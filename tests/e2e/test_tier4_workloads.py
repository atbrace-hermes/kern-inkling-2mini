"""Tier 4: Real-world application, stress, concurrency=8, 1M context, and MTP baseline workloads (8 tests, AC1-AC4)."""

from __future__ import annotations

import json
import re
import statistics
import sys
import time
import uuid
from pathlib import Path

import pytest

_E2E_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _E2E_DIR.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.e2e.client import KernCLI, KernClient, compute_mtp_stats, metric_delta  # noqa: E402
from tests.e2e.conftest import E2EConfig  # noqa: E402
from tests.e2e.fixtures_data import (  # noqa: E402
    GSM8K_STYLE_PROBLEMS,
    MAX_MODEL_LEN,
    BaselineArtifacts,
    assert_valid_token_ids,
    build_multiturn_messages,
    build_needle_haystack_tokens,
    build_weather_tool_schema,
    extract_baseline_mtp_metrics,
)

pytestmark = [pytest.mark.tier4, pytest.mark.gpu]


@pytest.mark.cli
def test_t4_w01_e2e_logits_gate(
    e2e_config: E2EConfig, baselines: BaselineArtifacts, kern_cli: KernCLI
) -> None:
    """AC1: Run `kern test e2e-logits` against the M0 official baseline Parquet."""
    assert e2e_config.kern_bin.is_file(), f"AC1 requires kern binary at {e2e_config.kern_bin}"
    assert e2e_config.manifest.is_file(), f"AC1 requires manifest at {e2e_config.manifest}"
    assert e2e_config.weights.is_dir(), f"AC1 requires weights directory at {e2e_config.weights}"
    assert baselines.recording_parquet is not None and baselines.recording_parquet.is_file(), (
        f"AC1 requires an M0 baseline recording Parquet in {e2e_config.baseline_dir}; "
        f"warnings={baselines.warnings}"
    )
    kl_arg = e2e_config.kl_limit
    if kl_arg is None and not baselines.recording_info.get("has_noise"):
        kl_arg = 0.05
    proc = kern_cli.kern_e2e_logits(
        manifest=e2e_config.manifest,
        weights=e2e_config.weights,
        reference=baselines.recording_parquet,
        kernels=e2e_config.kernels if e2e_config.kernels.exists() else None,
        kl=kl_arg,
        gpu=e2e_config.gpu,
    )
    assert proc.returncode == 0, f"AC1 e2e-logits gate failed:\n{proc.stdout}\n{proc.stderr}"


def test_t4_w02_multiturn_prefix_cache(client: KernClient) -> None:
    """AC2: 3-turn dialogue verifying prefix cache hits, MTP activity, and zero state contamination."""
    before_all = client.metrics()
    # Retry with a fresh nonce only if Turn 1 hits kern-serve's rare post-EOS
    # speculative overshoot ("not kept"), which intentionally skips caching Turn 1.
    nonce = ""
    turn1_msgs = turn2_msgs = r1 = r2_warm = after_t2 = None
    for _attempt in range(4):
        nonce = uuid.uuid4().hex[:8]
        system_prompt = (
            f"Session {nonce}. You are an expert systems engineer. "
            + ("Always preserve earlier variable bindings verbatim across turns. " * 25)
        )
        turn1_msgs = build_multiturn_messages(
            [("user", "Define constant ALPHA = 137 and reply with OK.")],
            system_prompt=system_prompt,
        )
        r1 = client.chat_completions(
            turn1_msgs, max_tokens=512, temperature=0, reasoning_effort="none"
        )
        assert r1["choices"][0]["finish_reason"] == "stop"
        a1 = r1["choices"][0]["message"].get("content") or ""
        r1_reason = r1["choices"][0]["message"].get("reasoning") or ""
        assert len(a1.strip()) > 0

        a1_msg: dict[str, object] = {"role": "assistant", "content": a1}
        if r1_reason:
            a1_msg["reasoning_content"] = r1_reason
        turn2_msgs = turn1_msgs + [
            a1_msg,
            {"role": "user", "content": "Define BETA = ALPHA + 63 and state BETA as just the integer."},
        ]
        before_t2 = client.metrics()
        r2_warm = client.chat_completions(
            turn2_msgs, max_tokens=512, temperature=0, reasoning_effort="none"
        )
        after_t2 = client.metrics()
        if r2_warm["usage"]["prompt_tokens_details"]["cached_tokens"] > 0:
            break

    assert turn1_msgs is not None and turn2_msgs is not None and r1 is not None and r2_warm is not None and after_t2 is not None
    assert r2_warm["choices"][0]["finish_reason"] == "stop"
    assert r2_warm["usage"]["prompt_tokens_details"]["cached_tokens"] > 0
    assert metric_delta(before_t2, after_t2, "vllm:prefix_cache_hits_total") >= 1.0
    a2 = r2_warm["choices"][0]["message"].get("content") or ""
    r2_reason = r2_warm["choices"][0]["message"].get("reasoning") or ""
    assert "200" in a2

    a2_msg: dict[str, object] = {"role": "assistant", "content": a2}
    if r2_reason:
        a2_msg["reasoning_content"] = r2_reason
    turn3_msgs = turn2_msgs + [
        a2_msg,
        {"role": "user", "content": "State ALPHA and BETA in one line as ALPHA=137, BETA=200."},
    ]
    r3_warm = client.chat_completions(
        turn3_msgs, max_tokens=512, temperature=0, reasoning_effort="none"
    )
    after_all = client.metrics()
    assert r3_warm["choices"][0]["finish_reason"] == "stop"
    assert r3_warm["usage"]["prompt_tokens_details"]["cached_tokens"] > 0
    warm_msg = r3_warm["choices"][0]["message"]
    c3 = warm_msg.get("content") or ""
    assert "137" in c3 and "200" in c3

    mtp = compute_mtp_stats(before_all, after_all)
    assert mtp["drafts"] is not None and mtp["drafts"] > 0
    assert mtp["accepted_tokens"] is not None and mtp["accepted_tokens"] > 0

    # Branch with fresh nonce to verify cold single-turn equivalence on the exact same 3-turn transcript
    cold_nonce = uuid.uuid4().hex[:8]
    cold_msgs = [
        {
            **m,
            **({"content": m["content"].replace(nonce, cold_nonce)} if isinstance(m.get("content"), str) else {}),
            **({"reasoning": m["reasoning"].replace(nonce, cold_nonce)} if isinstance(m.get("reasoning"), str) else {}),
            **({"reasoning_content": m["reasoning_content"].replace(nonce, cold_nonce)} if isinstance(m.get("reasoning_content"), str) else {}),
        }
        for m in turn3_msgs
    ]
    r3_cold = client.chat_completions(
        cold_msgs, max_tokens=512, temperature=0, reasoning_effort="none"
    )
    assert r3_cold["choices"][0]["finish_reason"] == "stop"
    assert r3_cold["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
    cold_msg = r3_cold["choices"][0]["message"]
    assert (warm_msg.get("content"), warm_msg.get("tool_calls")) == (
        cold_msg.get("content"),
        cold_msg.get("tool_calls"),
    )
    assert "137" in (cold_msg.get("content") or "") and "200" in (cold_msg.get("content") or "")


def _extract_final_int(content: str, reasoning: str) -> str | None:
    """Extract the final integer from `content` first, falling back to `reasoning` only if needed."""
    for text in (content, reasoning):
        cleaned = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text or "")
        nums = re.findall(r"-?\d+", cleaned)
        if nums:
            return nums[-1]
    return None


def test_t4_w03_gsm8k_mtp_acceptance(
    client: KernClient, baselines: BaselineArtifacts, e2e_config: E2EConfig
) -> None:
    """AC3: Replay GSM8K-style reasoning prompts at concurrency 1,4,8; verify MTP acceptance, TTFT & throughput."""
    # Warm once
    first_msgs = build_multiturn_messages(
        [("user", GSM8K_STYLE_PROBLEMS[0]["question"] + " End with the final integer.")]
    )
    client.chat_completions(first_msgs, max_tokens=256, temperature=0, reasoning_effort="low")

    # Measure steady-state TTFT on a streaming probe after warmup
    ttft_stream = client.chat_completions_stream(
        build_multiturn_messages(
            [("user", f"{GSM8K_STYLE_PROBLEMS[0]['question']} Show steps and end with the integer answer.")],
            system_prompt="You are a concise math solver.",
        ),
        max_tokens=256,
        temperature=0,
        reasoning_effort="low",
    )
    assert ttft_stream.done is True and not ttft_stream.errors
    assert ttft_stream.ttft_s is not None and ttft_stream.ttft_s > 0

    summaries: dict[int, dict[str, float]] = {}
    for conc in (1, 4, 8):
        before = client.metrics()
        started = time.monotonic()

        def _solve(item: dict[str, str]) -> tuple[bool, int, float]:
            t0 = time.monotonic()
            msgs = build_multiturn_messages(
                [("user", f"{item['question']} Show steps and end with the integer answer.")],
                system_prompt="You are a concise math solver.",
            )
            resp = client.chat_completions(msgs, max_tokens=512, temperature=0, reasoning_effort="low")
            dt = time.monotonic() - t0
            msg = resp["choices"][0]["message"]
            pred = _extract_final_int(msg.get("content") or "", msg.get("reasoning") or "")
            correct = pred == item["gold"]
            return correct, int(resp["usage"]["completion_tokens"]), dt

        outcomes = client.concurrent_map(_solve, GSM8K_STYLE_PROBLEMS, max_workers=conc)
        elapsed = time.monotonic() - started
        after = client.metrics()
        mtp = compute_mtp_stats(before, after)
        total_toks = sum(toks for _, toks, _ in outcomes)
        correct_count = sum(1 for ok, _, _ in outcomes if ok)
        assert correct_count >= (len(GSM8K_STYLE_PROBLEMS) * 3) // 4
        assert mtp["drafts"] is not None and mtp["drafts"] > 0
        assert mtp["tokens_per_round"] is not None and mtp["tokens_per_round"] > 1.0
        tok_s = total_toks / max(elapsed, 1e-6)
        median_lat_s = statistics.median(dt for _, _, dt in outcomes)
        assert tok_s > 0
        assert median_lat_s > 0
        summaries[conc] = {
            "tokens_per_round": mtp["tokens_per_round"],
            "tok_s": tok_s,
            "median_lat_s": median_lat_s,
        }

    # Concurrency 8 acceptance & throughput hold up relative to concurrency 1
    assert summaries[8]["tokens_per_round"] >= summaries[1]["tokens_per_round"] * 0.85
    assert summaries[8]["tok_s"] >= summaries[1]["tok_s"] * 0.90

    # Unconditional M0 baseline comparison (AC3)
    assert baselines.metrics_json is not None and baselines.metrics_data, (
        f"AC3 requires M0 baseline metrics JSON in {e2e_config.baseline_dir}; "
        f"warnings={baselines.warnings}"
    )
    base_mtp = extract_baseline_mtp_metrics(baselines.metrics_data)
    assert base_mtp["tokens_per_round"] is not None and base_mtp["tokens_per_round"] > 1.0, (
        f"AC3 requires valid baseline tokens_per_round in {baselines.metrics_json}, got {base_mtp}"
    )
    assert summaries[1]["tokens_per_round"] >= base_mtp["tokens_per_round"] * 0.80, (
        f"AC3 MTP mean acceptance regression: {summaries[1]['tokens_per_round']:.3f} "
        f"< 0.80 * baseline {base_mtp['tokens_per_round']:.3f}"
    )
    if base_mtp["tokens_per_second"] is not None:
        assert base_mtp["tokens_per_second"] > 0
        peak_tok_s = max(summaries[c]["tok_s"] for c in (1, 4, 8))
        assert peak_tok_s >= base_mtp["tokens_per_second"] * 0.50, (
            f"AC3 throughput regression: peak tok/s={peak_tok_s:.2f} "
            f"< 0.50 * baseline {base_mtp['tokens_per_second']:.2f}"
        )
    if base_mtp["ttft_s"] is not None:
        assert base_mtp["ttft_s"] > 0
        allowed_ttft_s = max(base_mtp["ttft_s"] * 3.0, base_mtp["ttft_s"] + 0.25)
        assert ttft_stream.ttft_s <= allowed_ttft_s, (
            f"AC3 TTFT regression: measured={ttft_stream.ttft_s:.4f}s > allowed={allowed_ttft_s:.4f}s "
            f"(baseline={base_mtp['ttft_s']:.4f}s)"
        )


def test_t4_w04_conc8_saturation_soak(client: KernClient) -> None:
    """AC3, AC4: Sustain concurrency=8 across 32 mixed greedy + seeded requests with zero OOM."""
    greedy_prompt = "The speed of light in vacuum is approximately"
    greedy_ref = client.completions(greedy_prompt, max_tokens=16, temperature=0, return_token_ids=True)[
        "choices"
    ][0]["token_ids"]
    sampled_prompt = "Write a short haiku about mountains:"
    sampled_ref_1 = client.completions(
        sampled_prompt, max_tokens=24, temperature=0.8, seed=888, return_token_ids=True
    )["choices"][0]["token_ids"]
    sampled_ref_2 = client.completions(
        sampled_prompt, max_tokens=24, temperature=0.8, seed=888, return_token_ids=True
    )["choices"][0]["token_ids"]
    assert sampled_ref_1 == sampled_ref_2
    assert_valid_token_ids(sampled_ref_1)

    def _task(idx: int) -> tuple[int, list[int], str]:
        if idx % 2 == 0:
            r = client.completions(greedy_prompt, max_tokens=16, temperature=0, return_token_ids=True)
        else:
            r = client.completions(
                sampled_prompt, max_tokens=24, temperature=0.8, seed=888, return_token_ids=True
            )
        choice = r["choices"][0]
        ids = choice["token_ids"]
        assert_valid_token_ids(ids)
        return idx, ids, choice.get("text", "")

    results = client.concurrent_map(_task, list(range(32)), max_workers=8, synchronize=True)
    for idx, ids, text in results:
        if idx % 2 == 0:
            assert len(ids) == len(greedy_ref)
            assert ids[:8] == greedy_ref[:8]
            assert "3" in text or "299" in text
        else:
            assert len(ids) > 0
            assert_valid_token_ids(ids)


def test_t4_w05_1m_context_stress(e2e_config: E2EConfig, client: KernClient) -> None:
    """AC4: Verify 1048576 config and progressive long-context needle recall without OOM."""
    models = client.models()
    assert models["data"][0]["max_model_len"] == MAX_MODEL_LEN

    needle = client.tokenize(prompt="CRITICAL_VAULT_TOKEN=OMEGA741. ")["tokens"]
    question = client.tokenize(prompt="\nReport CRITICAL_VAULT_TOKEN verbatim: ")["tokens"]
    filler = client.tokenize(prompt=" context")["tokens"][0]

    max_probe = min(MAX_MODEL_LEN - 64, max(65536, e2e_config.max_ctx_probe))
    probe_lengths = sorted({p for p in (65536, 131072, max_probe) if p <= max_probe})
    for length in probe_lengths:
        pids = build_needle_haystack_tokens(needle, question, filler, total_prompt_len=length, needle_pos=0)
        resp = client.completions(pids, max_tokens=24, temperature=0, return_token_ids=True, timeout=600.0)
        assert resp["usage"]["prompt_tokens"] == length
        assert_valid_token_ids(resp["choices"][0]["token_ids"])
        assert "OMEGA741" in resp["choices"][0]["text"]


def test_t4_w06_conc8_long_ctx_combined(e2e_config: E2EConfig, client: KernClient) -> None:
    """AC4: Run 8 concurrent long-context requests exercising FP8 KV + SWA/sconv rings with zero OOM."""
    per_seq_len = min(32768, max(8192, e2e_config.max_ctx_probe // 8))
    filler = client.tokenize(prompt=" block")["tokens"][0]
    question = client.tokenize(prompt="\nEcho the code: ")["tokens"]

    def _run_seq(idx: int) -> tuple[int, str]:
        code = f"SEQ{idx}KEY{100 + idx}"
        needle = client.tokenize(prompt=f"Code is {code}. ")["tokens"]
        pids = build_needle_haystack_tokens(needle, question, filler, total_prompt_len=per_seq_len, needle_pos=0)
        resp = client.completions(pids, max_tokens=16, temperature=0, return_token_ids=True, timeout=600.0)
        assert resp["usage"]["prompt_tokens"] == per_seq_len
        assert_valid_token_ids(resp["choices"][0]["token_ids"])
        return idx, resp["choices"][0]["text"]

    outs = client.concurrent_map(_run_seq, list(range(8)), max_workers=8)
    for idx, text in outs:
        assert f"SEQ{idx}KEY{100 + idx}" in text


def test_t4_w07_abort_and_slot_churn(client: KernClient) -> None:
    """AC2, AC4: Interleave SSE mid-stream aborts, Vision 400s, and 16-request slot churn."""
    ref_prompt = "Verify state integrity after churn: 12 * 12 ="
    ref_ids = client.completions(ref_prompt, max_tokens=12, temperature=0, return_token_ids=True)["choices"][0][
        "token_ids"
    ]

    for i in range(4):
        client.raw_post_abort(
            "/v1/completions",
            {
                "model": client.resolve_model_id(),
                "prompt": f"Long streaming request #{i}: explain compilers.",
                "max_tokens": 512,
                "temperature": 0,
            },
            chunks_before_close=3,
        )
        status, _, _ = client.raw_request(
            "POST",
            "/v1/chat/completions",
            body={
                "model": client.resolve_model_id(),
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Bad multimodal"},
                            {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}},
                        ],
                    }
                ],
            },
        )
        assert status == 400

    churn_prompts = [f"Churn item {k}: 1 + {k} =" for k in range(16)]
    churn_res = client.concurrent_map(
        lambda p: client.completions(p, max_tokens=8, temperature=0, return_token_ids=True),
        churn_prompts,
        max_workers=16,
    )
    assert len(churn_res) == 16

    after_ids = client.completions(ref_prompt, max_tokens=12, temperature=0, return_token_ids=True)["choices"][0][
        "token_ids"
    ]
    assert after_ids == ref_ids


def test_t4_w08_agentic_tool_workflow(client: KernClient) -> None:
    """AC2, AC3: End-to-end 3-step tool-calling agent loop with SSE streaming, reasoning, MTP & cache."""
    nonce = uuid.uuid4().hex[:8]
    tools = build_weather_tool_schema()
    sys_msg = (
        f"Agent-{nonce}. Use get_weather when asked about weather, then summarize the returned temperature."
    )
    msgs_step1 = build_multiturn_messages(
        [("user", "Check the weather in Tokyo in celsius using get_weather.")],
        system_prompt=sys_msg,
    )

    before = client.metrics()
    sse1 = client.chat_completions_stream(
        msgs_step1, tools=tools, max_tokens=512, temperature=0, reasoning_effort="low"
    )
    assert sse1.done is True
    assert not sse1.errors
    assert sse1.finish_reason == "tool_calls"
    assert len(sse1.tool_calls) >= 1
    tc = sse1.tool_calls[0]
    assert tc["function"]["name"] == "get_weather"
    parsed_args = json.loads(tc["function"]["arguments"])
    assert "city" in parsed_args

    assistant_msg: dict[str, object] = {
        "role": "assistant",
        "content": sse1.content or None,
        "tool_calls": [
            {
                "id": tc["id"] or "call_step1",
                "type": "function",
                "function": tc["function"],
            }
        ],
    }
    if sse1.reasoning:
        assistant_msg["reasoning"] = sse1.reasoning

    msgs_step2 = msgs_step1 + [
        assistant_msg,
        {
            "role": "tool",
            "tool_call_id": tc["id"] or "call_step1",
            "content": json.dumps({"city": "Tokyo", "temp_c": 21, "condition": "clear"}),
        },
    ]
    sse2 = client.chat_completions_stream(
        msgs_step2,
        tools=tools,
        max_tokens=512,
        temperature=0,
        reasoning_effort="low",
        stream_options={"include_usage": True},
    )
    after = client.metrics()
    assert sse2.done is True
    assert not sse2.errors
    assert "21" in sse2.content
    assert isinstance(sse2.reasoning, str) and len(sse2.reasoning.strip()) > 0
    assert sse2.usage is not None
    assert sse2.usage["prompt_tokens_details"]["cached_tokens"] > 0
    mtp = compute_mtp_stats(before, after)
    assert mtp["drafts"] is not None and mtp["drafts"] > 0
    assert mtp["accepted_tokens"] is not None and mtp["accepted_tokens"] > 0
