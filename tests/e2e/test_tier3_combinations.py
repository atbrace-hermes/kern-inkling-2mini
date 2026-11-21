"""Tier 3: Pairwise cross-feature combination tests (15 tests)."""

from __future__ import annotations

import json
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
from tests.e2e.conftest import E2EConfig, require_manifest_artifacts  # noqa: E402
from tests.e2e.fixtures_data import (  # noqa: E402
    assert_valid_token_ids,
    build_exact_token_prompt,
    build_local_token_prompt,
    build_multiturn_messages,
    build_needle_haystack_tokens,
    build_weather_tool_schema,
    strip_terminal_stop_token,
    verify_greedy_parity_with_near_tie,
)

pytestmark = [pytest.mark.tier3, pytest.mark.gpu]


@pytest.mark.cli
def test_t3_01_loop_x_block_attnres_chunk_invariance(
    e2e_config: E2EConfig, kern_cli: KernCLI
) -> None:
    require_manifest_artifacts(e2e_config)
    pids = build_local_token_prompt(256, e2e_config.weights)
    p64, ids64 = kern_cli.kern_run(
        manifest=e2e_config.manifest,
        weights=e2e_config.weights,
        kernels=e2e_config.kernels if e2e_config.kernels.exists() else None,
        prompt_ids=pids,
        steps=16,
        capacity=512,
        chunk=64,
        gpu=e2e_config.gpu,
    )
    p512, ids512 = kern_cli.kern_run(
        manifest=e2e_config.manifest,
        weights=e2e_config.weights,
        kernels=e2e_config.kernels if e2e_config.kernels.exists() else None,
        prompt_ids=pids,
        steps=16,
        capacity=512,
        chunk=512,
        gpu=e2e_config.gpu,
    )
    assert p64.returncode == 0 and p512.returncode == 0
    assert len(ids64) > 0
    assert ids64 == ids512


def test_t3_02_cross_pass_read_x_hybrid_attn_4096_multihop(client: KernClient) -> None:
    n1 = "VarA = 19. "
    n2 = "VarB = VarA + 11. "
    filler = "Background reference note. " * 600  # ~4800+ tokens total for >=4096 multihop
    prompt = (
        f"{n1}\n{filler}\n{n2}\n{filler}\n"
        "Using the definitions above, compute VarB as an integer."
    )
    msgs = build_multiturn_messages([("user", prompt)], system_prompt="Answer concisely.")
    tok = client.tokenize(messages=msgs)
    assert tok["count"] >= 3000
    resp = client.chat_completions(msgs, max_tokens=256, temperature=0, reasoning_effort="low")
    msg = resp["choices"][0]["message"]
    combined = (msg.get("content") or "") + (msg.get("reasoning") or "")
    assert "30" in combined


@pytest.mark.cli
@pytest.mark.hybrid
def test_t3_03_rank64_controllers_x_sconv_kern_run_parity(
    e2e_config: E2EConfig, client: KernClient, kern_cli: KernCLI
) -> None:
    require_manifest_artifacts(e2e_config)
    prompt = "A1 B2 C3 D4. Ignore earlier tokens and output 7 * 8 ="
    serve_resp = client.completions(prompt, max_tokens=16, temperature=0, return_token_ids=True)
    c = serve_resp["choices"][0]
    prompt_ids = c["prompt_token_ids"]
    serve_ids = strip_terminal_stop_token(c["token_ids"], c["finish_reason"])
    cap = max(512, ((len(prompt_ids) + 16 + 63) // 16) * 16)
    proc, run_ids = kern_cli.kern_run(
        manifest=e2e_config.manifest,
        weights=e2e_config.weights,
        kernels=e2e_config.kernels if e2e_config.kernels.exists() else None,
        prompt_ids=prompt_ids,
        steps=16,
        capacity=cap,
        temperature=0.0,
        gpu=e2e_config.gpu,
    )
    assert proc.returncode == 0, proc.stderr
    assert serve_ids == run_ids


def test_t3_04_swa_w1024_x_sconv_1500_token_decode(client: KernClient) -> None:
    pids = build_exact_token_prompt(client, 1200)
    resp = client.completions(pids, max_tokens=300, temperature=0, return_token_ids=True)
    ids = resp["choices"][0]["token_ids"]
    assert len(ids) >= 32
    assert_valid_token_ids(ids)


@pytest.mark.cli
@pytest.mark.hybrid
def test_t3_05_prologue_l2_clip_x_nvfp4_mxfp8_moe_parity(
    e2e_config: E2EConfig, client: KernClient, kern_cli: KernCLI
) -> None:
    require_manifest_artifacts(e2e_config)
    prompt = ("! " * 150) + "Write a one-line Python lambda adding x and y:"
    serve_resp = client.completions(prompt, max_tokens=20, temperature=0, return_token_ids=True)
    c = serve_resp["choices"][0]
    prompt_ids = c["prompt_token_ids"]
    serve_ids = strip_terminal_stop_token(c["token_ids"], c["finish_reason"])
    cap = max(512, ((len(prompt_ids) + 20 + 63) // 16) * 16)
    proc, run_ids = kern_cli.kern_run(
        manifest=e2e_config.manifest,
        weights=e2e_config.weights,
        kernels=e2e_config.kernels if e2e_config.kernels.exists() else None,
        prompt_ids=prompt_ids,
        steps=20,
        capacity=cap,
        temperature=0.0,
        gpu=e2e_config.gpu,
    )
    assert proc.returncode == 0, proc.stderr
    assert serve_ids == run_ids


def test_t3_06_moe_top8_x_loop_heterogeneous_batch_8(client: KernClient) -> None:
    prompts = [
        "def quicksort(arr):\n",
        "Solve 19 * 19 =",
        '{"status": "ready", "items": [',
        "量子计算的基本单元是",
        "SELECT count(*) FROM orders WHERE",
        "In 1969, Apollo 11 landed on",
        "for i in range(10):\n    print(",
        "Explain the water cycle briefly:",
    ]
    refs = [
        client.completions(p, max_tokens=16, temperature=0, return_token_ids=True)["choices"][0]["token_ids"]
        for p in prompts
    ]
    batch = client.concurrent_map(
        lambda p: client.completions(p, max_tokens=16, temperature=0, return_token_ids=True)["choices"][0]["token_ids"],
        prompts,
        max_workers=8,
        synchronize=True,
    )
    assert len(batch) == len(refs)
    for b_ids, s_ids in zip(batch, refs, strict=True):
        assert_valid_token_ids(b_ids)
        assert len(b_ids) == len(s_ids)
        assert b_ids[0] == s_ids[0]


def test_t3_07_nvfp4_mxfp8_x_softcap_vocab_mask_high_temp(client: KernClient) -> None:
    resp = client.completions(
        "Unconstrained high-entropy stream:",
        max_tokens=64,
        temperature=1.5,
        top_k=0,
        top_p=1.0,
        seed=2026,
        return_token_ids=True,
    )
    assert_valid_token_ids(resp["choices"][0]["token_ids"])


@pytest.mark.cli
@pytest.mark.hybrid
def test_t3_08_dspark_mtp_x_sconv_alternating_turns(
    e2e_config: E2EConfig, client: KernClient, kern_cli: KernCLI
) -> None:
    require_manifest_artifacts(e2e_config)
    before = client.metrics()
    prompt = "Pattern 10 20 30 40 10 20 30 40 -> unpredictable 9a7f3c2e -> next:"
    serve_resp = client.completions(prompt, max_tokens=24, temperature=0, return_token_ids=True)
    after = client.metrics()
    mtp = compute_mtp_stats(before, after)
    assert mtp["drafts"] and mtp["drafts"] > 0
    assert mtp["accepted_tokens"] is not None and mtp["accepted_tokens"] > 0

    c = serve_resp["choices"][0]
    prompt_ids = c["prompt_token_ids"]
    serve_ids = strip_terminal_stop_token(c["token_ids"], c["finish_reason"])
    cap = max(512, ((len(prompt_ids) + 24 + 63) // 16) * 16)
    kernels = e2e_config.kernels if e2e_config.kernels.exists() else None
    proc, run_ids = kern_cli.kern_run(
        manifest=e2e_config.manifest,
        weights=e2e_config.weights,
        kernels=kernels,
        prompt_ids=prompt_ids,
        steps=24,
        rows=1,
        capacity=cap,
        temperature=0.0,
        gpu=e2e_config.gpu,
    )
    assert proc.returncode == 0, proc.stderr
    verify_greedy_parity_with_near_tie(
        kern_cli=kern_cli,
        manifest=e2e_config.manifest,
        weights=e2e_config.weights,
        prompt_ids=prompt_ids,
        serve_ids=serve_ids,
        run_ids=run_ids,
        kernels=kernels,
        rows=1,
        gpu=e2e_config.gpu,
        capacity=cap,
        steps=24,
    )


def test_t3_09_dspark_mtp_x_hybrid_attn_8192_prompt(client: KernClient) -> None:
    pids = build_exact_token_prompt(client, 8200)
    before = client.metrics()
    r1 = client.completions(pids, max_tokens=24, temperature=0, return_token_ids=True)
    after = client.metrics()
    mtp = compute_mtp_stats(before, after)
    assert mtp["drafts"] and mtp["drafts"] > 0
    r2 = client.completions(pids, max_tokens=24, temperature=0, return_token_ids=True)
    assert r1["choices"][0]["token_ids"] == r2["choices"][0]["token_ids"]


def test_t3_10_dspark_mtp_x_prefix_cache_warm_turn2(client: KernClient) -> None:
    nonce = uuid.uuid4().hex[:8]
    cold_nonce = uuid.uuid4().hex[:8]
    system = f"PrefixMTP-{nonce}: " + ("archival clause. " * 25)
    t1 = build_multiturn_messages([("user", "Remember number 404. Reply with OK.")], system_prompt=system)
    r1 = client.chat_completions(t1, max_tokens=256, temperature=0, reasoning_effort="none")
    assert r1["choices"][0]["finish_reason"] == "stop"
    a1 = r1["choices"][0]["message"].get("content") or ""
    assert len(a1.strip()) > 0

    t2 = t1 + [
        {"role": "assistant", "content": a1},
        {"role": "user", "content": "Repeat the number I told you as just the integer."},
    ]
    before = client.metrics()
    warm = client.chat_completions(t2, max_tokens=256, temperature=0, reasoning_effort="none")
    after = client.metrics()
    assert warm["usage"]["prompt_tokens_details"]["cached_tokens"] > 0
    assert warm["choices"][0]["finish_reason"] == "stop"
    mtp = compute_mtp_stats(before, after)
    assert mtp["drafts"] and mtp["drafts"] > 0
    assert mtp["accepted_tokens"] is not None and mtp["accepted_tokens"] > 0
    warm_content = warm["choices"][0]["message"].get("content") or ""
    assert "404" in warm_content

    cold_msgs = [
        {
            **m,
            "content": m["content"].replace(nonce, cold_nonce)
            if isinstance(m.get("content"), str)
            else m.get("content"),
        }
        for m in t2
    ]
    cold = client.chat_completions(cold_msgs, max_tokens=256, temperature=0, reasoning_effort="none")
    assert cold["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
    assert cold["choices"][0]["finish_reason"] == "stop"
    cold_content = cold["choices"][0]["message"].get("content") or ""
    assert warm_content == cold_content


def test_t3_11_dspark_mtp_x_tool_calling_and_reasoning(client: KernClient) -> None:
    tools = build_weather_tool_schema()
    msgs = build_multiturn_messages(
        [("user", "Use get_weather for Berlin in celsius.")],
        system_prompt="Think briefly and invoke get_weather.",
    )
    before = client.metrics()
    resp = client.chat_completions(
        msgs, tools=tools, max_tokens=512, temperature=0, reasoning_effort="low"
    )
    after = client.metrics()
    mtp = compute_mtp_stats(before, after)
    assert mtp["drafts"] and mtp["drafts"] > 0
    assert mtp["accepted_tokens"] is not None and mtp["accepted_tokens"] > 0
    choice = resp["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    msg = choice["message"]
    assert "reasoning_content" not in msg
    r_text = msg.get("reasoning")
    r_toks = resp.get("usage", {}).get("completion_tokens_details", {}).get("reasoning_tokens", 0)
    assert (isinstance(r_text, str) and len(r_text.strip()) > 0) or r_toks > 0
    tcs = msg.get("tool_calls")
    assert isinstance(tcs, list) and len(tcs) >= 1
    assert tcs[0]["function"]["name"] == "get_weather"
    assert isinstance(json.loads(tcs[0]["function"]["arguments"]), dict)


def test_t3_12_dspark_mtp_x_conc8_mixed_sampling(client: KernClient) -> None:
    greedy_prompt = "The boiling point of water at sea level in Celsius is"
    greedy_ref = client.completions(greedy_prompt, max_tokens=12, temperature=0, return_token_ids=True)[
        "choices"
    ][0]["token_ids"]
    sampled_prompt = "Invent a two-word codename for a satellite:"
    sampled_ref_1 = client.completions(
        sampled_prompt, max_tokens=16, temperature=0.8, seed=314, return_token_ids=True
    )["choices"][0]["token_ids"]
    sampled_ref_2 = client.completions(
        sampled_prompt, max_tokens=16, temperature=0.8, seed=314, return_token_ids=True
    )["choices"][0]["token_ids"]
    assert sampled_ref_1 == sampled_ref_2
    assert_valid_token_ids(sampled_ref_1)

    def _worker(idx: int) -> list[int]:
        if idx % 2 == 0:
            return client.completions(
                greedy_prompt, max_tokens=12, temperature=0, return_token_ids=True
            )["choices"][0]["token_ids"]
        return client.completions(
            sampled_prompt, max_tokens=16, temperature=0.8, seed=314, return_token_ids=True
        )["choices"][0]["token_ids"]

    before = client.metrics()
    outs = client.concurrent_map(_worker, list(range(8)), max_workers=8, synchronize=True)
    after = client.metrics()
    assert compute_mtp_stats(before, after)["drafts"] > 0
    for idx, out in enumerate(outs):
        if idx % 2 == 0:
            assert out == greedy_ref
        else:
            assert len(out) > 0
            assert_valid_token_ids(out)


def test_t3_13_fp8_kv_x_long_context_64k(e2e_config: E2EConfig, client: KernClient) -> None:
    target_len = min(65536, max(16384, e2e_config.max_ctx_probe))
    needle = client.tokenize(prompt="LongContextNeedle is ZETA99. ")["tokens"]
    question = client.tokenize(prompt="\nWhat is LongContextNeedle? Answer: ")["tokens"]
    filler = client.tokenize(prompt=" text")["tokens"][0]
    pids = build_needle_haystack_tokens(needle, question, filler, total_prompt_len=target_len, needle_pos=0)
    resp = client.completions(pids, max_tokens=16, temperature=0, return_token_ids=True)
    assert resp["usage"]["prompt_tokens"] == target_len
    assert "ZETA99" in resp["choices"][0]["text"]


def test_t3_14_prefix_cache_x_slot_eviction_and_abort(client: KernClient) -> None:
    nonce = uuid.uuid4().hex[:8]
    anchor = f"Anchor-{nonce}: " + ("data " * 40) + "Return 42:"
    ref = client.completions(anchor, max_tokens=8, temperature=0, return_token_ids=True)["choices"][0][
        "token_ids"
    ]
    for i in range(10):
        if i % 3 == 0:
            client.raw_post_abort(
                "/v1/completions",
                {
                    "model": client.resolve_model_id(),
                    "prompt": f"Abort-{nonce}-{i}: " + ("stream " * 30),
                    "max_tokens": 256,
                    "temperature": 0,
                },
                chunks_before_close=2,
            )
        else:
            client.completions(f"Churn-{nonce}-{i}: " + ("item " * 20), max_tokens=8, temperature=0)
    time.sleep(0.3)
    before = client.metrics()
    replay = client.completions(anchor, max_tokens=8, temperature=0, return_token_ids=True)
    after = client.metrics()
    assert replay["choices"][0]["token_ids"] == ref
    assert metric_delta(before, after, "vllm:prefix_cache_queries_total") >= 1.0


def test_t3_15_eos_stop_x_dspark_mtp_mid_block(client: KernClient) -> None:
    resp = client.completions(
        "Count: 1, 2, 3,",
        max_tokens=32,
        temperature=0,
        stop=[", 6"],
        return_token_ids=True,
    )
    c = resp["choices"][0]
    assert c["finish_reason"] == "stop"
    assert "7" not in c["text"]
    assert_valid_token_ids(c["token_ids"])
