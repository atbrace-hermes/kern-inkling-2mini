"""Tier 1: Functional feature verification tests (66 tests across Features #1-#13)."""

from __future__ import annotations

import json
import sys
import uuid
from pathlib import Path
from typing import Any

import pytest

_E2E_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _E2E_DIR.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.e2e.client import KernCLI, KernClient, compute_mtp_stats, metric_delta  # noqa: E402
from tests.e2e.conftest import E2EConfig, require_manifest_artifacts  # noqa: E402
from tests.e2e.fixtures_data import (  # noqa: E402
    DIVERSE_PROMPTS,
    EOS_TOKEN_ID,
    MAX_MODEL_LEN,
    BaselineArtifacts,
    assert_valid_token_ids,
    build_exact_token_prompt,
    build_local_token_prompt,
    build_multiturn_messages,
    build_needle_haystack_tokens,
    build_weather_tool_schema,
    strip_terminal_stop_token,
    verify_greedy_parity_with_near_tie,
)

pytestmark = [pytest.mark.tier1, pytest.mark.gpu]


# ---------------------------------------------------------------------------
# Feature #1: Input Embedding + embed_norm (5 tests)
# ---------------------------------------------------------------------------


def test_t1_f01_01_ascii_tokenize_roundtrip(client: KernClient) -> None:
    text = "The quick brown fox jumps over 42 lazy dogs."
    tok = client.tokenize(prompt=text, return_token_strs=True)
    assert tok["count"] == len(tok["tokens"]) > 0
    assert_valid_token_ids(tok["tokens"])
    detok = client.detokenize(tok["tokens"])
    assert detok["prompt"] == text


def test_t1_f01_02_utf8_cjk_emoji_encoding(client: KernClient) -> None:
    text = "你好世界！Inkling-2-Mini 支持多语言与符号 🚀 αβγ."
    tok = client.tokenize(prompt=text)
    assert_valid_token_ids(tok["tokens"])
    resp = client.completions(text, max_tokens=16, temperature=0, return_token_ids=True)
    choice = resp["choices"][0]
    assert_valid_token_ids(choice["token_ids"])
    assert len(choice["token_ids"]) >= 1


def test_t1_f01_03_code_symbol_dense_tokens(client: KernClient) -> None:
    code = "def fib(n: int) -> int:\n    return n if n < 2 else fib(n - 1) + fib(n - 2)\n"
    tok = client.tokenize(prompt=code)
    assert_valid_token_ids(tok["tokens"])
    resp = client.completions(code, max_tokens=24, temperature=0, return_token_ids=True)
    assert resp["usage"]["completion_tokens"] >= 1
    assert_valid_token_ids(resp["choices"][0]["token_ids"])


def test_t1_f01_04_chat_control_token_rendering(client: KernClient) -> None:
    msgs = build_multiturn_messages([("user", "Say ok.")], system_prompt="Reply concisely.")
    tok = client.tokenize(messages=msgs)
    assert tok["count"] > 0
    assert_valid_token_ids(tok["tokens"])
    resp = client.chat_completions(msgs, max_tokens=128, temperature=0, reasoning_effort="none")
    msg = resp["choices"][0]["message"]
    content = msg.get("content")
    assert isinstance(content, str) and len(content.strip()) > 0
    assert "ok" in content.lower()


def test_t1_f01_05_repeated_request_identical_prompt_ids(client: KernClient) -> None:
    prompt = "Deterministic embedding check 12345:"
    r1 = client.completions(prompt, max_tokens=12, temperature=0, return_token_ids=True)
    r2 = client.completions(prompt, max_tokens=12, temperature=0, return_token_ids=True)
    assert r1["choices"][0]["prompt_token_ids"] == r2["choices"][0]["prompt_token_ids"]
    assert r1["choices"][0]["token_ids"] == r2["choices"][0]["token_ids"]


# ---------------------------------------------------------------------------
# Feature #2: 3-Phase Loop Topology (48 visits) (5 tests)
# ---------------------------------------------------------------------------


def test_t1_f02_01_multistep_arithmetic_loop(client: KernClient) -> None:
    msgs = build_multiturn_messages(
        [("user", "Compute (12 + 8) * 3 - 5. Respond with the final integer.")],
        system_prompt="You are a careful calculator.",
    )
    resp = client.chat_completions(msgs, max_tokens=512, temperature=0, reasoning_effort="low")
    choice = resp["choices"][0]
    combined = (choice["message"].get("reasoning") or "") + (choice["message"].get("content") or "")
    assert "55" in combined


@pytest.mark.cli
def test_t1_f02_02_kern_verify_manifest_protocol(e2e_config: E2EConfig, kern_cli: KernCLI) -> None:
    require_manifest_artifacts(e2e_config)
    proc = kern_cli.kern_verify(e2e_config.manifest)
    assert proc.returncode == 0, f"kern verify failed:\n{proc.stdout}\n{proc.stderr}"


@pytest.mark.cli
@pytest.mark.hybrid
def test_t1_f02_03_kern_run_vs_serve_greedy_parity(
    e2e_config: E2EConfig, client: KernClient, kern_cli: KernCLI
) -> None:
    require_manifest_artifacts(e2e_config)
    prompt = "The capital of France is"
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
    assert proc.returncode == 0, f"kern run failed:\n{proc.stderr}"
    assert serve_ids == run_ids


@pytest.mark.cli
def test_t1_f02_04_prefill_chunk_size_invariance(
    e2e_config: E2EConfig, kern_cli: KernCLI
) -> None:
    require_manifest_artifacts(e2e_config)
    prompt_ids = build_local_token_prompt(192, e2e_config.weights)
    p64, ids64 = kern_cli.kern_run(
        manifest=e2e_config.manifest,
        weights=e2e_config.weights,
        kernels=e2e_config.kernels if e2e_config.kernels.exists() else None,
        prompt_ids=prompt_ids,
        steps=12,
        capacity=512,
        chunk=64,
        gpu=e2e_config.gpu,
    )
    p128, ids128 = kern_cli.kern_run(
        manifest=e2e_config.manifest,
        weights=e2e_config.weights,
        kernels=e2e_config.kernels if e2e_config.kernels.exists() else None,
        prompt_ids=prompt_ids,
        steps=12,
        capacity=512,
        chunk=128,
        gpu=e2e_config.gpu,
    )
    assert p64.returncode == 0 and p128.returncode == 0
    assert len(ids64) > 0
    assert ids64 == ids128


def test_t1_f02_05_prompt_ids_vs_text_parity(client: KernClient) -> None:
    text = "In software engineering, regression testing verifies that"
    toks = client.tokenize(prompt=text)["tokens"]
    r_text = client.completions(text, max_tokens=16, temperature=0, return_token_ids=True)
    r_ids = client.completions(toks, max_tokens=16, temperature=0, return_token_ids=True)
    assert r_text["choices"][0]["token_ids"] == r_ids["choices"][0]["token_ids"]


# ---------------------------------------------------------------------------
# Feature #3: Cross-Pass Message Read (AttnRes) (5 tests)
# ---------------------------------------------------------------------------


def test_t1_f03_01_two_hop_variable_binding_recall(client: KernClient) -> None:
    prompt = (
        "Fact A: X equals 47. Fact B: Y equals X plus 10. "
        "Based on Fact A and Fact B, Y equals"
    )
    resp = client.completions(prompt, max_tokens=16, temperature=0)
    assert "57" in resp["choices"][0]["text"]


def test_t1_f03_02_multiconstraint_instruction_following(client: KernClient) -> None:
    msgs = build_multiturn_messages(
        [("user", "Output exactly two lowercase words separated by a space: hello world")],
        system_prompt="Follow instructions literally.",
    )
    resp = client.chat_completions(msgs, max_tokens=128, temperature=0, reasoning_effort="none")
    content = (resp["choices"][0]["message"].get("content") or "").strip().lower()
    assert "hello" in content and "world" in content


def test_t1_f03_03_cross_pass_numerical_consistency(client: KernClient) -> None:
    prompt = "Let a = 13, b = 29. Their sum a + b is"
    r1 = client.completions(prompt, max_tokens=12, temperature=0, return_token_ids=True)
    r2 = client.completions(prompt, max_tokens=12, temperature=0, return_token_ids=True)
    assert r1["choices"][0]["token_ids"] == r2["choices"][0]["token_ids"]
    assert "42" in r1["choices"][0]["text"]


def test_t1_f03_04_pass_depth_consistency_64tok(client: KernClient) -> None:
    prompt = "Write a short concise paragraph explaining why hash tables have O(1) average lookup:"
    resp = client.completions(prompt, max_tokens=64, temperature=0, return_token_ids=True)
    ids = resp["choices"][0]["token_ids"]
    assert len(ids) >= 8
    assert_valid_token_ids(ids)


def test_t1_f03_05_zero_drift_deterministic_replay(client: KernClient) -> None:
    prompt = DIVERSE_PROMPTS[0]
    runs = [
        client.completions(prompt, max_tokens=20, temperature=0, return_token_ids=True)["choices"][0]["token_ids"]
        for _ in range(3)
    ]
    assert runs[0] == runs[1] == runs[2]


# ---------------------------------------------------------------------------
# Feature #4: Rank-64 Gated Loop Controllers (5 tests)
# ---------------------------------------------------------------------------


def test_t1_f04_01_fp32_stability_256tok_greedy(client: KernClient) -> None:
    prompt = "Explain step by step how binary search works on a sorted array of integers:\n"
    resp = client.completions(prompt, max_tokens=256, temperature=0, return_token_ids=True)
    ids = resp["choices"][0]["token_ids"]
    assert len(ids) > 16
    assert_valid_token_ids(ids)
    assert len(set(ids[-32:])) > 2


def test_t1_f04_02_mid_prompt_topic_switch_gating(client: KernClient) -> None:
    prompt = (
        "Earlier we discussed ocean currents, tides, and coral reefs in detail. "
        "Now ignore marine biology completely and answer only this arithmetic question: 9 * 9 ="
    )
    resp = client.completions(prompt, max_tokens=12, temperature=0)
    assert "81" in resp["choices"][0]["text"]


def test_t1_f04_03_cross_domain_deterministic_continuation(client: KernClient) -> None:
    for prompt in DIVERSE_PROMPTS[:4]:
        a = client.completions(prompt, max_tokens=16, temperature=0, return_token_ids=True)
        b = client.completions(prompt, max_tokens=16, temperature=0, return_token_ids=True)
        assert a["choices"][0]["token_ids"] == b["choices"][0]["token_ids"]


def test_t1_f04_04_seeded_sampling_reproducibility(client: KernClient) -> None:
    prompt = "Invent a short motto for a mountain climbing club:"
    r1 = client.completions(
        prompt, max_tokens=24, temperature=0.8, top_k=20, top_p=0.9, seed=42, return_token_ids=True
    )
    r2 = client.completions(
        prompt, max_tokens=24, temperature=0.8, top_k=20, top_p=0.9, seed=42, return_token_ids=True
    )
    assert r1["choices"][0]["token_ids"] == r2["choices"][0]["token_ids"]


@pytest.mark.cli
def test_t1_f04_05_e2e_logits_kl_gate(
    e2e_config: E2EConfig, baselines: BaselineArtifacts, kern_cli: KernCLI
) -> None:
    assert e2e_config.kern_bin.is_file(), f"Requires kern binary at {e2e_config.kern_bin}"
    assert e2e_config.manifest.is_file(), f"Requires manifest at {e2e_config.manifest}"
    assert e2e_config.weights.is_dir(), f"Requires weights directory at {e2e_config.weights}"
    assert baselines.recording_parquet is not None and baselines.recording_parquet.is_file(), (
        f"Requires M0 baseline recording Parquet under {e2e_config.baseline_dir}; "
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
    assert proc.returncode == 0, f"e2e-logits failed:\n{proc.stdout}\n{proc.stderr}"


# ---------------------------------------------------------------------------
# Feature #5: Block Attention Residuals (attn_res) (5 tests)
# ---------------------------------------------------------------------------


def test_t1_f05_01_structured_json_generation(client: KernClient) -> None:
    msgs = build_multiturn_messages(
        [("user", 'Return valid JSON with exact keys "status" and "count": {"status":"ok","count":3}')],
        system_prompt="Output raw JSON only.",
    )
    resp = client.chat_completions(msgs, max_tokens=128, temperature=0, reasoning_effort="none")
    text = (resp["choices"][0]["message"].get("content") or "").strip()
    assert "{" in text and "}" in text
    candidate = text[text.index("{") : text.rindex("}") + 1]
    parsed = json.loads(candidate)
    assert isinstance(parsed, dict)


def test_t1_f05_02_chain_of_thought_coherency(client: KernClient) -> None:
    msgs = build_multiturn_messages(
        [("user", "If a rectangle has width 7 and height 6, what is its area?")],
    )
    resp = client.chat_completions(msgs, max_tokens=512, temperature=0, reasoning_effort="medium")
    msg = resp["choices"][0]["message"]
    assert (msg.get("reasoning") or msg.get("content")) is not None
    combined = (msg.get("reasoning") or "") + (msg.get("content") or "")
    assert "42" in combined


def test_t1_f05_03_multiturn_residual_consistency(client: KernClient) -> None:
    nonce = uuid.uuid4().hex[:8]
    msgs = [
        {"role": "system", "content": f"Session {nonce}. Answer concisely."},
        {"role": "user", "content": "Remember the number 731."},
        {"role": "assistant", "content": "I will remember 731."},
        {"role": "user", "content": "What number did I ask you to remember?"},
    ]
    r1 = client.chat_completions(msgs, max_tokens=128, temperature=0, reasoning_effort="none")
    r2 = client.chat_completions(msgs, max_tokens=128, temperature=0, reasoning_effort="none")
    assert r1["choices"][0]["message"]["content"] == r2["choices"][0]["message"]["content"]
    assert "731" in (r1["choices"][0]["message"]["content"] or "")


@pytest.mark.cli
def test_t1_f05_04_kern_run_single_vs_chunk_prefill_parity(
    e2e_config: E2EConfig, kern_cli: KernCLI
) -> None:
    require_manifest_artifacts(e2e_config)
    prompt_ids = build_local_token_prompt(36, e2e_config.weights)
    p1, ids1 = kern_cli.kern_run(
        manifest=e2e_config.manifest,
        weights=e2e_config.weights,
        kernels=e2e_config.kernels if e2e_config.kernels.exists() else None,
        prompt_ids=prompt_ids,
        steps=10,
        capacity=512,
        chunk=1,
        gpu=e2e_config.gpu,
    )
    p2, ids2 = kern_cli.kern_run(
        manifest=e2e_config.manifest,
        weights=e2e_config.weights,
        kernels=e2e_config.kernels if e2e_config.kernels.exists() else None,
        prompt_ids=prompt_ids,
        steps=10,
        capacity=512,
        chunk=36,
        gpu=e2e_config.gpu,
    )
    assert p1.returncode == 0 and p2.returncode == 0
    assert len(ids1) > 0
    assert ids1 == ids2


def test_t1_f05_05_batched_n4_vs_singleton_greedy_parity(client: KernClient) -> None:
    prompts = DIVERSE_PROMPTS[:4]
    singletons = [
        client.completions(p, max_tokens=16, temperature=0, return_token_ids=True)["choices"][0]["token_ids"]
        for p in prompts
    ]

    def _run(p: str) -> list[int]:
        return client.completions(p, max_tokens=16, temperature=0, return_token_ids=True)["choices"][0]["token_ids"]

    batched = client.concurrent_map(_run, prompts, max_workers=4)
    assert len(batched) == len(singletons)
    for b_ids, s_ids in zip(batched, singletons, strict=True):
        assert_valid_token_ids(b_ids)
        assert len(b_ids) == len(s_ids)
        # First prefill token must match identically; subsequent tokens may flip on M-dependent NVFP4/BF16 near-ties.
        assert b_ids[0] == s_ids[0]


# ---------------------------------------------------------------------------
# Feature #6: Hybrid Attention (40 SWA + 8 Full) (5 tests)
# ---------------------------------------------------------------------------


def test_t1_f06_01_short_range_swa_recall(client: KernClient) -> None:
    prompt = "The secret passcode is AMBER77. What is the secret passcode? Passcode:"
    resp = client.completions(prompt, max_tokens=12, temperature=0)
    assert "AMBER77" in resp["choices"][0]["text"]


def test_t1_f06_02_beyond_swa_2048_full_attn_recall(client: KernClient) -> None:
    needle = client.tokenize(prompt="The vault code is 849201. ")["tokens"]
    question = client.tokenize(prompt="\nQuestion: What is the vault code?\nAnswer: ")["tokens"]
    filler = client.tokenize(prompt=" alpha")["tokens"][0]
    prompt_ids = build_needle_haystack_tokens(needle, question, filler, total_prompt_len=2048, needle_pos=0)
    resp = client.completions(prompt_ids, max_tokens=16, temperature=0, return_token_ids=True)
    assert resp["usage"]["prompt_tokens"] == 2048
    assert "849201" in resp["choices"][0]["text"]


def test_t1_f06_03_past_log_scaling_floor_10000_recall(client: KernClient) -> None:
    needle = client.tokenize(prompt="Remember keycode K9135. ")["tokens"]
    question = client.tokenize(prompt="\nWhat keycode should you remember?\nKeycode: ")["tokens"]
    filler = client.tokenize(prompt=" alpha")["tokens"][0]
    prompt_ids = build_needle_haystack_tokens(needle, question, filler, total_prompt_len=10000, needle_pos=0)
    resp = client.completions(prompt_ids, max_tokens=16, temperature=0, return_token_ids=True)
    assert resp["usage"]["prompt_tokens"] == 10000
    assert "9135" in resp["choices"][0]["text"]


def test_t1_f06_04_multi_needle_across_swa_full_4096(client: KernClient) -> None:
    n1 = client.tokenize(prompt="TokenAlpha is 1111. ")["tokens"]
    n2 = client.tokenize(prompt="TokenBeta is 2222. ")["tokens"]
    q = client.tokenize(prompt="\nList TokenAlpha and TokenBeta: ")["tokens"]
    filler = client.tokenize(prompt=" note")["tokens"][0]
    buf = [filler] * 4096
    buf[0 : len(n1)] = n1
    buf[2048 : 2048 + len(n2)] = n2
    buf[4096 - len(q) :] = q
    resp = client.completions(buf, max_tokens=24, temperature=0)
    txt = resp["choices"][0]["text"]
    assert "1111" in txt and "2222" in txt


def test_t1_f06_05_paged_fp8_kv_16384_completion(client: KernClient) -> None:
    prompt_ids = build_exact_token_prompt(client, 16384)
    resp = client.completions(prompt_ids, max_tokens=16, temperature=0, return_token_ids=True)
    assert resp["usage"]["prompt_tokens"] == 16384
    assert_valid_token_ids(resp["choices"][0]["token_ids"])


# ---------------------------------------------------------------------------
# Feature #7: Attention Prologue + Gate/Sink (5 tests)
# ---------------------------------------------------------------------------


def test_t1_f07_01_first_token_anchor_sensitivity(client: KernClient) -> None:
    r1 = client.completions("First anchor: RED. Output the color:", max_tokens=8, temperature=0)
    r2 = client.completions("First anchor: BLUE. Output the color:", max_tokens=8, temperature=0)
    assert r1["choices"][0]["text"] != r2["choices"][0]["text"]


def test_t1_f07_02_repetitive_high_norm_prefix_stability(client: KernClient) -> None:
    prompt = ("the " * 200) + "Question: What is 6 + 7? Answer:"
    resp = client.completions(prompt, max_tokens=12, temperature=0, return_token_ids=True)
    assert_valid_token_ids(resp["choices"][0]["token_ids"])
    assert "13" in resp["choices"][0]["text"]


def test_t1_f07_03_punctuation_dense_input_stability(client: KernClient) -> None:
    prompt = "[[((!!??::;;--==++**//))]]: Next valid integer after 99 is"
    resp = client.completions(prompt, max_tokens=12, temperature=0, return_token_ids=True)
    assert_valid_token_ids(resp["choices"][0]["token_ids"])


def test_t1_f07_04_relative_position_order_sensitivity(client: KernClient) -> None:
    p_ab = "Order: First came Alpha, second came Beta. Which came second? Answer:"
    p_ba = "Order: First came Beta, second came Alpha. Which came second? Answer:"
    r_ab = client.completions(p_ab, max_tokens=12, temperature=0)
    r_ba = client.completions(p_ba, max_tokens=12, temperature=0)
    assert "Beta" in r_ab["choices"][0]["text"]
    assert "Alpha" in r_ba["choices"][0]["text"]


def test_t1_f07_05_relative_order_greedy_determinism(client: KernClient) -> None:
    p = "Sequence: item_one, item_two, item_three. Repeat the last item:"
    a = client.completions(p, max_tokens=12, temperature=0, return_token_ids=True)["choices"][0]["token_ids"]
    b = client.completions(p, max_tokens=12, temperature=0, return_token_ids=True)["choices"][0]["token_ids"]
    assert a == b


# ---------------------------------------------------------------------------
# Feature #8: 4-Stream Short Conv (sconv, W=4) (5 tests)
# ---------------------------------------------------------------------------


def test_t1_f08_01_local_4token_ngram_continuation(client: KernClient) -> None:
    prompt = "A1 B2 C3 D4 A1 B2 C3 D4 A1 B2 C3"
    resp = client.completions(prompt, max_tokens=8, temperature=0)
    assert "D4" in resp["choices"][0]["text"]


def test_t1_f08_02_bracket_delimiter_motif_matching(client: KernClient) -> None:
    prompt = "<<<[x]>>> <<<[y]>>> <<<[z]>>>"
    resp = client.completions(prompt, max_tokens=16, temperature=0, return_token_ids=True)
    assert_valid_token_ids(resp["choices"][0]["token_ids"])


def test_t1_f08_03_cold_vs_prefix_cached_sconv_parity(client: KernClient) -> None:
    nonce = uuid.uuid4().hex[:8]
    cold_nonce = uuid.uuid4().hex[:8]
    sys_prompt = f"Prefix-{nonce}: " + ("word " * 80)
    t1_msgs = build_multiturn_messages([("user", "Acknowledge with OK.")], system_prompt=sys_prompt)
    r1 = client.chat_completions(t1_msgs, max_tokens=256, temperature=0, reasoning_effort="none")
    assert r1["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
    assert r1["choices"][0]["finish_reason"] == "stop"
    a1 = r1["choices"][0]["message"].get("content") or ""
    r1_reason = r1["choices"][0]["message"].get("reasoning") or ""
    assert len(a1.strip()) > 0

    a1_msg: dict[str, object] = {"role": "assistant", "content": a1}
    if r1_reason:
        a1_msg["reasoning"] = r1_reason
    t2_msgs = t1_msgs + [
        a1_msg,
        {"role": "user", "content": "What is 15 + 27? Reply with only the integer."},
    ]
    warm = client.chat_completions(t2_msgs, max_tokens=128, temperature=0, reasoning_effort="none")
    assert warm["usage"]["prompt_tokens_details"]["cached_tokens"] > 0
    assert warm["choices"][0]["finish_reason"] == "stop"
    warm_content = warm["choices"][0]["message"].get("content") or ""
    assert "42" in warm_content

    cold_msgs = [
        {
            **m,
            "content": m["content"].replace(nonce, cold_nonce)
            if isinstance(m.get("content"), str)
            else m.get("content"),
        }
        for m in t2_msgs
    ]
    cold = client.chat_completions(cold_msgs, max_tokens=128, temperature=0, reasoning_effort="none")
    assert cold["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
    cold_content = cold["choices"][0]["message"].get("content") or ""
    assert "42" in cold_content
    assert warm_content == cold_content


def test_t1_f08_04_sequential_request_sconv_isolation(client: KernClient) -> None:
    clean_prompt = "The capital of Japan is"
    baseline = client.completions(clean_prompt, max_tokens=8, temperature=0, return_token_ids=True)
    # Send an unrelated noisy prompt to exercise sconv buffers
    client.completions("XYZQ 9988 !@# " * 20, max_tokens=16, temperature=0)
    after = client.completions(clean_prompt, max_tokens=8, temperature=0, return_token_ids=True)
    assert after["choices"][0]["token_ids"] == baseline["choices"][0]["token_ids"]


def test_t1_f08_05_prefill_chunk_crossing_sconv_w4(client: KernClient) -> None:
    prompt_ids = build_exact_token_prompt(client, 131)
    r1 = client.completions(prompt_ids, max_tokens=16, temperature=0, return_token_ids=True)
    r2 = client.completions(prompt_ids, max_tokens=16, temperature=0, return_token_ids=True)
    assert r1["choices"][0]["token_ids"] == r2["choices"][0]["token_ids"]


# ---------------------------------------------------------------------------
# Feature #9: MoE 128-Expert Top-8 SwiGLU (5 tests)
# ---------------------------------------------------------------------------


def test_t1_f09_01_python_code_expert_routing(client: KernClient) -> None:
    prompt = "```python\ndef factorial(n: int) -> int:\n"
    resp = client.completions(prompt, max_tokens=160, temperature=0, return_token_ids=True)
    assert "return" in resp["choices"][0]["text"]
    assert_valid_token_ids(resp["choices"][0]["token_ids"])


def test_t1_f09_02_math_derivation_expert_routing(client: KernClient) -> None:
    prompt = "Solve for x: 3x + 7 = 22. Step 1: 3x = 15. Step 2: x ="
    resp = client.completions(prompt, max_tokens=12, temperature=0)
    assert "5" in resp["choices"][0]["text"]


def test_t1_f09_03_sql_json_expert_routing(client: KernClient) -> None:
    prompt = "SELECT id, name FROM users WHERE active = true ORDER BY"
    resp = client.completions(prompt, max_tokens=20, temperature=0, return_token_ids=True)
    assert_valid_token_ids(resp["choices"][0]["token_ids"])
    assert len(resp["choices"][0]["text"].strip()) > 0


def test_t1_f09_04_multilingual_prose_expert_routing(client: KernClient) -> None:
    prompt = "北京是中国的首都，上海是"
    resp = client.completions(prompt, max_tokens=20, temperature=0, return_token_ids=True)
    assert_valid_token_ids(resp["choices"][0]["token_ids"])
    assert len(resp["choices"][0]["text"]) > 0


def test_t1_f09_05_heterogeneous_concurrent_batch_parity(client: KernClient) -> None:
    prompts = [
        "def add(a, b):\n    return",
        "Solve 4 * 11 =",
        '{"key": "value", "num":',
        "巴黎是法国的",
    ]
    singletons = [
        client.completions(p, max_tokens=16, temperature=0, return_token_ids=True)["choices"][0]["token_ids"]
        for p in prompts
    ]
    batched = client.concurrent_map(
        lambda p: client.completions(p, max_tokens=16, temperature=0, return_token_ids=True)["choices"][0]["token_ids"],
        prompts,
        max_workers=4,
    )
    assert len(batched) == len(singletons)
    for b_ids, s_ids in zip(batched, singletons, strict=True):
        assert_valid_token_ids(b_ids)
        assert len(b_ids) == len(s_ids)
        assert b_ids[0] == s_ids[0]


# ---------------------------------------------------------------------------
# Feature #10: Mixed NVFP4 (L0-19) + MXFP8 (L20-23) Quantization (5 tests)
# ---------------------------------------------------------------------------


@pytest.mark.cli
def test_t1_f10_01_manifest_mixed_quant_verify(e2e_config: E2EConfig, kern_cli: KernCLI) -> None:
    require_manifest_artifacts(e2e_config)
    proc = kern_cli.kern_verify(e2e_config.manifest)
    assert proc.returncode == 0, proc.stderr


@pytest.mark.cli
def test_t1_f10_02_e2e_logits_quant_kl_threshold(
    e2e_config: E2EConfig, baselines: BaselineArtifacts, kern_cli: KernCLI
) -> None:
    assert e2e_config.kern_bin.is_file(), f"Requires kern binary at {e2e_config.kern_bin}"
    assert e2e_config.manifest.is_file(), f"Requires manifest at {e2e_config.manifest}"
    assert e2e_config.weights.is_dir(), f"Requires weights directory at {e2e_config.weights}"
    assert baselines.recording_parquet is not None and baselines.recording_parquet.is_file(), (
        f"Requires M0 baseline recording Parquet under {e2e_config.baseline_dir}; "
        f"warnings={baselines.warnings}"
    )
    kl_limit = e2e_config.kl_limit if e2e_config.kl_limit is not None else 0.05
    proc = kern_cli.kern_e2e_logits(
        manifest=e2e_config.manifest,
        weights=e2e_config.weights,
        reference=baselines.recording_parquet,
        kernels=e2e_config.kernels if e2e_config.kernels.exists() else None,
        kl=kl_limit,
        gpu=e2e_config.gpu,
    )
    assert proc.returncode == 0, f"Quantized KL gate failed:\n{proc.stdout}\n{proc.stderr}"


def test_t1_f10_03_factual_qa_no_quant_degradation(client: KernClient) -> None:
    resp = client.completions("Water freezes at 0 degrees Celsius and boils at", max_tokens=12, temperature=0)
    assert "100" in resp["choices"][0]["text"]


def test_t1_f10_04_exact_integer_arithmetic_sanity(client: KernClient) -> None:
    resp = client.completions("17 * 23 =", max_tokens=12, temperature=0)
    assert "391" in resp["choices"][0]["text"]


def test_t1_f10_05_sustained_512tok_finite_generation(client: KernClient) -> None:
    prompt = "Write a detailed technical overview of how TCP congestion control works:\n"
    resp = client.completions(prompt, max_tokens=512, temperature=0, return_token_ids=True)
    ids = resp["choices"][0]["token_ids"]
    assert len(ids) >= 32
    assert_valid_token_ids(ids)


# ---------------------------------------------------------------------------
# Feature #11: LM Head + FP32 Softcap 30.0 + Vocab Mask (200058) (5 tests)
# ---------------------------------------------------------------------------


def test_t1_f11_01_greedy_token_ids_below_200058(client: KernClient) -> None:
    resp = client.completions(
        "List the first ten prime numbers:", max_tokens=64, temperature=0, return_token_ids=True
    )
    assert_valid_token_ids(resp["choices"][0]["token_ids"])


def test_t1_f11_02_sampled_token_ids_below_200058(client: KernClient) -> None:
    resp = client.completions(
        "Write a creative poem about stars:",
        max_tokens=64,
        temperature=1.0,
        top_p=1.0,
        seed=101,
        return_token_ids=True,
    )
    assert_valid_token_ids(resp["choices"][0]["token_ids"])


def test_t1_f11_03_eos_200006_finish_reason_stop(client: KernClient) -> None:
    msgs = build_multiturn_messages([("user", "Reply with just the word: ok")])
    resp = client.chat_completions(
        msgs, max_tokens=64, temperature=0, reasoning_effort="none", return_token_ids=True
    )
    choice = resp["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert isinstance(choice.get("token_ids"), list) and len(choice["token_ids"]) >= 1
    assert_valid_token_ids(choice["token_ids"])
    assert choice["token_ids"][-1] == EOS_TOKEN_ID or choice.get("stop_reason") == EOS_TOKEN_ID


def test_t1_f11_04_max_tokens_finish_reason_length(client: KernClient) -> None:
    resp = client.completions(
        "Count upwards starting from 1: 1, 2, 3, 4, 5,",
        max_tokens=5,
        temperature=0,
        ignore_eos=True,
        return_token_ids=True,
    )
    choice = resp["choices"][0]
    assert choice["finish_reason"] == "length"
    assert len(choice["token_ids"]) == 5


def test_t1_f11_05_custom_stop_sequences_honored(client: KernClient) -> None:
    resp = client.completions(
        "Count: 1, 2, 3, 4, 5, 6, 7",
        max_tokens=32,
        temperature=0,
        stop=[", 5", "5"],
    )
    choice = resp["choices"][0]
    assert choice["finish_reason"] == "stop"
    assert "6" not in choice["text"]


# ---------------------------------------------------------------------------
# Feature #12: DSpark MTP 3-Layer + RNN Markov Head (5 tests)
# ---------------------------------------------------------------------------


def test_t1_f12_01_mtp_prometheus_counters_increment(client: KernClient) -> None:
    before = client.metrics()
    client.completions(
        "def count_items(items):\n    total = 0\n    for item in items:\n",
        max_tokens=48,
        temperature=0,
    )
    after = client.metrics()
    mtp = compute_mtp_stats(before, after)
    assert mtp["drafts"] is not None and mtp["drafts"] > 0
    assert mtp["draft_tokens"] is not None and mtp["draft_tokens"] > 0
    assert mtp["accepted_tokens"] is not None and mtp["accepted_tokens"] >= 0


@pytest.mark.cli
@pytest.mark.hybrid
def test_t1_f12_02_mtp_greedy_matches_kern_run_rows1(
    e2e_config: E2EConfig, client: KernClient, kern_cli: KernCLI
) -> None:
    require_manifest_artifacts(e2e_config)
    prompt = "The three primary colors in additive RGB mixing are"
    serve_resp = client.completions(prompt, max_tokens=24, temperature=0, return_token_ids=True)
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


def test_t1_f12_03_high_acceptance_structured_prompt(client: KernClient) -> None:
    before = client.metrics()
    prompt = "Repeat the exact pattern:\n1 -> 1\n2 -> 2\n3 -> 3\n4 -> 4\n5 -> 5\n6 ->"
    client.completions(prompt, max_tokens=48, temperature=0)
    after = client.metrics()
    mtp = compute_mtp_stats(before, after)
    assert mtp["drafts"] and mtp["drafts"] > 0
    assert mtp["tokens_per_round"] is not None and mtp["tokens_per_round"] > 1.2


def test_t1_f12_04_probabilistic_sampling_seed_replay_mtp(client: KernClient) -> None:
    prompt = "Tell a very short story about a clockmaker:"
    r1 = client.completions(
        prompt, max_tokens=32, temperature=0.8, top_p=0.95, seed=77, return_token_ids=True
    )
    r2 = client.completions(
        prompt, max_tokens=32, temperature=0.8, top_p=0.95, seed=77, return_token_ids=True
    )
    assert r1["choices"][0]["token_ids"] == r2["choices"][0]["token_ids"]


@pytest.mark.cli
@pytest.mark.hybrid
def test_t1_f12_05_draft_rejection_recovery_parity(
    e2e_config: E2EConfig, client: KernClient, kern_cli: KernCLI
) -> None:
    require_manifest_artifacts(e2e_config)
    prompt = "Random SHA prefixes: 9f8a7c 1b2e3d 4a5f6c 7d8e9a 0c1b2a"
    serve_resp = client.completions(prompt, max_tokens=20, temperature=0, return_token_ids=True)
    c = serve_resp["choices"][0]
    prompt_ids = c["prompt_token_ids"]
    serve_ids = strip_terminal_stop_token(c["token_ids"], c["finish_reason"])
    cap = max(512, ((len(prompt_ids) + 20 + 63) // 16) * 16)
    kernels = e2e_config.kernels if e2e_config.kernels.exists() else None
    proc, run_ids = kern_cli.kern_run(
        manifest=e2e_config.manifest,
        weights=e2e_config.weights,
        kernels=kernels,
        prompt_ids=prompt_ids,
        steps=20,
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
        steps=20,
    )


# ---------------------------------------------------------------------------
# Feature #13: Serving Spec + Prefix Cache + 1M / Conc-8 (6 tests)
# ---------------------------------------------------------------------------


def test_t1_f13_01_health_and_models_1m_max_model_len(client: KernClient) -> None:
    status, _ = client.health()
    assert status == 200
    models = client.models()
    assert models["object"] == "list"
    card = models["data"][0]
    assert card["owned_by"] == "vllm-frontend-rs"
    assert card["max_model_len"] == MAX_MODEL_LEN


def test_t1_f13_02_tokenize_and_detokenize_endpoints(client: KernClient) -> None:
    sample = "Inkling-2-Mini serving endpoint check."
    tok = client.tokenize(prompt=sample, return_token_strs=True)
    assert tok["max_model_len"] == MAX_MODEL_LEN
    assert len(tok["token_strs"]) == tok["count"]
    detok = client.detokenize(tok["tokens"])
    assert detok["prompt"] == sample


def test_t1_f13_03_completions_sse_stream_parity(client: KernClient) -> None:
    prompt = "The capital of France is"
    non_stream = client.completions(prompt, max_tokens=16, temperature=0, return_token_ids=True)
    streamed = client.completions_stream(prompt, max_tokens=16, temperature=0, return_token_ids=True)
    assert streamed.done is True
    assert not streamed.errors
    assert streamed.content == non_stream["choices"][0]["text"]
    assert streamed.finish_reason == non_stream["choices"][0]["finish_reason"]


def test_t1_f13_04_chat_reasoning_and_reasoning_effort(client: KernClient) -> None:
    msgs = build_multiturn_messages([("user", "What is 14 * 12?")])
    no_think = client.chat_completions(msgs, max_tokens=128, temperature=0, reasoning_effort="none")
    no_msg = no_think["choices"][0]["message"]
    assert "reasoning_content" not in no_msg
    assert "168" in (no_msg.get("content") or "")
    no_r_toks = no_think["usage"]["completion_tokens_details"]["reasoning_tokens"]

    high_think = client.chat_completions(msgs, max_tokens=512, temperature=0, reasoning_effort="high")
    hi_msg = high_think["choices"][0]["message"]
    assert "reasoning_content" not in hi_msg
    assert isinstance(hi_msg.get("reasoning"), str) and len(hi_msg["reasoning"]) > 0
    hi_r_toks = high_think["usage"]["completion_tokens_details"]["reasoning_tokens"]
    assert hi_r_toks > 0
    assert no_r_toks <= hi_r_toks


def test_t1_f13_05_chat_tools_produces_tool_calls(client: KernClient) -> None:
    tools = build_weather_tool_schema()
    msgs = build_multiturn_messages(
        [("user", "Call get_weather for Tokyo in celsius.")],
        system_prompt="You must use the provided tool to answer weather queries.",
    )
    resp = client.chat_completions(
        msgs, tools=tools, max_tokens=512, temperature=0, reasoning_effort="low"
    )
    choice = resp["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    tcs = choice["message"].get("tool_calls")
    assert isinstance(tcs, list) and len(tcs) >= 1
    assert tcs[0]["function"]["name"] == "get_weather"
    parsed_args = json.loads(tcs[0]["function"]["arguments"])
    assert "city" in parsed_args


def test_t1_f13_06_prefix_cache_hit_and_output_parity(client: KernClient) -> None:
    sys_body = "Detailed archival specification paragraph. " * 25
    # kern-serve's recurrent scheduler intentionally skips caching a finished request
    # ("not kept") if its final speculative round accepted >= 1 token past EOS.
    # Retry with a fresh nonce if Turn 1 hits that rare post-EOS speculative overshoot.
    r1 = r2 = before = after = t2_ids = cold_t2_ids = None
    for _attempt in range(4):
        nonce = uuid.uuid4().hex[:8]
        t1_msgs = build_multiturn_messages(
            [("user", "Acknowledge with just the single word: OK.")],
            system_prompt=f"Context-{nonce}: {sys_body}",
        )
        before = client.metrics()
        r1 = client.chat_completions(t1_msgs, max_tokens=256, temperature=0, reasoning_effort="none")
        assert r1["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
        assert r1["choices"][0]["finish_reason"] == "stop"

        m1 = r1["choices"][0]["message"]
        asst_msg: dict[str, Any] = {"role": "assistant", "content": m1.get("content") or "OK"}
        if m1.get("reasoning"):
            asst_msg["reasoning_content"] = m1["reasoning"]
        t2_msgs = [
            *t1_msgs,
            asst_msg,
            {"role": "user", "content": "What is 2 + 2? Reply with just the integer."},
        ]
        t2_ids = client.tokenize(
            messages=t2_msgs,
            chat_template_kwargs={"reasoning_effort": "none"},
        )["tokens"]
        r2 = client.completions(t2_ids, max_tokens=32, temperature=0, return_token_ids=True)
        after = client.metrics()
        if r2["usage"]["prompt_tokens_details"]["cached_tokens"] >= 100:
            cold_nonce = uuid.uuid4().hex[:8]
            cold_t2_msgs = [
                *build_multiturn_messages(
                    [("user", "Acknowledge with just the single word: OK.")],
                    system_prompt=f"Context-{cold_nonce}: {sys_body}",
                ),
                asst_msg,
                {"role": "user", "content": "What is 2 + 2? Reply with just the integer."},
            ]
            cold_t2_ids = client.tokenize(
                messages=cold_t2_msgs,
                chat_template_kwargs={"reasoning_effort": "none"},
            )["tokens"]
            break

    assert r2 is not None and before is not None and after is not None and t2_ids is not None and cold_t2_ids is not None
    assert r2["usage"]["prompt_tokens_details"]["cached_tokens"] >= 100
    assert r2["usage"]["prompt_tokens_details"]["cached_tokens"] > 0
    assert metric_delta(before, after, "vllm:prefix_cache_queries_total") >= 2.0
    assert metric_delta(before, after, "vllm:prefix_cache_hits_total") >= 1.0
    assert metric_delta(before, after, "vllm:prompt_tokens_cached_total") > 0

    r2_repeat = client.completions(t2_ids, max_tokens=32, temperature=0, return_token_ids=True)
    assert r2_repeat["choices"][0]["token_ids"] == r2["choices"][0]["token_ids"]

    r2_cold = client.completions(cold_t2_ids, max_tokens=32, temperature=0, return_token_ids=True)
    assert r2_cold["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
    assert "4" in r2["choices"][0]["text"]
    assert "4" in r2_cold["choices"][0]["text"]
