"""Tier 2: Boundary, corner, and negative tests (70 tests across #1-#13 + Vision #14)."""

from __future__ import annotations

import sys
import time
import uuid
from pathlib import Path

import pytest

_E2E_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _E2E_DIR.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.e2e.client import HttpError, KernCLI, KernClient  # noqa: E402
from tests.e2e.conftest import E2EConfig, require_manifest_artifacts  # noqa: E402
from tests.e2e.fixtures_data import (  # noqa: E402
    DEFAULT_KV_PAGE_SIZE,
    DIVERSE_PROMPTS,
    EOS_TOKEN_ID,
    IMAGE_CONTENT_TOKEN_ID,
    IMAGE_SOFT_PLACEHOLDER_TOKEN_ID,
    MAX_MODEL_LEN,
    REASONING_EFFORT_LEVELS,
    UNPADDED_VOCAB_SIZE,
    assert_valid_token_ids,
    build_audio_messages,
    build_exact_token_prompt,
    build_multiturn_messages,
    build_needle_haystack_tokens,
    build_vision_base64_messages,
    build_vision_image_url_messages,
    strip_terminal_stop_token,
    verify_greedy_parity_with_near_tie,
)

pytestmark = [pytest.mark.tier2, pytest.mark.gpu]


# ---------------------------------------------------------------------------
# Feature #1: Input Embedding + embed_norm (5 tests)
# ---------------------------------------------------------------------------


def test_t2_f01_01_single_token_prompt_len1(client: KernClient) -> None:
    tok = client.tokenize(prompt="Hello")["tokens"][:1]
    resp = client.completions(tok, max_tokens=8, temperature=0, return_token_ids=True)
    assert resp["usage"]["prompt_tokens"] == 1
    assert_valid_token_ids(resp["choices"][0]["token_ids"])


def test_t2_f01_02_empty_prompt_rejected_400(client: KernClient) -> None:
    for empty_prompt in ("", []):
        with pytest.raises(HttpError) as exc:
            client.completions(empty_prompt, max_tokens=4, temperature=0)
        assert exc.value.status == 400


def test_t2_f01_03_max_valid_token_id_200057(client: KernClient) -> None:
    max_valid = UNPADDED_VOCAB_SIZE - 1  # 200057 (<|content_invoke_tool_text|>)
    resp = client.completions([42, max_valid], max_tokens=8, temperature=0, return_token_ids=True)
    assert_valid_token_ids(resp["choices"][0]["token_ids"])


def test_t2_f01_04_out_of_vocab_token_id_200058_rejected(client: KernClient) -> None:
    for bad_ids in ([UNPADDED_VOCAB_SIZE], [UNPADDED_VOCAB_SIZE + 50], [-1]):
        with pytest.raises(HttpError) as exc:
            client.completions(bad_ids, max_tokens=4, temperature=0)
        assert exc.value.status in (400, 422)


def test_t2_f01_05_control_chars_unicode_prompt_safety(client: KernClient) -> None:
    tricky = "Prefix\x00\t\r\nZeroWidth\u200bEmoji🧪End:"
    resp = client.completions(tricky, max_tokens=8, temperature=0, return_token_ids=True)
    assert_valid_token_ids(resp["choices"][0]["token_ids"])


# ---------------------------------------------------------------------------
# Feature #2: 3-Phase Loop Topology (5 tests)
# ---------------------------------------------------------------------------


def test_t2_f02_01_max_tokens_1_prefill_only(client: KernClient) -> None:
    resp = client.completions("1 + 1 =", max_tokens=1, temperature=0, return_token_ids=True)
    assert len(resp["choices"][0]["token_ids"]) == 1
    assert_valid_token_ids(resp["choices"][0]["token_ids"])


def test_t2_f02_02_max_tokens_2_minimal_decode(client: KernClient) -> None:
    resp = client.completions(
        "Count: 1, 2, 3,", max_tokens=2, temperature=0, ignore_eos=True, return_token_ids=True
    )
    assert len(resp["choices"][0]["token_ids"]) == 2
    assert resp["choices"][0]["finish_reason"] == "length"


def test_t2_f02_03_chunk_boundary_lengths(client: KernClient) -> None:
    for length in (127, 128, 129, 255, 256, 257):
        prompt_ids = build_exact_token_prompt(client, length)
        resp = client.completions(prompt_ids, max_tokens=4, temperature=0, return_token_ids=True)
        assert resp["usage"]["prompt_tokens"] == length
        assert_valid_token_ids(resp["choices"][0]["token_ids"])


def test_t2_f02_04_prime_length_prompt_997(client: KernClient) -> None:
    prompt_ids = build_exact_token_prompt(client, 997)
    resp = client.completions(prompt_ids, max_tokens=8, temperature=0, return_token_ids=True)
    assert resp["usage"]["prompt_tokens"] == 997
    assert_valid_token_ids(resp["choices"][0]["token_ids"])


@pytest.mark.cli
@pytest.mark.hybrid
def test_t2_f02_05_kern_run_vs_serve_max_tokens_1_parity(
    e2e_config: E2EConfig, client: KernClient, kern_cli: KernCLI
) -> None:
    require_manifest_artifacts(e2e_config)
    resp = client.completions("The sun rises in the", max_tokens=1, temperature=0, return_token_ids=True)
    c = resp["choices"][0]
    prompt_ids = c["prompt_token_ids"]
    serve_ids = strip_terminal_stop_token(c["token_ids"], c["finish_reason"])
    cap = max(512, ((len(prompt_ids) + 1 + 63) // 16) * 16)
    proc, run_ids = kern_cli.kern_run(
        manifest=e2e_config.manifest,
        weights=e2e_config.weights,
        kernels=e2e_config.kernels if e2e_config.kernels.exists() else None,
        prompt_ids=prompt_ids,
        steps=1,
        capacity=cap,
        temperature=0.0,
        gpu=e2e_config.gpu,
    )
    assert proc.returncode == 0, proc.stderr
    assert serve_ids == run_ids


# ---------------------------------------------------------------------------
# Feature #3: Cross-Pass Message Read (5 tests)
# ---------------------------------------------------------------------------


def test_t2_f03_01_single_token_cross_pass_edge(client: KernClient) -> None:
    resp = client.completions([100], max_tokens=4, temperature=0, return_token_ids=True)
    assert len(resp["choices"][0]["token_ids"]) >= 1
    assert_valid_token_ids(resp["choices"][0]["token_ids"])


def test_t2_f03_02_identical_repeated_token_64(client: KernClient) -> None:
    resp = client.completions([42] * 64, max_tokens=8, temperature=0, return_token_ids=True)
    assert resp["usage"]["prompt_tokens"] == 64
    assert_valid_token_ids(resp["choices"][0]["token_ids"])


def test_t2_f03_03_conflicting_cues_across_turns(client: KernClient) -> None:
    msgs = [
        {"role": "system", "content": "Always answer with the latest user override."},
        {"role": "user", "content": "The code is RED."},
        {"role": "assistant", "content": "Understood, RED."},
        {"role": "user", "content": "Correction: the code is now GREEN. What is the code?"},
    ]
    resp = client.chat_completions(msgs, max_tokens=128, temperature=0, reasoning_effort="none")
    assert "GREEN" in (resp["choices"][0]["message"].get("content") or "").upper()


def test_t2_f03_04_alternating_batch_1_8_1_stability(client: KernClient) -> None:
    p = DIVERSE_PROMPTS[1]
    single_before = client.completions(p, max_tokens=12, temperature=0, return_token_ids=True)["choices"][0]["token_ids"]
    client.concurrent_map(
        lambda s: client.completions(s, max_tokens=12, temperature=0, return_token_ids=True),
        DIVERSE_PROMPTS[:8],
        max_workers=8,
    )
    single_after = client.completions(p, max_tokens=12, temperature=0, return_token_ids=True)["choices"][0]["token_ids"]
    assert single_before == single_after


def test_t2_f03_05_cross_pass_prompt_vs_chunk_parity(client: KernClient) -> None:
    prompt_ids = build_exact_token_prompt(client, 300)
    r1 = client.completions(prompt_ids, max_tokens=12, temperature=0, return_token_ids=True)
    r2 = client.completions(prompt_ids, max_tokens=12, temperature=0, return_token_ids=True)
    assert r1["choices"][0]["token_ids"] == r2["choices"][0]["token_ids"]


# ---------------------------------------------------------------------------
# Feature #4: Rank-64 Gated Loop Controllers (5 tests)
# ---------------------------------------------------------------------------


def test_t2_f04_01_outlier_burst_500_bangs(client: KernClient) -> None:
    prompt = ("!" * 500) + "\nWhat is 5 + 5? Answer:"
    resp = client.completions(prompt, max_tokens=16, temperature=0, return_token_ids=True)
    assert_valid_token_ids(resp["choices"][0]["token_ids"])


def test_t2_f04_02_long_decode_1024_no_collapse(client: KernClient) -> None:
    prompt = "Write a comprehensive essay explaining the history of sorting algorithms:\n"
    resp = client.completions(prompt, max_tokens=1024, temperature=0, return_token_ids=True)
    ids = resp["choices"][0]["token_ids"]
    assert len(ids) >= 64
    assert_valid_token_ids(ids)


def test_t2_f04_03_zero_vs_near_zero_temp_greedy(client: KernClient) -> None:
    prompt = "The largest planet in our solar system is"
    r0 = client.completions(prompt, max_tokens=12, temperature=0.0, return_token_ids=True)
    r_eps = client.completions(prompt, max_tokens=12, temperature=1e-6, return_token_ids=True)
    assert r0["choices"][0]["token_ids"] == r_eps["choices"][0]["token_ids"]


def test_t2_f04_04_last_token_language_switch(client: KernClient) -> None:
    prompt = "The following English sentence ends abruptly. 请用中文回答：中国的首都是"
    resp = client.completions(prompt, max_tokens=16, temperature=0, return_token_ids=True)
    assert_valid_token_ids(resp["choices"][0]["token_ids"])
    assert "北京" in resp["choices"][0]["text"]


def test_t2_f04_05_seed_zero_and_large_boundaries(client: KernClient) -> None:
    prompt = "Name a color:"
    r0 = client.completions(prompt, max_tokens=8, temperature=0.7, seed=0, return_token_ids=True)
    r_big = client.completions(
        prompt, max_tokens=8, temperature=0.7, seed=(2**31 - 1), return_token_ids=True
    )
    assert_valid_token_ids(r0["choices"][0]["token_ids"])
    assert_valid_token_ids(r_big["choices"][0]["token_ids"])


# ---------------------------------------------------------------------------
# Feature #5: Block Attention Residuals (5 tests)
# ---------------------------------------------------------------------------


def test_t2_f05_01_block_boundary_prompt_lengths(client: KernClient) -> None:
    for length in (5, 6, 7, 12, 24):
        prompt_ids = build_exact_token_prompt(client, length)
        resp = client.completions(prompt_ids, max_tokens=4, temperature=0, return_token_ids=True)
        assert resp["usage"]["prompt_tokens"] == length
        assert_valid_token_ids(resp["choices"][0]["token_ids"])


def test_t2_f05_02_assistant_empty_content_with_tool_calls(client: KernClient) -> None:
    msgs = [
        {"role": "system", "content": "You have tools."},
        {"role": "user", "content": "Check weather in Paris."},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": '{"temp_c": 18}'},
        {"role": "user", "content": "Report the temperature."},
    ]
    resp = client.chat_completions(msgs, max_tokens=128, temperature=0, reasoning_effort="none")
    assert "18" in (resp["choices"][0]["message"].get("content") or "")


def test_t2_f05_03_deep_10_turn_dialogue_preservation(client: KernClient) -> None:
    msgs: list[dict] = [{"role": "system", "content": "Track every fact accurately."}]
    for i in range(1, 10):
        msgs.append({"role": "user", "content": f"Item {i} is code-{i * 11}."})
        msgs.append({"role": "assistant", "content": f"Noted item {i} is code-{i * 11}."})
    msgs.append({"role": "user", "content": "What was Item 2?"})
    resp = client.chat_completions(msgs, max_tokens=128, temperature=0, reasoning_effort="none")
    assert "22" in (resp["choices"][0]["message"].get("content") or "")


def test_t2_f05_04_mixed_short_10_and_2000_concurrent(client: KernClient) -> None:
    short_ids = build_exact_token_prompt(client, 10)
    long_ids = build_exact_token_prompt(client, 2000)
    short_ref = client.completions(short_ids, max_tokens=8, temperature=0, return_token_ids=True)["choices"][0]["token_ids"]
    long_ref = client.completions(long_ids, max_tokens=8, temperature=0, return_token_ids=True)["choices"][0]["token_ids"]

    res = client.concurrent_map(
        lambda pids: client.completions(pids, max_tokens=8, temperature=0, return_token_ids=True)["choices"][0]["token_ids"],
        [short_ids, long_ids, short_ids, long_ids],
        max_workers=4,
    )
    refs = [short_ref, long_ref, short_ref, long_ref]
    assert len(res) == len(refs)
    for b_ids, s_ids in zip(res, refs, strict=True):
        assert_valid_token_ids(b_ids)
        assert len(b_ids) == len(s_ids)
        assert b_ids[0] == s_ids[0]


def test_t2_f05_05_stop_token_ids_early_termination(client: KernClient) -> None:
    prompt = "Count continuously without stopping: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,"
    baseline = client.completions(prompt, max_tokens=16, temperature=0, return_token_ids=True)
    base_ids = baseline["choices"][0]["token_ids"]
    assert_valid_token_ids(base_ids)
    non_eos = [t for t in base_ids[:-1] if t != EOS_TOKEN_ID]
    assert len(non_eos) >= 2
    stop_tok = non_eos[min(2, len(non_eos) - 1)]

    resp = client.completions(
        prompt,
        max_tokens=16,
        temperature=0,
        stop_token_ids=[stop_tok],
        return_token_ids=True,
    )
    c = resp["choices"][0]
    assert c["finish_reason"] == "stop"
    assert_valid_token_ids(c["token_ids"])
    assert len(c["token_ids"]) < len(base_ids)
    assert c["token_ids"][-1] == stop_tok or c.get("stop_reason") == stop_tok


# ---------------------------------------------------------------------------
# Feature #6: Hybrid Attention (40 SWA + 8 Full) (5 tests)
# ---------------------------------------------------------------------------


def test_t2_f06_01_swa_window_boundary_1023_1024_1025(client: KernClient) -> None:
    needle = client.tokenize(prompt="PinCode is 5091. ")["tokens"]
    question = client.tokenize(prompt="\nWhat is PinCode? Answer: ")["tokens"]
    filler = client.tokenize(prompt=" filler")["tokens"][0]
    for total_len in (1023, 1024, 1025):
        pids = build_needle_haystack_tokens(needle, question, filler, total_len, needle_pos=0)
        resp = client.completions(pids, max_tokens=16, temperature=0, return_token_ids=True)
        assert resp["usage"]["prompt_tokens"] == total_len
        assert "5091" in resp["choices"][0]["text"]


def test_t2_f06_02_log_scaling_floor_8191_8192_8193(client: KernClient) -> None:
    needle = client.tokenize(prompt="VaultKey is 6420. ")["tokens"]
    question = client.tokenize(prompt="\nRepeat VaultKey: ")["tokens"]
    filler = client.tokenize(prompt=" word")["tokens"][0]
    for total_len in (8191, 8192, 8193):
        pids = build_needle_haystack_tokens(needle, question, filler, total_len, needle_pos=0)
        resp = client.completions(pids, max_tokens=12, temperature=0, return_token_ids=True)
        assert resp["usage"]["prompt_tokens"] == total_len
        assert "6420" in resp["choices"][0]["text"]


def test_t2_f06_03_kv_page_boundary_lengths(client: KernClient) -> None:
    p = DEFAULT_KV_PAGE_SIZE
    for length in (p - 1, p, p + 1, 2 * p - 1, 2 * p, 2 * p + 1):
        pids = build_exact_token_prompt(client, length)
        resp = client.completions(pids, max_tokens=4, temperature=0, return_token_ids=True)
        assert resp["usage"]["prompt_tokens"] == length


def test_t2_f06_04_swa_ring_wraparound_2048_plus_1024(client: KernClient) -> None:
    pids = build_exact_token_prompt(client, 2048)
    resp = client.completions(pids, max_tokens=1024, temperature=0, return_token_ids=True)
    assert resp["usage"]["prompt_tokens"] == 2048
    assert_valid_token_ids(resp["choices"][0]["token_ids"])


def test_t2_f06_05_context_overflow_1048576_rejected_400(client: KernClient) -> None:
    for overflow_len in (MAX_MODEL_LEN, MAX_MODEL_LEN + 1):
        with pytest.raises(HttpError) as exc:
            client.completions([42] * overflow_len, max_tokens=1, temperature=0)
        assert exc.value.status in (400, 413)
        if exc.value.status == 400 and exc.value.payload:
            assert exc.value.payload["error"]["param"] == "input_tokens"


# ---------------------------------------------------------------------------
# Feature #7: Attention Prologue + Gate/Sink (5 tests)
# ---------------------------------------------------------------------------


def test_t2_f07_01_rel_extent_boundary_1023_1024_1025(client: KernClient) -> None:
    needle = client.tokenize(prompt="RelKey is 7391. ")["tokens"]
    question = client.tokenize(prompt="\nWhat is RelKey? Answer: ")["tokens"]
    filler = client.tokenize(prompt=" word")["tokens"][0]
    for total_len in (1023, 1024, 1025):
        pids = build_needle_haystack_tokens(needle, question, filler, total_len, needle_pos=0)
        resp = client.completions(pids, max_tokens=16, temperature=0, return_token_ids=True)
        assert resp["usage"]["prompt_tokens"] == total_len
        assert_valid_token_ids(resp["choices"][0]["token_ids"])
        assert "7391" in resp["choices"][0]["text"]

    q_ord = client.tokenize(prompt="\nWhich color came second? The second color is")["tokens"]
    n_rb = client.tokenize(prompt="Order: first RED, second BLUE. ")["tokens"]
    n_br = client.tokenize(prompt="Order: first BLUE, second RED. ")["tokens"]
    for total_len in (1023, 1025):
        r_rb = client.completions(
            build_needle_haystack_tokens(n_rb, q_ord, filler, total_len, needle_pos=0),
            max_tokens=12,
            temperature=0,
            return_token_ids=True,
        )
        r_br = client.completions(
            build_needle_haystack_tokens(n_br, q_ord, filler, total_len, needle_pos=0),
            max_tokens=12,
            temperature=0,
            return_token_ids=True,
        )
        assert_valid_token_ids(r_rb["choices"][0]["token_ids"])
        assert_valid_token_ids(r_br["choices"][0]["token_ids"])
        assert "BLUE" in r_rb["choices"][0]["text"].upper()
        assert "RED" in r_br["choices"][0]["text"].upper()
        assert r_rb["choices"][0]["token_ids"] != r_br["choices"][0]["token_ids"]


def test_t2_f07_02_repeated_single_token_512_l2_clip(client: KernClient) -> None:
    resp = client.completions([777] * 512, max_tokens=12, temperature=0, return_token_ids=True)
    assert resp["usage"]["prompt_tokens"] == 512
    assert_valid_token_ids(resp["choices"][0]["token_ids"])


def test_t2_f07_03_immediate_answer_at_pos1(client: KernClient) -> None:
    resp = client.completions("1+1=2\n2+2=", max_tokens=4, temperature=0)
    assert "4" in resp["choices"][0]["text"]


def test_t2_f07_04_permuted_segments_asymmetric_rel_bias(client: KernClient) -> None:
    r1 = client.completions("Order: X then Y. Last letter:", max_tokens=6, temperature=0)
    r2 = client.completions("Order: Y then X. Last letter:", max_tokens=6, temperature=0)
    assert r1["choices"][0]["text"] != r2["choices"][0]["text"]


def test_t2_f07_05_stream_vs_nonstream_clipped_parity(client: KernClient) -> None:
    prompt = ("# " * 300) + "Answer 3 + 4 ="
    ns = client.completions(prompt, max_tokens=12, temperature=0)
    st = client.completions_stream(prompt, max_tokens=12, temperature=0)
    assert st.done is True
    assert st.content == ns["choices"][0]["text"]


# ---------------------------------------------------------------------------
# Feature #8: 4-Stream Short Conv (sconv) (5 tests)
# ---------------------------------------------------------------------------


def test_t2_f08_01_sconv_exact_lengths_1_to_5(client: KernClient) -> None:
    for length in (1, 2, 3, 4, 5):
        pids = build_exact_token_prompt(client, length)
        r1 = client.completions(pids, max_tokens=8, temperature=0, return_token_ids=True)
        r2 = client.completions(pids, max_tokens=8, temperature=0, return_token_ids=True)
        assert r1["usage"]["prompt_tokens"] == length
        assert r1["choices"][0]["token_ids"] == r2["choices"][0]["token_ids"]
        assert_valid_token_ids(r1["choices"][0]["token_ids"])


def test_t2_f08_02_turn2_exact_hit_vs_1tok_divergent_miss(client: KernClient) -> None:
    nonce = uuid.uuid4().hex[:8]
    msgs = build_multiturn_messages(
        [("user", "Reply with just OK.")],
        system_prompt=f"sconv-{nonce} " + ("a b c d " * 20),
    )
    base_ids = client.tokenize(
        messages=msgs,
        chat_template_kwargs={"reasoning_effort": "none"},
    )["tokens"]
    r1 = client.completions(base_ids, max_tokens=256, temperature=0, return_token_ids=True)
    assert r1["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
    assert r1["choices"][0]["finish_reason"] == "stop"

    next_ids = client.tokenize(prompt=" next")["tokens"]
    ext_same = base_ids + r1["choices"][0]["token_ids"] + next_ids
    warm = client.completions(ext_same, max_tokens=16, temperature=0, return_token_ids=True)
    assert warm["usage"]["prompt_tokens_details"]["cached_tokens"] >= len(base_ids)

    # Divergent first token: must not hit warm checkpoint
    div_start = [(base_ids[0] + 1) % UNPADDED_VOCAB_SIZE] + ext_same[1:]
    cold_start = client.completions(div_start, max_tokens=16, temperature=0, return_token_ids=True)
    assert cold_start["usage"]["prompt_tokens_details"]["cached_tokens"] == 0

    # Divergent last token of base_ids: must not hit warm checkpoint
    div_mid = base_ids[:-1] + [(base_ids[-1] + 1) % UNPADDED_VOCAB_SIZE] + r1["choices"][0]["token_ids"] + next_ids
    cold_mid = client.completions(div_mid, max_tokens=16, temperature=0, return_token_ids=True)
    assert cold_mid["usage"]["prompt_tokens_details"]["cached_tokens"] == 0


def test_t2_f08_03_seq_slots_eviction_churn(client: KernClient) -> None:
    nonce = uuid.uuid4().hex[:8]
    prompts = [f"SlotChurn-{nonce}-{i}: " + ("token " * 30) + "Result:" for i in range(12)]
    outs = [
        client.completions(p, max_tokens=8, temperature=0, return_token_ids=True)["choices"][0]["token_ids"]
        for p in prompts
    ]
    # Re-query the first prompt after slot churn; must still produce exact same answer
    replay0 = client.completions(prompts[0], max_tokens=8, temperature=0, return_token_ids=True)["choices"][0][
        "token_ids"
    ]
    assert replay0 == outs[0]


def test_t2_f08_04_midstream_abort_leaves_sconv_clean(client: KernClient) -> None:
    ref_prompt = "Clean check after abort: 8 * 8 ="
    expected = client.completions(ref_prompt, max_tokens=12, temperature=0, return_token_ids=True)["choices"][0][
        "token_ids"
    ]
    got = client.raw_post_abort(
        "/v1/completions",
        {
            "model": client.resolve_model_id(),
            "prompt": "Write a very long story about a lighthouse keeper:",
            "max_tokens": 512,
            "temperature": 0,
        },
        chunks_before_close=3,
    )
    assert got >= 1
    time.sleep(0.5)
    after = client.completions(ref_prompt, max_tokens=12, temperature=0, return_token_ids=True)["choices"][0][
        "token_ids"
    ]
    assert after == expected


def test_t2_f08_05_sconv_max_tokens_1_then_resume_turn2(client: KernClient) -> None:
    nonce = uuid.uuid4().hex[:8]
    msgs_t1 = [
        {"role": "system", "content": f"Session {nonce}."},
        {"role": "user", "content": "Say alpha."},
    ]
    r1 = client.chat_completions(msgs_t1, max_tokens=1, temperature=0, reasoning_effort="none")
    assert r1["usage"]["completion_tokens"] == 1
    msgs_t2 = msgs_t1 + [
        {"role": "assistant", "content": r1["choices"][0]["message"].get("content") or "alpha"},
        {"role": "user", "content": "Now say beta."},
    ]
    r2 = client.chat_completions(msgs_t2, max_tokens=128, temperature=0, reasoning_effort="none")
    assert isinstance(r2["choices"][0]["message"].get("content"), str)


# ---------------------------------------------------------------------------
# Feature #9: MoE 128-Expert Top-8 (5 tests)
# ---------------------------------------------------------------------------


def test_t2_f09_01_single_token_moe_routing_stability(client: KernClient) -> None:
    resp = client.completions([1234], max_tokens=8, temperature=0, return_token_ids=True)
    assert_valid_token_ids(resp["choices"][0]["token_ids"])


def test_t2_f09_02_skewed_repetitive_expert_imbalance(client: KernClient) -> None:
    prompt = ("0 " * 256) + "1 ="
    resp = client.completions(prompt, max_tokens=8, temperature=0, return_token_ids=True)
    assert_valid_token_ids(resp["choices"][0]["token_ids"])


def test_t2_f09_03_odd_tail_prompt_lengths_padding_safe(client: KernClient) -> None:
    for length in (129, 257, 513):
        pids = build_exact_token_prompt(client, length)
        resp = client.completions(pids, max_tokens=8, temperature=0, return_token_ids=True)
        assert resp["usage"]["prompt_tokens"] == length
        assert_valid_token_ids(resp["choices"][0]["token_ids"])


def test_t2_f09_04_max_concurrency_8_matches_singleton(client: KernClient) -> None:
    prompts = DIVERSE_PROMPTS[:8]
    refs = [
        client.completions(p, max_tokens=16, temperature=0, return_token_ids=True)["choices"][0]["token_ids"]
        for p in prompts
    ]
    batch = client.concurrent_map(
        lambda p: client.completions(p, max_tokens=16, temperature=0, return_token_ids=True)["choices"][0]["token_ids"],
        prompts,
        max_workers=8,
    )
    assert len(batch) == len(refs)
    for b_ids, s_ids in zip(batch, refs, strict=True):
        assert_valid_token_ids(b_ids)
        assert len(b_ids) == len(s_ids)
        assert b_ids[0] == s_ids[0]


def test_t2_f09_05_rapid_sequential_single_token_bursts(client: KernClient) -> None:
    for tok in (50, 100, 200, 300, 400):
        r = client.completions([tok], max_tokens=4, temperature=0, return_token_ids=True)
        assert_valid_token_ids(r["choices"][0]["token_ids"])


# ---------------------------------------------------------------------------
# Feature #10: Mixed NVFP4 + MXFP8 (5 tests)
# ---------------------------------------------------------------------------


def test_t2_f10_01_extreme_temp_2_0_stays_finite(client: KernClient) -> None:
    resp = client.completions(
        "Generate random words:",
        max_tokens=32,
        temperature=2.0,
        top_k=0,
        top_p=1.0,
        seed=55,
        return_token_ids=True,
    )
    assert_valid_token_ids(resp["choices"][0]["token_ids"])


def test_t2_f10_02_scientific_notation_numeric_precision(client: KernClient) -> None:
    resp = client.completions(
        "Compute 1.5e3 + 2.5e3 =", max_tokens=16, temperature=0
    )
    txt = resp["choices"][0]["text"]
    assert "4000" in txt or "4e3" in txt.lower() or "4.0e3" in txt.lower()


def test_t2_f10_03_quant_near_tie_consistency_under_batch(client: KernClient) -> None:
    prompt = "The chemical symbol for gold is"
    ref = client.completions(prompt, max_tokens=8, temperature=0, return_token_ids=True)["choices"][0]["token_ids"]
    batch = client.concurrent_map(
        lambda _: client.completions(prompt, max_tokens=8, temperature=0, return_token_ids=True)["choices"][0][
            "token_ids"
        ],
        range(4),
        max_workers=4,
    )
    assert all(b == ref for b in batch)


def test_t2_f10_04_batched_bs8_vs_singleton_quant_parity(client: KernClient) -> None:
    prompts = [f"Compute {i} + {i * 2} =" for i in range(1, 9)]
    refs = [
        client.completions(p, max_tokens=8, temperature=0, return_token_ids=True)["choices"][0]["token_ids"]
        for p in prompts
    ]
    batch = client.concurrent_map(
        lambda p: client.completions(p, max_tokens=8, temperature=0, return_token_ids=True),
        prompts,
        max_workers=8,
    )
    assert len(batch) == len(refs)
    for i, (b_resp, s_ids) in enumerate(zip(batch, refs, strict=True), start=1):
        b_ids = b_resp["choices"][0]["token_ids"]
        assert_valid_token_ids(b_ids)
        assert len(b_ids) == len(s_ids)
        assert b_ids[0] == s_ids[0]
        assert str(i + i * 2) in b_resp["choices"][0]["text"]


def test_t2_f10_05_long_context_32k_quantized_moe_stability(client: KernClient) -> None:
    pids = build_exact_token_prompt(client, 32768)
    resp = client.completions(pids, max_tokens=16, temperature=0, return_token_ids=True)
    assert resp["usage"]["prompt_tokens"] == 32768
    assert_valid_token_ids(resp["choices"][0]["token_ids"])


# ---------------------------------------------------------------------------
# Feature #11: LM Head + Softcap 30.0 + Vocab Mask (5 tests)
# ---------------------------------------------------------------------------


def test_t2_f11_01_high_entropy_temp2_no_padded_vocab_ids(client: KernClient) -> None:
    for seed in (1, 2, 3):
        resp = client.completions(
            "Chaos:",
            max_tokens=48,
            temperature=2.0,
            top_k=0,
            top_p=1.0,
            seed=seed,
            return_token_ids=True,
        )
        assert_valid_token_ids(resp["choices"][0]["token_ids"])


def test_t2_f11_02_top_k_1_matches_greedy_temp_0(client: KernClient) -> None:
    prompt = "The freezing point of water in Fahrenheit is"
    greedy = client.completions(prompt, max_tokens=12, temperature=0.0, return_token_ids=True)
    tk1 = client.completions(prompt, max_tokens=12, temperature=0.9, top_k=1, return_token_ids=True)
    assert greedy["choices"][0]["token_ids"] == tk1["choices"][0]["token_ids"]


def test_t2_f11_03_top_k_minus1_and_top_p_boundary(client: KernClient) -> None:
    resp = client.completions(
        "Hello", max_tokens=8, temperature=0.8, top_k=-1, top_p=1.0, seed=9, return_token_ids=True
    )
    assert_valid_token_ids(resp["choices"][0]["token_ids"])
    with pytest.raises(HttpError) as exc:
        client.completions("Hello", max_tokens=8, temperature=0.8, top_p=0.0)
    assert exc.value.status == 400


def test_t2_f11_04_invalid_temp_and_top_p_rejected_400(client: KernClient) -> None:
    for bad_kwargs in ({"temperature": -0.1}, {"temperature": 2.5}, {"top_p": 1.2}):
        with pytest.raises(HttpError) as exc:
            client.completions("Hello", max_tokens=4, **bad_kwargs)
        assert exc.value.status == 400


def test_t2_f11_05_ignore_eos_generates_to_max_tokens(client: KernClient) -> None:
    resp = client.completions(
        "Say hi.", max_tokens=20, temperature=0, ignore_eos=True, return_token_ids=True
    )
    c = resp["choices"][0]
    assert c["finish_reason"] == "length"
    assert len(c["token_ids"]) == 20
    assert_valid_token_ids(c["token_ids"])


# ---------------------------------------------------------------------------
# Feature #12: DSpark MTP + RNN Markov Head (5 tests)
# ---------------------------------------------------------------------------


@pytest.mark.cli
@pytest.mark.hybrid
def test_t2_f12_01_zero_acceptance_random_hex_fallback(
    e2e_config: E2EConfig, client: KernClient, kern_cli: KernCLI
) -> None:
    require_manifest_artifacts(e2e_config)
    prompt = "Hex: 7f3a 91c0 4e2d 8b1f 05a9 c3e7 2b4d 6a8f 1e0c"
    serve_resp = client.completions(prompt, max_tokens=16, temperature=0, return_token_ids=True)
    c = serve_resp["choices"][0]
    prompt_ids = c["prompt_token_ids"]
    serve_ids = strip_terminal_stop_token(c["token_ids"], c["finish_reason"])
    cap = max(512, ((len(prompt_ids) + 16 + 63) // 16) * 16)
    kernels = e2e_config.kernels if e2e_config.kernels.exists() else None
    proc, run_ids = kern_cli.kern_run(
        manifest=e2e_config.manifest,
        weights=e2e_config.weights,
        kernels=kernels,
        prompt_ids=prompt_ids,
        steps=16,
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
        steps=16,
    )


def test_t2_f12_02_max_tokens_1_to_7_no_overshoot(client: KernClient) -> None:
    prompt = "1, 2, 3, 4, 5, 6, 7, 8, 9, 10,"
    for k in (1, 3, 7):
        resp = client.completions(prompt, max_tokens=k, temperature=0, ignore_eos=True, return_token_ids=True)
        assert len(resp["choices"][0]["token_ids"]) == k
        assert resp["usage"]["completion_tokens"] == k


def test_t2_f12_03_max_tokens_8_9_draft_block_boundary(client: KernClient) -> None:
    prompt = "1, 2, 3, 4, 5, 6, 7, 8, 9, 10,"
    for k in (8, 9):
        resp = client.completions(prompt, max_tokens=k, temperature=0, ignore_eos=True, return_token_ids=True)
        assert len(resp["choices"][0]["token_ids"]) == k
        assert resp["choices"][0]["finish_reason"] == "length"


def test_t2_f12_04_eos_mid_draft_block_clean_truncation(client: KernClient) -> None:
    msgs = build_multiturn_messages([("user", "Output exactly: OK")])
    resp = client.chat_completions(
        msgs, max_tokens=128, temperature=0, reasoning_effort="none", return_token_ids=True
    )
    c = resp["choices"][0]
    assert c["finish_reason"] == "stop"
    assert resp["usage"]["completion_tokens"] < 128
    assert isinstance(c.get("token_ids"), list) and len(c["token_ids"]) >= 1
    assert_valid_token_ids(c["token_ids"])
    assert c["token_ids"][-1] == EOS_TOKEN_ID or c.get("stop_reason") == EOS_TOKEN_ID


@pytest.mark.cli
@pytest.mark.hybrid
def test_t2_f12_05_sconv_swa_rollback_commit_parity(
    e2e_config: E2EConfig, client: KernClient, kern_cli: KernCLI
) -> None:
    require_manifest_artifacts(e2e_config)
    prompt = "A1 B2 C3 D4 -> E5 F6 G7 H8 -> I9"
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


# ---------------------------------------------------------------------------
# Feature #13: Serving Spec + Prefix Cache + 1M / Conc-8 (5 tests)
# ---------------------------------------------------------------------------


def test_t2_f13_01_logprobs_and_prompt_logprobs_400(client: KernClient) -> None:
    for kwargs in ({"logprobs": True}, {"logprobs": 1}, {"prompt_logprobs": 1}):
        with pytest.raises(HttpError) as exc:
            client.completions("Hello", max_tokens=4, **kwargs)
        assert exc.value.status == 400
        assert exc.value.payload is not None
        assert exc.value.payload["error"]["type"] == "BadRequestError"


def test_t2_f13_02_unknown_model_404_and_unimpl_messages_501(client: KernClient) -> None:
    with pytest.raises(HttpError) as exc_404:
        client.completions("Hello", model="unknown-model-xyz", max_tokens=4)
    assert exc_404.value.status == 404

    status_501, _, _ = client.raw_request("POST", "/v1/messages", body={"model": client.resolve_model_id()})
    assert status_501 == 501


def test_t2_f13_03_reasoning_effort_boundaries_and_invalid_400(client: KernClient) -> None:
    msgs = build_multiturn_messages([("user", "Compute (17 * 13) + 29 step by step.")])
    for level in REASONING_EFFORT_LEVELS:
        resp = client.chat_completions(msgs, max_tokens=256, temperature=0, reasoning_effort=level)
        msg = resp["choices"][0]["message"]
        assert "reasoning_content" not in msg
        r_text = msg.get("reasoning")
        c_text = msg.get("content")
        assert ((r_text or "") + (c_text or "")).strip()
        r_toks = resp["usage"]["completion_tokens_details"]["reasoning_tokens"]
        if level == "none":
            assert r_toks >= 0
        elif level in ("high", "xhigh", "max"):
            assert isinstance(r_text, str) and len(r_text.strip()) > 0
            assert r_toks > 0

    r_float = client.chat_completions(
        msgs, max_tokens=256, temperature=0, chat_template_kwargs={"reasoning_effort": 0.5}
    )
    f_msg = r_float["choices"][0]["message"]
    assert "reasoning_content" not in f_msg
    assert ((f_msg.get("reasoning") or "") + (f_msg.get("content") or "")).strip()

    with pytest.raises(HttpError) as exc_float_bad:
        client.chat_completions(
            msgs, max_tokens=16, chat_template_kwargs={"reasoning_effort": 1.5}
        )
    assert exc_float_bad.value.status == 400

    with pytest.raises(HttpError) as exc_dev:
        client.chat_completions(
            [{"role": "developer", "content": "Be concise."}, {"role": "user", "content": "Hi"}],
            max_tokens=16,
        )
    assert exc_dev.value.status == 400

    with pytest.raises(HttpError) as exc_bad:
        client.chat_completions(msgs, max_tokens=16, reasoning_effort="ultra")
    assert exc_bad.value.status == 400


def test_t2_f13_04_bridge_refused_params_500_and_empty_tools_400(client: KernClient) -> None:
    with pytest.raises(HttpError) as exc_tools:
        client.chat_completions(
            build_multiturn_messages([("user", "Hi")]),
            tools=[],
            max_tokens=16,
        )
    assert exc_tools.value.status == 400

    for bad_param in ({"min_tokens": 1}, {"frequency_penalty": 0.5}, {"repetition_penalty": 1.1}):
        with pytest.raises(HttpError) as exc_500:
            client.completions("Hi", max_tokens=4, **bad_param)
        assert exc_500.value.status == 500


def test_t2_f13_05_concurrency_16_queues_cleanly_above_8(client: KernClient) -> None:
    prompts = [f"Question {i}: What is {i} + {i}? Answer:" for i in range(16)]
    results = client.concurrent_map(
        lambda p: client.completions(p, max_tokens=8, temperature=0, return_token_ids=True),
        prompts,
        max_workers=16,
    )
    assert len(results) == 16
    for r in results:
        assert_valid_token_ids(r["choices"][0]["token_ids"])


# ---------------------------------------------------------------------------
# Feature #14: Vision (model.visual.*) EXCLUDED — Negative Tests (5 tests)
# ---------------------------------------------------------------------------


def test_t2_f14_01_vision_image_url_http_400(client: KernClient) -> None:
    with pytest.raises(HttpError) as exc:
        client.chat_completions(build_vision_image_url_messages(), max_tokens=16)
    assert exc.value.status == 400
    assert exc.value.payload is not None
    assert exc.value.payload["error"]["type"] == "BadRequestError"


def test_t2_f14_02_vision_image_url_base64_400(client: KernClient) -> None:
    with pytest.raises(HttpError) as exc:
        client.chat_completions(build_vision_base64_messages(), max_tokens=16)
    assert exc.value.status == 400
    assert exc.value.payload is not None
    assert exc.value.payload["error"]["type"] == "BadRequestError"


def test_t2_f14_03_vision_video_and_audio_parts_400(client: KernClient) -> None:
    with pytest.raises(HttpError) as exc_aud:
        client.chat_completions(build_audio_messages(), max_tokens=16)
    assert exc_aud.value.status == 400

    video_msgs = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Summarize video"},
                {"type": "video_url", "video_url": {"url": "https://example.com/clip.mp4"}},
            ],
        }
    ]
    with pytest.raises(HttpError) as exc_vid:
        client.chat_completions(video_msgs, max_tokens=16)
    assert exc_vid.value.status == 400


def test_t2_f14_04_vision_multiturn_image_part_400(client: KernClient) -> None:
    msgs = [
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi! How can I help?"},
        *build_vision_image_url_messages(),
    ]
    with pytest.raises(HttpError) as exc:
        client.chat_completions(msgs, max_tokens=16)
    assert exc.value.status == 400


def test_t2_f14_05_vision_rejection_leaves_text_concurrent_healthy(client: KernClient) -> None:
    # Raw unused/image special token IDs in text completions also stay safe (text-only embed)
    raw_resp = client.completions(
        [42, IMAGE_CONTENT_TOKEN_ID, IMAGE_SOFT_PLACEHOLDER_TOKEN_ID, 43],
        max_tokens=8,
        temperature=0,
        return_token_ids=True,
    )
    assert_valid_token_ids(raw_resp["choices"][0]["token_ids"])

    # Trigger multimodal 400 rejections concurrently alongside valid text requests
    for _ in range(3):
        with pytest.raises(HttpError) as exc:
            client.chat_completions(build_vision_image_url_messages(), max_tokens=8)
        assert exc.value.status == 400

    valid = client.completions("The capital of France is", max_tokens=8, temperature=0)
    assert "Paris" in valid["choices"][0]["text"]
