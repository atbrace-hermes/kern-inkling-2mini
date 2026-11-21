"""Deterministic prompts, boundary builders, config checks, and Parquet baseline validators."""

from __future__ import annotations

import json
import math
import struct
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Inkling-2-Mini architecture & tokenizer constants (from weights/Inkling-2-Mini/config.json)
UNPADDED_VOCAB_SIZE = 200058
PADDED_VOCAB_SIZE = 201024
EOS_TOKEN_ID = 200006  # <|content_model_end_sampling|>
PAD_TOKEN_ID = 199999  # <|endoftext|>
MESSAGE_USER_TOKEN_ID = 200000
MESSAGE_MODEL_TOKEN_ID = 200001
MESSAGE_SYSTEM_TOKEN_ID = 200002
MESSAGE_TOOL_TOKEN_ID = 200003
CONTENT_TEXT_TOKEN_ID = 200004
IMAGE_CONTENT_TOKEN_ID = 200005  # <|content_image|>
CONTENT_THINKING_TOKEN_ID = 200008
END_MESSAGE_TOKEN_ID = 200010
IMAGE_SOFT_PLACEHOLDER_TOKEN_ID = 200054  # <|unused_200054|>

MAX_MODEL_LEN = 1048576
SCONV_KERNEL_SIZE = 4
SWA_WINDOW_SIZE = 1024
REL_EXTENT = 1024
LOG_SCALING_N_FLOOR = 8192
LOG_SCALING_ALPHA = 0.1
MTP_BLOCK_SIZE = 8
MTP_NUM_SPEC_TOKENS = 7
MAX_SEQS_DEFAULT = 8
ATTN_RES_BLOCK_SIZE = 6
DEFAULT_KV_PAGE_SIZE = 16

REASONING_EFFORT_LEVELS: tuple[str, ...] = (
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)

DIVERSE_PROMPTS: list[str] = [
    "The lighthouse keeper had not seen a ship in eleven days, and the silence of the bay was starting to feel like",
    "To make a proper risotto, begin by warming the stock in a separate pan, because cold stock",
    "A hash map trades memory for speed: instead of scanning every entry, it computes",
    "Compilers optimize loops by hoisting invariant computations out of them, which means",
    "Photosynthesis converts light into chemical energy in two stages. In the first stage, chlorophyll",
    "When the bridge was finally closed for repairs, the town council discovered that nobody",
    "A good chess opening does three things: it develops pieces, controls the center, and",
    "In distributed consensus protocols, a quorum intersection property guarantees that",
]

GSM8K_STYLE_PROBLEMS: list[dict[str, str]] = [
    {
        "id": "gsm_1",
        "question": (
            "A bakery makes 24 loaves of bread in the morning and 18 loaves in the afternoon. "
            "If they sell 31 loaves during the day, how many loaves are left?"
        ),
        "gold": "11",
    },
    {
        "id": "gsm_2",
        "question": (
            "Marcus has 5 boxes of pencils, and each box contains 12 pencils. "
            "He gives 17 pencils to his classmates. How many pencils does Marcus have left?"
        ),
        "gold": "43",
    },
    {
        "id": "gsm_3",
        "question": (
            "A train travels at 60 miles per hour for 2 hours and then at 45 miles per hour for 2 hours. "
            "What is the total distance in miles that the train travels?"
        ),
        "gold": "210",
    },
    {
        "id": "gsm_4",
        "question": (
            "Elena buys 3 notebooks at $4 each and 2 pens at $3 each. "
            "If she pays with a $50 bill, how much change in dollars does she receive?"
        ),
        "gold": "26",
    },
    {
        "id": "gsm_5",
        "question": (
            "A garden has 8 rows of tomato plants with 9 plants per row. "
            "If 14 plants fail to sprout, how many tomato plants grow?"
        ),
        "gold": "58",
    },
    {
        "id": "gsm_6",
        "question": (
            "Liam reads 15 pages a day from Monday through Friday, and 25 pages each on Saturday and Sunday. "
            "How many pages does he read in the whole week?"
        ),
        "gold": "125",
    },
    {
        "id": "gsm_7",
        "question": (
            "A warehouse packs 144 bottles into crates that hold 12 bottles each. "
            "If 5 crates are shipped away, how many crates remain in the warehouse?"
        ),
        "gold": "7",
    },
    {
        "id": "gsm_8",
        "question": (
            "Sophie saves $18 every week for 6 weeks and spends $29 on a jacket. "
            "How many dollars does she have remaining?"
        ),
        "gold": "79",
    },
]


def assert_valid_token_ids(
    token_ids: Sequence[int],
    unpadded_vocab_size: int = UNPADDED_VOCAB_SIZE,
    require_non_empty: bool = True,
) -> None:
    """Assert `token_ids` is non-empty (by default) and every ID lies strictly within [0, unpadded_vocab_size)."""
    if require_non_empty and len(token_ids) == 0:
        raise AssertionError("Expected non-empty token_ids sequence, got []")
    for idx, tok in enumerate(token_ids):
        if not isinstance(tok, int) or isinstance(tok, bool):
            raise AssertionError(f"Non-integer token ID at index {idx}: {tok!r}")
        if tok < 0 or tok >= unpadded_vocab_size:
            raise AssertionError(
                f"Token ID {tok} at index {idx} is outside valid vocab [0, {unpadded_vocab_size})"
            )


def strip_terminal_stop_token(token_ids: Sequence[int], finish_reason: str | None) -> list[int]:
    """Strip the trailing stop token emitted by OpenAI /v1/completions to align with `kern run`."""
    ids = [int(t) for t in token_ids]
    if finish_reason == "stop" and ids:
        return ids[:-1]
    return ids


def ulp_of_bf16(x: float) -> float:
    """BF16 ULP at the scale of `x` (8 significand bits)."""
    return 2.0 ** (math.frexp(abs(x))[1] - 8) if x else 2.0**-133


def is_near_tie_bf16(top_logit: float, alt_logit: float, max_ulps: float = 4.0) -> bool:
    """Check whether `alt_logit` is within `max_ulps` BF16 ULPs of `top_logit`."""
    return (top_logit - alt_logit) / ulp_of_bf16(top_logit) <= max_ulps


def manifest_logits_spec(manifest_path: Path) -> tuple[str, int]:
    """Return `(dtype, vocab_size)` for the target logits buffer declared in `manifest_path`."""
    try:
        m = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "f32", PADDED_VOCAB_SIZE
    buffers = m.get("buffers") or {}
    constants = m.get("constants") or {}
    logits_buf = buffers.get("logits") or buffers.get("d.verify_logits") or {}
    dtype = str(logits_buf.get("dtype") or "f32")
    shape = logits_buf.get("shape") or [None, PADDED_VOCAB_SIZE]
    vocab_raw = shape[-1] if shape else PADDED_VOCAB_SIZE
    if isinstance(vocab_raw, str):
        vocab_raw = constants.get(vocab_raw, PADDED_VOCAB_SIZE)
    vocab_size = int(vocab_raw) if isinstance(vocab_raw, int) else PADDED_VOCAB_SIZE
    return dtype, vocab_size


def read_probe_logits_row0(
    logits_path: Path,
    dtype: str = "f32",
    vocab_size: int = PADDED_VOCAB_SIZE,
) -> list[float]:
    """Read row 0 of a `kern run --probe-dir` `<tag>.logits.bin` dump (`f32` or `bf16`)."""
    raw = Path(logits_path).read_bytes()
    if dtype == "bf16":
        n = min(vocab_size, len(raw) // 2)
        return [struct.unpack("<f", b"\x00\x00" + raw[i * 2 : i * 2 + 2])[0] for i in range(n)]
    n = min(vocab_size, len(raw) // 4)
    return list(struct.unpack(f"<{n}f", raw[: n * 4]))


def read_probe_first_token(tokens_path: Path) -> set[int]:
    """Read the first token of a `kern run --probe-dir` `<tag>.tokens.bin` dump as i32/i64."""
    raw = Path(tokens_path).read_bytes()
    out: set[int] = set()
    if len(raw) >= 4:
        out.add(struct.unpack("<i", raw[:4])[0])
    if len(raw) >= 8:
        out.add(struct.unpack("<q", raw[:8])[0])
    return out


def verify_greedy_parity_with_near_tie(
    kern_cli: Any,
    manifest: Path,
    weights: Path | Sequence[Path],
    prompt_ids: Sequence[int],
    serve_ids: Sequence[int],
    run_ids: Sequence[int],
    *,
    kernels: Path | None = None,
    rows: int = 1,
    gpu: int = 0,
    max_excused: int = 3,
    max_ulps: float = 4.0,
    capacity: int | None = None,
    steps: int | None = None,
) -> None:
    """Compare `serve_ids` vs `run_ids`, excusing up to `max_excused` BF16 near-ties via `kern run --probe-dir`."""
    got = [int(t) for t in serve_ids]
    want = [int(t) for t in run_ids]
    if got == want:
        return
    base_prompt = [int(t) for t in prompt_ids]
    dtype, vocab_size = manifest_logits_spec(manifest)
    total_steps = steps if steps is not None else max(len(got), len(want), 1)
    cap = capacity if capacity is not None else max(512, ((len(base_prompt) + total_steps + 63) // 16) * 16)

    excused = 0
    search_from = 0
    while got != want:
        n_common = min(len(got), len(want))
        diff_idx: int | None = None
        for idx in range(search_from, n_common):
            if got[idx] != want[idx]:
                diff_idx = idx
                break
        if diff_idx is None:
            diff_idx = n_common
        got_tok = got[diff_idx] if diff_idx < len(got) else EOS_TOKEN_ID
        want_tok = want[diff_idx] if diff_idx < len(want) else EOS_TOKEN_ID

        context = base_prompt + want[:diff_idx]
        with tempfile.TemporaryDirectory(prefix="kern_probe_") as tmp_dir_str:
            probe_dir = Path(tmp_dir_str)
            kern_cli.kern_probe(
                manifest=manifest,
                weights=weights,
                probe_dir=probe_dir,
                prompt_ids=context,
                kernels=kernels,
                steps=2,
                rows=rows,
                capacity=cap,
                probe_steps=2,
                probe_labels="",
                gpu=gpu,
            )
            vals: list[float] | None = None
            order: list[int] | None = None
            for tag in ("chunk", "decode0"):
                logits_file = probe_dir / f"{tag}.logits.bin"
                tokens_file = probe_dir / f"{tag}.tokens.bin"
                if logits_file.is_file() and tokens_file.is_file():
                    cand_vals = read_probe_logits_row0(logits_file, dtype=dtype, vocab_size=vocab_size)
                    if not cand_vals:
                        continue
                    cand_order = sorted(range(len(cand_vals)), key=lambda i: -cand_vals[i])
                    if cand_order[0] in read_probe_first_token(tokens_file):
                        vals = cand_vals
                        order = cand_order
                        break
            if vals is None or order is None:
                raise AssertionError(
                    f"Greedy parity diverged at index {diff_idx} (serve={got_tok}, run={want_tok}) "
                    "and --probe-dir produced no matching logits/tokens dump"
                )

        if got_tok >= len(vals) or want_tok >= len(vals):
            raise AssertionError(
                f"Token out of probed logits range at index {diff_idx}: "
                f"serve={got_tok}, run={want_tok}, vocab={len(vals)}"
            )
        x1 = vals[order[0]]
        ulp = ulp_of_bf16(x1)
        got_gap = (x1 - vals[got_tok]) / ulp
        want_gap = (x1 - vals[want_tok]) / ulp
        near_tie = is_near_tie_bf16(x1, vals[got_tok], max_ulps=max_ulps) and is_near_tie_bf16(
            x1, vals[want_tok], max_ulps=max_ulps
        )
        top_str = ", ".join(f"{i}:{vals[i]:.4g}" for i in order[:4])
        if not near_tie or excused >= max_excused:
            raise AssertionError(
                f"Confident greedy parity flip at index {diff_idx} (excused={excused}/{max_excused}): "
                f"serve={got_tok} ({got_gap:.2f} ULPs down), run={want_tok} ({want_gap:.2f} ULPs down); "
                f"top=[{top_str}]"
            )
        excused += 1
        search_from = diff_idx + 1
        if got_tok == EOS_TOKEN_ID:
            want = want[:diff_idx]
            continue
        if diff_idx + 1 >= total_steps:
            want = want[:diff_idx] + [got_tok]
            continue
        remaining = max(1, total_steps - (diff_idx + 1))
        _, cont_ids = kern_cli.kern_run(
            manifest=manifest,
            weights=weights,
            kernels=kernels,
            prompt_ids=context + [got_tok],
            steps=remaining,
            rows=rows,
            capacity=cap,
            temperature=0.0,
            gpu=gpu,
        )
        want = want[:diff_idx] + [got_tok] + cont_ids


# Safe non-special token IDs verified in weights/Inkling-2-Mini/tokenizer.json (< 199998)
_SAFE_LOCAL_TOKEN_IDS: tuple[int, ...] = (1928, 154312, 2641, 3643, 3629, 268, 149876, 4096)


def build_local_token_prompt(target_len: int, weights_dir: Path | None = None) -> list[int]:
    """Build an exact-length valid token ID list on CPU without requiring live /tokenize."""
    if target_len < 1:
        raise ValueError(f"target_len must be >= 1, got {target_len}")
    seed_ids = list(_SAFE_LOCAL_TOKEN_IDS)
    if weights_dir is not None:
        tok_json = Path(weights_dir) / "tokenizer.json"
        if tok_json.is_file():
            try:
                vocab = json.loads(tok_json.read_text(encoding="utf-8")).get("model", {}).get("vocab", {})
                candidate = [
                    int(vocab[k])
                    for k in ("alpha", "Ġalpha", "beta", "gamma", "delta", "the", "Ġthe")
                    if k in vocab and 0 <= int(vocab[k]) < PAD_TOKEN_ID
                ]
                if candidate:
                    seed_ids = candidate
            except (OSError, json.JSONDecodeError, TypeError, ValueError):
                pass
    repeats = (target_len + len(seed_ids) - 1) // len(seed_ids)
    return (seed_ids * repeats)[:target_len]


def build_exact_token_prompt(
    client: Any,
    target_len: int,
    seed_text: str = "alpha beta gamma delta epsilon zeta eta theta ",
) -> list[int]:
    """Build an exact-length token ID sequence using the server's /tokenize endpoint."""
    if target_len < 1:
        raise ValueError(f"target_len must be >= 1, got {target_len}")
    base_tokens: list[int] = client.tokenize(prompt=seed_text)["tokens"]
    if not base_tokens:
        raise RuntimeError("/tokenize returned empty tokens for non-empty seed_text")
    assert_valid_token_ids(base_tokens)
    repeats = (target_len + len(base_tokens) - 1) // len(base_tokens)
    return (base_tokens * repeats)[:target_len]


def build_needle_haystack_tokens(
    needle_ids: Sequence[int],
    question_ids: Sequence[int],
    filler_token_id: int,
    total_prompt_len: int,
    needle_pos: int = 0,
) -> list[int]:
    """Construct an exact-length token prompt with `needle_ids` at `needle_pos` and `question_ids` at tail."""
    n_list = [int(x) for x in needle_ids]
    q_list = [int(x) for x in question_ids]
    min_len = needle_pos + len(n_list) + len(q_list)
    if total_prompt_len < min_len:
        raise ValueError(
            f"total_prompt_len={total_prompt_len} is smaller than needle+offset+question ({min_len})"
        )
    buf = [int(filler_token_id)] * total_prompt_len
    buf[needle_pos : needle_pos + len(n_list)] = n_list
    buf[total_prompt_len - len(q_list) :] = q_list
    return buf


def build_needle_haystack_text(
    needle_sentence: str,
    question: str,
    filler_repeats: int = 200,
    filler_unit: str = "The quick brown fox jumps over the lazy dog in the quiet valley. ",
) -> str:
    """Construct a plain-text needle-in-a-haystack prompt."""
    return f"{needle_sentence}\n\n{filler_unit * filler_repeats}\n\n{question}"


def build_multiturn_messages(
    turns: Sequence[tuple[str, str]],
    system_prompt: str | None = None,
) -> list[dict[str, Any]]:
    """Build OpenAI chat messages using 'system' role (never 'developer')."""
    messages: list[dict[str, Any]] = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    for role, content in turns:
        if role == "developer":
            raise ValueError("Use 'system' role instead of 'developer' for Inkling-2-Mini")
        messages.append({"role": role, "content": content})
    return messages


def build_weather_tool_schema() -> list[dict[str, Any]]:
    """Return a standard OpenAI function tool schema for tool-calling tests."""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather forecast for a given city and unit.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name, e.g. Tokyo"},
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature unit",
                        },
                    },
                    "required": ["city"],
                },
            },
        }
    ]


def build_vision_image_url_messages(
    url: str = "https://example.com/inkling_test_image.png",
    text: str = "Describe what is in this image.",
) -> list[dict[str, Any]]:
    """Build a multimodal image_url chat payload (used to test Vision exclusion rejection)."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "image_url", "image_url": {"url": url}},
            ],
        }
    ]


def build_vision_base64_messages() -> list[dict[str, Any]]:
    """Build an inline base64 PNG image_url chat payload (1x1 transparent PNG)."""
    tiny_png = (
        "data:image/png;base64,"
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
    )
    return build_vision_image_url_messages(url=tiny_png, text="What color is this pixel?")


def build_audio_messages() -> list[dict[str, Any]]:
    """Build a multimodal input_audio chat payload (used to test non-text exclusion)."""
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Transcribe this audio clip."},
                {"type": "input_audio", "input_audio": {"data": "AAAA", "format": "wav"}},
            ],
        }
    ]


def validate_weights_configs(weights_dir: Path) -> dict[str, Any]:
    """Validate Inkling-2-Mini config/generation/tokenizer JSON files on CPU."""
    weights_dir = Path(weights_dir)
    if not weights_dir.exists():
        return {"present": False, "warnings": [f"weights directory does not exist: {weights_dir}"]}

    warnings: list[str] = []
    cfg_path = weights_dir / "config.json"
    gen_path = weights_dir / "generation_config.json"
    tok_path = weights_dir / "tokenizer_config.json"
    for req in (cfg_path, gen_path, tok_path):
        if not req.exists():
            raise FileNotFoundError(f"Missing required config file: {req}")

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    gen = json.loads(gen_path.read_text(encoding="utf-8"))
    tok = json.loads(tok_path.read_text(encoding="utf-8"))

    if cfg.get("model_type") != "inkling2_loop":
        raise AssertionError(f"Expected model_type='inkling2_loop', got {cfg.get('model_type')!r}")
    if cfg.get("eos_token_id") != EOS_TOKEN_ID:
        raise AssertionError(f"Expected eos_token_id={EOS_TOKEN_ID}, got {cfg.get('eos_token_id')!r}")

    text_cfg = cfg.get("text_config") or {}
    if text_cfg.get("model_max_length") != MAX_MODEL_LEN:
        raise AssertionError(
            f"Expected text_config.model_max_length={MAX_MODEL_LEN}, got {text_cfg.get('model_max_length')!r}"
        )
    if text_cfg.get("unpadded_vocab_size") != UNPADDED_VOCAB_SIZE:
        raise AssertionError(
            f"Expected unpadded_vocab_size={UNPADDED_VOCAB_SIZE}, got {text_cfg.get('unpadded_vocab_size')!r}"
        )
    if text_cfg.get("vocab_size") != PADDED_VOCAB_SIZE:
        raise AssertionError(
            f"Expected vocab_size={PADDED_VOCAB_SIZE}, got {text_cfg.get('vocab_size')!r}"
        )
    if text_cfg.get("sconv_kernel_size") != SCONV_KERNEL_SIZE:
        raise AssertionError(
            f"Expected sconv_kernel_size={SCONV_KERNEL_SIZE}, got {text_cfg.get('sconv_kernel_size')!r}"
        )
    if text_cfg.get("sliding_window_size") != SWA_WINDOW_SIZE:
        raise AssertionError(
            f"Expected sliding_window_size={SWA_WINDOW_SIZE}, got {text_cfg.get('sliding_window_size')!r}"
        )
    if text_cfg.get("log_scaling_n_floor") != LOG_SCALING_N_FLOOR:
        raise AssertionError(
            f"Expected log_scaling_n_floor={LOG_SCALING_N_FLOOR}, got {text_cfg.get('log_scaling_n_floor')!r}"
        )
    mtp_cfg = cfg.get("mtp_config") or {}
    if mtp_cfg.get("block_size") != MTP_BLOCK_SIZE:
        raise AssertionError(
            f"Expected mtp_config.block_size={MTP_BLOCK_SIZE}, got {mtp_cfg.get('block_size')!r}"
        )
    if gen.get("eos_token_id") != EOS_TOKEN_ID:
        raise AssertionError(
            f"Expected generation_config eos_token_id={EOS_TOKEN_ID}, got {gen.get('eos_token_id')!r}"
        )
    if tok.get("model_max_length") != MAX_MODEL_LEN:
        raise AssertionError(
            f"Expected tokenizer_config.model_max_length={MAX_MODEL_LEN}, got {tok.get('model_max_length')!r}"
        )

    has_max_pos = "max_position_embeddings" in cfg or "max_position_embeddings" in text_cfg
    if not has_max_pos:
        warnings.append(
            "config.json lacks top-level/text_config max_position_embeddings; "
            "kern-serve must use a served overlay with max_position_embeddings=1048576 "
            "or infront-vllm falls back to 4096"
        )

    return {
        "present": True,
        "model_type": cfg["model_type"],
        "eos_token_id": cfg["eos_token_id"],
        "model_max_length": text_cfg["model_max_length"],
        "has_max_position_embeddings": has_max_pos,
        "warnings": warnings,
    }


def _is_int32_type(dt: Any) -> bool:
    import pyarrow as pa

    return bool(pa.types.is_int32(dt))


def _is_float32_type(dt: Any) -> bool:
    import pyarrow as pa

    return bool(pa.types.is_float32(dt))


def _is_float64_type(dt: Any) -> bool:
    import pyarrow as pa

    return bool(pa.types.is_float64(dt))


def _is_string_type(dt: Any) -> bool:
    import pyarrow as pa

    return bool(pa.types.is_string(dt) or pa.types.is_large_string(dt))


def validate_recording_parquet(path: Path) -> dict[str, Any]:
    """Validate a `kern test e2e-logits` baseline recording Parquet file via pyarrow."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    schema = table.schema
    required = ("source", "prompt", "output", "top_ids", "top_logprobs")
    for col in required:
        if col not in schema.names:
            raise AssertionError(f"{path}: missing required recording column {col!r}")

    if not _is_string_type(schema.field("source").type):
        raise AssertionError(f"{path}: column 'source' must be utf8/string, got {schema.field('source').type}")

    for list_col in ("prompt", "output"):
        t = schema.field(list_col).type
        if not (pa.types.is_list(t) or pa.types.is_large_list(t)) or not _is_int32_type(t.value_type):
            raise AssertionError(f"{path}: column {list_col!r} must be list<int32>, got {t}")

    t_ids = schema.field("top_ids").type
    if (
        not (pa.types.is_list(t_ids) or pa.types.is_large_list(t_ids))
        or not (pa.types.is_list(t_ids.value_type) or pa.types.is_large_list(t_ids.value_type))
        or not _is_int32_type(t_ids.value_type.value_type)
    ):
        raise AssertionError(f"{path}: column 'top_ids' must be list<list<int32>>, got {t_ids}")

    t_lps = schema.field("top_logprobs").type
    if (
        not (pa.types.is_list(t_lps) or pa.types.is_large_list(t_lps))
        or not (pa.types.is_list(t_lps.value_type) or pa.types.is_large_list(t_lps.value_type))
        or not _is_float32_type(t_lps.value_type.value_type)
    ):
        raise AssertionError(f"{path}: column 'top_logprobs' must be list<list<float32>>, got {t_lps}")

    has_noise = "noise" in schema.names
    if has_noise:
        t_noise = schema.field("noise").type
        if not (pa.types.is_list(t_noise) or pa.types.is_large_list(t_noise)) or not _is_float64_type(
            t_noise.value_type
        ):
            raise AssertionError(f"{path}: column 'noise' must be list<float64>, got {t_noise}")

    if table.num_rows == 0:
        raise AssertionError(f"{path}: recording parquet has 0 rows")

    data = table.to_pydict()
    for idx in range(table.num_rows):
        out_len = len(data["output"][idx])
        top_ids_row = data["top_ids"][idx]
        top_lps_row = data["top_logprobs"][idx]
        if len(top_ids_row) != out_len or len(top_lps_row) != out_len:
            raise AssertionError(
                f"{path}: row {idx} output len={out_len} vs top_ids={len(top_ids_row)} / top_logprobs={len(top_lps_row)}"
            )
        for step, (ids_k, lps_k) in enumerate(zip(top_ids_row, top_lps_row, strict=True)):
            if len(ids_k) != len(lps_k):
                raise AssertionError(
                    f"{path}: row {idx} step {step} top_ids len={len(ids_k)} != top_logprobs len={len(lps_k)}"
                )
        if has_noise and len(data["noise"][idx]) != out_len:
            raise AssertionError(
                f"{path}: row {idx} noise len={len(data['noise'][idx])} != output len={out_len}"
            )

    raw_meta = schema.metadata or {}
    meta = {
        (k.decode("utf-8", errors="replace") if isinstance(k, bytes) else str(k)): (
            v.decode("utf-8", errors="replace") if isinstance(v, bytes) else str(v)
        )
        for k, v in raw_meta.items()
    }
    return {
        "rows": table.num_rows,
        "has_noise": has_noise,
        "metadata": meta,
    }


def validate_questions_parquet(path: Path) -> dict[str, Any]:
    """Validate a `kern test e2e-logits record` question-set Parquet file."""
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    schema = table.schema
    for col in ("source", "messages"):
        if col not in schema.names:
            raise AssertionError(f"{path}: missing required question column {col!r}")
        if not _is_string_type(schema.field(col).type):
            raise AssertionError(f"{path}: column {col!r} must be utf8/string")
    if "tools" in schema.names and not _is_string_type(schema.field("tools").type):
        raise AssertionError(f"{path}: column 'tools' must be utf8/string")
    if table.num_rows == 0:
        raise AssertionError(f"{path}: question parquet has 0 rows")
    return {"rows": table.num_rows}


@dataclass
class BaselineArtifacts:
    """Discovered M0 baseline files under `--baseline-dir`."""

    baseline_dir: Path
    recording_parquet: Path | None = None
    recording_info: dict[str, Any] = field(default_factory=dict)
    questions_parquet: Path | None = None
    metrics_json: Path | None = None
    metrics_data: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _extract_numeric_metric_with_key(obj: Any, keys: Sequence[str]) -> tuple[str, float] | None:
    if isinstance(obj, dict):
        for k in keys:
            val = obj.get(k)
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                return k, float(val)
        for v in obj.values():
            found = _extract_numeric_metric_with_key(v, keys)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for item in obj:
            found = _extract_numeric_metric_with_key(item, keys)
            if found is not None:
                return found
    return None


def _extract_numeric_metric(obj: Any, keys: Sequence[str]) -> float | None:
    found = _extract_numeric_metric_with_key(obj, keys)
    return found[1] if found is not None else None


def extract_baseline_mtp_metrics(metrics_data: dict[str, Any]) -> dict[str, float | None]:
    """Extract normalized MTP / throughput numbers from heterogeneous M0 JSON schemas."""
    tokens_per_round = _extract_numeric_metric(
        metrics_data,
        (
            "tokens_per_round",
            "mean_accept_length",
            "mean_accepted_tokens",
            "mtp_tokens_per_round",
            "mtp_mean_acceptance_length",
        ),
    )
    accept_pct = _extract_numeric_metric(metrics_data, ("accept_pct",))
    accept_rate = _extract_numeric_metric(metrics_data, ("acceptance_rate", "mtp_acceptance_rate"))
    if tokens_per_round is None:
        if accept_pct is not None:
            # accept_pct is always a percentage in [0, 100]
            tokens_per_round = 1.0 + MTP_NUM_SPEC_TOKENS * (accept_pct / 100.0)
        elif accept_rate is not None:
            frac = accept_rate / 100.0 if accept_rate > 1.0 else accept_rate
            tokens_per_round = 1.0 + MTP_NUM_SPEC_TOKENS * frac
    tok_s = _extract_numeric_metric(
        metrics_data,
        (
            "end_to_end_tokens_per_second",
            "tokens_per_second",
            "tok_s",
            "throughput_tok_s",
            "output_throughput_tok_s",
        ),
    )
    ttft_match = _extract_numeric_metric_with_key(
        metrics_data,
        ("ttft_s", "median_ttft_s", "mean_ttft_s", "ttft_p50_ms", "ttft_mean_ms"),
    )
    ttft_s: float | None = None
    if ttft_match is not None:
        ttft_key, ttft_val = ttft_match
        ttft_s = ttft_val / 1000.0 if ttft_key.endswith("_ms") else ttft_val
    return {
        "tokens_per_round": tokens_per_round,
        "accept_pct": accept_pct if accept_pct is not None else accept_rate,
        "tokens_per_second": tok_s,
        "ttft_s": ttft_s,
    }


def discover_baselines(
    baseline_dir: Path,
    explicit_parquet: Path | None = None,
    explicit_metrics: Path | None = None,
) -> BaselineArtifacts:
    """Auto-discover M0 baseline recording Parquet, question Parquet, and metrics JSON."""
    import pyarrow as pa

    artifacts = BaselineArtifacts(baseline_dir=Path(baseline_dir))

    if explicit_parquet is not None:
        if not explicit_parquet.is_file():
            raise FileNotFoundError(f"Explicit --baseline-parquet not found: {explicit_parquet}")
        artifacts.recording_parquet = explicit_parquet
        artifacts.recording_info = validate_recording_parquet(explicit_parquet)
    elif artifacts.baseline_dir.exists():
        for pq_path in sorted(artifacts.baseline_dir.rglob("*.parquet")):
            try:
                info = validate_recording_parquet(pq_path)
                if artifacts.recording_parquet is None:
                    artifacts.recording_parquet = pq_path
                    artifacts.recording_info = info
                continue
            except (OSError, ValueError, pa.ArrowException) as exc:
                artifacts.warnings.append(f"Ignoring corrupt/unreadable parquet {pq_path}: {exc}")
                continue
            except AssertionError:
                pass
            try:
                validate_questions_parquet(pq_path)
                if artifacts.questions_parquet is None:
                    artifacts.questions_parquet = pq_path
            except (AssertionError, OSError, ValueError, pa.ArrowException) as exc:
                artifacts.warnings.append(f"Ignoring unrecognized/corrupt parquet {pq_path}: {exc}")

    if explicit_metrics is not None:
        if not explicit_metrics.is_file():
            raise FileNotFoundError(f"Explicit --baseline-metrics not found: {explicit_metrics}")
        parsed = json.loads(explicit_metrics.read_text(encoding="utf-8"))
        data = parsed if isinstance(parsed, dict) else {"entries": parsed}
        extracted = extract_baseline_mtp_metrics(data)
        if not any(v is not None for v in extracted.values()):
            raise ValueError(
                f"Explicit --baseline-metrics contains no recognized MTP/throughput metrics: {explicit_metrics}"
            )
        artifacts.metrics_json = explicit_metrics
        artifacts.metrics_data = data
    elif artifacts.baseline_dir.exists():
        for json_path in sorted(artifacts.baseline_dir.rglob("*.json")):
            try:
                parsed = json.loads(json_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            data = parsed if isinstance(parsed, dict) else {"entries": parsed}
            extracted = extract_baseline_mtp_metrics(data)
            if any(v is not None for v in extracted.values()):
                artifacts.metrics_json = json_path
                artifacts.metrics_data = data
                break

    return artifacts
