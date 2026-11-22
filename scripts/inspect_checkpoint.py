#!/usr/bin/env python3
"""CPU-only geometry, quantization, and safetensors header validator for Inkling-2-Mini (text + MTP)."""

import argparse
import json
import struct
import tomllib
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read_safetensors_header(path: Path) -> tuple[dict, int]:
    file_size = path.stat().st_size
    with path.open("rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        if not 0 < header_size <= min(64 * 1024 * 1024, file_size - 8):
            raise ValueError(f"{path.name}: invalid header length {header_size}")
        tensors = json.loads(f.read(header_size))
    tensors.pop("__metadata__", None)
    data_size = file_size - 8 - header_size
    end = 0
    for name, info in tensors.items():
        lo, hi = info["data_offsets"]
        if not 0 <= lo <= hi <= data_size:
            raise ValueError(f"{path.name}: invalid offsets for {name}: {[lo, hi]} vs {data_size}")
        end = max(end, hi)
    if end != data_size:
        raise ValueError(f"{path.name}: payload end {end} != data_size {data_size}")
    return tensors, data_size


def _check_tensor(tensors: dict, name: str, dtype: str, shape: list[int], problems: list[str]) -> None:
    info = tensors.get(name)
    if info is None:
        problems.append(f"missing tensor: {name}")
        return
    if info["dtype"] != dtype:
        problems.append(f"{name}: expected dtype {dtype}, got {info['dtype']}")
    if info["shape"] != shape:
        problems.append(f"{name}: expected shape {shape}, got {info['shape']}")


def inspect(root: Path) -> tuple[dict, list[str]]:
    problems = []
    policy = tomllib.loads((ROOT / "precision.toml").read_text())

    config = json.loads((root / "config.json").read_text())
    if config.get("model_type") != "inkling2_loop":
        problems.append(f"model_type: expected 'inkling2_loop', got {config.get('model_type')!r}")
    if config.get("architectures") != ["Inkling2LoopForConditionalGeneration"]:
        problems.append(f"architectures mismatch: {config.get('architectures')!r}")

    text = config.get("text_config", {})
    expected_text = {
        "hidden_size": 2304,
        "num_hidden_layers": 24,
        "vocab_size": 201024,
        "unpadded_vocab_size": 200058,
        "num_attention_heads": 24,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "d_rel": 16,
        "rel_extent": 1024,
        "sconv_kernel_size": 4,
        "sliding_window_size": 1024,
        "n_layers_per_attn_res_block": 6,
        "attn_res_normalize_values": True,
        "n_routed_experts": 128,
        "num_experts_per_tok": 8,
        "intermediate_size": 1152,
        "final_logit_softcapping": 30.0,
        "rms_norm_eps": 1e-6,
    }
    for k, v in expected_text.items():
        if text.get(k) != v:
            problems.append(f"text_config.{k}: expected {v!r}, got {text.get(k)!r}")

    expected_loop = {
        "core_start": 6,
        "core_end": 18,
        "num_passes": 3,
        "gate_rank": 64,
        "is_varloop": False,
    }
    if text.get("loop_config") != expected_loop:
        problems.append(f"text_config.loop_config mismatch: {text.get('loop_config')!r}")

    local_ids = set(text.get("local_layer_ids", []))
    expected_local = {0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22}
    if local_ids != expected_local:
        problems.append(f"local_layer_ids mismatch: {sorted(local_ids)}")

    mtp_cfg = config.get("mtp_config", {})
    expected_mtp = {
        "hidden_size": 2304,
        "num_hidden_layers": 3,
        "num_attention_heads": 24,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "intermediate_size": 9216,
        "block_size": 8,
        "markov_rank": 256,
        "markov_head_type": "rnn_hidden",
        "target_layer_ids": [5, 23, 41, 48],
    }
    for k, v in expected_mtp.items():
        if mtp_cfg.get(k) != v:
            problems.append(f"mtp_config.{k}: expected {v!r}, got {mtp_cfg.get(k)!r}")

    quant_cfg = config.get("quantization_config", {})
    if quant_cfg.get("quant_method") != "modelopt" or quant_cfg.get("quant_algo") != "NVFP4":
        problems.append(f"quantization_config mismatch: {quant_cfg!r}")

    # Validate hf_quant_config.json
    hfq_path = root / "hf_quant_config.json"
    hfq = json.loads(hfq_path.read_text()) if hfq_path.exists() else {}
    if not hfq_path.exists():
        problems.append("missing hf_quant_config.json")
    model_groups = hfq.get("model", {}).get("parameter_groups", [])
    mtp_groups = hfq.get("mtp", {}).get("parameter_groups", [])
    unembed_fwd = None
    for g in model_groups:
        if "model.llm.unembed.weight" in g.get("tensors", []):
            unembed_fwd = g.get("fwd_output_dtype")
    if unembed_fwd != "float32":
        problems.append(f"hf_quant_config unembed fwd_output_dtype: expected 'float32', got {unembed_fwd!r}")

    # Validate model.safetensors.index.json & main shard(s)
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    shards: dict[str, set[str]] = {}
    for tname, fname in index.items():
        if Path(fname).name != fname:
            problems.append(f"non-flat shard filename: {fname}")
        shards.setdefault(fname, set()).add(tname)

    all_main_tensors = {}
    main_data_bytes = 0
    for fname, expected_names in sorted(shards.items()):
        p = root / fname
        if not p.exists():
            problems.append(f"missing shard: {fname}")
            continue
        tensors, dsz = _read_safetensors_header(p)
        main_data_bytes += dsz
        missing = expected_names - tensors.keys()
        if missing:
            problems.append(f"{fname}: indexed tensors missing from header: {sorted(missing)[:5]}")
        all_main_tensors.update(tensors)

    visual_names = [k for k in all_main_tensors if k.startswith("model.visual.")]
    text_tensors = {k: v for k, v in all_main_tensors.items() if not k.startswith("model.visual.")}
    text_dtypes = Counter(v["dtype"] for v in text_tensors.values())
    text_bytes = sum(v["data_offsets"][1] - v["data_offsets"][0] for v in text_tensors.values())

    # Verify M1 & key text tensor shapes/dtypes
    H, V, R, E, I = 2304, 201024, 64, 128, 1152
    _check_tensor(text_tensors, "model.llm.embed.weight", "BF16", [V, H], problems)
    _check_tensor(text_tensors, "model.llm.embed_norm.weight", "BF16", [H], problems)
    _check_tensor(text_tensors, "model.llm.norm.weight", "BF16", [H], problems)
    _check_tensor(text_tensors, "model.llm.unembed.weight", "BF16", [V, H], problems)
    _check_tensor(text_tensors, "model.llm.middle_cycle_cross_pass_queries", "F32", [24, H], problems)
    _check_tensor(text_tensors, "model.llm.middle_cycle_cross_pass_kv_queries", "F32", [24, H], problems)
    _check_tensor(text_tensors, "model.llm.attn_res.output_query", "F32", [H], problems)

    for layer in range(24):
        is_full = layer not in expected_local
        if layer > 0:
            _check_tensor(
                text_tensors,
                f"model.llm.attn_res.attn_queries.{layer - 1}",
                "F32",
                [3, H] if is_full else [H],
                problems,
            )
        _check_tensor(text_tensors, f"model.llm.attn_res.mlp_queries.{layer}", "F32", [H], problems)
        if is_full:
            for norm in ("q_input_norm", "k_input_norm", "v_input_norm"):
                _check_tensor(text_tensors, f"model.llm.layers.{layer}.{norm}.weight", "BF16", [H], problems)
        else:
            _check_tensor(text_tensors, f"model.llm.layers.{layer}.attn_norm.weight", "BF16", [H], problems)
        _check_tensor(text_tensors, f"model.llm.layers.{layer}.mlp_norm.weight", "BF16", [H], problems)
        if 6 <= layer < 18:
            for s in ("attn", "k", "v", "mlp"):
                pfx = f"model.llm.layers.{layer}.controllers.{s}"
                _check_tensor(text_tensors, f"{pfx}.down_weight", "BF16", [R, 2 * H], problems)
                _check_tensor(text_tensors, f"{pfx}.forget_weight", "BF16", [H, R], problems)
                _check_tensor(text_tensors, f"{pfx}.gain_weight", "BF16", [H, R], problems)
        # Router & experts
        _check_tensor(text_tensors, f"model.llm.layers.{layer}.mlp.gate.weight", "BF16", [E, H], problems)
        _check_tensor(text_tensors, f"model.llm.layers.{layer}.mlp.gate.bias", "F32", [E], problems)
        if layer < 20:
            _check_tensor(text_tensors, f"model.llm.layers.{layer}.mlp.experts.w13_weight", "U8", [E, 2 * I, H // 2], problems)
            _check_tensor(text_tensors, f"model.llm.layers.{layer}.mlp.experts.w13_weight_scale", "F8_E4M3", [E, 2 * I, H // 16], problems)
            _check_tensor(text_tensors, f"model.llm.layers.{layer}.mlp.experts.w2_weight", "U8", [E, H, I // 2], problems)
            _check_tensor(text_tensors, f"model.llm.layers.{layer}.mlp.experts.w2_weight_scale", "F8_E4M3", [E, H, I // 16], problems)
        else:
            _check_tensor(text_tensors, f"model.llm.layers.{layer}.mlp.experts.w13_weight", "F8_E4M3", [E, 2 * I, H], problems)
            _check_tensor(text_tensors, f"model.llm.layers.{layer}.mlp.experts.w13_weight_scale", "U8", [E, 2 * I, H // 32], problems)
            _check_tensor(text_tensors, f"model.llm.layers.{layer}.mlp.experts.w2_weight", "F8_E4M3", [E, H, I], problems)
            _check_tensor(text_tensors, f"model.llm.layers.{layer}.mlp.experts.w2_weight_scale", "U8", [E, H, I // 32], problems)

    # Validate mtp.safetensors
    mtp_path = root / "mtp.safetensors"
    mtp_tensors = {}
    mtp_bytes = 0
    if not mtp_path.exists():
        problems.append("missing mtp.safetensors")
    else:
        mtp_tensors, mtp_bytes = _read_safetensors_header(mtp_path)
        _check_tensor(mtp_tensors, "fuse_weight", "BF16", [4, 4], problems)
        _check_tensor(mtp_tensors, "mask_embed", "BF16", [H], problems)
        _check_tensor(mtp_tensors, "norm.weight", "BF16", [H], problems)
    mtp_dtypes = Counter(v["dtype"] for v in mtp_tensors.values())

    # 48 unrolled visits: prelude 0..5, core 6..17 x3, coda 18..23
    visits = list(range(0, 6)) + list(range(6, 18)) * 3 + list(range(18, 24))
    full_visits = sum(1 for i in visits if i not in expected_local)
    swa_visits = sum(1 for i in visits if i in expected_local)
    total_full_kv = full_visits + 1  # +1 MTP full layer
    total_swa_kv = swa_visits + 2    # +2 MTP SWA layers
    full_kv_bytes_per_tok = total_full_kv * 2 * text["num_key_value_heads"] * text["head_dim"]
    swa_bytes_per_seq = total_swa_kv * 2 * text["sliding_window_size"] * text["num_key_value_heads"] * text["head_dim"]
    # 4-stream sconv: k(1024) + v(1024) + attn(2304? or 3072) + mlp(2304) * (W-1=3) * 2B
    report = {
        "model_type": config.get("model_type"),
        "logical_visits": len(visits),
        "full_attention_visits": full_visits,
        "swa_visits": swa_visits,
        "mtp_full_layers": 1,
        "mtp_swa_layers": 2,
        "precision": policy,
        "hf_quant_groups": {
            "model_groups": len(model_groups),
            "mtp_groups": len(mtp_groups),
            "unembed_fwd_output_dtype": unembed_fwd,
        },
        "payload_bytes": {
            "full_fp8_kv_per_token": full_kv_bytes_per_tok,
            "swa_fp8_kv_per_sequence": swa_bytes_per_seq,
            "text_weights_bytes": text_bytes,
            "mtp_weights_bytes": mtp_bytes,
        },
        "main_safetensors": {
            "total_tensors": len(all_main_tensors),
            "excluded_visual_tensors": len(visual_names),
            "text_tensors": len(text_tensors),
            "text_tensor_dtype_counts": dict(text_dtypes),
            "payload_bytes": main_data_bytes,
        },
        "mtp_safetensors": {
            "total_tensors": len(mtp_tensors),
            "tensor_dtype_counts": dict(mtp_dtypes),
            "payload_bytes": mtp_bytes,
        },
        "problems": problems,
    }
    return report, problems


def main() -> int:
    import os

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(os.environ.get("KERN_WEIGHTS", ROOT.parent / "weights" / "Inkling-2-Mini")),
    )
    ap.add_argument("--strict", action="store_true", help="exit 1 if any problems found")
    args = ap.parse_args()
    report, problems = inspect(args.checkpoint)
    print(json.dumps(report, indent=2))
    if problems and args.strict:
        return 1
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
