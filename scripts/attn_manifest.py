#!/usr/bin/env python3
"""M3 manifest builders & state contracts for Inkling-2-Mini (sconv, attn_prologue, rel_attention).

Exports:
  - Unified state constants (PAGE, RING, KV_BYTES_PER_TOKEN, SEQ_STATE_BYTES, offsets)
  - Typed builders: register_modules, register_states, register_buffers, register_ops,
    emit_load_calls (and alias emit_derive_calls), emit_sconv_kv, emit_sconv_attn,
    emit_sconv_mlp, emit_sconv_spec_commit, emit_layer_calls, emit_swa_spec_commit
  - Standalone helper: build_unit_manifest(...) for CPU kern verify & GPU unit tests
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Unified Architectural State & Geometry Constants
# ---------------------------------------------------------------------------
PAGE = 64
RING = 1024
HEADS = 24
KV_HEADS = 8
HEAD_DIM = 128
REL_DIM = 16
REL_EXTENT = 1024
HIDDEN_SIZE = 2304

KV_FULL_LAYERS = 9  # 8 target full visits + 1 MTP full layer (reserved for M4)
KV_HALF_PAGE_BYTES = KV_HEADS * PAGE * HEAD_DIM          # 65,536 B
KV_LAYER_PAGE_BYTES = 2 * KV_HALF_PAGE_BYTES            # 131,072 B
KV_PAGE_BYTES = KV_FULL_LAYERS * KV_LAYER_PAGE_BYTES    # 1,179,648 B
KV_BYTES_PER_TOKEN = KV_FULL_LAYERS * 2 * KV_HEADS * HEAD_DIM  # 18,432 B

SCONV_WIDTHS = {"k": 1024, "v": 1024, "attn": 2304, "mlp": 2304}
SCONV_STREAM_OFFSETS = {"k": 0, "v": 6144, "attn": 12288, "mlp": 26112}
SCONV_RAW_COL_OFFSETS = {"k": 0, "v": 1024, "attn": 2048, "mlp": 4352}
SCONV_RAW_ROW_ELEMS = 6656
SCONV_VISIT_BYTES = 3 * (1024 + 1024 + 2304 + 2304) * 2  # 39,936 B
SCONV_TOTAL_BYTES = 48 * SCONV_VISIT_BYTES              # 1,916,928 B

SWA_HALF_VISIT_BYTES = KV_HEADS * RING * HEAD_DIM       # 1,048,576 B
SWA_VISIT_BYTES = 2 * SWA_HALF_VISIT_BYTES              # 2,097,152 B
SWA_TARGET_OFFSET = SCONV_TOTAL_BYTES                   # 1,916,928 B
SWA_TARGET_TOTAL_BYTES = 40 * SWA_VISIT_BYTES           # 83,886,080 B
SWA_MTP_OFFSET = SWA_TARGET_OFFSET + SWA_TARGET_TOTAL_BYTES  # 85,803,008 B
SWA_MTP_TOTAL_BYTES = 2 * SWA_VISIT_BYTES               # 4,194,304 B
SEQ_STATE_TAIL_PAD_BYTES = 8064                         # 63 * 128 B tail alignment reserve (M3/M4 contract)
SEQ_STATE_BYTES = SWA_MTP_OFFSET + SWA_MTP_TOTAL_BYTES + SEQ_STATE_TAIL_PAD_BYTES  # 90,005,376 B
assert SEQ_STATE_BYTES == 90_005_376 and SEQ_STATE_BYTES % 128 == 0

NUM_PHYS_LAYERS = 24
NUM_VISITS = 48
FULL_LAYER_IDS = (5, 11, 17, 23)
FULL_VISIT_IDS = (5, 11, 17, 23, 29, 35, 41, 47)
VISIT_TO_PHYS = tuple(range(6)) + tuple(range(6, 18)) * 3 + tuple(range(18, 24))
VISIT_TO_FULL_IDX = {v: i for i, v in enumerate(FULL_VISIT_IDS)}
VISIT_TO_SWA_IDX = {
    v: idx
    for idx, v in enumerate(v for v in range(NUM_VISITS) if v not in VISIT_TO_FULL_IDX)
}

DEFAULT_CONTEXT = 1048576 + PAGE  # 1,048,640 (headroom for MTP rows=8)
DEFAULT_PAGE_COLUMNS = DEFAULT_CONTEXT // PAGE + 1  # 16,385


def _b(name: str, offset: int = 0) -> dict[str, Any]:
    return {"buf": name, "offset": offset} if offset else {"buf": name}


def _s(name: str, offset: int = 0) -> dict[str, Any]:
    return {"state": name, "offset": offset} if offset else {"state": name}


def _i32(v: int) -> dict[str, int]:
    return {"i32": int(v)}


def _i64(v: int) -> dict[str, int]:
    return {"i64": int(v)}


def _f32(v: float) -> dict[str, float]:
    return {"f32": float(v)}


def _ceil_div(expr: str | int, d: int) -> dict[str, list[Any]]:
    return {"ceil_div": [expr, d]}


def _mul(a: Any, b: Any) -> dict[str, list[Any]]:
    return {"mul": [a, b]}


ATTN_DSL_MODULES = (
    "attn_log_tau",
    "attn_full_prefill",
    "attn_full_prefill_split",
    "attn_full_decode1",
    "attn_full_decode8",
    "attn_full_verify",
    "attn_swa_prefill",
    "attn_swa_decode1",
    "attn_swa_decode8",
    "attn_swa_verify",
    "attn_combine_s8",
    "attn_combine_s16",
    "attn_combine_s32",
    "attn_combine_s64",
)


def load_dsl(kernels_dir: Path) -> dict[str, Any]:
    """Resolve M3 Triton AOT module metadata from dsl.json / <mod>.json / attn_dsl.json matching live .cubin SHA-256."""
    dsl_modules: dict[str, Any] = {}
    dsl_path = kernels_dir / "dsl.json"
    if dsl_path.exists():
        dsl_modules.update(json.loads(dsl_path.read_text()).get("modules", {}))
    attn_dsl_path = kernels_dir / "attn_dsl.json"
    attn_modules: dict[str, Any] = {}
    if attn_dsl_path.exists():
        attn_modules = json.loads(attn_dsl_path.read_text()).get("modules", {})

    resolved: dict[str, Any] = {}
    for mod in ATTN_DSL_MODULES:
        cubin_path = kernels_dir / f"{mod}.cubin"
        if not cubin_path.exists():
            raise FileNotFoundError(f"Missing M3 Triton cubin: {cubin_path}")
        live_sha = hashlib.sha256(cubin_path.read_bytes()).hexdigest()
        candidates: list[dict[str, Any]] = []
        if mod in dsl_modules:
            candidates.append(dsl_modules[mod])
        per_mod_path = kernels_dir / f"{mod}.json"
        if per_mod_path.exists():
            candidates.append(json.loads(per_mod_path.read_text()))
        if mod in attn_modules:
            candidates.append(attn_modules[mod])
        chosen = next((c for c in candidates if c.get("sha256") == live_sha), None)
        if chosen is None:
            raise RuntimeError(
                f"No DSL metadata matches live {cubin_path.name} sha256={live_sha} "
                f"(checked dsl.json, {mod}.json, attn_dsl.json)"
            )
        resolved[mod] = chosen
    return {"modules": resolved}


def register_modules(
    modules: dict[str, Any],
    kernels_dir: Path,
    dsl: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Register sconv.cubin, attn_prologue.cubin, and all attn_*.cubin Triton AOT modules."""
    if dsl is None:
        dsl = load_dsl(kernels_dir)
    for mod_name in ("sconv", "attn_prologue"):
        cubin_path = kernels_dir / f"{mod_name}.cubin"
        modules[mod_name] = {
            "source": cubin_path.name,
            "sha256": hashlib.sha256(cubin_path.read_bytes()).hexdigest(),
        }
    for mod_name, info in dsl["modules"].items():
        modules[mod_name] = {
            "source": info["source"],
            "sha256": info["sha256"],
        }
    return dsl


def register_states(states: dict[str, Any]) -> None:
    states["kv"] = {"bytes_per_token": KV_BYTES_PER_TOKEN}
    states["seq_state"] = {"bytes_per_seq": SEQ_STATE_BYTES}


def register_buffers(
    buffers: dict[str, Any],
    max_seqs: int = 8,
    chunk: int = 2048,
    context: int = DEFAULT_CONTEXT,
    page_columns: int = DEFAULT_PAGE_COLUMNS,
    include_inputs: bool = True,
    include_scales: bool = True,
) -> None:
    """Declare shared M3 input, workspace, and carry scale buffers."""
    if include_inputs:
        buffers.setdefault(
            "positions",
            {"dtype": "i64", "shape": ["tokens"], "kind": "input", "fill": "position", "domain": {"min": 0, "max": context - 1}},
        )
        buffers.setdefault(
            "valid",
            {"dtype": "i32", "shape": ["tokens"], "kind": "input", "fill": "valid", "domain": {"min": 0, "max": 1}},
        )
        buffers.setdefault(
            "lengths",
            {"dtype": "i32", "shape": ["seqs"], "kind": "input", "fill": "seq_len", "domain": {"min": 1, "max": context}},
        )
        buffers.setdefault(
            "pages",
            {"dtype": "i32", "shape": ["seqs", page_columns], "kind": "input", "domain": {"index_into": "kv", "stride": PAGE}},
        )
        buffers.setdefault(
            "seq_state.lines",
            {"dtype": "i32", "shape": [1, "seqs"], "kind": "input", "domain": {"index_into": "seq_state", "stride": SEQ_STATE_BYTES}},
        )

    part_tokens = max(64, max_seqs * 8)
    buffers["sconv.dyn_k"] = {"dtype": "bf16", "shape": ["tokens", 32], "kind": "workspace"}
    buffers["sconv.dyn_v"] = {"dtype": "bf16", "shape": ["tokens", 32], "kind": "workspace"}
    buffers["sconv.k_out"] = {"dtype": "bf16", "shape": ["tokens", 1024], "kind": "workspace"}
    buffers["sconv.v_out"] = {"dtype": "bf16", "shape": ["tokens", 1024], "kind": "workspace"}
    buffers["sconv.verify_raw"] = {
        "dtype": "bf16",
        "shape": [NUM_VISITS, max_seqs * 8, SCONV_RAW_ROW_ELEMS],
        "kind": "workspace",
    }

    buffers["attn.tau"] = {"dtype": "f32", "shape": ["tokens"], "kind": "workspace"}
    buffers["attn.q_clip"] = {"dtype": "bf16", "shape": ["tokens", HEADS, HEAD_DIM], "kind": "workspace"}
    buffers["attn.rel_bias"] = {"dtype": "bf16", "shape": ["tokens", HEADS, REL_EXTENT], "kind": "workspace"}
    buffers["attn.swa_k"] = {"dtype": "fp8e4m3", "shape": ["tokens", KV_HEADS, HEAD_DIM], "kind": "workspace"}
    buffers["attn.swa_v"] = {"dtype": "fp8e4m3", "shape": ["tokens", KV_HEADS, HEAD_DIM], "kind": "workspace"}
    buffers["attn.verify_swa_k"] = {
        "dtype": "fp8e4m3",
        "shape": [40, max_seqs * 8, KV_HEADS, HEAD_DIM],
        "kind": "workspace",
    }
    buffers["attn.verify_swa_v"] = {
        "dtype": "fp8e4m3",
        "shape": [40, max_seqs * 8, KV_HEADS, HEAD_DIM],
        "kind": "workspace",
    }
    buffers["attn.o_part"] = {"dtype": "f32", "shape": [64, part_tokens, HEADS, HEAD_DIM], "kind": "workspace"}
    buffers["attn.lse_part"] = {"dtype": "f32", "shape": [64, part_tokens, HEADS], "kind": "workspace"}
    buffers["attn.out"] = {"dtype": "bf16", "shape": ["tokens", HEADS * HEAD_DIM], "kind": "workspace"}

    if include_scales:
        for v in range(NUM_VISITS):
            buffers[f"attn.v{v}.k_scale"] = {"dtype": "f32", "shape": [1], "kind": "carry"}
            buffers[f"attn.v{v}.v_scale"] = {"dtype": "f32", "shape": [1], "kind": "carry"}


def _triton_launch(
    dsl: dict[str, Any],
    mod_name: str,
    params: list[str],
    grid: list[Any],
    args: list[Any],
    *,
    pdl: bool = False,
    when: dict[str, Any] | None = None,
) -> dict[str, Any]:
    info = dsl["modules"][mod_name]
    launch: dict[str, Any] = {
        "module": mod_name,
        "entry": info["entry"],
        "params": [*params, *info["abi_tail"]],
        "grid": grid,
        "block": info["block"],
        "args": [*args, _i64(0), _i64(0)],
    }
    if info["shared_mem"]:
        launch["shared_mem"] = info["shared_mem"]
    if pdl:
        launch["pdl"] = True
    if when is not None:
        launch["when"] = when
    return launch


def register_ops(
    ops: dict[str, Any],
    dsl: dict[str, Any],
    max_seqs: int = 8,
    page_columns: int = DEFAULT_PAGE_COLUMNS,
) -> None:
    """Register all M3 sconv, attn_prologue, SWA commit, and rel_attention ops."""
    # Linear projection for dynamic sconv weight_proj & general bf16 gemm
    ops.setdefault(
        "m3_gemm_bf16",
        {
            "params": ["in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32", "i32"],
            "impl": {"launches": [{"entry": "extern:cublaslt_bf16_tn"}]},
        },
    )
    ops["m3_fill_f32"] = {
        "params": ["out buffer<f32>", "i32", "f32"],
        "impl": {
            "launches": [
                {
                    "module": "attn_prologue",
                    "entry": "kern_fill_f32",
                    "grid": [1, 1, 1],
                    "block": [1, 1, 1],
                }
            ]
        },
    }
    ops["m3_gate_sink"] = {
        "params": [
            "in buffer<f32>", "in buffer<bf16>", "in buffer<f32>", "in buffer<bf16>",
            "out buffer<bf16>", "i32", "i32", "i32",
        ],
        "impl": {
            "launches": [
                {
                    "module": "attn_prologue",
                    "entry": "kern_gate_sink_bf16",
                    "grid": ["tokens", 1, 1],
                    "block": [768, 1, 1],
                }
            ]
        },
    }

    # ---------------- sconv ops ----------------
    for c_width in (1024, 2304):
        grid_y = (c_width + 1023) // 1024
        ops[f"sconv_static_{c_width}_prefill"] = {
            "params": [
                "in buffer<bf16>", "in buffer<bf16>", "in state", "in buffer<i32>", "in buffer<i64>",
                "out buffer<bf16>", "i32", "i32", "i64",
            ],
            "impl": {
                "launches": [
                    {
                        "module": "sconv",
                        "entry": "kern_sconv_static_prefill",
                        "params": [
                            "in buffer<bf16>", "in buffer<bf16>", "in state", "in buffer<i32>", "in buffer<i64>",
                            "out buffer<bf16>", "i32", "i32", "i32", "i64",
                        ],
                        "grid": ["tokens", grid_y, 1],
                        "block": [128, 1, 1],
                        "args": [
                            {"param": 0}, {"param": 1}, {"param": 2}, {"param": 3}, {"param": 4},
                            {"param": 5}, _i32(c_width), {"param": 6}, {"param": 7}, {"param": 8},
                        ],
                    }
                ]
            },
        }
        ops[f"sconv_static_{c_width}_decode"] = {
            "params": [
                "in buffer<bf16>", "in buffer<bf16>", "inout state", "in buffer<i32>", "in buffer<i64>",
                "in buffer<i32>", "out buffer<bf16>", "i32", "i64",
            ],
            "impl": {
                "launches": [
                    {
                        "module": "sconv",
                        "entry": "kern_sconv_static_decode",
                        "params": [
                            "in buffer<bf16>", "in buffer<bf16>", "inout state", "in buffer<i32>", "in buffer<i64>",
                            "in buffer<i32>", "out buffer<bf16>", "i32", "i32", "i64",
                        ],
                        "grid": ["seqs", grid_y, 1],
                        "block": [128, 1, 1],
                        "args": [
                            {"param": 0}, {"param": 1}, {"param": 2}, {"param": 3}, {"param": 4},
                            {"param": 5}, {"param": 6}, _i32(c_width), {"param": 7}, {"param": 8},
                        ],
                    }
                ]
            },
        }
        ops[f"sconv_static_{c_width}_verify"] = {
            "params": [
                "in buffer<bf16>", "in buffer<bf16>", "in state", "in buffer<i32>", "in buffer<i64>",
                "in buffer<i32>", "out buffer<bf16>", "out buffer<bf16>", "i32", "i64", "i64", "i32",
            ],
            "impl": {
                "launches": [
                    {
                        "module": "sconv",
                        "entry": "kern_sconv_static_verify",
                        "params": [
                            "in buffer<bf16>", "in buffer<bf16>", "in state", "in buffer<i32>", "in buffer<i64>",
                            "in buffer<i32>", "out buffer<bf16>", "out buffer<bf16>", "i32", "i32", "i64", "i64", "i32",
                        ],
                        "grid": ["tokens", grid_y, 1],
                        "block": [128, 1, 1],
                        "args": [
                            {"param": 0}, {"param": 1}, {"param": 2}, {"param": 3}, {"param": 4},
                            {"param": 5}, {"param": 6}, {"param": 7}, _i32(c_width),
                            {"param": 8}, {"param": 9}, {"param": 10}, {"param": 11},
                        ],
                    }
                ]
            },
        }
        ops[f"sconv_prefill_commit_{c_width}"] = {
            "params": ["in buffer<bf16>", "inout state", "in buffer<i32>", "i32", "i32", "i64"],
            "impl": {
                "launches": [
                    {
                        "module": "sconv",
                        "entry": "kern_sconv_prefill_commit",
                        "params": ["in buffer<bf16>", "inout state", "in buffer<i32>", "i32", "i32", "i32", "i64"],
                        "grid": [grid_y, 1, 1],
                        "block": [128, 1, 1],
                        "args": [{"param": 0}, {"param": 1}, {"param": 2}, _i32(c_width), {"param": 3}, {"param": 4}, {"param": 5}],
                    }
                ]
            },
        }

    # MLP static sconv with fused addend + optional residual
    ops["sconv_mlp_prefill"] = {
        "params": [
            "in buffer<bf16>", "in buffer<bf16>", "in state", "in buffer<i32>", "in buffer<i64>",
            "out buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "i32", "i32", "i64", "i32",
        ],
        "impl": {
            "launches": [
                {
                    "module": "sconv",
                    "entry": "kern_sconv_static_mlp_prefill",
                    "params": [
                        "in buffer<bf16>", "in buffer<bf16>", "in state", "in buffer<i32>", "in buffer<i64>",
                        "out buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "i32", "i32", "i32", "i64", "i32",
                    ],
                    "grid": ["tokens", 3, 1],
                    "block": [128, 1, 1],
                    "args": [
                        {"param": 0}, {"param": 1}, {"param": 2}, {"param": 3}, {"param": 4},
                        {"param": 5}, {"param": 6}, {"param": 7}, _i32(2304),
                        {"param": 8}, {"param": 9}, {"param": 10}, {"param": 11},
                    ],
                }
            ]
        },
    }
    ops["sconv_mlp_decode"] = {
        "params": [
            "in buffer<bf16>", "in buffer<bf16>", "inout state", "in buffer<i32>", "in buffer<i64>",
            "in buffer<i32>", "out buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "i32", "i64", "i32",
        ],
        "impl": {
            "launches": [
                {
                    "module": "sconv",
                    "entry": "kern_sconv_static_mlp_decode",
                    "params": [
                        "in buffer<bf16>", "in buffer<bf16>", "inout state", "in buffer<i32>", "in buffer<i64>",
                        "in buffer<i32>", "out buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "i32", "i32", "i64", "i32",
                    ],
                    "grid": ["seqs", 3, 1],
                    "block": [128, 1, 1],
                    "args": [
                        {"param": 0}, {"param": 1}, {"param": 2}, {"param": 3}, {"param": 4},
                        {"param": 5}, {"param": 6}, {"param": 7}, {"param": 8}, _i32(2304),
                        {"param": 9}, {"param": 10}, {"param": 11},
                    ],
                }
            ]
        },
    }
    ops["sconv_mlp_verify"] = {
        "params": [
            "in buffer<bf16>", "in buffer<bf16>", "in state", "in buffer<i32>", "in buffer<i64>",
            "in buffer<i32>", "out buffer<bf16>", "out buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>",
            "i32", "i64", "i64", "i32", "i32",
        ],
        "impl": {
            "launches": [
                {
                    "module": "sconv",
                    "entry": "kern_sconv_static_mlp_verify",
                    "params": [
                        "in buffer<bf16>", "in buffer<bf16>", "in state", "in buffer<i32>", "in buffer<i64>",
                        "in buffer<i32>", "out buffer<bf16>", "out buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>",
                        "i32", "i32", "i64", "i64", "i32", "i32",
                    ],
                    "grid": ["tokens", 3, 1],
                    "block": [128, 1, 1],
                    "args": [
                        {"param": 0}, {"param": 1}, {"param": 2}, {"param": 3}, {"param": 4},
                        {"param": 5}, {"param": 6}, {"param": 7}, {"param": 8}, {"param": 9},
                        _i32(2304), {"param": 10}, {"param": 11}, {"param": 12}, {"param": 13}, {"param": 14},
                    ],
                }
            ]
        },
    }

    # Dynamic sconv (C=1024)
    ops["sconv_dynamic_prefill"] = {
        "params": [
            "in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "in state", "in buffer<i32>", "in buffer<i64>",
            "out buffer<bf16>", "i32", "i32", "i64",
        ],
        "impl": {
            "launches": [
                {
                    "module": "sconv",
                    "entry": "kern_sconv_dynamic_prefill",
                    "params": [
                        "in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "in state", "in buffer<i32>",
                        "in buffer<i64>", "out buffer<bf16>", "i32", "i32", "i64",
                    ],
                    "grid": ["tokens", 1, 1],
                    "block": [128, 1, 1],
                }
            ]
        },
    }
    ops["sconv_dynamic_decode"] = {
        "params": [
            "in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "inout state", "in buffer<i32>",
            "in buffer<i64>", "in buffer<i32>", "out buffer<bf16>", "i32", "i64",
        ],
        "impl": {
            "launches": [
                {
                    "module": "sconv",
                    "entry": "kern_sconv_dynamic_decode",
                    "params": [
                        "in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "inout state", "in buffer<i32>",
                        "in buffer<i64>", "in buffer<i32>", "out buffer<bf16>", "i32", "i64",
                    ],
                    "grid": ["seqs", 1, 1],
                    "block": [128, 1, 1],
                }
            ]
        },
    }
    ops["sconv_dynamic_verify"] = {
        "params": [
            "in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "in state", "in buffer<i32>",
            "in buffer<i64>", "in buffer<i32>", "out buffer<bf16>", "out buffer<bf16>", "i32", "i64", "i64", "i32",
        ],
        "impl": {
            "launches": [
                {
                    "module": "sconv",
                    "entry": "kern_sconv_dynamic_verify",
                    "params": [
                        "in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "in state", "in buffer<i32>",
                        "in buffer<i64>", "in buffer<i32>", "out buffer<bf16>", "out buffer<bf16>",
                        "i32", "i64", "i64", "i32",
                    ],
                    "grid": ["tokens", 1, 1],
                    "block": [128, 1, 1],
                }
            ]
        },
    }
    ops["sconv_spec_commit_all"] = {
        "params": ["in buffer<bf16>", "inout state", "in buffer<i32>", "in buffer<i32>", "in buffer<i32>"],
        "impl": {
            "launches": [
                {
                    "module": "sconv",
                    "entry": "kern_sconv_spec_commit_all",
                    "params": ["in buffer<bf16>", "inout state", "in buffer<i32>", "in buffer<i32>", "in buffer<i32>", "i32"],
                    "grid": ["seqs", NUM_VISITS, 3],
                    "block": [128, 1, 1],
                    "args": [{"param": 0}, {"param": 1}, {"param": 2}, {"param": 3}, {"param": 4}, _i32(max_seqs)],
                }
            ]
        },
    }

    # ---------------- attn_prologue + SWA commit ops ----------------
    ops["attn_log_tau"] = {
        "params": ["in buffer<i64>", "out buffer<f32>", "i32"],
        "impl": {
            "launches": [
                _triton_launch(
                    dsl,
                    "attn_log_tau",
                    ["in buffer<i64>", "out buffer<f32>", "i32"],
                    [_ceil_div("tokens", 256), 1, 1],
                    [{"param": 0}, {"param": 1}, {"param": 2}],
                )
            ]
        },
    }
    ops["attn_log_tau_decode"] = {
        "params": ["in buffer<i64>", "out buffer<f32>", "i32"],
        "impl": {
            "launches": [
                _triton_launch(
                    dsl,
                    "attn_log_tau",
                    ["in buffer<i64>", "out buffer<f32>", "i32"],
                    [_ceil_div("seqs", 256), 1, 1],
                    [{"param": 0}, {"param": 1}, {"param": 2}],
                )
            ]
        },
    }
    ops["attn_prologue_full_prefill"] = {
        "params": [
            "in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>",
            "in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "in buffer<f32>",
            "in buffer<f32>", "in buffer<f32>", "in buffer<i64>", "in buffer<i32>",
            "inout state", "out buffer<bf16>", "out buffer<bf16>",
            "i32", "i32", "i32", "i32", "i32",
        ],
        "impl": {
            "launches": [
                {
                    "module": "attn_prologue",
                    "entry": "kern_attn_prologue_full_prefill",
                    "grid": ["tokens", 6, 1],
                    "block": [256, 1, 1],
                }
            ]
        },
    }
    for op_name, grid_x in (
        ("attn_prologue_full_decode", "seqs"),
        ("attn_prologue_full_verify", "tokens"),
    ):
        ops[op_name] = {
            "params": [
                "in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>",
                "in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "in buffer<f32>",
                "in buffer<f32>", "in buffer<f32>", "in buffer<i64>", "in buffer<i32>", "in buffer<i32>",
                "inout state", "out buffer<bf16>", "out buffer<bf16>",
                "i32", "i32", "i32", "i32", "i32",
            ],
            "impl": {
                "launches": [
                    {
                        "module": "attn_prologue",
                        "entry": "kern_attn_prologue_full_decode",
                        "params": [
                            "in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>",
                            "in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "in buffer<f32>",
                            "in buffer<f32>", "in buffer<f32>", "in buffer<i64>", "in buffer<i32>", "in buffer<i32>",
                            "inout state", "out buffer<bf16>", "out buffer<bf16>",
                            "i32", "i32", "i32", "i32", "i32", "i32",
                        ],
                        "grid": [grid_x, 6, 1],
                        "block": [256, 1, 1],
                        "args": [
                            *[{"param": j} for j in range(21)],
                            _i32(page_columns),
                        ],
                    }
                ]
            },
        }
    ops["attn_prologue_swa_prefill"] = {
        "params": [
            "in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>",
            "in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>",
            "in buffer<f32>", "in buffer<f32>",
            "out buffer<bf16>", "out buffer<bf16>", "out buffer<fp8e4m3>", "out buffer<fp8e4m3>",
            "i32", "i32", "i32", "i32", "i32",
        ],
        "impl": {
            "launches": [
                {
                    "module": "attn_prologue",
                    "entry": "kern_attn_prologue_swa_prefill",
                    "grid": ["tokens", 6, 1],
                    "block": [256, 1, 1],
                }
            ]
        },
    }
    ops["attn_prologue_swa_decode"] = {
        "params": [
            "in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>",
            "in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>",
            "in buffer<f32>", "in buffer<f32>",
            "in buffer<i64>", "in buffer<i32>", "in buffer<i32>", "inout state",
            "out buffer<bf16>", "out buffer<bf16>",
            "i32", "i32", "i32", "i32", "i64",
        ],
        "impl": {
            "launches": [
                {
                    "module": "attn_prologue",
                    "entry": "kern_attn_prologue_swa_decode",
                    "grid": ["seqs", 6, 1],
                    "block": [256, 1, 1],
                }
            ]
        },
    }
    ops["attn_prologue_swa_verify"] = {
        "params": [
            "in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>",
            "in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>",
            "in buffer<f32>", "in buffer<f32>",
            "in buffer<i32>", "in buffer<i32>",
            "out buffer<bf16>", "out buffer<bf16>", "out buffer<fp8e4m3>", "out buffer<fp8e4m3>",
            "i32", "i32", "i32", "i32",
        ],
        "impl": {
            "launches": [
                {
                    "module": "attn_prologue",
                    "entry": "kern_attn_prologue_swa_verify",
                    "grid": ["tokens", 6, 1],
                    "block": [256, 1, 1],
                }
            ]
        },
    }
    ops["swa_ring_commit_prefill"] = {
        "params": [
            "in buffer<fp8e4m3>", "in buffer<fp8e4m3>", "in buffer<i64>", "in buffer<i32>",
            "inout state", "i32", "i64",
        ],
        "impl": {
            "launches": [
                {
                    "module": "attn_prologue",
                    "entry": "kern_swa_ring_commit_prefill",
                    "grid": ["tokens", 8, 1],
                    "block": [32, 1, 1],
                }
            ]
        },
    }
    ops["spec_swa_commit"] = {
        "params": [
            "in buffer<fp8e4m3>", "in buffer<fp8e4m3>", "in buffer<i64>", "in buffer<i32>",
            "in buffer<i32>", "in buffer<i32>", "inout state",
        ],
        "impl": {
            "launches": [
                {
                    "module": "attn_prologue",
                    "entry": "kern_spec_swa_commit_all",
                    "params": [
                        "in buffer<fp8e4m3>", "in buffer<fp8e4m3>", "in buffer<i64>", "in buffer<i32>",
                        "in buffer<i32>", "in buffer<i32>", "inout state", "i32",
                    ],
                    "grid": ["seqs", 40, 1],
                    "block": [256, 1, 1],
                    "args": [*[{"param": j} for j in range(7)], _i32(max_seqs)],
                }
            ]
        },
    }

    # ---------------- rel_attention ops ----------------
    full_if_params = [
        "in buffer<bf16>", "in state", "in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>",
        "in buffer<i32>", "in buffer<i32>", "out buffer<bf16>",
        "i32", "i32", "i32", "f32", "f32",
    ]
    full_k_params_gated = [
        "in buffer<bf16>", "in state", "in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>",
        "in buffer<i32>", "in buffer<i32>", "out buffer<bf16>", "out buffer<f32>",
        "i32", "i32", "i32", "i32", "f32", "f32",
    ]
    full_k_params_part = [
        "in buffer<bf16>", "in state", "in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>",
        "in buffer<i32>", "in buffer<i32>", "out buffer<f32>", "out buffer<f32>",
        "i32", "i32", "i32", "i32", "f32", "f32",
    ]
    combine_params = [
        "in buffer<f32>", "in buffer<f32>", "in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>",
        "i32", "i32", "i32",
    ]

    # Full Prefill: split-KV (tokens <= 64) or single-split gated (tokens >= 65)
    ops["attn_full_prefill"] = {
        "params": full_if_params,
        "impl": {
            "scratch": {
                "o_part": {"dtype": "f32", "shape": [16, 64, HEADS, HEAD_DIM]},
                "lse_part": {"dtype": "f32", "shape": [16, 64, HEADS]},
                "dummy_lse": {"dtype": "f32", "shape": [1]},
            },
            "launches": [
                _triton_launch(
                    dsl,
                    "attn_full_prefill_split",
                    full_k_params_part,
                    [_ceil_div("tokens", 8), 8 * 16, 1],
                    [
                        {"param": 0}, {"param": 1}, {"param": 2}, {"param": 3}, {"param": 4},
                        {"param": 5}, {"param": 6}, {"scratch": "o_part"}, {"scratch": "lse_part"},
                        {"param": 8}, _i32(page_columns), {"param": 9}, {"param": 10},
                        {"param": 11}, {"param": 12},
                    ],
                    when={"var": "tokens", "max": 64},
                ),
                _triton_launch(
                    dsl,
                    "attn_combine_s16",
                    combine_params,
                    ["tokens", HEADS, 4],
                    [
                        {"scratch": "o_part"}, {"scratch": "lse_part"},
                        {"param": 3}, {"param": 4}, {"param": 7},
                        {"param": 8}, {"param": 9}, {"param": 10},
                    ],
                    pdl=True,
                    when={"var": "tokens", "max": 64},
                ),
                _triton_launch(
                    dsl,
                    "attn_full_prefill",
                    full_k_params_gated,
                    [_ceil_div("tokens", 8), 8, 1],
                    [
                        {"param": 0}, {"param": 1}, {"param": 2}, {"param": 3}, {"param": 4},
                        {"param": 5}, {"param": 6}, {"param": 7}, {"scratch": "dummy_lse"},
                        {"param": 8}, _i32(page_columns), {"param": 9}, {"param": 10},
                        {"param": 11}, {"param": 12},
                    ],
                    when={"var": "tokens", "min": 65},
                ),
            ],
        },
    }

    # Full Decode: 64 splits when seqs <= 1, 32 splits when seqs >= 2
    ops["attn_full_decode"] = {
        "params": full_if_params,
        "impl": {
            "scratch": {
                "o_part": {"dtype": "f32", "shape": [64, "seqs", HEADS, HEAD_DIM]},
                "lse_part": {"dtype": "f32", "shape": [64, "seqs", HEADS]},
            },
            "launches": [
                _triton_launch(
                    dsl,
                    "attn_full_decode1",
                    full_k_params_part,
                    [1, 8 * 64, 1],
                    [
                        {"param": 0}, {"param": 1}, {"param": 2}, {"param": 3}, {"param": 4},
                        {"param": 5}, {"param": 6}, {"scratch": "o_part"}, {"scratch": "lse_part"},
                        {"param": 8}, _i32(page_columns), {"param": 9}, {"param": 10},
                        {"param": 11}, {"param": 12},
                    ],
                    when={"var": "seqs", "max": 1},
                ),
                _triton_launch(
                    dsl,
                    "attn_combine_s64",
                    combine_params,
                    ["seqs", HEADS, 4],
                    [
                        {"scratch": "o_part"}, {"scratch": "lse_part"},
                        {"param": 3}, {"param": 4}, {"param": 7},
                        {"param": 8}, {"param": 9}, {"param": 10},
                    ],
                    pdl=True,
                    when={"var": "seqs", "max": 1},
                ),
                _triton_launch(
                    dsl,
                    "attn_full_decode8",
                    full_k_params_part,
                    [1, 8 * 32, "seqs"],
                    [
                        {"param": 0}, {"param": 1}, {"param": 2}, {"param": 3}, {"param": 4},
                        {"param": 5}, {"param": 6}, {"scratch": "o_part"}, {"scratch": "lse_part"},
                        {"param": 8}, _i32(page_columns), {"param": 9}, {"param": 10},
                        {"param": 11}, {"param": 12},
                    ],
                    when={"var": "seqs", "min": 2},
                ),
                _triton_launch(
                    dsl,
                    "attn_combine_s32",
                    combine_params,
                    ["seqs", HEADS, 4],
                    [
                        {"scratch": "o_part"}, {"scratch": "lse_part"},
                        {"param": 3}, {"param": 4}, {"param": 7},
                        {"param": 8}, {"param": 9}, {"param": 10},
                    ],
                    pdl=True,
                    when={"var": "seqs", "min": 2},
                ),
            ],
        },
    }

    ops["attn_full_verify"] = {
        "params": full_if_params,
        "impl": {
            "scratch": {
                "o_part": {"dtype": "f32", "shape": [32, "tokens", HEADS, HEAD_DIM]},
                "lse_part": {"dtype": "f32", "shape": [32, "tokens", HEADS]},
            },
            "launches": [
                _triton_launch(
                    dsl,
                    "attn_full_verify",
                    full_k_params_part,
                    [1, 8 * 32, "seqs"],
                    [
                        {"param": 0}, {"param": 1}, {"param": 2}, {"param": 3}, {"param": 4},
                        {"param": 5}, {"param": 6}, {"scratch": "o_part"}, {"scratch": "lse_part"},
                        {"param": 8}, _i32(page_columns), {"param": 9}, {"param": 10},
                        {"param": 11}, {"param": 12},
                    ],
                ),
                _triton_launch(
                    dsl,
                    "attn_combine_s32",
                    combine_params,
                    ["tokens", HEADS, 4],
                    [
                        {"scratch": "o_part"}, {"scratch": "lse_part"},
                        {"param": 3}, {"param": 4}, {"param": 7},
                        {"param": 8}, {"param": 9}, {"param": 10},
                    ],
                    pdl=True,
                ),
            ],
        },
    }

    # SWA ops (cur: prefill & verify; ring-only: decode1 & decode8)
    swa_cur_if_params = [
        "in buffer<bf16>", "in state", "in buffer<fp8e4m3>", "in buffer<fp8e4m3>",
        "in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>",
        "in buffer<i32>", "in buffer<i32>", "out buffer<bf16>",
        "i32", "i32", "i32", "f32", "f32",
    ]
    swa_cur_k_params_gated = [
        "in buffer<bf16>", "in state", "in buffer<fp8e4m3>", "in buffer<fp8e4m3>",
        "in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>",
        "in buffer<i32>", "in buffer<i32>", "out buffer<bf16>", "out buffer<f32>",
        "i32", "i32", "i32", "f32", "f32",
    ]
    swa_ring_if_params = [
        "in buffer<bf16>", "in state",
        "in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>",
        "in buffer<i32>", "in buffer<i32>", "out buffer<bf16>",
        "i32", "i32", "i32", "f32", "f32",
    ]
    swa_ring_k_params_gated = [
        "in buffer<bf16>", "in state",
        "in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>",
        "in buffer<i32>", "in buffer<i32>", "out buffer<bf16>", "out buffer<f32>",
        "i32", "i32", "i32", "f32", "f32",
    ]
    swa_ring_k_params_part = [
        "in buffer<bf16>", "in state",
        "in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>",
        "in buffer<i32>", "in buffer<i32>", "out buffer<f32>", "out buffer<f32>",
        "i32", "i32", "i32", "f32", "f32",
    ]

    for op_name, mod_name, grid_x, grid_z in (
        ("attn_swa_prefill", "attn_swa_prefill", _ceil_div("tokens", 8), 1),
        ("attn_swa_verify", "attn_swa_verify", 1, "seqs"),
    ):
        ops[op_name] = {
            "params": swa_cur_if_params,
            "impl": {
                "scratch": {"dummy_lse": {"dtype": "f32", "shape": [1]}},
                "launches": [
                    _triton_launch(
                        dsl,
                        mod_name,
                        swa_cur_k_params_gated,
                        [grid_x, 8, grid_z],
                        [
                            *[{"param": j} for j in range(10)],
                            {"scratch": "dummy_lse"},
                            *[{"param": j} for j in range(10, 15)],
                        ],
                    )
                ],
            },
        }

    # SWA Decode: 8 splits when seqs <= 1, single-split gated when seqs >= 2
    ops["attn_swa_decode"] = {
        "params": swa_ring_if_params,
        "impl": {
            "scratch": {
                "o_part": {"dtype": "f32", "shape": [8, "seqs", HEADS, HEAD_DIM]},
                "lse_part": {"dtype": "f32", "shape": [8, "seqs", HEADS]},
                "dummy_lse": {"dtype": "f32", "shape": [1]},
            },
            "launches": [
                _triton_launch(
                    dsl,
                    "attn_swa_decode1",
                    swa_ring_k_params_part,
                    [1, 8 * 8, 1],
                    [
                        *[{"param": j} for j in range(7)],
                        {"scratch": "o_part"}, {"scratch": "lse_part"},
                        *[{"param": j} for j in range(8, 13)],
                    ],
                    when={"var": "seqs", "max": 1},
                ),
                _triton_launch(
                    dsl,
                    "attn_combine_s8",
                    combine_params,
                    ["seqs", HEADS, 4],
                    [
                        {"scratch": "o_part"}, {"scratch": "lse_part"},
                        {"param": 3}, {"param": 4}, {"param": 7},
                        {"param": 8}, {"param": 9}, {"param": 10},
                    ],
                    pdl=True,
                    when={"var": "seqs", "max": 1},
                ),
                _triton_launch(
                    dsl,
                    "attn_swa_decode8",
                    swa_ring_k_params_gated,
                    [1, 8, "seqs"],
                    [
                        *[{"param": j} for j in range(8)],
                        {"scratch": "dummy_lse"},
                        *[{"param": j} for j in range(8, 13)],
                    ],
                    when={"var": "seqs", "min": 2},
                ),
            ],
        },
    }


def emit_load_calls(
    load_calls: list[dict[str, Any]],
    kv_scales: dict[str, dict[str, float]] | None,
    ops: dict[str, Any],
    buffers: dict[str, Any],
) -> dict[int, tuple[float, float]]:
    """Populate `load` (`once: true`) calls for all 48 visits' k_scale / v_scale carry buffers.

    Returns {visit_idx: (k_scale, v_scale)} for baking sm_scale and v_scale into rel_attention calls.
    """
    ops.setdefault(
        "m3_fill_f32",
        {
            "params": ["out buffer<f32>", "i32", "f32"],
            "impl": {
                "launches": [
                    {
                        "module": "attn_prologue",
                        "entry": "kern_fill_f32",
                        "grid": [1, 1, 1],
                        "block": [1, 1, 1],
                    }
                ]
            },
        },
    )
    scales: dict[int, tuple[float, float]] = {}
    for v in range(NUM_VISITS):
        entry = (kv_scales or {}).get(str(v), {"k": 1.0, "v": 1.0})
        ks = float(entry["k"])
        vs = float(entry["v"])
        scales[v] = (ks, vs)
        for suffix, val in (("k_scale", ks), ("v_scale", vs)):
            buf_name = f"attn.v{v}.{suffix}"
            if buf_name in buffers:
                load_calls.append(
                    {"label": buf_name, "op": "m3_fill_f32", "args": [_b(buf_name), _i32(1), _f32(val)]}
                )
    return scales


# Alias per PROJECT.md § Interface Contracts
emit_derive_calls = emit_load_calls


def _mode_tokens(mode: str) -> Any:
    if mode in ("prefill", "round"):
        return "tokens"
    if mode in ("decode", "decode_batch"):
        return "seqs"
    raise ValueError(f"Unsupported mode: {mode}")


def emit_sconv_kv(
    calls: list[dict[str, Any]],
    mode: str,
    visit_idx: int,
    phys_idx: int,
    k_in: dict[str, Any],
    v_in: dict[str, Any],
    x_k: dict[str, Any],
    x_v: dict[str, Any],
    k_out: dict[str, Any],
    v_out: dict[str, Any],
    *,
    stride_k: int = 1024,
    stride_v: int = 1024,
    max_seqs: int = 8,
    positions_buf: str = "positions",
    valid_buf: str = "valid",
    lines_buf: str = "seq_state.lines",
) -> None:
    """Emit k_sconv and v_sconv calls for visit `visit_idx` (physical `phys_idx`)."""
    is_local = phys_idx not in FULL_LAYER_IDS
    t_var = _mode_tokens(mode)
    raw_base = visit_idx * max_seqs * 8 * SCONV_RAW_ROW_ELEMS

    for stream, x_in, x_src, out_buf, stride in (
        ("k", k_in, x_k, k_out, stride_k),
        ("v", v_in, x_v, v_out, stride_v),
    ):
        stream_off = visit_idx * SCONV_VISIT_BYTES + SCONV_STREAM_OFFSETS[stream]
        raw_col_off = SCONV_RAW_COL_OFFSETS[stream]
        if is_local:
            w_buf = _b(f"model.llm.layers.{phys_idx}.attn.{stream}_sconv.weight")
            if mode == "prefill":
                calls.append({
                    "label": f"v{visit_idx}.{stream}_sconv",
                    "op": "sconv_static_1024_prefill",
                    "args": [x_in, w_buf, _s("seq_state"), _b(lines_buf), _b(positions_buf), out_buf, _i32(stride), {"var": "tokens"}, _i64(stream_off)],
                })
                calls.append({
                    "label": f"v{visit_idx}.{stream}_sconv.commit",
                    "op": "sconv_prefill_commit_1024",
                    "args": [x_in, _s("seq_state"), _b(lines_buf), _i32(stride), {"var": "tokens"}, _i64(stream_off)],
                })
            elif mode in ("decode", "decode_batch"):
                calls.append({
                    "label": f"v{visit_idx}.{stream}_sconv",
                    "op": "sconv_static_1024_decode",
                    "args": [x_in, w_buf, _s("seq_state"), _b(lines_buf), _b(positions_buf), _b(valid_buf), out_buf, _i32(stride), _i64(stream_off)],
                })
            elif mode == "round":
                calls.append({
                    "label": f"v{visit_idx}.{stream}_sconv.verify",
                    "op": "sconv_static_1024_verify",
                    "args": [x_in, w_buf, _s("seq_state"), _b(lines_buf), _b(positions_buf), _b(valid_buf), out_buf, _b("sconv.verify_raw"), _i32(stride), _i64(stream_off), _i64(raw_base), _i32(raw_col_off)],
                })
        else:
            dyn_buf = _b(f"sconv.dyn_{stream}")
            w_proj = _b(f"model.llm.layers.{phys_idx}.attn.{stream}_sconv.weight_proj")
            w_static = _b(f"model.llm.layers.{phys_idx}.attn.{stream}_sconv.static_w")
            calls.append({
                "label": f"v{visit_idx}.{stream}_sconv.dyn",
                "op": "m3_gemm_bf16",
                "args": [x_src, w_proj, dyn_buf, {"var": t_var}, _i32(32), _i32(HIDDEN_SIZE)],
            })
            if mode == "prefill":
                calls.append({
                    "label": f"v{visit_idx}.{stream}_sconv",
                    "op": "sconv_dynamic_prefill",
                    "args": [x_in, w_static, dyn_buf, _s("seq_state"), _b(lines_buf), _b(positions_buf), out_buf, _i32(stride), {"var": "tokens"}, _i64(stream_off)],
                })
                calls.append({
                    "label": f"v{visit_idx}.{stream}_sconv.commit",
                    "op": "sconv_prefill_commit_1024",
                    "args": [x_in, _s("seq_state"), _b(lines_buf), _i32(stride), {"var": "tokens"}, _i64(stream_off)],
                })
            elif mode in ("decode", "decode_batch"):
                calls.append({
                    "label": f"v{visit_idx}.{stream}_sconv",
                    "op": "sconv_dynamic_decode",
                    "args": [x_in, w_static, dyn_buf, _s("seq_state"), _b(lines_buf), _b(positions_buf), _b(valid_buf), out_buf, _i32(stride), _i64(stream_off)],
                })
            elif mode == "round":
                calls.append({
                    "label": f"v{visit_idx}.{stream}_sconv.verify",
                    "op": "sconv_dynamic_verify",
                    "args": [x_in, w_static, dyn_buf, _s("seq_state"), _b(lines_buf), _b(positions_buf), _b(valid_buf), out_buf, _b("sconv.verify_raw"), _i32(stride), _i64(stream_off), _i64(raw_base), _i32(raw_col_off)],
                })


def emit_sconv_attn(
    calls: list[dict[str, Any]],
    mode: str,
    visit_idx: int,
    phys_idx: int,
    attn_proj_out: dict[str, Any],
    attn_out: dict[str, Any],
    *,
    stride_x: int = 2304,
    max_seqs: int = 8,
    positions_buf: str = "positions",
    valid_buf: str = "valid",
    lines_buf: str = "seq_state.lines",
) -> None:
    """Emit attn_sconv call for visit `visit_idx` (physical `phys_idx`)."""
    w_buf = _b(f"model.llm.layers.{phys_idx}.attn_sconv.weight")
    stream_off = visit_idx * SCONV_VISIT_BYTES + SCONV_STREAM_OFFSETS["attn"]
    if mode == "prefill":
        calls.append({
            "label": f"v{visit_idx}.attn_sconv",
            "op": "sconv_static_2304_prefill",
            "args": [attn_proj_out, w_buf, _s("seq_state"), _b(lines_buf), _b(positions_buf), attn_out, _i32(stride_x), {"var": "tokens"}, _i64(stream_off)],
        })
        calls.append({
            "label": f"v{visit_idx}.attn_sconv.commit",
            "op": "sconv_prefill_commit_2304",
            "args": [attn_proj_out, _s("seq_state"), _b(lines_buf), _i32(stride_x), {"var": "tokens"}, _i64(stream_off)],
        })
    elif mode in ("decode", "decode_batch"):
        calls.append({
            "label": f"v{visit_idx}.attn_sconv",
            "op": "sconv_static_2304_decode",
            "args": [attn_proj_out, w_buf, _s("seq_state"), _b(lines_buf), _b(positions_buf), _b(valid_buf), attn_out, _i32(stride_x), _i64(stream_off)],
        })
    elif mode == "round":
        raw_base = visit_idx * max_seqs * 8 * SCONV_RAW_ROW_ELEMS
        calls.append({
            "label": f"v{visit_idx}.attn_sconv.verify",
            "op": "sconv_static_2304_verify",
            "args": [attn_proj_out, w_buf, _s("seq_state"), _b(lines_buf), _b(positions_buf), _b(valid_buf), attn_out, _b("sconv.verify_raw"), _i32(stride_x), _i64(stream_off), _i64(raw_base), _i32(SCONV_RAW_COL_OFFSETS["attn"])],
        })


def emit_sconv_mlp(
    calls: list[dict[str, Any]],
    mode: str,
    visit_idx: int,
    phys_idx: int,
    moe_out: dict[str, Any],
    out: dict[str, Any],
    addend: dict[str, Any],
    residual: dict[str, Any] | None,
    *,
    stride_x: int = 2304,
    max_seqs: int = 8,
    positions_buf: str = "positions",
    valid_buf: str = "valid",
    lines_buf: str = "seq_state.lines",
) -> None:
    """Emit mlp_sconv with fused (residual + addend) + y epilogue."""
    w_buf = _b(f"model.llm.layers.{phys_idx}.mlp_sconv.weight")
    stream_off = visit_idx * SCONV_VISIT_BYTES + SCONV_STREAM_OFFSETS["mlp"]
    has_res = 1 if (residual is not None and (visit_idx % 6 != 0)) else 0
    res_arg = residual if has_res else addend

    if mode == "prefill":
        calls.append({
            "label": f"v{visit_idx}.mlp_sconv",
            "op": "sconv_mlp_prefill",
            "args": [moe_out, w_buf, _s("seq_state"), _b(lines_buf), _b(positions_buf), out, res_arg, addend, _i32(stride_x), {"var": "tokens"}, _i64(stream_off), _i32(has_res)],
        })
        calls.append({
            "label": f"v{visit_idx}.mlp_sconv.commit",
            "op": "sconv_prefill_commit_2304",
            "args": [moe_out, _s("seq_state"), _b(lines_buf), _i32(stride_x), {"var": "tokens"}, _i64(stream_off)],
        })
    elif mode in ("decode", "decode_batch"):
        calls.append({
            "label": f"v{visit_idx}.mlp_sconv",
            "op": "sconv_mlp_decode",
            "args": [moe_out, w_buf, _s("seq_state"), _b(lines_buf), _b(positions_buf), _b(valid_buf), out, res_arg, addend, _i32(stride_x), _i64(stream_off), _i32(has_res)],
        })
    elif mode == "round":
        raw_base = visit_idx * max_seqs * 8 * SCONV_RAW_ROW_ELEMS
        calls.append({
            "label": f"v{visit_idx}.mlp_sconv.verify",
            "op": "sconv_mlp_verify",
            "args": [moe_out, w_buf, _s("seq_state"), _b(lines_buf), _b(positions_buf), _b(valid_buf), out, _b("sconv.verify_raw"), res_arg, addend, _i32(stride_x), _i64(stream_off), _i64(raw_base), _i32(SCONV_RAW_COL_OFFSETS["mlp"]), _i32(has_res)],
        })


def emit_sconv_spec_commit(
    calls: list[dict[str, Any]],
    accepted_buf: str = "spec.accepted",
    valid_buf: str = "valid",
    lines_buf: str = "seq_state.lines",
) -> None:
    calls.append({
        "label": "sconv.spec_commit_all",
        "op": "sconv_spec_commit_all",
        "args": [_b("sconv.verify_raw"), _s("seq_state"), _b(lines_buf), _b(accepted_buf), _b(valid_buf)],
    })


def emit_swa_spec_commit(
    calls: list[dict[str, Any]],
    accepted_buf: str = "spec.accepted",
    positions_buf: str = "positions",
    valid_buf: str = "valid",
    lines_buf: str = "seq_state.lines",
) -> None:
    calls.append({
        "label": "swa.spec_commit_all",
        "op": "spec_swa_commit",
        "args": [
            _b("attn.verify_swa_k"),
            _b("attn.verify_swa_v"),
            _b(positions_buf),
            _b(lines_buf),
            _b(accepted_buf),
            _b(valid_buf),
            _s("seq_state"),
        ],
    })


def emit_layer_calls(
    calls: list[dict[str, Any]],
    mode: str,
    visit_idx: int,
    phys_idx: int,
    q_buf: dict[str, Any],
    r_buf: dict[str, Any],
    k_buf: dict[str, Any],
    v_buf: dict[str, Any],
    gate_buf: dict[str, Any],
    sink_buf: dict[str, Any],
    out_buf: dict[str, Any],
    k_scale: float,
    v_scale: float,
    *,
    stride_q: int = 3072,
    stride_r: int = 384,
    stride_k: int = 1024,
    stride_v: int = 1024,
    stride_gate: int = 3072,
    stride_sink: int = 24,
    max_seqs: int = 8,
    positions_buf: str = "positions",
    valid_buf: str = "valid",
    lengths_buf: str = "lengths",
    pages_buf: str = "pages",
    lines_buf: str = "seq_state.lines",
) -> None:
    """Emit attn_prologue + rel_attention (+ SWA prefill ring commit) for one visit."""
    is_local = phys_idx not in FULL_LAYER_IDS
    t_var = _mode_tokens(mode)
    q_scale_w = _b(f"model.llm.layers.{phys_idx}.attn.q_l2_clip.scale")
    r_scale_w = _b(f"model.llm.layers.{phys_idx}.attn.rel_l2_clip.scale")
    rel_proj_w = _b(f"model.llm.layers.{phys_idx}.attn.rel_logits_proj.proj")
    ks_carry = _b(f"attn.v{visit_idx}.k_scale")
    vs_carry = _b(f"attn.v{visit_idx}.v_scale")
    sm_scale = (HEAD_DIM**-0.5) * float(k_scale)

    if not is_local:
        full_idx = VISIT_TO_FULL_IDX[visit_idx]
        kv_state_arg = _s("kv", full_idx * KV_LAYER_PAGE_BYTES)
        if mode == "prefill":
            calls.append({
                "label": f"v{visit_idx}.attn_prologue",
                "op": "attn_prologue_full_prefill",
                "args": [
                    q_buf, r_buf, k_buf, v_buf, q_scale_w, r_scale_w, rel_proj_w,
                    _b("attn.tau"), ks_carry, vs_carry, _b(positions_buf), _b(pages_buf),
                    kv_state_arg, _b("attn.q_clip"), _b("attn.rel_bias"),
                    _i32(stride_q), _i32(stride_r), _i32(stride_k), _i32(stride_v), {"var": "tokens"},
                ],
            })
            attn_op = "attn_full_prefill"
        else:
            rows_per_seq = 8 if mode == "round" else 1
            prologue_op = "attn_prologue_full_verify" if mode == "round" else "attn_prologue_full_decode"
            calls.append({
                "label": f"v{visit_idx}.attn_prologue",
                "op": prologue_op,
                "args": [
                    q_buf, r_buf, k_buf, v_buf, q_scale_w, r_scale_w, rel_proj_w,
                    _b("attn.tau"), ks_carry, vs_carry, _b(positions_buf), _b(valid_buf), _b(pages_buf),
                    kv_state_arg, _b("attn.q_clip"), _b("attn.rel_bias"),
                    _i32(stride_q), _i32(stride_r), _i32(stride_k), _i32(stride_v), _i32(rows_per_seq),
                ],
            })
            if mode in ("decode", "decode_batch"):
                attn_op = "attn_full_decode"
            elif mode == "round":
                attn_op = "attn_full_verify"
            else:
                raise ValueError(f"Unsupported mode: {mode}")

        calls.append({
            "label": f"v{visit_idx}.rel_attention",
            "op": attn_op,
            "args": [
                _b("attn.q_clip"), kv_state_arg, _b("attn.rel_bias"), gate_buf, sink_buf,
                _b(pages_buf), _b(lengths_buf), out_buf,
                {"var": t_var}, _i32(stride_gate), _i32(stride_sink), _f32(sm_scale), _f32(v_scale),
            ],
        })
        return

    # Local SWA layer
    swa_idx = VISIT_TO_SWA_IDX[visit_idx]
    swa_off = SWA_TARGET_OFFSET + swa_idx * SWA_VISIT_BYTES
    swa_state_arg = _s("seq_state", swa_off)
    if mode == "prefill":
        calls.append({
            "label": f"v{visit_idx}.attn_prologue",
            "op": "attn_prologue_swa_prefill",
            "args": [
                q_buf, r_buf, k_buf, v_buf, q_scale_w, r_scale_w, rel_proj_w,
                ks_carry, vs_carry, _b("attn.q_clip"), _b("attn.rel_bias"),
                _b("attn.swa_k"), _b("attn.swa_v"),
                _i32(stride_q), _i32(stride_r), _i32(stride_k), _i32(stride_v), {"var": "tokens"},
            ],
        })
        calls.append({
            "label": f"v{visit_idx}.rel_attention",
            "op": "attn_swa_prefill",
            "args": [
                _b("attn.q_clip"), swa_state_arg, _b("attn.swa_k"), _b("attn.swa_v"),
                _b("attn.rel_bias"), gate_buf, sink_buf, _b(lines_buf), _b(lengths_buf), out_buf,
                {"var": "tokens"}, _i32(stride_gate), _i32(stride_sink), _f32(sm_scale), _f32(v_scale),
            ],
        })
        calls.append({
            "label": f"v{visit_idx}.swa_ring_commit",
            "op": "swa_ring_commit_prefill",
            "args": [
                _b("attn.swa_k"), _b("attn.swa_v"), _b(positions_buf), _b(lines_buf),
                _s("seq_state"), {"var": "tokens"}, _i64(swa_off),
            ],
        })
    elif mode in ("decode", "decode_batch"):
        calls.append({
            "label": f"v{visit_idx}.attn_prologue",
            "op": "attn_prologue_swa_decode",
            "args": [
                q_buf, r_buf, k_buf, v_buf, q_scale_w, r_scale_w, rel_proj_w,
                ks_carry, vs_carry, _b(positions_buf), _b(valid_buf), _b(lines_buf),
                _s("seq_state"), _b("attn.q_clip"), _b("attn.rel_bias"),
                _i32(stride_q), _i32(stride_r), _i32(stride_k), _i32(stride_v), _i64(swa_off),
            ],
        })
        attn_op = "attn_swa_decode"
        calls.append({
            "label": f"v{visit_idx}.rel_attention",
            "op": attn_op,
            "args": [
                _b("attn.q_clip"), swa_state_arg,
                _b("attn.rel_bias"), gate_buf, sink_buf, _b(lines_buf), _b(lengths_buf), out_buf,
                {"var": "seqs"}, _i32(stride_gate), _i32(stride_sink), _f32(sm_scale), _f32(v_scale),
            ],
        })
    elif mode == "round":
        slice_bytes = swa_idx * max_seqs * 8 * KV_HEADS * HEAD_DIM
        k_slice = _b("attn.verify_swa_k", slice_bytes)
        v_slice = _b("attn.verify_swa_v", slice_bytes)
        calls.append({
            "label": f"v{visit_idx}.attn_prologue",
            "op": "attn_prologue_swa_verify",
            "args": [
                q_buf, r_buf, k_buf, v_buf, q_scale_w, r_scale_w, rel_proj_w,
                ks_carry, vs_carry, _b(valid_buf), _b(lines_buf),
                _b("attn.q_clip"), _b("attn.rel_bias"), k_slice, v_slice,
                _i32(stride_q), _i32(stride_r), _i32(stride_k), _i32(stride_v),
            ],
        })
        calls.append({
            "label": f"v{visit_idx}.rel_attention",
            "op": "attn_swa_verify",
            "args": [
                _b("attn.q_clip"), swa_state_arg, k_slice, v_slice,
                _b("attn.rel_bias"), gate_buf, sink_buf, _b(lines_buf), _b(lengths_buf), out_buf,
                {"var": "tokens"}, _i32(stride_gate), _i32(stride_sink), _f32(sm_scale), _f32(v_scale),
            ],
        })


def prune_unused(manifest: dict[str, Any]) -> dict[str, Any]:
    """Prune unreferenced buffers/ops/modules/states/vars so `kern verify` passes on minimal test manifests."""
    used_ops: set[str] = set()
    used_bufs: set[str] = set()
    used_states: set[str] = set()
    used_vars: set[str] = set()

    def scan_expr(e: Any) -> None:
        if isinstance(e, str) and e in manifest["vars"]:
            used_vars.add(e)
        elif isinstance(e, dict):
            for v in e.values():
                scan_expr(v)
        elif isinstance(e, list):
            for v in e:
                scan_expr(v)

    for p in manifest["programs"].values():
        if "batch" in p and p["batch"]:
            scan_expr(p["batch"].get("rows"))
            scan_expr(p["batch"].get("context"))
        for c in p["calls"]:
            used_ops.add(c["op"])
            for a in c["args"]:
                if isinstance(a, dict):
                    if "buf" in a:
                        used_bufs.add(a["buf"])
                    if "state" in a:
                        used_states.add(a["state"])
                    if "var" in a:
                        used_vars.add(a["var"])
                    if "expr" in a:
                        scan_expr(a["expr"])

    manifest["ops"] = {k: v for k, v in manifest["ops"].items() if k in used_ops}
    used_mods: set[str] = set()
    for op in manifest["ops"].values():
        for l in op["impl"]["launches"]:
            if "module" in l:
                used_mods.add(l["module"])
            if "grid" in l:
                scan_expr(l["grid"])
            if "shared_mem" in l:
                scan_expr(l["shared_mem"])
            if "when" in l and l["when"]:
                used_vars.add(l["when"]["var"])
            for s in op["impl"].get("scratch", {}).values():
                scan_expr(s["shape"])
    manifest["modules"] = {k: v for k, v in manifest["modules"].items() if k in used_mods}
    manifest["buffers"] = {k: v for k, v in manifest["buffers"].items() if k in used_bufs}
    for b in manifest["buffers"].values():
        scan_expr(b["shape"])
        dom = b.get("domain")
        if dom:
            if "index_into" in dom and dom["index_into"] in manifest["states"]:
                used_states.add(dom["index_into"])
            scan_expr(dom.get("min"))
            scan_expr(dom.get("max"))
    manifest["states"] = {k: v for k, v in manifest["states"].items() if k in used_states}
    manifest["vars"] = {k: v for k, v in manifest["vars"].items() if k in used_vars}
    return manifest


def build_unit_manifest(
    kernels_dir: Path,
    *,
    max_seqs: int = 8,
    chunk: int = 2048,
    context: int = DEFAULT_CONTEXT,
    page_columns: int = DEFAULT_PAGE_COLUMNS,
    include_full_visit: int = 5,
    include_swa_visit: int = 0,
    kv_scales: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Build a standalone valid schema-v7 manifest exercising M3 ops for CPU `kern verify`."""
    modules: dict[str, Any] = {}
    states: dict[str, Any] = {}
    buffers: dict[str, Any] = {}
    ops: dict[str, Any] = {}
    dsl = register_modules(modules, kernels_dir)
    register_states(states)
    register_buffers(buffers, max_seqs=max_seqs, chunk=chunk, context=context, page_columns=page_columns)
    register_ops(ops, dsl, max_seqs=max_seqs, page_columns=page_columns)

    # Synthetic input/output/weight buffers to exercise all programs cleanly
    buffers["test.x_k"] = {"dtype": "bf16", "shape": ["tokens", HIDDEN_SIZE], "kind": "input"}
    buffers["test.x_v"] = {"dtype": "bf16", "shape": ["tokens", HIDDEN_SIZE], "kind": "input"}
    buffers["test.q"] = {"dtype": "bf16", "shape": ["tokens", 3072], "kind": "input"}
    buffers["test.r"] = {"dtype": "bf16", "shape": ["tokens", 384], "kind": "input"}
    buffers["test.k"] = {"dtype": "bf16", "shape": ["tokens", 1024], "kind": "input"}
    buffers["test.v"] = {"dtype": "bf16", "shape": ["tokens", 1024], "kind": "input"}
    buffers["test.gate"] = {"dtype": "bf16", "shape": ["tokens", 3072], "kind": "input"}
    buffers["test.sink"] = {"dtype": "bf16", "shape": ["tokens", 24], "kind": "input"}
    buffers["test.attn_proj"] = {"dtype": "bf16", "shape": ["tokens", HIDDEN_SIZE], "kind": "input"}
    buffers["test.moe_out"] = {"dtype": "bf16", "shape": ["tokens", HIDDEN_SIZE], "kind": "input"}
    buffers["test.res"] = {"dtype": "bf16", "shape": ["tokens", HIDDEN_SIZE], "kind": "input"}
    buffers["test.attn_sconv_out"] = {"dtype": "bf16", "shape": ["tokens", HIDDEN_SIZE], "kind": "workspace"}
    buffers["test.mlp_out"] = {"dtype": "bf16", "shape": ["tokens", HIDDEN_SIZE], "kind": "output"}
    buffers["test.attn_y"] = {"dtype": "bf16", "shape": ["tokens", 3072], "kind": "output"}
    buffers["spec.accepted"] = {"dtype": "i32", "shape": ["seqs"], "kind": "input", "domain": {"min": 1, "max": 8}}

    for p in (VISIT_TO_PHYS[include_full_visit], VISIT_TO_PHYS[include_swa_visit]):
        buffers[f"model.llm.layers.{p}.attn.q_l2_clip.scale"] = {"dtype": "bf16", "shape": [24, 128], "kind": "input"}
        buffers[f"model.llm.layers.{p}.attn.rel_l2_clip.scale"] = {"dtype": "bf16", "shape": [24, 16], "kind": "input"}
        buffers[f"model.llm.layers.{p}.attn.rel_logits_proj.proj"] = {"dtype": "bf16", "shape": [16, 1024], "kind": "input"}
        buffers[f"model.llm.layers.{p}.attn_sconv.weight"] = {"dtype": "bf16", "shape": [2304, 4], "kind": "input"}
        buffers[f"model.llm.layers.{p}.mlp_sconv.weight"] = {"dtype": "bf16", "shape": [2304, 4], "kind": "input"}
        if p in FULL_LAYER_IDS:
            for st in ("k", "v"):
                buffers[f"model.llm.layers.{p}.attn.{st}_sconv.static_w"] = {"dtype": "bf16", "shape": [4, 1024], "kind": "input"}
                buffers[f"model.llm.layers.{p}.attn.{st}_sconv.weight_proj"] = {"dtype": "bf16", "shape": [32, HIDDEN_SIZE], "kind": "input"}
        else:
            for st in ("k", "v"):
                buffers[f"model.llm.layers.{p}.attn.{st}_sconv.weight"] = {"dtype": "bf16", "shape": [1024, 4], "kind": "input"}

    derive_calls: list[dict[str, Any]] = []
    unit_kv_scales = kv_scales if kv_scales is not None else {
        str(include_swa_visit): {"k": 0.0505, "v": 0.0252},
        str(include_full_visit): {"k": 0.0505, "v": 0.0252},
    }
    scales = emit_derive_calls(derive_calls, unit_kv_scales, ops, buffers)

    programs: dict[str, Any] = {
        "derive": {"derive": True, "calls": derive_calls},
    }
    for prog_name, mode, batch in (
        ("prefill", "prefill", {"groups": 1, "rows": "tokens", "context": "ctx"}),
        ("decode", "decode_batch", {"groups": max_seqs, "rows": 1}),
        ("round", "round", {"groups": max_seqs, "rows": 8}),
    ):
        tau_op = "attn_log_tau_decode" if mode in ("decode", "decode_batch") else "attn_log_tau"
        tau_var = "seqs" if mode in ("decode", "decode_batch") else "tokens"
        calls: list[dict[str, Any]] = [
            {"label": "log_tau", "op": tau_op, "args": [_b("positions"), _b("attn.tau"), {"var": tau_var}]}
        ]
        for v_id in (include_swa_visit, include_full_visit):
            p_id = VISIT_TO_PHYS[v_id]
            ks_v, vs_v = scales[v_id]
            emit_sconv_kv(
                calls, mode, v_id, p_id,
                _b("test.k"), _b("test.v"), _b("test.x_k"), _b("test.x_v"),
                _b("sconv.k_out"), _b("sconv.v_out"), max_seqs=max_seqs,
            )
            emit_layer_calls(
                calls, mode, v_id, p_id,
                _b("test.q"), _b("test.r"), _b("sconv.k_out"), _b("sconv.v_out"),
                _b("test.gate"), _b("test.sink"), _b("test.attn_y"),
                ks_v, vs_v, max_seqs=max_seqs,
            )
            emit_sconv_attn(calls, mode, v_id, p_id, _b("test.attn_proj"), _b("test.attn_sconv_out"), max_seqs=max_seqs)
            emit_sconv_mlp(
                calls, mode, v_id, p_id,
                _b("test.moe_out"), _b("test.mlp_out"), _b("test.attn_sconv_out"), _b("test.res"),
                max_seqs=max_seqs,
            )
        if mode == "round":
            emit_sconv_spec_commit(calls)
            emit_swa_spec_commit(calls)
        programs[prog_name] = {"batch": batch, "graph": mode != "prefill", "calls": calls}

    manifest = {
        "schema_version": 7,
        "model": "inkling-2-mini-m3-unit",
        "vars": {
            "tokens": {"max": max(chunk, max_seqs * 8), "axis": "rows"},
            "seqs": {"max": max_seqs, "axis": "groups"},
            "ctx": {"max": context},
        },
        "states": states,
        "buffers": buffers,
        "modules": modules,
        "ops": ops,
        "programs": programs,
    }
    return prune_unused(manifest)
