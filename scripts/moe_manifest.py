#!/usr/bin/env python3
"""M2 manifest builders for Inkling-2-Mini SM120 NVFP4 & MXFP8 MoE.

Exports:
  - Architectural constants (HIDDEN_SIZE=2304, INTER_SIZE=1152, NUM_EXPERTS=128, TOP_K=8, ...)
  - Typed builders: load_meta, register_modules, register_buffers, register_ops,
    emit_derive_calls, emit_moe_layer_calls
  - Standalone helper: build_unit_manifest(...) for CPU `kern verify` & tests
"""

from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any

HIDDEN_SIZE = 2304
INTER_SIZE = 1152
NUM_EXPERTS = 128
TOP_K = 8
ROUTE_SCALE = 8.0
GATE_SOFTCAP = 4.0
UP_SOFTCAP = 25.0

NUM_PHYSICAL_LAYERS = 24
NVFP4_LAYERS = range(0, 20)
MXFP8_LAYERS = range(20, 24)

NVFP4_GROUP = 16
MXFP8_GROUP = 32
DEFAULT_SM_COUNT = 128

M2_MODULES = (
    "moe_route",
    "moe_swiglu_quant",
    "moe_combine",
    "moe_nvfp4_sm120",
    "moe_mxfp8_sm120",
)

_b = lambda name, offset=0: {"buf": name, **({"offset": offset} if offset else {})}
_i32 = lambda v: {"i32": int(v)}
_expr = lambda e: {"expr": e}
_mul = lambda x, y: {"mul": [x, y]}
_ceil_div = lambda x, d: {"ceil_div": [x, d]}
_add = lambda x, y: {"add": [x, y]}
_clamp = lambda x, lo, len_: {"clamp": [x, lo, len_]}


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def max_swizzled_scale_rows(max_tokens: int, num_experts: int = NUM_EXPERTS) -> int:
    """Upper bound on 128-row-aligned scale rows across `num_experts` with per-expert 128-alignment."""
    max_n = max_tokens * TOP_K
    return (ceil_div(max_n, 128) + num_experts) * 128


def load_meta(kernels_dir: Path) -> dict[str, Any]:
    """Load M2 cubin + CUTLASS ABI metadata from `kernels_dir`."""
    build_json = kernels_dir / "moe_build.json"
    if build_json.exists():
        return json.loads(build_json.read_text())
    modules: dict[str, Any] = {}
    for name in M2_MODULES:
        p = kernels_dir / f"{name}.json"
        if p.exists():
            modules[name] = json.loads(p.read_text())
    if not modules:
        raise FileNotFoundError(f"No M2 metadata JSON found in {kernels_dir}")
    return {"architecture": "sm_120a", "modules": modules}


def register_modules(
    modules: dict[str, Any],
    kernels_dir: Path,
    meta: dict[str, Any] | None = None,
    *,
    include_nvfp4: bool = True,
    include_mxfp8: bool = True,
) -> dict[str, Any]:
    """SHA-256 pin M2 CUDA modules in `modules` and return loaded M2 metadata."""
    if meta is None:
        meta = load_meta(kernels_dir)
    for name in M2_MODULES:
        if name == "moe_nvfp4_sm120" and not include_nvfp4:
            continue
        if name == "moe_mxfp8_sm120" and not include_mxfp8:
            continue
        cubin = kernels_dir / f"{name}.cubin"
        if not cubin.exists():
            raise FileNotFoundError(f"Missing M2 cubin: {cubin}")
        sha = hashlib.sha256(cubin.read_bytes()).hexdigest()
        modules[name] = {"source": cubin.name, "sha256": sha}
    return meta


def layer_weight_names(phys_idx: int) -> dict[str, str]:
    """Canonical buffer names for physical layer `phys_idx` MoE weights & derived scales."""
    pfx = f"moe.l{phys_idx}"
    base = {
        "gate_weight": f"{pfx}.gate.weight",
        "gate_bias": f"{pfx}.gate.bias",
        "w13_weight": f"{pfx}.w13.weight",
        "w13_scale_src": f"{pfx}.w13.scale.src",
        "w13_scale": f"{pfx}.w13.scale",
        "w2_weight": f"{pfx}.w2.weight",
        "w2_scale_src": f"{pfx}.w2.scale.src",
        "w2_scale": f"{pfx}.w2.scale",
    }
    if phys_idx in NVFP4_LAYERS:
        base.update(
            {
                "w13_amax": f"{pfx}.w13.amax",
                "w13_denom": f"{pfx}.w13.denom",
                "w2_amax": f"{pfx}.w2.amax",
                "w2_denom": f"{pfx}.w2.denom",
            }
        )
    return base


def register_buffers(
    buffers: dict[str, Any],
    max_seqs: int = 8,
    max_tokens: int = 1024,
    *,
    num_layers: int = NUM_PHYSICAL_LAYERS,
    layers: range | list[int] | None = None,
    token_axis: str = "tokens",
    include_weights: bool = True,
    include_workspaces: bool = True,
    include_nvfp4: bool | None = None,
    include_mxfp8: bool | None = None,
    prefix: str = "model.llm.layers",
) -> dict[int, dict[str, str]]:
    """Declare shared MoE workspaces and (optionally) per-layer weight/source/carry buffers.

    Notes for M5 `gen.py` integration:
      - M5 `gen.py` must register `"moe_out"` (`{"dtype": "bf16", "shape": [token_axis, 2304], "kind": "workspace"}`)
        as the output buffer consumed by `emit_moe_layer_calls(..., output_buf="moe_out")`.
      - When `include_workspaces=True`, existing shared workspace buffers whose integer first
        dimension depends on `max_tokens` (e.g. `moe.xsf_*`, `moe.act_sf_*`) grow to
        `max(existing, required)` rather than staying clamped to an earlier smaller `max_tokens`.
    """
    _ = max_seqs

    def put(name: str, dtype: str, shape: list[Any], kind: str = "workspace", **extra: Any) -> None:
        existing = buffers.get(name)
        if existing is None:
            buffers[name] = {"dtype": dtype, "shape": shape, "kind": kind, **extra}
            return
        if (
            kind == "workspace"
            and existing.get("kind") == "workspace"
            and existing.get("dtype") == dtype
            and len(shape) >= 1
            and isinstance(shape[0], int)
            and len(existing.get("shape", [])) == len(shape)
            and isinstance(existing["shape"][0], int)
            and existing["shape"][1:] == shape[1:]
            and existing["shape"][0] < shape[0]
        ):
            existing["shape"] = [shape[0], *existing["shape"][1:]]

    phys_range = list(range(num_layers) if layers is None else layers)
    if include_workspaces:
        use_nvfp4 = any(i in NVFP4_LAYERS for i in phys_range) if include_nvfp4 is None else include_nvfp4
        use_mxfp8 = any(i in MXFP8_LAYERS for i in phys_range) if include_mxfp8 is None else include_mxfp8

        sf_rows = max_swizzled_scale_rows(max_tokens)

        # Shared routing workspaces
        put("moe.logits", "f32", [token_axis, NUM_EXPERTS])
        put("moe.ids", "i32", [token_axis, TOP_K])
        put("moe.weights", "f32", [token_axis, TOP_K])
        put("moe.sorted_ids", "i32", [token_axis, TOP_K])
        put("moe.src2dst", "i32", [token_axis, TOP_K])
        put("moe.dst2src", "i32", [token_axis, TOP_K])
        put("moe.gammas", "f32", [token_axis, TOP_K])
        put("moe.counts", "i32", [NUM_EXPERTS])
        put("moe.m_indptr", "i32", [NUM_EXPERTS + 1])

        if use_nvfp4 or use_mxfp8:
            put("moe.w13_out_bf16", "bf16", [token_axis, TOP_K, 2 * INTER_SIZE])
            put("moe.w2_out_bf16", "bf16", [token_axis, TOP_K, HIDDEN_SIZE])

        if use_nvfp4:
            # Shared NVFP4 workspaces (Layers 0..19)
            put("moe.xq_nvfp4", "u8", [token_axis, TOP_K, HIDDEN_SIZE // 2])
            put("moe.xsf_nvfp4", "fp8e4m3", [sf_rows, HIDDEN_SIZE // NVFP4_GROUP])
            put("moe.x_amax", "f32", [token_axis, TOP_K])
            put("moe.act_nvfp4", "u8", [token_axis, TOP_K, INTER_SIZE // 2])
            put("moe.act_sf_nvfp4", "fp8e4m3", [sf_rows, INTER_SIZE // NVFP4_GROUP])
            put("moe.act_amax", "f32", [token_axis, TOP_K])

        if use_mxfp8:
            # Shared MXFP8 workspaces (Layers 20..23)
            put("moe.xq_mxfp8", "fp8e4m3", [token_axis, TOP_K, HIDDEN_SIZE])
            put("moe.xsf_mxfp8", "u8", [sf_rows, HIDDEN_SIZE // MXFP8_GROUP])
            put("moe.act_mxfp8", "fp8e4m3", [token_axis, TOP_K, INTER_SIZE])
            put("moe.act_sf_mxfp8", "u8", [sf_rows, INTER_SIZE // MXFP8_GROUP])

    layer_weights: dict[int, dict[str, str]] = {}
    for phys_idx in phys_range:
        names = layer_weight_names(phys_idx)
        layer_weights[phys_idx] = names
        if not include_weights:
            continue
        lp = f"{prefix}.{phys_idx}.mlp"
        put(names["gate_weight"], "bf16", [NUM_EXPERTS, HIDDEN_SIZE], "weight", bind=[{"tensor": f"{lp}.gate.weight"}])
        put(names["gate_bias"], "f32", [NUM_EXPERTS], "weight", bind=[{"tensor": f"{lp}.gate.bias"}])
        if phys_idx in NVFP4_LAYERS:
            put(
                names["w13_weight"],
                "u8",
                [NUM_EXPERTS, 2 * INTER_SIZE, HIDDEN_SIZE // 2],
                "weight",
                bind=[{"tensor": f"{lp}.experts.w13_weight"}],
            )
            put(
                names["w13_scale_src"],
                "fp8e4m3",
                [NUM_EXPERTS, 2 * INTER_SIZE, HIDDEN_SIZE // NVFP4_GROUP],
                "source",
                bind=[{"tensor": f"{lp}.experts.w13_weight_scale"}],
            )
            put(
                names["w13_scale"],
                "fp8e4m3",
                [NUM_EXPERTS, 2 * INTER_SIZE, HIDDEN_SIZE // NVFP4_GROUP],
                "carry",
            )
            put(names["w13_amax"], "f32", [NUM_EXPERTS], "weight", bind=[{"tensor": f"{lp}.experts.w13_weight_amax"}])
            put(
                names["w13_denom"],
                "f32",
                [1],
                "weight",
                bind=[{"tensor": f"{lp}.experts.w13_weight_outer_scale_denominator"}],
            )
            put(
                names["w2_weight"],
                "u8",
                [NUM_EXPERTS, HIDDEN_SIZE, INTER_SIZE // 2],
                "weight",
                bind=[{"tensor": f"{lp}.experts.w2_weight"}],
            )
            put(
                names["w2_scale_src"],
                "fp8e4m3",
                [NUM_EXPERTS, HIDDEN_SIZE, INTER_SIZE // NVFP4_GROUP],
                "source",
                bind=[{"tensor": f"{lp}.experts.w2_weight_scale"}],
            )
            put(
                names["w2_scale"],
                "fp8e4m3",
                [NUM_EXPERTS, HIDDEN_SIZE, INTER_SIZE // NVFP4_GROUP],
                "carry",
            )
            put(names["w2_amax"], "f32", [NUM_EXPERTS], "weight", bind=[{"tensor": f"{lp}.experts.w2_weight_amax"}])
            put(
                names["w2_denom"],
                "f32",
                [1],
                "weight",
                bind=[{"tensor": f"{lp}.experts.w2_weight_outer_scale_denominator"}],
            )
        else:
            put(
                names["w13_weight"],
                "fp8e4m3",
                [NUM_EXPERTS, 2 * INTER_SIZE, HIDDEN_SIZE],
                "weight",
                bind=[{"tensor": f"{lp}.experts.w13_weight"}],
            )
            put(
                names["w13_scale_src"],
                "u8",
                [NUM_EXPERTS, 2 * INTER_SIZE, HIDDEN_SIZE // MXFP8_GROUP],
                "source",
                bind=[{"tensor": f"{lp}.experts.w13_weight_scale"}],
            )
            put(
                names["w13_scale"],
                "u8",
                [NUM_EXPERTS, 2 * INTER_SIZE, HIDDEN_SIZE // MXFP8_GROUP],
                "carry",
            )
            put(
                names["w2_weight"],
                "fp8e4m3",
                [NUM_EXPERTS, HIDDEN_SIZE, INTER_SIZE],
                "weight",
                bind=[{"tensor": f"{lp}.experts.w2_weight"}],
            )
            put(
                names["w2_scale_src"],
                "u8",
                [NUM_EXPERTS, HIDDEN_SIZE, INTER_SIZE // MXFP8_GROUP],
                "source",
                bind=[{"tensor": f"{lp}.experts.w2_weight_scale"}],
            )
            put(
                names["w2_scale"],
                "u8",
                [NUM_EXPERTS, HIDDEN_SIZE, INTER_SIZE // MXFP8_GROUP],
                "carry",
            )
    return layer_weights


def _build_gemm_pack(
    variant: dict[str, Any],
    sm_count: int,
    is_nvfp4: bool,
) -> dict[str, Any]:
    """Assemble `kern` Pack dict for SM120 CUTLASS NVFP4 or MXFP8 grouped GEMM."""
    abi = variant["abi"]
    off = abi["offsets"]
    tile_m = int(variant.get("tile_m", 128))
    tile_n = int(variant["tile_n"])

    # Patch sm_count into template byte image cleanly via structured fields
    fields: list[dict[str, Any]] = []
    covered: set[int] = set()

    def add_field(at: int, width: int, **kv: Any) -> None:
        covered.update(range(at, at + width))
        fields.append({"at": at, **kv})

    add_field(off["problem_shapes"], 8, scratch="problem_sizes")
    add_field(off["scheduler_problem_shapes"], 8, scratch="problem_sizes")
    add_field(off["ptr_A"], 8, scratch="A_ptr")
    add_field(off["dA"], 8, scratch="stride_A")
    add_field(off["ptr_B"], 8, scratch="B_ptr")
    add_field(off["dB"], 8, scratch="stride_B")
    add_field(off["ptr_SFA"], 8, scratch="SFA_ptr")
    add_field(off["layout_SFA"], 8, scratch="layout_SFA")
    add_field(off["ptr_SFB"], 8, scratch="SFB_ptr")
    add_field(off["layout_SFB"], 8, scratch="layout_SFB")
    add_field(off["ptr_D"], 8, scratch="D_ptr")
    add_field(off["dD"], 8, scratch="stride_D")
    if is_nvfp4:
        add_field(off["evt_row_scale_ptrs"], 8, scratch="x_amax_ptr")
        add_field(off["evt_group_amax_ptr"], 8, param=7)  # w_amax [E]
        add_field(off["evt_rcp_ptr"], 8, scratch="group_scale")  # rcp scalar [1]
    add_field(off["mainloop_tensormaps"], 8, scratch="main_ws")
    add_field(off["epilogue_tensormaps"], 8, scratch="epi_ws")
    add_field(off["hw_info_sm_count"], 4, i32=int(sm_count))
    add_field(off["scheduler_blocks_across_problem"], 8, i64=int(sm_count))

    # 5 initial TMA descriptors (dims/strides overwritten on device per expert,
    # while format, box, swizzle, l2_promotion are preserved from these templates).
    if is_nvfp4:
        m_sf_tiles = tile_m // 128
        add_field(
            off["tma_load_a"],
            128,
            tensormap={
                "param": 1,  # B_weights (>= 150 MB)
                "dtype": "u4packed",
                "dims": [128, tile_m],
                "strides": [64],
                "box": [128, tile_m],
                "swizzle": 64,
                "l2_promotion": 128,
            },
        )
        add_field(
            off["tma_load_b"],
            128,
            tensormap={
                "param": 1,
                "dtype": "u4packed",
                "dims": [128, tile_n],
                "strides": [64],
                "box": [128, tile_n],
                "swizzle": 64,
                "l2_promotion": 128,
            },
        )
        add_field(
            off["tma_load_sfa"],
            128,
            tensormap={
                "param": 3,  # SFB_weights (>= 20 MB)
                "dtype": "u16",
                "dims": [256, 2, m_sf_tiles, 1],
                "strides": [512 * m_sf_tiles, 512, 1024 * m_sf_tiles] if m_sf_tiles > 1 else [512, 1024, 1024],
                "box": [256, 2, m_sf_tiles, 1],
                "swizzle": 0,
                "l2_promotion": 128,
            },
        )
        add_field(
            off["tma_load_sfb"],
            128,
            tensormap={
                "param": 3,
                "dtype": "u16",
                "dims": [256, 2, m_sf_tiles, 1],
                "strides": [512, 1024, 1024 * m_sf_tiles],
                "box": [256, 2, 1, 1],
                "swizzle": 0,
                "l2_promotion": 128,
            },
        )
        add_field(
            off["tma_store_d"],
            128,
            tensormap={
                "param": 4,  # D_out bf16 (>= 36,864 B at tokens=1)
                "dtype": "bf16",
                "dims": [tile_m, tile_n],
                "strides": [tile_m * 2],
                "box": [64, 32],
                "swizzle": 128,
                "l2_promotion": 128,
            },
        )
    else:
        add_field(
            off["tma_load_a"],
            128,
            tensormap={
                "param": 1,
                "dtype": "u8",
                "dims": [128, 128],
                "strides": [128],
                "box": [128, 128],
                "swizzle": 128,
                "l2_promotion": 128,
            },
        )
        add_field(
            off["tma_load_b"],
            128,
            tensormap={
                "param": 1,
                "dtype": "u8",
                "dims": [128, tile_n],
                "strides": [128],
                "box": [128, tile_n],
                "swizzle": 128,
                "l2_promotion": 128,
            },
        )
        add_field(
            off["tma_load_sfa"],
            128,
            tensormap={
                "param": 3,
                "dtype": "u16",
                "dims": [256, 1, 1, 1],
                "strides": [512, 512, 512],
                "box": [256, 1, 1, 1],
                "swizzle": 0,
                "l2_promotion": 128,
            },
        )
        add_field(
            off["tma_load_sfb"],
            128,
            tensormap={
                "param": 3,
                "dtype": "u16",
                "dims": [256, 1, 1, 1],
                "strides": [512, 512, 512],
                "box": [256, 1, 1, 1],
                "swizzle": 0,
                "l2_promotion": 128,
            },
        )
        add_field(
            off["tma_store_d"],
            128,
            tensormap={
                "param": 4,  # D_out (>= 36,864 B at tokens=1)
                "dtype": "bf16",
                "dims": [128, tile_n],
                "strides": [256],
                "box": [64, 32],
                "swizzle": 128,
                "l2_promotion": 128,
            },
        )

    static_fields = [
        {"at": at, "u8": val}
        for at, val in enumerate(abi["bytes"])
        if val and at not in covered
    ]
    return {"size": abi["size"], "fields": static_fields + fields}


def _register_gemm_op(
    ops: dict[str, Any],
    op_name: str,
    mod_name: str,
    mod_meta: dict[str, Any],
    is_nvfp4: bool,
    max_tokens: int,
    sm_count: int,
    token_axis: str,
) -> None:
    if op_name in ops:
        return
    if is_nvfp4:
        params = [
            "in buffer<u8>",
            "in buffer<u8>",
            "in buffer<fp8e4m3>",
            "in buffer<fp8e4m3>",
            "out buffer<bf16>",
            "in buffer<i32>",
            "in buffer<f32>",
            "in buffer<f32>",
            "in buffer<f32>",
            "i32",
            "i32",
            "i32",
        ]
    else:
        params = [
            "in buffer<fp8e4m3>",
            "in buffer<fp8e4m3>",
            "in buffer<u8>",
            "in buffer<u8>",
            "out buffer<bf16>",
            "in buffer<i32>",
            "i32",
            "i32",
            "i32",
        ]

    variants = mod_meta["gemm_variants"]
    ref = variants["n64"]
    sz = ref["sizeof"]
    epi_ws_bytes = int(ref["epi_ws_per_sm"]) * sm_count
    main_ws_bytes = int(ref["main_ws_per_sm"]) * sm_count

    scratch = {
        "problem_sizes": {"dtype": "u8", "shape": [NUM_EXPERTS * int(sz["problem_shape"])]},
        "A_ptr": {"dtype": "u64", "shape": [NUM_EXPERTS]},
        "B_ptr": {"dtype": "u64", "shape": [NUM_EXPERTS]},
        "SFA_ptr": {"dtype": "u64", "shape": [NUM_EXPERTS]},
        "SFB_ptr": {"dtype": "u64", "shape": [NUM_EXPERTS]},
        "D_ptr": {"dtype": "u64", "shape": [NUM_EXPERTS]},
        "stride_A": {"dtype": "u8", "shape": [NUM_EXPERTS * int(sz["stride_a"])]},
        "stride_B": {"dtype": "u8", "shape": [NUM_EXPERTS * int(sz["stride_b"])]},
        "stride_D": {"dtype": "u8", "shape": [NUM_EXPERTS * int(sz["stride_d"])]},
        "layout_SFA": {"dtype": "u8", "shape": [NUM_EXPERTS * int(sz["layout_sfa"])]},
        "layout_SFB": {"dtype": "u8", "shape": [NUM_EXPERTS * int(sz["layout_sfb"])]},
        "epi_ws": {"dtype": "u8", "shape": [epi_ws_bytes]},
        "main_ws": {"dtype": "u8", "shape": [main_ws_bytes]},
    }
    if is_nvfp4:
        scratch["x_amax_ptr"] = {"dtype": "u64", "shape": [NUM_EXPERTS]}
        scratch["group_scale"] = {"dtype": "f32", "shape": [NUM_EXPERTS]}
        prep_params = (
            params
            + ["out buffer<u8>"]
            + ["out buffer<u64>"] * 6
            + ["out buffer<f32>"]
            + ["out buffer<u8>"] * 5
        )
        prep_args = [{"param": i} for i in range(len(params))] + [
            {"scratch": s}
            for s in (
                "problem_sizes",
                "A_ptr",
                "B_ptr",
                "SFA_ptr",
                "SFB_ptr",
                "D_ptr",
                "x_amax_ptr",
                "group_scale",
                "stride_A",
                "stride_B",
                "stride_D",
                "layout_SFA",
                "layout_SFB",
            )
        ]
    else:
        prep_params = params + ["out buffer<u8>"] + ["out buffer<u64>"] * 5 + ["out buffer<u8>"] * 5
        prep_args = [{"param": i} for i in range(len(params))] + [
            {"scratch": s}
            for s in (
                "problem_sizes",
                "A_ptr",
                "B_ptr",
                "SFA_ptr",
                "SFB_ptr",
                "D_ptr",
                "stride_A",
                "stride_B",
                "stride_D",
                "layout_SFA",
                "layout_SFB",
            )
        ]

    launches: list[dict[str, Any]] = [
        {
            "module": mod_name,
            "entry": mod_meta["prepare_args_entry"],
            "params": prep_params,
            "grid": [1, 1, 1],
            "block": [128, 1, 1],
            "pdl": True,
            "args": prep_args,
        }
    ]

    def gemm_launch(vname: str, when: dict[str, Any] | None) -> dict[str, Any]:
        v = variants[vname]
        abi_size = int(v["abi"]["size"])
        pack = _build_gemm_pack(v, sm_count, is_nvfp4)
        entry_launch: dict[str, Any] = {
            "module": mod_name,
            "entry": v["entry"],
            "params": [f"bytes<{abi_size}>"],
            "grid": [sm_count, 1, 1],
            "block": [int(v["max_threads"]), 1, 1],
            "shared_mem": int(v["shared_mem"]),
            "pdl": True,
            "args": [{"pack": pack}],
        }
        if when is not None:
            entry_launch["when"] = when
        return entry_launch

    lowm_entry = "moe_nvfp4_gemm_lowm" if is_nvfp4 else "moe_mxfp8_gemm_lowm"
    lowm_launch: dict[str, Any] = {
        "module": mod_name,
        "entry": lowm_entry,
        "params": params,
        "grid": [sm_count * 4, 1, 1],
        "block": [128, 1, 1],
        "pdl": True,
        "args": [{"param": i} for i in range(len(params))],
    }

    if max_tokens <= 32:
        ops[op_name] = {
            "params": params,
            "impl": {"launches": [lowm_launch]},
        }
        return

    lowm_launch["when"] = {"var": token_axis, "max": 32}
    launches[0]["when"] = {"var": token_axis, "min": 33}
    launches.insert(0, lowm_launch)

    if is_nvfp4 and "n128" in variants:
        if max_tokens <= 256:
            launches.append(gemm_launch("n32", {"var": token_axis, "min": 33}))
        elif max_tokens <= 768:
            launches.append(gemm_launch("n32", {"var": token_axis, "min": 33, "max": 256}))
            launches.append(gemm_launch("n64", {"var": token_axis, "min": 257}))
        else:
            launches.append(gemm_launch("n32", {"var": token_axis, "min": 33, "max": 256}))
            launches.append(gemm_launch("n64", {"var": token_axis, "min": 257, "max": 768}))
            launches.append(gemm_launch("n128", {"var": token_axis, "min": 769}))
    else:
        if max_tokens <= 256:
            launches.append(gemm_launch("n32", {"var": token_axis, "min": 33}))
        else:
            launches.append(gemm_launch("n32", {"var": token_axis, "min": 33, "max": 256}))
            launches.append(gemm_launch("n64", {"var": token_axis, "min": 257}))

    ops[op_name] = {
        "params": params,
        "impl": {"scratch": scratch, "launches": launches},
    }


def register_ops(
    ops: dict[str, Any],
    modules_or_buffers: dict[str, Any],
    meta: dict[str, Any] | None = None,
    *,
    max_tokens: int = 1024,
    max_seqs: int = 8,
    sm_count: int | None = None,
    target_sm: dict[str, Any] | None = None,
    token_axis: str = "tokens",
    include_nvfp4: bool = True,
    include_mxfp8: bool = True,
) -> None:
    """Register all M2 MoE ops (routing, quant, SwiGLU, SM120 NVFP4/MXFP8 GEMMs, combine, derive).

    Notes for M5 `gen.py` integration:
      - Pass `target_sm={"sms": <sm_count>}` (e.g. `target_sm={"sms": 110}` on RTX PRO 5000
        Blackwell) or `sm_count=<sm_count>` so persistent CUTLASS grouped GEMMs size scratch
        and grid dimensions to the target GPU's SM count instead of `DEFAULT_SM_COUNT` (128).
      - Register `moe_out` (`bf16`, `[token_axis, 2304]`, `kind="workspace"`) when using the
        default `output_buf="moe_out"` in `emit_moe_layer_calls`.
    """
    _ = (modules_or_buffers, max_seqs)
    if meta is None:
        meta = load_meta(Path(__file__).resolve().parents[1] / "build" / "sm_120")
    mods = meta["modules"] if "modules" in meta else meta
    if sm_count is None:
        sm_count = int(target_sm["sms"]) if target_sm and "sms" in target_sm else DEFAULT_SM_COUNT

    p = lambda idx: {"param": idx}
    n_expr = _mul(token_axis, TOP_K)

    # 1. Bit-exact MoE Router (cuBLAS BF16 GEMM + fused single-CTA topk/metadata for T <= 32,
    #    static-SMEM metadata_t64 for 33..64, standard metadata for >= 65)
    if "moe_route_fused" not in ops:
        rf_params = [
            "in buffer<bf16>",   # 0: x
            "in buffer<bf16>",   # 1: gate_weight
            "in buffer<f32>",    # 2: gate_bias
            "out buffer<f32>",   # 3: logits
            "out buffer<i32>",   # 4: ids
            "out buffer<f32>",   # 5: weights
            "out buffer<i32>",   # 6: sorted_ids
            "out buffer<i32>",   # 7: src2dst
            "out buffer<i32>",   # 8: dst2src
            "out buffer<f32>",   # 9: gammas
            "out buffer<i32>",   # 10: counts
            "out buffer<i32>",   # 11: m_indptr
            "i32",               # 12: T
            "i32",               # 13: n = T * 8
            "i32",               # 14: E = 128
            "i32",               # 15: H = 2304
        ]
        rf_scratch: dict[str, Any] = {}
        tm_params = [
            "in buffer<f32>",
            "in buffer<f32>",
            "out buffer<i32>",
            "out buffer<f32>",
            "out buffer<i32>",
            "out buffer<i32>",
            "out buffer<i32>",
            "out buffer<f32>",
            "out buffer<i32>",
            "out buffer<i32>",
            "i32",
        ]
        tm_args = [p(3), p(2), p(4), p(5), p(6), p(7), p(8), p(9), p(10), p(11), p(12)]
        rf_launches: list[dict[str, Any]] = [
            {
                "entry": "extern:cublas_bf16_tn_f32",
                "params": ["in buffer<bf16>", "in buffer<bf16>", "out buffer<f32>", "i32", "i32", "i32"],
                "args": [p(0), p(1), p(3), p(12), p(14), p(15)],
            }
        ]

        t8_launch: dict[str, Any] = {
            "module": "moe_route",
            "entry": "moe_router_topk_meta_t8",
            "params": tm_params,
            "grid": [1, 1, 1],
            "block": [256, 1, 1],
            "pdl": True,
            "args": tm_args,
        }
        if max_tokens > 8:
            t8_launch["when"] = {"var": token_axis, "max": 8}
        rf_launches.append(t8_launch)

        if max_tokens >= 9:
            meta_params_rf = [
                "in buffer<i32>",
                "in buffer<f32>",
                "out buffer<i32>",
                "out buffer<i32>",
                "out buffer<i32>",
                "out buffer<f32>",
                "out buffer<i32>",
                "out buffer<i32>",
                "i32",
            ]
            topk_params_rf = ["in buffer<f32>", "in buffer<f32>", "out buffer<i32>", "out buffer<f32>", "i32"]
            topk_1w_when: dict[str, Any] = {"var": token_axis, "min": 9}
            if max_tokens > 128:
                topk_1w_when["max"] = 128
            rf_launches.append(
                {
                    "module": "moe_route",
                    "entry": "moe_router_topk_1w",
                    "params": topk_params_rf,
                    "grid": [token_axis, 1, 1],
                    "block": [32, 1, 1],
                    "pdl": True,
                    "when": topk_1w_when,
                    "args": [p(3), p(2), p(4), p(5), p(12)],
                }
            )
            if max_tokens > 128:
                rf_launches.append(
                    {
                        "module": "moe_route",
                        "entry": "moe_router_topk",
                        "params": topk_params_rf,
                        "grid": [_ceil_div(token_axis, 8), 1, 1],
                        "block": [256, 1, 1],
                        "pdl": True,
                        "when": {"var": token_axis, "min": 129},
                        "args": [p(3), p(2), p(4), p(5), p(12)],
                    }
                )
            t32_when_rf: dict[str, Any] = {"var": token_axis, "min": 9}
            if max_tokens > 32:
                t32_when_rf["max"] = 32
            rf_launches.append(
                {
                    "module": "moe_route",
                    "entry": "moe_route_metadata_t32",
                    "params": meta_params_rf,
                    "grid": [1, 1, 1],
                    "block": [256, 1, 1],
                    "pdl": True,
                    "when": t32_when_rf,
                    "args": [p(4), p(5), p(6), p(7), p(8), p(9), p(10), p(11), p(13)],
                }
            )
            if max_tokens > 32:
                t64_when_rf: dict[str, Any] = {"var": token_axis, "min": 33}
                if max_tokens > 64:
                    t64_when_rf["max"] = 64
                rf_launches.append(
                    {
                        "module": "moe_route",
                        "entry": "moe_route_metadata_t64",
                        "params": meta_params_rf,
                        "grid": [1, 1, 1],
                        "block": [512, 1, 1],
                        "pdl": True,
                        "when": t64_when_rf,
                        "args": [p(4), p(5), p(6), p(7), p(8), p(9), p(10), p(11), p(13)],
                    }
                )
            if max_tokens > 64:
                small_smem_rf = _add(_mul(_ceil_div(_clamp(token_axis, 0, 1024), 4), 256), 1024)
                small_when_rf: dict[str, Any] = {"var": token_axis, "min": 65}
                if max_tokens > 1024:
                    small_when_rf["max"] = 1024
                rf_launches.append(
                    {
                        "module": "moe_route",
                        "entry": "moe_route_metadata_small",
                        "params": meta_params_rf,
                        "grid": [1, 1, 1],
                        "block": [1024, 1, 1],
                        "shared_mem": small_smem_rf,
                        "pdl": True,
                        "when": small_when_rf,
                        "args": [p(4), p(5), p(6), p(7), p(8), p(9), p(10), p(11), p(13)],
                    }
                )
            if max_tokens > 1024:
                max_blocks_rf = ceil_div(max_tokens * TOP_K, 256)
                rf_scratch["block_counts"] = {"dtype": "i32", "shape": [max_blocks_rf, NUM_EXPERTS]}
                rf_scratch["block_offsets"] = {"dtype": "i32", "shape": [max_blocks_rf, NUM_EXPERTS]}
                num_blocks_rf = _ceil_div(n_expr, 256)
                rf_launches.extend(
                    [
                        {
                            "module": "moe_route",
                            "entry": "moe_route_count",
                            "params": ["in buffer<i32>", "out buffer<i32>", "i32"],
                            "grid": [num_blocks_rf, 1, 1],
                            "block": [256, 1, 1],
                            "pdl": True,
                            "when": {"var": token_axis, "min": 1025},
                            "args": [p(4), {"scratch": "block_counts"}, p(13)],
                        },
                        {
                            "module": "moe_route",
                            "entry": "moe_route_scan",
                            "params": [
                                "in buffer<i32>",
                                "out buffer<i32>",
                                "out buffer<i32>",
                                "out buffer<i32>",
                                "i32",
                            ],
                            "grid": [1, 1, 1],
                            "block": [128, 1, 1],
                            "pdl": True,
                            "when": {"var": token_axis, "min": 1025},
                            "args": [
                                {"scratch": "block_counts"},
                                {"scratch": "block_offsets"},
                                p(10),
                                p(11),
                                p(13),
                            ],
                        },
                        {
                            "module": "moe_route",
                            "entry": "moe_route_scatter",
                            "params": [
                                "in buffer<i32>",
                                "in buffer<f32>",
                                "in buffer<i32>",
                                "out buffer<i32>",
                                "out buffer<i32>",
                                "out buffer<i32>",
                                "out buffer<f32>",
                                "i32",
                            ],
                            "grid": [num_blocks_rf, 1, 1],
                            "block": [256, 1, 1],
                            "pdl": True,
                            "when": {"var": token_axis, "min": 1025},
                            "args": [
                                p(4),
                                p(5),
                                {"scratch": "block_offsets"},
                                p(6),
                                p(7),
                                p(8),
                                p(9),
                                p(13),
                            ],
                        },
                    ]
                )

        rf_impl: dict[str, Any] = {"launches": rf_launches}
        if rf_scratch:
            rf_impl["scratch"] = rf_scratch
        ops["moe_route_fused"] = {
            "params": rf_params,
            "impl": rf_impl,
        }

    # 4. Derive weight-scale swizzles (NVFP4 fp8e4m3 and MXFP8 u8)
    derive_swizzle_specs = []
    if include_nvfp4:
        derive_swizzle_specs.append(("moe_swizzle_weight_scales_fp8", "fp8e4m3"))
    if include_mxfp8:
        derive_swizzle_specs.append(("moe_swizzle_weight_scales_u8", "u8"))
    for op_name, dt in derive_swizzle_specs:
        if op_name not in ops:
            swiz_params = [f"in buffer<{dt}>", f"out buffer<{dt}>", "i32", "i32", "i32"]
            # Grid sized per call via static max? Note derive args are static constants;
            # worst-case total elements = 128 * 2304 * 144 = 42,467,328 -> ceil_div(total, 256) = 165888.
            # Kernel checks `idx < experts * n * scale_k`, so launching 165888 blocks is valid for all 4 shapes.
            ops[op_name] = {
                "params": swiz_params,
                "impl": {
                    "launches": [
                        {
                            "module": "moe_swiglu_quant",
                            "entry": "moe_swizzle_weight_scales",
                            "params": swiz_params,
                            "grid": [ceil_div(NUM_EXPERTS * 2 * INTER_SIZE * (HIDDEN_SIZE // NVFP4_GROUP), 256), 1, 1],
                            "block": [256, 1, 1],
                            "args": [p(i) for i in range(5)],
                        }
                    ]
                },
            }

    # 5. Quantization & Softcapped SwiGLU kernels
    row_grid = [_clamp(n_expr, 0, 2048), 1, 1]

    def simple_op(name: str, mod: str, entry: str, params: list[str], grid: list[Any], block: int) -> None:
        if name in ops:
            return
        ops[name] = {
            "params": params,
            "impl": {
                "launches": [
                    {
                        "module": mod,
                        "entry": entry,
                        "params": params,
                        "grid": grid,
                        "block": [block, 1, 1],
                        "pdl": True,
                        "args": [p(i) for i in range(len(params))],
                    }
                ]
            },
        }

    def split_lowm_op(
        name: str,
        mod: str,
        lowm_entry: str,
        highm_entry: str,
        params: list[str],
        lowm_grid: list[Any],
        lowm_block: int,
        highm_grid: list[Any],
        highm_block: int,
        cutoff: int,
    ) -> None:
        if name in ops:
            return
        low_l: dict[str, Any] = {
            "module": mod,
            "entry": lowm_entry,
            "params": params,
            "grid": lowm_grid,
            "block": [lowm_block, 1, 1],
            "pdl": True,
            "args": [p(i) for i in range(len(params))],
        }
        if max_tokens <= cutoff:
            ops[name] = {"params": params, "impl": {"launches": [low_l]}}
            return
        low_l["when"] = {"var": token_axis, "max": cutoff}
        high_l: dict[str, Any] = {
            "module": mod,
            "entry": highm_entry,
            "params": params,
            "grid": highm_grid,
            "block": [highm_block, 1, 1],
            "pdl": True,
            "when": {"var": token_axis, "min": cutoff + 1},
            "args": [p(i) for i in range(len(params))],
        }
        ops[name] = {"params": params, "impl": {"launches": [low_l, high_l]}}

    if include_nvfp4:
        split_lowm_op(
            "moe_quant_nvfp4",
            "moe_swiglu_quant",
            "moe_quant_nvfp4_v2_nozero",
            "moe_quant_nvfp4_v2",
            [
                "in buffer<bf16>",
                "in buffer<i32>",
                "in buffer<i32>",
                "in buffer<i32>",
                "out buffer<u8>",
                "out buffer<fp8e4m3>",
                "out buffer<f32>",
                "i32",
            ],
            [token_axis, 1, 1],
            576,
            row_grid,
            576,
            32,
        )
        split_lowm_op(
            "moe_swiglu_quant_nvfp4",
            "moe_swiglu_quant",
            "moe_swiglu_quant_nvfp4_bf16_nozero",
            "moe_swiglu_quant_nvfp4_bf16",
            [
                "in buffer<bf16>",
                "in buffer<f32>",
                "in buffer<i32>",
                "out buffer<u8>",
                "out buffer<fp8e4m3>",
                "out buffer<f32>",
                "i32",
            ],
            [n_expr, 1, 1],
            288,
            row_grid,
            288,
            32,
        )
    if include_mxfp8:
        split_lowm_op(
            "moe_quant_mxfp8",
            "moe_swiglu_quant",
            "moe_quant_mxfp8_v2_nozero",
            "moe_quant_mxfp8_v2",
            [
                "in buffer<bf16>",
                "in buffer<i32>",
                "in buffer<i32>",
                "in buffer<i32>",
                "out buffer<fp8e4m3>",
                "out buffer<u8>",
                "i32",
            ],
            [token_axis, 1, 1],
            160,
            row_grid,
            160,
            32,
        )
        split_lowm_op(
            "moe_swiglu_quant_mxfp8",
            "moe_swiglu_quant",
            "moe_swiglu_quant_mxfp8_nozero",
            "moe_swiglu_quant_mxfp8",
            [
                "in buffer<bf16>",
                "in buffer<f32>",
                "in buffer<i32>",
                "out buffer<fp8e4m3>",
                "out buffer<u8>",
                "i32",
            ],
            [n_expr, 1, 1],
            96,
            row_grid,
            96,
            32,
        )

    # 6. SM120 CUTLASS Grouped GEMMs (NVFP4 + MXFP8)
    if include_nvfp4:
        _register_gemm_op(
            ops, "moe_nvfp4_gemm", "moe_nvfp4_sm120", mods["moe_nvfp4_sm120"], True, max_tokens, sm_count, token_axis
        )
    if include_mxfp8:
        _register_gemm_op(
            ops, "moe_mxfp8_gemm", "moe_mxfp8_sm120", mods["moe_mxfp8_sm120"], False, max_tokens, sm_count, token_axis
        )

    # 7. Combine kernels (cached L2 loads for T <= 64, 4-wide streaming prefetch for T >= 65)
    if include_nvfp4 or include_mxfp8:
        split_lowm_op(
            "moe_combine_bf16",
            "moe_combine",
            "moe_combine_bf16",
            "moe_combine_bf16_stream",
            ["in buffer<bf16>", "in buffer<i32>", "out buffer<bf16>", "i32"],
            [token_axis, 1, 1],
            288,
            [token_axis, 1, 1],
            288,
            64,
        )


def emit_derive_calls(
    buffers: dict[str, Any],
    derive_calls: list[dict[str, Any]],
    *,
    num_layers: int = NUM_PHYSICAL_LAYERS,
    layers: range | list[int] | None = None,
    prefix: str = "model.llm.layers",
    max_tokens: int | None = None,
    include_workspaces: bool = False,
) -> dict[int, dict[str, str]]:
    """Register any missing weight/source/carry buffers and emit `derive` scale-swizzle calls.

    By default (`include_workspaces=False`, `max_tokens=None`), this registers only per-layer
    weight/source/carry buffers so calling `emit_derive_calls` before `register_buffers` never
    clamps shared workspaces to a default `max_tokens=1024`.
    """
    use_workspaces = include_workspaces or (max_tokens is not None)
    layer_weights = register_buffers(
        buffers,
        max_tokens=1024 if max_tokens is None else max_tokens,
        num_layers=num_layers,
        layers=layers,
        include_weights=True,
        include_workspaces=use_workspaces,
        prefix=prefix,
    )
    phys_range = range(num_layers) if layers is None else layers
    for phys_idx in phys_range:
        w = layer_weights[phys_idx]
        if phys_idx in NVFP4_LAYERS:
            derive_calls.append(
                {
                    "label": f"derive.moe.l{phys_idx}.w13_scale",
                    "op": "moe_swizzle_weight_scales_fp8",
                    "args": [
                        _b(w["w13_scale_src"]),
                        _b(w["w13_scale"]),
                        _i32(NUM_EXPERTS),
                        _i32(2 * INTER_SIZE),
                        _i32(HIDDEN_SIZE // NVFP4_GROUP),
                    ],
                }
            )
            derive_calls.append(
                {
                    "label": f"derive.moe.l{phys_idx}.w2_scale",
                    "op": "moe_swizzle_weight_scales_fp8",
                    "args": [
                        _b(w["w2_scale_src"]),
                        _b(w["w2_scale"]),
                        _i32(NUM_EXPERTS),
                        _i32(HIDDEN_SIZE),
                        _i32(INTER_SIZE // NVFP4_GROUP),
                    ],
                }
            )
        else:
            derive_calls.append(
                {
                    "label": f"derive.moe.l{phys_idx}.w13_scale",
                    "op": "moe_swizzle_weight_scales_u8",
                    "args": [
                        _b(w["w13_scale_src"]),
                        _b(w["w13_scale"]),
                        _i32(NUM_EXPERTS),
                        _i32(2 * INTER_SIZE),
                        _i32(HIDDEN_SIZE // MXFP8_GROUP),
                    ],
                }
            )
            derive_calls.append(
                {
                    "label": f"derive.moe.l{phys_idx}.w2_scale",
                    "op": "moe_swizzle_weight_scales_u8",
                    "args": [
                        _b(w["w2_scale_src"]),
                        _b(w["w2_scale"]),
                        _i32(NUM_EXPERTS),
                        _i32(HIDDEN_SIZE),
                        _i32(INTER_SIZE // MXFP8_GROUP),
                    ],
                }
            )
    return layer_weights


def emit_moe_layer_calls(
    mode: str,
    visit_idx: int,
    phys_idx: int,
    layer_weights: dict[int, dict[str, str]] | None = None,
    input_buf: str = "x_mlp",
    output_buf: str = "moe_out",
    tokens_var: str = "tokens",
) -> list[dict[str, Any]]:
    """Emit ordered MoE calls for logical `visit_idx` reading physical layer `phys_idx` weights."""
    _ = mode
    w = layer_weights[phys_idx] if layer_weights is not None else layer_weight_names(phys_idx)
    t_var = {"var": tokens_var}
    n_expr = _expr(_mul(tokens_var, TOP_K))
    pfx = f"v{visit_idx}.moe"

    calls: list[dict[str, Any]] = [
        {
            "label": f"{pfx}.route_fused",
            "op": "moe_route_fused",
            "args": [
                _b(input_buf),
                _b(w["gate_weight"]),
                _b(w["gate_bias"]),
                _b("moe.logits"),
                _b("moe.ids"),
                _b("moe.weights"),
                _b("moe.sorted_ids"),
                _b("moe.src2dst"),
                _b("moe.dst2src"),
                _b("moe.gammas"),
                _b("moe.counts"),
                _b("moe.m_indptr"),
                t_var,
                n_expr,
                _i32(NUM_EXPERTS),
                _i32(HIDDEN_SIZE),
            ],
        },
    ]

    if phys_idx in NVFP4_LAYERS:
        calls.extend(
            [
                {
                    "label": f"{pfx}.quant_nvfp4",
                    "op": "moe_quant_nvfp4",
                    "args": [
                        _b(input_buf),
                        _b("moe.ids"),
                        _b("moe.src2dst"),
                        _b("moe.m_indptr"),
                        _b("moe.xq_nvfp4"),
                        _b("moe.xsf_nvfp4"),
                        _b("moe.x_amax"),
                        t_var,
                    ],
                },
                {
                    "label": f"{pfx}.w13_nvfp4",
                    "op": "moe_nvfp4_gemm",
                    "args": [
                        _b("moe.xq_nvfp4"),
                        _b(w["w13_weight"]),
                        _b("moe.xsf_nvfp4"),
                        _b(w["w13_scale"]),
                        _b("moe.w13_out_bf16"),
                        _b("moe.m_indptr"),
                        _b("moe.x_amax"),
                        _b(w["w13_amax"]),
                        _b(w["w13_denom"]),
                        _i32(2 * INTER_SIZE),
                        _i32(HIDDEN_SIZE),
                        _i32(NUM_EXPERTS),
                    ],
                },
                {
                    "label": f"{pfx}.swiglu_quant_nvfp4",
                    "op": "moe_swiglu_quant_nvfp4",
                    "args": [
                        _b("moe.w13_out_bf16"),
                        _b("moe.gammas"),
                        _b("moe.m_indptr"),
                        _b("moe.act_nvfp4"),
                        _b("moe.act_sf_nvfp4"),
                        _b("moe.act_amax"),
                        n_expr,
                    ],
                },
                {
                    "label": f"{pfx}.w2_nvfp4",
                    "op": "moe_nvfp4_gemm",
                    "args": [
                        _b("moe.act_nvfp4"),
                        _b(w["w2_weight"]),
                        _b("moe.act_sf_nvfp4"),
                        _b(w["w2_scale"]),
                        _b("moe.w2_out_bf16"),
                        _b("moe.m_indptr"),
                        _b("moe.act_amax"),
                        _b(w["w2_amax"]),
                        _b(w["w2_denom"]),
                        _i32(HIDDEN_SIZE),
                        _i32(INTER_SIZE),
                        _i32(NUM_EXPERTS),
                    ],
                },
                {
                    "label": f"{pfx}.combine_nvfp4",
                    "op": "moe_combine_bf16",
                    "args": [
                        _b("moe.w2_out_bf16"),
                        _b("moe.src2dst"),
                        _b(output_buf),
                        t_var,
                    ],
                },
            ]
        )
    else:
        calls.extend(
            [
                {
                    "label": f"{pfx}.quant_mxfp8",
                    "op": "moe_quant_mxfp8",
                    "args": [
                        _b(input_buf),
                        _b("moe.ids"),
                        _b("moe.src2dst"),
                        _b("moe.m_indptr"),
                        _b("moe.xq_mxfp8"),
                        _b("moe.xsf_mxfp8"),
                        t_var,
                    ],
                },
                {
                    "label": f"{pfx}.w13_mxfp8",
                    "op": "moe_mxfp8_gemm",
                    "args": [
                        _b("moe.xq_mxfp8"),
                        _b(w["w13_weight"]),
                        _b("moe.xsf_mxfp8"),
                        _b(w["w13_scale"]),
                        _b("moe.w13_out_bf16"),
                        _b("moe.m_indptr"),
                        _i32(2 * INTER_SIZE),
                        _i32(HIDDEN_SIZE),
                        _i32(NUM_EXPERTS),
                    ],
                },
                {
                    "label": f"{pfx}.swiglu_quant_mxfp8",
                    "op": "moe_swiglu_quant_mxfp8",
                    "args": [
                        _b("moe.w13_out_bf16"),
                        _b("moe.gammas"),
                        _b("moe.m_indptr"),
                        _b("moe.act_mxfp8"),
                        _b("moe.act_sf_mxfp8"),
                        n_expr,
                    ],
                },
                {
                    "label": f"{pfx}.w2_mxfp8",
                    "op": "moe_mxfp8_gemm",
                    "args": [
                        _b("moe.act_mxfp8"),
                        _b(w["w2_weight"]),
                        _b("moe.act_sf_mxfp8"),
                        _b(w["w2_scale"]),
                        _b("moe.w2_out_bf16"),
                        _b("moe.m_indptr"),
                        _i32(HIDDEN_SIZE),
                        _i32(INTER_SIZE),
                        _i32(NUM_EXPERTS),
                    ],
                },
                {
                    "label": f"{pfx}.combine_bf16",
                    "op": "moe_combine_bf16",
                    "args": [
                        _b("moe.w2_out_bf16"),
                        _b("moe.src2dst"),
                        _b(output_buf),
                        t_var,
                    ],
                },
            ]
        )
    return calls


def build_unit_manifest(
    kernels_dir: Path,
    *,
    max_seqs: int = 8,
    max_tokens: int = 2048,
    sm_count: int = DEFAULT_SM_COUNT,
    layers: tuple[int, ...] = (0, 20),
) -> dict[str, Any]:
    """Build a standalone schema-v7 manifest exercising M2 derive + NVFP4/MXFP8 across all programs."""
    modules: dict[str, Any] = {}
    buffers: dict[str, Any] = {}
    ops: dict[str, Any] = {}
    derive_calls: list[dict[str, Any]] = []

    meta = register_modules(modules, kernels_dir)
    buffers["x_mlp"] = {"dtype": "bf16", "shape": ["tokens", HIDDEN_SIZE], "kind": "input"}
    buffers["moe_out"] = {"dtype": "bf16", "shape": ["tokens", HIDDEN_SIZE], "kind": "output"}
    register_buffers(
        buffers,
        max_seqs=max_seqs,
        max_tokens=max_tokens,
        layers=list(layers),
        include_weights=True,
    )
    layer_weights = emit_derive_calls(buffers, derive_calls, layers=list(layers))
    register_ops(ops, modules, meta, max_tokens=max_tokens, max_seqs=max_seqs, sm_count=sm_count)

    def program_calls(mode: str) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []
        for idx, phys in enumerate(layers):
            calls.extend(emit_moe_layer_calls(mode, idx, phys, layer_weights, "x_mlp", "moe_out"))
        return calls

    return {
        "schema_version": 7,
        "model": "inkling-2-mini-m2-unit",
        "vars": {
            "tokens": {"max": max_tokens, "axis": "rows"},
        },
        "states": {},
        "buffers": buffers,
        "modules": modules,
        "ops": ops,
        "programs": {
            "derive": {"derive": True, "calls": derive_calls},
            "prefill": {"batch": {"groups": 1, "rows": "tokens"}, "calls": program_calls("prefill")},
            "decode": {"batch": {"groups": 1, "rows": 1}, "graph": True, "calls": program_calls("decode")},
            "decode_batch": {
                "batch": {"groups": max_seqs, "rows": 1},
                "graph": True,
                "calls": program_calls("decode_batch"),
            },
            "round": {
                "batch": {"groups": max_seqs, "rows": 8},
                "graph": True,
                "calls": program_calls("round"),
            },
        },
    }
