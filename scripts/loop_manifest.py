#!/usr/bin/env python3
"""Manifest builder helpers for Inkling-2-Mini 48-visit loop, Block AttnRes,
Cross-Pass AttnRes, Rank-64 Loop Controllers, Embedding, and FP32 Unembed/Softcap/Argmax.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

H = 2304
RANK = 64
VOCAB = 201024
UNPADDED_VOCAB = 200058
SOFTCAP = 30.0
BLOCK_SIZE = 6
EPS = 1e-6
INV_SQRT_H = 1.0 / 48.0

NUM_PHYSICAL_LAYERS = 24
CORE_START = 6
CORE_END = 18
NUM_PASSES = 3
NUM_LOGICAL_VISITS = 48

FULL_LAYERS = frozenset({5, 11, 17, 23})
LOCAL_LAYERS = frozenset(i for i in range(NUM_PHYSICAL_LAYERS) if i not in FULL_LAYERS)
AUX_CAPTURE_VISITS = (5, 23, 41, 48)
CTRL_STREAMS = ("attn", "k", "v", "mlp")

b = lambda name, offset=0: {"buf": name, **({"offset": offset} if offset else {})}
i32 = lambda value: {"i32": value}
i64 = lambda value: {"i64": value}
f32 = lambda value: {"f32": value}
expr = lambda e: {"expr": e}
mul = lambda x, y: {"mul": [x, y]}
ceil_div = lambda x, d: {"ceil_div": [x, d]}


def visit_schedule() -> list[tuple[int, int, str, int]]:
    """Return 48 (visit_idx, phys_idx, phase, pass_idx) entries."""
    schedule: list[tuple[int, int, str, int]] = []
    v = 0
    for phys in range(0, CORE_START):
        schedule.append((v, phys, "prelude", 0))
        v += 1
    for p in range(NUM_PASSES):
        for phys in range(CORE_START, CORE_END):
            schedule.append((v, phys, "core", p))
            v += 1
    for phys in range(CORE_END, NUM_PHYSICAL_LAYERS):
        schedule.append((v, phys, "coda", 0))
        v += 1
    assert len(schedule) == NUM_LOGICAL_VISITS
    return schedule


def register_modules(modules: dict, kernels_dir: Path, names: tuple[str, ...] = ("norm_attnres", "loop_controller", "unembed_sample")) -> dict[str, str]:
    """SHA-256 pin M1 CUDA modules in `modules`."""
    pinned: dict[str, str] = {}
    for name in names:
        cubin = kernels_dir / f"{name}.cubin"
        sha = hashlib.sha256(cubin.read_bytes()).hexdigest()
        modules[name] = {"source": cubin.name, "sha256": sha}
        pinned[name] = sha
    return pinned


def register_buffers(
    buffers: dict,
    max_seqs: int = 8,
    max_tokens: int = 1024,
    *,
    include_aux: bool = True,
    token_axis: str = "tokens",
) -> None:
    """Register M1 weights, source/carry pre-scaled queries, workspaces, and logits/output buffers."""
    _ = (max_seqs, max_tokens)

    def buf(name: str, dtype: str, shape: list, kind: str = "workspace", **extra) -> None:
        if name not in buffers:
            buffers[name] = {"dtype": dtype, "shape": shape, "kind": kind, **extra}

    def weight(name: str, shape: list, tensors: list[str] | None = None, dtype: str = "bf16") -> None:
        bind = [{"tensor": t} for t in (tensors or [name])]
        buf(name, dtype, shape, "weight", bind=bind)

    # Embedding, norms, unembed
    weight("w.embed", [VOCAB, H], ["model.llm.embed.weight"])
    weight("w.embed_norm", [H], ["model.llm.embed_norm.weight"])
    weight("w.final_norm", [H], ["model.llm.norm.weight"])
    weight("w.unembed", [VOCAB, H], ["model.llm.unembed.weight"])

    # Raw FP32 AttnRes & cross-pass queries (source -> scaled by 1/48 into carry during derive)
    buf("src.output_q", "f32", [H], "source", bind=[{"tensor": "model.llm.attn_res.output_query"}])
    buf("c.output_q", "f32", [H], "carry")

    for i in range(NUM_PHYSICAL_LAYERS):
        if i > 0:
            q_shape = [3, H] if i in FULL_LAYERS else [H]
            buf(
                f"src.attn_q.{i - 1}",
                "f32",
                q_shape,
                "source",
                bind=[{"tensor": f"model.llm.attn_res.attn_queries.{i - 1}"}],
            )
            buf(f"c.attn_q.{i - 1}", "f32", q_shape, "carry")
        buf(
            f"src.mlp_q.{i}",
            "f32",
            [H],
            "source",
            bind=[{"tensor": f"model.llm.attn_res.mlp_queries.{i}"}],
        )
        buf(f"c.mlp_q.{i}", "f32", [H], "carry")

        if i in FULL_LAYERS:
            if CORE_START <= i < CORE_END:
                for nm in ("q_input_norm", "k_input_norm", "v_input_norm"):
                    weight(f"w.l{i}.{nm}", [H], [f"model.llm.layers.{i}.{nm}.weight"])
            weight(
                f"w.l{i}.qkv_input_norm",
                [3, H],
                [f"model.llm.layers.{i}.{nm}.weight" for nm in ("q_input_norm", "k_input_norm", "v_input_norm")],
            )
        else:
            weight(f"w.l{i}.attn_norm", [H], [f"model.llm.layers.{i}.attn_norm.weight"])
        weight(f"w.l{i}.mlp_norm", [H], [f"model.llm.layers.{i}.mlp_norm.weight"])

        if CORE_START <= i < CORE_END:
            for s in CTRL_STREAMS:
                pfx = f"model.llm.layers.{i}.controllers.{s}"
                weight(f"w.l{i}.ctrl.{s}.down", [RANK, 2 * H], [f"{pfx}.down_weight"])
                weight(
                    f"w.l{i}.ctrl.{s}.fg",
                    [2 * H, RANK],
                    [f"{pfx}.forget_weight", f"{pfx}.gain_weight"],
                )

    # Cross-pass queries packed in [12, 4, H] order: (attn, k, v, mlp) per core layer c in 0..11
    cross_bind = []
    for c in range(CORE_END - CORE_START):
        cross_bind.append({"tensor": "model.llm.middle_cycle_cross_pass_queries", "rows": [2 * c, 2 * c + 1]})
        cross_bind.append({"tensor": "model.llm.middle_cycle_cross_pass_kv_queries", "rows": [2 * c, 2 * c + 1]})
        cross_bind.append({"tensor": "model.llm.middle_cycle_cross_pass_kv_queries", "rows": [2 * c + 1, 2 * c + 2]})
        cross_bind.append({"tensor": "model.llm.middle_cycle_cross_pass_queries", "rows": [2 * c + 1, 2 * c + 2]})
    buf("src.cross_pass_q", "f32", [CORE_END - CORE_START, 4, H], "source", bind=cross_bind)
    buf("c.cross_pass_q", "f32", [CORE_END - CORE_START, 4, H], "carry")

    # Loop & sublayer interface workspaces
    buf("tmp_embed", "bf16", [token_axis, H])
    buf("partial", "bf16", [token_axis, H])
    buf("stack", "bf16", [4, token_axis, H])
    buf("prelude_partial", "bf16", [token_axis, H])
    buf("prev_blocks", "bf16", [2, token_axis, H])
    buf("mix_qkv", "f32", [3, token_axis, H])
    buf("mix_mlp", "f32", [token_axis, H])
    buf("msg_buf", "f32", [4, token_axis, H])
    buf("x_qkv", "bf16", [3, token_axis, H])
    buf("x_mlp", "bf16", [token_axis, H])
    buf("attn_out", "bf16", [token_axis, H])
    buf("mlp_out", "bf16", [token_axis, H])
    buf("final_h", "bf16", [token_axis, H])
    buf("last_h", "bf16", ["seqs", H])
    buf("logits", "f32", ["seqs", VOCAB], "output")
    buf("next_token", "i64", ["seqs"], "output", fill="tokens", domain={"min": 0, "max": VOCAB - 1})

    if include_aux:
        buf("aux_hidden", "bf16", [token_axis, 4, H])


def register_ops(
    ops: dict,
    modules: dict,
    build_meta: dict | None = None,
    max_seqs: int = 8,
    max_tokens: int = 1024,
    target_sm: dict | None = None,
    *,
    include_sample_copy: bool = True,
) -> None:
    """Register M1 operators (norm_attnres, loop_controller, unembed_sample)."""
    _ = (build_meta, max_seqs, max_tokens, target_sm)
    for mod in ("norm_attnres", "loop_controller", "unembed_sample"):
        if mod not in modules:
            raise KeyError(f"Module `{mod}` must be registered before register_ops()")

    def single_op(name: str, mod: str, entry: str, params: list[str], grid: list, block: int = 256, pdl: bool = False) -> None:
        if name in ops:
            return
        launch = {
            "module": mod,
            "entry": entry,
            "params": params,
            "grid": grid,
            "block": [block, 1, 1],
            "args": [{"param": j} for j in range(len(params))],
        }
        if pdl:
            launch["pdl"] = True
        ops[name] = {
            "params": params,
            "impl": {"launches": [launch]},
        }

    # Embedding & norms
    single_op(
        "embed_bf16",
        "unembed_sample",
        "kern_embedding_bf16",
        ["in buffer<i64>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32"],
        [{"param": 0}, 1, 1],  # overridden below: grid uses var or expr if needed
    )
    # Fix grid on embed_bf16: kern requires grid exprs over vars/consts, not {"param": ...}
    ops["embed_bf16"]["impl"]["launches"][0]["grid"] = ["tokens", 1, 1]

    single_op(
        "rms_norm_bf16",
        "norm_attnres",
        "kern_inkling2_rms_norm_bf16",
        ["in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32", "i32", "f32"],
        ["tokens", 1, 1],
    )

    # AttnRes (4 non-null-pointer variants + cross-pass): Block [288, 1, 1] (9 warps x 8 elems = 2304)
    single_op(
        "attn_res_norm_bf16",
        "norm_attnres",
        "kern_attn_res_norm_bf16",
        [
            "in buffer<bf16>",
            "in buffer<bf16>",
            "in buffer<f32>",
            "in buffer<bf16>",
            "out buffer<bf16>",
            "i64",
            "i64",
            "i32",
            "i32",
            "i32",
            "i32",
            "f32",
            "f32",
        ],
        ["tokens", 1, 1],
        pdl=True,
    )
    single_op(
        "attn_res_addend_norm_bf16",
        "norm_attnres",
        "kern_attn_res_addend_norm_bf16",
        [
            "in buffer<bf16>",
            "in buffer<bf16>",
            "in buffer<bf16>",
            "in buffer<f32>",
            "in buffer<bf16>",
            "out buffer<bf16>",
            "i64",
            "i64",
            "i32",
            "i32",
            "i32",
            "i32",
            "f32",
            "f32",
        ],
        ["tokens", 1, 1],
        pdl=True,
    )
    if include_sample_copy:
        single_op(
            "attn_res_mix_f32",
            "norm_attnres",
            "kern_attn_res_mix_f32",
            [
                "in buffer<bf16>",
                "in buffer<bf16>",
                "in buffer<f32>",
                "out buffer<f32>",
                "i64",
                "i64",
                "i32",
                "i32",
                "i32",
                "i32",
                "f32",
            ],
            ["tokens", 1, 1],
            pdl=True,
        )
        single_op(
            "attn_res_addend_mix_f32",
            "norm_attnres",
            "kern_attn_res_addend_mix_f32",
            [
                "in buffer<bf16>",
                "in buffer<bf16>",
                "in buffer<bf16>",
                "in buffer<f32>",
                "out buffer<f32>",
                "i64",
                "i64",
                "i32",
                "i32",
                "i32",
                "i32",
                "f32",
            ],
            ["tokens", 1, 1],
            pdl=True,
        )
        if "cross_pass_attn_res_f32" not in ops:
            cp_params = [
                "in buffer<bf16>",
                "in buffer<f32>",
                "out buffer<f32>",
                "i64",
                "i64",
                "i32",
                "i32",
                "i32",
                "f32",
            ]
            p_cp = lambda idx: {"param": idx}
            ops["cross_pass_attn_res_f32"] = {
                "params": cp_params,
                "impl": {
                    "launches": [
                        {
                            "module": "norm_attnres",
                            "entry": "kern_cross_pass_attn_res_4q_f32",
                            "params": [
                                "in buffer<bf16>",
                                "in buffer<f32>",
                                "out buffer<f32>",
                                "i64",
                                "i64",
                                "i32",
                                "f32",
                            ],
                            "grid": ["tokens", 4, 1],
                            "block": [256, 1, 1],
                            "pdl": True,
                            "args": [p_cp(0), p_cp(1), p_cp(2), p_cp(3), p_cp(4), p_cp(5), p_cp(8)],
                            "when": {"var": "tokens", "max": 64},
                        },
                        {
                            "module": "norm_attnres",
                            "entry": "kern_cross_pass_attn_res_f32",
                            "params": cp_params,
                            "grid": ["tokens", 1, 1],
                            "block": [256, 1, 1],
                            "pdl": True,
                            "args": [p_cp(j) for j in range(9)],
                            "when": {"var": "tokens", "min": 65},
                        },
                    ]
                },
            }

    # Helpers: query scaling (for derive), copies, taps, residual adds
    single_op(
        "scale_f32",
        "norm_attnres",
        "kern_scale_f32",
        ["in buffer<f32>", "out buffer<f32>", "i64", "f32"],
        [ceil_div(12 * 4 * H, 256), 1, 1],
    )
    def split_vec8_op(name: str, entry_small: str, entry_vec8: str, params: list[str]) -> None:
        if name in ops:
            return
        if max_tokens <= 64:
            single_op(name, "norm_attnres", entry_small, params, [ceil_div(mul("tokens", H), 256), 1, 1])
            return
        ops[name] = {
            "params": params,
            "impl": {
                "launches": [
                    {
                        "module": "norm_attnres",
                        "entry": entry_small,
                        "params": params,
                        "grid": [ceil_div(mul("tokens", H), 256), 1, 1],
                        "block": [256, 1, 1],
                        "args": [{"param": j} for j in range(len(params))],
                        "when": {"var": "tokens", "max": 64},
                    },
                    {
                        "module": "norm_attnres",
                        "entry": entry_vec8,
                        "params": params,
                        "grid": [ceil_div(mul("tokens", H), 2048), 1, 1],
                        "block": [256, 1, 1],
                        "args": [{"param": j} for j in range(len(params))],
                        "when": {"var": "tokens", "min": 65},
                    },
                ]
            },
        }

    split_vec8_op("copy_bf16", "kern_copy_bf16", "kern_copy_bf16_vec8", ["in buffer<bf16>", "out buffer<bf16>", "i64"])
    single_op(
        "tap_bf16",
        "norm_attnres",
        "kern_tap_bf16",
        ["in buffer<bf16>", "out buffer<bf16>", "i32", "i32", "i32", "i32"],
        ["tokens", 1, 1],
    )
    split_vec8_op(
        "add2_bf16",
        "kern_residual_add2_bf16",
        "kern_residual_add2_bf16_vec8",
        ["in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i64"],
    )
    split_vec8_op(
        "add3_bf16",
        "kern_residual_add3_bf16",
        "kern_residual_add3_bf16_vec8",
        ["in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i64"],
    )

    # Rank-64 gated loop controller: standalone op for unit tests; fused ops for full model.
    if include_sample_copy and "loop_controller" not in ops:
        ctrl_params = [
            "in buffer<f32>",
            "in buffer<f32>",
            "in buffer<bf16>",
            "in buffer<bf16>",
            "in buffer<bf16>",
            "out buffer<bf16>",
            "i32",
            "f32",
        ]
        p = lambda idx: {"param": idx}
        scr = lambda name: {"scratch": name}
        ops["loop_controller"] = {
            "params": ctrl_params,
            "impl": {
                "scratch": {
                    "cat": {"dtype": "bf16", "shape": ["tokens", 2 * H]},
                    "h": {"dtype": "bf16", "shape": ["tokens", RANK]},
                    "fg": {"dtype": "f32", "shape": ["tokens", 2 * H]},
                },
                "launches": [
                    {
                        "module": "loop_controller",
                        "entry": "kern_ctrl_rms_concat_2stream_f32_bf16",
                        "params": ["in buffer<f32>", "in buffer<f32>", "out buffer<bf16>", "i32", "f32"],
                        "grid": ["tokens", 2, 1],
                        "block": [256, 1, 1],
                        "pdl": True,
                        "args": [p(0), p(1), scr("cat"), p(6), p(7)],
                        "when": {"var": "tokens", "max": 64},
                    },
                    {
                        "module": "loop_controller",
                        "entry": "kern_ctrl_rms_concat_f32_bf16",
                        "params": ["in buffer<f32>", "in buffer<f32>", "out buffer<bf16>", "i32", "i32", "f32"],
                        "grid": ["tokens", 1, 1],
                        "block": [256, 1, 1],
                        "pdl": True,
                        "args": [p(0), p(1), scr("cat"), p(6), i32(H), p(7)],
                        "when": {"var": "tokens", "min": 65},
                    },
                    {
                        "entry": "extern:cublaslt_bf16_tn",
                        "params": ["in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32", "i32"],
                        "args": [scr("cat"), p(2), scr("h"), p(6), i32(RANK), i32(2 * H)],
                    },
                    {
                        "entry": "extern:cublas_bf16_tn_f32",
                        "params": ["in buffer<bf16>", "in buffer<bf16>", "out buffer<f32>", "i32", "i32", "i32"],
                        "args": [scr("h"), p(3), scr("fg"), p(6), i32(2 * H), i32(RANK)],
                    },
                    {
                        "module": "loop_controller",
                        "entry": "kern_ctrl_merge_rmsnorm_f32_bf16",
                        "params": [
                            "in buffer<f32>",
                            "in buffer<f32>",
                            "in buffer<f32>",
                            "in buffer<bf16>",
                            "out buffer<bf16>",
                            "i32",
                            "i32",
                            "f32",
                        ],
                        "grid": ["tokens", 1, 1],
                        "block": [256, 1, 1],
                        "pdl": True,
                        "args": [p(0), p(1), scr("fg"), p(4), p(5), p(6), i32(H), p(7)],
                    },
                ],
            },
        }

    # Production fused AttnRes + Cross-Pass + Loop Controller ops (100% bit-identical):
    if not include_sample_copy:
        # Low-register high-occupancy AttnRes + RMSNorm (1q and 3q-split for round)
        ar1q_params = [
            "in buffer<bf16>",
            "in buffer<bf16>",
            "in buffer<f32>",
            "in buffer<bf16>",
            "out buffer<bf16>",
            "i64",
            "i64",
            "i32",
            "i32",
            "f32",
            "f32",
        ]
        single_op(
            "attn_res_norm_1q_bf16",
            "norm_attnres",
            "kern_attn_res_norm_1q_bf16",
            ar1q_params,
            ["tokens", 1, 1],
            pdl=True,
        )
        if "attn_res_norm_3q_bf16" not in ops:
            p3q = lambda idx: {"param": idx}
            ops["attn_res_norm_3q_bf16"] = {
                "params": ar1q_params,
                "impl": {
                    "launches": [
                        {
                            "module": "norm_attnres",
                            "entry": "kern_attn_res_norm_1q_bf16",
                            "params": ar1q_params,
                            "grid": ["tokens", 3, 1],
                            "block": [256, 1, 1],
                            "pdl": True,
                            "args": [p3q(j) for j in range(11)],
                            "when": {"var": "tokens", "max": 64},
                        },
                        {
                            "module": "norm_attnres",
                            "entry": "kern_attn_res_norm_bf16",
                            "params": [
                                "in buffer<bf16>",
                                "in buffer<bf16>",
                                "in buffer<f32>",
                                "in buffer<bf16>",
                                "out buffer<bf16>",
                                "i64",
                                "i64",
                                "i32",
                                "i32",
                                "i32",
                                "i32",
                                "f32",
                                "f32",
                            ],
                            "grid": ["tokens", 1, 1],
                            "block": [256, 1, 1],
                            "pdl": True,
                            "args": [
                                p3q(0),
                                p3q(1),
                                p3q(2),
                                p3q(3),
                                p3q(4),
                                p3q(5),
                                p3q(6),
                                p3q(7),
                                i32(H),
                                p3q(8),
                                i32(3),
                                p3q(9),
                                p3q(10),
                            ],
                            "when": {"var": "tokens", "min": 65},
                        },
                    ]
                },
            }

        # Fused Pre-Attention: attn_res_mix + cross_pass_attn_res + 3x loop_controller (s1..s4)
        for op_name, lowm_grid_y, num_q_val in (
            ("pre_attn_mix_cross_ctrl_swa", 5, 1),
            ("pre_attn_mix_cross_ctrl_full", 7, 3),
        ):
            if op_name not in ops:
                pa_params = [
                    "in buffer<bf16>",   # 0: stack
                    "in buffer<bf16>",   # 1: partial
                    "in buffer<f32>",    # 2: attn_q
                    "in buffer<bf16>",   # 3: prev_blocks
                    "in buffer<f32>",    # 4: cross_q
                    "in buffer<bf16>",   # 5: w_down_0
                    "in buffer<bf16>",   # 6: w_fg_0
                    "in buffer<bf16>",   # 7: w_down_1
                    "in buffer<bf16>",   # 8: w_fg_1
                    "in buffer<bf16>",   # 9: w_down_2
                    "in buffer<bf16>",   # 10: w_fg_2
                    "in buffer<bf16>",   # 11: norm_w
                    "out buffer<f32>",   # 12: mix_qkv
                    "out buffer<f32>",   # 13: msg_buf
                    "out buffer<bf16>",  # 14: x_qkv
                    "i64",               # 15: stack_stride (max_tokens*H)
                    "i64",               # 16: out_q_stride (max_tokens*H)
                    "i64",               # 17: in_stride (0 SWA, max_tokens*H Full)
                    "i64",               # 18: norm_stride (0 SWA, H Full)
                    "i32",               # 19: T
                    "i32",               # 20: n_stack
                    "f32",               # 21: eps
                ]
                pa = lambda idx: {"param": idx}
                ps = lambda name: {"scratch": name}
                s1_params = [
                    "in buffer<bf16>",
                    "in buffer<bf16>",
                    "in buffer<f32>",
                    "in buffer<bf16>",
                    "in buffer<f32>",
                    "out buffer<f32>",
                    "out buffer<f32>",
                    "out buffer<bf16>",
                    "out buffer<bf16>",
                    "out buffer<bf16>",
                    "i64",
                    "i64",
                    "i64",
                    "i32",
                    "i32",
                    "f32",
                ]
                s1_args = [
                    pa(0), pa(1), pa(2), pa(3), pa(4),
                    pa(12), pa(13), ps("cat0"), ps("cat1"), ps("cat2"),
                    pa(15), pa(16), pa(17), pa(19), pa(20), pa(21),
                ]
                s1_stream_params = [
                    "in buffer<f32>",
                    "in buffer<f32>",
                    "out buffer<bf16>",
                    "i64",
                    "i64",
                    "i32",
                    "i32",
                    "f32",
                ]
                s4_stream_params = [
                    "in buffer<f32>",
                    "in buffer<f32>",
                    "in buffer<f32>",
                    "in buffer<bf16>",
                    "out buffer<bf16>",
                    "i64",
                    "i64",
                    "i64",
                    "i64",
                    "i32",
                    "i32",
                    "f32",
                ]
                mid_when = {"var": "tokens", "min": 65, "max": 768}
                big_when = {"var": "tokens", "min": 769}
                pa_launches: list[dict[str, Any]] = [
                    # --- Low-M (T <= 64, round/decode): 1x fused s1 + 3x s2 + 3x s3 + 1x 3-stream s4 ---
                    {
                        "module": "norm_attnres",
                        "entry": "kern_pre_attn_mix_cross_s1_lowm",
                        "params": s1_params,
                        "grid": ["tokens", lowm_grid_y, 1],
                        "block": [256, 1, 1],
                        "pdl": True,
                        "args": s1_args,
                        "when": {"var": "tokens", "max": 64},
                    },
                    {
                        "entry": "extern:cublaslt_bf16_tn",
                        "params": ["in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32", "i32"],
                        "args": [ps("cat0"), pa(5), ps("h0"), pa(19), i32(RANK), i32(2 * H)],
                        "when": {"var": "tokens", "max": 64},
                    },
                    {
                        "entry": "extern:cublaslt_bf16_tn",
                        "params": ["in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32", "i32"],
                        "args": [ps("cat1"), pa(7), ps("h1"), pa(19), i32(RANK), i32(2 * H)],
                        "when": {"var": "tokens", "max": 64},
                    },
                    {
                        "entry": "extern:cublaslt_bf16_tn",
                        "params": ["in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32", "i32"],
                        "args": [ps("cat2"), pa(9), ps("h2"), pa(19), i32(RANK), i32(2 * H)],
                        "when": {"var": "tokens", "max": 64},
                    },
                    {
                        "entry": "extern:cublas_bf16_tn_f32",
                        "params": ["in buffer<bf16>", "in buffer<bf16>", "out buffer<f32>", "i32", "i32", "i32"],
                        "args": [ps("h0"), pa(6), ps("fg0"), pa(19), i32(2 * H), i32(RANK)],
                        "when": {"var": "tokens", "max": 64},
                    },
                    {
                        "entry": "extern:cublas_bf16_tn_f32",
                        "params": ["in buffer<bf16>", "in buffer<bf16>", "out buffer<f32>", "i32", "i32", "i32"],
                        "args": [ps("h1"), pa(8), ps("fg1"), pa(19), i32(2 * H), i32(RANK)],
                        "when": {"var": "tokens", "max": 64},
                    },
                    {
                        "entry": "extern:cublas_bf16_tn_f32",
                        "params": ["in buffer<bf16>", "in buffer<bf16>", "out buffer<f32>", "i32", "i32", "i32"],
                        "args": [ps("h2"), pa(10), ps("fg2"), pa(19), i32(2 * H), i32(RANK)],
                        "when": {"var": "tokens", "max": 64},
                    },
                    {
                        "module": "loop_controller",
                        "entry": "kern_ctrl_merge_rmsnorm_3x_f32_bf16",
                        "params": [
                            "in buffer<f32>",
                            "in buffer<f32>",
                            "in buffer<f32>",
                            "in buffer<f32>",
                            "in buffer<f32>",
                            "in buffer<bf16>",
                            "out buffer<bf16>",
                            "i64",
                            "i64",
                            "i64",
                            "i64",
                            "i32",
                            "f32",
                        ],
                        "grid": ["tokens", 3, 1],
                        "block": [256, 1, 1],
                        "pdl": True,
                        "args": [
                            pa(12), pa(13), ps("fg0"), ps("fg1"), ps("fg2"),
                            pa(11), pa(14), pa(17), pa(16), pa(16), pa(18), pa(19), pa(21),
                        ],
                        "when": {"var": "tokens", "max": 64},
                    },
                    # --- Mid Prefill (65 <= T <= 768, fits in 96 MiB L2): 1x fused s1 + 3x (s2->s3->s4) ---
                    {
                        "module": "norm_attnres",
                        "entry": "kern_pre_attn_mix_cross_s1_prefill",
                        "params": s1_params,
                        "grid": ["tokens", 2, 1],
                        "block": [256, 1, 1],
                        "pdl": True,
                        "args": s1_args,
                        "when": mid_when,
                    },
                ]
                for s_idx, (cat_s, h_s, wd_p, wfg_p) in enumerate((("cat0", "h0", 5, 6), ("cat1", "h1", 7, 8), ("cat2", "h2", 9, 10))):
                    pa_launches.extend([
                        {
                            "entry": "extern:cublaslt_bf16_tn",
                            "params": ["in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32", "i32"],
                            "args": [ps(cat_s), pa(wd_p), ps(h_s), pa(19), i32(RANK), i32(2 * H)],
                            "when": mid_when,
                        },
                        {
                            "entry": "extern:cublas_bf16_tn_f32",
                            "params": ["in buffer<bf16>", "in buffer<bf16>", "out buffer<f32>", "i32", "i32", "i32"],
                            "args": [ps(h_s), pa(wfg_p), ps("fg0"), pa(19), i32(2 * H), i32(RANK)],
                            "when": mid_when,
                        },
                        {
                            "module": "loop_controller",
                            "entry": "kern_ctrl_merge_rmsnorm_stream_f32_bf16",
                            "params": s4_stream_params,
                            "grid": ["tokens", 1, 1],
                            "block": [256, 1, 1],
                            "pdl": True,
                            "args": [pa(12), pa(13), ps("fg0"), pa(11), pa(14), pa(17), pa(16), pa(16), pa(18), i32(s_idx), pa(19), pa(21)],
                            "when": mid_when,
                        },
                    ])
                # --- Large Prefill (T >= 769, e.g. c=2048): L2-friendly sequential cat0/h0/fg0 reuse ---
                pa_launches.extend([
                    {
                        "module": "norm_attnres",
                        "entry": "kern_attn_res_mix_f32",
                        "params": [
                            "in buffer<bf16>",
                            "in buffer<bf16>",
                            "in buffer<f32>",
                            "out buffer<f32>",
                            "i64",
                            "i64",
                            "i32",
                            "i32",
                            "i32",
                            "i32",
                            "f32",
                        ],
                        "grid": ["tokens", 1, 1],
                        "block": [256, 1, 1],
                        "pdl": True,
                        "args": [pa(0), pa(1), pa(2), pa(12), pa(15), pa(16), pa(19), i32(H), pa(20), i32(num_q_val), pa(21)],
                        "when": big_when,
                    },
                    {
                        "module": "norm_attnres",
                        "entry": "kern_cross_pass_attn_res_f32",
                        "params": [
                            "in buffer<bf16>",
                            "in buffer<f32>",
                            "out buffer<f32>",
                            "i64",
                            "i64",
                            "i32",
                            "i32",
                            "i32",
                            "f32",
                        ],
                        "grid": ["tokens", 1, 1],
                        "block": [256, 1, 1],
                        "pdl": True,
                        "args": [pa(3), pa(4), pa(13), pa(15), pa(16), pa(19), i32(H), i32(4), pa(21)],
                        "when": big_when,
                    },
                ])
                for s_idx, (wd_p, wfg_p) in enumerate(((5, 6), (7, 8), (9, 10))):
                    pa_launches.extend([
                        {
                            "module": "loop_controller",
                            "entry": "kern_ctrl_rms_concat_stream_f32_bf16",
                            "params": s1_stream_params,
                            "grid": ["tokens", 1, 1],
                            "block": [256, 1, 1],
                            "pdl": True,
                            "args": [pa(12), pa(13), ps("cat0"), pa(17), pa(16), i32(s_idx), pa(19), pa(21)],
                            "when": big_when,
                        },
                        {
                            "entry": "extern:cublaslt_bf16_tn",
                            "params": ["in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32", "i32"],
                            "args": [ps("cat0"), pa(wd_p), ps("h0"), pa(19), i32(RANK), i32(2 * H)],
                            "when": big_when,
                        },
                        {
                            "entry": "extern:cublas_bf16_tn_f32",
                            "params": ["in buffer<bf16>", "in buffer<bf16>", "out buffer<f32>", "i32", "i32", "i32"],
                            "args": [ps("h0"), pa(wfg_p), ps("fg0"), pa(19), i32(2 * H), i32(RANK)],
                            "when": big_when,
                        },
                        {
                            "module": "loop_controller",
                            "entry": "kern_ctrl_merge_rmsnorm_stream_f32_bf16",
                            "params": s4_stream_params,
                            "grid": ["tokens", 1, 1],
                            "block": [256, 1, 1],
                            "pdl": True,
                            "args": [pa(12), pa(13), ps("fg0"), pa(11), pa(14), pa(17), pa(16), pa(16), pa(18), i32(s_idx), pa(19), pa(21)],
                            "when": big_when,
                        },
                    ])
                ops[op_name] = {
                    "params": pa_params,
                    "impl": {
                        "scratch": {
                            "cat0": {"dtype": "bf16", "shape": ["tokens", 2 * H]},
                            "cat1": {"dtype": "bf16", "shape": [768, 2 * H]},
                            "cat2": {"dtype": "bf16", "shape": [768, 2 * H]},
                            "h0": {"dtype": "bf16", "shape": ["tokens", RANK]},
                            "h1": {"dtype": "bf16", "shape": [768, RANK]},
                            "h2": {"dtype": "bf16", "shape": [768, RANK]},
                            "fg0": {"dtype": "f32", "shape": ["tokens", 2 * H]},
                            "fg1": {"dtype": "f32", "shape": [64, 2 * H]},
                            "fg2": {"dtype": "f32", "shape": [64, 2 * H]},
                        },
                        "launches": pa_launches,
                    },
                }

        # Fused Pre-MLP: attn_res_{mix,addend_mix} + loop_controller (s1..s4) for ALL T
        if "pre_mlp_mix_ctrl" not in ops:
            pm = lambda idx: {"param": idx}
            pms = lambda name: {"scratch": name}
            ops["pre_mlp_mix_ctrl"] = {
                "params": [
                    "in buffer<bf16>",   # 0: stack
                    "in buffer<bf16>",   # 1: partial (attn_out)
                    "in buffer<f32>",    # 2: mlp_q
                    "in buffer<f32>",    # 3: msg_f32 (msg_buf[3])
                    "in buffer<bf16>",   # 4: w_down
                    "in buffer<bf16>",   # 5: w_fg
                    "in buffer<bf16>",   # 6: norm_w
                    "out buffer<f32>",   # 7: mix_mlp
                    "out buffer<bf16>",  # 8: x_mlp
                    "i64",               # 9: stack_stride
                    "i32",               # 10: T
                    "i32",               # 11: n_stack
                    "f32",               # 12: eps
                ],
                "impl": {
                    "scratch": {
                        "cat": {"dtype": "bf16", "shape": ["tokens", 2 * H]},
                        "h": {"dtype": "bf16", "shape": ["tokens", RANK]},
                        "fg": {"dtype": "f32", "shape": ["tokens", 2 * H]},
                    },
                    "launches": [
                        {
                            "module": "norm_attnres",
                            "entry": "kern_pre_mlp_mix_s1_f32_bf16",
                            "params": [
                                "in buffer<bf16>",
                                "in buffer<bf16>",
                                "in buffer<f32>",
                                "in buffer<f32>",
                                "out buffer<f32>",
                                "out buffer<bf16>",
                                "i64",
                                "i32",
                                "i32",
                                "f32",
                            ],
                            "grid": ["tokens", 2, 1],
                            "block": [256, 1, 1],
                            "pdl": True,
                            "args": [pm(0), pm(1), pm(2), pm(3), pm(7), pms("cat"), pm(9), pm(10), pm(11), pm(12)],
                        },
                        {
                            "entry": "extern:cublaslt_bf16_tn",
                            "params": ["in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32", "i32"],
                            "args": [pms("cat"), pm(4), pms("h"), pm(10), i32(RANK), i32(2 * H)],
                        },
                        {
                            "entry": "extern:cublas_bf16_tn_f32",
                            "params": ["in buffer<bf16>", "in buffer<bf16>", "out buffer<f32>", "i32", "i32", "i32"],
                            "args": [pms("h"), pm(5), pms("fg"), pm(10), i32(2 * H), i32(RANK)],
                        },
                        {
                            "module": "loop_controller",
                            "entry": "kern_ctrl_merge_rmsnorm_f32_bf16",
                            "params": [
                                "in buffer<f32>",
                                "in buffer<f32>",
                                "in buffer<f32>",
                                "in buffer<bf16>",
                                "out buffer<bf16>",
                                "i32",
                                "i32",
                                "f32",
                            ],
                            "grid": ["tokens", 1, 1],
                            "block": [256, 1, 1],
                            "pdl": True,
                            "args": [pm(7), pm(3), pms("fg"), pm(6), pm(8), pm(10), i32(H), pm(12)],
                        },
                    ],
                },
            }

        if "pre_mlp_addend_mix_ctrl" not in ops:
            pma = lambda idx: {"param": idx}
            pmas = lambda name: {"scratch": name}
            ops["pre_mlp_addend_mix_ctrl"] = {
                "params": [
                    "in buffer<bf16>",   # 0: stack
                    "in buffer<bf16>",   # 1: partial
                    "in buffer<bf16>",   # 2: addend (attn_out)
                    "in buffer<f32>",    # 3: mlp_q
                    "in buffer<f32>",    # 4: msg_f32 (msg_buf[3])
                    "in buffer<bf16>",   # 5: w_down
                    "in buffer<bf16>",   # 6: w_fg
                    "in buffer<bf16>",   # 7: norm_w
                    "out buffer<f32>",   # 8: mix_mlp
                    "out buffer<bf16>",  # 9: x_mlp
                    "i64",               # 10: stack_stride
                    "i32",               # 11: T
                    "i32",               # 12: n_stack
                    "f32",               # 13: eps
                ],
                "impl": {
                    "scratch": {
                        "cat": {"dtype": "bf16", "shape": ["tokens", 2 * H]},
                        "h": {"dtype": "bf16", "shape": ["tokens", RANK]},
                        "fg": {"dtype": "f32", "shape": ["tokens", 2 * H]},
                    },
                    "launches": [
                        {
                            "module": "norm_attnres",
                            "entry": "kern_pre_mlp_addend_mix_s1_f32_bf16",
                            "params": [
                                "in buffer<bf16>",
                                "in buffer<bf16>",
                                "in buffer<bf16>",
                                "in buffer<f32>",
                                "in buffer<f32>",
                                "out buffer<f32>",
                                "out buffer<bf16>",
                                "i64",
                                "i32",
                                "i32",
                                "f32",
                            ],
                            "grid": ["tokens", 2, 1],
                            "block": [256, 1, 1],
                            "pdl": True,
                            "args": [
                                pma(0), pma(1), pma(2), pma(3), pma(4),
                                pma(8), pmas("cat"), pma(10), pma(11), pma(12), pma(13),
                            ],
                        },
                        {
                            "entry": "extern:cublaslt_bf16_tn",
                            "params": ["in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32", "i32"],
                            "args": [pmas("cat"), pma(5), pmas("h"), pma(11), i32(RANK), i32(2 * H)],
                        },
                        {
                            "entry": "extern:cublas_bf16_tn_f32",
                            "params": ["in buffer<bf16>", "in buffer<bf16>", "out buffer<f32>", "i32", "i32", "i32"],
                            "args": [pmas("h"), pma(6), pmas("fg"), pma(11), i32(2 * H), i32(RANK)],
                        },
                        {
                            "module": "loop_controller",
                            "entry": "kern_ctrl_merge_rmsnorm_f32_bf16",
                            "params": [
                                "in buffer<f32>",
                                "in buffer<f32>",
                                "in buffer<f32>",
                                "in buffer<bf16>",
                                "out buffer<bf16>",
                                "i32",
                                "i32",
                                "f32",
                            ],
                            "grid": ["tokens", 1, 1],
                            "block": [256, 1, 1],
                            "pdl": True,
                            "args": [pma(8), pma(4), pmas("fg"), pma(7), pma(9), pma(11), i32(H), pma(13)],
                        },
                    ],
                },
            }

    # Unembed + softcap + vocab mask + FP32 argmax
    single_op(
        "last_row_bf16",
        "unembed_sample",
        "kern_last_row_bf16",
        ["out buffer<bf16>", "in buffer<bf16>", "i32", "i32", "i32"],
        [1, 1, 1],
    )

    if "unembed_f32" not in ops:
        ops["unembed_f32"] = {
            "params": ["in buffer<bf16>", "in buffer<bf16>", "out buffer<f32>", "i32", "i32", "i32"],
            "impl": {
                "launches": [
                    {
                        "entry": "extern:cublas_bf16_tn_f32",
                        "params": ["in buffer<bf16>", "in buffer<bf16>", "out buffer<f32>", "i32", "i32", "i32"],
                        "args": [{"param": j} for j in range(6)],
                    }
                ]
            },
        }

    single_op(
        "softcap_mask_f32",
        "unembed_sample",
        "kern_softcap_tanh_mask_f32",
        ["inout buffer<f32>", "i32", "i32", "i32", "f32"],
        ["seqs", -(-VOCAB // 1024), 1],
    )

    if "argmax_f32" not in ops:
        ops["argmax_f32"] = {
            "params": ["in buffer<f32>", "out buffer<i64>", "i32", "i32"],
            "impl": {
                "scratch": {
                    "pmax": {"dtype": "f32", "shape": ["seqs", 64]},
                    "pidx": {"dtype": "i32", "shape": ["seqs", 64]},
                },
                "launches": [
                    {
                        "module": "unembed_sample",
                        "entry": "kern_argmax_partial_f32",
                        "params": ["in buffer<f32>", "out buffer<f32>", "out buffer<i32>", "i32"],
                        "grid": ["seqs", 64, 1],
                        "block": [1024, 1, 1],
                        "args": [{"param": 0}, {"scratch": "pmax"}, {"scratch": "pidx"}, {"param": 3}],
                    },
                    {
                        "module": "unembed_sample",
                        "entry": "kern_argmax_final_i64",
                        "params": ["in buffer<f32>", "in buffer<i32>", "out buffer<i64>", "i32"],
                        "grid": ["seqs", 1, 1],
                        "block": [64, 1, 1],
                        "args": [{"scratch": "pmax"}, {"scratch": "pidx"}, {"param": 1}, i32(64)],
                    },
                ],
            },
        }

    if include_sample_copy:
        single_op(
            "sample_copy_f32",
            "unembed_sample",
            "kern_sample_copy_f32",
            ["in buffer<f32>", "out buffer<f32>", "i32"],
            ["seqs", -(-VOCAB // 1024), 1],
            block=1024,
        )


def emit_derive_calls(buffers: dict, ops: dict, derive_calls: list) -> None:
    """Emit scale_f32 calls pre-scaling all AttnRes and cross-pass queries by 1/48."""
    _ = ops
    derive_calls.append(
        {
            "label": "derive.scale_output_q",
            "op": "scale_f32",
            "args": [b("src.output_q"), b("c.output_q"), i64(H), f32(INV_SQRT_H)],
        }
    )
    for i in range(NUM_PHYSICAL_LAYERS):
        if i > 0:
            q_elems = (3 * H) if i in FULL_LAYERS else H
            derive_calls.append(
                {
                    "label": f"derive.scale_attn_q.{i - 1}",
                    "op": "scale_f32",
                    "args": [b(f"src.attn_q.{i - 1}"), b(f"c.attn_q.{i - 1}"), i64(q_elems), f32(INV_SQRT_H)],
                }
            )
        derive_calls.append(
            {
                "label": f"derive.scale_mlp_q.{i}",
                "op": "scale_f32",
                "args": [b(f"src.mlp_q.{i}"), b(f"c.mlp_q.{i}"), i64(H), f32(INV_SQRT_H)],
            }
        )
    if "src.cross_pass_q" in buffers:
        derive_calls.append(
            {
                "label": "derive.scale_cross_pass_q",
                "op": "scale_f32",
                "args": [
                    b("src.cross_pass_q"),
                    b("c.cross_pass_q"),
                    i64((CORE_END - CORE_START) * 4 * H),
                    f32(INV_SQRT_H),
                ],
            }
        )


def emit_pre_loop_calls(token_ids_buf: str = "token_ids") -> list[dict]:
    """Emit embedding lookup + embed_norm writing initial `partial`."""
    T_var = {"var": "tokens"}
    return [
        {
            "label": "embed",
            "op": "embed_bf16",
            "args": [b(token_ids_buf), b("w.embed"), b("tmp_embed"), i32(H), i32(VOCAB)],
        },
        {
            "label": "embed_norm",
            "op": "rms_norm_bf16",
            "args": [b("tmp_embed"), b("w.embed_norm"), b("partial"), T_var, i32(H), i32(1), f32(EPS)],
        },
    ]


def emit_layer_calls(
    visit_idx: int,
    phys_idx: int,
    pass_idx: int,
    max_tokens: int = 1024,
    *,
    mode: str = "prefill",
    capture_aux: bool = True,
    aux_buf: str = "aux_hidden",
    token_var: str = "tokens",
) -> tuple[list[dict], list[dict], list[dict]]:
    """Emit (pre_attn_calls, pre_mlp_calls, post_mlp_calls) for logical visit `visit_idx`."""
    _ = mode
    T_var = {"var": token_var}
    t_elems = expr(mul(token_var, H))
    stack_stride = max_tokens * H
    out_q_stride = max_tokens * H

    first = (phys_idx % BLOCK_SIZE) == 0
    is_full = phys_idx in FULL_LAYERS
    n_blocks = 0 if phys_idx == 0 else (phys_idx - 1) // BLOCK_SIZE + 1

    pre_attn: list[dict] = []
    pre_mlp: list[dict] = []
    post_mlp: list[dict] = []

    # Aux capture before running layer at target_layer_ids in {5, 23, 41}
    if capture_aux and visit_idx in (5, 23, 41):
        tap_idx = {5: 0, 23: 1, 41: 2}[visit_idx]
        pre_attn.append(
            {
                "label": f"v{visit_idx}.aux_tap_{tap_idx}",
                "op": "tap_bf16",
                "args": [b("partial"), b(aux_buf), T_var, i32(H), i32(4), i32(tap_idx)],
            }
        )

    # Pass 1 & 2 start: snapshot prev pass's 2 core blocks (stack[2], partial) -> prev_blocks,
    # and reset (stack, partial) to ([embed_out], prelude_partial).
    if phys_idx == CORE_START and pass_idx in (1, 2):
        pre_attn.append(
            {
                "label": f"v{visit_idx}.save_prev_b0",
                "op": "copy_bf16",
                "args": [b("stack", 2 * max_tokens * H * 2), b("prev_blocks", 0), t_elems],
            }
        )
        pre_attn.append(
            {
                "label": f"v{visit_idx}.save_prev_b1",
                "op": "copy_bf16",
                "args": [b("partial"), b("prev_blocks", max_tokens * H * 2), t_elems],
            }
        )
        pre_attn.append(
            {
                "label": f"v{visit_idx}.reset_prelude_partial",
                "op": "copy_bf16",
                "args": [b("prelude_partial"), b("partial"), t_elems],
            }
        )

    # Pre-attention input preparation
    use_fused = (mode in ("round", "decode")) or (max_tokens >= 64)
    if pass_idx == 0:
        if phys_idx == 0:
            pre_attn.append(
                {
                    "label": f"v{visit_idx}.l0.attn_norm",
                    "op": "rms_norm_bf16",
                    "args": [b("partial"), b("w.l0.attn_norm"), b("x_qkv"), T_var, i32(H), i32(1), f32(EPS)],
                }
            )
        elif use_fused:
            norm_w = b(f"w.l{phys_idx}.qkv_input_norm") if is_full else b(f"w.l{phys_idx}.attn_norm")
            pre_attn.append(
                {
                    "label": f"v{visit_idx}.l{phys_idx}.attn_res_norm",
                    "op": "attn_res_norm_3q_bf16" if is_full else "attn_res_norm_1q_bf16",
                    "args": [
                        b("stack"),
                        b("partial"),
                        b(f"c.attn_q.{phys_idx - 1}"),
                        norm_w,
                        b("x_qkv"),
                        i64(stack_stride),
                        i64(out_q_stride),
                        T_var,
                        i32(n_blocks),
                        f32(EPS),
                        f32(EPS),
                    ],
                }
            )
        else:
            num_q = 3 if is_full else 1
            norm_w = b(f"w.l{phys_idx}.qkv_input_norm") if is_full else b(f"w.l{phys_idx}.attn_norm")
            pre_attn.append(
                {
                    "label": f"v{visit_idx}.l{phys_idx}.attn_res_norm",
                    "op": "attn_res_norm_bf16",
                    "args": [
                        b("stack"),
                        b("partial"),
                        b(f"c.attn_q.{phys_idx - 1}"),
                        norm_w,
                        b("x_qkv"),
                        i64(stack_stride),
                        i64(out_q_stride),
                        T_var,
                        i32(H),
                        i32(n_blocks),
                        i32(num_q),
                        f32(EPS),
                        f32(EPS),
                    ],
                }
            )
    else:
        c = phys_idx - CORE_START
        num_q = 3 if is_full else 1
        if use_fused:
            op_pa = "pre_attn_mix_cross_ctrl_full" if is_full else "pre_attn_mix_cross_ctrl_swa"
            nw_buf = b(f"w.l{phys_idx}.qkv_input_norm") if is_full else b(f"w.l{phys_idx}.attn_norm")
            in_stride = (max_tokens * H) if is_full else 0
            norm_stride = H if is_full else 0
            pre_attn.append(
                {
                    "label": f"v{visit_idx}.l{phys_idx}.pre_attn_ctrl",
                    "op": op_pa,
                    "args": [
                        b("stack"),
                        b("partial"),
                        b(f"c.attn_q.{phys_idx - 1}"),
                        b("prev_blocks"),
                        b("c.cross_pass_q", c * 4 * H * 4),
                        b(f"w.l{phys_idx}.ctrl.attn.down"),
                        b(f"w.l{phys_idx}.ctrl.attn.fg"),
                        b(f"w.l{phys_idx}.ctrl.k.down"),
                        b(f"w.l{phys_idx}.ctrl.k.fg"),
                        b(f"w.l{phys_idx}.ctrl.v.down"),
                        b(f"w.l{phys_idx}.ctrl.v.fg"),
                        nw_buf,
                        b("mix_qkv"),
                        b("msg_buf"),
                        b("x_qkv"),
                        i64(stack_stride),
                        i64(out_q_stride),
                        i64(in_stride),
                        i64(norm_stride),
                        T_var,
                        i32(n_blocks),
                        f32(EPS),
                    ],
                }
            )
        else:
            pre_attn.append(
                {
                    "label": f"v{visit_idx}.l{phys_idx}.attn_res_mix",
                    "op": "attn_res_mix_f32",
                    "args": [
                        b("stack"),
                        b("partial"),
                        b(f"c.attn_q.{phys_idx - 1}"),
                        b("mix_qkv"),
                        i64(stack_stride),
                        i64(out_q_stride),
                        T_var,
                        i32(H),
                        i32(n_blocks),
                        i32(num_q),
                        f32(EPS),
                    ],
                }
            )
            pre_attn.append(
                {
                    "label": f"v{visit_idx}.l{phys_idx}.cross_pass",
                    "op": "cross_pass_attn_res_f32",
                    "args": [
                        b("prev_blocks"),
                        b("c.cross_pass_q", c * 4 * H * 4),
                        b("msg_buf"),
                        i64(max_tokens * H),
                        i64(out_q_stride),
                        T_var,
                        i32(H),
                        i32(4),
                        f32(EPS),
                    ],
                }
            )
            stream_norms = (
                ("q_input_norm", "k_input_norm", "v_input_norm")
                if is_full
                else ("attn_norm", "attn_norm", "attn_norm")
            )
            for s_idx, (sname, nw_name) in enumerate(zip(("attn", "k", "v"), stream_norms)):
                mix_off = (s_idx * max_tokens * H * 4) if is_full else 0
                msg_off = s_idx * max_tokens * H * 4
                out_off = s_idx * max_tokens * H * 2
                pre_attn.append(
                    {
                        "label": f"v{visit_idx}.l{phys_idx}.ctrl.{sname}",
                        "op": "loop_controller",
                        "args": [
                            b("mix_qkv", mix_off),
                            b("msg_buf", msg_off),
                            b(f"w.l{phys_idx}.ctrl.{sname}.down"),
                            b(f"w.l{phys_idx}.ctrl.{sname}.fg"),
                            b(f"w.l{phys_idx}.{nw_name}"),
                            b("x_qkv", out_off),
                            T_var,
                            f32(EPS),
                        ],
                    }
                )

    # Pre-MLP input preparation
    if first:
        pre_mlp.append(
            {
                "label": f"v{visit_idx}.l{phys_idx}.push_block_{n_blocks}",
                "op": "copy_bf16",
                "args": [b("partial"), b("stack", n_blocks * max_tokens * H * 2), t_elems],
            }
        )
        mlp_n_stack = n_blocks + 1
        if pass_idx == 0:
            if use_fused:
                pre_mlp.append(
                    {
                        "label": f"v{visit_idx}.l{phys_idx}.mlp_attn_res_norm",
                        "op": "attn_res_norm_1q_bf16",
                        "args": [
                            b("stack"),
                            b("attn_out"),
                            b(f"c.mlp_q.{phys_idx}"),
                            b(f"w.l{phys_idx}.mlp_norm"),
                            b("x_mlp"),
                            i64(stack_stride),
                            i64(out_q_stride),
                            T_var,
                            i32(mlp_n_stack),
                            f32(EPS),
                            f32(EPS),
                        ],
                    }
                )
            else:
                pre_mlp.append(
                    {
                        "label": f"v{visit_idx}.l{phys_idx}.mlp_attn_res_norm",
                        "op": "attn_res_norm_bf16",
                        "args": [
                            b("stack"),
                            b("attn_out"),
                            b(f"c.mlp_q.{phys_idx}"),
                            b(f"w.l{phys_idx}.mlp_norm"),
                            b("x_mlp"),
                            i64(stack_stride),
                            i64(out_q_stride),
                            T_var,
                            i32(H),
                            i32(mlp_n_stack),
                            i32(1),
                            f32(EPS),
                            f32(EPS),
                        ],
                    }
                )
        elif use_fused:
            pre_mlp.append(
                {
                    "label": f"v{visit_idx}.l{phys_idx}.pre_mlp_ctrl",
                    "op": "pre_mlp_mix_ctrl",
                    "args": [
                        b("stack"),
                        b("attn_out"),
                        b(f"c.mlp_q.{phys_idx}"),
                        b("msg_buf", 3 * max_tokens * H * 4),
                        b(f"w.l{phys_idx}.ctrl.mlp.down"),
                        b(f"w.l{phys_idx}.ctrl.mlp.fg"),
                        b(f"w.l{phys_idx}.mlp_norm"),
                        b("mix_mlp"),
                        b("x_mlp"),
                        i64(stack_stride),
                        T_var,
                        i32(mlp_n_stack),
                        f32(EPS),
                    ],
                }
            )
        else:
            pre_mlp.append(
                {
                    "label": f"v{visit_idx}.l{phys_idx}.mlp_attn_res_mix",
                    "op": "attn_res_mix_f32",
                    "args": [
                        b("stack"),
                        b("attn_out"),
                        b(f"c.mlp_q.{phys_idx}"),
                        b("mix_mlp"),
                        i64(stack_stride),
                        i64(out_q_stride),
                        T_var,
                        i32(H),
                        i32(mlp_n_stack),
                        i32(1),
                        f32(EPS),
                    ],
                }
            )
            pre_mlp.append(
                {
                    "label": f"v{visit_idx}.l{phys_idx}.ctrl.mlp",
                    "op": "loop_controller",
                    "args": [
                        b("mix_mlp"),
                        b("msg_buf", 3 * max_tokens * H * 4),
                        b(f"w.l{phys_idx}.ctrl.mlp.down"),
                        b(f"w.l{phys_idx}.ctrl.mlp.fg"),
                        b(f"w.l{phys_idx}.mlp_norm"),
                        b("x_mlp"),
                        T_var,
                        f32(EPS),
                    ],
                }
            )
    else:
        mlp_n_stack = n_blocks
        if pass_idx == 0:
            pre_mlp.append(
                {
                    "label": f"v{visit_idx}.l{phys_idx}.mlp_attn_res_addend_norm",
                    "op": "attn_res_addend_norm_bf16",
                    "args": [
                        b("stack"),
                        b("partial"),
                        b("attn_out"),
                        b(f"c.mlp_q.{phys_idx}"),
                        b(f"w.l{phys_idx}.mlp_norm"),
                        b("x_mlp"),
                        i64(stack_stride),
                        i64(out_q_stride),
                        T_var,
                        i32(H),
                        i32(mlp_n_stack),
                        i32(1),
                        f32(EPS),
                        f32(EPS),
                    ],
                }
            )
        elif use_fused:
            pre_mlp.append(
                {
                    "label": f"v{visit_idx}.l{phys_idx}.pre_mlp_ctrl",
                    "op": "pre_mlp_addend_mix_ctrl",
                    "args": [
                        b("stack"),
                        b("partial"),
                        b("attn_out"),
                        b(f"c.mlp_q.{phys_idx}"),
                        b("msg_buf", 3 * max_tokens * H * 4),
                        b(f"w.l{phys_idx}.ctrl.mlp.down"),
                        b(f"w.l{phys_idx}.ctrl.mlp.fg"),
                        b(f"w.l{phys_idx}.mlp_norm"),
                        b("mix_mlp"),
                        b("x_mlp"),
                        i64(stack_stride),
                        T_var,
                        i32(mlp_n_stack),
                        f32(EPS),
                    ],
                }
            )
        else:
            pre_mlp.append(
                {
                    "label": f"v{visit_idx}.l{phys_idx}.mlp_attn_res_addend_mix",
                    "op": "attn_res_addend_mix_f32",
                    "args": [
                        b("stack"),
                        b("partial"),
                        b("attn_out"),
                        b(f"c.mlp_q.{phys_idx}"),
                        b("mix_mlp"),
                        i64(stack_stride),
                        i64(out_q_stride),
                        T_var,
                        i32(H),
                        i32(mlp_n_stack),
                        i32(1),
                        f32(EPS),
                    ],
                }
            )
            pre_mlp.append(
                {
                    "label": f"v{visit_idx}.l{phys_idx}.ctrl.mlp",
                    "op": "loop_controller",
                    "args": [
                        b("mix_mlp"),
                        b("msg_buf", 3 * max_tokens * H * 4),
                        b(f"w.l{phys_idx}.ctrl.mlp.down"),
                        b(f"w.l{phys_idx}.ctrl.mlp.fg"),
                        b(f"w.l{phys_idx}.mlp_norm"),
                        b("x_mlp"),
                        T_var,
                        f32(EPS),
                    ],
                }
            )

    # Post-MLP block accumulation
    if first:
        post_mlp.append(
            {
                "label": f"v{visit_idx}.l{phys_idx}.residual_add2",
                "op": "add2_bf16",
                "args": [b("attn_out"), b("mlp_out"), b("partial"), t_elems],
            }
        )
    else:
        post_mlp.append(
            {
                "label": f"v{visit_idx}.l{phys_idx}.residual_add3",
                "op": "add3_bf16",
                "args": [b("partial"), b("attn_out"), b("mlp_out"), b("partial"), t_elems],
            }
        )

    if visit_idx == 5:
        post_mlp.append(
            {
                "label": "v5.save_prelude_partial",
                "op": "copy_bf16",
                "args": [b("partial"), b("prelude_partial"), t_elems],
            }
        )

    return pre_attn, pre_mlp, post_mlp


def emit_post_loop_calls(
    max_tokens: int = 1024,
    *,
    mode: str = "prefill",
    capture_aux: bool = True,
    aux_buf: str = "aux_hidden",
    sampling_hook: list[dict] | None = None,
) -> list[dict]:
    """Emit final AttnRes + model.llm.norm, optional aux tap 48, unembed_f32, softcap+mask, and argmax/sampling."""
    T_var = {"var": "tokens"}
    S_var = {"var": "seqs"}
    stack_stride = max_tokens * H
    out_q_stride = max_tokens * H

    calls: list[dict] = [
        {
            "label": "final_attn_res_norm",
            "op": "attn_res_norm_bf16",
            "args": [
                b("stack"),
                b("partial"),
                b("c.output_q"),
                b("w.final_norm"),
                b("final_h"),
                i64(stack_stride),
                i64(out_q_stride),
                T_var,
                i32(H),
                i32(4),
                i32(1),
                f32(EPS),
                f32(EPS),
            ],
        }
    ]
    if capture_aux:
        calls.append(
            {
                "label": "v48.aux_tap_3",
                "op": "tap_bf16",
                "args": [b("final_h"), b(aux_buf), T_var, i32(H), i32(4), i32(3)],
            }
        )

    if mode == "prefill":
        calls.append(
            {
                "label": "last_row",
                "op": "last_row_bf16",
                "args": [b("last_h"), b("final_h"), i32(H), i32(H), T_var],
            }
        )
        unembed_in = b("last_h")
        rows_arg = S_var
    else:
        unembed_in = b("final_h")
        rows_arg = S_var

    calls += [
        {
            "label": "unembed",
            "op": "unembed_f32",
            "args": [unembed_in, b("w.unembed"), b("logits"), rows_arg, i32(VOCAB), i32(H)],
        },
        {
            "label": "softcap_mask",
            "op": "softcap_mask_f32",
            "args": [b("logits"), rows_arg, i32(VOCAB), i32(UNPADDED_VOCAB), f32(SOFTCAP)],
        },
    ]
    if sampling_hook is not None:
        calls += sampling_hook
    else:
        calls.append(
            {
                "label": "argmax",
                "op": "argmax_f32",
                "args": [b("logits"), b("next_token"), rows_arg, i32(VOCAB)],
            }
        )
    return calls
