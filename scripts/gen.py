#!/usr/bin/env python3
"""Assemble complete schema-v7 `model.json`, `kern.toml`, and `kernels/` bundle for Inkling-2-Mini."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

SCRIPTS_DIR = Path(__file__).resolve().parent
ROOT = SCRIPTS_DIR.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import attn_manifest  # noqa: E402
import dspark_manifest  # noqa: E402
import loop_manifest  # noqa: E402
import moe_manifest  # noqa: E402

DEFAULT_CHECKPOINT = Path("/workspace/weights/Inkling-2-Mini")
DEFAULT_KERNELS = ROOT / "build" / "sm_120"
DEFAULT_OUT = ROOT / "out" / "inkling-2-mini"
DEFAULT_KV_SCALES = ROOT / "results" / "kv-scales.json"

H = loop_manifest.H  # 2304
VOCAB = loop_manifest.VOCAB  # 201024
UNPADDED_VOCAB = loop_manifest.UNPADDED_VOCAB  # 200058
EPS = loop_manifest.EPS  # 1e-6


def _b(name: str, offset: int = 0) -> dict[str, Any]:
    return {"buf": name, "offset": offset} if offset else {"buf": name}


def _i32(v: int) -> dict[str, int]:
    return {"i32": int(v)}


def _i64(v: int) -> dict[str, int]:
    return {"i64": int(v)}


def _f32(v: float) -> dict[str, float]:
    return {"f32": float(v)}


def _load_kv_scales(path: Path | None) -> dict[str, dict[str, float]]:
    if path is not None and path.exists():
        return json.loads(path.read_text())
    k_fallback = 22.627416997969522 / 448.0
    v_fallback = 11.313708498984761 / 448.0
    return {
        str(v): {"k": k_fallback, "v": v_fallback}
        for v in range(loop_manifest.NUM_LOGICAL_VISITS)
    }


def _register_attn_sconv_buffers(buffers: dict[str, Any]) -> None:
    def buf(name: str, dtype: str, shape: list[Any], kind: str = "workspace", **extra: Any) -> None:
        if name not in buffers:
            buffers[name] = {"dtype": dtype, "shape": shape, "kind": kind, **extra}

    def weight(name: str, shape: list[int], tensor: str) -> None:
        buf(name, "bf16", shape, "weight", bind=[{"tensor": tensor}])

    # Attention / sconv / MoE / verify workspaces
    buf("attn.q", "bf16", ["tokens", 3072])
    buf("attn.r", "bf16", ["tokens", 384])
    buf("attn.k", "bf16", ["tokens", 1024])
    buf("attn.v", "bf16", ["tokens", 1024])
    buf("attn.gate", "bf16", ["tokens", 3072])
    buf("attn.sink", "bf16", ["tokens", 24])
    buf("attn.proj", "bf16", ["tokens", H])
    buf("moe_out", "bf16", ["tokens", H])
    buf("d.verify_logits", "f32", ["tokens", VOCAB])

    # Per-layer attention + sconv weights (24 physical layers)
    for p in range(loop_manifest.NUM_PHYSICAL_LAYERS):
        pfx = f"model.llm.layers.{p}"
        weight(f"{pfx}.attn.wq_du.weight", [3072, H], f"{pfx}.attn.wq_du.weight")
        weight(f"{pfx}.attn.wr_du.weight", [384, H], f"{pfx}.attn.wr_du.weight")
        weight(f"{pfx}.attn.wk_dv.weight", [1024, H], f"{pfx}.attn.wk_dv.weight")
        weight(f"{pfx}.attn.wv_dv.weight", [1024, H], f"{pfx}.attn.wv_dv.weight")
        weight(f"{pfx}.attn.gate_attn.weight", [3072, H], f"{pfx}.attn.gate_attn.weight")
        weight(f"{pfx}.attn.attn_sink_proj.weight", [24, H], f"{pfx}.attn.attn_sink_proj.weight")
        weight(f"{pfx}.attn.wo_ud.weight", [H, 3072], f"{pfx}.attn.wo_ud.weight")
        weight(f"{pfx}.attn.q_l2_clip.scale", [24, 128], f"{pfx}.attn.q_l2_clip.scale")
        weight(f"{pfx}.attn.rel_l2_clip.scale", [24, 16], f"{pfx}.attn.rel_l2_clip.scale")
        weight(f"{pfx}.attn.rel_logits_proj.proj", [16, 1024], f"{pfx}.attn.rel_logits_proj.proj")
        weight(f"{pfx}.attn_sconv.weight", [H, 4], f"{pfx}.attn_sconv.weight")
        weight(f"{pfx}.mlp_sconv.weight", [H, 4], f"{pfx}.mlp_sconv.weight")
        if p in loop_manifest.FULL_LAYERS:
            for st in ("k", "v"):
                weight(f"{pfx}.attn.{st}_sconv.static_w", [4, 1024], f"{pfx}.attn.{st}_sconv.static_w")
                weight(f"{pfx}.attn.{st}_sconv.weight_proj", [32, H], f"{pfx}.attn.{st}_sconv.weight_proj")
        else:
            for st in ("k", "v"):
                weight(f"{pfx}.attn.{st}_sconv.weight", [1024, 4], f"{pfx}.attn.{st}_sconv.weight")


def build_manifest(
    kernels_dir: Path,
    kv_scales: dict[str, dict[str, float]],
    *,
    max_seqs: int = 8,
    chunk: int = 2048,
    context: int = 1048640,
    page_columns: int = 16385,
    dspark_k_scale: float = 1.0,
    dspark_v_scale: float = 1.0,
    sms: int = 110,
    shared_mem: int = 101376,
) -> dict[str, Any]:
    max_tokens = max(chunk, max_seqs * 8)
    target_sm = {"sms": sms}
    sampling_target = {"sms": sms, "shared_mem": shared_mem}

    modules: dict[str, Any] = {}
    states: dict[str, Any] = {}
    buffers: dict[str, Any] = {}
    ops: dict[str, Any] = {}

    m: dict[str, Any] = {
        "schema_version": 7,
        "model": "inkling-2-mini",
        "chat": {"tool_parser": "inkling"},
        "vars": {
            "tokens": {"max": max_tokens, "axis": "rows"},
            "seqs": {"max": max_seqs, "axis": "groups"},
        },
        "states": states,
        "buffers": buffers,
        "modules": modules,
        "ops": ops,
        "programs": {},
    }

    # 1. Register modules
    loop_manifest.register_modules(modules, kernels_dir)
    moe_meta = moe_manifest.register_modules(modules, kernels_dir)
    dsl = attn_manifest.register_modules(modules, kernels_dir)

    # 2. Register states
    attn_manifest.register_states(states)

    # 3. Register shared & per-module buffers
    loop_manifest.register_buffers(
        buffers,
        max_seqs=max_seqs,
        max_tokens=max_tokens,
        include_aux=True,
        token_axis="tokens",
    )
    layer_weights = moe_manifest.register_buffers(
        buffers,
        max_seqs=max_seqs,
        max_tokens=max_tokens,
        token_axis="tokens",
    )
    attn_manifest.register_buffers(
        buffers,
        max_seqs=max_seqs,
        chunk=chunk,
        context=context,
        page_columns=page_columns,
    )

    buffers["token_ids"] = {
        "dtype": "i64",
        "shape": ["tokens"],
        "kind": "input",
        "fill": "token",
        "domain": {"min": 0, "max": UNPADDED_VOCAB - 1},
    }
    buffers["slots"] = {
        "dtype": "i32",
        "shape": ["tokens"],
        "kind": "input",
        "fill": "slot",
        "domain": {"index_into": "kv"},
    }
    buffers.update(dspark_manifest.sampling_fills())
    _register_attn_sconv_buffers(buffers)

    # 4. Register ops
    loop_manifest.register_ops(
        ops,
        modules,
        max_seqs=max_seqs,
        max_tokens=max_tokens,
        target_sm=target_sm,
        include_sample_copy=False,
    )
    moe_manifest.register_ops(
        ops,
        modules,
        moe_meta,
        max_tokens=max_tokens,
        max_seqs=max_seqs,
        target_sm=target_sm,
        token_axis="tokens",
    )
    attn_manifest.register_ops(
        ops,
        dsl,
        max_seqs=max_seqs,
        page_columns=page_columns,
    )

    layout = dspark_manifest.DsparkTargetLayout(
        page_size=64,
        page_columns=page_columns,
        kv_state="kv",
        num_kv_splits=16,
        target_embed_buf="w.embed",
        target_embed_norm_buf="w.embed_norm",
        target_lm_head_buf="w.unembed",
        target_verify_logits_buf="d.verify_logits",
        target_verify_logits_dtype="f32",
        taps_buf="aux_hidden",
        seq_state_name="seq_state",
        seq_lines_buf="seq_state.lines",
        max_seqs=max_seqs,
        valid_buf="valid",
        dspark_k_scale=dspark_k_scale,
        dspark_v_scale=dspark_v_scale,
    )
    loaded = dspark_manifest.register_ops(
        m,
        kernels_dir,
        layout,
        sampling_target,
        register_target_commit_ops=False,
    )
    ops["sample"] = dspark_manifest.sampling_op(
        VOCAB,
        max_seqs,
        loaded["spec"]["sampling"],
        sampling_target,
        logits_dtype="f32",
        axis_var="seqs",
    )

    # 5. Derive program
    derive_calls: list[dict[str, Any]] = []
    loop_manifest.emit_derive_calls(buffers, ops, derive_calls)
    moe_manifest.emit_derive_calls(buffers, derive_calls)
    scales = attn_manifest.emit_load_calls(derive_calls, kv_scales, ops, buffers)
    derive_calls.extend(dspark_manifest.emit_derive_calls(m, layout))
    m["programs"]["derive"] = {"derive": True, "calls": derive_calls}

    # 6. Shared 48-visit target body builder
    def build_target_body(mode: str) -> list[dict[str, Any]]:
        m1_mode = "decode" if mode == "decode_batch" else mode
        t_var = "seqs" if mode == "decode_batch" else "tokens"
        tau_op = "attn_log_tau_decode" if mode == "decode_batch" else "attn_log_tau"
        calls: list[dict[str, Any]] = [
            {
                "label": "attn.log_tau",
                "op": tau_op,
                "args": [_b("positions"), _b("attn.tau"), {"var": t_var}],
            }
        ]
        calls.extend(
            loop_manifest.emit_pre_loop_calls(
                token_ids_buf="d.verify_ids" if mode == "round" else "token_ids"
            )
        )

        for v, phys, _phase, pass_idx in loop_manifest.visit_schedule():
            pre_attn, pre_mlp, post_mlp = loop_manifest.emit_layer_calls(
                v,
                phys,
                pass_idx,
                max_tokens,
                mode=m1_mode,
                capture_aux=True,
                aux_buf="aux_hidden",
            )
            calls.extend(pre_attn)

            x_q = _b("x_qkv", 0)
            if pass_idx > 0 or phys in loop_manifest.FULL_LAYERS:
                x_k = _b("x_qkv", max_tokens * H * 2)
                x_v = _b("x_qkv", 2 * max_tokens * H * 2)
            else:
                x_k = x_q
                x_v = x_q

            pfx = f"model.llm.layers.{phys}"
            calls.extend(
                [
                    {
                        "label": f"v{v}.q_proj",
                        "op": "m3_gemm_bf16",
                        "args": [x_q, _b(f"{pfx}.attn.wq_du.weight"), _b("attn.q"), {"var": t_var}, _i32(3072), _i32(H)],
                    },
                    {
                        "label": f"v{v}.r_proj",
                        "op": "m3_gemm_bf16",
                        "args": [x_q, _b(f"{pfx}.attn.wr_du.weight"), _b("attn.r"), {"var": t_var}, _i32(384), _i32(H)],
                    },
                    {
                        "label": f"v{v}.k_proj",
                        "op": "m3_gemm_bf16",
                        "args": [x_k, _b(f"{pfx}.attn.wk_dv.weight"), _b("attn.k"), {"var": t_var}, _i32(1024), _i32(H)],
                    },
                    {
                        "label": f"v{v}.v_proj",
                        "op": "m3_gemm_bf16",
                        "args": [x_v, _b(f"{pfx}.attn.wv_dv.weight"), _b("attn.v"), {"var": t_var}, _i32(1024), _i32(H)],
                    },
                    {
                        "label": f"v{v}.gate_proj",
                        "op": "m3_gemm_bf16",
                        "args": [x_q, _b(f"{pfx}.attn.gate_attn.weight"), _b("attn.gate"), {"var": t_var}, _i32(3072), _i32(H)],
                    },
                    {
                        "label": f"v{v}.sink_proj",
                        "op": "m3_gemm_bf16",
                        "args": [x_q, _b(f"{pfx}.attn.attn_sink_proj.weight"), _b("attn.sink"), {"var": t_var}, _i32(24), _i32(H)],
                    },
                ]
            )

            attn_manifest.emit_sconv_kv(
                calls,
                mode,
                v,
                phys,
                _b("attn.k"),
                _b("attn.v"),
                x_k,
                x_v,
                _b("sconv.k_out"),
                _b("sconv.v_out"),
                max_seqs=max_seqs,
            )

            ks_v, vs_v = scales[v]
            attn_manifest.emit_layer_calls(
                calls,
                mode,
                v,
                phys,
                _b("attn.q"),
                _b("attn.r"),
                _b("sconv.k_out"),
                _b("sconv.v_out"),
                _b("attn.gate"),
                _b("attn.sink"),
                _b("attn.out"),
                ks_v,
                vs_v,
                max_seqs=max_seqs,
            )

            calls.append(
                {
                    "label": f"v{v}.o_proj",
                    "op": "m3_gemm_bf16",
                    "args": [_b("attn.out"), _b(f"{pfx}.attn.wo_ud.weight"), _b("attn.proj"), {"var": t_var}, _i32(H), _i32(3072)],
                }
            )
            attn_manifest.emit_sconv_attn(
                calls,
                mode,
                v,
                phys,
                _b("attn.proj"),
                _b("attn_out"),
                max_seqs=max_seqs,
            )

            calls.extend(pre_mlp)
            calls.extend(
                moe_manifest.emit_moe_layer_calls(
                    mode,
                    v,
                    phys,
                    layer_weights,
                    "x_mlp",
                    "moe_out",
                )
            )
            attn_manifest.emit_sconv_mlp(
                calls,
                mode,
                v,
                phys,
                _b("moe_out"),
                _b("partial"),
                _b("attn_out"),
                _b("partial"),
                max_seqs=max_seqs,
            )
            if v == 5:
                calls.append(post_mlp[-1])

        return calls

    sample_call = {
        "label": "sample",
        "op": "sample",
        "args": [
            _b("logits"),
            _b("lengths"),
            _b("temperature"),
            _b("top_k"),
            _b("top_p"),
            _b("min_p"),
            _b("seed"),
            _b("next_token"),
            {"var": "seqs"},
        ],
    }

    # 7. Prefill program (groups=1, rows=tokens)
    _, pre_tail = dspark_manifest.emit_prefill_capture(m, layout, taps_buf="aux_hidden")
    pre_calls = build_target_body("prefill")
    pre_calls.extend(
        loop_manifest.emit_post_loop_calls(
            max_tokens,
            mode="prefill",
            capture_aux=True,
            aux_buf="aux_hidden",
            sampling_hook=[sample_call],
        )
    )
    pre_calls.extend(pre_tail)
    m["programs"]["prefill"] = {"batch": {"groups": 1, "rows": "tokens"}, "calls": pre_calls}

    # 8. Decode program (groups=max_seqs, rows=1, graph=true)
    _, dec_tail = dspark_manifest.emit_decode_capture(m, layout, taps_buf="aux_hidden")
    dec_calls = build_target_body("decode_batch")
    dec_calls.extend(
        loop_manifest.emit_post_loop_calls(
            max_tokens,
            mode="decode",
            capture_aux=True,
            aux_buf="aux_hidden",
            sampling_hook=[sample_call],
        )
    )
    dec_calls.extend(dec_tail)
    m["programs"]["decode"] = {"batch": {"groups": max_seqs, "rows": 1}, "graph": True, "calls": dec_calls}

    # 9. Round program (groups=max_seqs, rows=8, graph=true)
    verify_calls = build_target_body("round")
    verify_calls.extend(
        [
            {
                "label": "final_attn_res_norm",
                "op": "attn_res_norm_bf16",
                "args": [
                    _b("stack"),
                    _b("partial"),
                    _b("c.output_q"),
                    _b("w.final_norm"),
                    _b("final_h"),
                    _i64(max_tokens * H),
                    _i64(max_tokens * H),
                    {"var": "tokens"},
                    _i32(H),
                    _i32(4),
                    _i32(1),
                    _f32(EPS),
                    _f32(EPS),
                ],
            },
            {
                "label": "v48.aux_tap_3",
                "op": "tap_bf16",
                "args": [_b("final_h"), _b("aux_hidden"), {"var": "tokens"}, _i32(H), _i32(4), _i32(3)],
            },
            {
                "label": "verify.unembed",
                "op": "unembed_f32",
                "args": [_b("final_h"), _b("w.unembed"), _b("d.verify_logits"), {"var": "tokens"}, _i32(VOCAB), _i32(H)],
            },
        ]
    )
    commit_calls: list[dict[str, Any]] = []
    attn_manifest.emit_sconv_spec_commit(commit_calls, accepted_buf="accepted")
    attn_manifest.emit_swa_spec_commit(commit_calls, accepted_buf="accepted")
    m["programs"]["round"] = dspark_manifest.emit_round_program(
        m,
        layout,
        verify_calls,
        target_commit_calls=commit_calls,
        taps_buf="aux_hidden",
    )

    return attn_manifest.prune_unused(m)


def _write_weights_overlay(
    checkpoint: Path,
    out_dir: Path,
    max_position_embeddings: int = 1048576,
) -> Path:
    """Create a symlinked weights overlay preserving 'inkling' in the path with 1M max_position_embeddings."""
    ckpt = checkpoint.resolve()
    overlay_dir = out_dir / "weights_inkling_1m"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    expected_entries = [
        entry
        for entry in sorted(ckpt.iterdir(), key=lambda p: p.name)
        if not entry.name.startswith(".")
    ]
    expected_names = {entry.name for entry in expected_entries}

    for existing in list(overlay_dir.iterdir()):
        if existing.name not in expected_names:
            if existing.is_symlink() or existing.is_file():
                existing.unlink()
            elif existing.is_dir():
                shutil.rmtree(existing)

    for entry in expected_entries:
        if entry.name == "config.json":
            continue
        dst = overlay_dir / entry.name
        if dst.is_symlink():
            if os.readlink(dst) == str(entry):
                continue
            dst.unlink()
        elif dst.is_file():
            dst.unlink()
        elif dst.is_dir():
            shutil.rmtree(dst)
        os.symlink(entry, dst)

    cfg_src = ckpt / "config.json"
    cfg = json.loads(cfg_src.read_text(encoding="utf-8"))
    resolved_max_pos = int(
        cfg.get("text_config", {}).get("model_max_length", max_position_embeddings)
    )
    cfg.setdefault("max_position_embeddings", resolved_max_pos)
    unpadded_vocab = cfg.get("text_config", {}).get("unpadded_vocab_size")
    if unpadded_vocab is not None:
        cfg.setdefault("vocab_size", int(unpadded_vocab))
    tmp_cfg = overlay_dir / ".config.json.tmp"
    tmp_cfg.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    tmp_cfg.replace(overlay_dir / "config.json")
    return overlay_dir


def write_bundle(
    manifest: dict[str, Any],
    kernels_src: Path,
    out_dir: Path,
    checkpoint: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle_kernels = out_dir / "kernels"
    bundle_kernels.mkdir(parents=True, exist_ok=True)
    for item in manifest["modules"].values():
        src = kernels_src / item["source"]
        dst = bundle_kernels / item["source"]
        if not src.exists():
            raise FileNotFoundError(f"Missing kernel cubin: {src}")
        shutil.copyfile(src, dst)

    model_path = out_dir / "model.json"
    tmp_model = out_dir / ".model.json.tmp"
    tmp_model.write_text(json.dumps(manifest, indent=2) + "\n")
    tmp_model.replace(model_path)

    overlay_dir = _write_weights_overlay(checkpoint, out_dir)
    overlay_rel = overlay_dir.name
    toml_lines = [
        "[targets.inkling_2_mini]",
        'manifest = "model.json"',
        'kernels = "kernels"',
        f'weights = ["{overlay_rel}"]',
        f'tokenizer = "{overlay_rel}/tokenizer.json"',
        "",
    ]
    (out_dir / "kern.toml").write_text("\n".join(toml_lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    ap.add_argument("--kernels", type=Path, default=DEFAULT_KERNELS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--kv-scales", type=Path, default=DEFAULT_KV_SCALES)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--max-seqs", type=int, default=8)
    ap.add_argument("--context", type=int, default=1048640)
    ap.add_argument("--page-columns", type=int, default=16385)
    ap.add_argument("--dspark-k-scale", type=float, default=1.0)
    ap.add_argument("--dspark-v-scale", type=float, default=1.0)
    ap.add_argument("--sms", type=int, default=110)
    ap.add_argument("--shared-mem", type=int, default=101376)
    args = ap.parse_args()

    kernels_dir = args.kernels if args.kernels.is_absolute() else ROOT / args.kernels
    out_dir = args.out if args.out.is_absolute() else ROOT / args.out
    kv_path = args.kv_scales if args.kv_scales.is_absolute() else ROOT / args.kv_scales

    kv_scales = _load_kv_scales(kv_path)
    manifest = build_manifest(
        kernels_dir,
        kv_scales,
        max_seqs=args.max_seqs,
        chunk=args.chunk,
        context=args.context,
        page_columns=args.page_columns,
        dspark_k_scale=args.dspark_k_scale,
        dspark_v_scale=args.dspark_v_scale,
        sms=args.sms,
        shared_mem=args.shared_mem,
    )
    write_bundle(manifest, kernels_dir, out_dir, args.checkpoint)
    print(
        f"Generated {out_dir / 'model.json'} "
        f"(modules={len(manifest['modules'])}, buffers={len(manifest['buffers'])}, "
        f"ops={len(manifest['ops'])}, programs={list(manifest['programs'].keys())})"
    )


if __name__ == "__main__":
    main()
