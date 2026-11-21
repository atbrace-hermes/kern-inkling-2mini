#!/usr/bin/env python3
"""GPU unit tests for M3 of kern-inkling-2mini:
  - 4-stream static/dynamic sconv (prefill, decode, verify, spec_commit_all, null/valid guards)
  - attn_prologue (Q/R/K/V clip/norm, log_scaling_tau, unsheared rel_bias, FP8 E4M3 KV write)
  - SM120 rel_attention (full paged + SWA ring, prefill split/single, 1M split-KV decode, verify causal & rollback, CUDA graphs)
"""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[2]
BUILD_DIR = ROOT / "build/sm_120"
KERN_BIN = Path("/workspace/kern/target/release/kern")
sys.path.insert(0, str(ROOT / "scripts"))

import attn_manifest as am  # noqa: E402
import calibrate_kv as ck  # noqa: E402


def _load_kern():
    """Import `kern` + `kern.host`, falling back to workspace `kern-py` + `lib_kern.so` if not pip-installed."""
    try:
        import kern  # type: ignore
        import kern.host  # type: ignore

        if hasattr(kern, "Runtime"):
            return kern, kern.host
    except ImportError:
        pass

    kern_py_dir = Path("/workspace/kern/crates/kern-py/python/kern")
    lib_kern_so = Path("/workspace/kern/target/release/lib_kern.so")
    if not lib_kern_so.exists():
        raise RuntimeError(f"Missing {lib_kern_so}")

    spec = importlib.util.spec_from_file_location("kern._kern", str(lib_kern_so))
    assert spec is not None and spec.loader is not None
    _kern = importlib.util.module_from_spec(spec)
    # Build minimal `kern` package in sys.modules
    pkg_spec = importlib.util.spec_from_file_location("kern", str(kern_py_dir / "host.py"))
    assert pkg_spec is not None
    import types

    kern_pkg = types.ModuleType("kern")
    kern_pkg.__path__ = [str(kern_py_dir)]
    sys.modules["kern"] = kern_pkg
    sys.modules["kern._kern"] = _kern
    spec.loader.exec_module(_kern)
    kern_pkg.Runtime = _kern.Runtime
    kern_pkg.__version__ = _kern.__version__

    host_spec = importlib.util.spec_from_file_location("kern.host", str(kern_py_dir / "host.py"))
    assert host_spec is not None and host_spec.loader is not None
    host_mod = importlib.util.module_from_spec(host_spec)
    sys.modules["kern.host"] = host_mod
    host_spec.loader.exec_module(host_mod)
    return kern_pkg, host_mod


# ---------------------------------------------------------------------------
# PyTorch FP64 / FP32 Reference Implementations
# ---------------------------------------------------------------------------
def ref_sconv_static(x: torch.Tensor, w: torch.Tensor, history: torch.Tensor | None, positions: torch.Tensor,
                     addend: torch.Tensor | None = None, residual: torch.Tensor | None = None) -> torch.Tensor:
    """Reference static sconv (5^-0.5, oldest-first k=0..3, FMA sum + non-FMA final scale)."""
    t_len, c_dim = x.shape
    scale = 5.0**-0.5  # fp32: 0.4472135901451111
    scale_f32 = float(torch.tensor(scale, dtype=torch.float32).item())
    w_f = w.to(torch.bfloat16).float()
    x_f = x.to(torch.bfloat16).float()
    h_f = history.to(torch.bfloat16).float() if history is not None else None
    out = torch.empty((t_len, c_dim), dtype=torch.bfloat16, device=x.device)
    for r in range(t_len):
        p = int(positions[r].item())
        acc = torch.zeros(c_dim, dtype=torch.float32, device=x.device)
        for k in range(4):
            lag = 3 - k
            if p - lag < 0:
                tap = torch.zeros(c_dim, dtype=torch.float32, device=x.device)
            elif r - lag >= 0:
                tap = x_f[r - lag]
            else:
                tap = h_f[3 + r - lag] if h_f is not None else torch.zeros(c_dim, dtype=torch.float32, device=x.device)
            acc = acc + tap * w_f[:, k]
        y_bf = ((acc + x_f[r]) * scale_f32).to(torch.bfloat16)
        if addend is not None:
            a_f = addend[r].to(torch.bfloat16).float()
            if residual is not None:
                a_f = residual[r].to(torch.bfloat16).float() + a_f
            y_bf = (a_f + y_bf.float()).to(torch.bfloat16)
        out[r] = y_bf
    return out


def ref_sconv_dynamic(x: torch.Tensor, static_w: torch.Tensor, dyn: torch.Tensor,
                      history: torch.Tensor | None, positions: torch.Tensor) -> torch.Tensor:
    """Reference dynamic sconv (9^-0.5, newest-first lag=0..3)."""
    t_len, c_dim = x.shape
    scale_f32 = float(torch.tensor(9.0**-0.5, dtype=torch.float32).item())
    sw_f = static_w.to(torch.bfloat16).float()
    dyn_f = dyn.to(torch.bfloat16).float().view(t_len, 8, 4)
    x_f = x.to(torch.bfloat16).float()
    h_f = history.to(torch.bfloat16).float() if history is not None else None
    out = torch.empty((t_len, c_dim), dtype=torch.bfloat16, device=x.device)
    for r in range(t_len):
        p = int(positions[r].item())
        acc = torch.zeros(c_dim, dtype=torch.float32, device=x.device)
        for lag in range(4):
            if p - lag < 0:
                tap = torch.zeros(c_dim, dtype=torch.float32, device=x.device)
            elif r - lag >= 0:
                tap = x_f[r - lag]
            else:
                tap = h_f[3 + r - lag] if h_f is not None else torch.zeros(c_dim, dtype=torch.float32, device=x.device)
            wt = sw_f[lag] + dyn_f[r, :, lag].repeat_interleave(128)
            acc = acc + tap * wt
        out[r] = ((acc + x_f[r]) * scale_f32).to(torch.bfloat16)
    return out


def ref_l2_clip(x: torch.Tensor, scale: torch.Tensor | None, thresh: float, tau: torch.Tensor | None = None) -> torch.Tensor:
    xf = x.float()
    if scale is not None:
        xf = xf * scale.float()
    norm = xf.pow(2).sum(dim=-1, keepdim=True).sqrt()
    coeff = torch.clamp(thresh / (norm + 1e-6), max=1.0)
    y = (xf * coeff).to(torch.bfloat16)
    if tau is not None:
        y = (y.float() * tau).to(torch.bfloat16)
    return y


def ref_v_rmsnorm(v: torch.Tensor) -> torch.Tensor:
    vf = v.float()
    rstd = torch.rsqrt(vf.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
    return vf * rstd


def ref_quant_fp8_dequant(x_f32: torch.Tensor, scale: float) -> torch.Tensor:
    q = torch.clamp(x_f32 / scale, -448.0, 448.0).to(torch.float8_e4m3fn)
    return q.float() * scale


def ref_rel_attention(
    q: torch.Tensor,         # [T_q, 24, 128] bf16
    k: torch.Tensor,         # [T_k, 8, 128] fp32 (already dequantized or scaled)
    v: torch.Tensor,         # [T_k, 8, 128] fp32
    rel_bias: torch.Tensor,  # [T_q, 24, 1024] bf16
    gate: torch.Tensor,      # [T_q, 24, 128] bf16
    sink: torch.Tensor,      # [T_q, 24] bf16
    past: int,
    window: int | None = None,
    sm_scale: float = 128.0**-0.5,
) -> torch.Tensor:
    out_dev = q.device
    t_q = q.shape[0]
    t_k = k.shape[0]
    # Run large N*M score matrices on CPU so kern's 71 GiB VMM KV pool doesn't OOM PyTorch.
    if t_q * t_k > 65536 and out_dev.type == "cuda":
        q, k, v, rel_bias, gate, sink = (
            q.cpu(), k.cpu(), v.cpu(), rel_bias.cpu(), gate.cpu(), sink.cpu()
        )
    dev = q.device
    q_f = q.float() * sm_scale
    k_rep = k.repeat_interleave(3, dim=1).float()
    v_rep = v.repeat_interleave(3, dim=1).float()
    scores = torch.einsum("qhd,khd->hqk", q_f, k_rep)  # [24, T_q, T_k]
    q_pos = torch.arange(past, past + t_q, device=dev)[:, None]
    k_pos = torch.arange(0, t_k, device=dev)[None, :]
    dist = q_pos - k_pos
    vis = dist >= 0
    if window is not None:
        vis = vis & (dist < window)
    in_ext = vis & (dist < 1024)
    safe_d = torch.where(in_ext, dist, torch.zeros_like(dist))
    b_slice = rel_bias.permute(1, 0, 2).float()
    bias_vals = torch.gather(b_slice, 2, safe_d.unsqueeze(0).expand(24, -1, -1))
    scores = scores + torch.where(in_ext.unsqueeze(0), bias_vals, torch.zeros_like(bias_vals))
    scores = torch.where(vis.unsqueeze(0), scores, torch.full_like(scores, float("-inf")))
    lse_nat = torch.logsumexp(scores, dim=-1)  # [24, T_q]
    probs = torch.softmax(scores, dim=-1)
    out = torch.einsum("hqk,khd->qhd", probs, v_rep)  # [T_q, 24, 128]
    gated = out * torch.sigmoid(gate.float()) * torch.sigmoid(lse_nat.t() - sink.float()).unsqueeze(-1)
    return gated.to(torch.bfloat16).contiguous().to(out_dev)


def _create_runtime(kern, manifest: dict):
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "model.json"
        p.write_text(json.dumps(manifest))
        return kern.Runtime(str(p), str(BUILD_DIR), [], 0)


def _region(host_mod, rt, m: dict, name: str) -> torch.Tensor:
    b = m["buffers"][name]
    if b["dtype"] == "fp8e4m3":
        ptr, nbytes = rt.region(name)
        shape = [host_mod.resolve(m, m["vars"][d]["max"] if d in m.get("vars", {}) else d) for d in b["shape"]]
        return torch.as_tensor(host_mod.Bytes(ptr, nbytes), device="cuda").view(torch.float8_e4m3fn).view(*shape)
    return host_mod.region(rt, m, name)


class TestM3ManifestCPU(unittest.TestCase):
    def test_seq_state_contract_and_dsl_consistency(self):
        self.assertEqual(am.SEQ_STATE_TAIL_PAD_BYTES, 8064)
        self.assertEqual(am.SEQ_STATE_BYTES, 90_005_376)
        self.assertEqual(am.SEQ_STATE_BYTES % 128, 0)

        dsl = am.load_dsl(BUILD_DIR)
        self.assertEqual(set(dsl["modules"]), set(am.ATTN_DSL_MODULES))
        dsl_json_path = BUILD_DIR / "dsl.json"
        self.assertTrue(dsl_json_path.exists(), "Missing build/sm_120/dsl.json")
        dsl_json = json.loads(dsl_json_path.read_text())

        for mod in am.ATTN_DSL_MODULES:
            cubin_sha = hashlib.sha256((BUILD_DIR / f"{mod}.cubin").read_bytes()).hexdigest()
            self.assertEqual(dsl["modules"][mod]["sha256"], cubin_sha)
            self.assertIn(mod, dsl_json["modules"])
            self.assertEqual(dsl_json["modules"][mod]["sha256"], cubin_sha)

        for mod_name, meta_name in (("sconv", "sconv_meta.json"), ("attn_prologue", "attn_prologue_meta.json")):
            meta = json.loads((BUILD_DIR / meta_name).read_text())
            cubin_sha = hashlib.sha256((BUILD_DIR / f"{mod_name}.cubin").read_bytes()).hexdigest()
            self.assertEqual(meta["sha256"], cubin_sha)
            for fn, info in meta["entries"].items():
                self.assertEqual(info["stack"], 0, f"{mod_name}:{fn} has non-zero stack")
                self.assertEqual(info["local"], 0, f"{mod_name}:{fn} has non-zero local")

        ops: dict = {}
        am.register_ops(ops, dsl)
        self.assertIn("m3_fill_f32", ops)
        self.assertEqual(ops["m3_fill_f32"]["impl"]["launches"][0]["module"], "attn_prologue")
        self.assertEqual(ops["m3_fill_f32"]["impl"]["launches"][0]["entry"], "kern_fill_f32")
        self.assertIn("m3_gate_sink", ops)
        self.assertEqual(ops["m3_gate_sink"]["impl"]["launches"][0]["module"], "attn_prologue")
        self.assertEqual(ops["m3_gate_sink"]["impl"]["launches"][0]["entry"], "kern_gate_sink_bf16")

        m = am.build_unit_manifest(BUILD_DIR)
        self.assertEqual(m["states"]["seq_state"]["bytes_per_seq"], 90_005_376)
        self.assertEqual(m["buffers"]["seq_state.lines"]["domain"]["stride"], 90_005_376)

    def test_kern_verify_with_emit_derive_calls(self):
        if not KERN_BIN.exists():
            self.skipTest(f"Missing {KERN_BIN}")
        m = am.build_unit_manifest(BUILD_DIR)
        self.assertIn("derive", m["programs"])
        self.assertTrue(m["programs"]["derive"].get("derive"))
        self.assertEqual(len(m["programs"]["derive"]["calls"]), 96)
        for c in m["programs"]["derive"]["calls"]:
            self.assertEqual(c["op"], "m3_fill_f32")
        self.assertNotIn("weight_prep", m["modules"])
        self.assertEqual(m["buffers"]["attn.v0.k_scale"]["kind"], "carry")
        self.assertEqual(m["buffers"]["attn.v5.v_scale"]["kind"], "carry")

        unembed_cubin = BUILD_DIR / "unembed_sample.cubin"
        if not unembed_cubin.exists():
            self.skipTest(f"Missing {unembed_cubin}")
        m["modules"]["unembed_sample"] = {
            "source": unembed_cubin.name,
            "sha256": hashlib.sha256(unembed_cubin.read_bytes()).hexdigest(),
        }
        m["buffers"]["_test.logits"] = {"dtype": "f32", "shape": ["seqs", 128], "kind": "input"}
        m["buffers"]["next_token"] = {
            "dtype": "i64",
            "shape": ["seqs"],
            "kind": "output",
            "fill": "tokens",
            "domain": {"min": 0, "max": 201023},
        }
        m["ops"]["_test_argmax"] = {
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
                        "args": [{"scratch": "pmax"}, {"scratch": "pidx"}, {"param": 1}, {"i32": 64}],
                    },
                ],
            },
        }
        for p in ("prefill", "decode", "round"):
            m["programs"][p]["calls"].append(
                {
                    "label": "test_next_token",
                    "op": "_test_argmax",
                    "args": [{"buf": "_test.logits"}, {"buf": "next_token"}, {"var": "seqs"}, {"i32": 128}],
                }
            )
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(m, f)
            p = f.name
        try:
            res = subprocess.run([str(KERN_BIN), "verify", p], capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"kern verify failed:\n{res.stderr}\n{res.stdout}")
        finally:
            Path(p).unlink(missing_ok=True)

    def test_calibrate_kv_mxfp8_safe_open_and_cli(self):
        # 1) _dequant_mxfp8 edge values: ue8m0 == 0 -> 2^-127, 115 -> 2^-12, 255 -> NaN
        q = torch.ones((1, 32), dtype=torch.float8_e4m3fn)
        s0 = torch.zeros((1, 1), dtype=torch.uint8)
        s115 = torch.full((1, 1), 115, dtype=torch.uint8)
        s255 = torch.full((1, 1), 255, dtype=torch.uint8)
        torch.testing.assert_close(ck._dequant_mxfp8(q, s0), torch.full((1, 32), 2.0**-127), rtol=0, atol=0)
        torch.testing.assert_close(ck._dequant_mxfp8(q, s115), torch.full((1, 32), 2.0**-12), rtol=0, atol=0)
        self.assertTrue(torch.isnan(ck._dequant_mxfp8(q, s255)).all())

        # 2) CLI --analytical-fallback & rejection of removed --mode kern_bf16
        with tempfile.TemporaryDirectory() as td:
            out_json = Path(td) / "kv.json"
            res_fb = subprocess.run(
                [sys.executable, str(ROOT / "scripts/calibrate_kv.py"), "--analytical-fallback", "--out", str(out_json)],
                capture_output=True,
                text=True,
            )
            self.assertEqual(res_fb.returncode, 0, res_fb.stderr)
            scales_fb = json.loads(out_json.read_text())
            self.assertEqual(len(scales_fb), 48)
            self.assertAlmostEqual(scales_fb["0"]["k"], ck.K_MAX_BOUND / ck.FP8_E4M3_MAX, places=12)
            self.assertAlmostEqual(scales_fb["47"]["v"], ck.V_MAX_BOUND / ck.FP8_E4M3_MAX, places=12)

            res_bad = subprocess.run(
                [sys.executable, str(ROOT / "scripts/calibrate_kv.py"), "--mode", "kern_bf16", "--out", str(out_json)],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(res_bad.returncode, 0)
            self.assertNotIn("NotImplementedError", res_bad.stderr)

        # 3) Synthetic CPU calibrate_torch_loop with safe_open skipping model.visual.*
        from safetensors.torch import save_file

        H = 32
        INTER = 32
        NUM_EXP = 4
        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td)
            cfg = {
                "text_config": {
                    "hidden_size": H,
                    "num_hidden_layers": 24,
                    "local_layer_ids": [i for i in range(24) if i not in (5, 11, 17, 23)],
                    "loop_config": {"core_start": 6, "core_end": 18, "num_passes": 3},
                    "n_layers_per_attn_res_block": 6,
                    "rms_norm_eps": 1e-6,
                    "qk_l2_clip_threshold": 22.627416997969522,
                    "rel_l2_clip_threshold": 8.0,
                    "log_scaling_n_floor": 8192,
                    "log_scaling_alpha": 0.1,
                    "mlp_gate_softcap": 4.0,
                    "mlp_up_softcap": 25.0,
                    "route_scale": 8.0,
                    "num_experts_per_tok": 2,
                }
            }
            (ckpt / "config.json").write_text(json.dumps(cfg))
            sd: dict[str, torch.Tensor] = {
                "model.visual.dummy": torch.ones((64, 64), dtype=torch.bfloat16),
                "model.llm.embed.weight": torch.randn((16, H), dtype=torch.bfloat16) * 0.1,
                "model.llm.embed_norm.weight": torch.ones((H,), dtype=torch.bfloat16),
                "model.llm.middle_cycle_cross_pass_queries": torch.randn((12, 2, H), dtype=torch.bfloat16) * 0.05,
                "model.llm.middle_cycle_cross_pass_kv_queries": torch.randn((12, 2, H), dtype=torch.bfloat16) * 0.05,
            }
            full_ids = {5, 11, 17, 23}
            for p in range(24):
                sd[f"model.llm.attn_res.mlp_queries.{p}"] = torch.randn((1, H), dtype=torch.bfloat16) * 0.05
                if p < 23:
                    nq = 3 if (p + 1) in full_ids else 1
                    sd[f"model.llm.attn_res.attn_queries.{p}"] = torch.randn((nq, H), dtype=torch.bfloat16) * 0.05
                pfx = f"model.llm.layers.{p}"
                if p in full_ids:
                    for sname in ("q_input_norm", "k_input_norm", "v_input_norm"):
                        sd[f"{pfx}.{sname}.weight"] = torch.ones((H,), dtype=torch.bfloat16)
                    for st in ("k", "v"):
                        sd[f"{pfx}.attn.{st}_sconv.static_w"] = torch.ones((4, 1024), dtype=torch.bfloat16) * 0.1
                        sd[f"{pfx}.attn.{st}_sconv.weight_proj"] = torch.zeros((32, H), dtype=torch.bfloat16)
                else:
                    sd[f"{pfx}.attn_norm.weight"] = torch.ones((H,), dtype=torch.bfloat16)
                    for st in ("k", "v"):
                        sd[f"{pfx}.attn.{st}_sconv.weight"] = torch.ones((1024, 4), dtype=torch.bfloat16) * 0.1
                if 6 <= p < 18:
                    for cname in ("attn", "k", "v", "mlp"):
                        sd[f"{pfx}.controllers.{cname}.down_weight"] = torch.zeros((4, 2 * H), dtype=torch.bfloat16)
                        sd[f"{pfx}.controllers.{cname}.gain_weight"] = torch.zeros((H, 4), dtype=torch.bfloat16)
                        sd[f"{pfx}.controllers.{cname}.forget_weight"] = torch.zeros((H, 4), dtype=torch.bfloat16)
                sd[f"{pfx}.mlp_norm.weight"] = torch.ones((H,), dtype=torch.bfloat16)
                sd[f"{pfx}.attn.wq_du.weight"] = torch.randn((3072, H), dtype=torch.bfloat16) * 0.05
                sd[f"{pfx}.attn.wk_dv.weight"] = torch.randn((1024, H), dtype=torch.bfloat16) * 0.05
                sd[f"{pfx}.attn.wv_dv.weight"] = torch.randn((1024, H), dtype=torch.bfloat16) * 0.05
                sd[f"{pfx}.attn.wr_du.weight"] = torch.randn((384, H), dtype=torch.bfloat16) * 0.05
                sd[f"{pfx}.attn.gate_attn.weight"] = torch.zeros((3072, H), dtype=torch.bfloat16)
                sd[f"{pfx}.attn.attn_sink_proj.weight"] = torch.zeros((24, H), dtype=torch.bfloat16)
                sd[f"{pfx}.attn.wo_ud.weight"] = torch.randn((H, 3072), dtype=torch.bfloat16) * 0.05
                sd[f"{pfx}.attn.q_l2_clip.scale"] = torch.ones((24, 128), dtype=torch.bfloat16)
                sd[f"{pfx}.attn.rel_l2_clip.scale"] = torch.ones((24, 16), dtype=torch.bfloat16)
                sd[f"{pfx}.attn.rel_logits_proj.proj"] = torch.zeros((16, 1024), dtype=torch.bfloat16)
                sd[f"{pfx}.attn_sconv.weight"] = torch.ones((H, 4), dtype=torch.bfloat16) * 0.1
                sd[f"{pfx}.mlp_sconv.weight"] = torch.ones((H, 4), dtype=torch.bfloat16) * 0.1
                sd[f"{pfx}.mlp.gate.weight"] = torch.randn((NUM_EXP, H), dtype=torch.bfloat16) * 0.05
                sd[f"{pfx}.mlp.gate.bias"] = torch.zeros((NUM_EXP,), dtype=torch.float32)
                if p < 20:
                    sd[f"{pfx}.mlp.experts.w13_weight"] = torch.zeros((NUM_EXP, 2 * INTER, H // 2), dtype=torch.uint8)
                    sd[f"{pfx}.mlp.experts.w13_weight_scale"] = torch.ones((NUM_EXP, 2 * INTER, H // 16), dtype=torch.uint8)
                    sd[f"{pfx}.mlp.experts.w13_weight_amax"] = torch.ones((NUM_EXP, 1, 1), dtype=torch.float32)
                    sd[f"{pfx}.mlp.experts.w13_weight_outer_scale_denominator"] = torch.tensor(1.0, dtype=torch.float32)
                    sd[f"{pfx}.mlp.experts.w2_weight"] = torch.zeros((NUM_EXP, H, INTER // 2), dtype=torch.uint8)
                    sd[f"{pfx}.mlp.experts.w2_weight_scale"] = torch.ones((NUM_EXP, H, INTER // 16), dtype=torch.uint8)
                    sd[f"{pfx}.mlp.experts.w2_weight_amax"] = torch.ones((NUM_EXP, 1, 1), dtype=torch.float32)
                    sd[f"{pfx}.mlp.experts.w2_weight_outer_scale_denominator"] = torch.tensor(1.0, dtype=torch.float32)
                else:
                    sd[f"{pfx}.mlp.experts.w13_weight"] = torch.ones((NUM_EXP, 2 * INTER, H), dtype=torch.uint8)
                    sd[f"{pfx}.mlp.experts.w13_weight_scale"] = torch.full((NUM_EXP, 2 * INTER, H // 32), 120, dtype=torch.uint8)
                    sd[f"{pfx}.mlp.experts.w2_weight"] = torch.ones((NUM_EXP, H, INTER), dtype=torch.uint8)
                    sd[f"{pfx}.mlp.experts.w2_weight_scale"] = torch.full((NUM_EXP, H, INTER // 32), 120, dtype=torch.uint8)
            save_file(sd, str(ckpt / "model-00001-of-00001.safetensors"))

            class _FakeTok:
                def encode(self, text, add_special_tokens=False):
                    return [1, 2, 3, 4]

            with mock.patch("transformers.AutoTokenizer.from_pretrained", return_value=_FakeTok()):
                scales, maxima = ck.calibrate_torch_loop(ckpt, headroom=1.25, device="cpu", max_tokens=4)
            self.assertEqual(len(scales), 48)
            for v in range(48):
                self.assertGreater(scales[str(v)]["k"], 0.0)
                self.assertGreater(scales[str(v)]["v"], 0.0)
                self.assertTrue(math.isfinite(maxima[str(v)]["k"]))
                self.assertTrue(math.isfinite(maxima[str(v)]["v"]))


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA GPU")
class TestM3AttnSconv(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kern, cls.host = _load_kern()
        cls.dsl = am.load_dsl(BUILD_DIR)
        torch.manual_seed(2026)
        cls.m = cls._build_full_test_manifest(cls, max_seqs=8, max_tokens=1600)
        cls.rt = _create_runtime(cls.kern, cls.m)
        cls.v = {name: _region(cls.host, cls.rt, cls.m, name) for name in cls.m["buffers"]}

    @classmethod
    def tearDownClass(cls):
        cls.v = {}
        if hasattr(cls, "rt"):
            del cls.rt
        gc.collect()
        torch.cuda.empty_cache()

    def _build_full_test_manifest(self, max_seqs: int = 8, max_tokens: int = 2048, page_columns: int = am.DEFAULT_PAGE_COLUMNS):
        modules: dict = {}
        states: dict = {}
        buffers: dict = {}
        ops: dict = {}
        am.register_modules(modules, BUILD_DIR, self.dsl)
        am.register_states(states)
        am.register_buffers(
            buffers, max_seqs=max_seqs, chunk=max_tokens, context=am.DEFAULT_CONTEXT,
            page_columns=page_columns, include_scales=False,
        )
        # In the unit-test manifest, verify and commit run as separate programs so tests can inspect
        # pre-commit rollback safety; mark staging buffers as "carry" so dataflow across programs is valid.
        for carry_buf in ("sconv.verify_raw", "attn.verify_swa_k", "attn.verify_swa_v"):
            buffers[carry_buf]["kind"] = "carry"
        am.register_ops(ops, self.dsl, max_seqs=max_seqs, page_columns=page_columns)

        # Test inputs/outputs/weights
        buffers["t.x_k"] = {"dtype": "bf16", "shape": ["tokens", 2304], "kind": "input"}
        buffers["t.x_v"] = {"dtype": "bf16", "shape": ["tokens", 2304], "kind": "input"}
        buffers["t.k_in"] = {"dtype": "bf16", "shape": ["tokens", 1024], "kind": "input"}
        buffers["t.v_in"] = {"dtype": "bf16", "shape": ["tokens", 1024], "kind": "input"}
        buffers["t.attn_in"] = {"dtype": "bf16", "shape": ["tokens", 2304], "kind": "input"}
        buffers["t.mlp_in"] = {"dtype": "bf16", "shape": ["tokens", 2304], "kind": "input"}
        buffers["t.res"] = {"dtype": "bf16", "shape": ["tokens", 2304], "kind": "input"}
        buffers["t.addend"] = {"dtype": "bf16", "shape": ["tokens", 2304], "kind": "input"}

        buffers["t.q_in"] = {"dtype": "bf16", "shape": ["tokens", 3072], "kind": "input"}
        buffers["t.r_in"] = {"dtype": "bf16", "shape": ["tokens", 384], "kind": "input"}
        buffers["t.gate"] = {"dtype": "bf16", "shape": ["tokens", 3072], "kind": "input"}
        buffers["t.sink"] = {"dtype": "bf16", "shape": ["tokens", 24], "kind": "input"}

        buffers["t.k_scale"] = {"dtype": "f32", "shape": [1], "kind": "input"}
        buffers["t.v_scale"] = {"dtype": "f32", "shape": [1], "kind": "input"}
        buffers["t.accepted"] = {"dtype": "i32", "shape": ["seqs"], "kind": "input"}

        buffers["t.w_k_static"] = {"dtype": "bf16", "shape": [1024, 4], "kind": "input"}
        buffers["t.w_v_static"] = {"dtype": "bf16", "shape": [1024, 4], "kind": "input"}
        buffers["t.w_k_dyn_st"] = {"dtype": "bf16", "shape": [4, 1024], "kind": "input"}
        buffers["t.w_k_dyn_pr"] = {"dtype": "bf16", "shape": [32, 2304], "kind": "input"}
        buffers["t.w_v_dyn_st"] = {"dtype": "bf16", "shape": [4, 1024], "kind": "input"}
        buffers["t.w_v_dyn_pr"] = {"dtype": "bf16", "shape": [32, 2304], "kind": "input"}
        buffers["t.w_attn"] = {"dtype": "bf16", "shape": [2304, 4], "kind": "input"}
        buffers["t.w_mlp"] = {"dtype": "bf16", "shape": [2304, 4], "kind": "input"}

        buffers["t.q_l2"] = {"dtype": "bf16", "shape": [24, 128], "kind": "input"}
        buffers["t.r_l2"] = {"dtype": "bf16", "shape": [24, 16], "kind": "input"}
        buffers["t.rel_proj"] = {"dtype": "bf16", "shape": [16, 1024], "kind": "input"}

        buffers["t.gs_attn_out"] = {"dtype": "f32", "shape": ["tokens", 24, 128], "kind": "input"}
        buffers["t.gs_lse"] = {"dtype": "f32", "shape": ["tokens", 24], "kind": "input"}
        buffers["t.gs_gate"] = {"dtype": "bf16", "shape": ["tokens", 4096], "kind": "input"}
        buffers["t.gs_sink"] = {"dtype": "bf16", "shape": ["tokens", 32], "kind": "input"}
        buffers["t.gs_y"] = {"dtype": "bf16", "shape": ["tokens", 3072], "kind": "output"}
        buffers["t.fill_k_scale"] = {"dtype": "f32", "shape": [1], "kind": "output"}
        buffers["t.fill_v_scale"] = {"dtype": "f32", "shape": [1], "kind": "output"}

        buffers["t.sconv_attn_out"] = {"dtype": "bf16", "shape": ["tokens", 2304], "kind": "output"}
        buffers["t.sconv_mlp_out"] = {"dtype": "bf16", "shape": ["tokens", 2304], "kind": "output"}
        buffers["t.attn_y"] = {"dtype": "bf16", "shape": ["tokens", 3072], "kind": "output"}
        buffers["t.q_clip_out"] = {"dtype": "bf16", "shape": ["tokens", 24, 128], "kind": "output"}
        buffers["t.rel_bias_out"] = {"dtype": "bf16", "shape": ["tokens", 24, 1024], "kind": "output"}
        buffers["t.swa_k_out"] = {"dtype": "fp8e4m3", "shape": ["tokens", 8, 128], "kind": "output"}
        buffers["t.swa_v_out"] = {"dtype": "fp8e4m3", "shape": ["tokens", 8, 128], "kind": "output"}

        b = am._b
        s = am._s
        i32 = am._i32
        i64 = am._i64
        f32 = am._f32

        v_local = 0
        v_full = 5
        v_tail = 47
        swa_off = am.SWA_TARGET_OFFSET + 0 * am.SWA_VISIT_BYTES
        swa39_off = am.SWA_TARGET_OFFSET + 39 * am.SWA_VISIT_BYTES
        swa_mtp1_off = am.SWA_MTP_OFFSET + 1 * am.SWA_VISIT_BYTES
        full_off = 0 * am.KV_LAYER_PAGE_BYTES

        # Register PDL-enabled variants of attn_prologue_full_{prefill,decode} to directly verify
        # griddepcontrol.wait ordering before tau_buf[row] load.
        for base_op in ("attn_prologue_full_prefill", "attn_prologue_full_decode"):
            op_copy = json.loads(json.dumps(ops[base_op]))
            op_copy["impl"]["launches"][0]["pdl"] = True
            ops[f"{base_op}_pdl"] = op_copy

        # Use deterministic unit scales in manifest; tests verify k_scale/v_scale via separate scaled programs too
        k_s_val = 0.05
        v_s_val = 0.025
        sm_scaled = (128.0**-0.5) * k_s_val

        programs = {
            # sconv local + dynamic tests
            "sconv_local_prefill": {
                "calls": [
                    {"op": "sconv_static_1024_prefill", "args": [b("t.k_in"), b("t.w_k_static"), s("seq_state"), b("seq_state.lines"), b("positions"), b("sconv.k_out"), i32(1024), {"var": "tokens"}, i64(v_local * am.SCONV_VISIT_BYTES + 0)]},
                    {"op": "sconv_prefill_commit_1024", "args": [b("t.k_in"), s("seq_state"), b("seq_state.lines"), i32(1024), {"var": "tokens"}, i64(v_local * am.SCONV_VISIT_BYTES + 0)]},
                    {"op": "sconv_static_2304_prefill", "args": [b("t.attn_in"), b("t.w_attn"), s("seq_state"), b("seq_state.lines"), b("positions"), b("t.sconv_attn_out"), i32(2304), {"var": "tokens"}, i64(v_local * am.SCONV_VISIT_BYTES + 12288)]},
                    {"op": "sconv_prefill_commit_2304", "args": [b("t.attn_in"), s("seq_state"), b("seq_state.lines"), i32(2304), {"var": "tokens"}, i64(v_local * am.SCONV_VISIT_BYTES + 12288)]},
                    {"op": "sconv_mlp_prefill", "args": [b("t.mlp_in"), b("t.w_mlp"), s("seq_state"), b("seq_state.lines"), b("positions"), b("t.sconv_mlp_out"), b("t.res"), b("t.addend"), i32(2304), {"var": "tokens"}, i64(v_local * am.SCONV_VISIT_BYTES + 26112), i32(1)]},
                    {"op": "sconv_prefill_commit_2304", "args": [b("t.mlp_in"), s("seq_state"), b("seq_state.lines"), i32(2304), {"var": "tokens"}, i64(v_local * am.SCONV_VISIT_BYTES + 26112)]},
                ]
            },
            "sconv_local_decode": {
                "calls": [
                    {"op": "sconv_static_1024_decode", "args": [b("t.k_in"), b("t.w_k_static"), s("seq_state"), b("seq_state.lines"), b("positions"), b("valid"), b("sconv.k_out"), i32(1024), i64(v_local * am.SCONV_VISIT_BYTES + 0)]},
                    {"op": "sconv_static_2304_decode", "args": [b("t.attn_in"), b("t.w_attn"), s("seq_state"), b("seq_state.lines"), b("positions"), b("valid"), b("t.sconv_attn_out"), i32(2304), i64(v_local * am.SCONV_VISIT_BYTES + 12288)]},
                    {"op": "sconv_mlp_decode", "args": [b("t.mlp_in"), b("t.w_mlp"), s("seq_state"), b("seq_state.lines"), b("positions"), b("valid"), b("t.sconv_mlp_out"), b("t.res"), b("t.addend"), i32(2304), i64(v_local * am.SCONV_VISIT_BYTES + 26112), i32(1)]},
                ]
            },
            "sconv_local_verify": {
                "calls": [
                    {"op": "sconv_static_1024_verify", "args": [b("t.k_in"), b("t.w_k_static"), s("seq_state"), b("seq_state.lines"), b("positions"), b("valid"), b("sconv.k_out"), b("sconv.verify_raw"), i32(1024), i64(v_local * am.SCONV_VISIT_BYTES + 0), i64(0), i32(0)]},
                    {"op": "sconv_static_2304_verify", "args": [b("t.attn_in"), b("t.w_attn"), s("seq_state"), b("seq_state.lines"), b("positions"), b("valid"), b("t.sconv_attn_out"), b("sconv.verify_raw"), i32(2304), i64(v_local * am.SCONV_VISIT_BYTES + 12288), i64(0), i32(2048)]},
                    {"op": "sconv_mlp_verify", "args": [b("t.mlp_in"), b("t.w_mlp"), s("seq_state"), b("seq_state.lines"), b("positions"), b("valid"), b("t.sconv_mlp_out"), b("sconv.verify_raw"), b("t.res"), b("t.addend"), i32(2304), i64(v_local * am.SCONV_VISIT_BYTES + 26112), i64(0), i32(4352), i32(0)]},
                ]
            },
            "sconv_dyn_prefill": {
                "calls": [
                    {"op": "m3_gemm_bf16", "args": [b("t.x_k"), b("t.w_k_dyn_pr"), b("sconv.dyn_k"), {"var": "tokens"}, i32(32), i32(2304)]},
                    {"op": "sconv_dynamic_prefill", "args": [b("t.k_in"), b("t.w_k_dyn_st"), b("sconv.dyn_k"), s("seq_state"), b("seq_state.lines"), b("positions"), b("sconv.k_out"), i32(1024), {"var": "tokens"}, i64(v_full * am.SCONV_VISIT_BYTES + 0)]},
                    {"op": "sconv_prefill_commit_1024", "args": [b("t.k_in"), s("seq_state"), b("seq_state.lines"), i32(1024), {"var": "tokens"}, i64(v_full * am.SCONV_VISIT_BYTES + 0)]},
                ]
            },
            "sconv_dyn_verify": {
                "calls": [
                    {"op": "m3_gemm_bf16", "args": [b("t.x_k"), b("t.w_k_dyn_pr"), b("sconv.dyn_k"), {"var": "tokens"}, i32(32), i32(2304)]},
                    {"op": "sconv_dynamic_verify", "args": [b("t.k_in"), b("t.w_k_dyn_st"), b("sconv.dyn_k"), s("seq_state"), b("seq_state.lines"), b("positions"), b("valid"), b("sconv.k_out"), b("sconv.verify_raw"), i32(1024), i64(v_full * am.SCONV_VISIT_BYTES + 0), i64(v_full * max_seqs * 8 * am.SCONV_RAW_ROW_ELEMS), i32(0)]},
                ]
            },
            "sconv_dyn_decode": {
                "calls": [
                    {"op": "m3_gemm_bf16", "args": [b("t.x_k"), b("t.w_k_dyn_pr"), b("sconv.dyn_k"), {"var": "seqs"}, i32(32), i32(2304)]},
                    {"op": "sconv_dynamic_decode", "args": [b("t.k_in"), b("t.w_k_dyn_st"), b("sconv.dyn_k"), s("seq_state"), b("seq_state.lines"), b("positions"), b("valid"), b("sconv.k_out"), i32(1024), i64(v_full * am.SCONV_VISIT_BYTES + 0)]},
                ]
            },
            "sconv_commit": {
                "calls": [
                    {"op": "sconv_spec_commit_all", "args": [b("sconv.verify_raw"), s("seq_state"), b("seq_state.lines"), b("t.accepted"), b("valid")]}
                ]
            },
            # Prologue + Attention (Full & SWA, scaled FP8)
            "full_prefill": {
                "calls": [
                    {"op": "attn_log_tau", "args": [b("positions"), b("attn.tau"), {"var": "tokens"}]},
                    {"op": "attn_prologue_full_prefill", "args": [b("t.q_in"), b("t.r_in"), b("t.k_in"), b("t.v_in"), b("t.q_l2"), b("t.r_l2"), b("t.rel_proj"), b("attn.tau"), b("t.k_scale"), b("t.v_scale"), b("positions"), b("pages"), s("kv", full_off), b("t.q_clip_out"), b("t.rel_bias_out"), i32(3072), i32(384), i32(1024), i32(1024), {"var": "tokens"}]},
                    {"op": "attn_full_prefill", "args": [b("t.q_clip_out"), s("kv", full_off), b("t.rel_bias_out"), b("t.gate"), b("t.sink"), b("pages"), b("lengths"), b("t.attn_y"), {"var": "tokens"}, i32(3072), i32(24), f32(sm_scaled), f32(v_s_val)]},
                ]
            },
            "full_decode": {
                "batch": {"groups": max_seqs, "rows": 1},
                "graph": True,
                "calls": [
                    {"op": "attn_log_tau_decode", "args": [b("positions"), b("attn.tau"), {"var": "seqs"}]},
                    {"op": "attn_prologue_full_decode", "args": [b("t.q_in"), b("t.r_in"), b("t.k_in"), b("t.v_in"), b("t.q_l2"), b("t.r_l2"), b("t.rel_proj"), b("attn.tau"), b("t.k_scale"), b("t.v_scale"), b("positions"), b("valid"), b("pages"), s("kv", full_off), b("t.q_clip_out"), b("t.rel_bias_out"), i32(3072), i32(384), i32(1024), i32(1024), i32(1)]},
                    {"op": "attn_full_decode", "args": [b("t.q_clip_out"), s("kv", full_off), b("t.rel_bias_out"), b("t.gate"), b("t.sink"), b("pages"), b("lengths"), b("t.attn_y"), {"var": "seqs"}, i32(3072), i32(24), f32(sm_scaled), f32(v_s_val)]},
                ],
            },
            "full_verify": {
                "batch": {"groups": max_seqs, "rows": 8},
                "graph": True,
                "calls": [
                    {"op": "attn_log_tau", "args": [b("positions"), b("attn.tau"), {"var": "tokens"}]},
                    {"op": "attn_prologue_full_verify", "args": [b("t.q_in"), b("t.r_in"), b("t.k_in"), b("t.v_in"), b("t.q_l2"), b("t.r_l2"), b("t.rel_proj"), b("attn.tau"), b("t.k_scale"), b("t.v_scale"), b("positions"), b("valid"), b("pages"), s("kv", full_off), b("t.q_clip_out"), b("t.rel_bias_out"), i32(3072), i32(384), i32(1024), i32(1024), i32(8)]},
                    {"op": "attn_full_verify", "args": [b("t.q_clip_out"), s("kv", full_off), b("t.rel_bias_out"), b("t.gate"), b("t.sink"), b("pages"), b("lengths"), b("t.attn_y"), {"var": "tokens"}, i32(3072), i32(24), f32(sm_scaled), f32(v_s_val)]},
                ],
            },
            "swa_prefill": {
                "calls": [
                    {"op": "attn_prologue_swa_prefill", "args": [b("t.q_in"), b("t.r_in"), b("t.k_in"), b("t.v_in"), b("t.q_l2"), b("t.r_l2"), b("t.rel_proj"), b("t.k_scale"), b("t.v_scale"), b("t.q_clip_out"), b("t.rel_bias_out"), b("t.swa_k_out"), b("t.swa_v_out"), i32(3072), i32(384), i32(1024), i32(1024), {"var": "tokens"}]},
                    {"op": "attn_swa_prefill", "args": [b("t.q_clip_out"), s("seq_state", swa_off), b("t.swa_k_out"), b("t.swa_v_out"), b("t.rel_bias_out"), b("t.gate"), b("t.sink"), b("seq_state.lines"), b("lengths"), b("t.attn_y"), {"var": "tokens"}, i32(3072), i32(24), f32(sm_scaled), f32(v_s_val)]},
                    {"op": "swa_ring_commit_prefill", "args": [b("t.swa_k_out"), b("t.swa_v_out"), b("positions"), b("seq_state.lines"), s("seq_state"), {"var": "tokens"}, i64(swa_off)]},
                ]
            },
            "swa_decode": {
                "calls": [
                    {"op": "attn_prologue_swa_decode", "args": [b("t.q_in"), b("t.r_in"), b("t.k_in"), b("t.v_in"), b("t.q_l2"), b("t.r_l2"), b("t.rel_proj"), b("t.k_scale"), b("t.v_scale"), b("positions"), b("valid"), b("seq_state.lines"), s("seq_state"), b("t.q_clip_out"), b("t.rel_bias_out"), i32(3072), i32(384), i32(1024), i32(1024), i64(swa_off)]},
                    {"op": "attn_swa_decode", "args": [b("t.q_clip_out"), s("seq_state", swa_off), b("t.rel_bias_out"), b("t.gate"), b("t.sink"), b("seq_state.lines"), b("lengths"), b("t.attn_y"), {"var": "seqs"}, i32(3072), i32(24), f32(sm_scaled), f32(v_s_val)]},
                ]
            },
            "swa_verify": {
                "calls": [
                    {"op": "attn_prologue_swa_verify", "args": [b("t.q_in"), b("t.r_in"), b("t.k_in"), b("t.v_in"), b("t.q_l2"), b("t.r_l2"), b("t.rel_proj"), b("t.k_scale"), b("t.v_scale"), b("valid"), b("seq_state.lines"), b("t.q_clip_out"), b("t.rel_bias_out"), b("attn.verify_swa_k"), b("attn.verify_swa_v"), i32(3072), i32(384), i32(1024), i32(1024)]},
                    {"op": "attn_swa_verify", "args": [b("t.q_clip_out"), s("seq_state", swa_off), b("attn.verify_swa_k"), b("attn.verify_swa_v"), b("t.rel_bias_out"), b("t.gate"), b("t.sink"), b("seq_state.lines"), b("lengths"), b("t.attn_y"), {"var": "tokens"}, i32(3072), i32(24), f32(sm_scaled), f32(v_s_val)]},
                ]
            },
            "swa_commit": {
                "calls": [
                    {"op": "spec_swa_commit", "args": [b("attn.verify_swa_k"), b("attn.verify_swa_v"), b("positions"), b("seq_state.lines"), b("t.accepted"), b("valid"), s("seq_state")]}
                ]
            },
            # Tail / max_seqs / canary boundary programs (v47, swa39, swa_mtp1)
            "v47_sconv_prefill": {
                "calls": [
                    {"op": "sconv_static_1024_prefill", "args": [b("t.k_in"), b("t.w_k_static"), s("seq_state"), b("seq_state.lines"), b("positions"), b("sconv.k_out"), i32(1024), {"var": "tokens"}, i64(v_tail * am.SCONV_VISIT_BYTES)]},
                    {"op": "sconv_prefill_commit_1024", "args": [b("t.k_in"), s("seq_state"), b("seq_state.lines"), i32(1024), {"var": "tokens"}, i64(v_tail * am.SCONV_VISIT_BYTES)]},
                ]
            },
            "v47_sconv_decode": {
                "calls": [
                    {"op": "sconv_static_1024_decode", "args": [b("t.k_in"), b("t.w_k_static"), s("seq_state"), b("seq_state.lines"), b("positions"), b("valid"), b("sconv.k_out"), i32(1024), i64(v_tail * am.SCONV_VISIT_BYTES)]},
                ]
            },
            "swa39_prefill": {
                "calls": [
                    {"op": "attn_prologue_swa_prefill", "args": [b("t.q_in"), b("t.r_in"), b("t.k_in"), b("t.v_in"), b("t.q_l2"), b("t.r_l2"), b("t.rel_proj"), b("t.k_scale"), b("t.v_scale"), b("t.q_clip_out"), b("t.rel_bias_out"), b("t.swa_k_out"), b("t.swa_v_out"), i32(3072), i32(384), i32(1024), i32(1024), {"var": "tokens"}]},
                    {"op": "attn_swa_prefill", "args": [b("t.q_clip_out"), s("seq_state", swa39_off), b("t.swa_k_out"), b("t.swa_v_out"), b("t.rel_bias_out"), b("t.gate"), b("t.sink"), b("seq_state.lines"), b("lengths"), b("t.attn_y"), {"var": "tokens"}, i32(3072), i32(24), f32(sm_scaled), f32(v_s_val)]},
                    {"op": "swa_ring_commit_prefill", "args": [b("t.swa_k_out"), b("t.swa_v_out"), b("positions"), b("seq_state.lines"), s("seq_state"), {"var": "tokens"}, i64(swa39_off)]},
                ]
            },
            "swa39_decode": {
                "calls": [
                    {"op": "attn_prologue_swa_decode", "args": [b("t.q_in"), b("t.r_in"), b("t.k_in"), b("t.v_in"), b("t.q_l2"), b("t.r_l2"), b("t.rel_proj"), b("t.k_scale"), b("t.v_scale"), b("positions"), b("valid"), b("seq_state.lines"), s("seq_state"), b("t.q_clip_out"), b("t.rel_bias_out"), i32(3072), i32(384), i32(1024), i32(1024), i64(swa39_off)]},
                    {"op": "attn_swa_decode", "args": [b("t.q_clip_out"), s("seq_state", swa39_off), b("t.rel_bias_out"), b("t.gate"), b("t.sink"), b("seq_state.lines"), b("lengths"), b("t.attn_y"), {"var": "seqs"}, i32(3072), i32(24), f32(sm_scaled), f32(v_s_val)]},
                ]
            },
            "swa_mtp1_prefill": {
                "calls": [
                    {"op": "attn_prologue_swa_prefill", "args": [b("t.q_in"), b("t.r_in"), b("t.k_in"), b("t.v_in"), b("t.q_l2"), b("t.r_l2"), b("t.rel_proj"), b("t.k_scale"), b("t.v_scale"), b("t.q_clip_out"), b("t.rel_bias_out"), b("t.swa_k_out"), b("t.swa_v_out"), i32(3072), i32(384), i32(1024), i32(1024), {"var": "tokens"}]},
                    {"op": "attn_swa_prefill", "args": [b("t.q_clip_out"), s("seq_state", swa_mtp1_off), b("t.swa_k_out"), b("t.swa_v_out"), b("t.rel_bias_out"), b("t.gate"), b("t.sink"), b("seq_state.lines"), b("lengths"), b("t.attn_y"), {"var": "tokens"}, i32(3072), i32(24), f32(sm_scaled), f32(v_s_val)]},
                    {"op": "swa_ring_commit_prefill", "args": [b("t.swa_k_out"), b("t.swa_v_out"), b("positions"), b("seq_state.lines"), s("seq_state"), {"var": "tokens"}, i64(swa_mtp1_off)]},
                ]
            },
            "swa_mtp1_decode": {
                "calls": [
                    {"op": "attn_prologue_swa_decode", "args": [b("t.q_in"), b("t.r_in"), b("t.k_in"), b("t.v_in"), b("t.q_l2"), b("t.r_l2"), b("t.rel_proj"), b("t.k_scale"), b("t.v_scale"), b("positions"), b("valid"), b("seq_state.lines"), s("seq_state"), b("t.q_clip_out"), b("t.rel_bias_out"), i32(3072), i32(384), i32(1024), i32(1024), i64(swa_mtp1_off)]},
                    {"op": "attn_swa_decode", "args": [b("t.q_clip_out"), s("seq_state", swa_mtp1_off), b("t.rel_bias_out"), b("t.gate"), b("t.sink"), b("seq_state.lines"), b("lengths"), b("t.attn_y"), {"var": "seqs"}, i32(3072), i32(24), f32(sm_scaled), f32(v_s_val)]},
                ]
            },
            "fill_f32_test": {
                "calls": [
                    {"op": "m3_fill_f32", "args": [b("t.fill_k_scale"), i32(1), f32(0.05)]},
                    {"op": "m3_fill_f32", "args": [b("t.fill_v_scale"), i32(1), f32(0.025)]},
                ]
            },
            "gate_sink_nat": {
                "calls": [
                    {"op": "m3_gate_sink", "args": [b("t.gs_attn_out"), b("t.gs_gate"), b("t.gs_lse"), b("t.gs_sink"), b("t.gs_y"), i32(4096), i32(32), i32(0)]},
                ]
            },
            "gate_sink_log2": {
                "calls": [
                    {"op": "m3_gate_sink", "args": [b("t.gs_attn_out"), b("t.gs_gate"), b("t.gs_lse"), b("t.gs_sink"), b("t.gs_y"), i32(4096), i32(32), i32(1)]},
                ]
            },
            "full_prefill_pdl": {
                "calls": [
                    {"op": "attn_log_tau", "args": [b("positions"), b("attn.tau"), {"var": "tokens"}]},
                    {"op": "attn_prologue_full_prefill_pdl", "args": [b("t.q_in"), b("t.r_in"), b("t.k_in"), b("t.v_in"), b("t.q_l2"), b("t.r_l2"), b("t.rel_proj"), b("attn.tau"), b("t.k_scale"), b("t.v_scale"), b("positions"), b("pages"), s("kv", full_off), b("t.q_clip_out"), b("t.rel_bias_out"), i32(3072), i32(384), i32(1024), i32(1024), {"var": "tokens"}]},
                ]
            },
            "full_decode_pdl": {
                "batch": {"groups": max_seqs, "rows": 1},
                "graph": True,
                "calls": [
                    {"op": "attn_log_tau_decode", "args": [b("positions"), b("attn.tau"), {"var": "seqs"}]},
                    {"op": "attn_prologue_full_decode_pdl", "args": [b("t.q_in"), b("t.r_in"), b("t.k_in"), b("t.v_in"), b("t.q_l2"), b("t.r_l2"), b("t.rel_proj"), b("attn.tau"), b("t.k_scale"), b("t.v_scale"), b("positions"), b("valid"), b("pages"), s("kv", full_off), b("t.q_clip_out"), b("t.rel_bias_out"), i32(3072), i32(384), i32(1024), i32(1024), i32(1)]},
                ],
            },
        }

        manifest = {
            "schema_version": 7,
            "model": "m3-unit-test",
            "vars": {
                "tokens": {"max": max_tokens, "axis": "rows"},
                "seqs": {"max": max_seqs, "axis": "groups"},
                "ctx": {"max": am.DEFAULT_CONTEXT},
            },
            "states": states,
            "buffers": buffers,
            "modules": modules,
            "ops": ops,
            "programs": programs,
        }
        return am.prune_unused(manifest)

    def test_sconv_static_dynamic_and_rollback(self):
        rt, v = self.rt, self.v
        if True:
            # Weights
            w_k = torch.randn((1024, 4), dtype=torch.bfloat16, device="cuda") * 0.25
            w_attn = torch.randn((2304, 4), dtype=torch.bfloat16, device="cuda") * 0.25
            w_mlp = torch.randn((2304, 4), dtype=torch.bfloat16, device="cuda") * 0.25
            w_dyn_st = torch.randn((4, 1024), dtype=torch.bfloat16, device="cuda") * 0.25
            w_dyn_pr = torch.randn((32, 2304), dtype=torch.bfloat16, device="cuda") * 0.05
            v["t.w_k_static"].copy_(w_k)
            v["t.w_attn"].copy_(w_attn)
            v["t.w_mlp"].copy_(w_mlp)
            v["t.w_k_dyn_st"].copy_(w_dyn_st)
            v["t.w_k_dyn_pr"].copy_(w_dyn_pr)

            # 1. Prefill T=5 (exercises pos=0,1,2 zero-padding + pos=3,4 intra-batch + commit last 3 rows)
            T = 5
            k_pf = torch.randn((T, 1024), dtype=torch.bfloat16, device="cuda")
            a_pf = torch.randn((T, 2304), dtype=torch.bfloat16, device="cuda")
            m_pf = torch.randn((T, 2304), dtype=torch.bfloat16, device="cuda")
            res_pf = torch.randn((T, 2304), dtype=torch.bfloat16, device="cuda")
            add_pf = torch.randn((T, 2304), dtype=torch.bfloat16, device="cuda")
            xk_pf = torch.randn((T, 2304), dtype=torch.bfloat16, device="cuda")

            v["seq_state.lines"].fill_(0)
            v["seq_state.lines"][0, 0] = 1
            v["positions"][:T].copy_(torch.arange(T, dtype=torch.int64, device="cuda"))
            v["valid"][:T].fill_(1)
            v["t.k_in"][:T].copy_(k_pf)
            v["t.attn_in"][:T].copy_(a_pf)
            v["t.mlp_in"][:T].copy_(m_pf)
            v["t.res"][:T].copy_(res_pf)
            v["t.addend"][:T].copy_(add_pf)
            v["t.x_k"][:T].copy_(xk_pf)

            rt.run("sconv_local_prefill", {"tokens": T})
            pos_pf = torch.arange(T, dtype=torch.int64, device="cuda")
            torch.testing.assert_close(v["sconv.k_out"][:T], ref_sconv_static(k_pf, w_k, None, pos_pf), rtol=0, atol=0)
            torch.testing.assert_close(v["t.sconv_attn_out"][:T], ref_sconv_static(a_pf, w_attn, None, pos_pf), rtol=0, atol=0)
            torch.testing.assert_close(v["t.sconv_mlp_out"][:T], ref_sconv_static(m_pf, w_mlp, None, pos_pf, add_pf, res_pf), rtol=0, atol=0)

            rt.run("sconv_dyn_prefill", {"tokens": T})
            dyn_ref = (xk_pf.float() @ w_dyn_pr.float().t()).to(torch.bfloat16)
            torch.testing.assert_close(v["sconv.k_out"][:T], ref_sconv_dynamic(k_pf, w_dyn_st, dyn_ref, None, pos_pf), rtol=2e-2, atol=2e-2)

            # 2. Decode batch (seqs=4: seq0=line1 active, seq1=line2 active @ pos=0, seq2=line0 null, seq3=line3 valid=0)
            k_dec = torch.randn((4, 1024), dtype=torch.bfloat16, device="cuda")
            a_dec = torch.randn((4, 2304), dtype=torch.bfloat16, device="cuda")
            m_dec = torch.randn((4, 2304), dtype=torch.bfloat16, device="cuda")
            res_dec = torch.randn((4, 2304), dtype=torch.bfloat16, device="cuda")
            add_dec = torch.randn((4, 2304), dtype=torch.bfloat16, device="cuda")
            xk_dec = torch.randn((4, 2304), dtype=torch.bfloat16, device="cuda")

            v["seq_state.lines"][0, :4].copy_(torch.tensor([1, 2, 0, 3], dtype=torch.int32, device="cuda"))
            v["positions"][:4].copy_(torch.tensor([5, 0, 5, 5], dtype=torch.int64, device="cuda"))
            v["valid"][:4].copy_(torch.tensor([1, 1, 1, 0], dtype=torch.int32, device="cuda"))
            v["t.k_in"][:4].copy_(k_dec)
            v["t.attn_in"][:4].copy_(a_dec)
            v["t.mlp_in"][:4].copy_(m_dec)
            v["t.res"][:4].copy_(res_dec)
            v["t.addend"][:4].copy_(add_dec)
            v["t.x_k"][:4].copy_(xk_dec)

            rt.run("sconv_local_decode", {"seqs": 4})
            # seq0 uses history k_pf[2:5], a_pf[2:5], m_pf[2:5]
            exp_k0 = ref_sconv_static(k_dec[0:1], w_k, k_pf[2:5], torch.tensor([5], device="cuda"))[0]
            exp_k1 = ref_sconv_static(k_dec[1:2], w_k, None, torch.tensor([0], device="cuda"))[0]
            torch.testing.assert_close(v["sconv.k_out"][0], exp_k0, rtol=0, atol=0)
            torch.testing.assert_close(v["sconv.k_out"][1], exp_k1, rtol=0, atol=0)
            torch.testing.assert_close(v["sconv.k_out"][2], torch.zeros_like(exp_k0), rtol=0, atol=0)
            torch.testing.assert_close(v["sconv.k_out"][3], torch.zeros_like(exp_k0), rtol=0, atol=0)

            rt.run("sconv_dyn_decode", {"seqs": 4, "tokens": 4})
            exp_dyn0 = ref_sconv_dynamic(k_dec[0:1], w_dyn_st, (xk_dec[0:1].float() @ w_dyn_pr.float().t()).to(torch.bfloat16), k_pf[2:5], torch.tensor([5], device="cuda"))[0]
            torch.testing.assert_close(v["sconv.k_out"][0], exp_dyn0, rtol=2e-2, atol=2e-2)

            # 3. Speculative verify (read-only!) + rollback + commit_all
            # seq0 history is now [k_pf[3], k_pf[4], k_dec[0]] at pos=6..13; seq1 history is [0, 0, k_dec[1]] at pos=1..8
            k_ver = torch.randn((32, 1024), dtype=torch.bfloat16, device="cuda")
            a_ver = torch.randn((32, 2304), dtype=torch.bfloat16, device="cuda")
            m_ver = torch.randn((32, 2304), dtype=torch.bfloat16, device="cuda")
            res_ver = torch.randn((32, 2304), dtype=torch.bfloat16, device="cuda")
            add_ver = torch.randn((32, 2304), dtype=torch.bfloat16, device="cuda")
            xk_ver = torch.randn((32, 2304), dtype=torch.bfloat16, device="cuda")

            pos_ver = torch.cat([
                torch.arange(6, 14, dtype=torch.int64, device="cuda"),
                torch.arange(1, 9, dtype=torch.int64, device="cuda"),
                torch.arange(6, 14, dtype=torch.int64, device="cuda"),
                torch.arange(6, 14, dtype=torch.int64, device="cuda"),
            ])
            valid_ver = torch.cat([
                torch.ones(8, dtype=torch.int32, device="cuda"),
                torch.ones(8, dtype=torch.int32, device="cuda"),
                torch.ones(8, dtype=torch.int32, device="cuda"),  # seq2 has line=0
                torch.zeros(8, dtype=torch.int32, device="cuda"), # seq3 has valid=0
            ])
            v["positions"][:32].copy_(pos_ver)
            v["valid"][:32].copy_(valid_ver)
            v["t.k_in"][:32].copy_(k_ver)
            v["t.attn_in"][:32].copy_(a_ver)
            v["t.mlp_in"][:32].copy_(m_ver)
            v["t.res"][:32].copy_(res_ver)
            v["t.addend"][:32].copy_(add_ver)
            v["t.x_k"][:32].copy_(xk_ver)

            rt.run("sconv_local_verify", {"tokens": 32})
            h0 = torch.stack([k_pf[3], k_pf[4], k_dec[0]])
            h1 = torch.stack([torch.zeros(1024, dtype=torch.bfloat16, device="cuda"),
                              torch.zeros(1024, dtype=torch.bfloat16, device="cuda"), k_dec[1]])
            torch.testing.assert_close(v["sconv.k_out"][:8], ref_sconv_static(k_ver[:8], w_k, h0, pos_ver[:8]), rtol=0, atol=0)
            torch.testing.assert_close(v["sconv.k_out"][8:16], ref_sconv_static(k_ver[8:16], w_k, h1, pos_ver[8:16]), rtol=0, atol=0)
            torch.testing.assert_close(v["sconv.k_out"][16:32], torch.zeros((16, 1024), dtype=torch.bfloat16, device="cuda"), rtol=0, atol=0)
            # Check mlp epilogue with has_res=0
            hm0 = torch.stack([m_pf[3], m_pf[4], m_dec[0]])
            torch.testing.assert_close(
                v["t.sconv_mlp_out"][:8],
                ref_sconv_static(m_ver[:8], w_mlp, hm0, pos_ver[:8], add_ver[:8], None),
                rtol=0, atol=0,
            )

            rt.run("sconv_dyn_verify", {"tokens": 32})
            torch.testing.assert_close(
                v["sconv.k_out"][:8],
                ref_sconv_dynamic(k_ver[:8], w_dyn_st, (xk_ver[:8].float() @ w_dyn_pr.float().t()).to(torch.bfloat16), h0, pos_ver[:8]),
                rtol=2e-2, atol=2e-2,
            )

            # Verify read-only: re-running verify produces identical result (history untouched!)
            snap = v["sconv.k_out"][:16].clone()
            rt.run("sconv_dyn_verify", {"tokens": 32})
            torch.testing.assert_close(v["sconv.k_out"][:16], snap, rtol=0, atol=0)

            # Now commit accepted = [1, 4, 8, 2]:
            # - seq0 accepted=1 -> new history [k_pf[4], k_dec[0], k_ver[0]] (next pos=7)
            # - seq1 accepted=4 -> new history [k_ver[9], k_ver[10], k_ver[11]] (next pos=5)
            v["t.accepted"][:4].copy_(torch.tensor([1, 4, 8, 2], dtype=torch.int32, device="cuda"))
            rt.run("sconv_commit", {"seqs": 4})

            k_after = torch.randn((4, 1024), dtype=torch.bfloat16, device="cuda")
            v["t.k_in"][:4].copy_(k_after)
            v["positions"][:4].copy_(torch.tensor([7, 5, 14, 14], dtype=torch.int64, device="cuda"))
            v["valid"][:4].copy_(torch.tensor([1, 1, 1, 0], dtype=torch.int32, device="cuda"))
            rt.run("sconv_local_decode", {"seqs": 4})

            exp_after0 = ref_sconv_static(k_after[0:1], w_k, torch.stack([k_pf[4], k_dec[0], k_ver[0]]), torch.tensor([7], device="cuda"))[0]
            exp_after1 = ref_sconv_static(k_after[1:2], w_k, k_ver[9:12], torch.tensor([5], device="cuda"))[0]
            torch.testing.assert_close(v["sconv.k_out"][0], exp_after0, rtol=0, atol=0)
            torch.testing.assert_close(v["sconv.k_out"][1], exp_after1, rtol=0, atol=0)

    def test_attn_prologue_clip_tau_rel_and_fp8_kv(self):
        rt, v = self.rt, self.v
        if True:
            T = 4
            # Large values so L2 clipping & RMSNorm saturate non-trivially, and positions spanning tau=1 & tau>1
            pos = torch.tensor([0, 8191, 16384, 1048575], dtype=torch.int64, device="cuda")
            q_in = torch.randn((T, 3072), dtype=torch.bfloat16, device="cuda") * 4.0
            r_in = torch.randn((T, 384), dtype=torch.bfloat16, device="cuda") * 4.0
            k_in = torch.randn((T, 1024), dtype=torch.bfloat16, device="cuda") * 4.0
            v_in = torch.randn((T, 1024), dtype=torch.bfloat16, device="cuda") * 4.0
            q_l2 = torch.randn((24, 128), dtype=torch.bfloat16, device="cuda") + 1.0
            r_l2 = torch.randn((24, 16), dtype=torch.bfloat16, device="cuda") + 1.0
            rel_proj = torch.randn((16, 1024), dtype=torch.bfloat16, device="cuda") * 0.25

            v["positions"][:T].copy_(pos)
            v["lengths"][0] = T
            v["pages"][0, :16385].copy_(torch.arange(16385, dtype=torch.int32, device="cuda") % 4)
            v["seq_state.lines"].fill_(0)
            v["seq_state.lines"][0, 0] = 1
            v["t.k_scale"].fill_(0.05)
            v["t.v_scale"].fill_(0.025)
            v["t.q_in"][:T].copy_(q_in)
            v["t.r_in"][:T].copy_(r_in)
            v["t.k_in"][:T].copy_(k_in)
            v["t.v_in"][:T].copy_(v_in)
            v["t.q_l2"].copy_(q_l2)
            v["t.r_l2"].copy_(r_l2)
            v["t.rel_proj"].copy_(rel_proj)
            v["t.gate"][:T].zero_()
            v["t.sink"][:T].zero_()

            # Full visit -> HAS_TAU = True
            rt.run("full_prefill", {"tokens": T})
            tau = 1.0 + 0.1 * torch.log(torch.clamp((pos.float() + 1.0) / 8192.0, min=1.0))
            tau_3 = tau.view(T, 1, 1)
            q_ref_full = ref_l2_clip(q_in.view(T, 24, 128), q_l2, 22.627416997969522, tau_3)
            r_ref_full = ref_l2_clip(r_in.view(T, 24, 16), r_l2, 8.0, tau_3)
            rel_ref_full = (r_ref_full.float() @ rel_proj.float()).to(torch.bfloat16)

            torch.testing.assert_close(v["t.q_clip_out"][:T], q_ref_full, rtol=1e-2, atol=2e-2)
            torch.testing.assert_close(v["t.rel_bias_out"][:T], rel_ref_full, rtol=2e-2, atol=5e-2)

            # SWA visit -> HAS_TAU = False, outputs swa_k / swa_v FP8
            rt.run("swa_prefill", {"tokens": T})
            q_ref_swa = ref_l2_clip(q_in.view(T, 24, 128), q_l2, 22.627416997969522, None)
            r_ref_swa = ref_l2_clip(r_in.view(T, 24, 16), r_l2, 8.0, None)
            rel_ref_swa = (r_ref_swa.float() @ rel_proj.float()).to(torch.bfloat16)

            torch.testing.assert_close(v["t.q_clip_out"][:T], q_ref_swa, rtol=1e-2, atol=2e-2)
            torch.testing.assert_close(v["t.rel_bias_out"][:T], rel_ref_swa, rtol=2e-2, atol=5e-2)

            # Check FP8 K/V quantization outputs via bytes view
            k_ref_f32 = ref_l2_clip(k_in.view(T, 8, 128), None, 22.627416997969522, None).float()
            # Recompute exact unrounded k_clipped in fp32 for FP8 check
            kf = k_in.view(T, 8, 128).float()
            kn = kf.pow(2).sum(dim=-1, keepdim=True).sqrt()
            k_clipped_f32 = kf * torch.clamp(22.627416997969522 / (kn + 1e-6), max=1.0)
            v_normed_f32 = ref_v_rmsnorm(v_in.view(T, 8, 128))
            k_q_ref = torch.clamp(k_clipped_f32 / 0.05, -448.0, 448.0).to(torch.float8_e4m3fn)
            v_q_ref = torch.clamp(v_normed_f32 / 0.025, -448.0, 448.0).to(torch.float8_e4m3fn)

            # Note: kern.host.region maps fp8e4m3 -> uint8; reinterpret as float8_e4m3fn
            swa_k_got = v["t.swa_k_out"][:T].view(torch.float8_e4m3fn).float()
            swa_v_got = v["t.swa_v_out"][:T].view(torch.float8_e4m3fn).float()
            torch.testing.assert_close(swa_k_got, k_q_ref.float(), rtol=0.1, atol=0.5)
            torch.testing.assert_close(swa_v_got, v_q_ref.float(), rtol=0.1, atol=0.5)

    def test_rel_attention_prefill_parity(self):
        rt, v = self.rt, self.v
        if True:
            v["t.k_scale"].fill_(0.05)
            v["t.v_scale"].fill_(0.025)
            v["t.q_l2"].fill_(1.0)
            v["t.r_l2"].fill_(1.0)
            v["t.rel_proj"].normal_(mean=0.0, std=0.15)
            v["seq_state.lines"].fill_(0)
            v["seq_state.lines"][0, 0] = 1

            # Test both T=17 (<=64 -> split-KV=16 + combine) and T=129 (>64 -> single-split) for Full,
            # and T=1500 (>1024 -> tests SWA ring non-overwrite!) for SWA.
            for kind, T in (("full", 17), ("full", 129), ("swa", 17), ("swa", 1500)):
                with self.subTest(kind=kind, T=T):
                    q_in = torch.randn((T, 3072), dtype=torch.bfloat16, device="cuda") * 0.5
                    r_in = torch.randn((T, 384), dtype=torch.bfloat16, device="cuda") * 0.5
                    k_in = torch.randn((T, 1024), dtype=torch.bfloat16, device="cuda") * 0.5
                    v_in = torch.randn((T, 1024), dtype=torch.bfloat16, device="cuda") * 0.5
                    gate = torch.randn((T, 3072), dtype=torch.bfloat16, device="cuda")
                    sink = torch.randn((T, 24), dtype=torch.bfloat16, device="cuda")

                    v["positions"][:T].copy_(torch.arange(T, dtype=torch.int64, device="cuda"))
                    v["lengths"][0] = T
                    num_p = (T + 63) // 64
                    v["pages"][0, :num_p].copy_(torch.arange(num_p, dtype=torch.int32, device="cuda"))
                    v["t.q_in"][:T].copy_(q_in)
                    v["t.r_in"][:T].copy_(r_in)
                    v["t.k_in"][:T].copy_(k_in)
                    v["t.v_in"][:T].copy_(v_in)
                    v["t.gate"][:T].copy_(gate)
                    v["t.sink"][:T].copy_(sink)

                    rt.run(f"{kind}_prefill", {"tokens": T})

                    # Compute reference using the exact prologue-quantized Q, RelBias, K, V
                    q_use = v["t.q_clip_out"][:T]
                    rel_use = v["t.rel_bias_out"][:T]
                    kf = k_in.view(T, 8, 128).float()
                    kn = kf.pow(2).sum(dim=-1, keepdim=True).sqrt()
                    k_clip = kf * torch.clamp(22.627416997969522 / (kn + 1e-6), max=1.0)
                    v_norm = ref_v_rmsnorm(v_in.view(T, 8, 128))
                    k_deq = ref_quant_fp8_dequant(k_clip, 0.05)
                    v_deq = ref_quant_fp8_dequant(v_norm, 0.025)

                    y_ref = ref_rel_attention(
                        q_use, k_deq, v_deq, rel_use, gate.view(T, 24, 128), sink,
                        past=0, window=(1024 if kind == "swa" else None), sm_scale=128.0**-0.5,
                    ).view(T, 3072)
                    torch.testing.assert_close(v["t.attn_y"][:T], y_ref, rtol=3e-2, atol=3e-2)

    def test_split_kv_decode_and_1m_simulation(self):
        rt, v = self.rt, self.v
        if True:
            v["t.k_scale"].fill_(0.05)
            v["t.v_scale"].fill_(0.025)
            v["t.q_l2"].fill_(1.0)
            v["t.r_l2"].fill_(1.0)
            v["t.rel_proj"].normal_(mean=0.0, std=0.1)

            # Populate 1 page (64 identical-pattern tokens) via prefill at positions 0..63 on page 0
            P = 64
            k_base = torch.randn((P, 1024), dtype=torch.bfloat16, device="cuda") * 0.25
            v_base = torch.randn((P, 1024), dtype=torch.bfloat16, device="cuda") * 0.25
            v["positions"][:P].copy_(torch.arange(P, dtype=torch.int64, device="cuda"))
            v["lengths"][0] = P
            v["pages"][0, 0] = 0
            v["t.q_in"][:P].zero_()
            v["t.r_in"][:P].zero_()
            v["t.k_in"][:P].copy_(k_base)
            v["t.v_in"][:P].copy_(v_base)
            v["t.gate"][:P].zero_()
            v["t.sink"][:P].zero_()
            rt.run("full_prefill", {"tokens": P})

            # 1) Short seqlen_k = 3 (tests empty splits in NUM_SPLITS=64 decode1!)
            v["pages"][0, 0] = 0
            v["positions"][0] = 2
            v["valid"][0] = 1
            v["lengths"][0] = 3
            q_d = torch.randn((1, 3072), dtype=torch.bfloat16, device="cuda") * 0.5
            r_d = torch.randn((1, 384), dtype=torch.bfloat16, device="cuda") * 0.5
            gate_d = torch.randn((1, 3072), dtype=torch.bfloat16, device="cuda")
            sink_d = torch.randn((1, 24), dtype=torch.bfloat16, device="cuda")
            v["t.q_in"][:1].copy_(q_d)
            v["t.r_in"][:1].copy_(r_d)
            v["t.k_in"][:1].copy_(k_base[2:3])
            v["t.v_in"][:1].copy_(v_base[2:3])
            v["t.gate"][:1].copy_(gate_d)
            v["t.sink"][:1].copy_(sink_d)
            rt.run("full_decode", {"seqs": 1})
            self.assertFalse(torch.isnan(v["t.attn_y"][:1]).any())

            # Compare seqlen_k = 3 against exact reference
            kf = k_base[:3].view(3, 8, 128).float()
            kn = kf.pow(2).sum(dim=-1, keepdim=True).sqrt()
            k_clip = kf * torch.clamp(22.627416997969522 / (kn + 1e-6), max=1.0)
            k_deq3 = ref_quant_fp8_dequant(k_clip, 0.05)
            v_deq3 = ref_quant_fp8_dequant(ref_v_rmsnorm(v_base[:3].view(3, 8, 128)), 0.025)
            y_ref3 = ref_rel_attention(
                v["t.q_clip_out"][:1], k_deq3, v_deq3, v["t.rel_bias_out"][:1],
                gate_d.view(1, 24, 128), sink_d, past=2,
            ).view(1, 3072)
            torch.testing.assert_close(v["t.attn_y"][:1], y_ref3, rtol=3e-2, atol=3e-2)

            # 2) Simulated 1M context decode (seqlen_k = 1,048,576, 16384 aliased pages -> page 0!)
            # With identical pages 0..16383 and uniform rel_proj=0, exact attention over 1M identical 64-token pages
            # equals attention over 1 single 64-token page!
            v["t.rel_proj"].zero_()
            v["pages"][0, :16384].fill_(0)
            v["positions"][0] = 1_048_575
            v["lengths"][0] = 1_048_576
            v["valid"][0] = 1
            v["t.k_in"][0].copy_(k_base[63])
            v["t.v_in"][0].copy_(v_base[63])
            rt.run("full_decode", {"seqs": 1})
            self.assertFalse(torch.isnan(v["t.attn_y"][:1]).any())
            y_1m_s1 = v["t.attn_y"][0].clone()

            # Also verify seqs=2 (NUM_SPLITS=32) at 131,072 and 1,048,576!
            v["pages"][1, :16384].fill_(0)
            v["positions"][:2].copy_(torch.tensor([1_048_575, 131_071], dtype=torch.int64, device="cuda"))
            v["lengths"][:2].copy_(torch.tensor([1_048_576, 131_072], dtype=torch.int32, device="cuda"))
            v["valid"][:2].fill_(1)
            v["t.q_in"][1].copy_(q_d[0])
            v["t.r_in"][1].copy_(r_d[0])
            v["t.k_in"][1].copy_(k_base[63])
            v["t.v_in"][1].copy_(v_base[63])
            v["t.gate"][1].copy_(gate_d[0])
            v["t.sink"][1].copy_(sink_d[0])
            rt.run("full_decode", {"seqs": 2})
            self.assertFalse(torch.isnan(v["t.attn_y"][:2]).any())
            torch.testing.assert_close(v["t.attn_y"][0], y_1m_s1, rtol=1e-2, atol=1e-2)

    def test_verify_causal_mask_and_swa_rollback(self):
        rt, v = self.rt, self.v
        if True:
            v["t.k_scale"].fill_(0.05)
            v["t.v_scale"].fill_(0.025)
            v["t.q_l2"].fill_(1.0)
            v["t.r_l2"].fill_(1.0)
            v["t.rel_proj"].normal_(mean=0.0, std=0.1)

            # Prefill seq0 (line=1) and seq1 (line=2) with 4 tokens each (past=4)
            k_pf = torch.randn((4, 1024), dtype=torch.bfloat16, device="cuda") * 0.3
            v_pf = torch.randn((4, 1024), dtype=torch.bfloat16, device="cuda") * 0.3
            for seq, line in ((0, 1), (1, 2)):
                v["seq_state.lines"].fill_(0)
                v["seq_state.lines"][0, 0] = line
                v["positions"][:4].copy_(torch.arange(4, dtype=torch.int64, device="cuda"))
                v["lengths"][0] = 4
                v["t.q_in"][:4].zero_()
                v["t.r_in"][:4].zero_()
                v["t.k_in"][:4].copy_(k_pf)
                v["t.v_in"][:4].copy_(v_pf)
                v["t.gate"][:4].zero_()
                v["t.sink"][:4].zero_()
                rt.run("swa_prefill", {"tokens": 4})

            # Verify 4 seqs x 8 rows = 32 tokens:
            # seq0: line=1, valid=1, accepted=1
            # seq1: line=2, valid=1, accepted=4
            # seq2: line=0 (null!), valid=1, accepted=8 -> outputs must be 0, no state writes
            # seq3: line=3, valid=0 (pad!), accepted=2 -> outputs must be 0, no state writes
            v["seq_state.lines"][0, :4].copy_(torch.tensor([1, 2, 0, 3], dtype=torch.int32, device="cuda"))
            pos_v = torch.arange(4, 12, dtype=torch.int64, device="cuda").repeat(4)
            valid_v = torch.cat([
                torch.ones(8, dtype=torch.int32, device="cuda"),
                torch.ones(8, dtype=torch.int32, device="cuda"),
                torch.ones(8, dtype=torch.int32, device="cuda"),
                torch.zeros(8, dtype=torch.int32, device="cuda"),
            ])
            v["positions"][:32].copy_(pos_v)
            v["valid"][:32].copy_(valid_v)
            v["lengths"][:4].copy_(torch.tensor([12, 12, 12, 12], dtype=torch.int32, device="cuda"))

            q_v = torch.randn((32, 3072), dtype=torch.bfloat16, device="cuda") * 0.5
            r_v = torch.randn((32, 384), dtype=torch.bfloat16, device="cuda") * 0.5
            k_v = torch.randn((32, 1024), dtype=torch.bfloat16, device="cuda") * 0.3
            v_v = torch.randn((32, 1024), dtype=torch.bfloat16, device="cuda") * 0.3
            gate_v = torch.randn((32, 3072), dtype=torch.bfloat16, device="cuda")
            sink_v = torch.randn((32, 24), dtype=torch.bfloat16, device="cuda")
            v["t.q_in"][:32].copy_(q_v)
            v["t.r_in"][:32].copy_(r_v)
            v["t.k_in"][:32].copy_(k_v)
            v["t.v_in"][:32].copy_(v_v)
            v["t.gate"][:32].copy_(gate_v)
            v["t.sink"][:32].copy_(sink_v)

            rt.run("swa_verify", {"tokens": 32, "seqs": 4})

            # Null line=0 (seq2) must output 0
            torch.testing.assert_close(v["t.attn_y"][16:24], torch.zeros((8, 3072), dtype=torch.bfloat16, device="cuda"), rtol=0, atol=0)

            # Check causal parity for seq0 (rows 0..7)
            k_all0 = torch.cat([k_pf, k_v[:8]], dim=0).view(12, 8, 128).float()
            kn0 = k_all0.pow(2).sum(dim=-1, keepdim=True).sqrt()
            k_clip0 = k_all0 * torch.clamp(22.627416997969522 / (kn0 + 1e-6), max=1.0)
            v_all0 = torch.cat([v_pf, v_v[:8]], dim=0).view(12, 8, 128)
            y_ver_ref0 = ref_rel_attention(
                v["t.q_clip_out"][:8],
                ref_quant_fp8_dequant(k_clip0, 0.05),
                ref_quant_fp8_dequant(ref_v_rmsnorm(v_all0), 0.025),
                v["t.rel_bias_out"][:8], gate_v[:8].view(8, 24, 128), sink_v[:8], past=4, window=1024,
            ).view(8, 3072)
            torch.testing.assert_close(v["t.attn_y"][:8], y_ver_ref0, rtol=3e-2, atol=3e-2)

            # Commit accepted = [1, 4, 8, 2]:
            # seq0 should now have committed tokens 0..4 (only k_v[0] committed; k_v[1..7] rolled back!)
            # seq1 should now have committed tokens 0..7 (k_v[8..11] committed; k_v[12..15] rolled back!)
            v["t.accepted"][:4].copy_(torch.tensor([1, 4, 8, 2], dtype=torch.int32, device="cuda"))
            rt.run("swa_commit", {"seqs": 4})

            # Decode 1 token at pos=5 for seq0 and pos=8 for seq1
            k_next = torch.randn((4, 1024), dtype=torch.bfloat16, device="cuda") * 0.3
            v_next = torch.randn((4, 1024), dtype=torch.bfloat16, device="cuda") * 0.3
            v["positions"][:4].copy_(torch.tensor([5, 8, 5, 5], dtype=torch.int64, device="cuda"))
            v["lengths"][:4].copy_(torch.tensor([6, 9, 6, 6], dtype=torch.int32, device="cuda"))
            v["valid"][:4].copy_(torch.tensor([1, 1, 1, 0], dtype=torch.int32, device="cuda"))
            v["t.k_in"][:4].copy_(k_next)
            v["t.v_in"][:4].copy_(v_next)
            rt.run("swa_decode", {"seqs": 4})

            # Reference for seq0 at len=6: [k_pf(4), k_v[0](1), k_next[0](1)]
            k_seq0 = torch.cat([k_pf, k_v[0:1], k_next[0:1]], dim=0).view(6, 8, 128).float()
            kn_s0 = k_seq0.pow(2).sum(dim=-1, keepdim=True).sqrt()
            kc_s0 = k_seq0 * torch.clamp(22.627416997969522 / (kn_s0 + 1e-6), max=1.0)
            v_seq0 = torch.cat([v_pf, v_v[0:1], v_next[0:1]], dim=0).view(6, 8, 128)
            y_dec_ref0 = ref_rel_attention(
                v["t.q_clip_out"][0:1],
                ref_quant_fp8_dequant(kc_s0, 0.05),
                ref_quant_fp8_dequant(ref_v_rmsnorm(v_seq0), 0.025),
                v["t.rel_bias_out"][0:1], v["t.gate"][0:1].view(1, 24, 128), v["t.sink"][0:1], past=5, window=1024,
            ).view(1, 3072)
            torch.testing.assert_close(v["t.attn_y"][0:1], y_dec_ref0, rtol=3e-2, atol=3e-2)

    def test_swa_verify_ring_wrap_and_batch_parity(self):
        """Exercise attn_swa_verify across cold start (past=0), exact window (past=1024), and ring wrap (past=1474) for seqs=1,4."""
        rt, v = self.rt, self.v
        v["t.k_scale"].fill_(0.05)
        v["t.v_scale"].fill_(0.025)
        v["t.q_l2"].fill_(1.0)
        v["t.r_l2"].fill_(1.0)
        v["t.rel_proj"].normal_(mean=0.0, std=0.2)

        for past, seqs in ((0, 1), (1024, 1), (1474, 4)):
            with self.subTest(past=past, seqs=seqs):
                k_pf = torch.randn((past, 1024), dtype=torch.bfloat16, device="cuda") * 0.4 if past > 0 else None
                v_pf = torch.randn((past, 1024), dtype=torch.bfloat16, device="cuda") * 0.4 if past > 0 else None
                if past > 0:
                    for si in range(seqs):
                        v["seq_state.lines"].fill_(0)
                        v["seq_state.lines"][0, 0] = si + 1
                        v["positions"][:past].copy_(torch.arange(past, dtype=torch.int64, device="cuda"))
                        v["lengths"][0] = past
                        v["t.q_in"][:past].zero_()
                        v["t.r_in"][:past].zero_()
                        v["t.k_in"][:past].copy_(k_pf)
                        v["t.v_in"][:past].copy_(v_pf)
                        v["t.gate"][:past].zero_()
                        v["t.sink"][:past].zero_()
                        rt.run("swa_prefill", {"tokens": past})

                T = seqs * 8
                v["seq_state.lines"][0, :seqs].copy_(torch.arange(1, seqs + 1, dtype=torch.int32, device="cuda"))
                v["positions"][:T].copy_(torch.arange(past, past + 8, dtype=torch.int64, device="cuda").repeat(seqs))
                v["valid"][:T].fill_(1)
                v["lengths"][:seqs].fill_(past + 8)
                q_v = torch.randn((T, 3072), dtype=torch.bfloat16, device="cuda") * 0.6
                r_v = torch.randn((T, 384), dtype=torch.bfloat16, device="cuda") * 0.6
                k_v = torch.randn((T, 1024), dtype=torch.bfloat16, device="cuda") * 0.4
                v_v = torch.randn((T, 1024), dtype=torch.bfloat16, device="cuda") * 0.4
                g_v = torch.randn((T, 3072), dtype=torch.bfloat16, device="cuda")
                s_v = torch.randn((T, 24), dtype=torch.bfloat16, device="cuda")
                v["t.q_in"][:T].copy_(q_v)
                v["t.r_in"][:T].copy_(r_v)
                v["t.k_in"][:T].copy_(k_v)
                v["t.v_in"][:T].copy_(v_v)
                v["t.gate"][:T].copy_(g_v)
                v["t.sink"][:T].copy_(s_v)

                rt.run("swa_verify", {"tokens": T, "seqs": seqs})

                for si in range(seqs):
                    r0, r1 = si * 8, (si + 1) * 8
                    k_all = (torch.cat([k_pf, k_v[r0:r1]], dim=0) if past > 0 else k_v[r0:r1]).view(past + 8, 8, 128).float()
                    kn = k_all.pow(2).sum(dim=-1, keepdim=True).sqrt()
                    kc = k_all * torch.clamp(22.627416997969522 / (kn + 1e-6), max=1.0)
                    v_all = (torch.cat([v_pf, v_v[r0:r1]], dim=0) if past > 0 else v_v[r0:r1]).view(past + 8, 8, 128)
                    y_ref = ref_rel_attention(
                        v["t.q_clip_out"][r0:r1],
                        ref_quant_fp8_dequant(kc, 0.05),
                        ref_quant_fp8_dequant(ref_v_rmsnorm(v_all), 0.025),
                        v["t.rel_bias_out"][r0:r1], g_v[r0:r1].view(8, 24, 128), s_v[r0:r1],
                        past=past, window=1024,
                    ).view(8, 3072)
                    torch.testing.assert_close(v["t.attn_y"][r0:r1], y_ref, rtol=3e-2, atol=3e-2)

    def test_cuda_graph_capture_replay(self):
        rt, v = self.rt, self.v
        if True:
            v["t.k_scale"].fill_(0.05)
            v["t.v_scale"].fill_(0.025)
            v["t.q_l2"].fill_(1.0)
            v["t.r_l2"].fill_(1.0)
            v["t.rel_proj"].normal_(mean=0.0, std=0.1)

            # Seed 16 tokens on each sequence's distinct page (pages 0..3) so verify writes never race
            P = 16
            v["positions"][:P].copy_(torch.arange(P, dtype=torch.int64, device="cuda"))
            v["lengths"][0] = P
            for seq_pg in range(4):
                v["pages"][0, 0] = seq_pg
                v["t.k_in"][:P].normal_(std=0.3)
                v["t.v_in"][:P].normal_(std=0.3)
                rt.run("full_prefill", {"tokens": P})

            # Capture full_verify (seqs=4, tokens=32) into a CUDAGraph with disjoint pages [0, 1, 2, 3]
            v["valid"][:32].fill_(1)
            v["positions"][:32].copy_(torch.arange(16, 24, dtype=torch.int64, device="cuda").repeat(4))
            v["lengths"][:4].fill_(24)
            v["pages"][:4, 0].copy_(torch.arange(4, dtype=torch.int32, device="cuda"))
            v["t.q_in"][:32].normal_(std=0.4)
            v["t.r_in"][:32].normal_(std=0.4)
            v["t.k_in"][:32].normal_(std=0.3)
            v["t.v_in"][:32].normal_(std=0.3)
            v["t.gate"][:32].normal_(std=0.5)
            v["t.sink"][:32].normal_(std=0.5)

            stream = torch.cuda.Stream()
            graph = torch.cuda.CUDAGraph()
            # Warmup capture stream
            with torch.cuda.stream(stream):
                rt.enqueue_after("full_verify", {"seqs": 4, "tokens": 32}, stream.cuda_stream)
            stream.synchronize()

            with torch.cuda.graph(graph, stream=stream):
                rt.enqueue_after("full_verify", {"seqs": 4, "tokens": 32}, stream.cuda_stream)

            # Replay 1 (seqlen=24) -> compare with eager
            graph.replay()
            torch.cuda.synchronize()
            y_graph_1 = v["t.attn_y"][:32].clone()
            rt.run("full_verify", {"seqs": 4, "tokens": 32})
            torch.testing.assert_close(y_graph_1, v["t.attn_y"][:32], rtol=0, atol=0)

            # Mutate lengths & pages between replays (short -> longer -> short)
            v["lengths"][:4].copy_(torch.tensor([12, 24, 9, 18], dtype=torch.int32, device="cuda"))
            v["positions"][:32].copy_(torch.cat([
                torch.arange(4, 12, dtype=torch.int64, device="cuda"),
                torch.arange(16, 24, dtype=torch.int64, device="cuda"),
                torch.arange(1, 9, dtype=torch.int64, device="cuda"),
                torch.arange(10, 18, dtype=torch.int64, device="cuda"),
            ]))
            graph.replay()
            torch.cuda.synchronize()
            y_graph_2 = v["t.attn_y"][:32].clone()
            rt.run("full_verify", {"seqs": 4, "tokens": 32})
            torch.testing.assert_close(y_graph_2, v["t.attn_y"][:32], rtol=0, atol=0)

    def test_seq_state_max_seqs_tail_and_canary(self):
        rt, v = self.rt, self.v
        v["t.k_scale"].fill_(0.05)
        v["t.v_scale"].fill_(0.025)
        v["t.q_l2"].fill_(1.0)
        v["t.r_l2"].fill_(1.0)
        v["t.rel_proj"].normal_(mean=0.0, std=0.1)

        w_k = torch.randn((1024, 4), dtype=torch.bfloat16, device="cuda") * 0.25
        w_attn = torch.randn((2304, 4), dtype=torch.bfloat16, device="cuda") * 0.25
        w_mlp = torch.randn((2304, 4), dtype=torch.bfloat16, device="cuda") * 0.25
        v["t.w_k_static"].copy_(w_k)
        v["t.w_attn"].copy_(w_attn)
        v["t.w_mlp"].copy_(w_mlp)

        # Seed canary patterns into line = 2 and line = 9 (slot 9 exists since seq_slots() == max_seqs + 2 == 10)
        T = 4
        canary_k = {}
        canary_v = {}
        canary_q = {}
        canary_r = {}
        canary_gate = {}
        canary_sink = {}
        for canary_line in (2, 9):
            ck_t = torch.randn((T, 1024), dtype=torch.bfloat16, device="cuda") * 0.5
            cv_t = torch.randn((T, 1024), dtype=torch.bfloat16, device="cuda") * 0.5
            cq_t = torch.randn((T, 3072), dtype=torch.bfloat16, device="cuda") * 0.4
            cr_t = torch.randn((T, 384), dtype=torch.bfloat16, device="cuda") * 0.4
            cg_t = torch.randn((T, 3072), dtype=torch.bfloat16, device="cuda")
            cs_t = torch.randn((T, 24), dtype=torch.bfloat16, device="cuda")
            canary_k[canary_line] = ck_t
            canary_v[canary_line] = cv_t
            canary_q[canary_line] = cq_t
            canary_r[canary_line] = cr_t
            canary_gate[canary_line] = cg_t
            canary_sink[canary_line] = cs_t

            v["seq_state.lines"].fill_(0)
            v["seq_state.lines"][0, 0] = canary_line
            v["positions"][:T].copy_(torch.arange(T, dtype=torch.int64, device="cuda"))
            v["valid"][:T].fill_(1)
            v["lengths"][0] = T
            v["t.k_in"][:T].copy_(ck_t)
            v["t.v_in"][:T].copy_(cv_t)
            v["t.attn_in"][:T].zero_()
            v["t.mlp_in"][:T].zero_()
            v["t.res"][:T].zero_()
            v["t.addend"][:T].zero_()
            v["t.q_in"][:T].copy_(cq_t)
            v["t.r_in"][:T].copy_(cr_t)
            v["t.gate"][:T].copy_(cg_t)
            v["t.sink"][:T].copy_(cs_t)
            rt.run("sconv_local_prefill", {"tokens": T})
            rt.run("swa_prefill", {"tokens": T})

        # Write high-offset tail regions (v47, swa39, swa_mtp1 up to byte 89,997,312 of slot) at line = 1 and line = 8
        active_data = {}
        for active_line in (1, 8):
            ak_t = torch.randn((T, 1024), dtype=torch.bfloat16, device="cuda") * 0.5
            av_t = torch.randn((T, 1024), dtype=torch.bfloat16, device="cuda") * 0.5
            aq_t = torch.randn((T, 3072), dtype=torch.bfloat16, device="cuda") * 0.4
            ar_t = torch.randn((T, 384), dtype=torch.bfloat16, device="cuda") * 0.4
            ag_t = torch.randn((T, 3072), dtype=torch.bfloat16, device="cuda")
            as_t = torch.randn((T, 24), dtype=torch.bfloat16, device="cuda")
            active_data[active_line] = (ak_t, av_t, aq_t, ar_t, ag_t, as_t)

            v["seq_state.lines"].fill_(0)
            v["seq_state.lines"][0, 0] = active_line
            v["positions"][:T].copy_(torch.arange(T, dtype=torch.int64, device="cuda"))
            v["valid"][:T].fill_(1)
            v["lengths"][0] = T
            v["t.k_in"][:T].copy_(ak_t)
            v["t.v_in"][:T].copy_(av_t)
            v["t.q_in"][:T].copy_(aq_t)
            v["t.r_in"][:T].copy_(ar_t)
            v["t.gate"][:T].copy_(ag_t)
            v["t.sink"][:T].copy_(as_t)
            rt.run("v47_sconv_prefill", {"tokens": T})
            rt.run("swa39_prefill", {"tokens": T})
            rt.run("swa_mtp1_prefill", {"tokens": T})

        # Verify line = 8 decode parity on v47, swa39, and swa_mtp1
        ak_8, av_8, aq_8, ar_8, ag_8, as_8 = active_data[8]
        kd8 = torch.randn((1, 1024), dtype=torch.bfloat16, device="cuda") * 0.5
        vd8 = torch.randn((1, 1024), dtype=torch.bfloat16, device="cuda") * 0.5
        qd8 = torch.randn((1, 3072), dtype=torch.bfloat16, device="cuda") * 0.4
        rd8 = torch.randn((1, 384), dtype=torch.bfloat16, device="cuda") * 0.4
        gd8 = torch.randn((1, 3072), dtype=torch.bfloat16, device="cuda")
        sd8 = torch.randn((1, 24), dtype=torch.bfloat16, device="cuda")
        v["seq_state.lines"].fill_(0)
        v["seq_state.lines"][0, 0] = 8
        v["positions"][0] = T
        v["valid"][0] = 1
        v["lengths"][0] = T + 1
        v["t.k_in"][0].copy_(kd8[0])
        v["t.v_in"][0].copy_(vd8[0])
        v["t.q_in"][0].copy_(qd8[0])
        v["t.r_in"][0].copy_(rd8[0])
        v["t.gate"][0].copy_(gd8[0])
        v["t.sink"][0].copy_(sd8[0])

        rt.run("v47_sconv_decode", {"seqs": 1})
        exp_v47 = ref_sconv_static(kd8, w_k, ak_8[1:4], torch.tensor([T], device="cuda"))[0]
        torch.testing.assert_close(v["sconv.k_out"][0], exp_v47, rtol=0, atol=0)

        for prog_name in ("swa39_decode", "swa_mtp1_decode"):
            v["t.k_in"][0].copy_(kd8[0])
            rt.run(prog_name, {"seqs": 1})
            k_seq = torch.cat([ak_8, kd8], dim=0).view(T + 1, 8, 128).float()
            kn_s = k_seq.pow(2).sum(dim=-1, keepdim=True).sqrt()
            kc_s = k_seq * torch.clamp(22.627416997969522 / (kn_s + 1e-6), max=1.0)
            v_seq = torch.cat([av_8, vd8], dim=0).view(T + 1, 8, 128)
            y_ref = ref_rel_attention(
                v["t.q_clip_out"][:1],
                ref_quant_fp8_dequant(kc_s, 0.05),
                ref_quant_fp8_dequant(ref_v_rmsnorm(v_seq), 0.025),
                v["t.rel_bias_out"][:1], gd8.view(1, 24, 128), sd8, past=T, window=1024,
            ).view(1, 3072)
            torch.testing.assert_close(v["t.attn_y"][:1], y_ref, rtol=3e-2, atol=3e-2)

        # Verify canary line = 2 and line = 9 (v0 sconv + swa0) suffered zero corruption
        for canary_line in (2, 9):
            ck_t = canary_k[canary_line]
            cv_t = canary_v[canary_line]
            kd_c = torch.randn((1, 1024), dtype=torch.bfloat16, device="cuda") * 0.5
            vd_c = torch.randn((1, 1024), dtype=torch.bfloat16, device="cuda") * 0.5
            qd_c = torch.randn((1, 3072), dtype=torch.bfloat16, device="cuda") * 0.4
            rd_c = torch.randn((1, 384), dtype=torch.bfloat16, device="cuda") * 0.4
            gd_c = torch.randn((1, 3072), dtype=torch.bfloat16, device="cuda")
            sd_c = torch.randn((1, 24), dtype=torch.bfloat16, device="cuda")

            v["seq_state.lines"].fill_(0)
            v["seq_state.lines"][0, 0] = canary_line
            v["positions"][0] = T
            v["valid"][0] = 1
            v["lengths"][0] = T + 1
            v["t.k_in"][0].copy_(kd_c[0])
            v["t.v_in"][0].copy_(vd_c[0])
            v["t.attn_in"][0].zero_()
            v["t.mlp_in"][0].zero_()
            v["t.res"][0].zero_()
            v["t.addend"][0].zero_()
            v["t.q_in"][0].copy_(qd_c[0])
            v["t.r_in"][0].copy_(rd_c[0])
            v["t.gate"][0].copy_(gd_c[0])
            v["t.sink"][0].copy_(sd_c[0])

            rt.run("sconv_local_decode", {"seqs": 1})
            exp_canary_sconv = ref_sconv_static(kd_c, w_k, ck_t[1:4], torch.tensor([T], device="cuda"))[0]
            torch.testing.assert_close(v["sconv.k_out"][0], exp_canary_sconv, rtol=0, atol=0)

            rt.run("swa_decode", {"seqs": 1})
            k_seq_c = torch.cat([ck_t, kd_c], dim=0).view(T + 1, 8, 128).float()
            kn_c = k_seq_c.pow(2).sum(dim=-1, keepdim=True).sqrt()
            kc_c = k_seq_c * torch.clamp(22.627416997969522 / (kn_c + 1e-6), max=1.0)
            v_seq_c = torch.cat([cv_t, vd_c], dim=0).view(T + 1, 8, 128)
            y_ref_c = ref_rel_attention(
                v["t.q_clip_out"][:1],
                ref_quant_fp8_dequant(kc_c, 0.05),
                ref_quant_fp8_dequant(ref_v_rmsnorm(v_seq_c), 0.025),
                v["t.rel_bias_out"][:1], gd_c.view(1, 24, 128), sd_c, past=T, window=1024,
            ).view(1, 3072)
            torch.testing.assert_close(v["t.attn_y"][:1], y_ref_c, rtol=3e-2, atol=3e-2)

    def test_rel_attention_past_gt_zero_prefill_parity(self):
        rt, v = self.rt, self.v
        v["t.k_scale"].fill_(0.05)
        v["t.v_scale"].fill_(0.025)
        v["t.q_l2"].fill_(1.0)
        v["t.r_l2"].fill_(1.0)
        v["t.rel_proj"].normal_(mean=0.0, std=0.12)

        # 1) Full prefill chunked past > 0: Turn 1 T1=70 (past=0), then Turn 2 past=70 with T2=17 (split) and T2=65 (single)
        T1 = 70
        k1 = torch.randn((T1, 1024), dtype=torch.bfloat16, device="cuda") * 0.4
        v1 = torch.randn((T1, 1024), dtype=torch.bfloat16, device="cuda") * 0.4
        v["pages"][0, :4].copy_(torch.arange(4, dtype=torch.int32, device="cuda"))
        v["positions"][:T1].copy_(torch.arange(T1, dtype=torch.int64, device="cuda"))
        v["lengths"][0] = T1
        v["t.q_in"][:T1].zero_()
        v["t.r_in"][:T1].zero_()
        v["t.k_in"][:T1].copy_(k1)
        v["t.v_in"][:T1].copy_(v1)
        v["t.gate"][:T1].zero_()
        v["t.sink"][:T1].zero_()
        rt.run("full_prefill", {"tokens": T1})

        for T2 in (17, 65):
            with self.subTest(kind="full", past=T1, T2=T2):
                q2 = torch.randn((T2, 3072), dtype=torch.bfloat16, device="cuda") * 0.5
                r2 = torch.randn((T2, 384), dtype=torch.bfloat16, device="cuda") * 0.5
                k2 = torch.randn((T2, 1024), dtype=torch.bfloat16, device="cuda") * 0.4
                v2 = torch.randn((T2, 1024), dtype=torch.bfloat16, device="cuda") * 0.4
                g2 = torch.randn((T2, 3072), dtype=torch.bfloat16, device="cuda")
                s2 = torch.randn((T2, 24), dtype=torch.bfloat16, device="cuda")

                v["positions"][:T2].copy_(torch.arange(T1, T1 + T2, dtype=torch.int64, device="cuda"))
                v["lengths"][0] = T1 + T2
                v["t.q_in"][:T2].copy_(q2)
                v["t.r_in"][:T2].copy_(r2)
                v["t.k_in"][:T2].copy_(k2)
                v["t.v_in"][:T2].copy_(v2)
                v["t.gate"][:T2].copy_(g2)
                v["t.sink"][:T2].copy_(s2)
                rt.run("full_prefill", {"tokens": T2})

                k_all = torch.cat([k1, k2], dim=0).view(T1 + T2, 8, 128).float()
                kn = k_all.pow(2).sum(dim=-1, keepdim=True).sqrt()
                k_clip = k_all * torch.clamp(22.627416997969522 / (kn + 1e-6), max=1.0)
                v_all = torch.cat([v1, v2], dim=0).view(T1 + T2, 8, 128)
                y_ref = ref_rel_attention(
                    v["t.q_clip_out"][:T2],
                    ref_quant_fp8_dequant(k_clip, 0.05),
                    ref_quant_fp8_dequant(ref_v_rmsnorm(v_all), 0.025),
                    v["t.rel_bias_out"][:T2], g2.view(T2, 24, 128), s2,
                    past=T1, window=None, sm_scale=128.0**-0.5,
                ).view(T2, 3072)
                torch.testing.assert_close(v["t.attn_y"][:T2], y_ref, rtol=3e-2, atol=3e-2)

        # 2) SWA prefill past > 0 with ring wrap-around: Turn 1 T1=1000, Turn 2 T2=64 (seqlen_k=1064 > 1024), then decode @ 1064
        v["seq_state.lines"].fill_(0)
        v["seq_state.lines"][0, 0] = 1
        S1 = 1000
        S2 = 64
        ks1 = torch.randn((S1, 1024), dtype=torch.bfloat16, device="cuda") * 0.4
        vs1 = torch.randn((S1, 1024), dtype=torch.bfloat16, device="cuda") * 0.4
        v["positions"][:S1].copy_(torch.arange(S1, dtype=torch.int64, device="cuda"))
        v["lengths"][0] = S1
        v["t.q_in"][:S1].zero_()
        v["t.r_in"][:S1].zero_()
        v["t.k_in"][:S1].copy_(ks1)
        v["t.v_in"][:S1].copy_(vs1)
        v["t.gate"][:S1].zero_()
        v["t.sink"][:S1].zero_()
        rt.run("swa_prefill", {"tokens": S1})

        qs2 = torch.randn((S2, 3072), dtype=torch.bfloat16, device="cuda") * 0.5
        rs2 = torch.randn((S2, 384), dtype=torch.bfloat16, device="cuda") * 0.5
        ks2 = torch.randn((S2, 1024), dtype=torch.bfloat16, device="cuda") * 0.4
        vs2 = torch.randn((S2, 1024), dtype=torch.bfloat16, device="cuda") * 0.4
        gs2 = torch.randn((S2, 3072), dtype=torch.bfloat16, device="cuda")
        ss2 = torch.randn((S2, 24), dtype=torch.bfloat16, device="cuda")
        v["positions"][:S2].copy_(torch.arange(S1, S1 + S2, dtype=torch.int64, device="cuda"))
        v["lengths"][0] = S1 + S2
        v["t.q_in"][:S2].copy_(qs2)
        v["t.r_in"][:S2].copy_(rs2)
        v["t.k_in"][:S2].copy_(ks2)
        v["t.v_in"][:S2].copy_(vs2)
        v["t.gate"][:S2].copy_(gs2)
        v["t.sink"][:S2].copy_(ss2)
        rt.run("swa_prefill", {"tokens": S2})

        k_swa_all = torch.cat([ks1, ks2], dim=0).view(S1 + S2, 8, 128).float()
        kn_s = k_swa_all.pow(2).sum(dim=-1, keepdim=True).sqrt()
        kc_s = k_swa_all * torch.clamp(22.627416997969522 / (kn_s + 1e-6), max=1.0)
        v_swa_all = torch.cat([vs1, vs2], dim=0).view(S1 + S2, 8, 128)
        y_swa_ref = ref_rel_attention(
            v["t.q_clip_out"][:S2],
            ref_quant_fp8_dequant(kc_s, 0.05),
            ref_quant_fp8_dequant(ref_v_rmsnorm(v_swa_all), 0.025),
            v["t.rel_bias_out"][:S2], gs2.view(S2, 24, 128), ss2,
            past=S1, window=1024, sm_scale=128.0**-0.5,
        ).view(S2, 3072)
        torch.testing.assert_close(v["t.attn_y"][:S2], y_swa_ref, rtol=3e-2, atol=3e-2)

        # Follow-up decode at pos = 1064 verifies ring wrap-around commit
        kd = torch.randn((1, 1024), dtype=torch.bfloat16, device="cuda") * 0.4
        vd = torch.randn((1, 1024), dtype=torch.bfloat16, device="cuda") * 0.4
        qd = torch.randn((1, 3072), dtype=torch.bfloat16, device="cuda") * 0.5
        rd = torch.randn((1, 384), dtype=torch.bfloat16, device="cuda") * 0.5
        gd = torch.randn((1, 3072), dtype=torch.bfloat16, device="cuda")
        sd = torch.randn((1, 24), dtype=torch.bfloat16, device="cuda")
        v["positions"][0] = S1 + S2
        v["valid"][0] = 1
        v["lengths"][0] = S1 + S2 + 1
        v["t.k_in"][0].copy_(kd[0])
        v["t.v_in"][0].copy_(vd[0])
        v["t.q_in"][0].copy_(qd[0])
        v["t.r_in"][0].copy_(rd[0])
        v["t.gate"][0].copy_(gd[0])
        v["t.sink"][0].copy_(sd[0])
        rt.run("swa_decode", {"seqs": 1})

        k_dec_all = torch.cat([ks1, ks2, kd], dim=0).view(S1 + S2 + 1, 8, 128).float()
        kn_d = k_dec_all.pow(2).sum(dim=-1, keepdim=True).sqrt()
        kc_d = k_dec_all * torch.clamp(22.627416997969522 / (kn_d + 1e-6), max=1.0)
        v_dec_all = torch.cat([vs1, vs2, vd], dim=0).view(S1 + S2 + 1, 8, 128)
        y_dec_ref = ref_rel_attention(
            v["t.q_clip_out"][:1],
            ref_quant_fp8_dequant(kc_d, 0.05),
            ref_quant_fp8_dequant(ref_v_rmsnorm(v_dec_all), 0.025),
            v["t.rel_bias_out"][:1], gd.view(1, 24, 128), sd,
            past=S1 + S2, window=1024, sm_scale=128.0**-0.5,
        ).view(1, 3072)
        torch.testing.assert_close(v["t.attn_y"][:1], y_dec_ref, rtol=3e-2, atol=3e-2)

    def test_m3_fill_f32_gpu(self):
        rt, v = self.rt, self.v
        v["t.fill_k_scale"].fill_(-1.0)
        v["t.fill_v_scale"].fill_(-1.0)
        rt.run("fill_f32_test", {})
        torch.testing.assert_close(v["t.fill_k_scale"], torch.tensor([0.05], dtype=torch.float32, device="cuda"), rtol=0, atol=0)
        torch.testing.assert_close(v["t.fill_v_scale"], torch.tensor([0.025], dtype=torch.float32, device="cuda"), rtol=0, atol=0)

        # Feed populated scales into swa_prefill
        v["t.k_scale"].copy_(v["t.fill_k_scale"])
        v["t.v_scale"].copy_(v["t.fill_v_scale"])
        T = 4
        v["seq_state.lines"].fill_(0)
        v["seq_state.lines"][0, 0] = 1
        v["positions"][:T].copy_(torch.arange(T, dtype=torch.int64, device="cuda"))
        v["lengths"][0] = T
        v["t.q_in"][:T].normal_(std=0.5)
        v["t.r_in"][:T].normal_(std=0.5)
        v["t.k_in"][:T].normal_(std=0.5)
        v["t.v_in"][:T].normal_(std=0.5)
        v["t.q_l2"].fill_(1.0)
        v["t.r_l2"].fill_(1.0)
        v["t.rel_proj"].zero_()
        v["t.gate"][:T].normal_(std=0.5)
        v["t.sink"][:T].normal_(std=0.5)
        rt.run("swa_prefill", {"tokens": T})
        self.assertFalse(torch.isnan(v["t.attn_y"][:T]).any())

    def test_gate_sink_nat_and_log2(self):
        rt, v = self.rt, self.v
        T = 7
        attn_out = torch.randn((T, 24, 128), dtype=torch.float32, device="cuda")
        gate_full = torch.randn((T, 4096), dtype=torch.bfloat16, device="cuda")
        lse = torch.randn((T, 24), dtype=torch.float32, device="cuda") * 2.0
        sink_full = torch.randn((T, 32), dtype=torch.bfloat16, device="cuda")

        v["t.gs_attn_out"][:T].copy_(attn_out)
        v["t.gs_gate"][:T].copy_(gate_full)
        v["t.gs_lse"][:T].copy_(lse)
        v["t.gs_sink"][:T].copy_(sink_full)

        gate_view = gate_full[:, :3072].view(T, 24, 128).float()
        sink_view = sink_full[:, :24].float()

        for prog, lse_is_log2 in (("gate_sink_nat", 0), ("gate_sink_log2", 1)):
            with self.subTest(lse_is_log2=lse_is_log2):
                rt.run(prog, {"tokens": T})
                lse_nat = lse * math.log(2.0) if lse_is_log2 else lse
                ref = (
                    attn_out
                    * torch.sigmoid(gate_view)
                    * torch.sigmoid(lse_nat - sink_view).unsqueeze(-1)
                ).to(torch.bfloat16).view(T, 3072)
                torch.testing.assert_close(v["t.gs_y"][:T], ref, rtol=1e-2, atol=2e-2)

    def test_attn_prologue_extreme_1e20_and_pdl(self):
        rt, v = self.rt, self.v
        T = 4
        v["t.k_scale"].fill_(0.05)
        v["t.v_scale"].fill_(0.025)
        v["t.q_l2"].fill_(1.0)
        v["t.r_l2"].fill_(1.0)
        v["t.rel_proj"].fill_(0.01)
        v["seq_state.lines"].fill_(0)
        v["seq_state.lines"][0, 0] = 1
        v["pages"][0, :16385].fill_(0)

        # 1) |x| = 1e20 BF16 inputs must produce finite non-NaN clipped/normed outputs (no +inf * 0 NaN!)
        pos_ext = torch.tensor([0, 8192, 16384, 65536], dtype=torch.int64, device="cuda")
        v["positions"][:T].copy_(pos_ext)
        v["valid"][:T].fill_(1)
        v["lengths"][0] = T
        v["t.q_in"][:T].fill_(1e20)
        v["t.r_in"][:T].fill_(-1e20)
        v["t.k_in"][:T].fill_(1e20)
        v["t.v_in"][:T].fill_(-1e20)
        v["t.gate"][:T].zero_()
        v["t.sink"][:T].zero_()

        for prog, kw in (
            ("swa_prefill", {"tokens": T}),
            ("full_prefill", {"tokens": T}),
            ("full_decode", {"seqs": 1}),
            ("swa_decode", {"seqs": 1}),
        ):
            with self.subTest(extreme_prog=prog):
                rt.run(prog, kw)
                rows = kw.get("tokens", kw.get("seqs"))
                self.assertFalse(torch.isnan(v["t.q_clip_out"][:rows]).any(), f"{prog} q_clip NaN")
                self.assertFalse(torch.isinf(v["t.q_clip_out"][:rows]).any(), f"{prog} q_clip Inf")
                self.assertFalse(torch.isnan(v["t.rel_bias_out"][:rows]).any(), f"{prog} rel_bias NaN")
                self.assertFalse(torch.isinf(v["t.rel_bias_out"][:rows]).any(), f"{prog} rel_bias Inf")
                if prog == "swa_prefill":
                    torch.testing.assert_close(
                        v["t.q_clip_out"][:rows].float(),
                        torch.full((rows, 24, 128), 2.0, dtype=torch.float32, device="cuda"),
                        rtol=2e-2, atol=5e-2,
                    )
                    # r_in = -1e20, dim=16, thresh=8.0 -> -8.0 / 4.0 = -2.0; rel_proj = 0.01 -> 16 * (-2.0) * 0.01 = -0.32
                    torch.testing.assert_close(
                        v["t.rel_bias_out"][:rows].float(),
                        torch.full((rows, 24, 1024), -0.32, dtype=torch.float32, device="cuda"),
                        rtol=2e-2, atol=5e-2,
                    )
                    k_f = v["t.swa_k_out"][:rows].view(torch.float8_e4m3fn).float() * 0.05
                    v_f = v["t.swa_v_out"][:rows].view(torch.float8_e4m3fn).float() * 0.025
                    self.assertFalse(torch.isnan(k_f).any())
                    self.assertFalse(torch.isnan(v_f).any())
                    # Each elem of constant-1e20 128-dim vector clips to 22.6274/sqrt(128) == 2.0, V rmsnorm -> -1.0
                    torch.testing.assert_close(k_f, torch.full_like(k_f, 2.0), rtol=0.1, atol=0.2)
                    torch.testing.assert_close(v_f, torch.full_like(v_f, -1.0), rtol=0.1, atol=0.2)

        # 2) PDL ordering check: pdl=True after attn_log_tau at pos >= 8192 matches pdl=False
        pos_pdl = torch.tensor([8192, 16384, 65536, 1048575], dtype=torch.int64, device="cuda")
        q_p = torch.randn((T, 3072), dtype=torch.bfloat16, device="cuda") * 2.0
        r_p = torch.randn((T, 384), dtype=torch.bfloat16, device="cuda") * 2.0
        k_p = torch.randn((T, 1024), dtype=torch.bfloat16, device="cuda") * 2.0
        v_p = torch.randn((T, 1024), dtype=torch.bfloat16, device="cuda") * 2.0
        v["t.rel_proj"].normal_(mean=0.0, std=0.25)
        v["positions"][:T].copy_(pos_pdl)
        v["valid"][:T].fill_(1)
        v["lengths"][0] = T
        v["t.q_in"][:T].copy_(q_p)
        v["t.r_in"][:T].copy_(r_p)
        v["t.k_in"][:T].copy_(k_p)
        v["t.v_in"][:T].copy_(v_p)

        rt.run("full_prefill", {"tokens": T})
        q_nopdl = v["t.q_clip_out"][:T].clone()
        rel_nopdl = v["t.rel_bias_out"][:T].clone()
        rt.run("full_prefill_pdl", {"tokens": T})
        torch.testing.assert_close(v["t.q_clip_out"][:T], q_nopdl, rtol=0, atol=0)
        torch.testing.assert_close(v["t.rel_bias_out"][:T], rel_nopdl, rtol=0, atol=0)

        rt.run("full_decode", {"seqs": 2})
        q_dec_nopdl = v["t.q_clip_out"][:2].clone()
        rel_dec_nopdl = v["t.rel_bias_out"][:2].clone()
        rt.run("full_decode_pdl", {"seqs": 2})
        torch.testing.assert_close(v["t.q_clip_out"][:2], q_dec_nopdl, rtol=0, atol=0)
        torch.testing.assert_close(v["t.rel_bias_out"][:2], rel_dec_nopdl, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
