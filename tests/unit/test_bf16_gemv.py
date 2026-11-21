#!/usr/bin/env python3
"""GPU unit tests for SM120 small-M BF16 GEMV/GEMM (`m3_gemm_bf16` / `bf16_gemv_sm120.cu`)."""

from __future__ import annotations

import gc
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
BUILD_DIR = ROOT / "build/sm_120"
sys.path.insert(0, str(ROOT / "scripts"))

import attn_manifest as am  # noqa: E402


def _load_kern():
    try:
        import kern  # type: ignore
        import kern.host  # type: ignore

        if hasattr(kern, "Runtime"):
            return kern, kern.host
    except ImportError:
        pass

    kern_py_dir = Path("/workspace/kern/crates/kern-py/python/kern")
    lib_kern_so = Path("/workspace/kern/target/release/lib_kern.so")
    spec = importlib.util.spec_from_file_location("kern._kern", str(lib_kern_so))
    assert spec is not None and spec.loader is not None
    _kern = importlib.util.module_from_spec(spec)
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
    kern_pkg.host = host_mod
    return kern_pkg, host_mod


def _create_runtime(kern, manifest: dict):
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "model.json"
        p.write_text(json.dumps(manifest))
        return kern.Runtime(str(p), str(BUILD_DIR), [], 0)


def _region(host, rt, manifest: dict, name: str) -> torch.Tensor:
    return host.region(rt, manifest, name)


@unittest.skipUnless(torch.cuda.is_available(), "requires a CUDA GPU")
class TestBf16GemvSm120(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kern, cls.host = _load_kern()
        cls.dsl = am.load_dsl(BUILD_DIR)

    def _run_shape_suite(self, shapes: list[tuple[int, int]], ms: list[int], use_graph: bool = False):
        max_m = max(ms)
        max_n = max(n for n, _ in shapes)
        max_k = max(k for _, k in shapes)
        modules: dict = {}
        ops: dict = {}
        am.register_modules(modules, BUILD_DIR, self.dsl)
        am.register_ops(ops, self.dsl)
        b, i32 = am._b, am._i32

        buffers = {
            "a": {"dtype": "bf16", "shape": ["tokens", max_k], "kind": "input"},
            "w": {"dtype": "bf16", "shape": [max_n, max_k], "kind": "input"},
            "c": {"dtype": "bf16", "shape": ["tokens", max_n], "kind": "output"},
            "c2": {"dtype": "bf16", "shape": ["tokens", max_n], "kind": "output"},
        }
        programs = {}
        for idx, (n, k) in enumerate(shapes):
            prog = {
                "calls": [
                    {"op": "m3_gemm_bf16", "args": [b("a"), b("w"), b("c"), {"var": "tokens"}, i32(n), i32(k)]},
                    # Back-to-back PDL launch check (second GEMM shares W/A, writes c2)
                    {"op": "m3_gemm_bf16", "args": [b("a"), b("w"), b("c2"), {"var": "tokens"}, i32(n), i32(k)]},
                ]
            }
            if use_graph:
                prog["batch"] = {"groups": max_m, "rows": 1}
                prog["graph"] = True
            programs[f"s{idx}"] = prog

        manifest = am.prune_unused({
            "schema_version": 7,
            "model": "test-bf16-gemv",
            "vars": {"tokens": {"max": max_m, "axis": "rows"}},
            "states": {},
            "buffers": buffers,
            "modules": modules,
            "ops": ops,
            "programs": programs,
        })
        rt = _create_runtime(self.kern, manifest)
        try:
            va = _region(self.host, rt, manifest, "a")
            vw = _region(self.host, rt, manifest, "w")
            vc = _region(self.host, rt, manifest, "c")
            vc2 = _region(self.host, rt, manifest, "c2")

            torch.manual_seed(2026)
            for idx, (n, k) in enumerate(shapes):
                w = torch.randn((n, k), dtype=torch.bfloat16, device="cuda") * (k ** -0.5)
                vw.view(-1)[: n * k].copy_(w.view(-1))
                for m in ms:
                    a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
                    va.view(-1)[: m * k].copy_(a.view(-1))
                    vc.zero_()
                    vc2.zero_()
                    rt.run(f"s{idx}", {"tokens": m})
                    got = vc.view(-1)[: m * n].view(m, n)
                    got2 = vc2.view(-1)[: m * n].view(m, n)
                    ref_f32 = a.float() @ w.float().t()
                    ref_bf16 = ref_f32.to(torch.bfloat16)
                    # Bit-identical across consecutive PDL calls
                    torch.testing.assert_close(got, got2, rtol=0, atol=0)
                    # Within <= 1-2 BF16 ULPs of exact FP32 matmul (and comparable to cuBLASLt)
                    cublas_err = ((a @ w.t()).float() - ref_f32).abs().max().item()
                    got_err = (got.float() - ref_f32).abs().max().item()
                    max_allowed = max(cublas_err * 2.0 + 1e-2, 3e-2)
                    self.assertLessEqual(
                        got_err,
                        max_allowed,
                        f"M={m} N={n} K={k}: got_err={got_err:.4e} > {max_allowed:.4e} (cublas_err={cublas_err:.4e})",
                    )
                    torch.testing.assert_close(got, ref_bf16, rtol=2e-2, atol=2e-2)
        finally:
            del rt
            gc.collect()
            torch.cuda.empty_cache()

    def test_all_model_shapes_and_buckets(self):
        shapes = [
            (3072, 2304),  # q_proj, gate_proj
            (2304, 3072),  # o_proj
            (1024, 2304),  # k_proj, v_proj
            (384, 2304),   # r_proj
            (32, 2304),    # k/v_sconv.dyn
            (24, 2304),    # sink_proj
            (1056, 2304),  # fused k/v + sconv.dyn
            (6552, 2304),  # fused q+r+gate+sink
            (8600, 2304),  # fused q+r+gate+sink+k+v
        ]
        # Exercises every `when` bucket: 1, 2..8, 9..16, 17..32, >=33
        ms = [1, 2, 4, 5, 8, 9, 16, 17, 24, 32, 33, 51, 64]
        self._run_shape_suite(shapes, ms, use_graph=False)

    def test_pair_ops_and_buckets(self):
        modules: dict = {}
        ops: dict = {}
        am.register_modules(modules, BUILD_DIR, self.dsl)
        am.register_ops(ops, self.dsl)
        b, i32 = am._b, am._i32

        max_m = 72
        k = 2304
        buffers = {
            "a0": {"dtype": "bf16", "shape": ["tokens", k], "kind": "input"},
            "a1": {"dtype": "bf16", "shape": ["tokens", k], "kind": "input"},
            "w0": {"dtype": "bf16", "shape": [1024, k], "kind": "input"},
            "w1": {"dtype": "bf16", "shape": [1024, k], "kind": "input"},
            "c0": {"dtype": "bf16", "shape": ["tokens", 1024], "kind": "output"},
            "c1": {"dtype": "bf16", "shape": ["tokens", 1024], "kind": "output"},
        }
        programs = {
            "pair_kv": {
                "calls": [
                    {
                        "op": "m3_gemm_bf16_pair",
                        "args": [b("a0"), b("w0"), b("c0"), b("a1"), b("w1"), b("c1"), {"var": "tokens"}, i32(1024), i32(k)],
                    }
                ]
            },
            "pair_dyn": {
                "calls": [
                    {
                        "op": "m3_gemm_bf16_pair_dyn",
                        "args": [b("a0"), b("w0"), b("c0"), b("a1"), b("w1"), b("c1"), {"var": "tokens"}, i32(32), i32(k)],
                    }
                ]
            },
        }
        manifest = am.prune_unused({
            "schema_version": 7,
            "model": "test-bf16-gemv-pair",
            "vars": {"tokens": {"max": max_m, "axis": "rows"}},
            "states": {},
            "buffers": buffers,
            "modules": modules,
            "ops": ops,
            "programs": programs,
        })
        rt = _create_runtime(self.kern, manifest)
        try:
            va0 = _region(self.host, rt, manifest, "a0")
            va1 = _region(self.host, rt, manifest, "a1")
            vw0 = _region(self.host, rt, manifest, "w0")
            vw1 = _region(self.host, rt, manifest, "w1")
            vc0 = _region(self.host, rt, manifest, "c0")
            vc1 = _region(self.host, rt, manifest, "c1")
            torch.manual_seed(2027)
            for prog, n in (("pair_kv", 1024), ("pair_dyn", 32)):
                w0 = torch.randn((n, k), dtype=torch.bfloat16, device="cuda") * (k ** -0.5)
                w1 = torch.randn((n, k), dtype=torch.bfloat16, device="cuda") * (k ** -0.5)
                vw0.view(-1)[: n * k].copy_(w0.view(-1))
                vw1.view(-1)[: n * k].copy_(w1.view(-1))
                for m in (1, 4, 8, 16, 24, 32, 33, 51, 64, 72):
                    a0 = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
                    a1 = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
                    va0.view(-1)[: m * k].copy_(a0.view(-1))
                    va1.view(-1)[: m * k].copy_(a1.view(-1))
                    vc0.zero_()
                    vc1.zero_()
                    rt.run(prog, {"tokens": m})
                    g0 = vc0.view(-1)[: m * n].view(m, n)
                    g1 = vc1.view(-1)[: m * n].view(m, n)
                    r0 = (a0.float() @ w0.float().t()).to(torch.bfloat16)
                    r1 = (a1.float() @ w1.float().t()).to(torch.bfloat16)
                    torch.testing.assert_close(g0, r0, rtol=2e-2, atol=2e-2)
                    torch.testing.assert_close(g1, r1, rtol=2e-2, atol=2e-2)
        finally:
            del rt
            gc.collect()
            torch.cuda.empty_cache()

    def test_fused_projection_visit_parity(self):
        """Verify fused 8600/6552 projections + pair k/v match unfused projections through sconv + prologue + rel_attn."""
        modules, states, buffers, ops = {}, {}, {}, {}
        am.register_modules(modules, BUILD_DIR, self.dsl)
        am.register_states(states)
        am.register_buffers(buffers, max_seqs=8, chunk=64, context=1024, page_columns=16)
        am.register_ops(ops, self.dsl, max_seqs=8, page_columns=16)
        b, i32 = am._b, am._i32
        H = 2304

        buffers["x_q"] = {"dtype": "bf16", "shape": ["tokens", H], "kind": "input"}
        buffers["x_k"] = {"dtype": "bf16", "shape": ["tokens", H], "kind": "input"}
        buffers["x_v"] = {"dtype": "bf16", "shape": ["tokens", H], "kind": "input"}
        buffers["w_qrgskv"] = {"dtype": "bf16", "shape": [8600, H], "kind": "input"}
        buffers["model.llm.layers.0.attn.q_l2_clip.scale"] = {"dtype": "bf16", "shape": [24, 128], "kind": "input"}
        buffers["model.llm.layers.0.attn.rel_l2_clip.scale"] = {"dtype": "bf16", "shape": [24, 16], "kind": "input"}
        buffers["model.llm.layers.0.attn.rel_logits_proj.proj"] = {"dtype": "bf16", "shape": [16, 1024], "kind": "input"}
        buffers["model.llm.layers.0.attn.k_sconv.weight"] = {"dtype": "bf16", "shape": [1024, 4], "kind": "input"}
        buffers["model.llm.layers.0.attn.v_sconv.weight"] = {"dtype": "bf16", "shape": [1024, 4], "kind": "input"}
        for nm, cols in (
            ("u_q", 3072), ("u_r", 384), ("u_g", 3072), ("u_s", 24), ("u_k", 1024), ("u_v", 1024),
            ("f_all", 8600), ("f_k", 1024), ("f_v", 1024), ("out_u", 3072), ("out_f", 3072),
        ):
            buffers[nm] = {"dtype": "bf16", "shape": ["tokens", cols], "kind": "output"}

        derive: list = []
        am.emit_load_calls(derive, {"0": {"k": 0.05, "v": 0.025}}, ops, buffers)
        programs: dict = {"derive": {"derive": True, "calls": derive}}

        for nm, mode, batch in (
            ("pf", "prefill", {"groups": 1, "rows": "tokens"}),
            ("dec", "decode_batch", {"groups": 8, "rows": 1}),
            ("rnd", "round", {"groups": 8, "rows": 8}),
        ):
            t_var = "seqs" if mode == "decode_batch" else "tokens"
            # pass_idx==0: unfused vs fused 8600
            cu0 = [
                {"op": "m3_gemm_bf16", "args": [b("x_q"), b("w_qrgskv", 0), b("u_q"), {"var": t_var}, i32(3072), i32(H)]},
                {"op": "m3_gemm_bf16", "args": [b("x_q"), b("w_qrgskv", 3072 * H * 2), b("u_r"), {"var": t_var}, i32(384), i32(H)]},
                {"op": "m3_gemm_bf16", "args": [b("x_q"), b("w_qrgskv", 3456 * H * 2), b("u_g"), {"var": t_var}, i32(3072), i32(H)]},
                {"op": "m3_gemm_bf16", "args": [b("x_q"), b("w_qrgskv", 6528 * H * 2), b("u_s"), {"var": t_var}, i32(24), i32(H)]},
                {"op": "m3_gemm_bf16", "args": [b("x_q"), b("w_qrgskv", 6552 * H * 2), b("u_k"), {"var": t_var}, i32(1024), i32(H)]},
                {"op": "m3_gemm_bf16", "args": [b("x_q"), b("w_qrgskv", 7576 * H * 2), b("u_v"), {"var": t_var}, i32(1024), i32(H)]},
            ]
            am.emit_sconv_kv(cu0, mode, 0, 0, b("u_k"), b("u_v"), b("x_q"), b("x_q"), b("sconv.k_out"), b("sconv.v_out"))
            am.emit_layer_calls(cu0, mode, 0, 0, b("u_q"), b("u_r"), b("sconv.k_out"), b("sconv.v_out"), b("u_g"), b("u_s"), b("out_u"), 0.05, 0.025)

            cf0 = [{"op": "m3_gemm_bf16", "args": [b("x_q"), b("w_qrgskv"), b("f_all"), {"var": t_var}, i32(8600), i32(H)]}]
            am.emit_sconv_kv(cf0, mode, 0, 0, b("f_all", 6552 * 2), b("f_all", 7576 * 2), b("x_q"), b("x_q"), b("sconv.k_out"), b("sconv.v_out"), stride_k=8600, stride_v=8600)
            am.emit_layer_calls(cf0, mode, 0, 0, b("f_all", 0), b("f_all", 3072 * 2), b("sconv.k_out"), b("sconv.v_out"), b("f_all", 3456 * 2), b("f_all", 6528 * 2), b("out_f"), 0.05, 0.025, stride_q=8600, stride_r=8600, stride_gate=8600, stride_sink=8600)

            # pass_idx>0: unfused vs fused 6552 + pair 1024
            cu1 = [
                {"op": "m3_gemm_bf16", "args": [b("x_q"), b("w_qrgskv", 0), b("u_q"), {"var": t_var}, i32(3072), i32(H)]},
                {"op": "m3_gemm_bf16", "args": [b("x_q"), b("w_qrgskv", 3072 * H * 2), b("u_r"), {"var": t_var}, i32(384), i32(H)]},
                {"op": "m3_gemm_bf16", "args": [b("x_q"), b("w_qrgskv", 3456 * H * 2), b("u_g"), {"var": t_var}, i32(3072), i32(H)]},
                {"op": "m3_gemm_bf16", "args": [b("x_q"), b("w_qrgskv", 6528 * H * 2), b("u_s"), {"var": t_var}, i32(24), i32(H)]},
                {"op": "m3_gemm_bf16", "args": [b("x_k"), b("w_qrgskv", 6552 * H * 2), b("u_k"), {"var": t_var}, i32(1024), i32(H)]},
                {"op": "m3_gemm_bf16", "args": [b("x_v"), b("w_qrgskv", 7576 * H * 2), b("u_v"), {"var": t_var}, i32(1024), i32(H)]},
            ]
            am.emit_sconv_kv(cu1, mode, 0, 0, b("u_k"), b("u_v"), b("x_k"), b("x_v"), b("sconv.k_out"), b("sconv.v_out"))
            am.emit_layer_calls(cu1, mode, 0, 0, b("u_q"), b("u_r"), b("sconv.k_out"), b("sconv.v_out"), b("u_g"), b("u_s"), b("out_u"), 0.05, 0.025)

            cf1 = [
                {"op": "m3_gemm_bf16", "args": [b("x_q"), b("w_qrgskv", 0), b("f_all"), {"var": t_var}, i32(6552), i32(H)]},
                {"op": "m3_gemm_bf16_pair", "args": [b("x_k"), b("w_qrgskv", 6552 * H * 2), b("f_k"), b("x_v"), b("w_qrgskv", 7576 * H * 2), b("f_v"), {"var": t_var}, i32(1024), i32(H)]},
            ]
            am.emit_sconv_kv(cf1, mode, 0, 0, b("f_k"), b("f_v"), b("x_k"), b("x_v"), b("sconv.k_out"), b("sconv.v_out"))
            am.emit_layer_calls(cf1, mode, 0, 0, b("f_all", 0), b("f_all", 3072 * 2), b("sconv.k_out"), b("sconv.v_out"), b("f_all", 3456 * 2), b("f_all", 6528 * 2), b("out_f"), 0.05, 0.025, stride_q=6552, stride_r=6552, stride_gate=6552, stride_sink=6552)

            programs[f"{nm}_p0_u"] = {"batch": batch, "calls": cu0}
            programs[f"{nm}_p0_f"] = {"batch": batch, "calls": cf0}
            programs[f"{nm}_p1_u"] = {"batch": batch, "calls": cu1}
            programs[f"{nm}_p1_f"] = {"batch": batch, "calls": cf1}

        manifest = am.prune_unused({
            "schema_version": 7,
            "model": "test-fused-visit-parity",
            "vars": {"tokens": {"max": 64, "axis": "rows"}, "seqs": {"max": 8, "axis": "groups"}},
            "states": states,
            "buffers": buffers,
            "modules": modules,
            "ops": ops,
            "programs": programs,
        })
        rt = _create_runtime(self.kern, manifest)
        try:
            torch.manual_seed(42)
            for k, spec in manifest["buffers"].items():
                if spec["kind"] == "input" and spec["dtype"] == "bf16":
                    t = _region(self.host, rt, manifest, k)
                    t.copy_(torch.randn_like(t) * 0.05)
            # Run unfused vs fused using separate seq_state lines (u: lines 1..S, f: lines 1..S on fresh run or distinct)
            for nm, kw in (
                ("pf", {"tokens": 64, "seqs": 1}),
                ("dec", {"tokens": 1, "seqs": 1}),
                ("dec", {"tokens": 4, "seqs": 4}),
                ("rnd", {"tokens": 8, "seqs": 1}),
                ("rnd", {"tokens": 32, "seqs": 4}),
                ("rnd", {"tokens": 64, "seqs": 8}),
            ):
                T, S = kw["tokens"], kw["seqs"]
                rows_per_seq = T // S
                _region(self.host, rt, manifest, "lengths").view(-1)[:S].fill_(rows_per_seq)
                pos = torch.arange(rows_per_seq, dtype=torch.int64, device="cuda").repeat(S)
                _region(self.host, rt, manifest, "positions").view(-1)[:T].copy_(pos)
                if "valid" in manifest["buffers"]:
                    _region(self.host, rt, manifest, "valid").view(-1)[:T].fill_(1)
                for ptag in ("p0", "p1"):
                    lines_u = torch.arange(1, S + 1, dtype=torch.int32, device="cuda")
                    lines_f = torch.arange(S + 1, 2 * S + 1, dtype=torch.int32, device="cuda") if 2 * S <= 8 else lines_u
                    _region(self.host, rt, manifest, "seq_state.lines").view(-1)[:S].copy_(lines_u)
                    rt.run(f"{nm}_{ptag}_u", kw)
                    ou = _region(self.host, rt, manifest, "out_u")[:T].clone()
                    uq = _region(self.host, rt, manifest, "u_q")[:T].clone()
                    ur = _region(self.host, rt, manifest, "u_r")[:T].clone()
                    ug = _region(self.host, rt, manifest, "u_g")[:T].clone()
                    us = _region(self.host, rt, manifest, "u_s")[:T].clone()
                    uk = _region(self.host, rt, manifest, "u_k")[:T].clone()
                    uv = _region(self.host, rt, manifest, "u_v")[:T].clone()

                    _region(self.host, rt, manifest, "seq_state.lines").view(-1)[:S].copy_(lines_f)
                    rt.run(f"{nm}_{ptag}_f", kw)
                    of = _region(self.host, rt, manifest, "out_f")[:T].clone()
                    stride = 8600 if ptag == "p0" else 6552
                    fa = _region(self.host, rt, manifest, "f_all").view(-1)[: T * stride].view(T, stride)
                    fk = fa[:, 6552:7576] if ptag == "p0" else _region(self.host, rt, manifest, "f_k")[:T]
                    fv = fa[:, 7576:8600] if ptag == "p0" else _region(self.host, rt, manifest, "f_v")[:T]

                    if T <= 32:
                        # Custom SM120 GEMV/MMA kernels are K-reduction-invariant across N -> bit-identical!
                        torch.testing.assert_close(fa[:, 0:3072], uq, rtol=0, atol=0)
                        torch.testing.assert_close(fa[:, 3072:3456], ur, rtol=0, atol=0)
                        torch.testing.assert_close(fa[:, 3456:6528], ug, rtol=0, atol=0)
                        torch.testing.assert_close(fa[:, 6528:6552], us, rtol=0, atol=0)
                        torch.testing.assert_close(fk, uk, rtol=0, atol=0)
                        torch.testing.assert_close(fv, uv, rtol=0, atol=0)
                        torch.testing.assert_close(of, ou, rtol=0, atol=0)
                    else:
                        # T >= 33 uses cuBLASLt (different N-tiling -> 1 BF16 ULP, up to 1 FP8 E4M3 step after V RMSNorm)
                        torch.testing.assert_close(fa[:, 0:3072], uq, rtol=2e-2, atol=2e-2)
                        torch.testing.assert_close(fa[:, 3072:3456], ur, rtol=2e-2, atol=2e-2)
                        torch.testing.assert_close(fa[:, 3456:6528], ug, rtol=2e-2, atol=2e-2)
                        torch.testing.assert_close(fa[:, 6528:6552], us, rtol=2e-2, atol=2e-2)
                        torch.testing.assert_close(fk, uk, rtol=2e-2, atol=2e-2)
                        torch.testing.assert_close(fv, uv, rtol=2e-2, atol=2e-2)
                        torch.testing.assert_close(of, ou, rtol=8e-2, atol=8e-2)
        finally:
            del rt
            gc.collect()
            torch.cuda.empty_cache()

    def test_cuda_graph_and_edge_shapes(self):
        # Odd N, odd K (triggers aligned fallback cleanly), tiny N, and CUDA graph replay
        shapes = [
            (1, 2304),
            (7, 2304),
            (25, 2304),
            (131, 2304),
            (384, 17),     # K % 16 != 0 -> generic fallback
            (3072, 2304),
            (24, 2304),
        ]
        ms = [1, 3, 8, 12, 16, 24, 32, 48]
        self._run_shape_suite(shapes, ms, use_graph=True)


if __name__ == "__main__":
    unittest.main()
