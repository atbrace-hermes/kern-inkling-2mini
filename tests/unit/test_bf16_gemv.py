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
            (8576, 2304),  # fused q+r+gate+sink+k+v
        ]
        # Exercises every `when` bucket: 1, 2..8, 9..16, 17..32, >=33
        ms = [1, 2, 4, 5, 8, 9, 16, 17, 24, 32, 33, 51, 64]
        self._run_shape_suite(shapes, ms, use_graph=False)

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
