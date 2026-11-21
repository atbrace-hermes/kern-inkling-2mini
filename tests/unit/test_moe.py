#!/usr/bin/env python3
"""Unit and integration tests for M2 Inkling-2-Mini SM120 NVFP4 & MXFP8 MoE.

CPU-only tests (`TestMoeCPU`) validate cubin metadata, absence of tcgen05 instructions,
checkpoint safetensors binding, and `kern verify` over a full schema-v7 manifest.

GPU tests (`TestMoeGPU`) require the Single-GPU lock and validate:
  - moe_router_topk (sigmoid+bias, deterministic tie-breaking, log-sigmoid softmax * 8.0)
  - moe_route_metadata_small (T <= 1024) & multi-CTA route_count/scan/scatter (T > 1024)
    including empty experts (m=0), m=1, 3, 129
  - moe_swizzle_weight_scales, moe_quant_nvfp4, moe_swiglu_quant_nvfp4 (fouroversix_mae + softcap)
  - moe_quant_mxfp8, moe_swiglu_quant_mxfp8 (UE8M0 block=32 + 128x4 swizzled scales + softcap)
  - SM120 NVFP4 (DTypeOut=f32) & MXFP8 (DTypeOut=bf16) grouped GEMMs + combine via kern.Runtime
    using real checkpoint slices from L0 (NVFP4) and L20 (MXFP8)
  - CUDA Graph capture and replay parity (torch.cuda.CUDAGraph + kern.Runtime decode/decode_batch/round)
"""

from __future__ import annotations

import ctypes
import hashlib
import importlib.util
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

# Preload system CUDA 13.5 cuBLAS before torch loads its bundled older cu13 libcublasLt.so.13,
# preventing libcublas.so / libcublasLt.so.13 version mismatch when lib_kern.so runs cublasGemmEx.
for _cu_lib in (
    "/usr/local/cuda/targets/x86_64-linux/lib/libcublasLt.so.13",
    "/usr/local/cuda/targets/x86_64-linux/lib/libcublas.so.13",
):
    if Path(_cu_lib).exists():
        ctypes.CDLL(_cu_lib, mode=ctypes.RTLD_GLOBAL)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import moe_manifest as mm

KERN_BIN = Path("/workspace/kern/target/release/kern")
KERN_PY_ROOT = Path("/workspace/kern/crates/kern-py/python")
KERN_LIB_SO = Path("/workspace/kern/target/release/lib_kern.so")
KERNELS_DIR = ROOT / "build" / "sm_120"
WEIGHTS_DIR = Path("/workspace/weights/Inkling-2-Mini")
CUOBJDUMP = Path("/usr/local/cuda-13.5/bin/cuobjdump")


def _load_kern():
    try:
        import kern
        from kern.host import region

        return kern, region
    except ImportError:
        if str(KERN_PY_ROOT) not in sys.path:
            sys.path.insert(0, str(KERN_PY_ROOT))
        spec = importlib.util.spec_from_file_location("kern._kern", KERN_LIB_SO)
        ext = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ext)
        if not hasattr(ext, "HostGraph"):
            ext.HostGraph = None
        sys.modules["kern._kern"] = ext
        import kern
        from kern.host import region

        return kern, region


def _read_safetensors_header(path: Path) -> tuple[dict, int]:
    with path.open("rb") as f:
        hdr_len = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hdr_len))
    hdr.pop("__metadata__", None)
    return hdr, 8 + hdr_len


def _read_safetensors_slice(path: Path, hdr: dict, data_off: int, name: str):
    import torch

    info = hdr[name]
    lo, hi = info["data_offsets"]
    shape = info["shape"]
    dt_map = {
        "BF16": torch.bfloat16,
        "F32": torch.float32,
        "F8_E4M3": torch.float8_e4m3fn,
        "U8": torch.uint8,
    }
    dtype = dt_map[info["dtype"]]
    with path.open("rb") as f:
        f.seek(data_off + lo)
        raw = bytearray(f.read(hi - lo))
    return torch.frombuffer(raw, dtype=dtype).reshape(shape).clone()


def _ref_swizzle_scales(scales):
    """Reference 128x4 Sm1xxBlkScaledConfig swizzle for [..., R, C] where R%128==0, C%4==0."""
    *lead, r, c = scales.shape
    assert r % 128 == 0 and c % 4 == 0
    return (
        scales.reshape(-1, r // 128, 4, 32, c // 4, 4)
        .permute(0, 1, 4, 3, 2, 5)
        .reshape(scales.shape)
        .contiguous()
    )


def _ref_unswizzle_expert_scales(swizzled, offsets, scale_k: int):
    """Extract per-row [n, scale_k] scales from 128x4 padded expert scale buffer."""
    import torch

    n = int(offsets[-1].item())
    out = torch.empty((n, scale_k), dtype=swizzled.dtype, device=swizzled.device)
    flat = swizzled.reshape(-1)
    for e in range(128):
        m0 = int(offsets[e].item())
        m1 = int(offsets[e + 1].item())
        cnt = m1 - m0
        if cnt == 0:
            continue
        base = ((m0 + e * 127) // 128) * 128 * scale_k
        tiles_m = (cnt + 127) // 128
        block = flat[base : base + tiles_m * 128 * scale_k]
        unswiz = (
            block.reshape(tiles_m, scale_k // 4, 32, 4, 4)
            .permute(0, 3, 2, 1, 4)
            .reshape(tiles_m * 128, scale_k)
        )
        out[m0:m1] = unswiz[:cnt]
        # Verify padding rows in last 128-tile are zeroed
        if cnt % 128 != 0:
            tail = unswiz[cnt:]
            assert torch.all(tail == 0), f"Expert {e} tail scales not zeroed!"
    return out


def _ref_dequant_nvfp4(packed_u8, scales_fp8, amax_f32, denom: float = 2688.0, outer_div: float = 1536.0):
    """Dequantize packed NVFP4 [..., K/2] + E4M3 scales [..., K/16] + amax."""
    import torch

    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
    lo = packed_u8 & 0x0F
    hi = (packed_u8 >> 4) & 0x0F
    nibbles = torch.stack([lo, hi], dim=-1).reshape(*packed_u8.shape[:-1], -1).to(torch.long)
    vals = lut[nibbles & 0x7] * torch.where((nibbles & 0x8) != 0, -1.0, 1.0)
    vals = vals.reshape(*packed_u8.shape[:-1], -1, 16)
    sc = scales_fp8.to(torch.float32).unsqueeze(-1)
    outer = (amax_f32.to(torch.float32) / (outer_div * denom)).reshape(
        *amax_f32.shape, *([1] * (vals.ndim - amax_f32.ndim))
    )
    return (vals * sc * outer).reshape(*packed_u8.shape[:-1], -1)


def _ref_dequant_mxfp8(vals_fp8, exps_u8):
    """Dequantize MXFP8 [..., K] FP8 E4M3 + UE8M0 exponents [..., K/32]."""
    import torch

    v = vals_fp8.to(torch.float32).reshape(*vals_fp8.shape[:-1], -1, 32)
    e = exps_u8.to(torch.int32)
    # UE8M0: 0 -> subnormal 2^-127, 1..254 -> 2^(e-127), 255 -> NaN
    scale = torch.where(
        e == 0,
        torch.full_like(e, 2.0**-127, dtype=torch.float32),
        torch.pow(2.0, (e - 127).to(torch.float32)),
    )
    return (v * scale.unsqueeze(-1)).reshape(vals_fp8.shape)


def _ref_router_topk(logits: "torch.Tensor", bias: "torch.Tensor"):
    import torch

    sig = torch.sigmoid(logits.float())
    sel = sig + bias.float().unsqueeze(0)
    # Exact tie-breaking: descending sel, smaller expert index wins ties
    T, E = sel.shape
    bits = sel.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    ord_bits = torch.where((bits & 0x80000000) != 0, (~bits) & 0xFFFFFFFF, bits | 0x80000000)
    e_idx = torch.arange(E, dtype=torch.int64).unsqueeze(0).expand(T, E)
    keys = (ord_bits << 16) | (E - e_idx)
    top_keys, _ = torch.topk(keys, 8, dim=-1, largest=True, sorted=True)
    ids = (E - (top_keys & 0xFFFF)).to(torch.int32)
    sel_logits = torch.gather(logits.float(), 1, ids.to(torch.int64))
    log_p = torch.minimum(sel_logits, torch.zeros_like(sel_logits)) - torch.log1p(torch.exp(-sel_logits.abs()))
    probs = torch.softmax(log_p, dim=-1) * 8.0
    return ids, probs


def _ref_capped_swiglu(gate_up_f32, gamma_f32):
    import torch

    gate = gate_up_f32[..., 0::2].float()
    up = gate_up_f32[..., 1::2].float()
    g_cap = 4.0 * torch.tanh(gate / 4.0)
    u_cap = 25.0 * torch.tanh(up / 25.0)
    out = (g_cap * torch.sigmoid(gate) * u_cap) * gamma_f32.unsqueeze(-1)
    return out.to(torch.bfloat16).to(torch.float32)


def _attach_verify_serving_tail(m: dict, max_seqs: int = 8) -> dict:
    m["vars"]["seqs"] = {"max": max_seqs, "axis": "groups"}
    unembed_cubin = KERNELS_DIR / "unembed_sample.cubin"
    m["modules"]["unembed_sample"] = {
        "source": unembed_cubin.name,
        "sha256": hashlib.sha256(unembed_cubin.read_bytes()).hexdigest(),
    }
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
    for p in ("prefill", "decode", "decode_batch", "round"):
        m["programs"][p]["calls"].append(
            {
                "label": "test_next_token",
                "op": "_test_argmax",
                "args": [{"buf": "moe.logits"}, {"buf": "next_token"}, {"var": "seqs"}, {"i32": 128}],
            }
        )
    return m


def _build_verify_manifest(max_tokens: int = 2048, max_seqs: int = 8) -> dict:
    m = mm.build_unit_manifest(KERNELS_DIR, max_seqs=max_seqs, max_tokens=max_tokens, layers=(0, 20))
    return _attach_verify_serving_tail(m, max_seqs=max_seqs)


class TestMoeCPU(unittest.TestCase):
    def test_cubins_metadata_smem_and_no_tcgen05(self):
        meta = mm.load_meta(KERNELS_DIR)
        for mod in mm.M2_MODULES:
            cubin = KERNELS_DIR / f"{mod}.cubin"
            js = KERNELS_DIR / f"{mod}.json"
            self.assertTrue(cubin.exists(), f"Missing {cubin}")
            self.assertTrue(js.exists(), f"Missing {js}")
            m = json.loads(js.read_text())
            self.assertEqual(m["sha256"], hashlib.sha256(cubin.read_bytes()).hexdigest())
            self.assertEqual(m["architecture"], "sm_120a")
            self.assertEqual(m["sha256"], meta["modules"][mod]["sha256"])

            if CUOBJDUMP.exists():
                sass = subprocess.check_output([str(CUOBJDUMP), "-sass", str(cubin)], text=True)
                self.assertNotIn("tcgen05", sass.lower(), f"{mod}.cubin contains forbidden tcgen05")

        for mod in ("moe_nvfp4_sm120", "moe_mxfp8_sm120"):
            variants = meta["modules"][mod]["gemm_variants"]
            for vname in ("n32", "n64"):
                self.assertLessEqual(variants[vname]["shared_mem"], 101376)
                self.assertEqual(variants[vname]["abi"]["size"], 2304)

    def test_manifest_exports_and_kern_verify(self):
        for fn in ("register_modules", "register_buffers", "register_ops", "emit_derive_calls", "emit_moe_layer_calls"):
            self.assertTrue(callable(getattr(mm, fn)), f"Missing export {fn}")

        for max_tokens, max_seqs in ((32, 4), (1024, 8), (2048, 8)):
            with self.subTest(max_tokens=max_tokens, max_seqs=max_seqs):
                m = _build_verify_manifest(max_tokens=max_tokens, max_seqs=max_seqs)
                with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
                    json.dump(m, f)
                    p = f.name
                try:
                    res = subprocess.run([str(KERN_BIN), "verify", p], capture_output=True, text=True)
                    self.assertEqual(res.returncode, 0, f"kern verify failed:\n{res.stderr}\n{res.stdout}")
                finally:
                    Path(p).unlink(missing_ok=True)

    def test_swizzled_scale_workspace_sizing_and_ordering(self):
        expected_2048 = mm.max_swizzled_scale_rows(2048)
        self.assertEqual(expected_2048, 32768)
        expected_shapes = {
            "moe.xsf_nvfp4": [32768, 144],
            "moe.act_sf_nvfp4": [32768, 72],
            "moe.xsf_mxfp8": [32768, 72],
            "moe.act_sf_mxfp8": [32768, 36],
        }

        # 1. build_unit_manifest(..., max_tokens=2048) must size all 4 swizzled scale buffers to 32768
        m_unit = mm.build_unit_manifest(KERNELS_DIR, max_seqs=8, max_tokens=2048, layers=(0, 20))
        self.assertEqual(m_unit["vars"]["tokens"]["max"], 2048)
        for name, shape in expected_shapes.items():
            self.assertEqual(m_unit["buffers"][name]["shape"], shape, f"unit manifest {name} shape mismatch")

        # 2. emit_derive_calls() before register_buffers(..., max_tokens=2048) must not clamp to 1024 (24576)
        bufs_derive_first: dict = {}
        derive_calls: list[dict] = []
        modules: dict = {}
        ops: dict = {}
        meta = mm.register_modules(modules, KERNELS_DIR)
        bufs_derive_first["x_mlp"] = {"dtype": "bf16", "shape": ["tokens", mm.HIDDEN_SIZE], "kind": "input"}
        bufs_derive_first["moe_out"] = {"dtype": "bf16", "shape": ["tokens", mm.HIDDEN_SIZE], "kind": "output"}
        lw = mm.emit_derive_calls(bufs_derive_first, derive_calls, layers=(0, 20))
        for name in expected_shapes:
            self.assertNotIn(name, bufs_derive_first, f"emit_derive_calls unexpectedly pre-allocated {name}")
        mm.register_buffers(bufs_derive_first, max_seqs=8, max_tokens=2048, layers=(0, 20))
        for name, shape in expected_shapes.items():
            self.assertEqual(bufs_derive_first[name]["shape"], shape, f"derive-first {name} shape mismatch")

        mm.register_ops(ops, modules, meta, max_tokens=2048, max_seqs=8, target_sm={"sms": 110})

        def prog_calls(mode: str) -> list[dict]:
            return mm.emit_moe_layer_calls(mode, 0, 0, lw) + mm.emit_moe_layer_calls(mode, 1, 20, lw)

        m_derive_first = _attach_verify_serving_tail(
            {
                "schema_version": 7,
                "model": "inkling-2-mini-m2-derive-first",
                "vars": {"tokens": {"max": 2048, "axis": "rows"}},
                "states": {},
                "buffers": bufs_derive_first,
                "modules": modules,
                "ops": ops,
                "programs": {
                    "derive": {"derive": True, "calls": derive_calls},
                    "prefill": {"batch": {"groups": 1, "rows": "tokens"}, "calls": prog_calls("prefill")},
                    "decode": {"batch": {"groups": 1, "rows": 1}, "graph": True, "calls": prog_calls("decode")},
                    "decode_batch": {"batch": {"groups": 8, "rows": 1}, "graph": True, "calls": prog_calls("decode_batch")},
                    "round": {"batch": {"groups": 8, "rows": 8}, "graph": True, "calls": prog_calls("round")},
                },
            },
            max_seqs=8,
        )
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(m_derive_first, f)
            p = f.name
        try:
            res = subprocess.run([str(KERN_BIN), "verify", p], capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"kern verify (derive-first) failed:\n{res.stderr}\n{res.stdout}")
        finally:
            Path(p).unlink(missing_ok=True)

        # 3. Even if workspaces were first registered at max_tokens=1024, a later max_tokens=2048 grows them (and 1024 never shrinks 2048)
        bufs_grow: dict = {}
        mm.emit_derive_calls(bufs_grow, [], layers=(0, 20), max_tokens=1024)
        self.assertEqual(bufs_grow["moe.xsf_nvfp4"]["shape"][0], mm.max_swizzled_scale_rows(1024))
        mm.register_buffers(bufs_grow, max_seqs=8, max_tokens=2048, layers=(0, 20))
        for name, shape in expected_shapes.items():
            self.assertEqual(bufs_grow[name]["shape"], shape, f"grow {name} shape mismatch")
        mm.register_buffers(bufs_grow, max_seqs=8, max_tokens=1024, layers=(0, 20))
        for name, shape in expected_shapes.items():
            self.assertEqual(bufs_grow[name]["shape"], shape, f"no-shrink {name} shape mismatch")

    def test_checkpoint_tensor_bindings_match_safetensors(self):
        st_path = WEIGHTS_DIR / "model-00001-of-00001.safetensors"
        if not st_path.exists():
            self.skipTest(f"Checkpoint not present at {st_path}")
        hdr, _ = _read_safetensors_header(st_path)
        dt_map = {"BF16": "bf16", "F32": "f32", "F8_E4M3": "fp8e4m3", "U8": "u8"}
        buffers: dict = {}
        mm.register_buffers(buffers, max_tokens=1024, include_weights=True)
        checked = 0
        for b in buffers.values():
            for seg in b.get("bind", []):
                tname = seg["tensor"]
                self.assertIn(tname, hdr, f"Missing checkpoint tensor {tname}")
                info = hdr[tname]
                self.assertEqual(dt_map[info["dtype"]], b["dtype"], f"dtype mismatch for {tname}")
                self.assertEqual(info["shape"], b["shape"], f"shape mismatch for {tname}")
                checked += 1
        # 24 layers * (2 router + 8 nvfp4 / 4 mxfp8) = 20*10 + 4*6 = 224 tensors
        self.assertEqual(checked, 224)


class TestMoeGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from cuda.bindings import driver as cu

        assert torch.cuda.is_available(), "CUDA GPU required for TestMoeGPU"
        torch.cuda.init()
        cls.cu = cu
        cls.torch = torch
        kern_mod, region_fn = _load_kern()
        cls.kern = kern_mod
        cls.region = staticmethod(region_fn)
        cls.meta = mm.load_meta(KERNELS_DIR)
        cls.sm_count = int(torch.cuda.get_device_properties(0).multi_processor_count)
        cls._cu_mods = {}
        cls._cu_funcs = {}
        cls._cu_max_smem = {}

    @classmethod
    def tearDownClass(cls):
        for mod in cls._cu_mods.values():
            cls.cu.cuModuleUnload(mod)
        cls._cu_mods.clear()
        cls._cu_funcs.clear()
        cls._cu_max_smem.clear()

    def _get_func(self, mod_name: str, entry: str, max_dyn_smem: int = 0):
        cu = self.cu
        if mod_name not in self._cu_mods:
            err, mod = cu.cuModuleLoad(str(KERNELS_DIR / f"{mod_name}.cubin").encode())
            self.assertEqual(err, cu.CUresult.CUDA_SUCCESS)
            self._cu_mods[mod_name] = mod
        key = (mod_name, entry)
        if key not in self._cu_funcs:
            err, fn = cu.cuModuleGetFunction(self._cu_mods[mod_name], entry.encode())
            self.assertEqual(err, cu.CUresult.CUDA_SUCCESS)
            self._cu_funcs[key] = fn
            self._cu_max_smem[key] = 48 * 1024
        fn = self._cu_funcs[key]
        if max_dyn_smem > self._cu_max_smem[key]:
            (err,) = cu.cuFuncSetAttribute(
                fn,
                cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                max_dyn_smem,
            )
            self.assertEqual(err, cu.CUresult.CUDA_SUCCESS)
            self._cu_max_smem[key] = max_dyn_smem
        return fn

    def _launch(self, mod_name: str, entry: str, grid: tuple[int, int, int], block: tuple[int, int, int], args: list, smem: int = 0, stream=None):
        cu = self.cu
        fn = self._get_func(mod_name, entry, smem)
        c_args = []
        ptrs = []
        for a in args:
            if hasattr(a, "data_ptr"):
                val = ctypes.c_uint64(a.data_ptr())
            elif isinstance(a, ctypes.Array):
                val = a
            elif isinstance(a, int):
                val = ctypes.c_int32(a)
            elif isinstance(a, float):
                val = ctypes.c_float(a)
            else:
                raise TypeError(type(a))
            c_args.append(val)
            ptrs.append(ctypes.addressof(val))
        arr = (ctypes.c_void_p * len(ptrs))(*ptrs)
        s = stream if stream is not None else self.torch.cuda.current_stream().cuda_stream
        (err,) = cu.cuLaunchKernel(
            fn,
            grid[0], grid[1], grid[2],
            block[0], block[1], block[2],
            smem,
            s,
            arr,
            0,
        )
        self.assertEqual(err, cu.CUresult.CUDA_SUCCESS)

    def test_router_topk_and_tiebreak(self):
        torch = self.torch
        torch.manual_seed(101)
        bias = torch.randn(128, dtype=torch.float32, device="cuda") * 0.25

        for T in (1, 7, 8, 64, 512, 1024):
            logits = torch.randn((T, 128), dtype=torch.float32, device="cuda") * 1.5
            ids = torch.empty((T, 8), dtype=torch.int32, device="cuda")
            weights = torch.empty((T, 8), dtype=torch.float32, device="cuda")
            self._launch("moe_route", "moe_router_topk", ((T + 7) // 8, 1, 1), (256, 1, 1), [logits, bias, ids, weights, T])
            torch.cuda.synchronize()
            ref_ids, ref_w = _ref_router_topk(logits.cpu(), bias.cpu())
            self.assertTrue(torch.equal(ids.cpu(), ref_ids), f"TopK IDs mismatch at T={T}")
            torch.testing.assert_close(weights.cpu(), ref_w, atol=1e-5, rtol=1e-5)

        # Deterministic tie-breaking: force experts 10..25 to identical sigmoid(logit) + bias
        logits = torch.full((4, 128), -10.0, dtype=torch.float32, device="cuda")
        bias_zero = torch.zeros(128, dtype=torch.float32, device="cuda")
        logits[:, 10:26] = 0.0
        ids = torch.empty((4, 8), dtype=torch.int32, device="cuda")
        weights = torch.empty((4, 8), dtype=torch.float32, device="cuda")
        self._launch("moe_route", "moe_router_topk", (1, 1, 1), (256, 1, 1), [logits, bias_zero, ids, weights, 4])
        torch.cuda.synchronize()
        expected = torch.arange(10, 18, dtype=torch.int32).unsqueeze(0).expand(4, 8)
        self.assertTrue(torch.equal(ids.cpu(), expected))
        torch.testing.assert_close(weights.cpu(), torch.ones((4, 8), dtype=torch.float32), atol=1e-6, rtol=1e-6)

    def test_route_metadata_small_and_multicta_and_edge_cases(self):
        torch = self.torch
        torch.manual_seed(202)

        def check_sort(ids_t, w_t, use_small: bool):
            T = ids_t.shape[0]
            n = T * 8
            sorted_ids = torch.empty(n, dtype=torch.int32, device="cuda")
            src2dst = torch.empty(n, dtype=torch.int32, device="cuda")
            dst2src = torch.empty(n, dtype=torch.int32, device="cuda")
            gammas = torch.empty(n, dtype=torch.float32, device="cuda")
            counts = torch.empty(128, dtype=torch.int32, device="cuda")
            m_indptr = torch.empty(129, dtype=torch.int32, device="cuda")
            if use_small:
                smem = ((n + 31) // 32) * 256 + 1024
                self._launch(
                    "moe_route",
                    "moe_route_metadata_small",
                    (1, 1, 1),
                    (1024, 1, 1),
                    [ids_t, w_t, sorted_ids, src2dst, dst2src, gammas, counts, m_indptr, n],
                    smem=smem,
                )
            else:
                nb = (n + 255) // 256
                bc = torch.empty((nb, 128), dtype=torch.int32, device="cuda")
                bo = torch.empty((nb, 128), dtype=torch.int32, device="cuda")
                self._launch("moe_route", "moe_route_count", (nb, 1, 1), (256, 1, 1), [ids_t, bc, n])
                self._launch("moe_route", "moe_route_scan", (1, 1, 1), (128, 1, 1), [bc, bo, counts, m_indptr, n])
                self._launch("moe_route", "moe_route_scatter", (nb, 1, 1), (256, 1, 1), [ids_t, w_t, bo, sorted_ids, src2dst, dst2src, gammas, n])
            torch.cuda.synchronize()

            flat_ids = ids_t.reshape(-1).cpu()
            flat_w = w_t.reshape(-1).cpu()
            ref_perm = torch.argsort(flat_ids, stable=True)
            ref_sorted = flat_ids[ref_perm]
            ref_gammas = flat_w[ref_perm]
            ref_counts = torch.bincount(flat_ids, minlength=128).to(torch.int32)
            ref_indptr = torch.zeros(129, dtype=torch.int32)
            ref_indptr[1:] = torch.cumsum(ref_counts, 0)
            ref_src2dst = torch.empty_like(ref_perm, dtype=torch.int32)
            ref_src2dst[ref_perm] = torch.arange(n, dtype=torch.int32)

            self.assertTrue(torch.equal(sorted_ids.cpu(), ref_sorted))
            self.assertTrue(torch.equal(dst2src.cpu(), ref_perm.to(torch.int32)))
            self.assertTrue(torch.equal(src2dst.cpu(), ref_src2dst))
            self.assertTrue(torch.equal(counts.cpu(), ref_counts))
            self.assertTrue(torch.equal(m_indptr.cpu(), ref_indptr))
            torch.testing.assert_close(gammas.cpu(), ref_gammas, atol=0, rtol=0)

        for T in (1, 7, 8, 64, 512, 1024):
            ids = torch.stack([torch.randperm(128, device="cuda")[:8] for _ in range(T)]).to(torch.int32)
            w = torch.rand((T, 8), dtype=torch.float32, device="cuda")
            check_sort(ids, w, use_small=True)
            check_sort(ids, w, use_small=False)

        for T in (1025, 1536, 2048):
            ids = torch.randint(0, 128, (T, 8), dtype=torch.int32, device="cuda")
            w = torch.rand((T, 8), dtype=torch.float32, device="cuda")
            check_sort(ids, w, use_small=False)

        # Edge cases: empty experts (m=0), m=1, m=3, m=129
        edge_ids = torch.full((17, 8), 5, dtype=torch.int32, device="cuda")  # 136 rows
        flat = edge_ids.reshape(-1)
        flat[:129] = 10   # m=129 for expert 10
        flat[129:132] = 20  # m=3 for expert 20
        flat[132:133] = 30  # m=1 for expert 30
        flat[133:] = 40     # m=3 for expert 40 (rest 124 experts have m=0)
        w = torch.rand((17, 8), dtype=torch.float32, device="cuda")
        check_sort(edge_ids, w, use_small=True)

    def test_swizzle_scales_and_quant_nvfp4_mxfp8(self):
        torch = self.torch
        torch.manual_seed(303)

        # 1. Weight scale swizzle (NVFP4 & MXFP8 shapes)
        for n_rows, sk, dtype in ((2304, 144, torch.float8_e4m3fn), (2304, 72, torch.uint8), (2304, 36, torch.uint8)):
            src = torch.randint(0, 127, (128, n_rows, sk), dtype=torch.uint8, device="cuda").view(dtype)
            dst = torch.empty_like(src)
            total = 128 * n_rows * sk
            self._launch("moe_swiglu_quant", "moe_swizzle_weight_scales", ((total + 255) // 256, 1, 1), (256, 1, 1), [src, dst, 128, n_rows, sk])
            torch.cuda.synchronize()
            ref = _ref_swizzle_scales(src.view(torch.uint8).cpu())
            self.assertTrue(torch.equal(dst.view(torch.uint8).cpu(), ref))

        # 2. NVFP4 quant + SwiGLU quant
        T = 21
        n = T * 8  # 168 rows; assign m=129, m=3, m=1, etc.
        x = torch.randn((T, 2304), dtype=torch.bfloat16, device="cuda") * 0.5
        ids = torch.randint(0, 128, (T, 8), dtype=torch.int32, device="cuda")
        flat = ids.reshape(-1)
        flat[:129] = 2
        flat[129:132] = 7
        flat[132:133] = 15
        w = torch.rand((T, 8), dtype=torch.float32, device="cuda")

        sorted_ids = torch.empty(n, dtype=torch.int32, device="cuda")
        src2dst = torch.empty(n, dtype=torch.int32, device="cuda")
        dst2src = torch.empty(n, dtype=torch.int32, device="cuda")
        gammas = torch.empty(n, dtype=torch.float32, device="cuda")
        counts = torch.empty(128, dtype=torch.int32, device="cuda")
        m_indptr = torch.empty(129, dtype=torch.int32, device="cuda")
        self._launch(
            "moe_route", "moe_route_metadata_small", (1, 1, 1), (1024, 1, 1),
            [ids, w, sorted_ids, src2dst, dst2src, gammas, counts, m_indptr, n],
            smem=((n + 31) // 32) * 256 + 1024,
        )

        sf_rows = mm.max_swizzled_scale_rows(T)
        xq4 = torch.empty((n, 1152), dtype=torch.uint8, device="cuda")
        xsf4 = torch.full((sf_rows, 144), 0x7F, dtype=torch.uint8, device="cuda").view(torch.float8_e4m3fn)
        x_amax = torch.empty(n, dtype=torch.float32, device="cuda")
        self._launch("moe_swiglu_quant", "moe_quant_nvfp4", (n, 1, 1), (576, 1, 1), [x, dst2src, m_indptr, xq4, xsf4, x_amax, n])
        torch.cuda.synchronize()

        unswiz_xsf = _ref_unswizzle_expert_scales(xsf4.view(torch.uint8).cpu(), m_indptr.cpu(), 144).view(torch.float8_e4m3fn)
        dq_x = _ref_dequant_nvfp4(xq4.cpu(), unswiz_xsf, x_amax.cpu(), denom=1.0, outer_div=1536.0)
        ref_x = x.cpu()[dst2src.cpu() // 8].float()
        # NVFP4 quantization relative error is well below ~15% on normal values; check cosine similarity > 0.99
        cos = torch.nn.functional.cosine_similarity(dq_x.reshape(-1), ref_x.reshape(-1), dim=0)
        self.assertGreater(float(cos), 0.992)

        # NVFP4 SwiGLU quant
        w13_f32 = torch.randn((n, 2304), dtype=torch.float32, device="cuda") * 100.0
        w13_amax = torch.rand(128, dtype=torch.float32, device="cuda") * 0.3 + 0.1
        w13_denom = torch.tensor([2688.0], dtype=torch.float32, device="cuda")
        act4 = torch.empty((n, 576), dtype=torch.uint8, device="cuda")
        actsf4 = torch.full((sf_rows, 72), 0x7F, dtype=torch.uint8, device="cuda").view(torch.float8_e4m3fn)
        act_amax = torch.empty(n, dtype=torch.float32, device="cuda")
        self._launch(
            "moe_swiglu_quant", "moe_swiglu_quant_nvfp4", (n, 1, 1), (288, 1, 1),
            [w13_f32, x_amax, w13_amax, w13_denom, gammas, m_indptr, act4, actsf4, act_amax, n],
        )
        torch.cuda.synchronize()
        unswiz_act = _ref_unswizzle_expert_scales(actsf4.view(torch.uint8).cpu(), m_indptr.cpu(), 72).view(torch.float8_e4m3fn)
        dq_act = _ref_dequant_nvfp4(act4.cpu(), unswiz_act, act_amax.cpu(), denom=1.0, outer_div=1536.0)
        row_scale = (x_amax.cpu() * w13_amax.cpu()[sorted_ids.cpu()]) / (1536.0 * 2688.0)
        scaled_w13 = (w13_f32.cpu() * row_scale.unsqueeze(-1)).to(torch.bfloat16).float()
        ref_act = _ref_capped_swiglu(scaled_w13, gammas.cpu())
        cos_act = torch.nn.functional.cosine_similarity(dq_act.reshape(-1), ref_act.reshape(-1), dim=0)
        self.assertGreater(float(cos_act), 0.992)

        # 3. MXFP8 quant + SwiGLU quant
        xq8 = torch.empty((n, 2304), dtype=torch.float8_e4m3fn, device="cuda")
        xsf8 = torch.full((sf_rows, 72), 0x7F, dtype=torch.uint8, device="cuda")
        self._launch("moe_swiglu_quant", "moe_quant_mxfp8", (n, 1, 1), (160, 1, 1), [x, dst2src, m_indptr, xq8, xsf8, n])
        w13_bf16 = torch.randn((n, 2304), dtype=torch.bfloat16, device="cuda")
        act8 = torch.empty((n, 1152), dtype=torch.float8_e4m3fn, device="cuda")
        actsf8 = torch.full((sf_rows, 36), 0x7F, dtype=torch.uint8, device="cuda")
        self._launch("moe_swiglu_quant", "moe_swiglu_quant_mxfp8", (n, 1, 1), (96, 1, 1), [w13_bf16, gammas, m_indptr, act8, actsf8, n])
        torch.cuda.synchronize()

        unswiz_x8 = _ref_unswizzle_expert_scales(xsf8.cpu(), m_indptr.cpu(), 72)
        dq_x8 = _ref_dequant_mxfp8(xq8.cpu(), unswiz_x8)
        torch.testing.assert_close(dq_x8, ref_x, atol=8e-2, rtol=8e-2)

        unswiz_a8 = _ref_unswizzle_expert_scales(actsf8.cpu(), m_indptr.cpu(), 36)
        dq_a8 = _ref_dequant_mxfp8(act8.cpu(), unswiz_a8)
        ref_a8 = _ref_capped_swiglu(w13_bf16.cpu().float(), gammas.cpu())
        torch.testing.assert_close(dq_a8, ref_a8, atol=8e-2, rtol=8e-2)

    def _run_e2e_layer_via_kern(
        self,
        phys_idx: int,
        weights: dict,
        T_list: list[int],
        max_tokens: int = 1024,
        custom_inputs: dict[int, "torch.Tensor"] | None = None,
        check_skewed_m129: bool = False,
    ):
        torch = self.torch
        from safetensors.torch import save_file

        modules: dict = {}
        buffers: dict = {}
        ops: dict = {}
        derive_calls: list[dict] = []

        use_nvfp4 = phys_idx in mm.NVFP4_LAYERS
        use_mxfp8 = phys_idx in mm.MXFP8_LAYERS
        meta = mm.register_modules(modules, KERNELS_DIR, self.meta, include_nvfp4=use_nvfp4, include_mxfp8=use_mxfp8)
        buffers["x_mlp"] = {"dtype": "bf16", "shape": ["tokens", mm.HIDDEN_SIZE], "kind": "input"}
        buffers["moe_out"] = {"dtype": "bf16", "shape": ["tokens", mm.HIDDEN_SIZE], "kind": "output"}
        lw = mm.emit_derive_calls(buffers, derive_calls, layers=[phys_idx])
        mm.register_buffers(buffers, max_seqs=8, max_tokens=max_tokens, layers=[phys_idx])
        expected_sf_rows = mm.max_swizzled_scale_rows(max_tokens)
        if use_nvfp4:
            self.assertEqual(buffers["moe.xsf_nvfp4"]["shape"], [expected_sf_rows, 144])
            self.assertEqual(buffers["moe.act_sf_nvfp4"]["shape"], [expected_sf_rows, 72])
        if use_mxfp8:
            self.assertEqual(buffers["moe.xsf_mxfp8"]["shape"], [expected_sf_rows, 72])
            self.assertEqual(buffers["moe.act_sf_mxfp8"]["shape"], [expected_sf_rows, 36])
        mm.register_ops(
            ops,
            modules,
            meta,
            max_tokens=max_tokens,
            max_seqs=8,
            sm_count=self.sm_count,
            include_nvfp4=use_nvfp4,
            include_mxfp8=use_mxfp8,
        )
        calls = mm.emit_moe_layer_calls("prefill", 0, phys_idx, lw, "x_mlp", "moe_out")

        manifest = {
            "schema_version": 7,
            "model": f"m2-e2e-l{phys_idx}",
            "vars": {"tokens": {"max": max_tokens, "axis": "rows"}},
            "states": {},
            "buffers": buffers,
            "modules": modules,
            "ops": ops,
            "programs": {
                "derive": {"derive": True, "calls": derive_calls},
                "run": {"calls": calls},
            },
        }

        with tempfile.TemporaryDirectory() as tmp:
            wpath = Path(tmp) / "w.safetensors"
            mpath = Path(tmp) / "model.json"
            save_file(weights, str(wpath))
            mpath.write_text(json.dumps(manifest))
            rt = self.kern.Runtime(str(mpath), str(KERNELS_DIR), [str(wpath)], 0)
            views = {
                name: self.region(rt, manifest, name)
                for name in ("x_mlp", "moe_out")
            }

            # Precompute dequantized reference weights on CPU
            lp = f"model.llm.layers.{phys_idx}.mlp"
            gw = weights[f"{lp}.gate.weight"].float()
            gb = weights[f"{lp}.gate.bias"].float()
            if phys_idx in mm.NVFP4_LAYERS:
                w13 = _ref_dequant_nvfp4(
                    weights[f"{lp}.experts.w13_weight"],
                    weights[f"{lp}.experts.w13_weight_scale"],
                    weights[f"{lp}.experts.w13_weight_amax"],
                    float(weights[f"{lp}.experts.w13_weight_outer_scale_denominator"].item()),
                    outer_div=1.0,
                )
                w2 = _ref_dequant_nvfp4(
                    weights[f"{lp}.experts.w2_weight"],
                    weights[f"{lp}.experts.w2_weight_scale"],
                    weights[f"{lp}.experts.w2_weight_amax"],
                    float(weights[f"{lp}.experts.w2_weight_outer_scale_denominator"].item()),
                    outer_div=1.0,
                )
            else:
                w13 = _ref_dequant_mxfp8(
                    weights[f"{lp}.experts.w13_weight"],
                    weights[f"{lp}.experts.w13_weight_scale"],
                )
                w2 = _ref_dequant_mxfp8(
                    weights[f"{lp}.experts.w2_weight"],
                    weights[f"{lp}.experts.w2_weight_scale"],
                )

            for T in T_list:
                with self.subTest(phys_idx=phys_idx, T=T):
                    if custom_inputs is not None and T in custom_inputs:
                        x_cpu = custom_inputs[T]
                    else:
                        x_cpu = torch.randn((T, mm.HIDDEN_SIZE), dtype=torch.bfloat16) * 0.5
                    views["x_mlp"][:T].copy_(x_cpu)
                    torch.cuda.synchronize()
                    rt.run("run", {"tokens": T})
                    torch.cuda.synchronize()
                    got = views["moe_out"][:T].cpu().float()

                    # PyTorch reference MoE (vectorized per expert)
                    logits_ref = x_cpu.float() @ gw.T
                    ids_ref, wts_ref = _ref_router_topk(logits_ref, gb)
                    counts_ref = torch.bincount(ids_ref.reshape(-1), minlength=128)
                    if check_skewed_m129:
                        self.assertGreaterEqual(int(counts_ref.max().item()), 129)
                        self.assertGreaterEqual(int((counts_ref == 0).sum().item()), 100)
                    if T == 2048:
                        indptr_ref = torch.zeros(129, dtype=torch.int64)
                        indptr_ref[1:] = torch.cumsum(counts_ref, 0)
                        max_sf_row = max(
                            ((int(indptr_ref[e]) + e * 127) // 128) * 128 + ((int(counts_ref[e]) + 127) // 128) * 128
                            for e in range(128)
                            if int(counts_ref[e]) > 0
                        )
                        self.assertGreater(max_sf_row, 24576, "Expected T=2048 to exceed old 24576-row limit")
                        self.assertLessEqual(max_sf_row, expected_sf_rows)

                    ref_out = torch.zeros((T, mm.HIDDEN_SIZE), dtype=torch.float32)
                    flat_ids = ids_ref.reshape(-1)
                    flat_tok = torch.arange(T).unsqueeze(1).expand(T, 8).reshape(-1)
                    flat_gam = wts_ref.reshape(-1)
                    for e in range(128):
                        mask = flat_ids == e
                        if not torch.any(mask):
                            continue
                        tok_e = flat_tok[mask]
                        gam_e = flat_gam[mask]
                        gu = x_cpu[tok_e].float() @ w13[e].T
                        act = _ref_capped_swiglu(gu, gam_e)
                        down_e = (act @ w2[e].T).to(torch.bfloat16).float()
                        ref_out.index_add_(0, tok_e, down_e)

                    self.assertFalse(torch.isnan(got).any().item(), f"NaN in L{phys_idx} T={T}")
                    self.assertFalse(torch.isinf(got).any().item(), f"Inf in L{phys_idx} T={T}")
                    cos = torch.nn.functional.cosine_similarity(got.reshape(-1), ref_out.reshape(-1), dim=0)
                    min_cos = 0.965 if phys_idx in mm.NVFP4_LAYERS else 0.995
                    self.assertGreater(float(cos), min_cos, f"L{phys_idx} T={T} cosine={cos:.5f}")

    def test_e2e_nvfp4_and_mxfp8_real_checkpoint_via_kern(self):
        st_path = WEIGHTS_DIR / "model-00001-of-00001.safetensors"
        if not st_path.exists():
            self.skipTest(f"Checkpoint not present at {st_path}")
        hdr, data_off = _read_safetensors_header(st_path)
        self.torch.manual_seed(404)

        # Real checkpoint weights from Layer 0 (NVFP4) and Layer 20 (MXFP8)
        for phys_idx in (0, 20):
            lp = f"model.llm.layers.{phys_idx}.mlp"
            keys = [k for k in hdr if k.startswith(lp + ".")]
            weights = {k: _read_safetensors_slice(st_path, hdr, data_off, k) for k in keys}
            # Exercise TileN=32 (T <= 32), TileN=64 (T >= 33), single-CTA (T <= 1024),
            # multi-CTA (T=1025), and max_tokens=2048 (T=2048, n=16384 > 24576 scale-row boundary)
            self._run_e2e_layer_via_kern(phys_idx, weights, [1, 7, 8, 64, 512, 1024, 1025, 2048], max_tokens=2048)

    def test_skewed_single_expert_m_ge_129_e2e_via_kern(self):
        torch = self.torch
        st_path = WEIGHTS_DIR / "model-00001-of-00001.safetensors"
        if not st_path.exists():
            self.skipTest(f"Checkpoint not present at {st_path}")
        hdr, data_off = _read_safetensors_header(st_path)
        torch.manual_seed(606)

        for phys_idx in (0, 20):
            lp = f"model.llm.layers.{phys_idx}.mlp"
            keys = [k for k in hdr if k.startswith(lp + ".")]
            weights = {k: _read_safetensors_slice(st_path, hdr, data_off, k) for k in keys}
            gw = weights[f"{lp}.gate.weight"].float()
            base_dir = (gw[0] / gw[0].norm().clamp_min(1e-3)).unsqueeze(0).expand(160, -1)
            x_skew = (base_dir * 4.0).to(torch.bfloat16) + torch.randn((160, mm.HIDDEN_SIZE), dtype=torch.bfloat16) * 0.02
            self._run_e2e_layer_via_kern(
                phys_idx,
                weights,
                [160],
                max_tokens=2048,
                custom_inputs={160: x_skew},
                check_skewed_m129=True,
            )

    def test_cuda_graph_capture_and_replay_via_kern(self):
        torch = self.torch
        from safetensors.torch import save_file

        st_path = WEIGHTS_DIR / "model-00001-of-00001.safetensors"
        if not st_path.exists():
            self.skipTest(f"Checkpoint not present at {st_path}")
        hdr, data_off = _read_safetensors_header(st_path)
        torch.manual_seed(505)

        for phys_idx in (0, 20):
            lp = f"model.llm.layers.{phys_idx}.mlp"
            weights = {k: _read_safetensors_slice(st_path, hdr, data_off, k) for k in hdr if k.startswith(lp + ".")}

            modules: dict = {}
            buffers: dict = {}
            ops: dict = {}
            derive_calls: list[dict] = []
            use_nvfp4 = phys_idx in mm.NVFP4_LAYERS
            use_mxfp8 = phys_idx in mm.MXFP8_LAYERS
            meta = mm.register_modules(modules, KERNELS_DIR, self.meta, include_nvfp4=use_nvfp4, include_mxfp8=use_mxfp8)
            buffers["x_mlp"] = {"dtype": "bf16", "shape": ["tokens", mm.HIDDEN_SIZE], "kind": "input"}
            buffers["moe_out"] = {"dtype": "bf16", "shape": ["tokens", mm.HIDDEN_SIZE], "kind": "output"}
            mm.register_buffers(buffers, max_seqs=8, max_tokens=64, layers=[phys_idx])
            lw = mm.emit_derive_calls(buffers, derive_calls, layers=[phys_idx])
            mm.register_ops(
                ops,
                modules,
                meta,
                max_tokens=64,
                max_seqs=8,
                sm_count=self.sm_count,
                include_nvfp4=use_nvfp4,
                include_mxfp8=use_mxfp8,
            )
            calls = mm.emit_moe_layer_calls("round", 0, phys_idx, lw, "x_mlp", "moe_out")

            manifest = {
                "schema_version": 7,
                "model": f"m2-graph-l{phys_idx}",
                "vars": {"tokens": {"max": 64, "axis": "rows"}},
                "states": {},
                "buffers": buffers,
                "modules": modules,
                "ops": ops,
                "programs": {
                    "derive": {"derive": True, "calls": derive_calls},
                    "round_moe": {"calls": calls},
                },
            }

            with tempfile.TemporaryDirectory() as tmp:
                wpath = Path(tmp) / "w.safetensors"
                mpath = Path(tmp) / "model.json"
                save_file(weights, str(wpath))
                mpath.write_text(json.dumps(manifest))
                rt = self.kern.Runtime(str(mpath), str(KERNELS_DIR), [str(wpath)], 0)
                views = {
                    name: self.region(rt, manifest, name)
                    for name in ("x_mlp", "moe_out")
                }

                for T in (1, 8, 64):
                    x1 = torch.randn((T, mm.HIDDEN_SIZE), dtype=torch.bfloat16) * 0.5
                    x2 = torch.randn((T, mm.HIDDEN_SIZE), dtype=torch.bfloat16) * 0.7

                    # Eager reference for x2
                    views["x_mlp"][:T].copy_(x2)
                    torch.cuda.synchronize()
                    rt.run("round_moe", {"tokens": T})
                    torch.cuda.synchronize()
                    eager_x2 = views["moe_out"][:T].cpu().clone()

                    # Capture CUDA Graph at shape T
                    views["x_mlp"][:T].copy_(x1)
                    stream = torch.cuda.Stream()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.stream(stream):
                        rt.enqueue_after("round_moe", {"tokens": T}, stream.cuda_stream)
                    stream.synchronize()
                    with torch.cuda.graph(graph, stream=stream):
                        rt.enqueue_after("round_moe", {"tokens": T}, stream.cuda_stream)

                    # Replay with new input x2 (routes to different experts dynamically on device!)
                    views["x_mlp"][:T].copy_(x2)
                    torch.cuda.synchronize()
                    graph.replay()
                    torch.cuda.synchronize()
                    replayed_x2 = views["moe_out"][:T].cpu().clone()

                    self.assertTrue(
                        torch.equal(replayed_x2, eager_x2),
                        f"CUDA graph replay differed from eager for L{phys_idx} T={T}",
                    )


if __name__ == "__main__":
    unittest.main()
