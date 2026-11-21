# SPDX-License-Identifier: Apache-2.0
"""Unit tests for M4: DSpark 3-layer drafter, Inkling2MarkovHead, SpecRound, and rollback-safe commit.

Phase A (CPU-only):
  python -m unittest tests/unit/test_dspark.py
Phase B (GPU tests, only after GPU lock):
  KERN_RUN_GPU=1 python -m unittest tests/unit/test_dspark.py
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

import dspark_manifest as dm  # noqa: E402

KERN_BIN = Path("/workspace/kern/target/release/kern")
KERN_PY_DIR = Path("/workspace/kern/crates/kern-py/python")
LIB_KERN_SO = Path("/workspace/kern/target/release/lib_kern.so")
KERNELS_DIR = ROOT / "build/sm_120"

RUN_GPU = os.environ.get("KERN_RUN_GPU") == "1"


def _run_kern_verify(manifest: dict) -> subprocess.CompletedProcess:
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "model.json"
        path.write_text(json.dumps(manifest))
        return subprocess.run(
            [str(KERN_BIN), "verify", str(path)],
            capture_output=True,
            text=True,
        )


def _load_kern_runtime():
    """Load `kern` and `kern.host` without modifying external packages."""
    try:
        import kern
        from kern.host import Bytes, DTYPES, resolve

        def region_ex(rt, m: dict, name: str):
            import torch

            ptr, nbytes = rt.region(name)
            b = m["buffers"][name]
            shape = [resolve(m, m["vars"][d]["max"] if d in m.get("vars", {}) else d) for d in b["shape"]]
            dt_map = {**DTYPES, "fp8e4m3": torch.float8_e4m3fn}
            return torch.as_tensor(Bytes(ptr, nbytes), device="cuda").view(dt_map[b["dtype"]]).view(*shape)

        return kern, region_ex
    except ImportError:
        tmp_pkg = Path(tempfile.mkdtemp(prefix="kern_py_shim_"))
        pkg_dir = tmp_pkg / "kern"
        pkg_dir.mkdir()
        for fname in ("host.py", "group.py", "trace.py"):
            shutil.copy2(KERN_PY_DIR / "kern" / fname, pkg_dir / fname)
        os.symlink(LIB_KERN_SO, pkg_dir / "_kern.so")
        (pkg_dir / "__init__.py").write_text(
            "from kern._kern import Runtime, __version__\n"
            "try:\n"
            "    from kern._kern import HostGraph\n"
            "except ImportError:\n"
            "    HostGraph = None\n"
        )
        sys.path.insert(0, str(tmp_pkg))
        import kern
        from kern.host import Bytes, DTYPES, resolve

        def region_ex(rt, m: dict, name: str):
            import torch

            ptr, nbytes = rt.region(name)
            b = m["buffers"][name]
            shape = [resolve(m, m["vars"][d]["max"] if d in m.get("vars", {}) else d) for d in b["shape"]]
            dt_map = {**DTYPES, "fp8e4m3": torch.float8_e4m3fn}
            return torch.as_tensor(Bytes(ptr, nbytes), device="cuda").view(dt_map[b["dtype"]]).view(*shape)

        return kern, region_ex


class TestDsparkCPU(unittest.TestCase):
    def test_sm120_artifacts_and_ptx_safety(self):
        dsl_path = KERNELS_DIR / "dspark_dsl.json"
        spec_path = KERNELS_DIR / "spec_build.json"
        self.assertTrue(dsl_path.exists(), f"Missing {dsl_path}; run dspark_manifest.py --build")
        self.assertTrue(spec_path.exists(), f"Missing {spec_path}; run dspark_manifest.py --build")

        dsl = json.loads(dsl_path.read_text())
        self.assertEqual(dsl["target"], "cuda:120:32")
        for name, info in dsl["modules"].items():
            self.assertTrue(name.startswith("dspark_"))
            cubin = KERNELS_DIR / info["source"]
            self.assertTrue(cubin.exists(), f"Missing {cubin}")
            self.assertEqual(hashlib.sha256(cubin.read_bytes()).hexdigest(), info["sha256"])
            ptx = (KERNELS_DIR / f"{name}.ptx").read_text()
            for forbidden in ("tcgen05", "utmastg", "wgmma.mma_async"):
                self.assertNotIn(forbidden, ptx, f"{name} PTX contains {forbidden}")

        spec = json.loads(spec_path.read_text())
        self.assertEqual(len(spec["sampling"]["entries"]), 4)
        for name, info in spec["modules"].items():
            cubin = KERNELS_DIR / info["source"]
            self.assertTrue(cubin.exists(), f"Missing {cubin}")
            self.assertEqual(hashlib.sha256(cubin.read_bytes()).hexdigest(), info["sha256"])

    def test_standalone_dspark_manifest_kern_verify_p64(self):
        m = dm.build_standalone_test_manifest(KERNELS_DIR, page_size=64, num_kv_splits=16)
        proc = _run_kern_verify(m)
        self.assertEqual(proc.returncode, 0, f"kern verify failed:\n{proc.stdout}\n{proc.stderr}")
        out = proc.stdout + proc.stderr
        self.assertIn("round", out)
        self.assertIn("verify_tokens", out)
        self.assertIn("accepted", out)
        self.assertIn("verified", out)

    def test_standalone_dspark_manifest_kern_verify_p128_direct(self):
        m = dm.build_standalone_test_manifest(KERNELS_DIR, page_size=128, num_kv_splits=1)
        proc = _run_kern_verify(m)
        self.assertEqual(proc.returncode, 0, f"kern verify failed:\n{proc.stdout}\n{proc.stderr}")

    def test_negative_manifest_rejections(self):
        base = dm.build_standalone_test_manifest(KERNELS_DIR)

        # (a) Duplicate rows=1 program (decode + decode_batch forbidden by protocol.rs)
        m_dup = json.loads(json.dumps(base))
        m_dup["programs"]["decode_batch"] = json.loads(json.dumps(m_dup["programs"]["decode"]))
        p = _run_kern_verify(m_dup)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("a shape names one program", p.stdout + p.stderr)

        # (b) input buffer used inside derive program (rejected by verify.rs:982)
        m_var = json.loads(json.dumps(base))
        m_var["programs"]["derive"]["calls"][0]["args"][0] = {"buf": "token_ids"}
        p = _run_kern_verify(m_var)
        self.assertNotEqual(p.returncode, 0)

        # (c) Partial sampling fills (missing seed)
        m_samp = json.loads(json.dumps(base))
        del m_samp["buffers"]["seed"]["fill"]
        p = _run_kern_verify(m_samp)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("sampling fills", p.stdout + p.stderr)

        # (d) index_into combined with min/max on same domain
        m_dom = json.loads(json.dumps(base))
        m_dom["buffers"]["pages"]["domain"]["max"] = 100
        p = _run_kern_verify(m_dom)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("mutually exclusive", p.stdout + p.stderr)

        # (e) d.lines stride not dividing dspark_state bytes_per_seq
        m_stride = json.loads(json.dumps(base))
        m_stride["buffers"]["d.lines"]["domain"]["stride"] = 123
        p = _run_kern_verify(m_stride)
        self.assertNotEqual(p.returncode, 0)
        self.assertIn("divide", p.stdout + p.stderr)

    def test_custom_lines_buf_kern_verify(self):
        m = dm.build_standalone_test_manifest(KERNELS_DIR, page_size=64, num_kv_splits=16)
        layout = dm.DsparkTargetLayout()
        m["buffers"]["custom.lines"] = m["buffers"].pop("d.lines")

        pre_hooks, pre_tail = dm.emit_prefill_capture(m, layout, lines_buf="custom.lines")
        pre_calls = [
            dm._call("stub_check_meta", dm._b("valid"), dm._b("q_indptr"), label="check_meta"),
            dm._call("stub_embed", dm._b("token_ids"), dm._i32(1), dm._b(layout.target_embed_buf), dm._b("h"), dm._i32(dm.DSPARK_HIDDEN), label="embed"),
            dm._call("dspark_tap_capture_0", dm._b("h"), dm._b("d.taps"), dm._i32(0), dm._i32(dm.DSPARK_HIDDEN), dm._i32(4), label="touch_pos_valid_qindptr"),
        ]
        for vid in dm.TARGET_LAYER_IDS:
            pre_calls.extend(pre_hooks[vid])
        pre_calls.extend(pre_tail)
        pre_calls += [
            dm._call("dspark_gemm_bf16_f32", dm._b("h"), dm._b(layout.target_lm_head_buf), dm._b(layout.target_verify_logits_buf), dm._var("seqs"), dm._i32(dm.VOCAB_PADDED), dm._i32(dm.DSPARK_HIDDEN), label="lm_head"),
            dm._call("sample", dm._b(layout.target_verify_logits_buf), dm._b("lengths"), dm._b("temperature"), dm._b("top_k"), dm._b("top_p"), dm._b("min_p"), dm._b("seed"), dm._b("next_token"), dm._var("seqs"), label="sample"),
        ]
        m["programs"]["prefill"]["calls"] = pre_calls

        dec_hooks, dec_tail = dm.emit_decode_capture(m, layout, lines_buf="custom.lines")
        dec_calls = [dm._call("stub_embed", dm._b("token_ids"), dm._i32(1), dm._b(layout.target_embed_buf), dm._b("h"), dm._i32(dm.DSPARK_HIDDEN), label="embed")]
        for vid in dm.TARGET_LAYER_IDS:
            dec_calls.extend(dec_hooks[vid])
        dec_calls.extend(dec_tail)
        dec_calls += [
            dm._call("dspark_gemm_bf16_f32", dm._b("h"), dm._b(layout.target_lm_head_buf), dm._b(layout.target_verify_logits_buf), dm._var("seqs"), dm._i32(dm.VOCAB_PADDED), dm._i32(dm.DSPARK_HIDDEN), label="lm_head"),
            dm._call("sample", dm._b(layout.target_verify_logits_buf), dm._b("lengths"), dm._b("temperature"), dm._b("top_k"), dm._b("top_p"), dm._b("min_p"), dm._b("seed"), dm._b("next_token"), dm._var("seqs"), label="sample"),
        ]
        m["programs"]["decode"]["calls"] = dec_calls

        verify_calls = [
            dm._call("stub_embed", dm._b("d.verify_ids"), dm._i32(1), dm._b(layout.target_embed_buf), dm._b("h"), dm._i32(dm.DSPARK_HIDDEN), label="verify.embed"),
            dm._call("stub_copy_sconv", dm._b(layout.target_sconv_raw_buf), dm._i32(1), label="verify.stage_sconv"),
            dm._call("stub_zero_swa", dm._b(layout.target_swa_kv_buf), label="verify.stage_swa"),
        ] + [
            dm._call("dspark_tap_capture_0" if idx == 0 else "dspark_tap_capture", dm._b("h"), dm._b("d.taps"), dm._i32(idx), dm._i32(dm.DSPARK_HIDDEN), dm._i32(len(dm.TARGET_LAYER_IDS)), label=f"verify.tap.{vid}")
            for idx, vid in enumerate(dm.TARGET_LAYER_IDS)
        ] + [
            dm._call("dspark_gemm_bf16_f32", dm._b("h"), dm._b(layout.target_lm_head_buf), dm._b(layout.target_verify_logits_buf), dm._var("tokens"), dm._i32(dm.VOCAB_PADDED), dm._i32(dm.DSPARK_HIDDEN), label="verify.lm_head")
        ]
        m["programs"]["round"] = dm.emit_round_program(m, layout, verify_calls, lines_buf="custom.lines")

        self.assertNotIn('"d.lines"', json.dumps(m["programs"]))
        ring_lines = [
            c["args"][4]["buf"]
            for c in m["programs"]["round"]["calls"]
            if c["op"] == "dspark_attn_ring"
        ]
        self.assertEqual(ring_lines, ["custom.lines", "custom.lines"])
        proc = _run_kern_verify(m)
        self.assertEqual(proc.returncode, 0, f"kern verify failed with custom.lines:\n{proc.stdout}\n{proc.stderr}")


@unittest.skipUnless(RUN_GPU, "GPU tests skipped in Phase A (set KERN_RUN_GPU=1 in Phase B)")
class TestDsparkGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA GPU not available")
        cls.torch = torch
        kern_mod, region_fn = _load_kern_runtime()
        cls.kern = kern_mod
        cls.region = staticmethod(region_fn)
        cls.dsl = json.loads((KERNELS_DIR / "dspark_dsl.json").read_text())["modules"]
        cls.spec = json.loads((KERNELS_DIR / "spec_build.json").read_text())
        major, minor = torch.cuda.get_device_capability()
        cls.arch = f"sm_{major}{minor}"

        # Query real GPU attributes for sampling
        import ctypes

        driver = ctypes.CDLL("libcuda.so.1")
        dev = ctypes.c_int()
        driver.cuDeviceGet(ctypes.byref(dev), 0)
        sms, smem = ctypes.c_int(), ctypes.c_int()
        driver.cuDeviceGetAttribute(ctypes.byref(sms), 16, dev)
        driver.cuDeviceGetAttribute(ctypes.byref(smem), 97, dev)
        cls.sampling_target = {"sms": sms.value, "shared_mem": smem.value}

    def _run_manifest(self, m: dict, setup_fn, program: str, vars_dict: dict):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "model.json"
            p.write_text(json.dumps(m))
            rt = self.kern.Runtime(str(p), str(KERNELS_DIR), [], 0)
        views = {name: self.region(rt, m, name) for name in m["buffers"]}
        setup_fn(rt, views)
        self.torch.cuda.synchronize()
        rt.run(program, vars_dict)
        self.torch.cuda.synchronize()
        return rt, views

    def _triton_mod(self, name: str):
        info = self.dsl[name]
        return {name: {"source": info["source"], "sha256": info["sha256"]}}

    def _spec_mod(self, name: str):
        info = self.spec["modules"][name]
        return {name: {"source": info["source"], "sha256": info["sha256"]}}

    def test_dflare_mix_rmsnorm(self):
        torch = self.torch
        for strided in (False, True):
            for rows in (1, 17):
                with self.subTest(strided=strided, rows=rows):
                    g = torch.Generator().manual_seed(851 + rows)
                    hidden = dm.DSPARK_HIDDEN
                    taps = 4
                    features = torch.randn(rows, taps, hidden * 2, generator=g).bfloat16().cuda()
                    feats = features[..., ::2] if strided else features[..., :hidden]
                    alpha = torch.randn(taps, generator=g).softmax(-1).bfloat16().cuda()
                    weight = (0.5 + torch.rand(hidden, generator=g)).bfloat16().cuda()

                    rec = self.dsl["dspark_mix_rmsnorm"]
                    params = ["in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i64", "i64", "i64"]
                    m = {
                        "schema_version": 7,
                        "model": "mix-rmsnorm-test",
                        "vars": {},
                        "states": {},
                        "buffers": {
                            "X": {"dtype": "bf16", "shape": [rows, taps, hidden * 2], "kind": "input"},
                            "ALPHA": {"dtype": "bf16", "shape": [taps], "kind": "input"},
                            "WEIGHT": {"dtype": "bf16", "shape": [hidden], "kind": "input"},
                            "OUT": {"dtype": "bf16", "shape": [rows, hidden], "kind": "output"},
                        },
                        "modules": self._triton_mod("dspark_mix_rmsnorm"),
                        "ops": {
                            "mix": {
                                "params": params,
                                "impl": {
                                    "launches": [
                                        {
                                            "module": "dspark_mix_rmsnorm",
                                            "entry": rec["entry"],
                                            "params": params + rec["abi_tail"],
                                            "grid": [rows, 1, 1],
                                            "block": rec["block"],
                                            "shared_mem": rec["shared_mem"],
                                            "args": [{"param": j} for j in range(len(params))] + [{"i64": 0}, {"i64": 0}],
                                        }
                                    ]
                                },
                            }
                        },
                        "programs": {
                            "run": {
                                "calls": [
                                    {
                                        "op": "mix",
                                        "args": [
                                            {"buf": "X"},
                                            {"buf": "ALPHA"},
                                            {"buf": "WEIGHT"},
                                            {"buf": "OUT"},
                                            {"i64": feats.stride(0)},
                                            {"i64": feats.stride(1)},
                                            {"i64": feats.stride(2)},
                                        ],
                                    }
                                ]
                            }
                        },
                    }

                    def setup(rt, v):
                        v["X"].copy_(features)
                        v["ALPHA"].copy_(alpha)
                        v["WEIGHT"].copy_(weight)

                    _, views = self._run_manifest(m, setup, "run", {})
                    mixed = (feats.float() * alpha.float()[:, None]).sum(1)
                    ref = mixed * torch.rsqrt(mixed.square().mean(-1, keepdim=True) + dm.DSPARK_RMS_EPS)
                    ref = (ref * weight.float()).bfloat16()
                    torch.testing.assert_close(views["OUT"], ref, atol=0, rtol=1e-2)

    def test_qk_norm_rope_1m_and_silu_mul(self):
        torch = self.torch
        g = torch.Generator().manual_seed(852)
        T = 4
        qkv = torch.randn(T, 5120, generator=g).bfloat16().cuda()
        q_w = (0.5 + torch.rand(128, generator=g)).bfloat16().cuda()
        k_w = (0.5 + torch.rand(128, generator=g)).bfloat16().cuda()
        pos = torch.tensor([0, 17, 8192, 1048575], dtype=torch.int64, device="cuda")
        inv_freq = 1.0 / (dm.DSPARK_ROPE_THETA ** (torch.arange(0, 128, 2, dtype=torch.float32, device="cuda") / 128.0))
        gate_up = torch.randn(T, 2 * dm.DSPARK_INTER, generator=g).bfloat16().cuda()

        rec_rope = self.dsl["dspark_qk_norm_rope"]
        rec_silu = self.dsl["dspark_silu_mul"]
        p_rope = ["inout buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "in buffer<f32>", "in buffer<i64>", "i64", "i64"]
        p_silu = ["in buffer<bf16>", "out buffer<bf16>", "i64"]
        m = {
            "schema_version": 7,
            "model": "rope-silu-test",
            "vars": {},
            "states": {},
            "buffers": {
                "QKV": {"dtype": "bf16", "shape": [T, 5120], "kind": "inout"},
                "QW": {"dtype": "bf16", "shape": [128], "kind": "input"},
                "KW": {"dtype": "bf16", "shape": [128], "kind": "input"},
                "INV": {"dtype": "f32", "shape": [64], "kind": "input"},
                "POS": {"dtype": "i64", "shape": [T], "kind": "input"},
                "GU": {"dtype": "bf16", "shape": [T, 2 * dm.DSPARK_INTER], "kind": "input"},
                "ACT": {"dtype": "bf16", "shape": [T, dm.DSPARK_INTER], "kind": "output"},
            },
            "modules": {**self._triton_mod("dspark_qk_norm_rope"), **self._triton_mod("dspark_silu_mul")},
            "ops": {
                "rope": {
                    "params": p_rope,
                    "impl": {
                        "launches": [
                            {
                                "module": "dspark_qk_norm_rope",
                                "entry": rec_rope["entry"],
                                "params": p_rope + rec_rope["abi_tail"],
                                "grid": [T, 32, 1],
                                "block": rec_rope["block"],
                                "shared_mem": rec_rope["shared_mem"],
                                "args": [{"param": j} for j in range(len(p_rope))] + [{"i64": 0}, {"i64": 0}],
                            }
                        ]
                    },
                },
                "silu": {
                    "params": p_silu,
                    "impl": {
                        "launches": [
                            {
                                "module": "dspark_silu_mul",
                                "entry": rec_silu["entry"],
                                "params": p_silu + rec_silu["abi_tail"],
                                "grid": [T, 9, 1],
                                "block": rec_silu["block"],
                                "shared_mem": rec_silu["shared_mem"],
                                "args": [{"param": j} for j in range(len(p_silu))] + [{"i64": 0}, {"i64": 0}],
                            }
                        ]
                    },
                },
            },
            "programs": {
                "run": {
                    "calls": [
                        {"op": "rope", "args": [{"buf": "QKV"}, {"buf": "QW"}, {"buf": "KW"}, {"buf": "INV"}, {"buf": "POS"}, {"i64": 5120}, {"i64": 3072}]},
                        {"op": "silu", "args": [{"buf": "GU"}, {"buf": "ACT"}, {"i64": 2 * dm.DSPARK_INTER}]},
                    ]
                }
            },
        }

        def setup(rt, v):
            v["QKV"].copy_(qkv)
            v["QW"].copy_(q_w)
            v["KW"].copy_(k_w)
            v["INV"].copy_(inv_freq)
            v["POS"].copy_(pos)
            v["GU"].copy_(gate_up)

        _, views = self._run_manifest(m, setup, "run", {})

        # Reference Neox RoPE + QK-RMSNorm
        ref_qkv = qkv.clone()
        freqs = pos.float()[:, None] * inv_freq[None, :]
        cos = freqs.cos()[:, None, :]
        sin = freqs.sin()[:, None, :]
        for sl, nh, w in ((slice(0, 3072), 24, q_w), (slice(3072, 4096), 8, k_w)):
            sub = ref_qkv[:, sl].view(T, nh, 128).float()
            inv = torch.rsqrt(sub.square().mean(-1, keepdim=True) + dm.DSPARK_RMS_EPS)
            normed = (sub * inv * w.float()).bfloat16().float()
            x1, x2 = normed[..., :64], normed[..., 64:]
            o1 = x1 * cos - x2 * sin
            o2 = x2 * cos + x1 * sin
            ref_qkv[:, sl] = torch.cat([o1, o2], dim=-1).bfloat16().view(T, nh * 128)
        torch.testing.assert_close(views["QKV"], ref_qkv, atol=1e-2, rtol=1e-2)
        # V slice must be untouched
        torch.testing.assert_close(views["QKV"][:, 4096:], qkv[:, 4096:], atol=0, rtol=0)

        g_in, u_in = gate_up.float().chunk(2, dim=-1)
        ref_act = (torch.nn.functional.silu(g_in) * u_in).bfloat16()
        torch.testing.assert_close(views["ACT"], ref_act, atol=1e-2, rtol=1e-2)

    def test_dspark_derive_fold_fp64_and_prep(self):
        torch = self.torch
        g = torch.Generator().manual_seed(853)
        V, H, R = 1024, 2304, 256
        lm_head = (torch.randn(V, H, generator=g) * 0.03).bfloat16().cuda()
        hidden_w = (torch.randn(H, R, generator=g) * 0.03).bfloat16().cuda()
        hidden_b = (torch.randn(H, generator=g) * 0.03).bfloat16().cuda()
        fuse_w = torch.randn(4, 4, generator=g).bfloat16().cuda()
        rnn_w = torch.randn(768, 5120, generator=g).bfloat16().cuda()
        rnn_b = torch.randn(768, generator=g).bfloat16().cuda()

        p_fw = ["in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32", "i32"]
        p_fb = ["in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32"]
        p_pr = [
            "in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>",
            "out buffer<bf16>", "out buffer<bf16>", "out buffer<bf16>", "out buffer<f32>", "out buffer<f32>",
            "i32", "i32", "i32", "i32", "i32", "i32", "f32",
        ]
        m = {
            "schema_version": 7,
            "model": "derive-test",
            "vars": {},
            "states": {},
            "buffers": {
                "lm": {"dtype": "bf16", "shape": [V, H], "kind": "input"},
                "hw": {"dtype": "bf16", "shape": [H, R], "kind": "input"},
                "hb": {"dtype": "bf16", "shape": [H], "kind": "input"},
                "fuse_w": {"dtype": "bf16", "shape": [4, 4], "kind": "input"},
                "rnn_w": {"dtype": "bf16", "shape": [768, 5120], "kind": "input"},
                "rnn_b": {"dtype": "bf16", "shape": [768], "kind": "input"},
                "fw": {"dtype": "bf16", "shape": [V, R], "kind": "output"},
                "fb": {"dtype": "bf16", "shape": [V], "kind": "output"},
                "alpha": {"dtype": "bf16", "shape": [4, 4], "kind": "output"},
                "ws": {"dtype": "bf16", "shape": [768, 512], "kind": "output"},
                "wc": {"dtype": "bf16", "shape": [768, 4608], "kind": "output"},
                "rb": {"dtype": "f32", "shape": [768], "kind": "output"},
                "inv": {"dtype": "f32", "shape": [64], "kind": "output"},
            },
            "modules": self._spec_mod("dspark_derive"),
            "ops": {
                "fold_w": {"params": p_fw, "impl": {"launches": [{"module": "dspark_derive", "entry": "dspark_fold_fp64", "params": p_fw, "grid": [dm.ceil_div(V, 32), dm.ceil_div(R, 32), 1], "block": [32, 8, 1], "args": [{"param": j} for j in range(6)]}]}},
                "fold_b": {"params": p_fb, "impl": {"launches": [{"module": "dspark_derive", "entry": "dspark_fold_bias_fp64", "params": p_fb, "grid": [dm.ceil_div(V, 8), 1, 1], "block": [256, 1, 1], "args": [{"param": j} for j in range(5)]}]}},
                "prep": {"params": p_pr, "impl": {"launches": [{"module": "dspark_derive", "entry": "dspark_prep_weights", "params": p_pr, "grid": [dm.ceil_div(768 * 5120, 256), 1, 1], "block": [256, 1, 1], "args": [{"param": j} for j in range(15)]}]}},
            },
            "programs": {
                "run": {
                    "calls": [
                        {"op": "fold_w", "args": [{"buf": "lm"}, {"buf": "hw"}, {"buf": "fw"}, {"i32": V}, {"i32": H}, {"i32": R}]},
                        {"op": "fold_b", "args": [{"buf": "lm"}, {"buf": "hb"}, {"buf": "fb"}, {"i32": V}, {"i32": H}]},
                        {"op": "prep", "args": [{"buf": "fuse_w"}, {"buf": "rnn_w"}, {"buf": "rnn_b"}, {"buf": "alpha"}, {"buf": "ws"}, {"buf": "wc"}, {"buf": "rb"}, {"buf": "inv"}, {"i32": 4}, {"i32": 4}, {"i32": 768}, {"i32": 512}, {"i32": 4608}, {"i32": 128}, {"f32": 1e6}]},
                    ]
                }
            },
        }

        def setup(rt, v):
            v["lm"].copy_(lm_head)
            v["hw"].copy_(hidden_w)
            v["hb"].copy_(hidden_b)
            v["fuse_w"].copy_(fuse_w)
            v["rnn_w"].copy_(rnn_w)
            v["rnn_b"].copy_(rnn_b)

        _, views = self._run_manifest(m, setup, "run", {})
        ref_fw = (lm_head.double() @ hidden_w.double()).bfloat16()
        ref_fb = (lm_head.double() @ hidden_b.double()).bfloat16()
        torch.testing.assert_close(views["fw"], ref_fw, atol=0, rtol=2**-7)
        torch.testing.assert_close(views["fb"], ref_fb, atol=0, rtol=2**-7)
        torch.testing.assert_close(views["alpha"], torch.softmax(fuse_w.float(), dim=-1).bfloat16(), atol=1e-2, rtol=1e-2)
        self.assertTrue(torch.equal(views["ws"], rnn_w[:, :512]))
        self.assertTrue(torch.equal(views["wc"], rnn_w[:, 512:]))
        self.assertTrue(torch.equal(views["rb"], rnn_b.float()))
        ref_inv = 1.0 / (1e6 ** (torch.arange(0, 128, 2, dtype=torch.float32, device="cuda") / 128.0))
        torch.testing.assert_close(views["inv"], ref_inv, atol=1e-7, rtol=1e-6)

    def test_markov_head_gate_softcap_and_7step_loop(self):
        torch = self.torch
        g = torch.Generator().manual_seed(854)
        seqs = 3
        V, VP, H, R = dm.VOCAB_UNPADDED, dm.VOCAB_PADDED, dm.DSPARK_HIDDEN, dm.DSPARK_MARKOV_RANK
        raw_ctx = torch.randn(seqs, 7, 3 * R, generator=g).mul(2).float().cuda()
        raw_st = torch.randn(seqs, 3 * R, generator=g).mul(2).float().cuda()
        state_in = torch.randn(seqs, R, generator=g).bfloat16().cuda()
        base_logits = (torch.randn(seqs, 7, VP, generator=g) * 10.0).bfloat16().cuda()
        step_bias = torch.randn(seqs, VP, generator=g).bfloat16().cuda()
        fold_bias = torch.randn(VP, generator=g).bfloat16().cuda()
        # Ensure max token sits in unpadded range or padded range to test mask >= 200058!
        base_logits[:, 2, V + 10 :] = 100.0
        base_logits[:, 2, 42] = 5.0

        rec_gate = self.dsl["dspark_markov_gate_add"]
        rec_cap = self.dsl["dspark_add_softcap_mask"]
        p_gate = ["in buffer<f32>", "in buffer<f32>", "in buffer<bf16>", "out buffer<bf16>", "out buffer<bf16>", "i32"]
        p_cap = ["in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i32", "f32"]
        p_am1 = ["in buffer<bf16>", "i64", "out buffer<f32>", "out buffer<i32>", "i32"]
        p_am2 = ["in buffer<f32>", "in buffer<i32>", "out buffer<i64>", "i32", "i32"]
        m = {
            "schema_version": 7,
            "model": "markov-gate-softcap-test",
            "vars": {},
            "states": {},
            "buffers": {
                "ctx": {"dtype": "f32", "shape": [seqs, 7, 3 * R], "kind": "input"},
                "sp": {"dtype": "f32", "shape": [seqs, 3 * R], "kind": "input"},
                "st_in": {"dtype": "bf16", "shape": [seqs, R], "kind": "input"},
                "st_out": {"dtype": "bf16", "shape": [seqs, R], "kind": "output"},
                "latent": {"dtype": "bf16", "shape": [seqs, R], "kind": "output"},
                "base": {"dtype": "bf16", "shape": [seqs, 7, VP], "kind": "input"},
                "bias": {"dtype": "bf16", "shape": [seqs, VP], "kind": "input"},
                "fb": {"dtype": "bf16", "shape": [VP], "kind": "input"},
                "logits": {"dtype": "bf16", "shape": [seqs, VP], "kind": "output"},
                "pmax": {"dtype": "f32", "shape": [seqs, 64], "kind": "workspace"},
                "pidx": {"dtype": "i32", "shape": [seqs, 64], "kind": "workspace"},
                "tok": {"dtype": "i64", "shape": [seqs], "kind": "output"},
            },
            "modules": {
                **self._triton_mod("dspark_markov_gate_add"),
                **self._triton_mod("dspark_add_softcap_mask"),
                **self._spec_mod("spec_markov_rows"),
            },
            "ops": {
                "gate": {"params": p_gate, "impl": {"launches": [{"module": "dspark_markov_gate_add", "entry": rec_gate["entry"], "params": p_gate + rec_gate["abi_tail"], "grid": [seqs, 1, 1], "block": rec_gate["block"], "shared_mem": rec_gate["shared_mem"], "args": [{"param": j} for j in range(6)] + [{"i64": 0}, {"i64": 0}]}]}},
                "cap": {"params": p_cap, "impl": {"launches": [{"module": "dspark_add_softcap_mask", "entry": rec_cap["entry"], "params": p_cap + rec_cap["abi_tail"], "grid": [dm.ceil_div(VP, 1024), seqs, 1], "block": rec_cap["block"], "shared_mem": rec_cap["shared_mem"], "args": [{"param": j} for j in range(6)] + [{"i64": 0}, {"i64": 0}]}]}},
                "am1": {"params": p_am1, "impl": {"launches": [{"module": "spec_markov_rows", "entry": "kern_argmax_rows_partial_bf16", "params": p_am1, "grid": [seqs, 64, 1], "block": [1024, 1, 1], "args": [{"param": j} for j in range(5)]}]}},
                "am2": {"params": p_am2, "impl": {"launches": [{"module": "spec_markov_rows", "entry": "kern_argmax_rows_final_i64", "params": p_am2, "grid": [seqs, 1, 1], "block": [64, 1, 1], "args": [{"param": j} for j in range(5)]}]}},
            },
            "programs": {
                "run": {
                    "calls": [
                        {"op": "gate", "args": [{"buf": "ctx"}, {"buf": "sp"}, {"buf": "st_in"}, {"buf": "st_out"}, {"buf": "latent"}, {"i32": 2}]},
                        {"op": "cap", "args": [{"buf": "base"}, {"buf": "bias"}, {"buf": "fb"}, {"buf": "logits"}, {"i32": 2}, {"f32": 30.0}]},
                        {"op": "am1", "args": [{"buf": "logits"}, {"i64": VP}, {"buf": "pmax"}, {"buf": "pidx"}, {"i32": VP}]},
                        {"op": "am2", "args": [{"buf": "pmax"}, {"buf": "pidx"}, {"buf": "tok"}, {"i32": 1}, {"i32": 64}]},
                    ]
                }
            },
        }

        def setup(rt, v):
            v["ctx"].copy_(raw_ctx)
            v["sp"].copy_(raw_st)
            v["st_in"].copy_(state_in)
            v["base"].copy_(base_logits)
            v["bias"].copy_(step_bias)
            v["fb"].copy_(fold_bias)

        _, views = self._run_manifest(m, setup, "run", {})
        # Bitwise equality check for Markov gate
        gate_raw, cand_raw, out_raw = (raw_ctx[:, 2] + raw_st).bfloat16().chunk(3, dim=-1)
        gate_ref = torch.sigmoid(gate_raw)
        ref_state = gate_ref * state_in + (1.0 - gate_ref) * torch.tanh(cand_raw)
        ref_latent = torch.tanh(out_raw)
        self.assertTrue(torch.equal(views["st_out"], ref_state))
        self.assertTrue(torch.equal(views["latent"], ref_latent))

        # Softcap + vocab mask >= 200058 check
        bias_bf = (step_bias.float() + fold_bias.float()[None, :]).bfloat16()
        ref_logits = (((base_logits[:, 2] + bias_bf).float() / 30.0).tanh() * 30.0).bfloat16()
        ref_logits[:, V:] = float("-inf")
        torch.testing.assert_close(views["logits"][:, :V], ref_logits[:, :V], atol=0, rtol=1e-2)
        self.assertTrue((views["logits"][:, V:] == float("-inf")).all())
        self.assertTrue((views["tok"] < V).all())

    def test_dspark_block_attention_local_ring_and_full_paged(self):
        torch = self.torch
        g = torch.Generator().manual_seed(855)
        context_lens = [5, 70, 1050, 1200]
        seqs = len(context_lens) + 1  # last seq is padding/null line
        block_size = 8
        window = 1024
        page_sz = 64
        max_pages = 25
        H, HKV, D = 24, 8, 128

        # Build live Q/K/V and committed FP8 caches
        q = torch.randn(seqs * block_size, 3072, generator=g).bfloat16().cuda()
        k_live = torch.randn(seqs * block_size, 1024, generator=g).bfloat16().cuda()
        v_live = torch.randn(seqs * block_size, 1024, generator=g).bfloat16().cuda()

        # Generate committed FP8 history per sequence
        k_hist_bf16 = [torch.randn(n, HKV, D, generator=g).bfloat16().cuda() for n in context_lens]
        v_hist_bf16 = [torch.randn(n, HKV, D, generator=g).bfloat16().cuda() for n in context_lens]
        k_hist_fp8 = [x.to(torch.float8_e4m3fn) for x in k_hist_bf16]
        v_hist_fp8 = [x.to(torch.float8_e4m3fn) for x in v_hist_bf16]

        # Populate ring state (dspark_state) and paged state (kv)
        ring_stride = dm.DSPARK_STATE_BYTES_PER_SEQ
        ring_cache = torch.zeros((seqs + 1, ring_stride), dtype=torch.uint8, device="cuda")
        lines = torch.tensor([1, 2, 3, 4, 0], dtype=torch.int32, device="cuda")

        total_pages = seqs * max_pages
        # Match M3 states.kv layout: [pages, 9, 2, 8, 64, 128]
        kv_page_stride = 9 * 2 * HKV * page_sz * D
        kv_layer_off = 8 * 2 * HKV * page_sz * D
        kv_head_stride = page_sz * D
        kv_slot_stride = D
        kv_v_off = HKV * page_sz * D
        paged_cache = torch.zeros((total_pages, kv_page_stride), dtype=torch.uint8, device="cuda")
        pages_tbl = torch.arange(total_pages, dtype=torch.int32, device="cuda").view(seqs, max_pages)

        for s_idx, n in enumerate(context_lens):
            line = s_idx + 1
            kf = k_hist_fp8[s_idx].view(torch.uint8).reshape(n, 1024)
            vf = v_hist_fp8[s_idx].view(torch.uint8).reshape(n, 1024)
            # Ring: write last min(n, window) positions at pos % 1024
            start = max(0, n - window)
            for p in range(start, n):
                slot = p % window
                off = slot * 2048
                ring_cache[line, off : off + 1024] = kf[p]
                ring_cache[line, off + 1024 : off + 2048] = vf[p]
            # Paged: write all 0..n-1 into [9, 2, 8, 64, 128]
            kh_u8 = k_hist_fp8[s_idx].view(torch.uint8)
            vh_u8 = v_hist_fp8[s_idx].view(torch.uint8)
            for p in range(n):
                pg = int(pages_tbl[s_idx, p // page_sz].item())
                in_pg = p % page_sz
                for hd in range(HKV):
                    base_k = kv_layer_off + hd * kv_head_stride + in_pg * kv_slot_stride
                    paged_cache[pg, base_k : base_k + D] = kh_u8[p, hd]
                    paged_cache[pg, base_k + kv_v_off : base_k + kv_v_off + D] = vh_u8[p, hd]

        lengths = torch.tensor([n + block_size for n in context_lens] + [0], dtype=torch.int32, device="cuda")

        rec_ring = self.dsl["dspark_attn_ring"]
        rec_paged = self.dsl["dspark_attn_paged_partial"]
        rec_comb = self.dsl["dspark_attn_combine_16"]
        p_ring = [
            "in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "in state",
            "in buffer<i32>", "in buffer<i32>", "out buffer<bf16>",
            "f32", "f32", "f32", "i64", "i64", "i64", "i64", "i64",
        ]
        p_paged = [
            "in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "in state",
            "in buffer<i32>", "in buffer<i32>", "out buffer<f32>", "out buffer<f32>",
            "f32", "f32", "f32", "i64", "i64", "i64", "i64", "i64",
            "i64", "i64", "i64", "i64", "i64", "i32", "i32",
        ]
        p_comb = ["in buffer<f32>", "in buffer<f32>", "out buffer<bf16>", "i32"]
        m = {
            "schema_version": 7,
            "model": "dspark-attn-test",
            "vars": {},
            "states": {
                "dspark_state": {"host": {"dtype": "u8", "shape": [seqs + 1, ring_stride], "strides": [ring_stride, 1]}},
                "kv": {"host": {"dtype": "u8", "shape": [total_pages, kv_page_stride], "strides": [kv_page_stride, 1]}},
            },
            "buffers": {
                "Q": {"dtype": "bf16", "shape": [seqs * block_size, 3072], "kind": "input"},
                "K": {"dtype": "bf16", "shape": [seqs * block_size, 1024], "kind": "input"},
                "V": {"dtype": "bf16", "shape": [seqs * block_size, 1024], "kind": "input"},
                "Lines": {"dtype": "i32", "shape": [seqs], "kind": "input"},
                "Pages": {"dtype": "i32", "shape": [seqs, max_pages], "kind": "input"},
                "Lengths": {"dtype": "i32", "shape": [seqs], "kind": "input"},
                "OutRing": {"dtype": "bf16", "shape": [seqs * block_size, 3072], "kind": "output"},
                "OPart": {"dtype": "f32", "shape": [16, seqs * block_size, H, D], "kind": "workspace"},
                "LPart": {"dtype": "f32", "shape": [16, seqs * block_size, H], "kind": "workspace"},
                "OutPaged": {"dtype": "bf16", "shape": [seqs * block_size, 3072], "kind": "output"},
            },
            "modules": {
                **self._triton_mod("dspark_attn_ring"),
                **self._triton_mod("dspark_attn_paged_partial"),
                **self._triton_mod("dspark_attn_combine_16"),
            },
            "ops": {
                "attn_ring": {"params": p_ring, "impl": {"launches": [{"module": "dspark_attn_ring", "entry": rec_ring["entry"], "params": p_ring + rec_ring["abi_tail"], "grid": [HKV, seqs, 1], "block": rec_ring["block"], "shared_mem": rec_ring["shared_mem"], "args": [{"param": j} for j in range(15)] + [{"i64": 0}, {"i64": 0}]}]}},
                "attn_paged": {"params": p_paged, "impl": {"launches": [{"module": "dspark_attn_paged_partial", "entry": rec_paged["entry"], "params": p_paged + rec_paged["abi_tail"], "grid": [HKV * 16, seqs, 1], "block": rec_paged["block"], "shared_mem": rec_paged["shared_mem"], "args": [{"param": j} for j in range(23)] + [{"i64": 0}, {"i64": 0}]}]}},
                "attn_comb": {"params": p_comb, "impl": {"launches": [{"module": "dspark_attn_combine_16", "entry": rec_comb["entry"], "params": p_comb + rec_comb["abi_tail"], "grid": [seqs * block_size, H, 1], "block": rec_comb["block"], "shared_mem": rec_comb["shared_mem"], "args": [{"param": j} for j in range(4)] + [{"i64": 0}, {"i64": 0}]}]}},
            },
            "programs": {
                "run": {
                    "calls": [
                        {
                            "op": "attn_ring",
                            "args": [
                                {"buf": "Q"}, {"buf": "K"}, {"buf": "V"}, {"state": "dspark_state"},
                                {"buf": "Lines"}, {"buf": "Lengths"}, {"buf": "OutRing"},
                                {"f32": 1.0}, {"f32": 1.0}, {"f32": D**-0.5},
                                {"i64": 3072}, {"i64": 1024}, {"i64": 1024}, {"i64": 3072}, {"i64": ring_stride},
                            ],
                        },
                        {
                            "op": "attn_paged",
                            "args": [
                                {"buf": "Q"}, {"buf": "K"}, {"buf": "V"}, {"state": "kv"},
                                {"buf": "Pages"}, {"buf": "Lengths"}, {"buf": "OPart"}, {"buf": "LPart"},
                                {"f32": 1.0}, {"f32": 1.0}, {"f32": D**-0.5},
                                {"i64": 3072}, {"i64": 1024}, {"i64": 1024}, {"i64": max_pages}, {"i64": 3072},
                                {"i64": kv_layer_off}, {"i64": kv_page_stride}, {"i64": kv_slot_stride}, {"i64": kv_head_stride}, {"i64": kv_v_off},
                                {"i32": seqs * block_size}, {"i32": 16},
                            ],
                        },
                        {
                            "op": "attn_comb",
                            "args": [{"buf": "OPart"}, {"buf": "LPart"}, {"buf": "OutPaged"}, {"i32": seqs * block_size}],
                        },
                    ]
                }
            },
        }

        def setup(rt, v):
            rt.bind_host({
                "dspark_state": (ring_cache.data_ptr(), "u8", list(ring_cache.shape), list(ring_cache.stride())),
                "kv": (paged_cache.data_ptr(), "u8", list(paged_cache.shape), list(paged_cache.stride())),
            })
            v["Q"].copy_(q)
            v["K"].copy_(k_live)
            v["V"].copy_(v_live)
            v["Lines"].copy_(lines)
            v["Pages"].copy_(pages_tbl)
            v["Lengths"].copy_(lengths)

        _, views = self._run_manifest(m, setup, "run", {})
        out_ring = views["OutRing"].view(seqs, block_size, H, D)
        out_paged = views["OutPaged"].view(seqs, block_size, H, D)

        for i, n in enumerate(context_lens):
            kh_all = torch.cat([k_hist_fp8[i].bfloat16(), k_live[i * 8 : (i + 1) * 8].view(8, HKV, D)], dim=0)
            vh_all = torch.cat([v_hist_fp8[i].bfloat16(), v_live[i * 8 : (i + 1) * 8].view(8, HKV, D)], dim=0)
            qi = q[i * 8 : (i + 1) * 8].view(8, H, D).float()
            for is_local, out_t in ((True, out_ring[i]), (False, out_paged[i])):
                start = max(0, n - window) if is_local else 0
                keys = kh_all[start:].float().repeat_interleave(3, dim=1)
                vals = vh_all[start:].float().repeat_interleave(3, dim=1)
                scores = torch.einsum("qhd,khd->hqk", qi, keys) * (D**-0.5)
                ref = torch.einsum("hqk,khd->qhd", scores.softmax(-1), vals).bfloat16()
                torch.testing.assert_close(out_t.float(), ref.float(), atol=2e-2, rtol=2e-2)

        # Padding sequence (Lengths=0) must return zeros
        self.assertTrue((out_ring[-1] == 0).all())
        self.assertTrue((out_paged[-1] == 0).all())

    def test_spec_round_and_rollback_safe_commit(self):
        torch = self.torch
        g = torch.Generator().manual_seed(856)
        seqs = 3  # seq 0: accepted=1 at pos=1030 (wraparound!); seq 1: accepted=3; seq 2: null line=0
        block_size = 8
        n_visits = 48
        n_swa = 40
        sconv_dim = 6656
        window = 1024
        tok_bytes = 2048

        swa_off = 0
        sconv_off = n_swa * window * tok_bytes
        state_stride = sconv_off + n_visits * 3 * sconv_dim * 2

        seq_state = torch.randint(0, 255, (seqs + 1, state_stride), dtype=torch.uint8, device="cuda")
        # Keep sentinel clone to verify untouched slots!
        sentinel_state = seq_state.clone()

        spec_swa_kv = torch.randint(0, 255, (n_swa, seqs * block_size, tok_bytes), dtype=torch.uint8, device="cuda")
        spec_sconv_raw = torch.randn(n_visits, seqs * block_size, sconv_dim, generator=g).bfloat16().cuda()

        anchor = torch.tensor([10, 20, 30], dtype=torch.int64, device="cuda")
        drafts = torch.tensor(
            [[100, 101, 102, 103, 104, 105, 106], [200, 201, 202, 203, 204, 205, 206], [300, 301, 302, 303, 304, 305, 306]],
            dtype=torch.int64,
            device="cuda",
        )
        # seq 0 matches 0 drafts -> count=1; seq 1 matches 2 drafts -> count=3; seq 2 matches 7 -> count=8
        verify = torch.tensor(
            [
                [999, 101, 102, 103, 104, 105, 106, 107],
                [200, 201, 888, 203, 204, 205, 206, 207],
                [300, 301, 302, 303, 304, 305, 306, 307],
            ],
            dtype=torch.int64,
            device="cuda",
        )
        lines = torch.tensor([1, 2, 0], dtype=torch.int32, device="cuda")
        # anchor_pos = lengths - 8 -> [1030, 50, 10]
        lengths = torch.tensor([1030 + 8, 50 + 8, 10 + 8], dtype=torch.int32, device="cuda")

        p_cnt = ["in buffer<i64>", "in buffer<i64>", "out buffer<i32>", "i32", "i32"]
        p_sconv = ["in buffer<bf16>", "inout state", "in buffer<i32>", "in buffer<i32>", "i64", "i64", "i32", "i32", "i32"]
        p_swa = ["in buffer<fp8e4m3>", "inout state", "in buffer<i32>", "in buffer<i32>", "in buffer<i32>", "i64", "i64", "i32", "i32", "i32", "i32"]
        m = {
            "schema_version": 7,
            "model": "rollback-commit-test",
            "vars": {},
            "states": {"seq_state": {"host": {"dtype": "u8", "shape": [seqs + 1, state_stride], "strides": [state_stride, 1]}}},
            "buffers": {
                "drafts": {"dtype": "i64", "shape": [seqs, 7], "kind": "input"},
                "verify": {"dtype": "i64", "shape": [seqs, 8], "kind": "input"},
                "count": {"dtype": "i32", "shape": [seqs], "kind": "output"},
                "lines": {"dtype": "i32", "shape": [seqs], "kind": "input"},
                "lengths": {"dtype": "i32", "shape": [seqs], "kind": "input"},
                "sconv_raw": {"dtype": "bf16", "shape": [n_visits, seqs * block_size, sconv_dim], "kind": "input"},
                "swa_kv": {"dtype": "fp8e4m3", "shape": [n_swa, seqs * block_size, tok_bytes], "kind": "input"},
            },
            "modules": {**self._spec_mod("spec_round"), **self._spec_mod("spec_state_commit")},
            "ops": {
                "cnt": {"params": p_cnt, "impl": {"launches": [{"module": "spec_round", "entry": "kern_spec_count", "params": p_cnt, "grid": [seqs, 1, 1], "block": [32, 1, 1], "args": [{"param": j} for j in range(5)]}]}},
                "conv_c": {"params": p_sconv, "impl": {"launches": [{"module": "spec_state_commit", "entry": "spec_conv_commit", "params": p_sconv, "grid": [seqs, dm.ceil_div(sconv_dim, 256), n_visits], "block": [256, 1, 1], "args": [{"param": j} for j in range(9)]}]}},
                "swa_c": {"params": p_swa, "impl": {"launches": [{"module": "spec_state_commit", "entry": "spec_swa_commit", "params": p_swa, "grid": [seqs, n_swa, block_size], "block": [128, 1, 1], "args": [{"param": j} for j in range(11)]}]}},
            },
            "programs": {
                "run": {
                    "calls": [
                        {"op": "cnt", "args": [{"buf": "drafts"}, {"buf": "verify"}, {"buf": "count"}, {"i32": block_size}, {"i32": 7}]},
                        {"op": "conv_c", "args": [{"buf": "sconv_raw"}, {"state": "seq_state"}, {"buf": "lines"}, {"buf": "count"}, {"i64": state_stride}, {"i64": sconv_off}, {"i32": seqs * block_size}, {"i32": sconv_dim}, {"i32": block_size}]},
                        {"op": "swa_c", "args": [{"buf": "swa_kv"}, {"state": "seq_state"}, {"buf": "lines"}, {"buf": "lengths"}, {"buf": "count"}, {"i64": state_stride}, {"i64": swa_off}, {"i32": seqs * block_size}, {"i32": window}, {"i32": block_size}, {"i32": tok_bytes}]},
                    ]
                }
            },
        }

        def setup(rt, v):
            rt.bind_host({
                "seq_state": (seq_state.data_ptr(), "u8", list(seq_state.shape), list(seq_state.stride())),
            })
            v["drafts"].copy_(drafts)
            v["verify"].copy_(verify)
            v["lines"].copy_(lines)
            v["lengths"].copy_(lengths)
            v["sconv_raw"].copy_(spec_sconv_raw)
            v["swa_kv"].view(torch.uint8).copy_(spec_swa_kv)

        _, views = self._run_manifest(m, setup, "run", {})
        self.assertEqual(views["count"].tolist(), [1, 3, 8])

        # 1. Null line (slot 0) must be completely untouched!
        self.assertTrue(torch.equal(seq_state[0], sentinel_state[0]))

        # 2. SWA wraparound rollback check for seq 0 (line=1, anchor_pos=1030, accepted=1):
        # Only slot 1030 % 1024 = 6 may change; slots 7..13 (rejected rows 1..7) MUST be bit-identical!
        for v_idx in range(n_swa):
            base = swa_off + v_idx * window * tok_bytes
            slot6 = seq_state[1, base + 6 * tok_bytes : base + 7 * tok_bytes]
            self.assertTrue(torch.equal(slot6, spec_swa_kv[v_idx, 0]))
            for rej_slot in range(7, 14):
                got = seq_state[1, base + rej_slot * tok_bytes : base + (rej_slot + 1) * tok_bytes]
                exp = sentinel_state[1, base + rej_slot * tok_bytes : base + (rej_slot + 1) * tok_bytes]
                self.assertTrue(torch.equal(got, exp), f"SWA wraparound corrupted slot {rej_slot}!")

        # 3. sconv RAW-safe rollback check for seq 0 (n=1) and seq 1 (n=3):
        for s_idx, line, n_acc in ((0, 1, 1), (1, 2, 3)):
            old_hist = sentinel_state[line, sconv_off:].view(torch.bfloat16).view(n_visits, 3, sconv_dim)
            new_hist = seq_state[line, sconv_off:].view(torch.bfloat16).view(n_visits, 3, sconv_dim)
            for j in range(3):
                row = n_acc - 3 + j
                exp = old_hist[:, row + 3, :] if row < 0 else spec_sconv_raw[:, s_idx * block_size + row, :]
                self.assertTrue(torch.equal(new_hist[:, j, :].view(torch.uint16), exp.view(torch.uint16)))

    def test_cuda_graph_round_replay(self):
        torch = self.torch
        seqs = 2
        block = 8
        p_sp = ["in buffer<i64>", "in buffer<i64>", "out buffer<i64>", "i32", "i32"]
        p_cnt = ["in buffer<i64>", "in buffer<i64>", "out buffer<i32>", "i32", "i32"]
        m = {
            "schema_version": 7,
            "model": "graph-round-test",
            "vars": {"seqs": {"max": seqs}},
            "states": {},
            "buffers": {
                "anchor": {"dtype": "i64", "shape": ["seqs"], "kind": "input"},
                "draft": {"dtype": "i64", "shape": ["seqs", 7], "kind": "input"},
                "verify_ids": {"dtype": "i64", "shape": ["seqs", 8], "kind": "output"},
                "verify_toks": {"dtype": "i64", "shape": ["seqs", 8], "kind": "input"},
                "count": {"dtype": "i32", "shape": ["seqs"], "kind": "output"},
            },
            "modules": self._spec_mod("spec_round"),
            "ops": {
                "splice": {"params": p_sp, "impl": {"launches": [{"module": "spec_round", "entry": "kern_splice_verify", "params": p_sp, "grid": ["seqs", 1, 1], "block": [32, 1, 1], "args": [{"param": j} for j in range(5)]}]}},
                "count": {"params": p_cnt, "impl": {"launches": [{"module": "spec_round", "entry": "kern_spec_count", "params": p_cnt, "grid": ["seqs", 1, 1], "block": [32, 1, 1], "args": [{"param": j} for j in range(5)]}]}},
            },
            "programs": {
                "round": {
                    "calls": [
                        {"op": "splice", "args": [{"buf": "anchor"}, {"buf": "draft"}, {"buf": "verify_ids"}, {"i32": block}, {"i32": 7}]},
                        {"op": "count", "args": [{"buf": "draft"}, {"buf": "verify_toks"}, {"buf": "count"}, {"i32": block}, {"i32": 7}]},
                    ]
                }
            },
        }
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "model.json"
            p.write_text(json.dumps(m))
            rt = self.kern.Runtime(str(p), str(KERNELS_DIR), [], 0)
        v = {name: self.region(rt, m, name) for name in m["buffers"]}
        v["anchor"].copy_(torch.tensor([11, 22], dtype=torch.int64, device="cuda"))
        v["draft"].copy_(torch.arange(14, dtype=torch.int64, device="cuda").view(2, 7))
        v["verify_toks"].copy_(torch.tensor([[0, 1, 99, 3, 4, 5, 6, 7], [7, 8, 9, 10, 99, 12, 13, 14]], dtype=torch.int64, device="cuda"))

        stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream):
            rt.enqueue_after("round", {"seqs": seqs}, stream.cuda_stream)
        stream.synchronize()
        with torch.cuda.graph(graph, stream=stream):
            rt.enqueue_after("round", {"seqs": seqs}, stream.cuda_stream)
        graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(v["count"].tolist(), [3, 5])
        self.assertEqual(v["verify_ids"][0].tolist(), [11, 0, 1, 2, 3, 4, 5, 6])

    def test_multi_seq_frnn_isolation_and_spec_sampling(self):
        """Verify prefill(seq B) never clobbers seq A's f_rnn in dspark_state,
        spec_sampling expands block_size=8 lengths to match 1-token decode,
        and dspark_verify_mask_vocab_f32 softcaps & masks >= 200058."""
        torch = self.torch
        g = torch.Generator().manual_seed(857)
        stride = dm.DSPARK_STATE_BYTES_PER_SEQ
        frnn_off = dm.DSPARK_SWA_BYTES
        H = dm.DSPARK_HIDDEN
        dspark_state = torch.zeros((4, stride), dtype=torch.uint8, device="cuda")

        frnn_a = torch.randn(1, H, generator=g).bfloat16().cuda()
        frnn_b = torch.randn(1, H, generator=g).bfloat16().cuda()

        # Also test spec_sampling (block_size=8) + dspark_verify_mask_vocab_f32
        seqs = 2
        VP, VU = dm.VOCAB_PADDED, dm.VOCAB_UNPADDED
        logits_in = torch.randn(seqs * 8, VP, generator=g).float().cuda() * 15.0
        temp = torch.tensor([0.0, 0.8], dtype=torch.float32, device="cuda")
        top_k = torch.tensor([0, 20], dtype=torch.int32, device="cuda")
        top_p = torch.tensor([1.0, 0.9], dtype=torch.float32, device="cuda")
        min_p = torch.tensor([0.0, 0.05], dtype=torch.float32, device="cuda")
        seed = torch.tensor([111, 222], dtype=torch.int64, device="cuda")
        lengths = torch.tensor([33, 79], dtype=torch.int32, device="cuda")

        p_st = ["in buffer<bf16>", "inout state", "in buffer<i32>", "i64", "i64", "i32"]
        p_ld = ["in state", "in buffer<i32>", "out buffer<bf16>", "i64", "i64", "i32"]
        p_ss = [
            "in buffer<f32>", "in buffer<i32>", "in buffer<f32>", "in buffer<f32>", "in buffer<i64>", "in buffer<i32>",
            "out buffer<f32>", "out buffer<i32>", "out buffer<f32>", "out buffer<f32>", "out buffer<i64>", "out buffer<i32>", "i32",
        ]
        p_vm = ["inout buffer<f32>", "f32", "i32", "i32"]
        v_op = dm.sampling_op(VP, seqs * 8, self.spec["sampling"], self.sampling_target, logits_dtype="f32", axis_var="tok")
        m = {
            "schema_version": 7,
            "model": "frnn-iso-test",
            "vars": {"s": {"max": 2}, "tok": {"max": seqs * 8}},
            "states": {"dspark_state": {"host": {"dtype": "u8", "shape": [4, stride], "strides": [stride, 1]}}},
            "buffers": {
                "fa": {"dtype": "bf16", "shape": [1, H], "kind": "input"},
                "fb": {"dtype": "bf16", "shape": [1, H], "kind": "input"},
                "la": {"dtype": "i32", "shape": [1], "kind": "input"},
                "lb": {"dtype": "i32", "shape": [1], "kind": "input"},
                "l_ab": {"dtype": "i32", "shape": [2], "kind": "input"},
                "out_ab": {"dtype": "bf16", "shape": [2, H], "kind": "output"},
                "out_null": {"dtype": "bf16", "shape": [1, H], "kind": "output"},
                "l0": {"dtype": "i32", "shape": [1], "kind": "input"},
                "temp": {"dtype": "f32", "shape": [seqs], "kind": "input"},
                "topk": {"dtype": "i32", "shape": [seqs], "kind": "input"},
                "topp": {"dtype": "f32", "shape": [seqs], "kind": "input"},
                "minp": {"dtype": "f32", "shape": [seqs], "kind": "input"},
                "seed": {"dtype": "i64", "shape": [seqs], "kind": "input"},
                "lens": {"dtype": "i32", "shape": [seqs], "kind": "input"},
                "r_temp": {"dtype": "f32", "shape": [seqs * 8], "kind": "output"},
                "r_topk": {"dtype": "i32", "shape": [seqs * 8], "kind": "output"},
                "r_topp": {"dtype": "f32", "shape": [seqs * 8], "kind": "output"},
                "r_minp": {"dtype": "f32", "shape": [seqs * 8], "kind": "output"},
                "r_seed": {"dtype": "i64", "shape": [seqs * 8], "kind": "output"},
                "r_lens": {"dtype": "i32", "shape": [seqs * 8], "kind": "output"},
                "vlogits": {"dtype": "f32", "shape": ["tok", VP], "kind": "inout"},
                "vtokens": {"dtype": "i64", "shape": ["tok"], "kind": "output"},
            },
            "modules": {
                **self._spec_mod("spec_state_commit"),
                **self._spec_mod("spec_sampling"),
                **self._spec_mod("spec_sample_glue"),
                **self._spec_mod("spec_argmax"),
            },
            "ops": {
                "st": {"params": p_st, "impl": {"launches": [{"module": "spec_state_commit", "entry": "dspark_store_frnn", "params": p_st, "grid": [1, 1, 1], "block": [256, 1, 1], "args": [{"param": j} for j in range(6)]}]}},
                "ld": {"params": p_ld, "impl": {"launches": [{"module": "spec_state_commit", "entry": "dspark_load_frnn", "params": p_ld, "grid": ["s", 1, 1], "block": [256, 1, 1], "args": [{"param": j} for j in range(6)]}]}},
                "ss": {"params": p_ss, "impl": {"launches": [{"module": "spec_state_commit", "entry": "spec_sampling", "params": p_ss, "grid": [seqs, 1, 1], "block": [32, 1, 1], "args": [{"param": j} for j in range(13)]}]}},
                "vm": {"params": p_vm, "impl": {"launches": [{"module": "spec_state_commit", "entry": "dspark_verify_mask_vocab_f32", "params": p_vm, "grid": ["tok", dm.ceil_div(VP, 1024), 1], "block": [1024, 1, 1], "args": [{"param": j} for j in range(4)]}]}},
                "vs": v_op,
            },
            "programs": {
                "run": {
                    "calls": [
                        {"op": "st", "args": [{"buf": "fa"}, {"state": "dspark_state"}, {"buf": "la"}, {"i64": stride}, {"i64": frnn_off}, {"i32": H}]},
                        # Interleave prefill of seq B on line 2
                        {"op": "st", "args": [{"buf": "fb"}, {"state": "dspark_state"}, {"buf": "lb"}, {"i64": stride}, {"i64": frnn_off}, {"i32": H}]},
                        # Also test line=0 store is no-op
                        {"op": "st", "args": [{"buf": "fb"}, {"state": "dspark_state"}, {"buf": "l0"}, {"i64": stride}, {"i64": frnn_off}, {"i32": H}]},
                        {"op": "ld", "args": [{"state": "dspark_state"}, {"buf": "l_ab"}, {"buf": "out_ab"}, {"i64": stride}, {"i64": frnn_off}, {"i32": H}]},
                        {"op": "ld", "args": [{"state": "dspark_state"}, {"buf": "l0"}, {"buf": "out_null"}, {"i64": stride}, {"i64": frnn_off}, {"i32": H}]},
                        {"op": "ss", "args": [{"buf": "temp"}, {"buf": "topk"}, {"buf": "topp"}, {"buf": "minp"}, {"buf": "seed"}, {"buf": "lens"}, {"buf": "r_temp"}, {"buf": "r_topk"}, {"buf": "r_topp"}, {"buf": "r_minp"}, {"buf": "r_seed"}, {"buf": "r_lens"}, {"i32": 8}]},
                        {"op": "vm", "args": [{"buf": "vlogits"}, {"f32": 30.0}, {"i32": VP}, {"i32": VU}]},
                        {"op": "vs", "args": [{"buf": "vlogits"}, {"buf": "r_lens"}, {"buf": "r_temp"}, {"buf": "r_topk"}, {"buf": "r_topp"}, {"buf": "r_minp"}, {"buf": "r_seed"}, {"buf": "vtokens"}, {"var": "tok"}]},
                    ]
                }
            },
        }

        def setup(rt, v):
            rt.bind_host({
                "dspark_state": (dspark_state.data_ptr(), "u8", list(dspark_state.shape), list(dspark_state.stride())),
            })
            v["fa"].copy_(frnn_a)
            v["fb"].copy_(frnn_b)
            v["la"].fill_(1)
            v["lb"].fill_(2)
            v["l_ab"].copy_(torch.tensor([1, 2], dtype=torch.int32, device="cuda"))
            v["l0"].fill_(0)
            v["temp"].copy_(temp)
            v["topk"].copy_(top_k)
            v["topp"].copy_(top_p)
            v["minp"].copy_(min_p)
            v["seed"].copy_(seed)
            v["lens"].copy_(lengths)
            v["vlogits"].copy_(logits_in)

        _, views = self._run_manifest(m, setup, "run", {"s": 2, "tok": seqs * 8})
        self.assertTrue(torch.equal(views["out_ab"][0:1], frnn_a))
        self.assertTrue(torch.equal(views["out_ab"][1:2], frnn_b))
        self.assertTrue((views["out_null"] == 0).all())
        self.assertTrue((dspark_state[0] == 0).all())

        expected_rlens = lengths.repeat_interleave(8) - 7 + torch.arange(8, dtype=torch.int32, device="cuda").repeat(seqs)
        self.assertTrue(torch.equal(views["r_lens"], expected_rlens))
        self.assertTrue((views["vlogits"][:, VU:] == float("-inf")).all())
        ref_v = 30.0 * torch.tanh(logits_in[:, :VU] / 30.0)
        torch.testing.assert_close(views["vlogits"][:, :VU], ref_v, atol=1e-5, rtol=1e-5)
        # Verify FlashInfer sampling + greedy argmax select valid unpadded tokens (< 200058)
        self.assertTrue(((views["vtokens"] >= 0) & (views["vtokens"] < VU)).all())
        self.assertTrue(torch.equal(views["vtokens"][:8], ref_v[:8].argmax(dim=-1)))

    def test_ctx_store_paged_slot0_and_valid_mask(self):
        """Verify dspark_ctx_store_paged_{direct,commit} writes slot==0 when valid==1
        and skips writing when valid==0 (even if slot>=0) or slot<0."""
        torch = self.torch
        g = torch.Generator().manual_seed(858)
        page_sz = 64
        HKV, D = 8, 128
        pages = 2
        kv_page_stride = 9 * 2 * HKV * page_sz * D
        kv_layer_off = 8 * 2 * HKV * page_sz * D
        kv_head_stride = page_sz * D
        kv_slot_stride = D
        kv_v_off = HKV * page_sz * D
        sentinel = 0x5A

        def read_slot_kv(cache_u8, slot_idx: int):
            pg = slot_idx // page_sz
            in_pg = slot_idx % page_sz
            ks = torch.stack([cache_u8[pg, kv_layer_off + h * kv_head_stride + in_pg * kv_slot_stride : kv_layer_off + h * kv_head_stride + in_pg * kv_slot_stride + D] for h in range(HKV)])
            vs = torch.stack([cache_u8[pg, kv_layer_off + kv_v_off + h * kv_head_stride + in_pg * kv_slot_stride : kv_layer_off + kv_v_off + h * kv_head_stride + in_pg * kv_slot_stride + D] for h in range(HKV)])
            return ks, vs

        # Case A: direct mode (prefill/decode)
        # row 0: slot=0, valid=1 -> MUST write
        # row 1: slot=1, valid=0 -> MUST skip
        # row 2: slot=-1, valid=1 -> MUST skip
        # row 3: slot=64, valid=1 -> MUST write
        tokens_d = 4
        kv_in_d = (torch.randn(tokens_d, 2048, generator=g) + 3.0).bfloat16().cuda()
        slots_d = torch.tensor([0, 1, -1, 64], dtype=torch.int32, device="cuda")
        valid_d = torch.tensor([1, 0, 1, 1], dtype=torch.int32, device="cuda")
        lens_d = torch.tensor([1], dtype=torch.int32, device="cuda")
        cache_d = torch.full((pages, kv_page_stride), sentinel, dtype=torch.uint8, device="cuda")

        rec_dir = self.dsl["dspark_ctx_store_paged_direct"]
        rec_com = self.dsl["dspark_ctx_store_paged_commit"]
        p_paged = [
            "in buffer<bf16>", "in buffer<i32>", "in buffer<i32>", "in buffer<i32>", "inout state",
            "f32", "f32", "i64", "i64", "i64", "i64", "i64", "i64", "i32",
        ]
        m_dir = {
            "schema_version": 7,
            "model": "ctx-paged-direct-test",
            "vars": {},
            "states": {"kv": {"host": {"dtype": "u8", "shape": [pages, kv_page_stride], "strides": [kv_page_stride, 1]}}},
            "buffers": {
                "KV": {"dtype": "bf16", "shape": [tokens_d, 2048], "kind": "input"},
                "Slots": {"dtype": "i32", "shape": [tokens_d], "kind": "input"},
                "Valid": {"dtype": "i32", "shape": [tokens_d], "kind": "input"},
                "Count": {"dtype": "i32", "shape": [1], "kind": "input"},
            },
            "modules": self._triton_mod("dspark_ctx_store_paged_direct"),
            "ops": {
                "store": {
                    "params": p_paged,
                    "impl": {
                        "launches": [
                            {
                                "module": "dspark_ctx_store_paged_direct",
                                "entry": rec_dir["entry"],
                                "params": p_paged + rec_dir["abi_tail"],
                                "grid": [tokens_d, 4, 1],
                                "block": rec_dir["block"],
                                "shared_mem": rec_dir["shared_mem"],
                                "args": [{"param": j} for j in range(14)] + [{"i64": 0}, {"i64": 0}],
                            }
                        ]
                    },
                }
            },
            "programs": {
                "run": {
                    "calls": [
                        {
                            "op": "store",
                            "args": [
                                {"buf": "KV"}, {"buf": "Slots"}, {"buf": "Valid"}, {"buf": "Count"}, {"state": "kv"},
                                {"f32": 1.0}, {"f32": 1.0},
                                {"i64": kv_layer_off}, {"i64": kv_page_stride}, {"i64": kv_slot_stride},
                                {"i64": kv_head_stride}, {"i64": kv_v_off}, {"i64": 2048}, {"i32": tokens_d},
                            ],
                        }
                    ]
                }
            },
        }

        def setup_dir(rt, v):
            rt.bind_host({"kv": (cache_d.data_ptr(), "u8", list(cache_d.shape), list(cache_d.stride()))})
            v["KV"].copy_(kv_in_d)
            v["Slots"].copy_(slots_d)
            v["Valid"].copy_(valid_d)
            v["Count"].copy_(lens_d)

        self._run_manifest(m_dir, setup_dir, "run", {})
        for r_idx, s_idx in ((0, 0), (3, 64)):
            ks, vs = read_slot_kv(cache_d, s_idx)
            exp_k = kv_in_d[r_idx, :1024].float().to(torch.float8_e4m3fn).view(torch.uint8).view(HKV, D)
            exp_v = kv_in_d[r_idx, 1024:].float().to(torch.float8_e4m3fn).view(torch.uint8).view(HKV, D)
            self.assertTrue(torch.equal(ks, exp_k), f"direct slot={s_idx} K mismatch")
            self.assertTrue(torch.equal(vs, exp_v), f"direct slot={s_idx} V mismatch")
        ks1, vs1 = read_slot_kv(cache_d, 1)
        self.assertTrue((ks1 == sentinel).all() and (vs1 == sentinel).all())

        # Verify direct mode with slot==0, valid==0 skips writing slot 0
        cache_d0 = torch.full((pages, kv_page_stride), sentinel, dtype=torch.uint8, device="cuda")

        def setup_dir0(rt, v):
            rt.bind_host({"kv": (cache_d0.data_ptr(), "u8", list(cache_d0.shape), list(cache_d0.stride()))})
            v["KV"].copy_(kv_in_d)
            v["Slots"].copy_(torch.tensor([0, 1, 2, 3], dtype=torch.int32, device="cuda"))
            v["Valid"].zero_()
            v["Count"].copy_(lens_d)

        self._run_manifest(m_dir, setup_dir0, "run", {})
        self.assertTrue((cache_d0 == sentinel).all())

        # Case B: commit mode (seqs=2, block_size=8, tokens=16)
        # seq 0 (valid=1, accepted=3): slots [0, -1, 2, 3, 4, 5, 6, 7] -> writes 0 and 2 ONLY; 1 and 3..7 stay sentinel
        # seq 1 (valid=0, accepted=8): slots [8..15] -> ALL skipped (stay sentinel)
        tokens_c = 16
        kv_in_c = (torch.randn(tokens_c, 2048, generator=g) - 2.5).bfloat16().cuda()
        slots_c = torch.tensor([0, -1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=torch.int32, device="cuda")
        valid_c = torch.tensor([1] * 8 + [0] * 8, dtype=torch.int32, device="cuda")
        acc_c = torch.tensor([3, 8], dtype=torch.int32, device="cuda")
        cache_c = torch.full((pages, kv_page_stride), sentinel, dtype=torch.uint8, device="cuda")

        m_com = json.loads(json.dumps(m_dir))
        m_com["buffers"]["KV"]["shape"] = [tokens_c, 2048]
        m_com["buffers"]["Slots"]["shape"] = [tokens_c]
        m_com["buffers"]["Valid"]["shape"] = [tokens_c]
        m_com["buffers"]["Count"]["shape"] = [2]
        m_com["modules"] = self._triton_mod("dspark_ctx_store_paged_commit")
        m_com["ops"]["store"]["impl"]["launches"][0]["module"] = "dspark_ctx_store_paged_commit"
        m_com["ops"]["store"]["impl"]["launches"][0]["entry"] = rec_com["entry"]
        m_com["ops"]["store"]["impl"]["launches"][0]["grid"] = [tokens_c, 4, 1]
        m_com["ops"]["store"]["impl"]["launches"][0]["shared_mem"] = rec_com["shared_mem"]
        m_com["programs"]["run"]["calls"][0]["args"][-1] = {"i32": tokens_c}

        def setup_com(rt, v):
            rt.bind_host({"kv": (cache_c.data_ptr(), "u8", list(cache_c.shape), list(cache_c.stride()))})
            v["KV"].copy_(kv_in_c)
            v["Slots"].copy_(slots_c)
            v["Valid"].copy_(valid_c)
            v["Count"].copy_(acc_c)

        self._run_manifest(m_com, setup_com, "run", {})
        for r_idx, s_idx in ((0, 0), (2, 2)):
            ks, vs = read_slot_kv(cache_c, s_idx)
            exp_k = kv_in_c[r_idx, :1024].float().to(torch.float8_e4m3fn).view(torch.uint8).view(HKV, D)
            exp_v = kv_in_c[r_idx, 1024:].float().to(torch.float8_e4m3fn).view(torch.uint8).view(HKV, D)
            self.assertTrue(torch.equal(ks, exp_k), f"commit slot={s_idx} K mismatch")
            self.assertTrue(torch.equal(vs, exp_v), f"commit slot={s_idx} V mismatch")
        for untouched_slot in (1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15):
            ks, vs = read_slot_kv(cache_c, untouched_slot)
            self.assertTrue((ks == sentinel).all() and (vs == sentinel).all(), f"slot={untouched_slot} should be untouched")

    def test_attn_ring_and_paged_zero_stale_out_on_padding_seq(self):
        """Verify stale non-zero Out (999.0) is zeroed on seqlen_k==0 padding sequences
        in dspark_attn_ring, dspark_attn_paged_partial+combine, and dspark_attn_paged_direct."""
        torch = self.torch
        g = torch.Generator().manual_seed(859)
        seqs, block_size, H, HKV, D = 2, 8, 24, 8, 128
        page_sz, max_pages = 64, 2
        q = torch.randn(seqs * block_size, 3072, generator=g).bfloat16().cuda()
        k = torch.randn(seqs * block_size, 1024, generator=g).bfloat16().cuda()
        v = torch.randn(seqs * block_size, 1024, generator=g).bfloat16().cuda()
        ring_stride = dm.DSPARK_STATE_BYTES_PER_SEQ
        ring_cache = torch.zeros((seqs + 1, ring_stride), dtype=torch.uint8, device="cuda")
        kv_page_stride = 9 * 2 * HKV * page_sz * D
        kv_layer_off = 8 * 2 * HKV * page_sz * D
        kv_head_stride = page_sz * D
        kv_slot_stride = D
        kv_v_off = HKV * page_sz * D
        paged_cache = torch.zeros((seqs * max_pages, kv_page_stride), dtype=torch.uint8, device="cuda")
        lines = torch.tensor([1, 0], dtype=torch.int32, device="cuda")
        pages_tbl = torch.arange(seqs * max_pages, dtype=torch.int32, device="cuda").view(seqs, max_pages)
        lengths = torch.tensor([10, 0], dtype=torch.int32, device="cuda")

        rec_ring = self.dsl["dspark_attn_ring"]
        rec_pp = self.dsl["dspark_attn_paged_partial"]
        rec_pd = self.dsl["dspark_attn_paged_direct"]
        rec_comb = self.dsl["dspark_attn_combine_16"]
        p_ring = [
            "in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "in state",
            "in buffer<i32>", "in buffer<i32>", "out buffer<bf16>",
            "f32", "f32", "f32", "i64", "i64", "i64", "i64", "i64",
        ]
        p_paged = [
            "in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "in state",
            "in buffer<i32>", "in buffer<i32>", "out buffer<f32>", "out buffer<f32>",
            "f32", "f32", "f32", "i64", "i64", "i64", "i64", "i64",
            "i64", "i64", "i64", "i64", "i64", "i32", "i32",
        ]
        p_paged_dir = p_paged[:6] + ["out buffer<bf16>"] + p_paged[7:]
        p_comb = ["in buffer<f32>", "in buffer<f32>", "out buffer<bf16>", "i32"]

        m = {
            "schema_version": 7,
            "model": "stale-out-zero-test",
            "vars": {},
            "states": {
                "dspark_state": {"host": {"dtype": "u8", "shape": [seqs + 1, ring_stride], "strides": [ring_stride, 1]}},
                "kv": {"host": {"dtype": "u8", "shape": [seqs * max_pages, kv_page_stride], "strides": [kv_page_stride, 1]}},
            },
            "buffers": {
                "Q": {"dtype": "bf16", "shape": [seqs * block_size, 3072], "kind": "input"},
                "K": {"dtype": "bf16", "shape": [seqs * block_size, 1024], "kind": "input"},
                "V": {"dtype": "bf16", "shape": [seqs * block_size, 1024], "kind": "input"},
                "Lines": {"dtype": "i32", "shape": [seqs], "kind": "input"},
                "Pages": {"dtype": "i32", "shape": [seqs, max_pages], "kind": "input"},
                "Lengths": {"dtype": "i32", "shape": [seqs], "kind": "input"},
                "OutRing": {"dtype": "bf16", "shape": [seqs * block_size, 3072], "kind": "output"},
                "OPart": {"dtype": "f32", "shape": [16, seqs * block_size, H, D], "kind": "output"},
                "LPart": {"dtype": "f32", "shape": [16, seqs * block_size, H], "kind": "output"},
                "OutPaged": {"dtype": "bf16", "shape": [seqs * block_size, 3072], "kind": "output"},
                "OutPagedDirect": {"dtype": "bf16", "shape": [seqs * block_size, 3072], "kind": "output"},
                "LDirect": {"dtype": "f32", "shape": [1], "kind": "workspace"},
            },
            "modules": {
                **self._triton_mod("dspark_attn_ring"),
                **self._triton_mod("dspark_attn_paged_partial"),
                **self._triton_mod("dspark_attn_paged_direct"),
                **self._triton_mod("dspark_attn_combine_16"),
            },
            "ops": {
                "attn_ring": {"params": p_ring, "impl": {"launches": [{"module": "dspark_attn_ring", "entry": rec_ring["entry"], "params": p_ring + rec_ring["abi_tail"], "grid": [HKV, seqs, 1], "block": rec_ring["block"], "shared_mem": rec_ring["shared_mem"], "args": [{"param": j} for j in range(15)] + [{"i64": 0}, {"i64": 0}]}]}},
                "attn_pp": {"params": p_paged, "impl": {"launches": [{"module": "dspark_attn_paged_partial", "entry": rec_pp["entry"], "params": p_paged + rec_pp["abi_tail"], "grid": [HKV * 16, seqs, 1], "block": rec_pp["block"], "shared_mem": rec_pp["shared_mem"], "args": [{"param": j} for j in range(23)] + [{"i64": 0}, {"i64": 0}]}]}},
                "attn_pd": {"params": p_paged_dir, "impl": {"launches": [{"module": "dspark_attn_paged_direct", "entry": rec_pd["entry"], "params": p_paged_dir + rec_pd["abi_tail"], "grid": [HKV, seqs, 1], "block": rec_pd["block"], "shared_mem": rec_pd["shared_mem"], "args": [{"param": j} for j in range(23)] + [{"i64": 0}, {"i64": 0}]}]}},
                "attn_comb": {"params": p_comb, "impl": {"launches": [{"module": "dspark_attn_combine_16", "entry": rec_comb["entry"], "params": p_comb + rec_comb["abi_tail"], "grid": [seqs * block_size, H, 1], "block": rec_comb["block"], "shared_mem": rec_comb["shared_mem"], "args": [{"param": j} for j in range(4)] + [{"i64": 0}, {"i64": 0}]}]}},
            },
            "programs": {
                "run": {
                    "calls": [
                        {"op": "attn_ring", "args": [{"buf": "Q"}, {"buf": "K"}, {"buf": "V"}, {"state": "dspark_state"}, {"buf": "Lines"}, {"buf": "Lengths"}, {"buf": "OutRing"}, {"f32": 1.0}, {"f32": 1.0}, {"f32": D**-0.5}, {"i64": 3072}, {"i64": 1024}, {"i64": 1024}, {"i64": 3072}, {"i64": ring_stride}]},
                        {"op": "attn_pp", "args": [{"buf": "Q"}, {"buf": "K"}, {"buf": "V"}, {"state": "kv"}, {"buf": "Pages"}, {"buf": "Lengths"}, {"buf": "OPart"}, {"buf": "LPart"}, {"f32": 1.0}, {"f32": 1.0}, {"f32": D**-0.5}, {"i64": 3072}, {"i64": 1024}, {"i64": 1024}, {"i64": max_pages}, {"i64": 3072}, {"i64": kv_layer_off}, {"i64": kv_page_stride}, {"i64": kv_slot_stride}, {"i64": kv_head_stride}, {"i64": kv_v_off}, {"i32": seqs * block_size}, {"i32": 16}]},
                        {"op": "attn_comb", "args": [{"buf": "OPart"}, {"buf": "LPart"}, {"buf": "OutPaged"}, {"i32": seqs * block_size}]},
                        {"op": "attn_pd", "args": [{"buf": "Q"}, {"buf": "K"}, {"buf": "V"}, {"state": "kv"}, {"buf": "Pages"}, {"buf": "Lengths"}, {"buf": "OutPagedDirect"}, {"buf": "LDirect"}, {"f32": 1.0}, {"f32": 1.0}, {"f32": D**-0.5}, {"i64": 3072}, {"i64": 1024}, {"i64": 1024}, {"i64": max_pages}, {"i64": 3072}, {"i64": kv_layer_off}, {"i64": kv_page_stride}, {"i64": kv_slot_stride}, {"i64": kv_head_stride}, {"i64": kv_v_off}, {"i32": seqs * block_size}, {"i32": 1}]},
                    ]
                }
            },
        }

        def setup(rt, views):
            rt.bind_host({
                "dspark_state": (ring_cache.data_ptr(), "u8", list(ring_cache.shape), list(ring_cache.stride())),
                "kv": (paged_cache.data_ptr(), "u8", list(paged_cache.shape), list(paged_cache.stride())),
            })
            views["Q"].copy_(q)
            views["K"].copy_(k)
            views["V"].copy_(v)
            views["Lines"].copy_(lines)
            views["Pages"].copy_(pages_tbl)
            views["Lengths"].copy_(lengths)
            views["OutRing"].fill_(999.0)
            views["OPart"].fill_(999.0)
            views["LPart"].fill_(5.0)
            views["OutPaged"].fill_(999.0)
            views["OutPagedDirect"].fill_(999.0)

        _, views = self._run_manifest(m, setup, "run", {})
        self.assertTrue((views["OutRing"][8:] == 0).all())
        self.assertTrue((views["OPart"][:, 8:] == 0).all())
        self.assertTrue((views["LPart"][:, 8:] == float("-inf")).all())
        self.assertTrue((views["OutPaged"][8:] == 0).all())
        self.assertTrue((views["OutPagedDirect"][8:] == 0).all())
        # Active seq 0 should remain non-zero and finite
        self.assertTrue(torch.isfinite(views["OutRing"][:8]).all() and views["OutRing"][:8].abs().max() > 0)
        self.assertTrue(torch.isfinite(views["OutPaged"][:8]).all() and views["OutPaged"][:8].abs().max() > 0)
        self.assertTrue(torch.isfinite(views["OutPagedDirect"][:8]).all() and views["OutPagedDirect"][:8].abs().max() > 0)

    def test_argmax_nan_unpadded_selects_zero_not_padded(self):
        """Verify NaN unpadded logits in spec_markov_rows & spec_argmax select token 0 (not 200058)."""
        torch = self.torch
        rows = 2
        VP, VU = dm.VOCAB_PADDED, dm.VOCAB_UNPADDED
        logits_bf16 = torch.full((rows, VP), float("nan"), dtype=torch.bfloat16, device="cuda")
        logits_bf16[:, VU:] = float("-inf")
        logits_bf16[1, 1234] = -50.0
        logits_f32 = logits_bf16.float()

        p_mr1 = ["in buffer<bf16>", "i64", "out buffer<f32>", "out buffer<i32>", "i32"]
        p_mr2 = ["in buffer<f32>", "in buffer<i32>", "out buffer<i64>", "i32", "i32"]
        p_a1_bf = ["in buffer<bf16>", "out buffer<f32>", "out buffer<i32>", "i32"]
        p_a1_f32 = ["in buffer<f32>", "out buffer<f32>", "out buffer<i32>", "i32"]
        p_a2 = ["in buffer<f32>", "in buffer<i32>", "out buffer<i64>", "i32"]

        m = {
            "schema_version": 7,
            "model": "argmax-nan-test",
            "vars": {},
            "states": {},
            "buffers": {
                "Lbf16": {"dtype": "bf16", "shape": [rows, VP], "kind": "input"},
                "Lf32": {"dtype": "f32", "shape": [rows, VP], "kind": "input"},
                "pmax": {"dtype": "f32", "shape": [rows, 64], "kind": "workspace"},
                "pidx": {"dtype": "i32", "shape": [rows, 64], "kind": "workspace"},
                "out_mr": {"dtype": "i64", "shape": [rows], "kind": "output"},
                "out_bf16": {"dtype": "i64", "shape": [rows], "kind": "output"},
                "out_f32": {"dtype": "i64", "shape": [rows], "kind": "output"},
            },
            "modules": {**self._spec_mod("spec_markov_rows"), **self._spec_mod("spec_argmax")},
            "ops": {
                "mr1": {"params": p_mr1, "impl": {"launches": [{"module": "spec_markov_rows", "entry": "kern_argmax_rows_partial_bf16", "params": p_mr1, "grid": [rows, 64, 1], "block": [1024, 1, 1], "args": [{"param": j} for j in range(5)]}]}},
                "mr2": {"params": p_mr2, "impl": {"launches": [{"module": "spec_markov_rows", "entry": "kern_argmax_rows_final_i64", "params": p_mr2, "grid": [rows, 1, 1], "block": [64, 1, 1], "args": [{"param": j} for j in range(5)]}]}},
                "a1_bf": {"params": p_a1_bf, "impl": {"launches": [{"module": "spec_argmax", "entry": "kern_argmax_partial_bf16", "params": p_a1_bf, "grid": [rows, 64, 1], "block": [1024, 1, 1], "args": [{"param": j} for j in range(4)]}]}},
                "a1_f32": {"params": p_a1_f32, "impl": {"launches": [{"module": "spec_argmax", "entry": "kern_argmax_partial_f32", "params": p_a1_f32, "grid": [rows, 64, 1], "block": [1024, 1, 1], "args": [{"param": j} for j in range(4)]}]}},
                "a2": {"params": p_a2, "impl": {"launches": [{"module": "spec_argmax", "entry": "kern_argmax_final_i64", "params": p_a2, "grid": [rows, 1, 1], "block": [64, 1, 1], "args": [{"param": j} for j in range(4)]}]}},
            },
            "programs": {
                "run": {
                    "calls": [
                        {"op": "mr1", "args": [{"buf": "Lbf16"}, {"i64": VP}, {"buf": "pmax"}, {"buf": "pidx"}, {"i32": VP}]},
                        {"op": "mr2", "args": [{"buf": "pmax"}, {"buf": "pidx"}, {"buf": "out_mr"}, {"i32": 1}, {"i32": 64}]},
                        {"op": "a1_bf", "args": [{"buf": "Lbf16"}, {"buf": "pmax"}, {"buf": "pidx"}, {"i32": VP}]},
                        {"op": "a2", "args": [{"buf": "pmax"}, {"buf": "pidx"}, {"buf": "out_bf16"}, {"i32": 64}]},
                        {"op": "a1_f32", "args": [{"buf": "Lf32"}, {"buf": "pmax"}, {"buf": "pidx"}, {"i32": VP}]},
                        {"op": "a2", "args": [{"buf": "pmax"}, {"buf": "pidx"}, {"buf": "out_f32"}, {"i32": 64}]},
                    ]
                }
            },
        }

        def setup(rt, v):
            v["Lbf16"].copy_(logits_bf16)
            v["Lf32"].copy_(logits_f32)

        _, views = self._run_manifest(m, setup, "run", {})
        self.assertEqual(views["out_mr"].tolist(), [0, 1234])
        self.assertEqual(views["out_bf16"].tolist(), [0, 1234])
        self.assertEqual(views["out_f32"].tolist(), [0, 1234])

    def test_dspark_verify_mask_vocab_bf16(self):
        """Verify dspark_verify_mask_vocab_bf16 masks >= 200058 to -inf and softcaps 30.0."""
        torch = self.torch
        g = torch.Generator().manual_seed(860)
        tokens = 3
        VP, VU = dm.VOCAB_PADDED, dm.VOCAB_UNPADDED
        logits_in = (torch.randn(tokens, VP, generator=g) * 25.0).bfloat16().cuda()
        logits_in[:, VU:] = 50.0
        logits_in[:, VU + 5] = float("nan")
        logits_in[0, 0] = 100.0
        logits_in[0, 1] = -100.0

        p_vm = ["inout buffer<bf16>", "f32", "i32", "i32"]
        m = {
            "schema_version": 7,
            "model": "verify-mask-bf16-test",
            "vars": {},
            "states": {},
            "buffers": {"logits": {"dtype": "bf16", "shape": [tokens, VP], "kind": "inout"}},
            "modules": self._spec_mod("spec_state_commit"),
            "ops": {
                "vm": {
                    "params": p_vm,
                    "impl": {
                        "launches": [
                            {
                                "module": "spec_state_commit",
                                "entry": "dspark_verify_mask_vocab_bf16",
                                "params": p_vm,
                                "grid": [tokens, dm.ceil_div(VP, 1024), 1],
                                "block": [1024, 1, 1],
                                "args": [{"param": j} for j in range(4)],
                            }
                        ]
                    },
                }
            },
            "programs": {"run": {"calls": [{"op": "vm", "args": [{"buf": "logits"}, {"f32": 30.0}, {"i32": VP}, {"i32": VU}]}]}},
        }

        def setup(rt, v):
            v["logits"].copy_(logits_in)

        _, views = self._run_manifest(m, setup, "run", {})
        self.assertTrue((views["logits"][:, VU:] == float("-inf")).all())
        ref = (30.0 * torch.tanh(logits_in[:, :VU].float() / 30.0)).bfloat16()
        torch.testing.assert_close(views["logits"][:, :VU], ref, atol=1e-2, rtol=1e-2)

    def test_markov_fused_fold_softcap_argmax_parity(self):
        """Verify fused kern_markov_fold_softcap_argmax_partial + kern_markov_argmax_final_pdl
        matches PyTorch bf16(fold_proj) + fold_bias + base_logits + softcap + vocab mask + argmax
        across seqs in (1, 2, 3, 4, 8), including padded vocab > 200058 poison values."""
        torch = self.torch
        g = torch.Generator().manual_seed(861)
        VP, VU, R, NSTEPS = dm.VOCAB_PADDED, dm.VOCAB_UNPADDED, dm.DSPARK_MARKOV_RANK, dm.DSPARK_NUM_SPEC
        fold_w = (torch.randn(VP, R, generator=g) * 0.05).bfloat16().cuda()
        fold_b = (torch.randn(VP, generator=g) * 0.5).bfloat16().cuda()

        p_part = [
            "in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>",
            "out buffer<f32>", "out buffer<i32>",
            "i32", "i32", "f32", "i32", "i32", "i32",
        ]
        p_fin = ["in buffer<f32>", "in buffer<i32>", "out buffer<i64>", "i32", "i32"]

        for seqs, step in ((1, 0), (2, 3), (3, 1), (4, 5), (8, 6)):
            with self.subTest(seqs=seqs, step=step):
                latent = (torch.randn(seqs, R, generator=g) * 0.5).bfloat16().cuda()
                base = (torch.randn(seqs, NSTEPS, VP, generator=g) * 4.0).bfloat16().cuda()
                # Inject clear winners in unpadded vocab and poison padded vocab >= 200058
                winners = [(17 + 13107 * s) % VU for s in range(seqs)]
                for s, w_tok in enumerate(winners):
                    base[s, step, w_tok] = 25.0
                base[:, step, VU:] = 100.0
                base[:, step, VU + 3] = float("nan")

                m = {
                    "schema_version": 7,
                    "model": "markov-fused-fold-test",
                    "vars": {},
                    "states": {},
                    "buffers": {
                        "lat": {"dtype": "bf16", "shape": [seqs, R], "kind": "input"},
                        "fw": {"dtype": "bf16", "shape": [VP, R], "kind": "input"},
                        "fb": {"dtype": "bf16", "shape": [VP], "kind": "input"},
                        "base": {"dtype": "bf16", "shape": [seqs, NSTEPS, VP], "kind": "input"},
                        "pmax": {"dtype": "f32", "shape": [seqs, 220], "kind": "workspace"},
                        "pidx": {"dtype": "i32", "shape": [seqs, 220], "kind": "workspace"},
                        "draft": {"dtype": "i64", "shape": [seqs, NSTEPS], "kind": "output"},
                    },
                    "modules": self._spec_mod("spec_markov_rows"),
                    "ops": {
                        "fused": {
                            "params": p_part[:6] + ["out buffer<i64>"] + p_part[6:],
                            "impl": {
                                "launches": [
                                    {
                                        "module": "spec_markov_rows",
                                        "entry": "kern_markov_fold_softcap_argmax_partial",
                                        "params": p_part,
                                        "grid": [220, 1, 1],
                                        "block": [256, 1, 1],
                                        "args": [{"param": j} for j in range(6)] + [{"param": j} for j in range(7, 13)],
                                    },
                                    {
                                        "module": "spec_markov_rows",
                                        "entry": "kern_markov_argmax_final_pdl",
                                        "params": p_fin,
                                        "grid": [seqs, 1, 1],
                                        "block": [256, 1, 1],
                                        "pdl": True,
                                        "args": [{"param": 4}, {"param": 5}, {"param": 6}, {"param": 12}, {"i32": 220}],
                                    },
                                ]
                            },
                        }
                    },
                    "programs": {
                        "run": {
                            "calls": [
                                {
                                    "op": "fused",
                                    "args": [
                                        {"buf": "lat"}, {"buf": "fw"}, {"buf": "fb"}, {"buf": "base"},
                                        {"buf": "pmax"}, {"buf": "pidx"}, {"buf": "draft", "offset": step * 8},
                                        {"i32": seqs}, {"i32": step}, {"f32": 30.0},
                                        {"i32": VP}, {"i32": VU}, {"i32": NSTEPS},
                                    ],
                                }
                            ]
                        }
                    },
                }

                def setup(rt, v):
                    v["lat"].copy_(latent)
                    v["fw"].copy_(fold_w)
                    v["fb"].copy_(fold_b)
                    v["base"].copy_(base)

                _, views = self._run_manifest(m, setup, "run", {})
                step_bias = (latent.float() @ fold_w[:VU].float().T).bfloat16()
                bias_bf = (step_bias.float() + fold_b[:VU].float()[None, :]).bfloat16()
                x_bf = (base[:, step, :VU].float() + bias_bf.float()).bfloat16()
                logits = (30.0 * torch.tanh(x_bf.float() / 30.0)).bfloat16()
                ref_tok = logits.argmax(dim=-1)
                self.assertTrue(torch.equal(views["draft"][:, step], ref_tok))
                self.assertEqual(views["draft"][:, step].tolist(), winners)


if __name__ == "__main__":
    unittest.main()
