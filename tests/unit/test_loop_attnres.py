"""Unit and integration tests for M1 Inkling-2-Mini kernels and loop_manifest.py."""

import ctypes
import importlib.util
import json
import os
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
import loop_manifest as lm

_WORKSPACE = ROOT.parent
KERN_BIN = Path(os.environ.get("KERN_BIN", _WORKSPACE / "kern/target/release/kern"))
KERN_PY_ROOT = Path(os.environ.get("KERN_PY_ROOT", _WORKSPACE / "kern/crates/kern-py/python"))
KERN_LIB_SO = Path(os.environ.get("KERN_LIB_SO", _WORKSPACE / "kern/target/release/lib_kern.so"))
KERNELS_DIR = ROOT / "build" / "sm_120"


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


def build_m1_verify_manifest(max_seqs: int = 2, max_tokens: int = 16) -> dict:
    """Build a complete 48-visit M1 schema-v7 manifest (sublayers stubbed with copy_bf16)."""
    modules: dict = {}
    buffers: dict = {}
    ops: dict = {}

    lm.register_modules(modules, KERNELS_DIR)
    lm.register_buffers(buffers, max_seqs=max_seqs, max_tokens=max_tokens, include_aux=True)
    buffers["token_ids"] = {"dtype": "i64", "shape": ["tokens"], "kind": "input", "fill": "token"}
    buffers["sample_probs"] = {"dtype": "f32", "shape": ["seqs", lm.VOCAB], "kind": "output"}
    lm.register_ops(ops, modules, None, max_seqs=max_seqs, max_tokens=max_tokens)

    derive_calls: list[dict] = []
    lm.emit_derive_calls(buffers, ops, derive_calls)

    t_elems = lm.expr(lm.mul("tokens", lm.H))
    prefill_calls: list[dict] = lm.emit_pre_loop_calls("token_ids")
    for visit_idx, phys_idx, _phase, pass_idx in lm.visit_schedule():
        pre_attn, pre_mlp, post_mlp = lm.emit_layer_calls(
            visit_idx,
            phys_idx,
            pass_idx,
            max_tokens=max_tokens,
            mode="prefill",
            capture_aux=True,
        )
        prefill_calls += pre_attn
        # Stub M3 attention+sconv: project x_qkv[0] -> attn_out
        prefill_calls.append(
            {
                "label": f"v{visit_idx}.stub_attn",
                "op": "copy_bf16",
                "args": [lm.b("x_qkv", 0), lm.b("attn_out"), t_elems],
            }
        )
        prefill_calls += pre_mlp
        # Stub M2 MoE+mlp_sconv: project x_mlp -> mlp_out
        prefill_calls.append(
            {
                "label": f"v{visit_idx}.stub_mlp",
                "op": "copy_bf16",
                "args": [lm.b("x_mlp"), lm.b("mlp_out"), t_elems],
            }
        )
        prefill_calls += post_mlp

    prefill_calls += lm.emit_post_loop_calls(max_tokens=max_tokens, mode="prefill", capture_aux=True)
    prefill_calls.append(
        {
            "label": "sample_copy_check",
            "op": "sample_copy_f32",
            "args": [lm.b("logits"), lm.b("sample_probs"), lm.i32(lm.VOCAB)],
        }
    )

    return {
        "schema_version": 7,
        "model": "inkling-2-mini-m1-test",
        "vars": {
            "tokens": {"max": max_tokens, "axis": "rows"},
            "seqs": {"max": max_seqs, "axis": "groups"},
        },
        "states": {},
        "buffers": buffers,
        "modules": modules,
        "ops": ops,
        "programs": {
            "derive": {"derive": True, "calls": derive_calls},
            "prefill": {"batch": {"groups": 1, "rows": "tokens"}, "calls": prefill_calls},
        },
    }


class TestLoopManifestCPU(unittest.TestCase):
    def test_visit_schedule_and_query_mapping(self):
        sched = lm.visit_schedule()
        self.assertEqual(len(sched), 48)
        self.assertEqual([x[1] for x in sched[0:6]], list(range(0, 6)))
        self.assertEqual([x[1] for x in sched[6:18]], list(range(6, 18)))
        self.assertEqual([x[1] for x in sched[18:30]], list(range(6, 18)))
        self.assertEqual([x[1] for x in sched[30:42]], list(range(6, 18)))
        self.assertEqual([x[1] for x in sched[42:48]], list(range(18, 24)))
        self.assertEqual([x[3] for x in sched[6:18]], [0] * 12)
        self.assertEqual([x[3] for x in sched[18:30]], [1] * 12)
        self.assertEqual([x[3] for x in sched[30:42]], [2] * 12)

        # Layer 0 skips attn_res on attn input and uses layers.0.attn_norm directly
        pre0, mlp0, _ = lm.emit_layer_calls(0, 0, 0, max_tokens=16)
        self.assertEqual([c["op"] for c in pre0], ["rms_norm_bf16"])
        self.assertEqual([c["op"] for c in mlp0], ["copy_bf16", "attn_res_norm_bf16"])

        # Pass 1 & 2 start (visits 18 & 30): save prev_blocks + reset prelude_partial + mix + cross_pass + 3 controllers
        pre18, mlp18, _ = lm.emit_layer_calls(18, 6, 1, max_tokens=16)
        self.assertEqual(
            [c["op"] for c in pre18],
            [
                "copy_bf16",
                "copy_bf16",
                "copy_bf16",
                "attn_res_mix_f32",
                "cross_pass_attn_res_f32",
                "loop_controller",
                "loop_controller",
                "loop_controller",
            ],
        )
        # Local layer 6 broadcasts mix_qkv[0] to all 3 controllers and reuses attn_norm
        ctrl_calls = [c for c in pre18 if c["op"] == "loop_controller"]
        for c in ctrl_calls:
            self.assertEqual(c["args"][0], {"buf": "mix_qkv"})
            self.assertEqual(c["args"][4], {"buf": "w.l6.attn_norm"})

        # Full layer 11 on Pass 1 (visit 23) captures aux tap 1 and uses distinct q/k/v norms & mix offsets
        pre23, _, _ = lm.emit_layer_calls(23, 11, 1, max_tokens=16, capture_aux=True)
        self.assertEqual(pre23[0]["op"], "tap_bf16")
        self.assertEqual(pre23[0]["args"][-1], {"i32": 1})
        ctrl23 = [c for c in pre23 if c["op"] == "loop_controller"]
        self.assertEqual([c["args"][4]["buf"] for c in ctrl23], ["w.l11.q_input_norm", "w.l11.k_input_norm", "w.l11.v_input_norm"])
        self.assertEqual([c["args"][0].get("offset", 0) for c in ctrl23], [0, 16 * lm.H * 4, 2 * 16 * lm.H * 4])

    def test_kern_verify_static_cpu(self):
        if not KERN_BIN.exists():
            self.skipTest(f"kern binary not found at {KERN_BIN}")
        manifest = build_m1_verify_manifest(max_seqs=2, max_tokens=16)
        with tempfile.TemporaryDirectory() as tmp:
            mpath = Path(tmp) / "model.json"
            mpath.write_text(json.dumps(manifest))
            res = subprocess.run([str(KERN_BIN), "verify", str(mpath)], capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"kern verify failed:\n{res.stdout}\n{res.stderr}")


def _ref_rms_norm(x, w, eps=1e-6):
    import torch

    xf = x.float()
    rstd = torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + eps)
    if w is None:
        return (xf * rstd).to(torch.bfloat16)
    return (xf * rstd * w.float()).to(torch.bfloat16)


def _ref_attn_res(sources, queries, addend=None, norm_w=None, eps=1e-6, norm_eps=1e-6):
    import torch

    srcs = [s.float() for s in sources]
    if addend is not None:
        srcs[-1] = srcs[-1] + addend.float()
    inv_rms = [torch.rsqrt(s.square().mean(dim=-1) + eps) for s in srcs]
    logits = torch.stack(
        [
            torch.stack([(s @ queries[q].float()) * inv_rms[n] for n, s in enumerate(srcs)], dim=-1)
            for q in range(queries.shape[0])
        ],
        dim=0,
    )
    probs = torch.softmax(logits, dim=-1)
    weights = probs * torch.stack(inv_rms, dim=-1).unsqueeze(0)
    stacked = torch.stack(srcs, dim=-2)
    mix = torch.einsum("qtn,tnd->qtd", weights, stacked)
    if norm_w is None:
        return mix
    nw = norm_w.float().unsqueeze(1) if norm_w.dim() == 2 else norm_w.float().view(1, 1, -1)
    rstd = torch.rsqrt(mix.square().mean(dim=-1, keepdim=True) + norm_eps)
    return (mix * rstd * nw).to(torch.bfloat16)


def _ref_loop_controller(dest_f32, msg_f32, w_down, w_forget, w_gain, norm_w, eps=1e-6):
    import torch

    u = torch.cat([_ref_rms_norm(dest_f32, None, eps), _ref_rms_norm(msg_f32, None, eps)], dim=-1)
    h = (u.float() @ w_down.float().T).to(torch.bfloat16)
    f_pre = h.float() @ w_forget.float().T
    g_pre = h.float() @ w_gain.float().T
    merged = (1.0 - torch.tanh(f_pre)) * dest_f32 + torch.tanh(g_pre) * msg_f32
    return _ref_rms_norm(merged, norm_w, eps)


class TestM1KernelsGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        if not torch.cuda.is_available():
            raise unittest.SkipTest("requires a CUDA GPU")
        kern_mod, region_fn = _load_kern()
        cls.kern = kern_mod
        cls.region = staticmethod(region_fn)
        cls.build_meta = json.loads((KERNELS_DIR / "build.json").read_text())

    def _run_manifest(self, m: dict, weights_path: Path | None = None):
        with tempfile.TemporaryDirectory() as tmp:
            mpath = Path(tmp) / "model.json"
            mpath.write_text(json.dumps(m))
            wlist = [str(weights_path)] if weights_path else []
            rt = self.kern.Runtime(str(mpath), str(KERNELS_DIR), wlist, 0)
        views = {
            name: self.region(rt, m, name)
            for name, spec in m["buffers"].items()
            if spec["kind"] != "source"
        }
        return rt, views

    def _mod(self, name: str) -> dict:
        return {"source": f"{name}.cubin", "sha256": self.build_meta["modules"][name]["sha256"]}

    def test_rms_norm_bf16_and_f32(self):
        import torch

        torch.manual_seed(11)
        D = 2304
        max_t = 257
        for num_inputs in (1, 3):
            with self.subTest(num_inputs=num_inputs):
                m = {
                    "schema_version": 7,
                    "model": "test-rms-norm",
                    "vars": {"tokens": {"max": max_t}},
                    "states": {},
                    "buffers": {
                        "x_bf16": {"dtype": "bf16", "shape": ["tokens", num_inputs, D], "kind": "input"},
                        "x_f32": {"dtype": "f32", "shape": ["tokens", num_inputs, D], "kind": "input"},
                        "w": {"dtype": "bf16", "shape": [num_inputs, D], "kind": "input"},
                        "y_bf16": {"dtype": "bf16", "shape": ["tokens", num_inputs, D], "kind": "output"},
                        "y_f32": {"dtype": "bf16", "shape": ["tokens", num_inputs, D], "kind": "output"},
                    },
                    "modules": {"norm_attnres": self._mod("norm_attnres")},
                    "ops": {
                        "norm_bf16": {
                            "params": ["in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32", "i32", "f32"],
                            "impl": {
                                "launches": [
                                    {
                                        "module": "norm_attnres",
                                        "entry": "kern_inkling2_rms_norm_bf16",
                                        "params": ["in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32", "i32", "f32"],
                                        "grid": ["tokens", num_inputs, 1],
                                        "block": [256, 1, 1],
                                        "args": [{"param": j} for j in range(7)],
                                    }
                                ]
                            },
                        },
                        "norm_f32": {
                            "params": ["in buffer<f32>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32", "i32", "f32"],
                            "impl": {
                                "launches": [
                                    {
                                        "module": "norm_attnres",
                                        "entry": "kern_inkling2_rms_norm_f32_bf16",
                                        "params": ["in buffer<f32>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32", "i32", "f32"],
                                        "grid": ["tokens", num_inputs, 1],
                                        "block": [256, 1, 1],
                                        "args": [{"param": j} for j in range(7)],
                                    }
                                ]
                            },
                        },
                    },
                    "programs": {
                        "run": {
                            "calls": [
                                {
                                    "op": "norm_bf16",
                                    "args": [{"buf": "x_bf16"}, {"buf": "w"}, {"buf": "y_bf16"}, {"var": "tokens"}, {"i32": D}, {"i32": num_inputs}, {"f32": 1e-6}],
                                },
                                {
                                    "op": "norm_f32",
                                    "args": [{"buf": "x_f32"}, {"buf": "w"}, {"buf": "y_f32"}, {"var": "tokens"}, {"i32": D}, {"i32": num_inputs}, {"f32": 1e-6}],
                                },
                            ]
                        }
                    },
                }
                rt, views = self._run_manifest(m)
                for T in (1, 7, 64, 257):
                    x_b = torch.randn(T, num_inputs, D, dtype=torch.bfloat16)
                    x_f = torch.randn(T, num_inputs, D, dtype=torch.float32)
                    # Include negative/non-unit weights to verify NO +1 Gemma offset
                    w = torch.randn(num_inputs, D, dtype=torch.bfloat16) * 0.5 - 0.3
                    views["x_bf16"][:T].copy_(x_b)
                    views["x_f32"][:T].copy_(x_f)
                    views["w"].copy_(w)
                    torch.cuda.synchronize()
                    rt.run("run", {"tokens": T})
                    torch.cuda.synchronize()
                    ref_b = torch.stack([_ref_rms_norm(x_b[:, q, :], w[q]) for q in range(num_inputs)], dim=1)
                    ref_f = torch.stack([_ref_rms_norm(x_f[:, q, :], w[q]) for q in range(num_inputs)], dim=1)
                    torch.testing.assert_close(views["y_bf16"][:T].cpu(), ref_b, atol=2e-2, rtol=1e-2)
                    torch.testing.assert_close(views["y_f32"][:T].cpu(), ref_f, atol=2e-2, rtol=1e-2)

    def test_attn_res_all_variants_and_cross_pass_eager_and_graph(self):
        import torch

        torch.manual_seed(22)
        D, max_t = 2304, 64
        for n_stack, num_q in ((1, 1), (2, 3), (4, 1)):
            with self.subTest(n_stack=n_stack, num_q=num_q):
                m = {
                    "schema_version": 7,
                    "model": "test-attnres",
                    "vars": {"tokens": {"max": max_t}},
                    "states": {},
                    "buffers": {
                        "stack": {"dtype": "bf16", "shape": [n_stack, "tokens", D], "kind": "input"},
                        "partial": {"dtype": "bf16", "shape": ["tokens", D], "kind": "input"},
                        "addend": {"dtype": "bf16", "shape": ["tokens", D], "kind": "input"},
                        "raw_q": {"dtype": "f32", "shape": [num_q, D], "kind": "input"},
                        "q_scaled": {"dtype": "f32", "shape": [num_q, D], "kind": "output"},
                        "norm_w": {"dtype": "bf16", "shape": [num_q, D], "kind": "input"},
                        "out_norm": {"dtype": "bf16", "shape": [num_q, "tokens", D], "kind": "output"},
                        "out_add_norm": {"dtype": "bf16", "shape": [num_q, "tokens", D], "kind": "output"},
                        "out_mix": {"dtype": "f32", "shape": [num_q, "tokens", D], "kind": "output"},
                        "out_add_mix": {"dtype": "f32", "shape": [num_q, "tokens", D], "kind": "output"},
                        "prev_blocks": {"dtype": "bf16", "shape": [2, "tokens", D], "kind": "input"},
                        "cross_q": {"dtype": "f32", "shape": [4, D], "kind": "input"},
                        "out_msgs": {"dtype": "f32", "shape": [4, "tokens", D], "kind": "output"},
                    },
                    "modules": {
                        "norm_attnres": self._mod("norm_attnres"),
                        "loop_controller": self._mod("loop_controller"),
                        "unembed_sample": self._mod("unembed_sample"),
                    },
                    "ops": {},
                    "programs": {},
                }
                lm.register_ops(m["ops"], m["modules"])
                # Remove unused unembed ops from minimal manifest
                for unused in ("embed_bf16", "rms_norm_bf16", "rms_norm3_bf16", "copy_bf16", "tap_bf16", "add2_bf16", "add3_bf16", "loop_controller", "last_row_bf16", "unembed_f32", "softcap_mask_f32", "argmax_f32", "sample_copy_f32"):
                    m["ops"].pop(unused, None)
                m["modules"] = {"norm_attnres": self._mod("norm_attnres")}

                s_stride = lm.i64(max_t * D)
                q_stride = lm.i64(max_t * D)
                T_var = {"var": "tokens"}
                m["programs"]["run"] = {
                    "calls": [
                        {
                            "op": "scale_f32",
                            "args": [lm.b("raw_q"), lm.b("q_scaled"), lm.i64(num_q * D), lm.f32(lm.INV_SQRT_H)],
                        },
                        {
                            "op": "attn_res_norm_bf16",
                            "args": [lm.b("stack"), lm.b("partial"), lm.b("q_scaled"), lm.b("norm_w"), lm.b("out_norm"), s_stride, q_stride, T_var, lm.i32(D), lm.i32(n_stack), lm.i32(num_q), lm.f32(1e-6), lm.f32(1e-6)],
                        },
                        {
                            "op": "attn_res_addend_norm_bf16",
                            "args": [lm.b("stack"), lm.b("partial"), lm.b("addend"), lm.b("q_scaled"), lm.b("norm_w"), lm.b("out_add_norm"), s_stride, q_stride, T_var, lm.i32(D), lm.i32(n_stack), lm.i32(num_q), lm.f32(1e-6), lm.f32(1e-6)],
                        },
                        {
                            "op": "attn_res_mix_f32",
                            "args": [lm.b("stack"), lm.b("partial"), lm.b("q_scaled"), lm.b("out_mix"), s_stride, q_stride, T_var, lm.i32(D), lm.i32(n_stack), lm.i32(num_q), lm.f32(1e-6)],
                        },
                        {
                            "op": "attn_res_addend_mix_f32",
                            "args": [lm.b("stack"), lm.b("partial"), lm.b("addend"), lm.b("q_scaled"), lm.b("out_add_mix"), s_stride, q_stride, T_var, lm.i32(D), lm.i32(n_stack), lm.i32(num_q), lm.f32(1e-6)],
                        },
                        {
                            "op": "cross_pass_attn_res_f32",
                            "args": [lm.b("prev_blocks"), lm.b("cross_q"), lm.b("out_msgs"), s_stride, q_stride, T_var, lm.i32(D), lm.i32(4), lm.f32(1e-6)],
                        },
                    ]
                }
                rt, views = self._run_manifest(m)

                for T in (1, 7, 64):
                    stack_h = torch.randn(n_stack, max_t, D, dtype=torch.bfloat16)
                    partial_h = torch.randn(max_t, D, dtype=torch.bfloat16)
                    addend_h = torch.randn(max_t, D, dtype=torch.bfloat16)
                    raw_q = torch.randn(num_q, D, dtype=torch.float32) * 2.0
                    norm_w = torch.randn(num_q, D, dtype=torch.bfloat16)
                    prev_h = torch.randn(2, max_t, D, dtype=torch.bfloat16)
                    cross_q = torch.randn(4, D, dtype=torch.float32) * (1.0 / 48.0)

                    views["stack"].copy_(stack_h)
                    views["partial"].copy_(partial_h)
                    views["addend"].copy_(addend_h)
                    views["raw_q"].copy_(raw_q)
                    views["norm_w"].copy_(norm_w)
                    views["prev_blocks"].copy_(prev_h)
                    views["cross_q"].copy_(cross_q)
                    torch.cuda.synchronize()
                    rt.run("run", {"tokens": T})
                    torch.cuda.synchronize()

                    q_s = raw_q * (1.0 / 48.0)
                    srcs = [stack_h[n, :T] for n in range(n_stack)] + [partial_h[:T]]
                    ref_norm = _ref_attn_res(srcs, q_s, norm_w=norm_w)
                    ref_add_norm = _ref_attn_res(srcs, q_s, addend=addend_h[:T], norm_w=norm_w)
                    ref_mix = _ref_attn_res(srcs, q_s)
                    ref_add_mix = _ref_attn_res(srcs, q_s, addend=addend_h[:T])
                    ref_msgs = _ref_attn_res([prev_h[0, :T], prev_h[1, :T]], cross_q)

                    torch.testing.assert_close(views["out_norm"][:, :T].cpu(), ref_norm, atol=2e-2, rtol=1e-2)
                    torch.testing.assert_close(views["out_add_norm"][:, :T].cpu(), ref_add_norm, atol=2e-2, rtol=1e-2)
                    torch.testing.assert_close(views["out_mix"][:, :T].cpu(), ref_mix, atol=2e-5, rtol=1e-5)
                    torch.testing.assert_close(views["out_add_mix"][:, :T].cpu(), ref_add_mix, atol=2e-5, rtol=1e-5)
                    torch.testing.assert_close(views["out_msgs"][:, :T].cpu(), ref_msgs, atol=2e-5, rtol=1e-5)

                # CUDA Graph capture & replay check for T=19
                T = 19
                stream = torch.cuda.Stream()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.stream(stream):
                    rt.enqueue_after("run", {"tokens": T}, stream.cuda_stream)
                stream.synchronize()
                with torch.cuda.graph(graph, stream=stream):
                    rt.enqueue_after("run", {"tokens": T}, stream.cuda_stream)
                new_partial = torch.randn(max_t, D, dtype=torch.bfloat16)
                views["partial"].copy_(new_partial)
                torch.cuda.synchronize()
                graph.replay()
                torch.cuda.synchronize()
                srcs2 = [stack_h[n, :T] for n in range(n_stack)] + [new_partial[:T]]
                ref_mix2 = _ref_attn_res(srcs2, raw_q * (1.0 / 48.0))
                torch.testing.assert_close(views["out_mix"][:, :T].cpu(), ref_mix2, atol=2e-5, rtol=1e-5)

    def test_loop_controller_no_fma_eager_and_graph(self):
        import torch

        torch.manual_seed(33)
        D, R, max_t = 2304, 64, 257
        m = {
            "schema_version": 7,
            "model": "test-loop-ctrl",
            "vars": {"tokens": {"max": max_t}},
            "states": {},
            "buffers": {
                "in_f32": {"dtype": "f32", "shape": ["tokens", D], "kind": "input"},
                "msg_f32": {"dtype": "f32", "shape": ["tokens", D], "kind": "input"},
                "w_down": {"dtype": "bf16", "shape": [R, 2 * D], "kind": "input"},
                "w_fg": {"dtype": "bf16", "shape": [2 * D, R], "kind": "input"},
                "norm_w": {"dtype": "bf16", "shape": [D], "kind": "input"},
                "out_bf16": {"dtype": "bf16", "shape": ["tokens", D], "kind": "output"},
            },
            "modules": {
                "norm_attnres": self._mod("norm_attnres"),
                "loop_controller": self._mod("loop_controller"),
                "unembed_sample": self._mod("unembed_sample"),
            },
            "ops": {},
            "programs": {},
        }
        lm.register_ops(m["ops"], m["modules"])
        m["ops"] = {"loop_controller": m["ops"]["loop_controller"]}
        m["modules"] = {"loop_controller": self._mod("loop_controller")}
        m["programs"]["ctrl"] = {
            "calls": [
                {
                    "op": "loop_controller",
                    "args": [
                        lm.b("in_f32"),
                        lm.b("msg_f32"),
                        lm.b("w_down"),
                        lm.b("w_fg"),
                        lm.b("norm_w"),
                        lm.b("out_bf16"),
                        {"var": "tokens"},
                        lm.f32(1e-6),
                    ],
                }
            ]
        }
        rt, views = self._run_manifest(m)

        w_down = torch.randn(R, 2 * D, dtype=torch.bfloat16) * 0.1
        w_forget = torch.randn(D, R, dtype=torch.bfloat16) * 0.1
        w_gain = torch.randn(D, R, dtype=torch.bfloat16) * 0.1
        norm_w = torch.randn(D, dtype=torch.bfloat16)
        views["w_down"].copy_(w_down)
        views["w_fg"][:D].copy_(w_forget)
        views["w_fg"][D:].copy_(w_gain)
        views["norm_w"].copy_(norm_w)

        for T in (1, 8, 64, 257):
            dest = torch.randn(T, D, dtype=torch.float32) * 1.5
            msg = torch.randn(T, D, dtype=torch.float32) * 1.5
            views["in_f32"][:T].copy_(dest)
            views["msg_f32"][:T].copy_(msg)
            torch.cuda.synchronize()
            rt.run("ctrl", {"tokens": T})
            torch.cuda.synchronize()
            ref = _ref_loop_controller(dest, msg, w_down, w_forget, w_gain, norm_w)
            torch.testing.assert_close(views["out_bf16"][:T].cpu(), ref, atol=7e-2, rtol=3e-2)
            self.assertLess((views["out_bf16"][:T].cpu().float() - ref.float()).abs().mean().item(), 2e-3)

        # CUDAGraph capture & replay
        T = 13
        stream = torch.cuda.Stream()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.stream(stream):
            rt.enqueue_after("ctrl", {"tokens": T}, stream.cuda_stream)
        stream.synchronize()
        with torch.cuda.graph(graph, stream=stream):
            rt.enqueue_after("ctrl", {"tokens": T}, stream.cuda_stream)
        dest2 = torch.randn(T, D, dtype=torch.float32) * 2.0
        msg2 = torch.randn(T, D, dtype=torch.float32) * 2.0
        views["in_f32"][:T].copy_(dest2)
        views["msg_f32"][:T].copy_(msg2)
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        ref2 = _ref_loop_controller(dest2, msg2, w_down, w_forget, w_gain, norm_w)
        torch.testing.assert_close(views["out_bf16"][:T].cpu(), ref2, atol=7e-2, rtol=3e-2)
        self.assertLess((views["out_bf16"][:T].cpu().float() - ref2.float()).abs().mean().item(), 2e-3)

    def test_embed_unembed_softcap_mask_argmax_and_sample_copy(self):
        import torch

        torch.manual_seed(44)
        D, V_PAD, V_VALID = 2304, lm.VOCAB, lm.UNPADDED_VOCAB
        max_s, max_t = 4, 12
        m = {
            "schema_version": 7,
            "model": "test-unembed",
            "vars": {"seqs": {"max": max_s}, "tokens": {"max": max_t}},
            "states": {},
            "buffers": {
                "ids": {"dtype": "i64", "shape": ["tokens"], "kind": "input"},
                "embed_w": {"dtype": "bf16", "shape": [V_PAD, D], "kind": "input"},
                "embed_out": {"dtype": "bf16", "shape": ["tokens", D], "kind": "output"},
                "last_h": {"dtype": "bf16", "shape": ["seqs", D], "kind": "output"},
                "unembed_w": {"dtype": "bf16", "shape": [V_PAD, D], "kind": "input"},
                "logits": {"dtype": "f32", "shape": ["seqs", V_PAD], "kind": "output"},
                "probs": {"dtype": "f32", "shape": ["seqs", V_PAD], "kind": "output"},
                "next_token": {"dtype": "i64", "shape": ["seqs"], "kind": "output"},
            },
            "modules": {
                "norm_attnres": self._mod("norm_attnres"),
                "loop_controller": self._mod("loop_controller"),
                "unembed_sample": self._mod("unembed_sample"),
            },
            "ops": {},
            "programs": {},
        }
        lm.register_ops(m["ops"], m["modules"])
        keep_ops = ("embed_bf16", "last_row_bf16", "unembed_f32", "softcap_mask_f32", "argmax_f32", "sample_copy_f32")
        m["ops"] = {k: m["ops"][k] for k in keep_ops}
        m["modules"] = {"unembed_sample": self._mod("unembed_sample")}
        m["programs"]["run_prefill_one"] = {
            "calls": [
                {"op": "embed_bf16", "args": [lm.b("ids"), lm.b("embed_w"), lm.b("embed_out"), lm.i32(D), lm.i32(V_PAD)]},
                {"op": "last_row_bf16", "args": [lm.b("last_h"), lm.b("embed_out"), lm.i32(D), lm.i32(D), {"var": "tokens"}]},
                {"op": "unembed_f32", "args": [lm.b("last_h"), lm.b("unembed_w"), lm.b("logits"), {"var": "seqs"}, lm.i32(V_PAD), lm.i32(D)]},
                {"op": "softcap_mask_f32", "args": [lm.b("logits"), {"var": "seqs"}, lm.i32(V_PAD), lm.i32(V_VALID), lm.f32(30.0)]},
                {"op": "sample_copy_f32", "args": [lm.b("logits"), lm.b("probs"), lm.i32(V_PAD)]},
                {"op": "argmax_f32", "args": [lm.b("logits"), lm.b("next_token"), {"var": "seqs"}, lm.i32(V_PAD)]},
            ]
        }
        rt, views = self._run_manifest(m)

        # Put huge positive weights in padded tail [200058:] so unmasked unembed would pick >=200058!
        embed_w = torch.randn(V_PAD, D, dtype=torch.bfloat16) * 0.25
        unembed_w = torch.randn(V_PAD, D, dtype=torch.bfloat16) * 0.1
        ids = torch.tensor([0, 42, 12345, 200057, 7], dtype=torch.int64)
        # Ensure padded tail would dominate if not masked
        unembed_w[V_VALID:] = embed_w[ids[-1]].unsqueeze(0) * 10.0

        views["embed_w"].copy_(embed_w)
        views["unembed_w"].copy_(unembed_w)
        views["ids"][: len(ids)].copy_(ids)
        torch.cuda.synchronize()
        rt.run("run_prefill_one", {"seqs": 1, "tokens": len(ids)})
        torch.cuda.synchronize()

        ref_emb = embed_w[ids]
        torch.testing.assert_close(views["embed_out"][: len(ids)].cpu(), ref_emb, atol=0, rtol=0)
        torch.testing.assert_close(views["last_h"][:1].cpu(), ref_emb[-1:].cpu(), atol=0, rtol=0)

        raw_logits = ref_emb[-1:].float() @ unembed_w.float().T
        ref_logits = 30.0 * torch.tanh(raw_logits / 30.0)
        ref_logits[:, V_VALID:] = float("-inf")
        got_logits = views["logits"][:1].cpu()
        self.assertTrue(torch.all(torch.isneginf(got_logits[:, V_VALID:])))
        torch.testing.assert_close(got_logits[:, :V_VALID], ref_logits[:, :V_VALID], atol=2e-2, rtol=1e-3)
        torch.testing.assert_close(views["probs"][:1, :V_VALID].cpu(), got_logits[:, :V_VALID], atol=0, rtol=0)
        self.assertEqual(int(views["next_token"][0].item()), int(ref_logits.argmax(dim=-1).item()))
        self.assertLess(int(views["next_token"][0].item()), V_VALID)

    def test_multi_visit_loop_manifest_integration(self):
        import torch
        from safetensors.torch import save_file

        torch.manual_seed(55)
        max_seqs, max_tokens = 2, 16
        manifest = build_m1_verify_manifest(max_seqs=max_seqs, max_tokens=max_tokens)

        # Build deterministic synthetic safetensors matching checkpoint tensor names
        weights: dict[str, torch.Tensor] = {
            "model.llm.embed.weight": torch.randn(lm.VOCAB, lm.H, dtype=torch.bfloat16) * 0.2,
            "model.llm.embed_norm.weight": torch.randn(lm.H, dtype=torch.bfloat16) * 0.25,
            "model.llm.norm.weight": torch.randn(lm.H, dtype=torch.bfloat16) * 0.25,
            "model.llm.unembed.weight": torch.randn(lm.VOCAB, lm.H, dtype=torch.bfloat16) * 0.05,
            "model.llm.attn_res.output_query": torch.randn(lm.H, dtype=torch.float32),
            "model.llm.middle_cycle_cross_pass_queries": torch.randn(24, lm.H, dtype=torch.float32),
            "model.llm.middle_cycle_cross_pass_kv_queries": torch.randn(24, lm.H, dtype=torch.float32),
        }
        # Force padded tail to be large so masking is exercised end-to-end
        weights["model.llm.unembed.weight"][lm.UNPADDED_VOCAB:] = 5.0

        for i in range(lm.NUM_PHYSICAL_LAYERS):
            if i > 0:
                qshape = (3, lm.H) if i in lm.FULL_LAYERS else (lm.H,)
                weights[f"model.llm.attn_res.attn_queries.{i - 1}"] = torch.randn(*qshape, dtype=torch.float32)
            weights[f"model.llm.attn_res.mlp_queries.{i}"] = torch.randn(lm.H, dtype=torch.float32)
            if i in lm.FULL_LAYERS:
                for nm in ("q_input_norm", "k_input_norm", "v_input_norm"):
                    weights[f"model.llm.layers.{i}.{nm}.weight"] = torch.randn(lm.H, dtype=torch.bfloat16) * 0.25
            else:
                weights[f"model.llm.layers.{i}.attn_norm.weight"] = torch.randn(lm.H, dtype=torch.bfloat16) * 0.25
            weights[f"model.llm.layers.{i}.mlp_norm.weight"] = torch.randn(lm.H, dtype=torch.bfloat16) * 0.25
            if lm.CORE_START <= i < lm.CORE_END:
                for s in lm.CTRL_STREAMS:
                    pfx = f"model.llm.layers.{i}.controllers.{s}"
                    weights[f"{pfx}.down_weight"] = torch.randn(lm.RANK, 2 * lm.H, dtype=torch.bfloat16) * 0.05
                    weights[f"{pfx}.forget_weight"] = torch.randn(lm.H, lm.RANK, dtype=torch.bfloat16) * 0.05
                    weights[f"{pfx}.gain_weight"] = torch.randn(lm.H, lm.RANK, dtype=torch.bfloat16) * 0.05

        with tempfile.TemporaryDirectory() as tmp:
            wpath = Path(tmp) / "weights.safetensors"
            save_file(weights, str(wpath))
            rt, views = self._run_manifest(manifest, weights_path=wpath)

            T = 8
            ids = torch.tensor([1, 5, 19, 100, 500, 1234, 42, 200057], dtype=torch.int64)
            views["token_ids"][:T].copy_(ids)
            torch.cuda.synchronize()
            rt.run("prefill", {"seqs": 1, "tokens": T})
            torch.cuda.synchronize()

            # PyTorch golden 48-visit reference matching loop_manifest.py + stub sublayers
            partial = _ref_rms_norm(weights["model.llm.embed.weight"][ids], weights["model.llm.embed_norm.weight"])
            stack = []
            aux_ref = {}
            prelude_partial = None
            prev_blocks = None

            cross_q = weights["model.llm.middle_cycle_cross_pass_queries"] * lm.INV_SQRT_H
            cross_kv = weights["model.llm.middle_cycle_cross_pass_kv_queries"] * lm.INV_SQRT_H

            for visit_idx, phys_idx, _phase, pass_idx in lm.visit_schedule():
                if visit_idx in (5, 23, 41):
                    aux_ref[visit_idx] = partial.clone()
                if phys_idx == lm.CORE_START and pass_idx in (1, 2):
                    prev_blocks = [stack[2].clone(), partial.clone()]
                    stack = [stack[0]]
                    partial = prelude_partial.clone()

                first = (phys_idx % lm.BLOCK_SIZE) == 0
                is_full = phys_idx in lm.FULL_LAYERS
                if pass_idx == 0:
                    if phys_idx == 0:
                        x_q = _ref_rms_norm(partial, weights["model.llm.layers.0.attn_norm.weight"])
                    else:
                        q_attn = weights[f"model.llm.attn_res.attn_queries.{phys_idx - 1}"].view(-1, lm.H) * lm.INV_SQRT_H
                        nw = (
                            torch.stack([weights[f"model.llm.layers.{phys_idx}.{nm}.weight"] for nm in ("q_input_norm", "k_input_norm", "v_input_norm")], dim=0)
                            if is_full
                            else weights[f"model.llm.layers.{phys_idx}.attn_norm.weight"]
                        )
                        x_qkv = _ref_attn_res(stack + [partial], q_attn, norm_w=nw)
                        x_q = x_qkv[0]
                else:
                    c = phys_idx - lm.CORE_START
                    q_attn = weights[f"model.llm.attn_res.attn_queries.{phys_idx - 1}"].view(-1, lm.H) * lm.INV_SQRT_H
                    dest_qkv = _ref_attn_res(stack + [partial], q_attn)  # [1 or 3, T, H]
                    q_cp = torch.stack([cross_q[2 * c], cross_kv[2 * c], cross_kv[2 * c + 1], cross_q[2 * c + 1]], dim=0)
                    msgs = _ref_attn_res(prev_blocks, q_cp)  # [4, T, H]
                    nw0 = weights[f"model.llm.layers.{phys_idx}.q_input_norm.weight"] if is_full else weights[f"model.llm.layers.{phys_idx}.attn_norm.weight"]
                    x_q = _ref_loop_controller(
                        dest_qkv[0],
                        msgs[0],
                        weights[f"model.llm.layers.{phys_idx}.controllers.attn.down_weight"],
                        weights[f"model.llm.layers.{phys_idx}.controllers.attn.forget_weight"],
                        weights[f"model.llm.layers.{phys_idx}.controllers.attn.gain_weight"],
                        nw0,
                    )

                attn_out = x_q  # matches stub_attn copy_bf16(x_qkv[0], attn_out)
                if first:
                    stack.append(partial.clone())
                    mlp_srcs = stack
                    mlp_last = attn_out
                    mlp_addend = None
                else:
                    mlp_srcs = stack
                    mlp_last = partial
                    mlp_addend = attn_out

                q_mlp = weights[f"model.llm.attn_res.mlp_queries.{phys_idx}"].view(1, lm.H) * lm.INV_SQRT_H
                if pass_idx == 0:
                    x_mlp = _ref_attn_res(mlp_srcs + [mlp_last], q_mlp, addend=mlp_addend, norm_w=weights[f"model.llm.layers.{phys_idx}.mlp_norm.weight"])[0]
                else:
                    dest_mlp = _ref_attn_res(mlp_srcs + [mlp_last], q_mlp, addend=mlp_addend)[0]
                    x_mlp = _ref_loop_controller(
                        dest_mlp,
                        msgs[3],
                        weights[f"model.llm.layers.{phys_idx}.controllers.mlp.down_weight"],
                        weights[f"model.llm.layers.{phys_idx}.controllers.mlp.forget_weight"],
                        weights[f"model.llm.layers.{phys_idx}.controllers.mlp.gain_weight"],
                        weights[f"model.llm.layers.{phys_idx}.mlp_norm.weight"],
                    )
                mlp_out = x_mlp  # matches stub_mlp copy_bf16(x_mlp, mlp_out)
                if first:
                    partial = (attn_out.float() + mlp_out.float()).to(torch.bfloat16)
                else:
                    partial = (partial.float() + attn_out.float() + mlp_out.float()).to(torch.bfloat16)
                if visit_idx == 5:
                    prelude_partial = partial.clone()

            final_h_ref = _ref_attn_res(
                stack + [partial],
                weights["model.llm.attn_res.output_query"].view(1, lm.H) * lm.INV_SQRT_H,
                norm_w=weights["model.llm.norm.weight"],
            )[0]
            aux_ref[48] = final_h_ref

            got_aux = views["aux_hidden"][:T].cpu()
            for idx, vid in enumerate(lm.AUX_CAPTURE_VISITS):
                if vid == 5:
                    torch.testing.assert_close(got_aux[:, idx, :], aux_ref[vid], atol=2e-2, rtol=1e-2)
                else:
                    diff = (got_aux[:, idx, :].float() - aux_ref[vid].float()).abs()
                    self.assertLess(diff.mean().item(), 5e-3, f"vid={vid} mean_abs too high")
                    torch.testing.assert_close(got_aux[:, idx, :], aux_ref[vid], atol=0.5, rtol=5e-2)
            self.assertLess((views["final_h"][:T].cpu().float() - final_h_ref.float()).abs().mean().item(), 5e-3)
            torch.testing.assert_close(views["final_h"][:T].cpu(), final_h_ref, atol=0.5, rtol=5e-2)

            logits_ref = 30.0 * torch.tanh((final_h_ref[-1:].float() @ weights["model.llm.unembed.weight"].float().T) / 30.0)
            logits_ref[:, lm.UNPADDED_VOCAB :] = float("-inf")
            got_logits = views["logits"][:1].cpu()
            self.assertTrue(torch.all(torch.isneginf(got_logits[:, lm.UNPADDED_VOCAB :])))
            self.assertLess(
                (got_logits[:, : lm.UNPADDED_VOCAB] - logits_ref[:, : lm.UNPADDED_VOCAB]).abs().mean().item(),
                1e-2,
            )
            torch.testing.assert_close(got_logits[:, : lm.UNPADDED_VOCAB], logits_ref[:, : lm.UNPADDED_VOCAB], atol=0.5, rtol=5e-2)
            self.assertEqual(int(views["next_token"][0].item()), int(logits_ref.argmax(dim=-1).item()))

    def test_cross_pass_4q_bitexact_parity(self):
        """Verify 4-stream kern_cross_pass_attn_res_4q_f32 is 100% bit-identical to 1-stream."""
        import torch

        torch.manual_seed(66)
        D = lm.H
        max_t = 64
        m = {
            "schema_version": 7,
            "model": "test-cp-4q",
            "vars": {"tokens": {"max": max_t}},
            "states": {},
            "buffers": {
                "prev": {"dtype": "bf16", "shape": [2, "tokens", D], "kind": "input"},
                "q": {"dtype": "f32", "shape": [4, D], "kind": "input"},
                "out_1q": {"dtype": "f32", "shape": [4, "tokens", D], "kind": "output"},
                "out_4q": {"dtype": "f32", "shape": [4, "tokens", D], "kind": "output"},
            },
            "modules": {"norm_attnres": self._mod("norm_attnres")},
            "ops": {
                "cp_1q": {
                    "params": ["in buffer<bf16>", "in buffer<f32>", "out buffer<f32>", "i64", "i64", "i32", "i32", "i32", "f32"],
                    "impl": {
                        "launches": [
                            {
                                "module": "norm_attnres",
                                "entry": "kern_cross_pass_attn_res_f32",
                                "params": ["in buffer<bf16>", "in buffer<f32>", "out buffer<f32>", "i64", "i64", "i32", "i32", "i32", "f32"],
                                "grid": ["tokens", 1, 1],
                                "block": [256, 1, 1],
                                "pdl": True,
                                "args": [{"param": j} for j in range(9)],
                            }
                        ]
                    },
                },
                "cp_4q": {
                    "params": ["in buffer<bf16>", "in buffer<f32>", "out buffer<f32>", "i64", "i64", "i32", "f32"],
                    "impl": {
                        "launches": [
                            {
                                "module": "norm_attnres",
                                "entry": "kern_cross_pass_attn_res_4q_f32",
                                "params": ["in buffer<bf16>", "in buffer<f32>", "out buffer<f32>", "i64", "i64", "i32", "f32"],
                                "grid": ["tokens", 4, 1],
                                "block": [256, 1, 1],
                                "pdl": True,
                                "args": [{"param": j} for j in range(7)],
                            }
                        ]
                    },
                },
            },
            "programs": {
                "run": {
                    "calls": [
                        {"op": "cp_1q", "args": [lm.b("prev"), lm.b("q"), lm.b("out_1q"), lm.i64(max_t * D), lm.i64(max_t * D), {"var": "tokens"}, lm.i32(D), lm.i32(4), lm.f32(1e-6)]},
                        {"op": "cp_4q", "args": [lm.b("prev"), lm.b("q"), lm.b("out_4q"), lm.i64(max_t * D), lm.i64(max_t * D), {"var": "tokens"}, lm.f32(1e-6)]},
                    ]
                }
            },
        }
        rt, views = self._run_manifest(m)
        prev = torch.randn(2, max_t, D, dtype=torch.bfloat16)
        q = torch.randn(4, D, dtype=torch.float32) * lm.INV_SQRT_H
        views["prev"].copy_(prev)
        views["q"].copy_(q)
        for T in (1, 8, 32, 64):
            with self.subTest(T=T):
                torch.cuda.synchronize()
                rt.run("run", {"tokens": T})
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(views["out_4q"][:, :T], views["out_1q"][:, :T]))


if __name__ == "__main__":
    unittest.main()
