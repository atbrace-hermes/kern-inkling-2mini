#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""DSpark MTP + Inkling2MarkovHead + SpecRound SM120 AOT builder and manifest helper."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TARGET_LAYER_IDS = (5, 23, 41, 48)
DSPARK_HIDDEN = 2304
DSPARK_INTER = 9216
DSPARK_Q_HEADS = 24
DSPARK_KV_HEADS = 8
DSPARK_HEAD_DIM = 128
DSPARK_LAYERS = 3
DSPARK_LOCAL_LAYERS = (0, 2)
DSPARK_SWA_WINDOW = 1024
DSPARK_BLOCK_SIZE = 8
DSPARK_NUM_SPEC = DSPARK_BLOCK_SIZE - 1
DSPARK_MARKOV_RANK = 256
DSPARK_RMS_EPS = 1e-6
DSPARK_ROPE_THETA = 1_000_000.0
VOCAB_PADDED = 201024
VOCAB_UNPADDED = 200058
SOFTCAP = 30.0
CONTEXT_LIMIT = 1048576 + 7

DSPARK_SWA_BYTES = 2 * DSPARK_SWA_WINDOW * (DSPARK_KV_HEADS * DSPARK_HEAD_DIM * 2)
DSPARK_FRNN_BYTES = DSPARK_HIDDEN * 2
DSPARK_STATE_BYTES_PER_SEQ = DSPARK_SWA_BYTES + DSPARK_FRNN_BYTES

DEFAULT_FLASHINFER_INCLUDE = Path(
    "/workspace/kern-qwen38/third_party/flashinfer/include"
)

BLOCK_SAMPLING = 1024
VEC_SAMPLING = 4


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


@dataclass(frozen=True)
class DsparkTargetLayout:
    """Layout parameters connecting M4 DSpark/SpecRound to M1/M3 target buffers and states."""

    page_size: int = 64
    page_columns: int = ceil_div(CONTEXT_LIMIT, 64)
    kv_state: str = "kv"
    kv_layer_offset_bytes: int = 8 * 2 * 8 * 64 * 128
    kv_page_stride_bytes: int = 9 * 2 * 8 * 64 * 128
    kv_slot_stride_bytes: int = 128
    kv_head_stride_bytes: int = 64 * 128
    kv_v_offset_bytes: int = 8 * 64 * 128
    kv_k_scale: float = 1.0
    kv_v_scale: float = 1.0
    dspark_k_scale: float = 1.0
    dspark_v_scale: float = 1.0
    dspark_swa_fp8: bool = True
    mask_token_id: int = 0
    num_kv_splits: int = 16
    target_embed_buf: str = "embed.weight"
    target_embed_norm_buf: str = "embed_norm.weight"
    target_lm_head_buf: str = "lm_head.weight"
    target_verify_logits_buf: str = "d.verify_logits"
    target_verify_logits_dtype: str = "f32"
    target_sconv_raw_buf: str = "spec.sconv_raw"
    target_swa_kv_buf: str = "spec.swa_kv"
    taps_buf: str = "d.taps"
    seq_state_name: str = "seq_state"
    seq_lines_buf: str = "seq_state.lines"
    swa_off_bytes: int = 1916928
    sconv_off_bytes: int = 0
    seq_state_stride_bytes: int = 90005376
    target_n_visits: int = 48
    target_n_swa_visits: int = 40
    target_sconv_dim: int = 6656
    swa_window: int = 1024
    swa_token_bytes: int = 2048
    max_seqs: int = 8
    valid_buf: str = "valid"


# ---------------------------------------------------------------------------
# SM120 AOT build helper (CPU-only safe)
# ---------------------------------------------------------------------------


def _load_py_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _check_ptx_sm120_safe(name: str, ptx: str) -> None:
    for bad in ("tcgen05", "utmastg", "wgmma.mma_async"):
        if bad in ptx:
            raise ValueError(f"{name}: forbidden SM90/SM100 instruction `{bad}` found in PTX")


def build_m4_kernels(
    root: Path | None = None,
    nvcc: str | None = None,
    flashinfer_include: Path | None = None,
    jobs: int = 4,
) -> dict:
    """Compile all M4 CUDA `.cu` and Triton `.py` kernels to `build/sm_120/{dspark_*,spec_*}`."""
    import triton
    from triton.backends.compiler import GPUTarget
    from triton.compiler import ASTSource

    repo = Path(root) if root is not None else Path(__file__).resolve().parents[1]
    out = repo / "build/sm_120"
    out.mkdir(parents=True, exist_ok=True)
    nvcc_bin = nvcc or shutil.which("nvcc")
    if not nvcc_bin:
        raise RuntimeError("nvcc not found in PATH")
    fi_inc = flashinfer_include or (repo / "third_party/flashinfer/include")
    if not fi_inc.exists():
        fi_inc = DEFAULT_FLASHINFER_INCLUDE
    if not fi_inc.exists():
        raise FileNotFoundError(f"FlashInfer include dir not found: {fi_inc}")

    cu_specs = [
        ("dspark_derive", "dspark_derive", []),
        ("spec_round", "spec_round", []),
        ("spec_markov_rows", "markov_rows", []),
        ("spec_argmax", "spec_argmax", []),
        ("spec_sampling", "sampling", [f"-I{fi_inc}"]),
        ("spec_sample_glue", "sample_glue", []),
        ("spec_state_commit", "spec_state_commit", []),
    ]

    spec_path = out / "spec_build.json"
    prev_spec_mods = json.loads(spec_path.read_text()).get("modules", {}) if spec_path.exists() else {}

    def compile_cu(item):
        name, src_stem, extra = item
        src = repo / "source" / f"{src_stem}.cu"
        cubin = out / f"{name}.cubin"
        src_sha = hashlib.sha256(src.read_bytes()).hexdigest()
        prev = prev_spec_mods.get(name)
        if prev and cubin.exists() and prev.get("source_sha256") == src_sha:
            cubin_sha = hashlib.sha256(cubin.read_bytes()).hexdigest()
            if cubin_sha == prev.get("sha256"):
                return name, prev
        cmd = [
            nvcc_bin,
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "-O3",
            "-lineinfo",
            "-cubin",
            "-arch=sm_120a",
            *extra,
            f"source/{src_stem}.cu",
            "-o",
            str(cubin),
        ]
        subprocess.run(cmd, cwd=repo, check=True)
        return name, {
            "source": cubin.name,
            "sha256": hashlib.sha256(cubin.read_bytes()).hexdigest(),
            "source_sha256": src_sha,
            "command": cmd,
        }

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        cu_built = dict(pool.map(compile_cu, cu_specs))

    # Host-only sampling metadata probe + cuobjdump resource inspection (CPU-only!)
    probe = out / "spec_sampling_metadata"
    subprocess.run(
        [
            nvcc_bin,
            "-std=c++17",
            "--expt-relaxed-constexpr",
            f"-I{fi_inc}",
            str(repo / "source/sampling_metadata.cu"),
            "-o",
            str(probe),
        ],
        check=True,
    )
    sampling_sizes = json.loads(subprocess.check_output([str(probe)], text=True))
    cuobjdump = str(Path(nvcc_bin).resolve().with_name("cuobjdump"))
    usage = subprocess.check_output([cuobjdump, "-res-usage", str(out / "spec_sampling.cubin")], text=True)
    entries = {}
    for entry, resources in re.findall(r"Function ([^:]+):\s*([^\n]+)", usage):
        tag = next(
            (
                n
                for n in (
                    "OnlineSoftmaxFusedKernel",
                    "RadixTopKRenormProbKernel_MultiCTA",
                    "TopPRenormProbKernel",
                    "MinPSamplingFromProbKernel",
                )
                if n in entry
            ),
            None,
        )
        if tag:
            if tag in entries:
                raise ValueError(f"Ambiguous sampling entry {tag}")
            m_sh = re.search(r"SHARED:(\d+)", resources)
            entries[tag] = {"entry": entry, "static_shared": int(m_sh.group(1)) if m_sh else 0}
    if len(entries) != 4:
        raise ValueError(f"Expected 4 sampling entries, got {entries}")

    spec_build = {
        "architecture": "sm_120a",
        "compiler": subprocess.check_output([nvcc_bin, "--version"], text=True),
        "modules": cu_built,
        "sampling": {"sizes": sampling_sizes, "entries": entries},
    }
    (out / "spec_build.json").write_text(json.dumps(spec_build, indent=2) + "\n")

    # Triton AOT kernels: GPUTarget("cuda", 120, 32) (warp_size=32, NOT 120!)
    dspark_py = _load_py_module(repo / "source/dspark_kernels.py")
    markov_py = _load_py_module(repo / "source/markov_head.py")

    triton_specs = [
        (
            "dspark_mix_rmsnorm",
            dspark_py._mix_rmsnorm_kernel,
            "*bf16,*bf16,*bf16,*bf16,i64,i64,i64,4,2304,EPS,4096",
            8,
            {"EPS": DSPARK_RMS_EPS},
        ),
        (
            "dspark_qk_norm_rope",
            dspark_py._qk_norm_rope_kernel,
            "*bf16,*bf16,*bf16,*fp32,*i64,i64,i64,24,128,EPS,1",
            1,
            {"EPS": DSPARK_RMS_EPS},
        ),
        (
            "dspark_k_norm_rope_ctx",
            dspark_py._qk_norm_rope_kernel,
            "*bf16,*bf16,*bf16,*fp32,*i64,i64,i64,0,128,EPS,1",
            1,
            {"EPS": DSPARK_RMS_EPS},
        ),
        (
            "dspark_rope_only",
            dspark_py._qk_norm_rope_kernel,
            "*bf16,*bf16,*bf16,*fp32,*i64,i64,i64,0,128,EPS,0",
            1,
            {"EPS": 0.0},
        ),
        (
            "dspark_silu_mul",
            dspark_py._silu_and_mul_kernel,
            "*bf16,*bf16,i64,9216,1024",
            4,
            {},
        ),
        (
            "dspark_rms_norm",
            dspark_py._rms_norm_residual_kernel,
            "*bf16,*bf16,*bf16,*bf16,2304,EPS,0,4096",
            8,
            {"EPS": DSPARK_RMS_EPS},
        ),
        (
            "dspark_add_rms_norm",
            dspark_py._rms_norm_residual_kernel,
            "*bf16,*bf16,*bf16,*bf16,2304,EPS,1,4096",
            8,
            {"EPS": DSPARK_RMS_EPS},
        ),
        (
            "dspark_ctx_store_ring_prefill",
            dspark_py._dspark_ctx_store_ring_kernel,
            "*bf16,*i64,*i32,*i32,*i32,*fp8e4nv,fp32,fp32,i64,i64,i32,1,0,1024,8,1024",
            4,
            {},
        ),
        (
            "dspark_ctx_store_ring_decode",
            dspark_py._dspark_ctx_store_ring_kernel,
            "*bf16,*i64,*i32,*i32,*i32,*fp8e4nv,fp32,fp32,i64,i64,i32,0,0,1024,8,1024",
            4,
            {},
        ),
        (
            "dspark_ctx_store_ring_commit",
            dspark_py._dspark_ctx_store_ring_kernel,
            "*bf16,*i64,*i32,*i32,*i32,*fp8e4nv,fp32,fp32,i64,i64,i32,0,1,1024,8,1024",
            4,
            {},
        ),
        (
            "dspark_attn_ring",
            dspark_py._dspark_attn_ring_kernel,
            "*bf16,*bf16,*bf16,*fp8e4nv,*i32,*i32,*bf16,fp32,fp32,fp32,i64,i64,i64,i64,i64,24,8,128,8,1024,1032,32,64",
            4,
            {},
        ),
        (
            "dspark_attn_combine_16",
            dspark_py._dspark_attn_combine_kernel,
            "*fp32,*fp32,*bf16,i32,24,128,16,16,128",
            2,
            {},
        ),
        (
            "dspark_skip_anchor",
            markov_py._skip_anchor_reorder_kernel,
            "*bf16,*bf16,2304,8,1024",
            4,
            {},
        ),
        (
            "dspark_markov_ctx_concat",
            markov_py._markov_ctx_concat_kernel,
            "*bf16,*bf16,*bf16,2304,7,1024",
            4,
            {},
        ),
        (
            "dspark_markov_add_bias",
            markov_py._markov_add_bias_kernel,
            "*fp32,*fp32,768,1024",
            4,
            {},
        ),
        (
            "dspark_markov_state_concat",
            markov_py._markov_state_concat_kernel,
            "*bf16,*i64,*bf16,*bf16,i32,i32,256",
            2,
            {},
        ),
        (
            "dspark_markov_gate_add",
            markov_py._markov_gate_add_kernel,
            "*fp32,*fp32,*bf16,*bf16,*bf16,i32,7,256",
            2,
            {},
        ),
        (
            "dspark_add_softcap_mask",
            markov_py._add_softcap_mask_kernel,
            "*bf16,*bf16,*bf16,*bf16,i32,fp32,7,201024,200058,1024",
            4,
            {},
        ),
    ]
    paged_div16 = {(i,): [["tt.divisibility", 16]] for i in (0, 1, 2, 3, 6, 11, 12, 13, 15, 16, 17, 18, 19, 20)}
    for page_sz in (64, 128):
        suf = "" if page_sz == 64 else f"_p{page_sz}"
        triton_specs += [
            (
                f"dspark_ctx_store_paged_direct{suf}",
                dspark_py._dspark_ctx_store_paged_kernel,
                f"*bf16,*i32,*i32,*i32,*fp8e4nv,fp32,fp32,i64,i64,i64,i64,i64,i64,i32,{page_sz},0,8,1024",
                4,
                {},
            ),
            (
                f"dspark_ctx_store_paged_commit{suf}",
                dspark_py._dspark_ctx_store_paged_kernel,
                f"*bf16,*i32,*i32,*i32,*fp8e4nv,fp32,fp32,i64,i64,i64,i64,i64,i64,i32,{page_sz},1,8,1024",
                4,
                {},
            ),
            (
                f"dspark_attn_paged_direct{suf}",
                dspark_py._dspark_attn_paged_kernel,
                f"*bf16,*bf16,*bf16,*fp8e4nv,*i32,*i32,*bf16,*fp32,fp32,fp32,fp32,i64,i64,i64,i64,i64,i64,i64,i64,i64,i64,i32,i32,24,8,128,8,{page_sz},0,32,32",
                4,
                {"__opts__": {"num_stages": 3, "enable_fp_fusion": True}, "__attrs__": paged_div16},
            ),
            (
                f"dspark_attn_paged_partial{suf}",
                dspark_py._dspark_attn_paged_kernel,
                f"*bf16,*bf16,*bf16,*fp8e4nv,*i32,*i32,*fp32,*fp32,fp32,fp32,fp32,i64,i64,i64,i64,i64,i64,i64,i64,i64,i64,i32,i32,24,8,128,8,{page_sz},1,32,32",
                4,
                {"__opts__": {"num_stages": 3, "enable_fp_fusion": True}, "__attrs__": paged_div16},
            ),
        ]

    dsl_path = out / "dspark_dsl.json"
    prev_dsl_mods = json.loads(dsl_path.read_text()).get("modules", {}) if dsl_path.exists() else {}

    def _strip_ptx_debug(text: str) -> str:
        out_lines = []
        for ln in text.splitlines():
            s = ln.lstrip()
            if s.startswith((".file", ".loc", "//")):
                continue
            if s.startswith(".section\t.debug") or s.startswith(".section .debug"):
                break
            out_lines.append(ln)
        return "\n".join(out_lines)

    dsl_records = {}
    for name, kernel, sig_str, warps, extra_const in triton_specs:
        extra_const = dict(extra_const)
        extra_opts = extra_const.pop("__opts__", {})
        extra_attrs = extra_const.pop("__attrs__", {})
        sig, constants = {}, {}
        for arg, typ in zip(kernel.arg_names, sig_str.split(","), strict=True):
            if typ.isdecimal():
                constants[arg] = int(typ)
            elif arg in extra_const:
                constants[arg] = extra_const[arg]
            else:
                sig[arg] = typ
        constants.update(extra_const)
        compile_opts = {"num_warps": warps, "num_stages": 2, "enable_fp_fusion": False}
        compile_opts.update(extra_opts)
        compiled = triton.compile(
            ASTSource(kernel, signature=sig, constexprs=constants, attrs=extra_attrs),
            target=GPUTarget("cuda", 120, 32),
            options=compile_opts,
        )
        if (
            compiled.metadata.global_scratch_size
            or compiled.metadata.profile_scratch_size
            or compiled.metadata.num_ctas != 1
        ):
            raise ValueError(f"{name}: unsupported scratch or cluster requirements")
        ptx = compiled.asm["ptx"]
        _check_ptx_sm120_safe(name, ptx)
        cubin_path = out / f"{name}.cubin"
        ptx_path = out / f"{name}.ptx"
        prev = prev_dsl_mods.get(name)
        if (
            prev
            and cubin_path.exists()
            and ptx_path.exists()
            and prev.get("entry") == compiled.name
            and prev.get("shared_mem") == compiled.metadata.shared
            and prev.get("block") == [32 * warps, 1, 1]
            and prev.get("signature") == list(sig.values())
            and _strip_ptx_debug(ptx_path.read_text()) == _strip_ptx_debug(ptx)
        ):
            dsl_records[name] = prev
            continue
        cubin_path.write_bytes(compiled.asm["cubin"])
        ptx_path.write_text(ptx)
        dsl_records[name] = {
            "source": cubin_path.name,
            "sha256": hashlib.sha256(cubin_path.read_bytes()).hexdigest(),
            "entry": compiled.name,
            "shared_mem": compiled.metadata.shared,
            "block": [32 * warps, 1, 1],
            "signature": list(sig.values()),
            "arguments": list(sig),
            "abi_tail": ["i64", "i64"],
        }

    dsl_report = {"triton": triton.__version__, "target": "cuda:120:32", "modules": dsl_records}
    (out / "dspark_dsl.json").write_text(json.dumps(dsl_report, indent=2) + "\n")
    return {"spec_build": spec_build, "dspark_dsl": dsl_report}


# ---------------------------------------------------------------------------
# FlashInfer sampling op helper
# ---------------------------------------------------------------------------


def sampling_fills() -> dict:
    return {
        "temperature": {"dtype": "f32", "shape": ["seqs"], "kind": "input", "fill": "temperature"},
        "top_k": {"dtype": "i32", "shape": ["seqs"], "kind": "input", "fill": "top_k"},
        "top_p": {"dtype": "f32", "shape": ["seqs"], "kind": "input", "fill": "top_p"},
        "min_p": {"dtype": "f32", "shape": ["seqs"], "kind": "input", "fill": "min_p"},
        "seed": {"dtype": "i64", "shape": ["seqs"], "kind": "input", "fill": "seed"},
    }


def sampling_radix(vocab: int, metadata: dict, target: dict) -> tuple[int, int, int, int]:
    fixed = metadata["sizes"]["radix_fixed"]
    static = metadata["entries"]["RadixTopKRenormProbKernel_MultiCTA"]["static_shared"]
    limit = (target["shared_mem"] - static - fixed) // (4 * VEC_SAMPLING) * VEC_SAMPLING
    if vocab < 1 or vocab % VEC_SAMPLING or limit < VEC_SAMPLING * BLOCK_SAMPLING:
        raise ValueError("Invalid sampling vocab or shared memory")
    ctas = ceil_div(vocab, limit)
    if ctas > target["sms"]:
        raise ValueError(f"Sampling requires {ctas} resident CTAs, target has {target['sms']} SMs")
    chunk = ceil_div(ceil_div(vocab, ctas), VEC_SAMPLING) * VEC_SAMPLING
    return ctas, chunk, target["sms"] // ctas, fixed + chunk * 4


def sampling_op(
    vocab: int,
    seqs_max: int,
    metadata: dict,
    target: dict,
    logits_dtype: str = "bf16",
    axis_var: str = "seqs",
) -> dict:
    """Build FlashInfer sampling op for BF16 or FP32 logits."""
    ctas, chunk, groups, radix_smem = sampling_radix(vocab, metadata, target)
    chain, glue, argmax_mod = "spec_sampling", "spec_sample_glue", "spec_argmax"
    sizes = metadata["sizes"]
    groups = min(groups, seqs_max)

    def launch(mod, name, params, block, grid, args, smem=None):
        entry = metadata["entries"][name]["entry"] if mod == chain else name
        s = {
            "entry": entry,
            "params": params,
            "block": [block, 1, 1],
            "grid": grid,
            "args": args,
            "module": mod,
        }
        if smem is not None:
            s["shared_mem"] = smem
        return s

    p = lambda idx: {"param": idx}
    scr = lambda n: {"scratch": n}
    i32 = lambda v: {"i32": v}
    i64 = lambda v: {"i64": v}
    f32 = lambda v: {"f32": v}

    S = axis_var
    logits, seq_lens, temperature, top_k, top_p, min_p, seed, tokens, seqs = (p(i) for i in range(9))
    probs = scr("probs")
    in_logit_type = f"in buffer<{logits_dtype}>"
    partial_entry = "kern_argmax_partial_f32" if logits_dtype == "f32" else "kern_argmax_partial_bf16"
    cast_entry = "kern_sample_copy_f32" if logits_dtype == "f32" else "kern_sample_cast_bf16"

    return {
        "params": [
            in_logit_type,
            "in buffer<i32>",
            "in buffer<f32>",
            "in buffer<i32>",
            "in buffer<f32>",
            "in buffer<f32>",
            "in buffer<i64>",
            "out buffer<i64>",
            "i32",
        ],
        "impl": {
            "scratch": {
                "greedy": {"dtype": "i64", "shape": [S]},
                "max": {"dtype": "f32", "shape": [S, 64]},
                "idx": {"dtype": "i32", "shape": [S, 64]},
                "probs": {"dtype": "f32", "shape": [S, vocab]},
                "chain_temperature": {"dtype": "f32", "shape": [S]},
                "chain_top_k": {"dtype": "i32", "shape": [S]},
                "offset": {"dtype": "u64", "shape": [S]},
                "states": {"dtype": "u32", "shape": [groups, sizes["state_words"]]},
                "drawn": {"dtype": "i32", "shape": [S]},
                "valid": {"dtype": "u8", "shape": [S]},
            },
            "launches": [
                launch(
                    argmax_mod,
                    partial_entry,
                    [in_logit_type, "out buffer<f32>", "out buffer<i32>", "i32"],
                    1024,
                    [S, 64, 1],
                    [logits, scr("max"), scr("idx"), i32(vocab)],
                ),
                launch(
                    argmax_mod,
                    "kern_argmax_final_i64",
                    ["in buffer<f32>", "in buffer<i32>", "out buffer<i64>", "i32"],
                    64,
                    [S, 1, 1],
                    [scr("max"), scr("idx"), scr("greedy"), i32(64)],
                ),
                launch(
                    glue,
                    cast_entry,
                    [in_logit_type, "out buffer<f32>", "i32"],
                    BLOCK_SAMPLING,
                    [S, ceil_div(vocab, BLOCK_SAMPLING), 1],
                    [logits, probs, i32(vocab)],
                ),
                launch(
                    glue,
                    "kern_sample_prep",
                    [
                        "in buffer<f32>",
                        "in buffer<i32>",
                        "in buffer<i32>",
                        "out buffer<f32>",
                        "out buffer<i32>",
                        "out buffer<u64>",
                        "out buffer<u32>",
                        "i32",
                        "i32",
                        "i32",
                    ],
                    BLOCK_SAMPLING,
                    [1, 1, 1],
                    [
                        temperature,
                        top_k,
                        seq_lens,
                        scr("chain_temperature"),
                        scr("chain_top_k"),
                        scr("offset"),
                        scr("states"),
                        seqs,
                        i32(vocab),
                        i32(groups * sizes["state_words"]),
                    ],
                ),
                launch(
                    chain,
                    "OnlineSoftmaxFusedKernel",
                    ["inout buffer<f32>", "inout buffer<f32>", "in buffer<f32>", "f32", "i32"],
                    BLOCK_SAMPLING,
                    [S, 1, 1],
                    [probs, probs, scr("chain_temperature"), f32(1.0), i32(vocab)],
                    sizes["softmax"],
                ),
                launch(
                    chain,
                    "RadixTopKRenormProbKernel_MultiCTA",
                    [
                        "inout buffer<f32>",
                        "inout buffer<f32>",
                        "in buffer<i32>",
                        "i32",
                        "i32",
                        "i32",
                        "inout buffer<u32>",
                        "i32",
                        "i32",
                    ],
                    BLOCK_SAMPLING,
                    [{"mul": [{"clamp": [S, 0, groups]}, ctas]}, 1, 1],
                    [
                        probs,
                        probs,
                        scr("chain_top_k"),
                        i32(0),
                        i32(vocab),
                        seqs,
                        scr("states"),
                        i32(chunk),
                        i32(ctas),
                    ],
                    radix_smem,
                ),
                launch(
                    chain,
                    "TopPRenormProbKernel",
                    ["inout buffer<f32>", "inout buffer<f32>", "in buffer<f32>", "f32", "i32"],
                    BLOCK_SAMPLING,
                    [S, 1, 1],
                    [probs, probs, top_p, f32(0.0), i32(vocab)],
                    sizes["topp"],
                ),
                launch(
                    chain,
                    "MinPSamplingFromProbKernel",
                    [
                        "inout buffer<f32>",
                        "in buffer<f32>",
                        "out buffer<i32>",
                        "out buffer<u8>",
                        "i64",
                        "f32",
                        "i32",
                        "in buffer<i64>",
                        "i64",
                        "in buffer<u64>",
                        "i64",
                        "i32",
                        "i32",
                    ],
                    BLOCK_SAMPLING,
                    [S, 1, 1],
                    [
                        probs,
                        min_p,
                        scr("drawn"),
                        scr("valid"),
                        i64(0),
                        f32(0.0),
                        i32(vocab),
                        seed,
                        i64(0),
                        scr("offset"),
                        i64(0),
                        i32(1),
                        i32(1),
                    ],
                    sizes["minp"],
                ),
                launch(
                    glue,
                    "kern_sample_select",
                    [
                        "in buffer<f32>",
                        "in buffer<i32>",
                        "in buffer<i64>",
                        "in buffer<u8>",
                        "out buffer<i64>",
                        "i32",
                    ],
                    BLOCK_SAMPLING,
                    [1, 1, 1],
                    [temperature, scr("drawn"), scr("greedy"), scr("valid"), tokens, seqs],
                ),
            ],
        },
    }


# ---------------------------------------------------------------------------
# Manifest registration & emitters
# ---------------------------------------------------------------------------

_b = lambda n, o=0: {"buf": n, **({"offset": o} if o else {})}
_st = lambda n, o=0: {"state": n, **({"offset": o} if o else {})}
_i32 = lambda v: {"i32": v}
_i64 = lambda v: {"i64": v}
_f32 = lambda v: {"f32": v}
_ceil = lambda x, d: {"ceil_div": [x, d]}
_mul = lambda x, y: {"mul": [x, y]}
_var = lambda n: {"var": n}
_expr = lambda x: {"expr": x}


def register_modules(
    m: dict,
    kernels_dir: Path,
    layout: DsparkTargetLayout | None = None,
) -> dict[str, dict]:
    """Load `dspark_dsl.json` and `spec_build.json` from `kernels_dir`, register modules in `m['modules']`."""
    modules = m.setdefault("modules", {})
    dsl = json.loads((kernels_dir / "dspark_dsl.json").read_text())
    spec = json.loads((kernels_dir / "spec_build.json").read_text())
    if layout is None:
        keep_dsl = set(dsl["modules"])
    else:
        suf = "" if layout.page_size == 64 else f"_p{layout.page_size}"
        keep_dsl = {
            "dspark_mix_rmsnorm",
            "dspark_qk_norm_rope",
            "dspark_k_norm_rope_ctx",
            "dspark_silu_mul",
            "dspark_rms_norm",
            "dspark_add_rms_norm",
            "dspark_ctx_store_ring_prefill",
            "dspark_ctx_store_ring_decode",
            "dspark_ctx_store_ring_commit",
            f"dspark_ctx_store_paged_direct{suf}",
            f"dspark_ctx_store_paged_commit{suf}",
            "dspark_attn_ring",
            f"dspark_attn_paged_partial{suf}" if layout.num_kv_splits > 1 else f"dspark_attn_paged_direct{suf}",
            "dspark_skip_anchor",
            "dspark_markov_ctx_concat",
            "dspark_markov_add_bias",
            "dspark_markov_state_concat",
            "dspark_markov_gate_add",
            "dspark_add_softcap_mask",
        }
        if layout.num_kv_splits > 1:
            keep_dsl.add("dspark_attn_combine_16")
    for name, info in dsl["modules"].items():
        if name in keep_dsl:
            modules[name] = {"source": info["source"], "sha256": info["sha256"]}
    for name, info in spec["modules"].items():
        modules[name] = {"source": info["source"], "sha256": info["sha256"]}
    return {"dsl": dsl, "spec": spec}


def _triton_op(ops: dict, name: str, record: dict, params: list[str], grid: list, out_mod_name: str | None = None):
    ops[name] = {
        "params": params,
        "impl": {
            "launches": [
                {
                    "module": out_mod_name or name,
                    "entry": record["entry"],
                    "params": params + record["abi_tail"],
                    "grid": grid,
                    "block": record["block"],
                    "shared_mem": record["shared_mem"],
                    "args": [{"param": j} for j in range(len(params))] + [_i64(0), _i64(0)],
                }
            ]
        },
    }


def _cuda_op(ops: dict, name: str, mod: str, entry: str, params: list[str], grid: list, block: int | list[int] = 256):
    blk = [block, 1, 1] if isinstance(block, int) else block
    ops[name] = {
        "params": params,
        "impl": {
            "launches": [
                {
                    "module": mod,
                    "entry": entry,
                    "params": params,
                    "grid": grid,
                    "block": blk,
                    "args": [{"param": j} for j in range(len(params))],
                }
            ]
        },
    }


def _gemm_op(ops: dict, name: str, entry: str, c_dtype: str = "bf16"):
    ops[name] = {
        "params": [
            "in buffer<bf16>",
            "in buffer<bf16>",
            f"{'inout' if entry.endswith('_acc') else 'out'} buffer<{c_dtype}>",
            "i32",
            "i32",
            "i32",
        ],
        "impl": {
            "launches": [
                {
                    "entry": entry,
                    "params": [
                        "in buffer<bf16>",
                        "in buffer<bf16>",
                        f"{'inout' if entry.endswith('_acc') else 'out'} buffer<{c_dtype}>",
                        "i32",
                        "i32",
                        "i32",
                    ],
                    "args": [{"param": j} for j in range(6)],
                }
            ]
        },
    }


def register_ops(
    m: dict,
    kernels_dir: Path,
    layout: DsparkTargetLayout,
    sampling_target: dict,
    *,
    register_target_commit_ops: bool = True,
) -> dict:
    """Register all DSpark/Markov/SpecRound/Sampling states, buffers, and ops."""
    loaded = register_modules(m, kernels_dir, layout)
    dsl = loaded["dsl"]["modules"]
    spec = loaded["spec"]
    buffers = m.setdefault("buffers", {})
    ops = m.setdefault("ops", {})
    states = m.setdefault("states", {})

    states.setdefault("dspark_state", {"bytes_per_seq": DSPARK_STATE_BYTES_PER_SEQ})

    def buf(name, dtype, shape, kind="workspace", **kw):
        buffers[name] = {"dtype": dtype, "shape": shape, "kind": kind, **kw}
        return _b(name)

    def dweight(name, shape, tensor=None):
        return buf("d." + name, "bf16", shape, "weight", bind=[{"tensor": tensor or name}])

    def dsource(name, shape, tensor=None):
        return buf("d." + name, "bf16", shape, "source", bind=[{"tensor": tensor or name}])

    # Per-sequence line table into dspark_state
    buf(
        "d.lines",
        "i32",
        [1, "seqs"],
        "input",
        domain={"index_into": "dspark_state", "stride": DSPARK_STATE_BYTES_PER_SEQ},
    )

    # MTP checkpoint sources (used only inside derive) & weights
    dsource("fuse_weight", [4, 4], "fuse_weight")
    dsource("hidden_out.weight", [DSPARK_HIDDEN, DSPARK_MARKOV_RANK], "markov_head.hidden_out.weight")
    dsource("hidden_out.bias", [DSPARK_HIDDEN], "markov_head.hidden_out.bias")
    dsource("rnn_proj.weight", [3 * DSPARK_MARKOV_RANK, 2 * DSPARK_MARKOV_RANK + 2 * DSPARK_HIDDEN], "markov_head.rnn_proj.weight")
    dsource("rnn_proj.bias", [3 * DSPARK_MARKOV_RANK], "markov_head.rnn_proj.bias")

    dweight("mask_embed", [DSPARK_HIDDEN], "mask_embed")
    dweight("norm.weight", [DSPARK_HIDDEN], "norm.weight")
    dweight("markov_head.dflare_norm.weight", [DSPARK_HIDDEN], "markov_head.dflare_norm.weight")
    dweight("markov_head.markov_w1.weight", [VOCAB_PADDED, DSPARK_MARKOV_RANK], "markov_head.markov_w1.weight")

    for l in range(DSPARK_LAYERS):
        p = f"layers.{l}."
        dweight(p + "dflare_norm.weight", [DSPARK_HIDDEN])
        dweight(p + "input_layernorm.weight", [DSPARK_HIDDEN])
        dweight(p + "post_attention_layernorm.weight", [DSPARK_HIDDEN])
        dweight(p + "self_attn.q_norm.weight", [DSPARK_HEAD_DIM])
        dweight(p + "self_attn.k_norm.weight", [DSPARK_HEAD_DIM])
        dweight(p + "self_attn.k_norm_ctx.weight", [DSPARK_HEAD_DIM])
        buf(
            f"d.{p}qkv.weight",
            "bf16",
            [5120, DSPARK_HIDDEN],
            "weight",
            bind=[
                {"tensor": f"{p}self_attn.q_proj.weight"},
                {"tensor": f"{p}self_attn.k_proj.weight"},
                {"tensor": f"{p}self_attn.v_proj.weight"},
            ],
        )
        buf(
            f"d.{p}kv_ctx.weight",
            "bf16",
            [2048, DSPARK_HIDDEN],
            "weight",
            bind=[
                {"tensor": f"{p}self_attn.k_proj_ctx.weight"},
                {"tensor": f"{p}self_attn.v_proj_ctx.weight"},
            ],
        )
        dweight(p + "self_attn.o_proj.weight", [DSPARK_HIDDEN, 3072])
        buf(
            f"d.{p}gate_up.weight",
            "bf16",
            [2 * DSPARK_INTER, DSPARK_HIDDEN],
            "weight",
            bind=[
                {"tensor": f"{p}mlp.gate_proj.weight"},
                {"tensor": f"{p}mlp.up_proj.weight"},
            ],
        )
        dweight(p + "mlp.down_proj.weight", [DSPARK_HIDDEN, DSPARK_INTER])

    # Derived carry buffers
    buf("d.fold_weight", "bf16", [VOCAB_PADDED, DSPARK_MARKOV_RANK], "carry")
    buf("d.fold_bias", "bf16", [VOCAB_PADDED], "carry")
    buf("d.fuse_alpha", "bf16", [4, 4], "carry")
    buf("d.w_state", "bf16", [3 * DSPARK_MARKOV_RANK, 2 * DSPARK_MARKOV_RANK], "carry")
    buf("d.w_ctx", "bf16", [3 * DSPARK_MARKOV_RANK, 2 * DSPARK_HIDDEN], "carry")
    buf("d.rnn_bias", "f32", [3 * DSPARK_MARKOV_RANK], "carry")
    buf("d.inv_freq", "f32", [DSPARK_HEAD_DIM // 2], "carry")

    # Round inputs & outputs
    buf("d.anchor", "i64", ["seqs"], "input", fill="token", domain={"min": 0, "max": VOCAB_UNPADDED - 1})
    buf("verify_tokens", "i64", ["seqs", DSPARK_BLOCK_SIZE], "output", fill="tokens", domain={"min": 0, "max": VOCAB_UNPADDED - 1})
    buf("accepted", "i32", ["seqs"], "output", fill="count", domain={"min": 1, "max": DSPARK_BLOCK_SIZE})

    # Workspaces
    if layout.taps_buf not in buffers:
        buf(layout.taps_buf, "bf16", ["tokens", 4, DSPARK_HIDDEN])
    buf("d.last_taps", "bf16", ["seqs", 4, DSPARK_HIDDEN])
    buf("d.ctx", "bf16", ["tokens", DSPARK_HIDDEN])
    buf("d.ctx_kv", "bf16", ["tokens", 2048])
    buf("d.f_rnn", "bf16", ["seqs", DSPARK_HIDDEN])
    buf("d.h", "bf16", ["tokens", DSPARK_HIDDEN])
    buf("d.x", "bf16", ["tokens", DSPARK_HIDDEN])
    buf("d.branch", "bf16", ["tokens", DSPARK_HIDDEN])
    buf("d.qkv", "bf16", ["tokens", 5120])
    buf("d.attn", "bf16", ["tokens", 3072])
    if layout.num_kv_splits > 1:
        buf("d.attn_o_part", "f32", [layout.num_kv_splits, "tokens", DSPARK_Q_HEADS, DSPARK_HEAD_DIM])
        buf("d.attn_lse_part", "f32", [layout.num_kv_splits, "tokens", DSPARK_Q_HEADS])
    else:
        buf("d.attn_lse_part", "f32", [1])
    buf("d.gate_up", "bf16", ["tokens", 2 * DSPARK_INTER])
    buf("d.act", "bf16", ["tokens", DSPARK_INTER])
    buf("d.step_hidden", "bf16", ["seqs", DSPARK_NUM_SPEC, DSPARK_HIDDEN])
    buf("d.base_logits", "bf16", ["seqs", DSPARK_NUM_SPEC, VOCAB_PADDED])
    buf("d.markov_ctx_in", "bf16", ["seqs", DSPARK_NUM_SPEC, 2 * DSPARK_HIDDEN])
    buf("d.markov_ctx", "f32", ["seqs", DSPARK_NUM_SPEC, 3 * DSPARK_MARKOV_RANK])
    buf("d.markov_state", "bf16", ["seqs", DSPARK_MARKOV_RANK])
    buf("d.markov_in", "bf16", ["seqs", 2 * DSPARK_MARKOV_RANK])
    buf("d.markov_state_part", "f32", ["seqs", 3 * DSPARK_MARKOV_RANK])
    buf("d.markov_latent", "bf16", ["seqs", DSPARK_MARKOV_RANK])
    buf("d.step_bias", "bf16", ["seqs", VOCAB_PADDED])
    buf("d.step_logits", "bf16", ["seqs", VOCAB_PADDED])
    buf("d.argmax_pmax", "f32", ["seqs", 64])
    buf("d.argmax_pidx", "i32", ["seqs", 64])
    buf("d.draft", "i64", ["seqs", DSPARK_NUM_SPEC])
    buf("d.verify_ids", "i64", ["tokens"])
    for n, dt in (
        ("temperature", "f32"),
        ("top_k", "i32"),
        ("top_p", "f32"),
        ("min_p", "f32"),
        ("seed", "i64"),
        ("lengths", "i32"),
    ):
        buf(f"d.sample.{n}", dt, ["tokens"])

    # Builtin GEMM ops
    _gemm_op(ops, "dspark_gemm_bf16", "extern:cublaslt_bf16_tn", "bf16")
    _gemm_op(ops, "dspark_gemm_bf16_f32", "extern:cublas_bf16_tn_f32", "f32")

    # Derive ops (no vars in grid/args)
    _cuda_op(
        ops,
        "dspark_fold_fp64",
        "dspark_derive",
        "dspark_fold_fp64",
        ["in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32", "i32"],
        [ceil_div(VOCAB_PADDED, 32), ceil_div(DSPARK_MARKOV_RANK, 32), 1],
        [32, 8, 1],
    )
    _cuda_op(
        ops,
        "dspark_fold_bias_fp64",
        "dspark_derive",
        "dspark_fold_bias_fp64",
        ["in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32"],
        [ceil_div(VOCAB_PADDED, 8), 1, 1],
        256,
    )
    _cuda_op(
        ops,
        "dspark_prep_weights",
        "dspark_derive",
        "dspark_prep_weights",
        [
            "in buffer<bf16>",
            "in buffer<bf16>",
            "in buffer<bf16>",
            "out buffer<bf16>",
            "out buffer<bf16>",
            "out buffer<bf16>",
            "out buffer<f32>",
            "out buffer<f32>",
            "i32",
            "i32",
            "i32",
            "i32",
            "i32",
            "i32",
            "f32",
        ],
        [ceil_div(3 * DSPARK_MARKOV_RANK * (2 * DSPARK_MARKOV_RANK + 2 * DSPARK_HIDDEN), 256), 1, 1],
        256,
    )

    # State & capture CUDA ops
    _cuda_op(
        ops,
        "dspark_tap_capture",
        "spec_state_commit",
        "dspark_tap_capture",
        ["in buffer<bf16>", "inout buffer<bf16>", "i32", "i32", "i32"],
        ["tokens", 1, 1],
        256,
    )
    _cuda_op(
        ops,
        "dspark_tap_capture_0",
        "spec_state_commit",
        "dspark_tap_capture",
        ["in buffer<bf16>", "out buffer<bf16>", "i32", "i32", "i32"],
        ["tokens", 1, 1],
        256,
    )
    _cuda_op(
        ops,
        "dspark_embed_block",
        "spec_state_commit",
        "dspark_embed_block",
        [
            "in buffer<i64>",
            "in buffer<bf16>",
            "in buffer<bf16>",
            "in buffer<bf16>",
            "out buffer<bf16>",
            "i32",
            "i32",
            "f32",
        ],
        ["seqs", DSPARK_BLOCK_SIZE, 1],
        256,
    )
    _cuda_op(
        ops,
        "dspark_gather_last_tap",
        "spec_state_commit",
        "dspark_gather_last_tap",
        ["in buffer<bf16>", "in buffer<i32>", "out buffer<bf16>", "i32", "i32", "i32", "i32"],
        ["seqs", 1, 1],
        256,
    )
    _cuda_op(
        ops,
        "dspark_gather_last_tap_prefill",
        "spec_state_commit",
        "dspark_gather_last_tap",
        ["in buffer<bf16>", "in buffer<i32>", "out buffer<bf16>", "i32", "i32", "i32", "i32"],
        [1, 1, 1],
        256,
    )
    _cuda_op(
        ops,
        "dspark_load_frnn",
        "spec_state_commit",
        "dspark_load_frnn",
        ["in state", "in buffer<i32>", "out buffer<bf16>", "i64", "i64", "i32"],
        ["seqs", 1, 1],
        256,
    )
    _cuda_op(
        ops,
        "dspark_store_frnn",
        "spec_state_commit",
        "dspark_store_frnn",
        ["in buffer<bf16>", "inout state", "in buffer<i32>", "i64", "i64", "i32"],
        ["seqs", 1, 1],
        256,
    )
    _cuda_op(
        ops,
        "dspark_store_frnn_prefill",
        "spec_state_commit",
        "dspark_store_frnn",
        ["in buffer<bf16>", "inout state", "in buffer<i32>", "i64", "i64", "i32"],
        [1, 1, 1],
        256,
    )
    if layout.target_verify_logits_dtype == "f32":
        _cuda_op(
            ops,
            "dspark_verify_mask_vocab_f32",
            "spec_state_commit",
            "dspark_verify_mask_vocab_f32",
            ["inout buffer<f32>", "f32", "i32", "i32"],
            ["tokens", ceil_div(VOCAB_PADDED, 1024), 1],
            1024,
        )
    else:
        _cuda_op(
            ops,
            "dspark_verify_mask_vocab_bf16",
            "spec_state_commit",
            "dspark_verify_mask_vocab_bf16",
            ["inout buffer<bf16>", "f32", "i32", "i32"],
            ["tokens", ceil_div(VOCAB_PADDED, 1024), 1],
            1024,
        )
    _cuda_op(
        ops,
        "spec_sampling",
        "spec_state_commit",
        "spec_sampling",
        [
            "in buffer<f32>",
            "in buffer<i32>",
            "in buffer<f32>",
            "in buffer<f32>",
            "in buffer<i64>",
            "in buffer<i32>",
            "out buffer<f32>",
            "out buffer<i32>",
            "out buffer<f32>",
            "out buffer<f32>",
            "out buffer<i64>",
            "out buffer<i32>",
            "i32",
        ],
        ["seqs", 1, 1],
        32,
    )
    _cuda_op(
        ops,
        "spec_zeros_bf16",
        "spec_state_commit",
        "spec_zeros_bf16",
        ["out buffer<bf16>", "i32"],
        [_ceil(_mul("seqs", DSPARK_MARKOV_RANK), 256), 1, 1],
        256,
    )
    _cuda_op(
        ops,
        "spec_splice_verify",
        "spec_round",
        "kern_splice_verify",
        ["in buffer<i64>", "in buffer<i64>", "out buffer<i64>", "i32", "i32"],
        ["seqs", 1, 1],
        32,
    )
    _cuda_op(
        ops,
        "kern_spec_count",
        "spec_round",
        "kern_spec_count",
        ["in buffer<i64>", "in buffer<i64>", "out buffer<i32>", "i32", "i32"],
        ["seqs", 1, 1],
        32,
    )
    _cuda_op(
        ops,
        "dspark_markov_argmax_partial",
        "spec_markov_rows",
        "kern_argmax_rows_partial_bf16",
        ["in buffer<bf16>", "i64", "out buffer<f32>", "out buffer<i32>", "i32"],
        ["seqs", 64, 1],
        1024,
    )
    _cuda_op(
        ops,
        "dspark_markov_argmax_final",
        "spec_markov_rows",
        "kern_argmax_rows_final_i64",
        ["in buffer<f32>", "in buffer<i32>", "out buffer<i64>", "i32", "i32"],
        ["seqs", 1, 1],
        64,
    )

    if register_target_commit_ops:
        _cuda_op(
            ops,
            "spec_conv_commit",
            "spec_state_commit",
            "spec_conv_commit",
            [
                "in buffer<bf16>",
                "inout state",
                "in buffer<i32>",
                "in buffer<i32>",
                "i64",
                "i64",
                "i32",
                "i32",
                "i32",
            ],
            ["seqs", ceil_div(layout.target_sconv_dim, 256), layout.target_n_visits],
            256,
        )
        _cuda_op(
            ops,
            "spec_swa_commit",
            "spec_state_commit",
            "spec_swa_commit",
            [
                "in buffer<fp8e4m3>",
                "inout state",
                "in buffer<i32>",
                "in buffer<i32>",
                "in buffer<i32>",
                "i64",
                "i64",
                "i32",
                "i32",
                "i32",
                "i32",
            ],
            ["seqs", layout.target_n_swa_visits, DSPARK_BLOCK_SIZE],
            128,
        )

    # Triton ops
    _triton_op(
        ops,
        "dspark_mix_rmsnorm",
        dsl["dspark_mix_rmsnorm"],
        ["in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i64", "i64", "i64"],
        ["tokens", 1, 1],
    )
    _triton_op(
        ops,
        "dspark_mix_rmsnorm_seqs",
        dsl["dspark_mix_rmsnorm"],
        ["in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i64", "i64", "i64"],
        ["seqs", 1, 1],
        out_mod_name="dspark_mix_rmsnorm",
    )
    _triton_op(
        ops,
        "dspark_mix_rmsnorm_one",
        dsl["dspark_mix_rmsnorm"],
        ["in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i64", "i64", "i64"],
        [1, 1, 1],
        out_mod_name="dspark_mix_rmsnorm",
    )
    _triton_op(
        ops,
        "dspark_qk_norm_rope",
        dsl["dspark_qk_norm_rope"],
        [
            "inout buffer<bf16>",
            "in buffer<bf16>",
            "in buffer<bf16>",
            "in buffer<f32>",
            "in buffer<i64>",
            "i64",
            "i64",
        ],
        ["tokens", DSPARK_Q_HEADS + DSPARK_KV_HEADS, 1],
    )
    _triton_op(
        ops,
        "dspark_k_norm_rope_ctx",
        dsl["dspark_k_norm_rope_ctx"],
        [
            "inout buffer<bf16>",
            "in buffer<bf16>",
            "in buffer<bf16>",
            "in buffer<f32>",
            "in buffer<i64>",
            "i64",
            "i64",
        ],
        ["tokens", DSPARK_KV_HEADS, 1],
    )
    _triton_op(
        ops,
        "dspark_silu_mul",
        dsl["dspark_silu_mul"],
        ["in buffer<bf16>", "out buffer<bf16>", "i64"],
        ["tokens", ceil_div(DSPARK_INTER, 1024), 1],
    )
    _triton_op(
        ops,
        "dspark_rms_norm",
        dsl["dspark_rms_norm"],
        ["in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>"],
        ["tokens", 1, 1],
    )
    _triton_op(
        ops,
        "dspark_add_rms_norm",
        dsl["dspark_add_rms_norm"],
        ["in buffer<bf16>", "inout buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>"],
        ["tokens", 1, 1],
    )
    for tag in ("prefill", "decode", "commit"):
        mod_n = f"dspark_ctx_store_ring_{tag}"
        _triton_op(
            ops,
            mod_n,
            dsl[mod_n],
            [
                "in buffer<bf16>",
                "in buffer<i64>",
                "in buffer<i32>",
                "in buffer<i32>",
                "in buffer<i32>",
                "inout state",
                "f32",
                "f32",
                "i64",
                "i64",
                "i32",
            ],
            ["tokens", 4, 1],
        )
    suf = "" if layout.page_size == 64 else f"_p{layout.page_size}"
    for tag in ("direct", "commit"):
        mod_n = f"dspark_ctx_store_paged_{tag}{suf}"
        _triton_op(
            ops,
            f"dspark_ctx_store_paged_{tag}",
            dsl[mod_n],
            [
                "in buffer<bf16>",
                "in buffer<i32>",
                "in buffer<i32>",
                "in buffer<i32>",
                "inout state",
                "f32",
                "f32",
                "i64",
                "i64",
                "i64",
                "i64",
                "i64",
                "i64",
                "i32",
            ],
            ["tokens", 4, 1],
            out_mod_name=mod_n,
        )
    _triton_op(
        ops,
        "dspark_attn_ring",
        dsl["dspark_attn_ring"],
        [
            "in buffer<bf16>",
            "in buffer<bf16>",
            "in buffer<bf16>",
            "in state",
            "in buffer<i32>",
            "in buffer<i32>",
            "out buffer<bf16>",
            "f32",
            "f32",
            "f32",
            "i64",
            "i64",
            "i64",
            "i64",
            "i64",
        ],
        [DSPARK_KV_HEADS, "seqs", 1],
    )
    if layout.num_kv_splits > 1:
        mod_p = f"dspark_attn_paged_partial{suf}"
        _triton_op(
            ops,
            "dspark_attn_paged_partial",
            dsl[mod_p],
            [
                "in buffer<bf16>",
                "in buffer<bf16>",
                "in buffer<bf16>",
                "in state",
                "in buffer<i32>",
                "in buffer<i32>",
                "out buffer<f32>",
                "out buffer<f32>",
                "f32",
                "f32",
                "f32",
                "i64",
                "i64",
                "i64",
                "i64",
                "i64",
                "i64",
                "i64",
                "i64",
                "i64",
                "i64",
                "i32",
                "i32",
            ],
            [DSPARK_KV_HEADS * layout.num_kv_splits, "seqs", 1],
            out_mod_name=mod_p,
        )
        _triton_op(
            ops,
            "dspark_attn_combine_16",
            dsl["dspark_attn_combine_16"],
            ["in buffer<f32>", "in buffer<f32>", "out buffer<bf16>", "i32"],
            ["tokens", DSPARK_Q_HEADS, 1],
        )
    else:
        mod_d = f"dspark_attn_paged_direct{suf}"
        _triton_op(
            ops,
            "dspark_attn_paged_direct",
            dsl[mod_d],
            [
                "in buffer<bf16>",
                "in buffer<bf16>",
                "in buffer<bf16>",
                "in state",
                "in buffer<i32>",
                "in buffer<i32>",
                "out buffer<bf16>",
                "out buffer<f32>",
                "f32",
                "f32",
                "f32",
                "i64",
                "i64",
                "i64",
                "i64",
                "i64",
                "i64",
                "i64",
                "i64",
                "i64",
                "i64",
                "i32",
                "i32",
            ],
            [DSPARK_KV_HEADS, "seqs", 1],
            out_mod_name=mod_d,
        )

    _triton_op(
        ops,
        "dspark_skip_anchor",
        dsl["dspark_skip_anchor"],
        ["in buffer<bf16>", "out buffer<bf16>"],
        [_mul("seqs", DSPARK_NUM_SPEC), ceil_div(DSPARK_HIDDEN, 1024), 1],
    )
    _triton_op(
        ops,
        "dspark_markov_ctx_concat",
        dsl["dspark_markov_ctx_concat"],
        ["in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>"],
        [_mul("seqs", DSPARK_NUM_SPEC), ceil_div(DSPARK_HIDDEN, 1024), 1],
    )
    _triton_op(
        ops,
        "dspark_markov_add_bias",
        dsl["dspark_markov_add_bias"],
        ["inout buffer<f32>", "in buffer<f32>"],
        [_mul("seqs", DSPARK_NUM_SPEC), 1, 1],
    )
    _triton_op(
        ops,
        "dspark_markov_state_concat",
        dsl["dspark_markov_state_concat"],
        ["in buffer<bf16>", "in buffer<i64>", "in buffer<bf16>", "out buffer<bf16>", "i32", "i32"],
        ["seqs", 1, 1],
    )
    _triton_op(
        ops,
        "dspark_markov_gate_add",
        dsl["dspark_markov_gate_add"],
        [
            "in buffer<f32>",
            "in buffer<f32>",
            "in buffer<bf16>",
            "out buffer<bf16>",
            "out buffer<bf16>",
            "i32",
        ],
        ["seqs", 1, 1],
    )
    _triton_op(
        ops,
        "dspark_add_softcap_mask",
        dsl["dspark_add_softcap_mask"],
        ["in buffer<bf16>", "in buffer<bf16>", "in buffer<bf16>", "out buffer<bf16>", "i32", "f32"],
        [ceil_div(VOCAB_PADDED, 1024), "seqs", 1],
    )

    # Target verify sampling op
    rows_max = layout.max_seqs * DSPARK_BLOCK_SIZE
    v_op = sampling_op(
        VOCAB_PADDED,
        rows_max,
        spec["sampling"],
        sampling_target,
        logits_dtype=layout.target_verify_logits_dtype,
        axis_var="tokens",
    )
    ops["verify_sample"] = v_op
    return loaded


def _call(op: str, *args, label: str | None = None) -> dict:
    return {"op": op, "args": list(args), **({"label": label} if label else {})}


def emit_derive_calls(m: dict, layout: DsparkTargetLayout) -> list[dict]:
    """Emit calls for `derive` program: FP64 fold + fuse_alpha + split W_state/W_ctx + inv_freq."""
    return [
        _call(
            "dspark_fold_fp64",
            _b(layout.target_lm_head_buf),
            _b("d.hidden_out.weight"),
            _b("d.fold_weight"),
            _i32(VOCAB_PADDED),
            _i32(DSPARK_HIDDEN),
            _i32(DSPARK_MARKOV_RANK),
            label="dspark.derive.fold_weight",
        ),
        _call(
            "dspark_fold_bias_fp64",
            _b(layout.target_lm_head_buf),
            _b("d.hidden_out.bias"),
            _b("d.fold_bias"),
            _i32(VOCAB_PADDED),
            _i32(DSPARK_HIDDEN),
            label="dspark.derive.fold_bias",
        ),
        _call(
            "dspark_prep_weights",
            _b("d.fuse_weight"),
            _b("d.rnn_proj.weight"),
            _b("d.rnn_proj.bias"),
            _b("d.fuse_alpha"),
            _b("d.w_state"),
            _b("d.w_ctx"),
            _b("d.rnn_bias"),
            _b("d.inv_freq"),
            _i32(4),
            _i32(4),
            _i32(3 * DSPARK_MARKOV_RANK),
            _i32(2 * DSPARK_MARKOV_RANK),
            _i32(2 * DSPARK_HIDDEN),
            _i32(DSPARK_HEAD_DIM),
            _f32(DSPARK_ROPE_THETA),
            label="dspark.derive.prep_weights",
        ),
    ]


def _emit_context_store_calls(
    layout: DsparkTargetLayout,
    mode: str,
    taps_buf: str | None = None,
    lines_buf: str = "d.lines",
    valid_buf: str | None = None,
) -> list[dict]:
    """mode in {'prefill', 'decode', 'commit'}."""
    taps_name = taps_buf or layout.taps_buf
    valid_name = valid_buf or layout.valid_buf
    calls = []
    swa_idx = 0
    count_arg = _b("accepted") if mode == "commit" else _b("lengths")
    ring_op = f"dspark_ctx_store_ring_{mode}"
    paged_op = "dspark_ctx_store_paged_commit" if mode == "commit" else "dspark_ctx_store_paged_direct"
    for l in range(DSPARK_LAYERS):
        p = f"layers.{l}."
        calls += [
            _call(
                "dspark_mix_rmsnorm",
                _b(taps_name),
                _b("d.fuse_alpha", l * 4 * 2),
                _b(f"d.{p}dflare_norm.weight"),
                _b("d.ctx"),
                _i64(4 * DSPARK_HIDDEN),
                _i64(DSPARK_HIDDEN),
                _i64(1),
                label=f"dspark.ctx.{l}.mix_norm",
            ),
            _call(
                "dspark_gemm_bf16",
                _b("d.ctx"),
                _b(f"d.{p}kv_ctx.weight"),
                _b("d.ctx_kv"),
                _var("tokens"),
                _i32(2048),
                _i32(DSPARK_HIDDEN),
                label=f"dspark.ctx.{l}.kv_proj",
            ),
            _call(
                "dspark_k_norm_rope_ctx",
                _b("d.ctx_kv"),
                _b(f"d.{p}self_attn.k_norm_ctx.weight"),
                _b(f"d.{p}self_attn.k_norm_ctx.weight"),
                _b("d.inv_freq"),
                _b("positions"),
                _i64(2048),
                _i64(0),
                label=f"dspark.ctx.{l}.k_norm_rope",
            ),
        ]
        if l in DSPARK_LOCAL_LAYERS:
            calls.append(
                _call(
                    ring_op,
                    _b("d.ctx_kv"),
                    _b("positions"),
                    _b(lines_buf),
                    _b("lengths"),
                    count_arg,
                    _st("dspark_state", swa_idx * (DSPARK_SWA_WINDOW * 2048)),
                    _f32(layout.dspark_k_scale),
                    _f32(layout.dspark_v_scale),
                    _i64(DSPARK_STATE_BYTES_PER_SEQ),
                    _i64(2048),
                    _var("tokens"),
                    label=f"dspark.ctx.{l}.store_ring",
                )
            )
            swa_idx += 1
        else:
            calls.append(
                _call(
                    paged_op,
                    _b("d.ctx_kv"),
                    _b("slots"),
                    _b(valid_name),
                    count_arg,
                    _st(layout.kv_state),
                    _f32(layout.dspark_k_scale),
                    _f32(layout.dspark_v_scale),
                    _i64(layout.kv_layer_offset_bytes),
                    _i64(layout.kv_page_stride_bytes),
                    _i64(layout.kv_slot_stride_bytes),
                    _i64(layout.kv_head_stride_bytes),
                    _i64(layout.kv_v_offset_bytes),
                    _i64(2048),
                    _var("tokens"),
                    label=f"dspark.ctx.{l}.store_paged",
                )
            )
    return calls


def emit_prefill_capture(
    m: dict,
    layout: DsparkTargetLayout,
    taps_buf: str = "d.taps",
    lines_buf: str = "d.lines",
    valid_buf: str | None = None,
) -> tuple[dict[int, list[dict]], list[dict]]:
    """Return `(tap_hooks_by_visit_id, tail_calls)` for `prefill` (`groups=1, rows=tokens`)."""
    tap_hooks = {
        vid: [
            _call(
                "dspark_tap_capture_0" if idx == 0 else "dspark_tap_capture",
                _b("h"),
                _b(taps_buf),
                _i32(idx),
                _i32(DSPARK_HIDDEN),
                _i32(len(TARGET_LAYER_IDS)),
                label=f"dspark.tap.{vid}",
            )
        ]
        for idx, vid in enumerate(TARGET_LAYER_IDS)
    }
    tail_calls = _emit_context_store_calls(
        layout, "prefill", taps_buf=taps_buf, lines_buf=lines_buf, valid_buf=valid_buf
    ) + [
        _call(
            "dspark_gather_last_tap_prefill",
            _b(taps_buf),
            _b("lengths"),
            _b("d.last_taps"),
            _i32(0),
            _i32(0),
            _var("tokens"),
            _i32(4 * DSPARK_HIDDEN),
            label="dspark.prefill.gather_last_tap",
        ),
        _call(
            "dspark_mix_rmsnorm_one",
            _b("d.last_taps"),
            _b("d.fuse_alpha", 3 * 4 * 2),
            _b("d.markov_head.dflare_norm.weight"),
            _b("d.f_rnn"),
            _i64(4 * DSPARK_HIDDEN),
            _i64(DSPARK_HIDDEN),
            _i64(1),
            label="dspark.prefill.frnn_mix",
        ),
        _call(
            "dspark_store_frnn_prefill",
            _b("d.f_rnn"),
            _st("dspark_state"),
            _b(lines_buf),
            _i64(DSPARK_STATE_BYTES_PER_SEQ),
            _i64(DSPARK_SWA_BYTES),
            _i32(DSPARK_HIDDEN),
            label="dspark.prefill.store_frnn",
        ),
    ]
    return tap_hooks, tail_calls


def emit_decode_capture(
    m: dict,
    layout: DsparkTargetLayout,
    taps_buf: str = "d.taps",
    lines_buf: str = "d.lines",
    valid_buf: str | None = None,
) -> tuple[dict[int, list[dict]], list[dict]]:
    """Return `(tap_hooks_by_visit_id, tail_calls)` for `decode` (`groups=max_seqs, rows=1`)."""
    tap_hooks = {
        vid: [
            _call(
                "dspark_tap_capture_0" if idx == 0 else "dspark_tap_capture",
                _b("h"),
                _b(taps_buf),
                _i32(idx),
                _i32(DSPARK_HIDDEN),
                _i32(len(TARGET_LAYER_IDS)),
                label=f"dspark.tap.{vid}",
            )
        ]
        for idx, vid in enumerate(TARGET_LAYER_IDS)
    }
    tail_calls = _emit_context_store_calls(
        layout, "decode", taps_buf=taps_buf, lines_buf=lines_buf, valid_buf=valid_buf
    ) + [
        _call(
            "dspark_gather_last_tap",
            _b(taps_buf),
            _b("lengths"),
            _b("d.last_taps"),
            _i32(1),
            _i32(0),
            _var("tokens"),
            _i32(4 * DSPARK_HIDDEN),
            label="dspark.decode.gather_last_tap",
        ),
        _call(
            "dspark_mix_rmsnorm_seqs",
            _b("d.last_taps"),
            _b("d.fuse_alpha", 3 * 4 * 2),
            _b("d.markov_head.dflare_norm.weight"),
            _b("d.f_rnn"),
            _i64(4 * DSPARK_HIDDEN),
            _i64(DSPARK_HIDDEN),
            _i64(1),
            label="dspark.decode.frnn_mix",
        ),
        _call(
            "dspark_store_frnn",
            _b("d.f_rnn"),
            _st("dspark_state"),
            _b(lines_buf),
            _i64(DSPARK_STATE_BYTES_PER_SEQ),
            _i64(DSPARK_SWA_BYTES),
            _i32(DSPARK_HIDDEN),
            label="dspark.decode.store_frnn",
        ),
    ]
    return tap_hooks, tail_calls


def emit_round_program(
    m: dict,
    layout: DsparkTargetLayout,
    target_verify_calls: list[dict],
    target_commit_calls: list[dict] | None = None,
    plan_calls: list[dict] | None = None,
    taps_buf: str | None = None,
    lines_buf: str = "d.lines",
    valid_buf: str | None = None,
) -> dict:
    """Assemble `programs['round']` with `batch: {groups: max_seqs, rows: 8}`, `graph: True`."""
    taps_name = taps_buf or layout.taps_buf
    calls: list[dict] = list(plan_calls or [])

    # 1. Load persistent f_rnn from dspark_state[d.lines[seq]] & embed draft block
    calls += [
        _call(
            "dspark_load_frnn",
            _st("dspark_state"),
            _b(lines_buf),
            _b("d.f_rnn"),
            _i64(DSPARK_STATE_BYTES_PER_SEQ),
            _i64(DSPARK_SWA_BYTES),
            _i32(DSPARK_HIDDEN),
            label="dspark.load_frnn",
        ),
        _call(
            "dspark_embed_block",
            _b("d.anchor"),
            _b(layout.target_embed_buf),
            _b(layout.target_embed_norm_buf),
            _b("d.mask_embed"),
            _b("d.h"),
            _i32(DSPARK_BLOCK_SIZE),
            _i32(DSPARK_HIDDEN),
            _f32(DSPARK_RMS_EPS),
            label="dspark.embed_block",
        ),
    ]

    # 2. 3-layer bidirectional block drafter
    swa_idx = 0
    for l in range(DSPARK_LAYERS):
        p = f"layers.{l}."
        if l == 0:
            calls.append(
                _call(
                    "dspark_rms_norm",
                    _b("d.h"),
                    _b("d.h"),
                    _b(f"d.{p}input_layernorm.weight"),
                    _b("d.x"),
                    label=f"dspark.l{l}.in_norm",
                )
            )
        else:
            calls.append(
                _call(
                    "dspark_add_rms_norm",
                    _b("d.branch"),
                    _b("d.h"),
                    _b(f"d.{p}input_layernorm.weight"),
                    _b("d.x"),
                    label=f"dspark.l{l}.in_norm",
                )
            )
        calls += [
            _call(
                "dspark_gemm_bf16",
                _b("d.x"),
                _b(f"d.{p}qkv.weight"),
                _b("d.qkv"),
                _var("tokens"),
                _i32(5120),
                _i32(DSPARK_HIDDEN),
                label=f"dspark.l{l}.qkv",
            ),
            _call(
                "dspark_qk_norm_rope",
                _b("d.qkv"),
                _b(f"d.{p}self_attn.q_norm.weight"),
                _b(f"d.{p}self_attn.k_norm.weight"),
                _b("d.inv_freq"),
                _b("positions"),
                _i64(5120),
                _i64(DSPARK_Q_HEADS * DSPARK_HEAD_DIM),
                label=f"dspark.l{l}.qk_norm_rope",
            ),
        ]
        if l in DSPARK_LOCAL_LAYERS:
            calls.append(
                _call(
                    "dspark_attn_ring",
                    _b("d.qkv"),
                    _b("d.qkv", 3072 * 2),
                    _b("d.qkv", 4096 * 2),
                    _st("dspark_state", swa_idx * (DSPARK_SWA_WINDOW * 2048)),
                    _b(lines_buf),
                    _b("lengths"),
                    _b("d.attn"),
                    _f32(layout.dspark_k_scale),
                    _f32(layout.dspark_v_scale),
                    _f32(DSPARK_HEAD_DIM**-0.5),
                    _i64(5120),
                    _i64(5120),
                    _i64(5120),
                    _i64(3072),
                    _i64(DSPARK_STATE_BYTES_PER_SEQ),
                    label=f"dspark.l{l}.attn_ring",
                )
            )
            swa_idx += 1
        else:
            if layout.num_kv_splits > 1:
                calls += [
                    _call(
                        "dspark_attn_paged_partial",
                        _b("d.qkv"),
                        _b("d.qkv", 3072 * 2),
                        _b("d.qkv", 4096 * 2),
                        _st(layout.kv_state),
                        _b("pages"),
                        _b("lengths"),
                        _b("d.attn_o_part"),
                        _b("d.attn_lse_part"),
                        _f32(layout.dspark_k_scale),
                        _f32(layout.dspark_v_scale),
                        _f32(DSPARK_HEAD_DIM**-0.5),
                        _i64(5120),
                        _i64(5120),
                        _i64(5120),
                        _i64(layout.page_columns),
                        _i64(3072),
                        _i64(layout.kv_layer_offset_bytes),
                        _i64(layout.kv_page_stride_bytes),
                        _i64(layout.kv_slot_stride_bytes),
                        _i64(layout.kv_head_stride_bytes),
                        _i64(layout.kv_v_offset_bytes),
                        _var("tokens"),
                        _i32(layout.num_kv_splits),
                        label=f"dspark.l{l}.attn_paged",
                    ),
                    _call(
                        "dspark_attn_combine_16",
                        _b("d.attn_o_part"),
                        _b("d.attn_lse_part"),
                        _b("d.attn"),
                        _var("tokens"),
                        label=f"dspark.l{l}.attn_combine",
                    ),
                ]
            else:
                calls.append(
                    _call(
                        "dspark_attn_paged_direct",
                        _b("d.qkv"),
                        _b("d.qkv", 3072 * 2),
                        _b("d.qkv", 4096 * 2),
                        _st(layout.kv_state),
                        _b("pages"),
                        _b("lengths"),
                        _b("d.attn"),
                        _b("d.attn_lse_part"),
                        _f32(layout.dspark_k_scale),
                        _f32(layout.dspark_v_scale),
                        _f32(DSPARK_HEAD_DIM**-0.5),
                        _i64(5120),
                        _i64(5120),
                        _i64(5120),
                        _i64(layout.page_columns),
                        _i64(3072),
                        _i64(layout.kv_layer_offset_bytes),
                        _i64(layout.kv_page_stride_bytes),
                        _i64(layout.kv_slot_stride_bytes),
                        _i64(layout.kv_head_stride_bytes),
                        _i64(layout.kv_v_offset_bytes),
                        _var("tokens"),
                        _i32(1),
                        label=f"dspark.l{l}.attn_paged",
                    )
                )

        calls += [
            _call(
                "dspark_gemm_bf16",
                _b("d.attn"),
                _b(f"d.{p}self_attn.o_proj.weight"),
                _b("d.branch"),
                _var("tokens"),
                _i32(DSPARK_HIDDEN),
                _i32(3072),
                label=f"dspark.l{l}.o_proj",
            ),
            _call(
                "dspark_add_rms_norm",
                _b("d.branch"),
                _b("d.h"),
                _b(f"d.{p}post_attention_layernorm.weight"),
                _b("d.x"),
                label=f"dspark.l{l}.post_norm",
            ),
            _call(
                "dspark_gemm_bf16",
                _b("d.x"),
                _b(f"d.{p}gate_up.weight"),
                _b("d.gate_up"),
                _var("tokens"),
                _i32(2 * DSPARK_INTER),
                _i32(DSPARK_HIDDEN),
                label=f"dspark.l{l}.gate_up",
            ),
            _call(
                "dspark_silu_mul",
                _b("d.gate_up"),
                _b("d.act"),
                _i64(2 * DSPARK_INTER),
                label=f"dspark.l{l}.silu_mul",
            ),
            _call(
                "dspark_gemm_bf16",
                _b("d.act"),
                _b(f"d.{p}mlp.down_proj.weight"),
                _b("d.branch"),
                _var("tokens"),
                _i32(DSPARK_HIDDEN),
                _i32(DSPARK_INTER),
                label=f"dspark.l{l}.down_proj",
            ),
        ]

    calls.append(
        _call(
            "dspark_add_rms_norm",
            _b("d.branch"),
            _b("d.h"),
            _b("d.norm.weight"),
            _b("d.x"),
            label="dspark.final_norm",
        )
    )

    # 3. Inkling2MarkovHead 7-step unrolled loop
    calls += [
        _call("dspark_skip_anchor", _b("d.x"), _b("d.step_hidden"), label="markov.skip_anchor"),
        _call(
            "dspark_gemm_bf16",
            _b("d.step_hidden"),
            _b(layout.target_lm_head_buf),
            _b("d.base_logits"),
            _expr(_mul("seqs", DSPARK_NUM_SPEC)),
            _i32(VOCAB_PADDED),
            _i32(DSPARK_HIDDEN),
            label="markov.base_logits",
        ),
        _call(
            "dspark_markov_ctx_concat",
            _b("d.step_hidden"),
            _b("d.f_rnn"),
            _b("d.markov_ctx_in"),
            label="markov.ctx_concat",
        ),
        _call(
            "dspark_gemm_bf16_f32",
            _b("d.markov_ctx_in"),
            _b("d.w_ctx"),
            _b("d.markov_ctx"),
            _expr(_mul("seqs", DSPARK_NUM_SPEC)),
            _i32(3 * DSPARK_MARKOV_RANK),
            _i32(2 * DSPARK_HIDDEN),
            label="markov.ctx_proj",
        ),
        _call("dspark_markov_add_bias", _b("d.markov_ctx"), _b("d.rnn_bias"), label="markov.ctx_bias"),
        _call(
            "spec_zeros_bf16",
            _b("d.markov_state"),
            _expr(_mul("seqs", DSPARK_MARKOV_RANK)),
            label="markov.zero_state",
        ),
    ]

    for step in range(DSPARK_NUM_SPEC):
        prev_buf = _b("d.anchor") if step == 0 else _b("d.draft")
        prev_off = 0 if step == 0 else step - 1
        prev_stride = 1 if step == 0 else DSPARK_NUM_SPEC
        calls += [
            _call(
                "dspark_markov_state_concat",
                _b("d.markov_state"),
                prev_buf,
                _b("d.markov_head.markov_w1.weight"),
                _b("d.markov_in"),
                _i32(prev_off),
                _i32(prev_stride),
                label=f"markov.s{step}.state_concat",
            ),
            _call(
                "dspark_gemm_bf16_f32",
                _b("d.markov_in"),
                _b("d.w_state"),
                _b("d.markov_state_part"),
                _var("seqs"),
                _i32(3 * DSPARK_MARKOV_RANK),
                _i32(2 * DSPARK_MARKOV_RANK),
                label=f"markov.s{step}.state_proj",
            ),
            _call(
                "dspark_markov_gate_add",
                _b("d.markov_ctx"),
                _b("d.markov_state_part"),
                _b("d.markov_state"),
                _b("d.markov_state"),
                _b("d.markov_latent"),
                _i32(step),
                label=f"markov.s{step}.gate",
            ),
            _call(
                "dspark_gemm_bf16",
                _b("d.markov_latent"),
                _b("d.fold_weight"),
                _b("d.step_bias"),
                _var("seqs"),
                _i32(VOCAB_PADDED),
                _i32(DSPARK_MARKOV_RANK),
                label=f"markov.s{step}.fold_proj",
            ),
            _call(
                "dspark_add_softcap_mask",
                _b("d.base_logits"),
                _b("d.step_bias"),
                _b("d.fold_bias"),
                _b("d.step_logits"),
                _i32(step),
                _f32(SOFTCAP),
                label=f"markov.s{step}.softcap_mask",
            ),
            _call(
                "dspark_markov_argmax_partial",
                _b("d.step_logits"),
                _i64(VOCAB_PADDED),
                _b("d.argmax_pmax"),
                _b("d.argmax_pidx"),
                _i32(VOCAB_PADDED),
                label=f"markov.s{step}.argmax_partial",
            ),
            _call(
                "dspark_markov_argmax_final",
                _b("d.argmax_pmax"),
                _b("d.argmax_pidx"),
                _b("d.draft", step * 8),
                _i32(DSPARK_NUM_SPEC),
                _i32(64),
                label=f"markov.s{step}.argmax_final",
            ),
        ]

    # 4. Splice verify ids + target verify forward
    calls.append(
        _call(
            "spec_splice_verify",
            _b("d.anchor"),
            _b("d.draft"),
            _b("d.verify_ids"),
            _i32(DSPARK_BLOCK_SIZE),
            _i32(DSPARK_NUM_SPEC),
            label="spec.splice_verify",
        )
    )
    calls.extend(target_verify_calls)

    # 5. Target softcap + vocab mask + sampling expansion + verify sampling + spec_count
    mask_op = (
        "dspark_verify_mask_vocab_f32"
        if layout.target_verify_logits_dtype == "f32"
        else "dspark_verify_mask_vocab_bf16"
    )
    sampling_names = ("temperature", "top_k", "top_p", "min_p", "seed", "lengths")
    calls += [
        _call(
            mask_op,
            _b(layout.target_verify_logits_buf),
            _f32(SOFTCAP),
            _i32(VOCAB_PADDED),
            _i32(VOCAB_UNPADDED),
            label="verify.softcap_mask",
        ),
        _call(
            "spec_sampling",
            *[_b(n) for n in sampling_names],
            *[_b("d.sample." + n) for n in sampling_names],
            _i32(DSPARK_BLOCK_SIZE),
            label="verify.spec_sampling",
        ),
        _call(
            "verify_sample",
            _b(layout.target_verify_logits_buf),
            _b("d.sample.lengths"),
            *[_b("d.sample." + n) for n in sampling_names[:-1]],
            _b("verify_tokens"),
            _var("tokens"),
            label="verify.sample",
        ),
        _call(
            "kern_spec_count",
            _b("d.draft"),
            _b("verify_tokens"),
            _b("accepted"),
            _i32(DSPARK_BLOCK_SIZE),
            _i32(DSPARK_NUM_SPEC),
            label="spec.count",
        ),
    ]

    # 6. Rollback-safe target & DSpark state commits (ONLY 0 .. accepted[seq]-1)
    if target_commit_calls is not None:
        calls.extend(target_commit_calls)
    else:
        calls += [
            _call(
                "spec_conv_commit",
                _b(layout.target_sconv_raw_buf),
                _st(layout.seq_state_name),
                _b(layout.seq_lines_buf),
                _b("accepted"),
                _i64(layout.seq_state_stride_bytes),
                _i64(layout.sconv_off_bytes),
                _i32(layout.max_seqs * DSPARK_BLOCK_SIZE),
                _i32(layout.target_sconv_dim),
                _i32(DSPARK_BLOCK_SIZE),
                label="spec.conv_commit",
            ),
            _call(
                "spec_swa_commit",
                _b(layout.target_swa_kv_buf),
                _st(layout.seq_state_name),
                _b(layout.seq_lines_buf),
                _b("lengths"),
                _b("accepted"),
                _i64(layout.seq_state_stride_bytes),
                _i64(layout.swa_off_bytes),
                _i32(layout.max_seqs * DSPARK_BLOCK_SIZE),
                _i32(layout.swa_window),
                _i32(DSPARK_BLOCK_SIZE),
                _i32(layout.swa_token_bytes),
                label="spec.swa_commit",
            ),
        ]

    calls += _emit_context_store_calls(
        layout, "commit", taps_buf=taps_name, lines_buf=lines_buf, valid_buf=valid_buf
    )
    calls += [
        _call(
            "dspark_gather_last_tap",
            _b(taps_name),
            _b("accepted"),
            _b("d.last_taps"),
            _i32(DSPARK_BLOCK_SIZE),
            _i32(1),
            _var("tokens"),
            _i32(4 * DSPARK_HIDDEN),
            label="dspark.commit.gather_last_tap",
        ),
        _call(
            "dspark_mix_rmsnorm_seqs",
            _b("d.last_taps"),
            _b("d.fuse_alpha", 3 * 4 * 2),
            _b("d.markov_head.dflare_norm.weight"),
            _b("d.f_rnn"),
            _i64(4 * DSPARK_HIDDEN),
            _i64(DSPARK_HIDDEN),
            _i64(1),
            label="dspark.commit.frnn_mix",
        ),
        _call(
            "dspark_store_frnn",
            _b("d.f_rnn"),
            _st("dspark_state"),
            _b(lines_buf),
            _i64(DSPARK_STATE_BYTES_PER_SEQ),
            _i64(DSPARK_SWA_BYTES),
            _i32(DSPARK_HIDDEN),
            label="dspark.commit.store_frnn",
        ),
    ]

    return {
        "batch": {"groups": layout.max_seqs, "rows": DSPARK_BLOCK_SIZE},
        "graph": True,
        "calls": calls,
    }


def build_standalone_test_manifest(
    kernels_dir: Path,
    sampling_target: dict | None = None,
    page_size: int = 64,
    num_kv_splits: int = 16,
) -> dict:
    """Construct a complete self-contained schema v7 manifest exercising `derive`, `prefill`, `decode`, and `round`
    for CPU `kern verify` and protocol validation."""
    target_limits = sampling_target or {"sms": 110, "shared_mem": 101376}
    layout = DsparkTargetLayout(
        page_size=page_size,
        page_columns=ceil_div(CONTEXT_LIMIT, page_size),
        kv_layer_offset_bytes=8 * 2 * 8 * page_size * 128,
        kv_page_stride_bytes=9 * 2 * 8 * page_size * 128,
        kv_slot_stride_bytes=128,
        kv_head_stride_bytes=page_size * 128,
        kv_v_offset_bytes=8 * page_size * 128,
        num_kv_splits=num_kv_splits,
    )
    m: dict = {
        "schema_version": 7,
        "model": "inkling-2-mini-dspark-standalone",
        "vars": {
            "tokens": {"max": 1024, "axis": "rows"},
            "seqs": {"max": layout.max_seqs, "axis": "groups"},
        },
        "states": {
            layout.kv_state: {"bytes_per_token": 9 * 2048},
            layout.seq_state_name: {"bytes_per_seq": layout.seq_state_stride_bytes},
        },
        "buffers": {
            "token_ids": {"dtype": "i64", "shape": ["tokens"], "kind": "input", "fill": "token", "domain": {"min": 0, "max": VOCAB_UNPADDED - 1}},
            "positions": {"dtype": "i64", "shape": ["tokens"], "kind": "input", "fill": "position", "domain": {"min": 0, "max": CONTEXT_LIMIT - 1}},
            "slots": {"dtype": "i32", "shape": ["tokens"], "kind": "input", "fill": "slot", "domain": {"index_into": layout.kv_state}},
            "lengths": {"dtype": "i32", "shape": ["seqs"], "kind": "input", "fill": "seq_len", "domain": {"min": 1, "max": CONTEXT_LIMIT}},
            "valid": {"dtype": "i32", "shape": ["tokens"], "kind": "input", "fill": "valid"},
            "q_indptr": {"dtype": "i32", "shape": [layout.max_seqs + 1], "kind": "input", "fill": "cu_seqlens", "domain": {"min": 0, "max": "tokens", "monotone": True}},
            "pages": {"dtype": "i32", "shape": ["seqs", layout.page_columns], "kind": "input", "domain": {"index_into": layout.kv_state}},
            layout.seq_lines_buf: {
                "dtype": "i32",
                "shape": [1, "seqs"],
                "kind": "input",
                "domain": {"index_into": layout.seq_state_name, "stride": layout.seq_state_stride_bytes},
            },
            **sampling_fills(),
            layout.target_embed_buf: {"dtype": "bf16", "shape": [VOCAB_PADDED, DSPARK_HIDDEN], "kind": "weight", "bind": [{"tensor": "model.llm.embed.weight"}]},
            layout.target_embed_norm_buf: {"dtype": "bf16", "shape": [DSPARK_HIDDEN], "kind": "weight", "bind": [{"tensor": "model.llm.embed_norm.weight"}]},
            layout.target_lm_head_buf: {"dtype": "bf16", "shape": [VOCAB_PADDED, DSPARK_HIDDEN], "kind": "weight", "bind": [{"tensor": "model.llm.unembed.weight"}]},
            "h": {"dtype": "bf16", "shape": ["tokens", DSPARK_HIDDEN], "kind": "workspace"},
            layout.target_verify_logits_buf: {"dtype": "f32", "shape": ["tokens", VOCAB_PADDED], "kind": "workspace"},
            "next_token": {"dtype": "i64", "shape": ["seqs"], "kind": "output", "fill": "tokens", "domain": {"min": 0, "max": VOCAB_UNPADDED - 1}},
            layout.target_sconv_raw_buf: {"dtype": "bf16", "shape": [layout.target_n_visits, layout.max_seqs * DSPARK_BLOCK_SIZE, layout.target_sconv_dim], "kind": "workspace"},
            layout.target_swa_kv_buf: {"dtype": "fp8e4m3", "shape": [layout.target_n_swa_visits, layout.max_seqs * DSPARK_BLOCK_SIZE, 2, DSPARK_KV_HEADS, DSPARK_HEAD_DIM], "kind": "workspace"},
        },
        "modules": {},
        "ops": {},
        "programs": {},
    }

    loaded = register_ops(m, kernels_dir, layout, target_limits)
    spec = loaded["spec"]

    # Standalone minimal target embed/logits/sample ops so prefill/decode/round verify cleanly
    _cuda_op(
        m["ops"],
        "stub_embed",
        "spec_markov_rows",
        "kern_embedding_rows_i64_bf16",
        ["in buffer<i64>", "i32", "in buffer<bf16>", "out buffer<bf16>", "i32"],
        ["tokens", 1, 1],
        256,
    )
    _cuda_op(
        m["ops"],
        "stub_copy_sconv",
        "spec_state_commit",
        "spec_zeros_bf16",
        ["out buffer<bf16>", "i32"],
        [1, 1, 1],
        256,
    )
    m["ops"]["stub_zero_swa"] = {
        "params": ["out buffer<fp8e4m3>"],
        "impl": {
            "launches": [
                {
                    "module": "spec_state_commit",
                    "entry": "spec_zeros_bf16",
                    "params": ["out buffer<fp8e4m3>", "i32"],
                    "grid": [1, 1, 1],
                    "block": [256, 1, 1],
                    "args": [{"param": 0}, _i32(1)],
                }
            ]
        },
    }
    m["ops"]["sample"] = sampling_op(
        VOCAB_PADDED,
        layout.max_seqs,
        spec["sampling"],
        target_limits,
        logits_dtype="f32",
        axis_var="seqs",
    )

    m["programs"]["derive"] = {"derive": True, "calls": emit_derive_calls(m, layout)}

    # Prefill (groups: 1, rows: "tokens", emits next_token)
    pre_hooks, pre_tail = emit_prefill_capture(m, layout)
    pre_calls = [
        _call("stub_embed", _b("token_ids"), _i32(1), _b(layout.target_embed_buf), _b("h"), _i32(DSPARK_HIDDEN), label="embed"),
        _call("dspark_tap_capture_0", _b("h"), _b("d.taps"), _i32(0), _i32(DSPARK_HIDDEN), _i32(4), label="touch_pos_valid_qindptr"),
    ]
    for vid in TARGET_LAYER_IDS:
        pre_calls.extend(pre_hooks[vid])
    pre_calls.extend(pre_tail)
    # Touch remaining protocol inputs and emit next_token
    pre_calls += [
        _call("dspark_gemm_bf16_f32", _b("h"), _b(layout.target_lm_head_buf), _b(layout.target_verify_logits_buf), _var("seqs"), _i32(VOCAB_PADDED), _i32(DSPARK_HIDDEN), label="lm_head"),
        _call("sample", _b(layout.target_verify_logits_buf), _b("lengths"), _b("temperature"), _b("top_k"), _b("top_p"), _b("min_p"), _b("seed"), _b("next_token"), _var("seqs"), label="sample"),
    ]
    m["programs"]["prefill"] = {"batch": {"groups": 1, "rows": "tokens"}, "calls": pre_calls}

    # Decode (groups: 8, rows: 1, graph: true, emits next_token)
    dec_hooks, dec_tail = emit_decode_capture(m, layout)
    dec_calls = [_call("stub_embed", _b("token_ids"), _i32(1), _b(layout.target_embed_buf), _b("h"), _i32(DSPARK_HIDDEN), label="embed")]
    for vid in TARGET_LAYER_IDS:
        dec_calls.extend(dec_hooks[vid])
    dec_calls.extend(dec_tail)
    dec_calls += [
        _call("dspark_gemm_bf16_f32", _b("h"), _b(layout.target_lm_head_buf), _b(layout.target_verify_logits_buf), _var("seqs"), _i32(VOCAB_PADDED), _i32(DSPARK_HIDDEN), label="lm_head"),
        _call("sample", _b(layout.target_verify_logits_buf), _b("lengths"), _b("temperature"), _b("top_k"), _b("top_p"), _b("min_p"), _b("seed"), _b("next_token"), _var("seqs"), label="sample"),
    ]
    m["programs"]["decode"] = {"batch": {"groups": layout.max_seqs, "rows": 1}, "graph": True, "calls": dec_calls}

    # Round (groups: 8, rows: 8, graph: true)
    verify_calls = [
        _call("stub_embed", _b("d.verify_ids"), _i32(1), _b(layout.target_embed_buf), _b("h"), _i32(DSPARK_HIDDEN), label="verify.embed"),
        _call("stub_copy_sconv", _b(layout.target_sconv_raw_buf), _i32(1), label="verify.stage_sconv"),
        _call("stub_zero_swa", _b(layout.target_swa_kv_buf), label="verify.stage_swa"),
    ]
    for idx, vid in enumerate(TARGET_LAYER_IDS):
        verify_calls.append(
            _call(
                "dspark_tap_capture_0" if idx == 0 else "dspark_tap_capture",
                _b("h"),
                _b("d.taps"),
                _i32(idx),
                _i32(DSPARK_HIDDEN),
                _i32(len(TARGET_LAYER_IDS)),
                label=f"verify.tap.{vid}",
            )
        )
    verify_calls.append(
        _call(
            "dspark_gemm_bf16_f32",
            _b("h"),
            _b(layout.target_lm_head_buf),
            _b(layout.target_verify_logits_buf),
            _var("tokens"),
            _i32(VOCAB_PADDED),
            _i32(DSPARK_HIDDEN),
            label="verify.lm_head",
        )
    )
    m["programs"]["round"] = emit_round_program(m, layout, verify_calls)

    # Touch `valid` and `q_indptr` in prefill so all protocol buffers are referenced.
    m["ops"]["stub_check_meta"] = {
        "params": ["in buffer<i32>", "in buffer<i32>"],
        "impl": {
            "launches": [
                {
                    "module": "spec_round",
                    "entry": "kern_ones_i32",
                    "params": ["out buffer<i32>", "i32"],
                    "grid": [1, 1, 1],
                    "block": [32, 1, 1],
                    "args": [{"scratch": "tmp"}, _i32(1)],
                }
            ],
            "scratch": {"tmp": {"dtype": "i32", "shape": [1]}},
        },
    }
    m["programs"]["prefill"]["calls"].insert(0, _call("stub_check_meta", _b("valid"), _b("q_indptr"), label="check_meta"))

    return m


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", help="Build all M4 SM120 .cubin and JSON artifacts")
    parser.add_argument("--nvcc", default=None)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--verify-standalone", action="store_true", help="Write & verify standalone manifest via kern verify")
    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    if args.build:
        res = build_m4_kernels(repo_root, nvcc=args.nvcc, jobs=args.jobs)
        print(
            f"Built M4 SM120 artifacts: {len(res['spec_build']['modules'])} CUDA modules, "
            f"{len(res['dspark_dsl']['modules'])} Triton DSL modules"
        )
    if args.verify_standalone:
        import tempfile
        m = build_standalone_test_manifest(repo_root / "build/sm_120")
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "model.json"
            p.write_text(json.dumps(m, indent=2))
            subprocess.run(["/workspace/kern/target/release/kern", "verify", str(p)], check=True)
