#!/usr/bin/env python3
"""Incremental SM120 AOT builder for loop/make (CPU-only, worktree-safe)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def combo_sha256(paths: list[Path]) -> str:
    h = hashlib.sha256()
    for p in paths:
        h.update(str(p.name).encode())
        h.update(b"\0")
        h.update(p.read_bytes() if p.exists() else b"<missing>")
        h.update(b"\0")
    return h.hexdigest()


def ensure_worktree_deps(tree: Path, main: Path) -> None:
    # Ensure third_party/flashinfer is populated (worktrees have empty submodule dirs).
    fi = tree / "third_party" / "flashinfer"
    main_fi = main / "third_party" / "flashinfer"
    if main_fi.exists() and not (fi / "include").exists():
        fi.parent.mkdir(parents=True, exist_ok=True)
        if fi.is_symlink() or fi.is_file():
            fi.unlink()
        elif fi.is_dir():
            shutil.rmtree(fi)
        os.symlink(main_fi, fi)

    # Seed build/sm_120 from MAIN if missing in this worktree.
    build_dir = tree / "build" / "sm_120"
    main_build = main / "build" / "sm_120"
    if not (build_dir / "build.json").exists() and main_build.exists():
        build_dir.mkdir(parents=True, exist_ok=True)
        for item in main_build.iterdir():
            if item.is_file():
                dst = build_dir / item.name
                if not dst.exists():
                    shutil.copy2(item, dst)


def _match_variant_entry(variant_name: str, functions: dict) -> str:
    # Match e.g. "n32" -> "*KernelN32*", "n64" -> "*KernelN64*", or exact substring match.
    v_up = variant_name.upper()
    cands = [fn for fn in functions if "device_kernel" in fn and v_up in fn.upper()]
    if len(cands) == 1:
        return cands[0]
    # Fallback: match trailing digits/suffix (e.g. N32, N64, N16, N128)
    m = re.search(r"([a-zA-Z]*\d+)$", variant_name)
    if m:
        tok = m.group(1).upper()
        cands = [fn for fn in functions if "device_kernel" in fn and tok in fn.upper()]
        if len(cands) == 1:
            return cands[0]
    # Fallback: direct substring in any function name
    cands = [fn for fn in functions if variant_name.lower() in fn.lower()]
    if len(cands) == 1:
        return cands[0]
    raise RuntimeError(
        f"Cannot uniquely match CUTLASS variant '{variant_name}' in functions={list(functions.keys())}"
    )


def refresh_moe_cutlass_probe(tree: Path, name: str, kcfg: dict, toml_cfg: dict, nvcc: str) -> None:
    out = tree / "build" / "sm_120"
    src = tree / kcfg["source"]
    if "MOE_METADATA_PROBE" not in src.read_text():
        return
    flags = list(toml_cfg["flags"][kcfg["flags"]]) if "flags" in kcfg else []
    incs = [f"-I{tree / i}" for s in kcfg.get("includes", []) for i in toml_cfg["includes"][s]]
    defs = [f"-D{a}={b}" for a, b in sorted(kcfg.get("defines", {}).items())]
    probe_bin = out / f".{name}_probe"
    cmd = [nvcc, *flags, *incs, *defs, "-DMOE_METADATA_PROBE", "-o", str(probe_bin), str(src)]
    print(f"[make] probing CUTLASS ABI for {name}...", flush=True)
    subprocess.run(cmd, cwd=tree, check=True)
    try:
        raw = subprocess.check_output([str(probe_bin), "128"], cwd=tree, text=True)
        variants = json.loads(raw)
    finally:
        if probe_bin.exists():
            probe_bin.unlink()

    mod_json_path = out / f"{name}.json"
    mod_meta = json.loads(mod_json_path.read_text())
    functions = mod_meta.get("functions", {})
    for vname, vinfo in variants.items():
        if "entry" not in vinfo:
            vinfo["entry"] = _match_variant_entry(vname, functions)
    mod_meta["gemm_variants"] = variants
    if "prepare_args_entry" not in mod_meta:
        prep = [fn for fn in functions if "prepare_args" in fn]
        if len(prep) == 1:
            mod_meta["prepare_args_entry"] = prep[0]
    tmp = out / f".{name}.json.tmp"
    tmp.write_text(json.dumps(mod_meta, indent=2) + "\n")
    tmp.replace(mod_json_path)

    moe_build_path = out / "moe_build.json"
    if moe_build_path.exists():
        mb = json.loads(moe_build_path.read_text())
        mb.setdefault("modules", {})[name] = mod_meta
        tmp_mb = out / ".moe_build.json.tmp"
        tmp_mb.write_text(json.dumps(mb, indent=2) + "\n")
        tmp_mb.replace(moe_build_path)


def sync_kernels_toml_shas(tree: Path, names: list[str]) -> None:
    toml_path = tree / "kernels.toml"
    out = tree / "build" / "sm_120"
    text = toml_path.read_text()
    changed = False
    for name in names:
        cubin = out / f"{name}.cubin"
        head = f'[kernels."{name}"]\n'
        if not cubin.exists() or head not in text:
            continue
        sha = file_sha256(cubin)
        i = text.index(head)
        j = text.find("\n[", i + len(head))
        j = len(text) if j < 0 else j + 1
        block = text[i:j]
        new_block = re.sub(r'sha256 = "[0-9a-f]*"\n', "", block).replace(
            head, f'{head}sha256 = "{sha}"\n', 1
        )
        if new_block != block:
            text = text[:i] + new_block + text[j:]
            changed = True
    if changed:
        toml_path.write_text(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", type=Path, required=True)
    ap.add_argument("--main", type=Path, required=True)
    ap.add_argument("--py", required=True)
    ap.add_argument("--nvcc", required=True)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()

    tree = args.tree.resolve()
    main_dir = args.main.resolve()
    ensure_worktree_deps(tree, main_dir)

    out = tree / "build" / "sm_120"
    out.mkdir(parents=True, exist_ok=True)
    toml_cfg = tomllib.loads((tree / "kernels.toml").read_text())
    kernels_cfg = toml_cfg.get("kernels", {})

    # 1. Detect changed M4 (DSpark/Spec) sources -> scripts/dspark_manifest.py --build
    m4_cuda_names = [n for n, k in kernels_cfg.items() if k.get("milestone") == "M4"]
    m4_watch = [
        tree / "source" / "dspark_kernels.py",
        tree / "source" / "markov_head.py",
        tree / "scripts" / "dspark_manifest.py",
        tree / "source" / "sampling_metadata.cu",
        *[tree / kernels_cfg[n]["source"] for n in m4_cuda_names if "source" in kernels_cfg[n]],
    ]
    m4_sha_file = out / ".m4_build.sha"
    m4_sha = combo_sha256(m4_watch)
    need_m4 = (
        args.all
        or not (out / "dspark_dsl.json").exists()
        or not (out / "spec_build.json").exists()
        or not m4_sha_file.exists()
        or m4_sha_file.read_text().strip() != m4_sha
    )
    if need_m4:
        print("[make] building M4 DSpark/Spec kernels...", flush=True)
        subprocess.run(
            [args.py, "scripts/dspark_manifest.py", "--build", "--nvcc", args.nvcc, "--jobs", str(args.jobs)],
            cwd=tree,
            check=True,
        )
        sync_kernels_toml_shas(tree, m4_cuda_names)
        m4_sha_file.write_text(m4_sha + "\n")

    # 2. Detect changed non-M4 CUDA kernels declared in kernels.toml
    changed_cuda: list[str] = []
    for name, k in kernels_cfg.items():
        if k.get("milestone") == "M4":
            continue
        cubin = out / f"{name}.cubin"
        meta_path = out / f"{name}.json"
        if args.all or not cubin.exists() or not meta_path.exists():
            changed_cuda.append(name)
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            changed_cuda.append(name)
            continue
        src_rel = k.get("source") or k.get("prebuilt")
        src_path = tree / src_rel
        if not src_path.exists() or file_sha256(src_path) != meta.get("source_sha256"):
            changed_cuda.append(name)
            continue
        # Also check if flags/defines/includes changed in kernels.toml
        sig_str = json.dumps(
            {
                "flags": toml_cfg["flags"].get(k.get("flags", ""), []),
                "includes": [toml_cfg["includes"][s] for s in k.get("includes", [])],
                "defines": k.get("defines", {}),
                "arch": toml_cfg["toolchain"]["arch"],
            },
            sort_keys=True,
        )
        cfg_sha = hashlib.sha256(sig_str.encode()).hexdigest()
        cfg_marker = out / f".{name}.cfgsha"
        if cfg_marker.exists() and cfg_marker.read_text().strip() != cfg_sha:
            changed_cuda.append(name)

    if changed_cuda:
        print(f"[make] building CUDA modules: {', '.join(changed_cuda)}", flush=True)
        subprocess.run(
            [args.py, "scripts/build.py", "--update", "--nvcc", args.nvcc, "--jobs", str(args.jobs), *changed_cuda],
            cwd=tree,
            check=True,
        )
        for name in changed_cuda:
            k = kernels_cfg[name]
            sig_str = json.dumps(
                {
                    "flags": toml_cfg["flags"].get(k.get("flags", ""), []),
                    "includes": [toml_cfg["includes"][s] for s in k.get("includes", [])],
                    "defines": k.get("defines", {}),
                    "arch": toml_cfg["toolchain"]["arch"],
                },
                sort_keys=True,
            )
            (out / f".{name}.cfgsha").write_text(hashlib.sha256(sig_str.encode()).hexdigest() + "\n")
            if name in ("moe_nvfp4_sm120", "moe_mxfp8_sm120"):
                refresh_moe_cutlass_probe(tree, name, k, toml_cfg, args.nvcc)

    # 3. Detect changed M3 Triton DSL (source/rel_attention.py, scripts/attn_build.py)
    m3_watch = [
        tree / "source" / "rel_attention.py",
        tree / "scripts" / "attn_build.py",
    ]
    m3_sha_file = out / ".m3_dsl.sha"
    m3_sha = combo_sha256(m3_watch)
    need_m3 = (
        args.all
        or not (out / "dsl.json").exists()
        or not m3_sha_file.exists()
        or m3_sha_file.read_text().strip() != m3_sha
    )
    if need_m3:
        print("[make] building M3 Triton DSL kernels...", flush=True)
        subprocess.run(
            [args.py, "scripts/build_dsl.py", "--milestone", "M3"],
            cwd=tree,
            check=True,
        )
        m3_sha_file.write_text(m3_sha + "\n")


if __name__ == "__main__":
    main()
