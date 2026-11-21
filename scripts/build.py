#!/usr/bin/env python3
"""Build SM120 CUDA cubins declared in kernels.toml with non-destructive metadata merging."""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tomllib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOML = ROOT / "kernels.toml"


def _cuobjdump_path(nvcc: str | None) -> str:
    if nvcc:
        cand = Path(nvcc).resolve().with_name("cuobjdump")
        if cand.exists():
            return str(cand)
    found = shutil.which("cuobjdump")
    if found:
        return found
    return str(Path(os.environ.get("CUDA_HOME", "/usr/local/cuda")) / "bin" / "cuobjdump")


def extract_cubin_metadata(cubin_path: Path, cuobjdump_bin: str) -> dict:
    """Extract per-function register, static shared memory, and KPARAM sizes via cuobjdump."""
    res_out = subprocess.check_output([cuobjdump_bin, "-res-usage", str(cubin_path)], text=True)
    elf_out = subprocess.check_output([cuobjdump_bin, "-elf", str(cubin_path)], text=True)
    functions = {}
    cur = None
    for line in res_out.splitlines():
        s = line.strip()
        if s.startswith("Function "):
            cur = s.split()[1].rstrip(":")
        elif cur and "REG:" in s:
            reg_m = re.search(r"REG:(\d+)", s)
            sh_m = re.search(r"SHARED:(\d+)", s)
            regs = int(reg_m.group(1)) if reg_m else 0
            static_shared = int(sh_m.group(1)) if sh_m else 0

            param_sizes = []
            hdr = re.search(r"^\.nv\.info\.%s\s*$" % re.escape(cur), elf_out, re.M)
            if hdr:
                nxt = re.search(r"^\.[A-Za-z]", elf_out[hdr.end():], re.M)
                sect = elf_out[hdr.end():hdr.end() + nxt.start() if nxt else None]
                params = {
                    int(o, 16): int(sz, 16)
                    for o, sz in re.findall(
                        r"Ordinal\s*:\s*0x([0-9a-f]+)\s+Offset\s*:\s*0x[0-9a-f]+\s+Size\s*:\s*0x([0-9a-f]+)",
                        sect,
                    )
                }
                param_sizes = [params[i] for i in range(len(params))]
            functions[cur] = {
                "registers": regs,
                "static_shared": static_shared,
                "param_sizes": param_sizes,
            }
            cur = None
    return functions


def pin_sha256(text: str, name: str, sha: str) -> str:
    head = f'[kernels."{name}"]\n'
    i = text.index(head)
    j = text.find("\n[", i + len(head))
    j = len(text) if j < 0 else j + 1
    block = re.sub(r'sha256 = "[0-9a-f]*"\n', "", text[i:j])
    return text[:i] + block.replace(head, f'{head}sha256 = "{sha}"\n') + text[j:]


def check_flashinfer(cfg: dict) -> str:
    fi_cfg = cfg.get("upstreams", {}).get("flashinfer")
    if not fi_cfg:
        return ""
    upstream = ROOT / fi_cfg["path"]
    if not upstream.exists():
        raise SystemExit(f"Missing upstream directory: {upstream}")
    rev = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=upstream, text=True).strip()
    want = fi_cfg.get("revision", "")
    if want and rev != want:
        raise SystemExit(f"FlashInfer: expected {want}, got {rev}")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=upstream, text=True
    ).strip()
    if dirty:
        raise SystemExit(f"FlashInfer has local modifications:\n{dirty}")
    return rev


def _sync_module_report(report_path: Path, name: str, cubin_sha: str, src_sha: str, functions: dict) -> None:
    if not report_path.exists():
        return
    data = json.loads(report_path.read_text())
    mods = data.get("modules")
    if not isinstance(mods, dict) or name not in mods:
        return
    entry = dict(mods[name])
    entry["sha256"] = cubin_sha
    entry["source_sha256"] = src_sha
    if "functions" in entry:
        entry["functions"] = functions
    mods[name] = entry
    tmp = report_path.with_name(f".{report_path.name}.tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.replace(report_path)


def build(args) -> None:
    cfg = tomllib.loads(TOML.read_text())
    arch = cfg["toolchain"]["arch"]
    arch_dir = arch.rstrip("a")  # sm_120a -> build/sm_120
    out = ROOT / "build" / arch_dir
    out.mkdir(parents=True, exist_ok=True)

    selected = []
    for name, k in cfg.get("kernels", {}).items():
        if args.names and name not in args.names:
            continue
        if args.milestone and k.get("milestone") != args.milestone:
            continue
        if args.prebuilt_only and "prebuilt" not in k:
            continue
        selected.append((name, k))

    if not selected:
        print("No matching CUDA kernel targets selected.", flush=True)
        return

    if args.verify_sha and not args.rebuild and not args.update:
        bad = []
        for name, k in selected:
            target = out / f"{name}.cubin"
            expected_sha = k.get("sha256")
            if not target.exists():
                print(f"{name}: missing ({target})", flush=True)
                bad.append(name)
                continue
            sha = hashlib.sha256(target.read_bytes()).hexdigest()
            if expected_sha and sha == expected_sha:
                print(f"{name}: {sha} (matches kernels.toml)", flush=True)
            else:
                print(f"{name}: {sha} (differs from kernels.toml: {expected_sha})", flush=True)
                bad.append(name)
        if bad:
            raise SystemExit("on-disk cubins missing or differ from kernels.toml: " + ", ".join(bad))
        return

    if args.clean:
        for name, _ in selected:
            for suffix in (".cubin", ".json"):
                p = out / f"{name}{suffix}"
                if p.exists():
                    p.unlink()

    fi_rev = check_flashinfer(cfg)
    nvcc = args.nvcc or os.environ.get("NVCC") or shutil.which("nvcc")
    compiler_ver = ""
    if not args.prebuilt_only:
        if not nvcc:
            raise SystemExit("nvcc not found (pass --nvcc or set NVCC)")
        compiler_ver = subprocess.check_output([nvcc, "--version"], text=True).strip()
        want_nvcc = cfg["toolchain"]["nvcc"]
        if not args.skip_version_check and f"V{want_nvcc}" not in compiler_ver:
            raise SystemExit(
                f"nvcc V{want_nvcc} expected (kernels.toml), got: {compiler_ver.splitlines()[-1]}"
            )
    cuobjdump_bin = _cuobjdump_path(nvcc)

    def compile_one(item):
        name, k = item
        target = out / f"{name}.cubin"
        if "prebuilt" in k:
            src = ROOT / k["prebuilt"]
            shutil.copyfile(src, target)
            cmd = ["prebuilt", str(src)]
            src_rel = k["prebuilt"]
            src_sha = hashlib.sha256(src.read_bytes()).hexdigest()
        else:
            src = ROOT / k["source"]
            flags = cfg["flags"][k["flags"]] if "flags" in k else []
            incs = [f"-I{ROOT / i}" for s in k.get("includes", []) for i in cfg["includes"][s]]
            defs = [f"-D{a}={b}" for a, b in sorted(k.get("defines", {}).items())]
            cmd = [nvcc, "-cubin", f"-arch={arch}", *flags, *incs, *defs, "-o", str(target), str(src)]
            subprocess.run(cmd, cwd=ROOT, check=True)
            src_rel = k["source"]
            src_sha = hashlib.sha256(src.read_bytes()).hexdigest()

        cubin_sha = hashlib.sha256(target.read_bytes()).hexdigest()
        functions = extract_cubin_metadata(target, cuobjdump_bin)
        mod_json_path = out / f"{name}.json"
        existing_mod = {}
        if mod_json_path.exists():
            try:
                existing_mod = json.loads(mod_json_path.read_text())
            except json.JSONDecodeError:
                existing_mod = {}
        meta = dict(existing_mod)
        meta.update({
            "name": name,
            "milestone": k.get("milestone", ""),
            "kind": "cuda",
            "architecture": arch,
            "source": target.name,
            "source_file": src_rel,
            "sha256": cubin_sha,
            "source_sha256": src_sha,
            "command": cmd,
            "block": k.get("block"),
            "shared_mem": k.get("shared_mem", 0),
            "signature": k.get("signature"),
            "functions": functions,
        })
        tmp_mod = out / f".{name}.json.tmp"
        tmp_mod.write_text(json.dumps(meta, indent=2) + "\n")
        tmp_mod.replace(mod_json_path)
        print(f"Built {name} ({cubin_sha[:12]}...) functions={list(functions.keys())}", flush=True)
        return name, k, cubin_sha, src_sha, functions, meta

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        results = list(pool.map(compile_one, selected))

    toml_text = TOML.read_text()
    bad = []
    for name, k, sha, _, _, _ in results:
        expected_sha = k.get("sha256")
        if expected_sha and sha == expected_sha:
            print(f"{name}: {sha} (matches kernels.toml)", flush=True)
        elif args.update:
            print(f"{name}: {sha} (pinned in kernels.toml)", flush=True)
            toml_text = pin_sha256(toml_text, name, sha)
        elif expected_sha or args.verify_sha:
            print(f"{name}: {sha} (differs from kernels.toml: {expected_sha})", flush=True)
            bad.append(name)

    if args.update:
        TOML.write_text(toml_text)

    # Non-destructive merge into build/sm_120/build.json and module-group reports
    build_json_path = out / "build.json"
    if build_json_path.exists():
        existing = json.loads(build_json_path.read_text())
        merged_modules = {
            mname: mmeta
            for mname, mmeta in existing.get("modules", {}).items()
            if (out / f"{mname}.cubin").exists()
        }
    else:
        existing = {}
        merged_modules = {}

    for name, _, cubin_sha, src_sha, functions, meta in results:
        merged_modules[name] = meta
        _sync_module_report(out / "moe_build.json", name, cubin_sha, src_sha, functions)
        _sync_module_report(out / "spec_build.json", name, cubin_sha, src_sha, functions)

    report = {
        "architecture": arch,
        "flashinfer_revision": fi_rev or existing.get("flashinfer_revision", ""),
        "compiler": compiler_ver or existing.get("compiler", ""),
        "modules": dict(sorted(merged_modules.items())),
    }
    tmp_report = out / ".build.json.tmp"
    tmp_report.write_text(json.dumps(report, indent=2) + "\n")
    tmp_report.replace(build_json_path)

    if bad:
        raise SystemExit("built cubins differ from kernels.toml: " + ", ".join(bad))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("names", nargs="*", help="optional module names from kernels.toml")
    ap.add_argument("--milestone", choices=["M1", "M2", "M3", "M4"], help="build only one milestone")
    ap.add_argument("--nvcc", default=shutil.which("nvcc"))
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--update", action="store_true", help="pin computed sha256 into kernels.toml")
    ap.add_argument("--verify-sha", action="store_true", help="verify on-disk cubin sha256 against kernels.toml (recompiles only if --rebuild is also set)")
    ap.add_argument("--rebuild", action="store_true", help="force recompilation of selected kernels")
    ap.add_argument("--prebuilt-only", action="store_true")
    ap.add_argument("--skip-version-check", action="store_true")
    ap.add_argument("--clean", action="store_true", help="wipe cubin/json artifacts for selected kernels before building")
    parsed = ap.parse_args()
    if parsed.jobs < 1:
        ap.error("--jobs must be >= 1")
    build(parsed)
