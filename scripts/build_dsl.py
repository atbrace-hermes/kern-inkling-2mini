#!/usr/bin/env python3
"""Compile Triton DSL operators to standalone SM120 cubins with non-destructive metadata merging."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

ROOT = Path(__file__).resolve().parents[1]

# Per-milestone optional DSL spec provider modules under scripts/
MILESTONE_PROVIDERS = {
    "M1": "loop_build",
    "M2": "moe_build",
    "M3": "attn_build",
    "M4": "dspark_build",
}


def load_source(name: str):
    path = ROOT / "source" / f"{name}.py"
    if not path.exists():
        raise FileNotFoundError(f"Missing Triton source: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_provider(module_name: str):
    path = ROOT / "scripts" / f"{module_name}.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def collect_specs(milestone: str | None) -> list[dict]:
    specs = []
    for ms, mod_name in MILESTONE_PROVIDERS.items():
        if milestone and ms != milestone:
            continue
        mod = _load_provider(mod_name)
        if mod is None:
            continue
        fn = getattr(mod, "specifications", None) or getattr(mod, "specs", None)
        if fn is None:
            continue
        for item in fn(load_source):
            if isinstance(item, dict):
                entry = dict(item)
                entry.setdefault("milestone", ms)
            else:
                # Tuple format: (name, kernel, signature, warps[, constants[, options[, attrs]]])
                entry = {
                    "name": item[0],
                    "milestone": ms,
                    "kernel": item[1],
                    "signature": item[2],
                    "num_warps": item[3],
                    "constants": item[4] if len(item) >= 5 else {},
                    "options": item[5] if len(item) >= 6 else {},
                    "attrs": item[6] if len(item) >= 7 else {},
                }
            specs.append(entry)
    return specs


def build(args) -> None:
    if not args.skip_version_check and triton.__version__ != "3.8.0":
        raise SystemExit(f"Expected Triton 3.8.0, got {triton.__version__}")

    out = ROOT / "build" / "sm_120"
    out.mkdir(parents=True, exist_ok=True)
    if args.clean:
        for p in out.glob("*.ptx"):
            p.unlink()

    all_specs = collect_specs(args.milestone)
    if args.names:
        all_specs = [s for s in all_specs if s["name"] in args.names]

    dsl_json_path = out / "dsl.json"
    if dsl_json_path.exists() and not args.clean:
        existing = json.loads(dsl_json_path.read_text())
        merged_modules = {
            mname: mmeta
            for mname, mmeta in existing.get("modules", {}).items()
            if (out / f"{mname}.cubin").exists()
        }
    else:
        existing = {}
        merged_modules = {}

    if not all_specs:
        print(
            f"No Triton DSL specs found for milestone={args.milestone or 'ALL'}; dsl.json intact.",
            flush=True,
        )
        payload = {
            "architecture": "sm_120a",
            "triton": triton.__version__,
            "warp_size": args.warp_size,
            "modules": dict(sorted(merged_modules.items())),
        }
        for extra_k in ("gemm_tuning", "prefill_gemm_tuning"):
            if extra_k in existing:
                payload[extra_k] = existing[extra_k]
        tmp = out / ".dsl.json.tmp"
        tmp.write_text(json.dumps(payload, indent=2) + "\n")
        tmp.replace(dsl_json_path)
        return

    target = GPUTarget("cuda", 120, args.warp_size)
    for spec in all_specs:
        name = spec["name"]
        kernel = spec["kernel"]
        raw_sig = spec["signature"]
        warps = spec["num_warps"]
        sig, constants = {}, {}
        if isinstance(raw_sig, dict):
            sig = dict(raw_sig)
        else:
            for arg, typ in zip(kernel.arg_names, raw_sig.split(","), strict=True):
                if typ.isdecimal():
                    constants[arg] = int(typ)
                else:
                    sig[arg] = typ
        constants.update(spec.get("constants") or {})
        options = {"num_warps": warps, "num_stages": 2}
        if args.no_fp_fusion:
            options["enable_fp_fusion"] = False
        options.update(spec.get("options") or {})
        attrs = spec.get("attrs") or {}

        compiled = triton.compile(
            ASTSource(kernel, signature=sig, constexprs=constants, attrs=attrs),
            target=target,
            options=options,
        )
        if (
            compiled.metadata.global_scratch_size
            or compiled.metadata.profile_scratch_size
            or compiled.metadata.num_ctas != 1
        ):
            raise ValueError(f"{name}: unsupported scratch or cluster launch requirements")

        cubin_path = out / f"{name}.cubin"
        ptx_path = out / f"{name}.ptx"
        cubin_path.write_bytes(compiled.asm["cubin"])
        ptx_path.write_text(compiled.asm["ptx"])
        sha = hashlib.sha256(cubin_path.read_bytes()).hexdigest()

        mod_meta = {
            "name": name,
            "milestone": spec["milestone"],
            "kind": "triton",
            "architecture": "sm_120a",
            "triton": triton.__version__,
            "source": cubin_path.name,
            "ptx": ptx_path.name,
            "sha256": sha,
            "entry": compiled.name,
            "shared_mem": compiled.metadata.shared,
            "block": [args.warp_size * warps, 1, 1],
            "signature": list(sig.values()),
            "arguments": list(sig),
            "abi_tail": ["i64", "i64"],
        }
        tmp_mod = out / f".{name}.json.tmp"
        tmp_mod.write_text(json.dumps(mod_meta, indent=2) + "\n")
        tmp_mod.replace(out / f"{name}.json")
        merged_modules[name] = mod_meta
        print(f"Built DSL {name} ({sha[:12]}...) shared={compiled.metadata.shared}", flush=True)

    payload = {
        "architecture": "sm_120a",
        "triton": triton.__version__,
        "warp_size": args.warp_size,
        "modules": dict(sorted(merged_modules.items())),
    }
    for extra_k in ("gemm_tuning", "prefill_gemm_tuning"):
        if extra_k in existing:
            payload[extra_k] = existing[extra_k]
    tmp = out / ".dsl.json.tmp"
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(dsl_json_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("names", nargs="*", help="optional DSL kernel names")
    ap.add_argument("--milestone", choices=["M1", "M2", "M3", "M4"])
    ap.add_argument("--warp-size", type=int, default=32)
    ap.add_argument("--no-fp-fusion", action="store_true")
    ap.add_argument("--skip-version-check", action="store_true")
    ap.add_argument("--clean", action="store_true")
    parsed = ap.parse_args()
    build(parsed)
