#!/usr/bin/env python3
"""Modal RTX PRO 6000 Blackwell (SM120 / sm_120a) remote GPU runner for kern-inkling-2mini.

Deploy once (or after editing this file):
    MODAL_DISABLE_API_PROXY=1 modal deploy loop/modal_runner.py

Populate persistent Volume (cheap CPU-only function, idempotent):
    loop/onmodal populate-weights

Invoke via wrapper `loop/onmodal`:
    loop/onmodal probe
    loop/onmodal unit <worktree_dir> <test_path> [extra unittest args...]
    loop/onmodal e2e <bundle_dir>
    loop/onmodal check <cand_bundle> [base_bundle] [--capacity N]
    loop/onmodal bench <bundle_dir> [workload_toml] [--out <local_json>] [--base <base_bundle>] [--ops]
    loop/onmodal exec <local_dir> "<shell_cmd>"
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import modal

APP_NAME = "kern-inkling-2mini-runner"
VOL_NAME = "inkling-2-mini-data"
GPU_SPEC = "RTX-PRO-6000"
DATA_DIR = "/data"
MAX_PAYLOAD_BYTES = 64 * 1024 * 1024  # 64 MiB compressed safety cap

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOL_NAME, create_if_missing=True)

cpu_loader_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("huggingface_hub>=0.28.0", "hf_transfer>=0.1.9")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

_IMAGE_SETUP_CMDS = [
    'bash -c \'set -euo pipefail '
    '&& /.uv/uv pip install --target /opt/cuda135 --no-deps nvidia-cublas==13.10.1.6 nvidia-cuda-runtime==13.5.37 nvidia-cuda-cupti==13.5.25 '
    '&& CU135_LIB=/opt/cuda135/nvidia/cu13/lib '
    '&& find /opt/cuda135 -name "libcupti*.so*" ! -path "$CU135_LIB/*" -exec ln -sf {} "$CU135_LIB"/ \\; '
    '&& ls -la "$CU135_LIB"/libcupti* '
    '&& CU13_TORCH_LIB=$(python -c "import os, nvidia.cu13 as c; print(os.path.join(c.__path__[0], \\"lib\\"))") '
    '&& echo "CUDA 13.5 libs: $CU135_LIB (fallback: $CU13_TORCH_LIB)" '
    '&& printf "%s\\n%s\\n" "$CU135_LIB" "$CU13_TORCH_LIB" > /etc/ld.so.conf.d/nvidia-cu13.conf '
    '&& ldconfig '
    '&& mkdir -p /usr/local/cuda/targets/x86_64-linux '
    '&& ln -sfn "$CU135_LIB" /usr/local/cuda/targets/x86_64-linux/lib '
    '&& ln -sfn "$CU135_LIB" /usr/local/cuda/lib64\'',
    "mkdir -p "
    "/workspace/kern/target/release "
    "/workspace/kern/crates/kern-py "
    "/workspace/weights "
    "/workspace/baselines/inkling-2-mini "
    "/workspace/kern-inkling-2mini/results",
    "ln -sfn /data/bin/kern /workspace/kern/target/release/kern "
    "&& ln -sfn /data/bin/lib_kern.so /workspace/kern/target/release/lib_kern.so "
    "&& ln -sfn /data/kern-py/python /workspace/kern/crates/kern-py/python "
    "&& ln -sfn /data/weights/Inkling-2-Mini /workspace/weights/Inkling-2-Mini "
    "&& ln -sfn /data/smoke_reference.parquet /workspace/baselines/inkling-2-mini/smoke_reference.parquet "
    "&& ln -sfn /data/results/kv-scales.json /workspace/kern-inkling-2mini/results/kv-scales.json",
]

gpu_image = (
    modal.Image.from_registry("ubuntu:24.04", add_python="3.12")
    .apt_install(
        "ca-certificates",
        "gcc",
        "g++",
        "libc6-dev",
        "libssl3",
        "libstdc++6",
        "libgomp1",
    )
    .uv_pip_install(
        "torch==2.14.0",
        "triton==3.8.0",
        "safetensors>=0.8.0",
        "numpy>=2.0.0",
        "pyarrow>=18.0.0",
        "pytest>=8.0.0",
    )
    .run_commands(*_IMAGE_SETUP_CMDS)
    .env({"PYTHONUNBUFFERED": "1"})
)


def _format_bench_ops(report: dict[str, Any]) -> list[str]:
    seconds: dict[tuple[str, str], float] = {}
    for s in report.get("scenarios", []):
        weight, kind = s["scenario"]["weight"], s["program"]
        for c in s.get("calls", []):
            key = (c["op"], kind)
            seconds[key] = seconds.get(key, 0.0) + weight * c["in_program"]["stats"]["p50"] / 1e6
    total = sum(seconds.values())
    cost_s = report.get("cost", {}).get("total_us", 0.0) / 1e6
    rows = sorted(seconds.items(), key=lambda kv: -kv[1])
    lines = ["", f"{'op':28s} {'program':8s} {'GPU s':>9s} {'share':>6s}"]
    for (op, kind), sec in rows:
        if total > 0 and sec / total >= 0.002:
            lines.append(f"{op:28s} {kind:8s} {sec:9.1f} {100 * sec / total:5.1f}%")
    lines.append(f"total of calls {total:.1f} s (cost {cost_s:.1f} s; the rest is gaps between kernels)")
    return lines


def _format_bench_summary(report: dict[str, Any], include_ops: bool = False) -> str:
    lines: list[str] = []
    for s in report.get("scenarios", []):
        sid = s["scenario"]["id"]
        p50_ms = s["graph"]["stats"]["p50"] / 1e3
        lines.append(f"{sid:36s} {p50_ms:9.3f} ms")
    cost_obj = report.get("cost", {})
    cost_s = cost_obj.get("total_us", 0.0) / 1e6
    dropped = cost_obj.get("dropped_calls", 0)
    by_op = cost_obj.get("by_op", [])[:6]
    ops_str = " · ".join(f"{o['op']} {100 * o['share']:.1f}%" for o in by_op)
    lines.append(f"cost {cost_s:.3f} s · dropped {dropped} · {ops_str}")
    if include_ops:
        lines.extend(_format_bench_ops(report))
    return "\n".join(lines) + "\n"


def _format_bench_ab(
    base_report: dict[str, Any],
    cand_report: dict[str, Any],
    include_ops: bool = False,
) -> str:
    sa = {s["scenario"]["id"]: s["graph"]["stats"]["p50"] / 1e3 for s in base_report.get("scenarios", [])}
    sb = {s["scenario"]["id"]: s["graph"]["stats"]["p50"] / 1e3 for s in cand_report.get("scenarios", [])}
    ca = base_report.get("cost", {}).get("total_us", 0.0) / 1e6
    cb = cand_report.get("cost", {}).get("total_us", 0.0) / 1e6
    da = base_report.get("cost", {}).get("dropped_calls", 0)
    db = cand_report.get("cost", {}).get("dropped_calls", 0)
    pct = lambda a, b: 100.0 * (b / a - 1.0) if a > 0 else 0.0

    common = [k for k in sa if k in sb]
    moved = sorted(((pct(sa[k], sb[k]), k) for k in common), key=lambda x: x[0])
    lines: list[str] = []
    for change, k in moved:
        if abs(change) >= 0.5:
            lines.append(f"{k:36s} {sa[k]:9.3f} -> {sb[k]:9.3f} ms  {change:+6.2f}%")
    if not lines:
        for change, k in moved:
            lines.append(f"{k:36s} {sa[k]:9.3f} -> {sb[k]:9.3f} ms  {change:+6.2f}%")

    worst_pct, worst_id = max(((pct(sa[k], sb[k]), k) for k in common), default=(0.0, "none"))
    cost_pct = pct(ca, cb)
    dropped = max(da, db)
    by_op = cand_report.get("cost", {}).get("by_op", [])[:6]
    ops_str = " · ".join(f"{o['op']} {100 * o['share']:.1f}%" for o in by_op)
    lines.append(f"worst {worst_id} {worst_pct:+.2f}%")
    lines.append(f"cost {ca:.3f} -> {cb:.3f} s ({cost_pct:+.2f}%) · dropped {dropped} · {ops_str}")
    lines.append(
        json.dumps(
            {
                "a": round(ca, 3),
                "b": round(cb, 3),
                "change_pct": round(cost_pct, 3),
                "worst": worst_id,
                "worst_pct": round(worst_pct, 3),
                "dropped": dropped,
            }
        )
    )
    if include_ops:
        lines.extend(_format_bench_ops(cand_report))
    return "\n".join(lines) + "\n"


def _build_patched_config(src_cfg_path: Path, max_position_embeddings: int = 1048576) -> dict[str, Any]:
    """Replicate scripts/gen.py `_write_weights_overlay` config.json patch."""
    cfg = json.loads(src_cfg_path.read_text(encoding="utf-8"))
    resolved_max_pos = int(cfg.get("text_config", {}).get("model_max_length", max_position_embeddings))
    cfg.setdefault("max_position_embeddings", resolved_max_pos)
    unpadded_vocab = cfg.get("text_config", {}).get("unpadded_vocab_size")
    if unpadded_vocab is not None:
        cfg.setdefault("vocab_size", int(unpadded_vocab))
    return cfg


def _materialize_weights_1m(ckpt_dir: Path, overlay_dir: Path) -> Path:
    """Create a local symlinked weights overlay preserving 'inkling' in the path."""
    overlay_dir.mkdir(parents=True, exist_ok=True)
    for entry in sorted(ckpt_dir.iterdir(), key=lambda p: p.name):
        if entry.name.startswith(".") or entry.name == "config.json":
            continue
        dst = overlay_dir / entry.name
        if dst.is_symlink():
            if os.readlink(dst) == str(entry):
                continue
            dst.unlink()
        elif dst.is_file():
            dst.unlink()
        elif dst.is_dir():
            shutil.rmtree(dst)
        os.symlink(entry, dst)

    patched_cache = ckpt_dir.parent / "weights_inkling_1m_config.json"
    if patched_cache.exists():
        cfg_text = patched_cache.read_text(encoding="utf-8")
    else:
        cfg_text = json.dumps(_build_patched_config(ckpt_dir / "config.json"), indent=2) + "\n"
    (overlay_dir / "config.json").write_text(cfg_text, encoding="utf-8")
    return overlay_dir


@app.function(
    image=cpu_loader_image,
    cpu=4.0,
    memory=8192,
    timeout=1800,
    volumes={DATA_DIR: vol},
)
def populate_weights(force: bool = False) -> dict[str, Any]:
    """Download thinkingmachines/Inkling-2-Mini to persistent Volume on cheap CPU."""
    from huggingface_hub import snapshot_download

    t0 = time.monotonic()
    weights_dir = Path(DATA_DIR) / "weights" / "Inkling-2-Mini"
    marker = weights_dir / "model-00001-of-00001.safetensors"

    downloaded = False
    if force or not (marker.exists() and (weights_dir / "mtp.safetensors").exists()):
        weights_dir.mkdir(parents=True, exist_ok=True)
        snapshot_download(
            repo_id="thinkingmachines/Inkling-2-Mini",
            local_dir=str(weights_dir),
            ignore_patterns=[".cache/*", ".git/*", ".gitattributes"],
        )
        downloaded = True

    cfg = _build_patched_config(weights_dir / "config.json")
    patched_path = Path(DATA_DIR) / "weights" / "weights_inkling_1m_config.json"
    patched_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")

    # Ensure uploaded kern binary and shared library are executable
    for bin_name in ("kern", "lib_kern.so"):
        p = Path(DATA_DIR) / "bin" / bin_name
        if p.exists():
            os.chmod(p, 0o755)

    vol.commit()
    files = sorted(p.name for p in weights_dir.iterdir() if not p.name.startswith("."))
    total_bytes = sum(p.stat().st_size for p in weights_dir.rglob("*") if p.is_file())
    return {
        "downloaded": downloaded,
        "elapsed_s": round(time.monotonic() - t0, 2),
        "weights_dir": str(weights_dir),
        "file_count": len(files),
        "total_gib": round(total_bytes / (1024**3), 2),
        "patched_max_position_embeddings": cfg.get("max_position_embeddings"),
        "patched_vocab_size": cfg.get("vocab_size"),
    }


@app.cls(
    image=gpu_image,
    gpu=GPU_SPEC,
    cpu=4.0,
    memory=32768,
    scaledown_window=60,
    timeout=600,
    max_containers=4,
    volumes={DATA_DIR: vol.read_only()},
)
class Runner:
    @modal.enter()
    def setup(self) -> None:
        self.kern_bin = Path(DATA_DIR) / "bin" / "kern"
        self.weights_dir = Path(DATA_DIR) / "weights" / "Inkling-2-Mini"
        self.weights_1m_dir = Path("/tmp/weights_inkling_1m")
        self.smoke_ref = Path(DATA_DIR) / "smoke_reference.parquet"
        self.kv_scales = Path(DATA_DIR) / "results" / "kv-scales.json"

        # If Volume binary lacks +x bit, copy to /tmp and chmod +x
        if self.kern_bin.exists() and not os.access(self.kern_bin, os.X_OK):
            tmp_bin = Path("/tmp/kern_bin")
            tmp_bin.mkdir(parents=True, exist_ok=True)
            dst_kern = tmp_bin / "kern"
            shutil.copy2(self.kern_bin, dst_kern)
            os.chmod(dst_kern, 0o755)
            self.kern_bin = dst_kern
            os.unlink("/workspace/kern/target/release/kern")
            os.symlink(dst_kern, "/workspace/kern/target/release/kern")

        if self.weights_dir.exists() and (self.weights_dir / "config.json").exists():
            _materialize_weights_1m(self.weights_dir, self.weights_1m_dir)

        import torch

        self.gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"

    def _extract_tgz(self, payload_tgz: bytes, dest: Path) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(fileobj=io.BytesIO(payload_tgz), mode="r:gz") as tf:
            tf.extractall(path=dest, filter="data")

    def _base_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["KERN_RUN_GPU"] = "1"
        env["PYTHONUNBUFFERED"] = "1"
        env["KERN"] = str(self.kern_bin)
        env["KERN_BIN"] = str(self.kern_bin)
        env["KERN_LIB_SO"] = str(Path(DATA_DIR) / "bin" / "lib_kern.so")
        env["KERN_PY_ROOT"] = str(Path(DATA_DIR) / "kern-py" / "python")
        env["KERN_WEIGHTS"] = str(self.weights_dir)
        env["WEIGHTS"] = str(self.weights_dir)
        env["WEIGHTS_1M"] = str(self.weights_1m_dir)
        env["SMOKE_REF"] = str(self.smoke_ref)
        env["KV_SCALES"] = str(self.kv_scales)
        return env

    @modal.method()
    def probe(self, sample_cubin: bytes | None = None) -> str:
        import ctypes
        import traceback

        t0 = time.monotonic()
        try:
            import torch
            import triton

            smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
            kver = subprocess.run([str(self.kern_bin), "--version"], capture_output=True, text=True)

            cubin_load = "skipped"
            if sample_cubin:
                try:
                    torch.cuda.init()
                    _ = torch.empty(1, device="cuda")
                    libcuda = ctypes.CDLL("libcuda.so.1")
                    mod = ctypes.c_void_p()
                    rc = libcuda.cuModuleLoadData(ctypes.byref(mod), ctypes.c_char_p(sample_cubin))
                    if rc == 0:
                        libcuda.cuModuleUnload(mod)
                        cubin_load = "OK (cuModuleLoadData=0)"
                    else:
                        err_str = ctypes.c_char_p()
                        libcuda.cuGetErrorString(rc, ctypes.byref(err_str))
                        cubin_load = f"ERR rc={rc}: {err_str.value.decode() if err_str.value else '?'}"
                except Exception as exc:
                    cubin_load = f"EXC: {exc}"

            cap = list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else []
            vram = (
                round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
                if torch.cuda.is_available()
                else 0.0
            )
            out = {
                "gpu_name": f"{self.gpu_name}",
                "capability": [int(x) for x in cap],
                "vram_gib": float(vram),
                "torch": f"{torch.__version__}",
                "torch_cuda": f"{torch.version.cuda}",
                "triton": f"{triton.__version__}",
                "kern_version": f"{(kver.stdout + kver.stderr).strip()}",
                "kern_rc": int(kver.returncode),
                "cubin_load": f"{cubin_load}",
                "weights_ready": bool((self.weights_1m_dir / "model-00001-of-00001.safetensors").exists()),
                "smoke_ref_ready": bool(self.smoke_ref.exists()),
                "nvidia_smi": f"{smi.stdout}",
                "elapsed_s": float(round(time.monotonic() - t0, 2)),
            }
            return json.dumps(out)
        except Exception:
            return json.dumps({"error": traceback.format_exc(), "kern_rc": 1})

    @modal.method()
    def run_unit(self, payload_tgz: bytes, test_args: list[str]) -> dict[str, Any]:
        t0 = time.monotonic()
        work = Path(tempfile.mkdtemp(prefix=f"unit_{uuid.uuid4().hex[:8]}_"))
        try:
            self._extract_tgz(payload_tgz, work)
            kv_dst = work / "results" / "kv-scales.json"
            if not kv_dst.exists() and self.kv_scales.exists():
                kv_dst.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(self.kv_scales, kv_dst)

            cmd = [sys.executable, "-m", "unittest", "-v", *test_args]
            proc = subprocess.run(cmd, cwd=work, env=self._base_env(), capture_output=True, text=True)
            return {
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "elapsed_s": round(time.monotonic() - t0, 2),
                "gpu_name": self.gpu_name,
            }
        finally:
            shutil.rmtree(work, ignore_errors=True)

    @modal.method()
    def run_e2e(self, payload_tgz: bytes) -> dict[str, Any]:
        t0 = time.monotonic()
        work = Path(tempfile.mkdtemp(prefix=f"e2e_{uuid.uuid4().hex[:8]}_"))
        try:
            bundle = work / "bundle"
            self._extract_tgz(payload_tgz, bundle)
            if not self.weights_1m_dir.exists():
                if not self.weights_dir.exists():
                    return {
                        "returncode": 2,
                        "stdout": "",
                        "stderr": "Volume weights missing! Run: loop/onmodal populate-weights\n",
                        "elapsed_s": round(time.monotonic() - t0, 2),
                        "gpu_name": self.gpu_name,
                    }
                _materialize_weights_1m(self.weights_dir, self.weights_1m_dir)

            out_parquet = work / "e2e_out.parquet"
            cmd = [
                str(self.kern_bin),
                "test",
                "e2e-logits",
                "--manifest",
                str(bundle / "model.json"),
                "--kernels",
                str(bundle / "kernels"),
                "--weights",
                str(self.weights_1m_dir),
                "--reference",
                str(self.smoke_ref),
                "--gpu",
                "0",
                "--out",
                str(out_parquet),
            ]
            proc = subprocess.run(cmd, cwd=work, env=self._base_env(), capture_output=True, text=True)
            return {
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "elapsed_s": round(time.monotonic() - t0, 2),
                "gpu_name": self.gpu_name,
            }
        finally:
            shutil.rmtree(work, ignore_errors=True)

    @modal.method()
    def run_check(self, payload_tgz: bytes, capacity: int = 8192) -> dict[str, Any]:
        """Run `kern test` numerics gate comparing cand vs base bundle on the same RTX PRO 6000."""
        t0 = time.monotonic()
        work = Path(tempfile.mkdtemp(prefix=f"check_{uuid.uuid4().hex[:8]}_"))
        try:
            self._extract_tgz(payload_tgz, work)
            if not self.weights_1m_dir.exists():
                _materialize_weights_1m(self.weights_dir, self.weights_1m_dir)

            base_dir = work / "base"
            cand_dir = work / "cand"
            merged_dir = work / "merged"
            (merged_dir / "kernels").mkdir(parents=True, exist_ok=True)
            for cubin in (base_dir / "kernels").glob("*.cubin"):
                shutil.copyfile(cubin, merged_dir / "kernels" / cubin.name)

            base_m = json.loads((base_dir / "model.json").read_text())
            cand_m = json.loads((cand_dir / "model.json").read_text())
            changed = {
                k
                for k, v in cand_m["modules"].items()
                if base_m["modules"].get(k, {}).get("sha256") != v["sha256"]
            }
            for k in changed:
                src = cand_dir / "kernels" / cand_m["modules"][k]["source"]
                dst_name = f"{k}+candidate.cubin"
                shutil.copyfile(src, merged_dir / "kernels" / dst_name)
                cand_m["modules"][k]["source"] = dst_name

            cand_m["modules"] = {
                (f"{k}+candidate" if k in changed else k): v for k, v in cand_m["modules"].items()
            }

            def _rename(obj: Any) -> None:
                if isinstance(obj, dict):
                    if "module" in obj and obj["module"] in changed:
                        obj["module"] = f"{obj['module']}+candidate"
                    for val in obj.values():
                        _rename(val)
                elif isinstance(obj, list):
                    for val in obj:
                        _rename(val)

            _rename(cand_m["ops"])
            (merged_dir / "model.json").write_text(json.dumps(cand_m))
            header = f"modules changed: {', '.join(sorted(changed)) or 'none'}\n"

            out_json = work / "check.json"
            cmd = [
                str(self.kern_bin),
                "test",
                "--capacity",
                str(int(capacity)),
                "--reference",
                str(base_dir / "model.json"),
                "--manifest",
                str(merged_dir / "model.json"),
                "--kernels",
                str(merged_dir / "kernels"),
                "--weights",
                str(self.weights_1m_dir),
                "--tokenizer",
                str(self.weights_1m_dir / "tokenizer.json"),
                "--gpu",
                "0",
                "--out",
                str(out_json),
            ]
            proc = subprocess.run(cmd, cwd=work, env=self._base_env(), capture_output=True, text=True)
            return {
                "returncode": proc.returncode,
                "stdout": header + proc.stdout,
                "stderr": proc.stderr,
                "elapsed_s": round(time.monotonic() - t0, 2),
                "gpu_name": self.gpu_name,
            }
        finally:
            shutil.rmtree(work, ignore_errors=True)

    @modal.method()
    def run_bench(
        self,
        payload_tgz: bytes,
        return_json: bool = False,
        has_base: bool = False,
        show_ops: bool = False,
    ) -> dict[str, Any]:
        """Run `kern bench` on the warm RTX PRO 6000 container (single bundle or same-GPU A/B)."""
        t0 = time.monotonic()
        work = Path(tempfile.mkdtemp(prefix=f"bench_{uuid.uuid4().hex[:8]}_"))
        try:
            self._extract_tgz(payload_tgz, work)
            if not self.weights_1m_dir.exists():
                if not self.weights_dir.exists():
                    return {
                        "returncode": 2,
                        "stdout": "",
                        "stderr": "Volume weights missing! Run: loop/onmodal populate-weights\n",
                        "elapsed_s": round(time.monotonic() - t0, 2),
                        "gpu_name": self.gpu_name,
                    }
                _materialize_weights_1m(self.weights_dir, self.weights_1m_dir)

            workload_path = work / "workload.toml"

            def _run_one(bundle_dir: Path, out_json: Path) -> subprocess.CompletedProcess[str]:
                cmd = [
                    str(self.kern_bin),
                    "bench",
                    "--manifest",
                    str(bundle_dir / "model.json"),
                    "--kernels",
                    str(bundle_dir / "kernels"),
                    "--weights",
                    str(self.weights_1m_dir),
                    "--tokenizer",
                    str(self.weights_1m_dir / "tokenizer.json"),
                    "--gpu",
                    "0",
                    "--workload",
                    str(workload_path),
                    "--out",
                    str(out_json),
                ]
                return subprocess.run(cmd, cwd=work, env=self._base_env(), capture_output=True, text=True)

            cand_json = work / "bench_out.json"
            if has_base:
                base_json = work / "base_out.json"
                proc_a = _run_one(work / "base", base_json)
                if proc_a.returncode != 0:
                    return {
                        "returncode": proc_a.returncode,
                        "stdout": proc_a.stdout,
                        "stderr": f"[onmodal bench] base failed:\n{proc_a.stderr}",
                        "elapsed_s": round(time.monotonic() - t0, 2),
                        "gpu_name": self.gpu_name,
                    }
                proc_b = _run_one(work / "cand", cand_json)
                if proc_b.returncode != 0:
                    return {
                        "returncode": proc_b.returncode,
                        "stdout": proc_b.stdout,
                        "stderr": f"[onmodal bench] cand failed:\n{proc_b.stderr}",
                        "elapsed_s": round(time.monotonic() - t0, 2),
                        "gpu_name": self.gpu_name,
                    }
                base_report = json.loads(base_json.read_text(encoding="utf-8"))
                cand_report = json.loads(cand_json.read_text(encoding="utf-8"))
                summary = _format_bench_ab(base_report, cand_report, include_ops=show_ops)
            else:
                proc = _run_one(work / "bundle", cand_json)
                if proc.returncode != 0:
                    return {
                        "returncode": proc.returncode,
                        "stdout": proc.stdout,
                        "stderr": proc.stderr,
                        "elapsed_s": round(time.monotonic() - t0, 2),
                        "gpu_name": self.gpu_name,
                    }
                cand_report = json.loads(cand_json.read_text(encoding="utf-8"))
                summary = _format_bench_summary(cand_report, include_ops=show_ops)

            res: dict[str, Any] = {
                "returncode": 0,
                "stdout": summary,
                "stderr": "",
                "elapsed_s": round(time.monotonic() - t0, 2),
                "gpu_name": self.gpu_name,
            }
            if return_json:
                res["bench_json_gz"] = gzip.compress(cand_json.read_bytes(), compresslevel=1)
            return res
        finally:
            shutil.rmtree(work, ignore_errors=True)

    @modal.method()
    def run_exec(self, payload_tgz: bytes, shell_cmd: str) -> dict[str, Any]:
        t0 = time.monotonic()
        work = Path(tempfile.mkdtemp(prefix=f"exec_{uuid.uuid4().hex[:8]}_"))
        try:
            if payload_tgz:
                self._extract_tgz(payload_tgz, work)
            kv_dst = work / "results" / "kv-scales.json"
            if not kv_dst.exists() and self.kv_scales.exists():
                kv_dst.parent.mkdir(parents=True, exist_ok=True)
                os.symlink(self.kv_scales, kv_dst)

            proc = subprocess.run(
                ["bash", "-c", shell_cmd],
                cwd=work,
                env=self._base_env(),
                capture_output=True,
                text=True,
            )
            return {
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "elapsed_s": round(time.monotonic() - t0, 2),
                "gpu_name": self.gpu_name,
            }
        finally:
            shutil.rmtree(work, ignore_errors=True)


# ---------------------------------------------------------------------------
# Local client packaging & CLI (invoked by `loop/onmodal`)
# ---------------------------------------------------------------------------

_SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    "__pycache__",
    ".ruff_cache",
    ".pytest_cache",
    "third_party",
    "node_modules",
    "weights_inkling_1m",
}

_SKIP_SUFFIXES = {".safetensors", ".pt", ".pth", ".ckpt", ".parquet", ".pyc", ".pyo"}


def _pack_tgz(root: Path, rel_paths: list[Path]) -> bytes:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=1) as gz:
        with tarfile.open(fileobj=gz, mode="w", dereference=True) as tf:
            seen: set[str] = set()
            for rel in rel_paths:
                arcname = rel.as_posix()
                if arcname in seen:
                    continue
                full = root / rel
                if not full.exists() or not full.is_file():
                    continue
                seen.add(arcname)
                tf.add(full, arcname=arcname, recursive=False)
    data = buf.getvalue()
    if len(data) > MAX_PAYLOAD_BYTES:
        raise RuntimeError(
            f"Upload payload too large ({len(data) / 1048576:.1f} MiB > "
            f"{MAX_PAYLOAD_BYTES / 1048576:.0f} MiB cap). Narrow <local_dir> or exclude large artifacts."
        )
    return data


def _pack_check_tgz(cand_dir: Path, base_dir: Path) -> bytes:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=1) as gz:
        with tarfile.open(fileobj=gz, mode="w", dereference=True) as tf:
            for prefix, bdir in (("base", base_dir), ("cand", cand_dir)):
                for rel in _collect_e2e_files(bdir):
                    tf.add(bdir / rel, arcname=f"{prefix}/{rel.as_posix()}", recursive=False)
    data = buf.getvalue()
    if len(data) > MAX_PAYLOAD_BYTES:
        raise RuntimeError(f"Check payload too large ({len(data) / 1048576:.1f} MiB)")
    return data


def _pack_bench_tgz(bundle_dir: Path, workload_path: Path, base_dir: Path | None = None) -> bytes:
    if not workload_path.is_file():
        raise FileNotFoundError(f"Missing workload TOML: {workload_path}")
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb", compresslevel=1) as gz:
        with tarfile.open(fileobj=gz, mode="w", dereference=True) as tf:
            tf.add(workload_path, arcname="workload.toml", recursive=False)
            if base_dir is not None:
                for prefix, bdir in (("base", base_dir), ("cand", bundle_dir)):
                    for rel in _collect_e2e_files(bdir):
                        tf.add(bdir / rel, arcname=f"{prefix}/{rel.as_posix()}", recursive=False)
            else:
                for rel in _collect_e2e_files(bundle_dir):
                    tf.add(bundle_dir / rel, arcname=f"bundle/{rel.as_posix()}", recursive=False)
    data = buf.getvalue()
    if len(data) > MAX_PAYLOAD_BYTES:
        raise RuntimeError(f"Bench payload too large ({len(data) / 1048576:.1f} MiB)")
    return data


def _collect_unit_files(worktree: Path) -> list[Path]:
    files: list[Path] = []
    for sub in ("source", "scripts", "tests"):
        d = worktree / sub
        if not d.exists():
            continue
        for p in sorted(d.rglob("*")):
            if any(part in _SKIP_DIR_NAMES for part in p.relative_to(worktree).parts):
                continue
            if p.is_file() and p.suffix not in _SKIP_SUFFIXES:
                files.append(p.relative_to(worktree))

    cubin_dir = worktree / "build" / "sm_120"
    if cubin_dir.exists():
        for pat in ("*.cubin", "*.json"):
            for p in sorted(cubin_dir.glob(pat)):
                files.append(p.relative_to(worktree))

    for top in ("kernels.toml", "precision.toml", "pyproject.toml", "workload.toml"):
        p = worktree / top
        if p.is_file():
            files.append(p.relative_to(worktree))

    kv = worktree / "results" / "kv-scales.json"
    if kv.is_file():
        files.append(kv.relative_to(worktree))
    return files


def _collect_e2e_files(bundle_dir: Path) -> list[Path]:
    model_json = bundle_dir / "model.json"
    kernels_dir = bundle_dir / "kernels"
    if not model_json.is_file():
        raise FileNotFoundError(f"Missing {model_json}")
    if not kernels_dir.is_dir():
        raise FileNotFoundError(f"Missing {kernels_dir}")
    files = [Path("model.json")]
    cubins = sorted(kernels_dir.glob("*.cubin"))
    if not cubins:
        raise FileNotFoundError(f"No .cubin files found in {kernels_dir}")
    for p in cubins:
        files.append(p.relative_to(bundle_dir))
    return files


def _collect_exec_files(local_dir: Path) -> list[Path]:
    is_repo = (local_dir / "kernels.toml").is_file() and (local_dir / "scripts").is_dir()
    skip_top = {"out", "state", "results"} if is_repo else set()
    files: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(local_dir):
        rel_dir = Path(dirpath).relative_to(local_dir)
        dirnames[:] = sorted(
            d
            for d in dirnames
            if d not in _SKIP_DIR_NAMES and not (rel_dir == Path(".") and d in skip_top)
        )
        for fn in sorted(filenames):
            p = Path(dirpath) / fn
            if p.suffix in _SKIP_SUFFIXES:
                continue
            if p.is_file() and not p.is_symlink():
                files.append(p.relative_to(local_dir))
    if is_repo:
        kv = local_dir / "results" / "kv-scales.json"
        if kv.is_file():
            files.append(kv.relative_to(local_dir))
    return files


def _print_result(res: dict[str, Any], upload_bytes: int, wall_s: float) -> int:
    if res.get("stdout"):
        sys.stdout.write(res["stdout"])
        sys.stdout.flush()
    if res.get("stderr"):
        sys.stderr.write(res["stderr"])
        sys.stderr.flush()
    sys.stderr.write(
        f"[onmodal] gpu={res.get('gpu_name', '?')} "
        f"upload={upload_bytes / 1048576:.2f}MB "
        f"remote={res.get('elapsed_s', '?')}s "
        f"wall={wall_s:.2f}s rc={res.get('returncode', 1)}\n"
    )
    return int(res.get("returncode", 1))


def _cli() -> int:
    ap = argparse.ArgumentParser(prog="loop/onmodal", description="Modal RTX PRO 6000 runner")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_probe = sub.add_parser("probe", help="Verify Modal RTX PRO 6000 driver, kern, and cubin load")
    p_probe.add_argument("--cubin", type=Path, default=None, help="Optional .cubin file to test cuModuleLoadData")

    p_pop = sub.add_parser("populate-weights", help="Populate persistent Volume weights via cheap CPU function")
    p_pop.add_argument("--force", action="store_true", help="Re-download even if weights exist")

    p_unit = sub.add_parser("unit", help="Run unittest on Modal RTX PRO 6000")
    p_unit.add_argument("worktree_dir", type=Path)
    p_unit.add_argument("test_args", nargs="+")

    p_e2e = sub.add_parser("e2e", help="Run kern test e2e-logits on Modal RTX PRO 6000")
    p_e2e.add_argument("bundle_dir", type=Path)

    p_chk = sub.add_parser("check", help="Run kern test cand vs base bundle on Modal RTX PRO 6000")
    p_chk.add_argument("cand_bundle", type=Path)
    p_chk.add_argument("base_bundle", type=Path, nargs="?", default=None)
    p_chk.add_argument("--capacity", type=int, default=int(os.environ.get("KERN_CHECK_CAPACITY", "8192")))

    p_bench = sub.add_parser("bench", help="Run kern bench on Modal RTX PRO 6000")
    p_bench.add_argument("bundle_dir", type=Path)
    p_bench.add_argument("workload_toml", type=Path, nargs="?", default=None)
    p_bench.add_argument("--out", type=Path, default=None, help="Write full kern bench JSON report locally")
    p_bench.add_argument("--base", type=Path, default=None, help="Optional base bundle for same-GPU A/B comparison")
    p_bench.add_argument("--ops", action="store_true", help="Also print per-op GPU-seconds breakdown")

    p_exec = sub.add_parser("exec", help="Upload small local_dir and run shell command on Modal RTX PRO 6000")
    p_exec.add_argument("local_dir", type=Path)
    p_exec.add_argument("shell_cmd", type=str)

    args = ap.parse_args()
    t0 = time.monotonic()

    if args.cmd == "populate-weights":
        fn = modal.Function.from_name(APP_NAME, "populate_weights")
        res = fn.remote(force=args.force)
        print(json.dumps(res, indent=2))
        return 0

    runner_cls = modal.Cls.from_name(APP_NAME, "Runner")
    runner = runner_cls()

    if args.cmd == "probe":
        cubin_bytes = args.cubin.read_bytes() if args.cubin and args.cubin.is_file() else None
        raw = runner.probe.remote(sample_cubin=cubin_bytes)
        res = json.loads(raw) if isinstance(raw, str) else raw
        smi = res.pop("nvidia_smi", "")
        print(json.dumps(res, indent=2))
        if smi:
            print(smi)
        sys.stderr.write(f"[onmodal] probe wall={time.monotonic() - t0:.2f}s\n")
        return 0 if res.get("kern_rc") == 0 else 1

    if args.cmd == "unit":
        wt = args.worktree_dir.resolve()
        files = _collect_unit_files(wt)
        payload = _pack_tgz(wt, files)
        res = runner.run_unit.remote(payload, args.test_args)
        return _print_result(res, len(payload), time.monotonic() - t0)

    if args.cmd == "e2e":
        bd = args.bundle_dir.resolve()
        files = _collect_e2e_files(bd)
        payload = _pack_tgz(bd, files)
        res = runner.run_e2e.remote(payload)
        return _print_result(res, len(payload), time.monotonic() - t0)

    if args.cmd == "check":
        cand = args.cand_bundle.resolve()
        if args.base_bundle is not None:
            base = args.base_bundle.resolve()
        else:
            base = (Path(__file__).resolve().parents[1] / "state" / "main" / "bundle").resolve()
        payload = _pack_check_tgz(cand, base)
        res = runner.run_check.remote(payload, capacity=args.capacity)
        return _print_result(res, len(payload), time.monotonic() - t0)

    if args.cmd == "bench":
        bd = args.bundle_dir.resolve()
        if args.workload_toml is not None:
            wl = args.workload_toml.resolve()
        else:
            wl = (Path(__file__).resolve().parents[1] / "workload.toml").resolve()
        base = args.base.resolve() if args.base is not None else None
        payload = _pack_bench_tgz(bd, wl, base_dir=base)
        res = runner.run_bench.remote(
            payload,
            return_json=args.out is not None,
            has_base=base is not None,
            show_ops=args.ops,
        )
        if args.out is not None and res.get("bench_json_gz"):
            out_path = args.out.resolve()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            gz_bytes = res["bench_json_gz"]
            raw_json = gzip.decompress(gz_bytes)
            out_path.write_bytes(raw_json)
            sys.stderr.write(
                f"[onmodal] wrote {args.out} ({len(raw_json) / 1048576:.2f}MB, gz={len(gz_bytes) / 1048576:.2f}MB)\n"
            )
        return _print_result(res, len(payload), time.monotonic() - t0)

    if args.cmd == "exec":
        ld = args.local_dir.resolve()
        files = _collect_exec_files(ld)
        payload = _pack_tgz(ld, files)
        res = runner.run_exec.remote(payload, args.shell_cmd)
        return _print_result(res, len(payload), time.monotonic() - t0)

    return 2


if __name__ == "__main__":
    raise SystemExit(_cli())
