#!/usr/bin/env python3
"""Unified CLI runner for the 4-tier opaque-box kern-inkling-2mini E2E test suite."""

from __future__ import annotations

import argparse
import json
import os
import py_compile
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

_E2E_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _E2E_DIR.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.e2e.conftest import (  # noqa: E402
    DEFAULT_BASE_URL,
    DEFAULT_BASELINE_DIR,
    DEFAULT_KERN_BIN,
    DEFAULT_KERNELS,
    DEFAULT_MANIFEST,
    DEFAULT_TRACE_URL,
    DEFAULT_WEIGHTS,
)
from tests.e2e.fixtures_data import discover_baselines, validate_weights_configs  # noqa: E402

TIER_FILES: dict[str, Path] = {
    "1": _E2E_DIR / "test_tier1_features.py",
    "2": _E2E_DIR / "test_tier2_boundaries.py",
    "3": _E2E_DIR / "test_tier3_combinations.py",
    "4": _E2E_DIR / "test_tier4_workloads.py",
    "selftest": _E2E_DIR / "test_suite_selftest.py",
}

TIER_MIN_COUNTS: dict[str, int] = {
    "1": 65,
    "2": 65,
    "3": 13,
    "4": 7,
}


def _collect_tier_tests(tier_file: Path, env: dict[str, str]) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(tier_file),
        "--collect-only",
        "-q",
        "-c",
        str(_E2E_DIR / "pytest.ini"),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(_E2E_DIR))
    if proc.returncode != 0:
        raise RuntimeError(
            f"pytest --collect-only failed for {tier_file.name} (rc={proc.returncode}):\n"
            f"{proc.stdout}\n{proc.stderr}"
        )
    node_ids: list[str] = []
    for line in proc.stdout.splitlines():
        stripped = line.strip()
        if "::" in stripped and not stripped.startswith("="):
            node_ids.append(stripped)
    return node_ids


def _parse_junit_details(junit_path: Path) -> tuple[dict[str, int], int]:
    if not junit_path.exists():
        return {}, 0
    root = ET.parse(junit_path).getroot()
    suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
    totals = {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}
    tier4_skipped = 0
    for suite in suites:
        for key in totals:
            totals[key] += int(suite.attrib.get(key, 0))
        for tc in suite.findall("testcase"):
            cname = tc.attrib.get("classname") or ""
            if "test_tier4_workloads" in cname and tc.find("skipped") is not None:
                tier4_skipped += 1
    totals["passed"] = max(0, totals["tests"] - totals["failures"] - totals["errors"] - totals["skipped"])
    return totals, tier4_skipped


def _parse_junit_counts(junit_path: Path) -> dict[str, int]:
    totals, _ = _parse_junit_details(junit_path)
    return totals


def _run_dry_run(args: argparse.Namespace) -> int:
    started = time.monotonic()
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{_REPO_ROOT}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(_REPO_ROOT)
    env["KERN_E2E_IN_DRY_RUN"] = "1"

    # 1. py_compile all .py files in tests/e2e/
    py_files = sorted(_E2E_DIR.glob("*.py"))
    compiled: list[str] = []
    for py_file in py_files:
        py_compile.compile(str(py_file), doraise=True)
        compiled.append(py_file.name)

    # 2. Run CPU-only selftests
    selftest_cmd = [
        sys.executable,
        "-m",
        "pytest",
        str(TIER_FILES["selftest"]),
        "-v",
        "-c",
        str(_E2E_DIR / "pytest.ini"),
    ]
    if args.junit_xml:
        junit_path = Path(args.junit_xml)
        junit_path.parent.mkdir(parents=True, exist_ok=True)
        selftest_cmd.extend(["--junitxml", str(junit_path)])
    selftest_proc = subprocess.run(selftest_cmd, capture_output=True, text=True, env=env, cwd=str(_E2E_DIR))
    print(selftest_proc.stdout, end="")
    if selftest_proc.stderr:
        print(selftest_proc.stderr, file=sys.stderr, end="")

    # 3. Collect all 4 GPU tiers and verify minimum coverage counts
    collected: dict[str, int] = {}
    collection_errors: list[str] = []
    for tier_key, min_count in TIER_MIN_COUNTS.items():
        nodes = _collect_tier_tests(TIER_FILES[tier_key], env=env)
        collected[f"tier{tier_key}"] = len(nodes)
        if len(nodes) < min_count:
            collection_errors.append(
                f"Tier {tier_key} collected {len(nodes)} tests, expected >= {min_count}"
            )

    # 4. Validate weights config if present
    weights_info = validate_weights_configs(Path(args.weights))

    # 5. Baseline discovery status
    baselines = discover_baselines(
        baseline_dir=Path(args.baseline_dir),
        explicit_parquet=Path(args.baseline_parquet) if args.baseline_parquet else None,
        explicit_metrics=Path(args.baseline_metrics) if args.baseline_metrics else None,
    )

    ok = selftest_proc.returncode == 0 and not collection_errors
    report: dict[str, Any] = {
        "mode": "dry-run",
        "ok": ok,
        "duration_s": round(time.monotonic() - started, 3),
        "compiled_files": compiled,
        "selftest_returncode": selftest_proc.returncode,
        "collected_counts": collected,
        "min_thresholds": {f"tier{k}": v for k, v in TIER_MIN_COUNTS.items()},
        "collection_errors": collection_errors,
        "weights_check": weights_info,
        "baseline_discovery": {
            "baseline_dir": str(baselines.baseline_dir),
            "recording_parquet": str(baselines.recording_parquet) if baselines.recording_parquet else None,
            "questions_parquet": str(baselines.questions_parquet) if baselines.questions_parquet else None,
            "metrics_json": str(baselines.metrics_json) if baselines.metrics_json else None,
            "warnings": baselines.warnings,
        },
        "manifest_exists": Path(args.manifest).exists(),
    }
    if args.junit_xml:
        report["junit_summary"] = _parse_junit_counts(Path(args.junit_xml))

    if args.json_report:
        out_path = Path(args.json_report)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({"dry_run_ok": ok, "collected_counts": collected, "errors": collection_errors}))
    return 0 if ok else 1


def _run_live(args: argparse.Namespace, extra_pytest_args: list[str]) -> int:
    started = time.monotonic()
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{_REPO_ROOT}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(_REPO_ROOT)

    if args.tier == "all":
        targets = [str(TIER_FILES[k]) for k in ("1", "2", "3", "4")]
    else:
        targets = [str(TIER_FILES[args.tier])]

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        *targets,
        "-c",
        str(_E2E_DIR / "pytest.ini"),
        "--base-url",
        args.base_url,
        "--baseline-dir",
        args.baseline_dir,
        "--manifest",
        args.manifest,
        "--kernels",
        args.kernels,
        "--weights",
        args.weights,
        "--kern-bin",
        args.kern_bin,
        "--max-ctx-probe",
        str(args.max_ctx_probe),
        "--gpu",
        str(args.gpu),
    ]
    if args.trace_url:
        cmd.extend(["--trace-url", args.trace_url])
    if args.baseline_parquet:
        cmd.extend(["--baseline-parquet", args.baseline_parquet])
    if args.baseline_metrics:
        cmd.extend(["--baseline-metrics", args.baseline_metrics])
    if args.model_id:
        cmd.extend(["--model-id", args.model_id])
    if args.kl_limit:
        cmd.extend(["--kl-limit", args.kl_limit])

    mode = getattr(args, "mode", "all")
    if args.tier != "selftest":
        if mode == "http":
            cmd.extend(["-m", "not cli"])
        elif mode == "cli":
            cmd.extend(["-m", "cli and not hybrid", "--strict-cli"])

    cmd.extend(extra_pytest_args)

    def _exec_and_report(junit_path: Path) -> int:
        junit_path.parent.mkdir(parents=True, exist_ok=True)
        run_cmd = [*cmd, "--junitxml", str(junit_path)]
        proc = subprocess.run(run_cmd, env=env, cwd=str(_E2E_DIR))
        summary, tier4_skipped = _parse_junit_details(junit_path)
        tier4_gate_failed = args.tier in ("4", "all") and tier4_skipped > 0
        effective_rc = 1 if (tier4_gate_failed and proc.returncode == 0) else proc.returncode
        if tier4_gate_failed:
            print(
                f"ERROR: Tier 4 acceptance gate failed: {tier4_skipped} Tier 4 test(s) were skipped.",
                file=sys.stderr,
            )

        if args.json_report:
            report = {
                "mode": "live",
                "phase_mode": mode,
                "tier": args.tier,
                "base_url": args.base_url,
                "returncode": effective_rc,
                "pytest_returncode": proc.returncode,
                "ok": effective_rc == 0,
                "tier4_skipped": tier4_skipped,
                "duration_s": round(time.monotonic() - started, 3),
                "summary": summary,
            }
            out_path = Path(args.json_report)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

        return effective_rc

    if args.junit_xml:
        return _exec_and_report(Path(args.junit_xml))

    with tempfile.TemporaryDirectory(prefix="kern_e2e_junit_") as tmp_dir:
        return _exec_and_report(Path(tmp_dir) / "results.junit.xml")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Live kern-serve OpenAI URL")
    p.add_argument("--trace-url", default=DEFAULT_TRACE_URL, help="Optional kern-serve --trace-port URL")
    p.add_argument(
        "--tier",
        choices=("1", "2", "3", "4", "all", "selftest"),
        default="all",
        help="Test tier to execute (default: all)",
    )
    p.add_argument(
        "--mode",
        choices=("all", "http", "cli"),
        default="all",
        help="Phase filter: all, http (pure HTTP against kern-serve), or cli (pure kern CLI without serve VRAM contention)",
    )
    p.add_argument("--baseline-dir", default=DEFAULT_BASELINE_DIR, help="M0 baseline directory")
    p.add_argument("--baseline-parquet", default="", help="Explicit M0 baseline Parquet path")
    p.add_argument("--baseline-metrics", default="", help="Explicit M0 baseline metrics JSON path")
    p.add_argument("--manifest", default=DEFAULT_MANIFEST, help="Manifest model.json path")
    p.add_argument("--kernels", default=DEFAULT_KERNELS, help="SM120 cubins directory")
    p.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Inkling-2-Mini weights directory")
    p.add_argument("--kern-bin", default=DEFAULT_KERN_BIN, help="Path to kern CLI binary")
    p.add_argument("--model-id", default="", help="Override served model ID")
    p.add_argument("--kl-limit", default="", help="Optional KL threshold (nats) for e2e-logits")
    p.add_argument("--max-ctx-probe", type=int, default=262144, help="Max Tier 4 context length probe")
    p.add_argument("--gpu", type=int, default=0, help="CUDA device ordinal for kern CLI calls")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="CPU-only validation: py_compile, selftests, pytest --collect-only counts, config validation",
    )
    p.add_argument("--junit-xml", default="", help="Write JUnit XML report to this path")
    p.add_argument("--json-report", default="", help="Write structured JSON report to this path")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)
    if args.dry_run:
        return _run_dry_run(args)
    return _run_live(args, extra)


if __name__ == "__main__":
    raise SystemExit(main())
