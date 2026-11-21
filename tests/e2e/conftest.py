"""Pytest CLI options and session/function fixtures for the kern-inkling-2mini E2E suite."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tests.e2e.client import KernCLI, KernClient  # noqa: E402
from tests.e2e.fixtures_data import BaselineArtifacts, discover_baselines  # noqa: E402

DEFAULT_BASE_URL = os.environ.get("KERN_E2E_BASE_URL", "http://127.0.0.1:8000")
DEFAULT_TRACE_URL = os.environ.get("KERN_E2E_TRACE_URL", "")
DEFAULT_BASELINE_DIR = "/workspace/baselines/inkling-2-mini"
DEFAULT_MANIFEST = "/workspace/kern-inkling-2mini/out/inkling-2-mini/model.json"
DEFAULT_KERNELS = "/workspace/kern-inkling-2mini/build/sm_120"
DEFAULT_WEIGHTS = "/workspace/weights/Inkling-2-Mini"
DEFAULT_KERN_BIN = "/workspace/kern/target/release/kern"


@dataclass(frozen=True)
class E2EConfig:
    base_url: str
    trace_url: str | None
    baseline_dir: Path
    baseline_parquet: Path | None
    baseline_metrics: Path | None
    manifest: Path
    kernels: Path
    weights: Path
    kern_bin: Path
    model_id: str | None
    kl_limit: float | None
    max_ctx_probe: int
    gpu: int
    strict_cli: bool = False


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("kern-e2e", "kern-inkling-2mini E2E suite options")
    group.addoption("--base-url", default=DEFAULT_BASE_URL, help="Live kern-serve OpenAI base URL")
    group.addoption("--trace-url", default=DEFAULT_TRACE_URL, help="Optional kern-serve --trace-port URL")
    group.addoption("--baseline-dir", default=DEFAULT_BASELINE_DIR, help="M0 baseline directory")
    group.addoption("--baseline-parquet", default="", help="Explicit M0 recording Parquet path")
    group.addoption("--baseline-metrics", default="", help="Explicit M0 metrics JSON path")
    group.addoption("--manifest", default=DEFAULT_MANIFEST, help="Manifest model.json path")
    group.addoption("--kernels", default=DEFAULT_KERNELS, help="Compiled SM120 cubins directory")
    group.addoption("--weights", default=DEFAULT_WEIGHTS, help="Inkling-2-Mini weights directory")
    group.addoption("--kern-bin", default=DEFAULT_KERN_BIN, help="Path to kern CLI binary")
    group.addoption("--model-id", default="", help="Override served model ID")
    group.addoption("--kl-limit", default="", help="Optional KL threshold (nats) for e2e-logits")
    group.addoption(
        "--max-ctx-probe",
        type=int,
        default=262144,
        help="Max single-GPU long-context probe length for Tier 4 stress",
    )
    group.addoption("--gpu", type=int, default=0, help="CUDA device ordinal for kern CLI calls")
    group.addoption(
        "--strict-cli",
        action="store_true",
        default=False,
        help="Fail instead of skipping when CLI artifacts (kern_bin/manifest/weights) are missing",
    )


@pytest.fixture(scope="session")
def e2e_config(request: pytest.FixtureRequest) -> E2EConfig:
    opts = request.config
    trace_raw = str(opts.getoption("--trace-url") or "").strip()
    pq_raw = str(opts.getoption("--baseline-parquet") or "").strip()
    met_raw = str(opts.getoption("--baseline-metrics") or "").strip()
    model_raw = str(opts.getoption("--model-id") or "").strip()
    kl_raw = str(opts.getoption("--kl-limit") or "").strip()
    return E2EConfig(
        base_url=str(opts.getoption("--base-url")).rstrip("/"),
        trace_url=trace_raw.rstrip("/") if trace_raw else None,
        baseline_dir=Path(str(opts.getoption("--baseline-dir"))),
        baseline_parquet=Path(pq_raw) if pq_raw else None,
        baseline_metrics=Path(met_raw) if met_raw else None,
        manifest=Path(str(opts.getoption("--manifest"))),
        kernels=Path(str(opts.getoption("--kernels"))),
        weights=Path(str(opts.getoption("--weights"))),
        kern_bin=Path(str(opts.getoption("--kern-bin"))),
        model_id=model_raw or None,
        kl_limit=float(kl_raw) if kl_raw else None,
        max_ctx_probe=int(opts.getoption("--max-ctx-probe")),
        gpu=int(opts.getoption("--gpu")),
        strict_cli=bool(opts.getoption("--strict-cli")),
    )


@pytest.fixture(scope="session")
def client(e2e_config: E2EConfig) -> KernClient:
    cli = KernClient(
        base_url=e2e_config.base_url,
        trace_url=e2e_config.trace_url,
        model_id=e2e_config.model_id,
    )
    status, body = cli.health(timeout=10.0)
    if status != 200:
        raise RuntimeError(
            f"Live kern-serve server at {e2e_config.base_url} is unhealthy (HTTP {status}: {body})"
        )
    cli.resolve_model_id()
    return cli


@pytest.fixture(scope="session")
def model_id(client: KernClient) -> str:
    return client.resolve_model_id()


@pytest.fixture(scope="session")
def baselines(e2e_config: E2EConfig) -> BaselineArtifacts:
    return discover_baselines(
        baseline_dir=e2e_config.baseline_dir,
        explicit_parquet=e2e_config.baseline_parquet,
        explicit_metrics=e2e_config.baseline_metrics,
    )


@pytest.fixture(scope="session")
def kern_cli(e2e_config: E2EConfig) -> KernCLI:
    return KernCLI(e2e_config.kern_bin)


def require_manifest_artifacts(cfg: E2EConfig) -> None:
    """Require CLI/manifest/weights artifacts; fail in `--strict-cli`, skip only in ad-hoc optional parity runs."""
    missing: list[str] = []
    if not cfg.kern_bin.is_file():
        missing.append(f"kern binary not found: {cfg.kern_bin}")
    if not cfg.manifest.is_file():
        missing.append(f"manifest not found: {cfg.manifest}")
    if not cfg.weights.is_dir():
        missing.append(f"weights directory not found: {cfg.weights}")
    if missing:
        msg = "; ".join(missing)
        if cfg.strict_cli:
            raise AssertionError(msg)
        pytest.skip(msg)
