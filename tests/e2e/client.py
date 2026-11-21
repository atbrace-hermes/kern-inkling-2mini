"""Pure-stdlib HTTP, SSE, OpenMetrics Prometheus, and kern-CLI client for opaque-box E2E tests."""

from __future__ import annotations

import http.client
import json
import math
import os
import re
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class HttpError(Exception):
    """HTTP error raised by KernClient with parsed JSON envelope when available."""

    def __init__(
        self,
        status: int,
        url: str,
        body: str,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status = status
        self.url = url
        self.body = body
        self.headers = headers or {}
        self.payload: dict[str, Any] | None = None
        try:
            parsed = json.loads(body)
            if isinstance(parsed, dict):
                self.payload = parsed
        except json.JSONDecodeError:
            pass
        super().__init__(f"HTTP {status} from {url}: {body[:400]}")


@dataclass
class SSEStreamResult:
    """Assembled result of an OpenAI /v1/completions or /v1/chat/completions SSE stream."""

    chunks: list[dict[str, Any]] = field(default_factory=list)
    content: str = ""
    reasoning: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    token_ids: list[int] = field(default_factory=list)
    prompt_token_ids: list[int] | None = None
    finish_reason: str | None = None
    stop_reason: Any = None
    usage: dict[str, Any] | None = None
    errors: list[dict[str, Any]] = field(default_factory=list)
    done: bool = False
    raw_lines: list[str] = field(default_factory=list)
    ttft_s: float | None = None
    elapsed_s: float | None = None


@dataclass(frozen=True)
class MetricSample:
    """Single Prometheus/OpenMetrics sample line."""

    name: str
    labels: dict[str, str]
    value: float


_LABEL_RE = re.compile(r'(\w+)\s*=\s*"((?:\\.|[^"\\])*)"')


def _parse_labels(raw: str) -> dict[str, str]:
    if not raw:
        return {}
    unesc = {'"': '"', "\\": "\\", "n": "\n"}
    return {
        k: re.sub(r'\\(["\\n])', lambda m: unesc[m.group(1)], v)
        for k, v in _LABEL_RE.findall(raw)
    }


def _find_label_block(line: str) -> tuple[int, int] | None:
    """Return `(brace_start, brace_end)` for metric labels, ignoring quotes/escapes and exemplars."""
    i, n = 0, len(line)
    while i < n and not line[i].isspace() and line[i] != "{":
        i += 1
    while i < n and line[i].isspace():
        i += 1
    if i >= n or line[i] != "{":
        return None
    brace_start = i
    i += 1
    in_quote = False
    escaped = False
    while i < n:
        ch = line[i]
        if in_quote:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_quote = False
        else:
            if ch == '"':
                in_quote = True
            elif ch == "}":
                return brace_start, i
        i += 1
    return None


def parse_openmetrics_samples(text: str) -> list[MetricSample]:
    """Parse Prometheus / OpenMetrics exposition text into MetricSample records."""
    samples: list[MetricSample] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        span = _find_label_block(line)
        if span is not None:
            brace_start, brace_end = span
            name = line[:brace_start].strip()
            labels = _parse_labels(line[brace_start + 1 : brace_end])
            rest = line[brace_end + 1 :].strip()
        else:
            parts = line.split(None, 1)
            if len(parts) != 2 or "{" in parts[0]:
                continue
            name, rest = parts[0].strip(), parts[1].strip()
            labels = {}
        val_tokens = rest.split()
        if not val_tokens:
            continue
        try:
            val = float(val_tokens[0])
        except ValueError:
            continue
        samples.append(MetricSample(name=name, labels=labels, value=val))
    return samples


def parse_openmetrics(text: str) -> dict[str, float]:
    """Aggregate Prometheus / OpenMetrics sample values by metric name (summing label sets)."""
    totals: dict[str, float] = {}
    for sample in parse_openmetrics_samples(text):
        if math.isnan(sample.value):
            continue
        totals[sample.name] = totals.get(sample.name, 0.0) + sample.value
    return totals


def metric_delta(before: dict[str, float], after: dict[str, float], name: str) -> float:
    """Return counter delta between two OpenMetrics scrapes."""
    return after.get(name, 0.0) - before.get(name, 0.0)


def compute_mtp_stats(before: dict[str, float], after: dict[str, float]) -> dict[str, float | None]:
    """Compute speculative decoding counters and derived MTP acceptance stats."""
    drafts = metric_delta(before, after, "vllm:spec_decode_num_drafts_total")
    draft_tokens = metric_delta(before, after, "vllm:spec_decode_num_draft_tokens_total")
    accepted = metric_delta(before, after, "vllm:spec_decode_num_accepted_tokens_total")
    tokens_per_round = (accepted + drafts) / drafts if drafts > 0 else None
    accept_rate = accepted / draft_tokens if draft_tokens > 0 else None
    return {
        "drafts": drafts,
        "draft_tokens": draft_tokens,
        "accepted_tokens": accepted,
        "tokens_per_round": tokens_per_round,
        "accept_rate": accept_rate,
    }


def _assemble_sse_lines(lines: Iterable[bytes | str], started: float | None = None) -> SSEStreamResult:
    """Parse SSE event stream from kern-serve OpenAI endpoints."""
    t0 = started if started is not None else time.monotonic()
    res = SSEStreamResult()
    tool_map: dict[int, dict[str, Any]] = {}
    first_choice_dt: float | None = None

    for raw in lines:
        line = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
        stripped = line.rstrip("\r\n")
        res.raw_lines.append(stripped)
        if not stripped.startswith("data:"):
            continue
        payload = stripped[5:].strip()
        if payload == "[DONE]":
            res.done = True
            break
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if not isinstance(chunk, dict):
            continue
        res.chunks.append(chunk)
        err = chunk.get("error")
        if err is not None and err is not False:
            res.errors.append(err if isinstance(err, dict) else {"message": str(err)})
            continue
        if chunk.get("object") == "error":
            res.errors.append(chunk)
            continue
        if chunk.get("usage"):
            res.usage = chunk["usage"]
        if chunk.get("prompt_token_ids") is not None and res.prompt_token_ids is None:
            res.prompt_token_ids = [int(t) for t in chunk["prompt_token_ids"]]

        now_dt = time.monotonic() - t0
        for choice in chunk.get("choices") or []:
            if first_choice_dt is None:
                first_choice_dt = now_dt
            if choice.get("prompt_token_ids") is not None and res.prompt_token_ids is None:
                res.prompt_token_ids = [int(t) for t in choice["prompt_token_ids"]]
            if choice.get("finish_reason") is not None:
                res.finish_reason = choice["finish_reason"]
            if choice.get("stop_reason") is not None:
                res.stop_reason = choice["stop_reason"]
            has_content_token = False
            if isinstance(choice.get("token_ids"), list):
                res.token_ids.extend(int(t) for t in choice["token_ids"])
                if choice["token_ids"]:
                    has_content_token = True
            # /v1/completions chunk
            if isinstance(choice.get("text"), str):
                res.content += choice["text"]
                if choice["text"]:
                    has_content_token = True
            # /v1/chat/completions delta chunk
            delta = choice.get("delta")
            if isinstance(delta, dict):
                # kern-serve exposes reasoning under delta.reasoning (NOT reasoning_content)
                r_piece = delta.get("reasoning")
                if isinstance(r_piece, str):
                    res.reasoning += r_piece
                    if r_piece:
                        has_content_token = True
                c_piece = delta.get("content")
                if isinstance(c_piece, str):
                    res.content += c_piece
                    if c_piece:
                        has_content_token = True
                tc_list = delta.get("tool_calls")
                if isinstance(tc_list, list):
                    if tc_list:
                        has_content_token = True
                    for tc in tc_list:
                        if not isinstance(tc, dict):
                            continue
                        idx = int(tc.get("index", 0))
                        entry = tool_map.setdefault(
                            idx,
                            {
                                "index": idx,
                                "id": "",
                                "type": tc.get("type") or "function",
                                "function": {"name": "", "arguments": ""},
                            },
                        )
                        if tc.get("id"):
                            entry["id"] = tc["id"]
                        if tc.get("type"):
                            entry["type"] = tc["type"]
                        fn = tc.get("function")
                        if isinstance(fn, dict):
                            if isinstance(fn.get("name"), str) and fn["name"]:
                                if not entry["function"]["name"]:
                                    entry["function"]["name"] = fn["name"]
                            if isinstance(fn.get("arguments"), str):
                                entry["function"]["arguments"] += fn["arguments"]
            if res.ttft_s is None and has_content_token:
                res.ttft_s = now_dt

    if res.ttft_s is None and first_choice_dt is not None:
        res.ttft_s = first_choice_dt
    res.elapsed_s = time.monotonic() - t0
    if tool_map:
        res.tool_calls = [tool_map[idx] for idx in sorted(tool_map)]
    return res


class KernClient:
    """Pure-stdlib HTTP/SSE/OpenMetrics client for kern-serve."""

    def __init__(
        self,
        base_url: str,
        trace_url: str | None = None,
        model_id: str | None = None,
        timeout: float = 180.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.trace_url = trace_url.rstrip("/") if trace_url else None
        self.model_id = model_id
        self.timeout = timeout

    def _url(self, path: str, base: str | None = None) -> str:
        root = (base or self.base_url).rstrip("/")
        return f"{root}{path}" if path.startswith("/") else f"{root}/{path}"

    def raw_request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | bytes | str | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
        base: str | None = None,
    ) -> tuple[int, dict[str, str], str]:
        """Execute an HTTP request and return (status, headers, body_text) without raising on 4xx/5xx."""
        url = self._url(path, base=base)
        req_headers = dict(headers or {})
        data: bytes | None = None
        if isinstance(body, dict):
            data = json.dumps(body).encode("utf-8")
            req_headers.setdefault("Content-Type", "application/json")
        elif isinstance(body, str):
            data = body.encode("utf-8")
            req_headers.setdefault("Content-Type", "application/json")
        elif isinstance(body, bytes):
            data = body

        req = urllib.request.Request(url, data=data, headers=req_headers, method=method.upper())
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                resp_headers = {k.lower(): v for k, v in resp.headers.items()}
                text = resp.read().decode("utf-8", errors="replace")
                return resp.status, resp_headers, text
        except urllib.error.HTTPError as exc:
            resp_headers = {k.lower(): v for k, v in exc.headers.items()} if exc.headers else {}
            text = exc.read().decode("utf-8", errors="replace")
            return exc.code, resp_headers, text

    def _json_request(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
        timeout: float | None = None,
        base: str | None = None,
    ) -> dict[str, Any]:
        url = self._url(path, base=base)
        status, headers, text = self.raw_request(method, path, body=body, timeout=timeout, base=base)
        if status >= 400:
            raise HttpError(status, url, text, headers)
        return json.loads(text) if text else {}

    def health(self, timeout: float = 10.0) -> tuple[int, str]:
        status, _, text = self.raw_request("GET", "/health", timeout=timeout)
        return status, text

    def models(self, timeout: float = 15.0) -> dict[str, Any]:
        return self._json_request("GET", "/v1/models", timeout=timeout)

    def resolve_model_id(self) -> str:
        if self.model_id:
            return self.model_id
        data = self.models().get("data") or []
        if not data:
            raise RuntimeError(f"GET {self.base_url}/v1/models returned empty data list")
        self.model_id = str(data[0]["id"])
        return self.model_id

    def load(self, timeout: float = 10.0) -> dict[str, Any]:
        return self._json_request("GET", "/load", timeout=timeout)

    def version(self, timeout: float = 10.0) -> dict[str, Any]:
        return self._json_request("GET", "/version", timeout=timeout)

    def tokenize(
        self,
        prompt: str | None = None,
        messages: list[dict[str, Any]] | None = None,
        model: str | None = None,
        return_token_strs: bool = False,
        chat_template_kwargs: dict[str, Any] | None = None,
        timeout: float = 60.0,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"return_token_strs": return_token_strs}
        if model is not None:
            body["model"] = model
        elif self.model_id is not None:
            body["model"] = self.model_id
        if prompt is not None:
            body["prompt"] = prompt
        if messages is not None:
            body["messages"] = messages
        if chat_template_kwargs is not None:
            body["chat_template_kwargs"] = chat_template_kwargs
        return self._json_request("POST", "/tokenize", body=body, timeout=timeout)

    def detokenize(
        self,
        tokens: Sequence[int],
        model: str | None = None,
        timeout: float = 60.0,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"tokens": [int(t) for t in tokens]}
        if model is not None:
            body["model"] = model
        elif self.model_id is not None:
            body["model"] = self.model_id
        return self._json_request("POST", "/detokenize", body=body, timeout=timeout)

    def completions(
        self,
        prompt: str | Sequence[int],
        timeout: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"prompt": list(prompt) if isinstance(prompt, tuple) else prompt, **kwargs}
        body.setdefault("model", self.resolve_model_id())
        body["stream"] = False
        return self._json_request("POST", "/v1/completions", body=body, timeout=timeout)

    def completions_stream(
        self,
        prompt: str | Sequence[int],
        timeout: float | None = None,
        **kwargs: Any,
    ) -> SSEStreamResult:
        body: dict[str, Any] = {"prompt": list(prompt) if isinstance(prompt, tuple) else prompt, **kwargs}
        body.setdefault("model", self.resolve_model_id())
        body["stream"] = True
        return self._stream_post("/v1/completions", body, timeout=timeout)

    def chat_completions(
        self,
        messages: list[dict[str, Any]],
        timeout: float | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"messages": messages, **kwargs}
        body.setdefault("model", self.resolve_model_id())
        body["stream"] = False
        return self._json_request("POST", "/v1/chat/completions", body=body, timeout=timeout)

    def chat_completions_stream(
        self,
        messages: list[dict[str, Any]],
        timeout: float | None = None,
        **kwargs: Any,
    ) -> SSEStreamResult:
        body: dict[str, Any] = {"messages": messages, **kwargs}
        body.setdefault("model", self.resolve_model_id())
        body["stream"] = True
        return self._stream_post("/v1/chat/completions", body, timeout=timeout)

    def _stream_post(
        self,
        path: str,
        body: dict[str, Any],
        timeout: float | None = None,
    ) -> SSEStreamResult:
        url = self._url(path)
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
            method="POST",
        )
        try:
            started = time.monotonic()
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                return _assemble_sse_lines(resp, started=started)
        except urllib.error.HTTPError as exc:
            headers = {k.lower(): v for k, v in exc.headers.items()} if exc.headers else {}
            text = exc.read().decode("utf-8", errors="replace")
            raise HttpError(exc.code, url, text, headers) from exc

    def raw_post_abort(
        self,
        path: str,
        body: dict[str, Any],
        chunks_before_close: int = 3,
        timeout: float = 120.0,
    ) -> int:
        """Open a streaming request, read N `data:` SSE lines, and forcibly close the socket."""
        parsed = urllib.parse.urlparse(self._url(path))
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        conn_cls = (
            http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        )
        conn = conn_cls(host, port, timeout=timeout)
        payload = json.dumps({**body, "stream": True}).encode("utf-8")
        try:
            conn.request(
                "POST",
                parsed.path or "/",
                body=payload,
                headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
            )
            resp = conn.getresponse()
            if resp.status >= 400:
                err_body = resp.read().decode("utf-8", errors="replace")
                raise HttpError(resp.status, self._url(path), err_body)
            got = 0
            while got < chunks_before_close:
                line = resp.readline()
                if not line:
                    break
                if line.startswith(b"data:"):
                    got += 1
                    if line.strip() == b"data: [DONE]":
                        break
            return got
        finally:
            if conn.sock is not None:
                try:
                    conn.sock.close()
                except OSError:
                    pass
            conn.close()

    def metrics_raw(self, timeout: float = 15.0) -> str:
        url = self._url("/metrics")
        status, headers, text = self.raw_request("GET", "/metrics", timeout=timeout)
        if status >= 400:
            raise HttpError(status, url, text, headers)
        return text

    def metrics(self, timeout: float = 15.0) -> dict[str, float]:
        return parse_openmetrics(self.metrics_raw(timeout=timeout))

    def metrics_samples(self, timeout: float = 15.0) -> list[MetricSample]:
        return parse_openmetrics_samples(self.metrics_raw(timeout=timeout))

    def trace_metrics(self, timeout: float = 15.0) -> dict[str, float]:
        if not self.trace_url:
            raise RuntimeError("trace_url is not configured")
        url = self._url("/kern/metrics", base=self.trace_url)
        status, headers, text = self.raw_request(
            "GET", "/kern/metrics", timeout=timeout, base=self.trace_url
        )
        if status >= 400:
            raise HttpError(status, url, text, headers)
        return parse_openmetrics(text)

    def trace_rotate(self, timeout: float = 15.0) -> dict[str, Any]:
        if not self.trace_url:
            raise RuntimeError("trace_url is not configured")
        return self._json_request("POST", "/kern/trace/rotate", body={}, timeout=timeout, base=self.trace_url)

    def concurrent_map(
        self,
        fn: Callable[[Any], Any],
        items: Sequence[Any],
        max_workers: int,
        synchronize: bool = True,
        barrier_timeout: float = 30.0,
    ) -> list[Any]:
        item_list = list(items)
        if not item_list:
            return []
        parties = min(max_workers, len(item_list))
        barrier = (
            threading.Barrier(parties, timeout=barrier_timeout)
            if (synchronize and parties > 1)
            else None
        )
        gate_lock = threading.Lock()
        waiting = 0

        def _run(item: Any) -> Any:
            nonlocal waiting
            if barrier is not None:
                should_wait = False
                with gate_lock:
                    if waiting < parties:
                        waiting += 1
                        should_wait = True
                if should_wait:
                    barrier.wait()
            return fn(item)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            return list(pool.map(_run, item_list))


class KernCLI:
    """Subprocess wrapper around `kern verify`, `kern run`, and `kern test e2e-logits`."""

    def __init__(self, kern_bin: Path) -> None:
        self.kern_bin = Path(kern_bin)

    def kern_verify(
        self,
        manifest: Path,
        config: Path | None = None,
        timeout: float = 120.0,
    ) -> subprocess.CompletedProcess[str]:
        cmd = [str(self.kern_bin), "verify", str(manifest)]
        if config is not None:
            cmd.extend(["--config", str(config)])
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)

    def kern_run(
        self,
        manifest: Path,
        weights: Path | Sequence[Path],
        kernels: Path | None = None,
        tokenizer: Path | None = None,
        prompt: str | None = None,
        prompt_ids: Sequence[int] | None = None,
        steps: int = 32,
        rows: int | None = None,
        capacity: int | None = None,
        chunk: int | None = None,
        eager: bool = False,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        min_p: float = 0.0,
        seed: int = 0,
        stop_tokens: Sequence[int] | None = None,
        gpu: int = 0,
        timeout: float = 1800.0,
    ) -> tuple[subprocess.CompletedProcess[str], list[int]]:
        cmd = [
            str(self.kern_bin),
            "run",
            "--manifest",
            str(manifest),
            "--steps",
            str(steps),
            "--temperature",
            str(temperature),
            "--top-k",
            str(top_k),
            "--top-p",
            str(top_p),
            "--min-p",
            str(min_p),
            "--seed",
            str(seed),
            "--gpu",
            str(gpu),
        ]
        weight_list = [weights] if isinstance(weights, Path) else list(weights)
        for w in weight_list:
            cmd.extend(["--weights", str(w)])
        if kernels is not None:
            cmd.extend(["--kernels", str(kernels)])
        if tokenizer is not None:
            cmd.extend(["--tokenizer", str(tokenizer)])
        if prompt is not None:
            cmd.extend(["--prompt", prompt])
        if prompt_ids is not None:
            cmd.extend(["--prompt-ids", ",".join(str(int(x)) for x in prompt_ids)])
        if rows is not None:
            cmd.extend(["--rows", str(rows)])
        if capacity is not None:
            cmd.extend(["--capacity", str(capacity)])
        if chunk is not None:
            cmd.extend(["--chunk", str(chunk)])
        if eager:
            cmd.append("--eager")
        if stop_tokens:
            cmd.extend(["--stop-tokens", ",".join(str(int(x)) for x in stop_tokens)])

        env = dict(os.environ)
        if env.get("RUST_LOG", "").strip().lower() in ("", "warn", "error", "off"):
            env["RUST_LOG"] = "info"
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        if proc.returncode != 0:
            raise RuntimeError(
                f"kern run failed (rc={proc.returncode}):\n"
                f"stdout:\n{proc.stdout[-400:]}\nstderr:\n{proc.stderr[-800:]}"
            )
        m = re.search(r"generated ids:\s*\[([^\]]*)\]", proc.stderr)
        if m is None:
            raise RuntimeError(
                f"kern run succeeded (rc=0) without 'generated ids:' in stderr:\n{proc.stderr[-800:]}"
            )
        ids = [int(x.strip()) for x in m.group(1).split(",") if x.strip()]
        return proc, ids

    def kern_probe(
        self,
        manifest: Path,
        weights: Path | Sequence[Path],
        probe_dir: Path,
        prompt_ids: Sequence[int],
        kernels: Path | None = None,
        tokenizer: Path | None = None,
        steps: int = 2,
        rows: int | None = None,
        capacity: int | None = None,
        probe_steps: int = 2,
        probe_labels: str = "",
        gpu: int = 0,
        timeout: float = 1800.0,
    ) -> subprocess.CompletedProcess[str]:
        cmd = [
            str(self.kern_bin),
            "run",
            "--manifest",
            str(manifest),
            "--steps",
            str(steps),
            "--probe-dir",
            str(probe_dir),
            "--probe-steps",
            str(probe_steps),
            "--probe-labels",
            probe_labels,
            "--prompt-ids",
            ",".join(str(int(x)) for x in prompt_ids),
            "--gpu",
            str(gpu),
        ]
        weight_list = [weights] if isinstance(weights, Path) else list(weights)
        for w in weight_list:
            cmd.extend(["--weights", str(w)])
        if kernels is not None:
            cmd.extend(["--kernels", str(kernels)])
        if tokenizer is not None:
            cmd.extend(["--tokenizer", str(tokenizer)])
        if rows is not None:
            cmd.extend(["--rows", str(rows)])
        if capacity is not None:
            cmd.extend(["--capacity", str(capacity)])

        env = dict(os.environ)
        if env.get("RUST_LOG", "").strip().lower() in ("", "warn", "error", "off"):
            env["RUST_LOG"] = "info"
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
        if proc.returncode != 0:
            raise RuntimeError(
                f"kern run --probe-dir failed (rc={proc.returncode}):\n"
                f"stdout:\n{proc.stdout[-400:]}\nstderr:\n{proc.stderr[-800:]}"
            )
        return proc

    def kern_e2e_logits(
        self,
        manifest: Path,
        weights: Path | Sequence[Path],
        reference: Path,
        kernels: Path | None = None,
        kl: float | None = None,
        out: Path | None = None,
        gpu: int = 0,
        timeout: float = 3600.0,
    ) -> subprocess.CompletedProcess[str]:
        cmd = [
            str(self.kern_bin),
            "test",
            "e2e-logits",
            "--manifest",
            str(manifest),
            "--reference",
            str(reference),
            "--gpu",
            str(gpu),
        ]
        weight_list = [weights] if isinstance(weights, Path) else list(weights)
        for w in weight_list:
            cmd.extend(["--weights", str(w)])
        if kernels is not None:
            cmd.extend(["--kernels", str(kernels)])
        if kl is not None:
            cmd.extend(["--kl", str(kl)])
        if out is not None:
            cmd.extend(["--out", str(out)])
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
