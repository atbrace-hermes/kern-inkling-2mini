"""100% CPU-only self-tests verifying client, SSE, OpenMetrics, Parquet, config, and CLI runner."""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import threading
import time
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from unittest import mock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

_E2E_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _E2E_DIR.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tests.e2e import run_e2e  # noqa: E402
from tests.e2e.client import (  # noqa: E402
    HttpError,
    KernCLI,
    KernClient,
    _assemble_sse_lines,
    compute_mtp_stats,
    metric_delta,
    parse_openmetrics,
    parse_openmetrics_samples,
)
from tests.e2e.fixtures_data import (  # noqa: E402
    EOS_TOKEN_ID,
    MAX_MODEL_LEN,
    REASONING_EFFORT_LEVELS,
    UNPADDED_VOCAB_SIZE,
    assert_valid_token_ids,
    build_audio_messages,
    build_vision_base64_messages,
    build_vision_image_url_messages,
    discover_baselines,
    extract_baseline_mtp_metrics,
    is_near_tie_bf16,
    manifest_logits_spec,
    read_probe_first_token,
    read_probe_logits_row0,
    strip_terminal_stop_token,
    ulp_of_bf16,
    validate_questions_parquet,
    validate_recording_parquet,
    validate_weights_configs,
    verify_greedy_parity_with_near_tie,
)

pytestmark = pytest.mark.selftest


class _MockServerState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.model_id = "Inkling-2-Mini"
        self.max_model_len = MAX_MODEL_LEN
        self.prefix_queries = 0
        self.prefix_hits = 0
        self.prompt_tokens_cached = 0
        self.spec_drafts = 0
        self.spec_draft_tokens = 0
        self.spec_accepted_tokens = 0
        self.stateful_snapshots: set[tuple[int, ...]] = set()
        self.aborted_requests = 0

    def tokenize_str(self, text: str) -> list[int]:
        return [100 + (ord(ch) % 50000) for ch in text] or [42]


def _make_handler(state: _MockServerState) -> type[BaseHTTPRequestHandler]:
    class _MockHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
            return

        def _send_json(self, code: int, obj: Any) -> None:
            raw = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def _error(self, code: int, err_type: str, msg: str, param: str | None = None) -> None:
            self._send_json(
                code,
                {"error": {"message": msg, "type": err_type, "param": param, "code": code}},
            )

        def _read_json_body(self) -> dict[str, Any] | None:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            try:
                data = json.loads(raw.decode("utf-8"))
            except json.JSONDecodeError:
                self._error(400, "Bad Request", "Malformed JSON body")
                return None
            if not isinstance(data, dict):
                self._error(400, "Bad Request", "Expected JSON object")
                return None
            return data

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/health":
                self.send_response(200)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            if self.path == "/v1/models":
                self._send_json(
                    200,
                    {
                        "object": "list",
                        "data": [
                            {
                                "id": state.model_id,
                                "object": "model",
                                "created": 1700000000,
                                "owned_by": "vllm-frontend-rs",
                                "root": "/weights/Inkling-2-Mini",
                                "parent": None,
                                "max_model_len": state.max_model_len,
                            }
                        ],
                    },
                )
                return
            if self.path == "/metrics":
                with state.lock:
                    text = (
                        "# HELP vllm:prefix_cache_queries_total Prefix cache queries\n"
                        "# TYPE vllm:prefix_cache_queries_total counter\n"
                        f"vllm:prefix_cache_queries_total {state.prefix_queries}\n"
                        f"vllm:prefix_cache_hits_total {state.prefix_hits}\n"
                        f"vllm:prompt_tokens_cached_total {state.prompt_tokens_cached}\n"
                        f'vllm:prompt_tokens_cached_by_source_total{{source="device"}} {state.prompt_tokens_cached}\n'
                        f"vllm:spec_decode_num_drafts_total {state.spec_drafts}\n"
                        f"vllm:spec_decode_num_draft_tokens_total {state.spec_draft_tokens}\n"
                        f"vllm:spec_decode_num_accepted_tokens_total {state.spec_accepted_tokens}\n"
                        f'vllm:spec_decode_num_accepted_tokens_per_pos_total{{position="0"}} {state.spec_accepted_tokens}\n'
                        "vllm:num_requests_running 0\n"
                        "vllm:num_requests_waiting 0\n"
                        "# EOF\n"
                    )
                raw = text.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/openmetrics-text; version=1.0.0; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return
            if self.path == "/kern/metrics":
                raw = b'kern_steps_total{rank="0",program="round",shape="8",kv="0"} 5\n# EOF\n'
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return
            self._error(404, "NotFoundError", f"Unknown path {self.path}")

        def do_POST(self) -> None:  # noqa: N802
            if self.path == "/v1/messages":
                self._error(501, "NotImplementedError", "Not implemented")
                return
            if self.path == "/kern/trace/rotate":
                self._send_json(200, {"closed": ["trace_0001.json"]})
                return

            body = self._read_json_body()
            if body is None:
                return

            model = body.get("model")
            if model not in (None, "", state.model_id):
                self._error(404, "NotFoundError", f"The model `{model}` does not exist.", param="model")
                return

            if self.path == "/tokenize":
                if "prompt" in body:
                    toks = state.tokenize_str(str(body["prompt"]))
                else:
                    toks = state.tokenize_str(json.dumps(body.get("messages", [])))
                resp: dict[str, Any] = {
                    "count": len(toks),
                    "max_model_len": state.max_model_len,
                    "tokens": toks,
                }
                if body.get("return_token_strs"):
                    resp["token_strs"] = [f"tok_{t}" for t in toks]
                self._send_json(200, resp)
                return

            if self.path == "/detokenize":
                toks = body.get("tokens", [])
                text = " ".join(f"w{t}" for t in toks)
                self._send_json(200, {"prompt": text})
                return

            # Shaping check: Capabilities.logprobs == false
            for key in ("logprobs", "prompt_logprobs"):
                val = body.get(key)
                if val is not None and val is not False:
                    self._error(400, "BadRequestError", f"{key} is not supported by this engine")
                    return

            # Bridge-refused params check (returns 500)
            if (
                (body.get("min_tokens") or 0) != 0
                or float(body.get("frequency_penalty") or 0.0) != 0.0
                or float(body.get("presence_penalty") or 0.0) != 0.0
                or float(body.get("repetition_penalty") if body.get("repetition_penalty") is not None else 1.0)
                != 1.0
            ):
                self._error(500, "InternalServerError", "Internal server error")
                return

            if self.path == "/v1/completions":
                self._handle_completions(body)
                return
            if self.path == "/v1/chat/completions":
                self._handle_chat_completions(body)
                return
            self._error(404, "NotFoundError", f"Unknown path {self.path}")

        def _handle_completions(self, body: dict[str, Any]) -> None:
            prompt = body.get("prompt")
            if prompt in ("", []):
                self._error(400, "BadRequestError", "Empty prompt is not allowed", param="prompt")
                return
            if isinstance(prompt, list):
                for tok in prompt:
                    if not isinstance(tok, int) or tok < 0 or tok >= UNPADDED_VOCAB_SIZE:
                        self._error(400, "BadRequestError", f"Token ID {tok} out of vocab", param="prompt")
                        return
                prompt_ids = list(prompt)
            else:
                prompt_ids = state.tokenize_str(str(prompt))

            if len(prompt_ids) >= state.max_model_len:
                self._error(
                    400,
                    "BadRequestError",
                    f"This model's maximum context length is {state.max_model_len} tokens.",
                    param="input_tokens",
                )
                return

            max_tokens = body.get("max_tokens")
            if max_tokens == 0 and not body.get("echo"):
                self._error(400, "BadRequestError", "max_tokens must be >= 1 unless echo=true")
                return

            out_text = " Paris."
            out_ids = [1234, 5678, EOS_TOKEN_ID]

            with state.lock:
                state.prefix_queries += 1
                cached = 0
                for snap in state.stateful_snapshots:
                    if len(snap) < len(prompt_ids) and tuple(prompt_ids[: len(snap)]) == snap:
                        if len(snap) > cached:
                            cached = len(snap)
                if cached > 0:
                    state.prefix_hits += 1
                    state.prompt_tokens_cached += cached
                # Model KernScheduler::finish(): max_tokens=1 MTP overshoot -> "not kept"
                if max_tokens != 1:
                    state.stateful_snapshots.add(tuple(prompt_ids) + tuple(out_ids[:-1]))
                state.spec_drafts += 2
                state.spec_draft_tokens += 14
                state.spec_accepted_tokens += 6

            usage = {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(out_ids),
                "total_tokens": len(prompt_ids) + len(out_ids),
                "prompt_tokens_details": {"cached_tokens": cached},
            }

            if body.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                if body.get("prompt") == "__TRIGGER_SSE_ERROR__":
                    err_evt = {"error": {"message": "Internal server error", "type": "InternalServerError", "code": 500}}
                    self.wfile.write(f"data: {json.dumps(err_evt)}\n\n".encode())
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                    return
                pieces = [" Paris", "."]
                try:
                    for idx, piece in enumerate(pieces):
                        chunk = {
                            "id": "cmpl-mock",
                            "object": "text_completion",
                            "choices": [{"index": 0, "text": piece, "finish_reason": None}],
                        }
                        if body.get("return_token_ids"):
                            chunk["choices"][0]["token_ids"] = [out_ids[idx]]
                        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                        self.wfile.flush()
                        if body.get("max_tokens", 16) >= 2048:
                            time.sleep(0.01)
                    if body.get("max_tokens", 16) >= 2048:
                        for extra in range(10):
                            chunk = {
                                "id": "cmpl-mock",
                                "object": "text_completion",
                                "choices": [{"index": 0, "text": f" x{extra}", "finish_reason": None}],
                            }
                            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                            self.wfile.flush()
                            time.sleep(0.01)
                    fin = {
                        "id": "cmpl-mock",
                        "object": "text_completion",
                        "choices": [{"index": 0, "text": "", "finish_reason": "stop"}],
                        "usage": usage,
                    }
                    if body.get("return_token_ids"):
                        fin["choices"][0]["token_ids"] = [EOS_TOKEN_ID]
                    self.wfile.write(f"data: {json.dumps(fin)}\n\n".encode())
                    self.wfile.write(b"data: [DONE]\n\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    with state.lock:
                        state.aborted_requests += 1
                return

            choice: dict[str, Any] = {
                "index": 0,
                "text": out_text,
                "finish_reason": "stop",
                "stop_reason": EOS_TOKEN_ID,
            }
            if body.get("return_token_ids"):
                choice["token_ids"] = out_ids
                choice["prompt_token_ids"] = prompt_ids
            self._send_json(
                200,
                {
                    "id": "cmpl-mock",
                    "object": "text_completion",
                    "model": state.model_id,
                    "choices": [choice],
                    "usage": usage,
                },
            )

        def _handle_chat_completions(self, body: dict[str, Any]) -> None:
            messages = body.get("messages") or []
            for msg in messages:
                if msg.get("role") == "developer":
                    self._error(400, "BadRequestError", "Unknown message role: developer")
                    return
                content = msg.get("content")
                if isinstance(content, list):
                    for part in content:
                        ptype = part.get("type") if isinstance(part, dict) else None
                        if ptype in ("image_url", "video_url", "audio_url", "input_audio", "image", "input_image"):
                            self._error(
                                400,
                                "BadRequestError",
                                f"unsupported multimodal content: {ptype}",
                            )
                            return

            effort = body.get("reasoning_effort")
            if effort is not None and effort not in REASONING_EFFORT_LEVELS:
                self._error(400, "BadRequestError", f"Invalid reasoning_effort: {effort!r}", param="reasoning_effort")
                return
            kwargs_effort = (body.get("chat_template_kwargs") or {}).get("reasoning_effort")
            if isinstance(kwargs_effort, (int, float)) and not (0.0 <= float(kwargs_effort) <= 0.99):
                self._error(400, "BadRequestError", f"Invalid reasoning_effort float: {kwargs_effort!r}")
                return

            tools = body.get("tools")
            if tools == []:
                self._error(400, "BadRequestError", "tools cannot be empty", param="tools")
                return

            with state.lock:
                state.prefix_queries += 1
                state.spec_drafts += 1
                state.spec_draft_tokens += 7
                state.spec_accepted_tokens += 4

            include_reasoning = body.get("include_reasoning", True)
            reasoning_str = "Let me check step by step." if (include_reasoning and effort != "none") else None
            tool_calls_out = None
            content_out: str | None = "Hello from Inkling-2-Mini."
            finish_reason = "stop"
            if tools:
                content_out = None
                finish_reason = "tool_calls"
                tool_calls_out = [
                    {
                        "id": "call_mock_1",
                        "type": "function",
                        "function": {"name": tools[0]["function"]["name"], "arguments": '{"city": "Tokyo"}'},
                    }
                ]

            usage = {
                "prompt_tokens": 24,
                "completion_tokens": 12,
                "total_tokens": 36,
                "prompt_tokens_details": {"cached_tokens": 0},
                "completion_tokens_details": {"reasoning_tokens": 5 if effort != "none" else 0},
            }

            if body.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                chunks: list[dict[str, Any]] = [
                    {"choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}
                ]
                if reasoning_str is not None:
                    chunks.append(
                        {"choices": [{"index": 0, "delta": {"reasoning": "Let me check "}, "finish_reason": None}]}
                    )
                    chunks.append(
                        {"choices": [{"index": 0, "delta": {"reasoning": "step by step."}, "finish_reason": None}]}
                    )
                if tool_calls_out:
                    chunks.append(
                        {
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "tool_calls": [
                                            {
                                                "index": 0,
                                                "id": "call_mock_1",
                                                "type": "function",
                                                "function": {"name": tools[0]["function"]["name"], "arguments": ""},
                                            }
                                        ]
                                    },
                                    "finish_reason": None,
                                }
                            ]
                        }
                    )
                    chunks.append(
                        {
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "tool_calls": [
                                            {"index": 0, "function": {"arguments": '{"city": '}}
                                        ]
                                    },
                                    "finish_reason": None,
                                }
                            ]
                        }
                    )
                    chunks.append(
                        {
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {
                                        "tool_calls": [
                                            {"index": 0, "function": {"arguments": '"Tokyo"}'}}
                                        ]
                                    },
                                    "finish_reason": None,
                                }
                            ]
                        }
                    )
                else:
                    chunks.append(
                        {"choices": [{"index": 0, "delta": {"content": "Hello from "}, "finish_reason": None}]}
                    )
                    chunks.append(
                        {"choices": [{"index": 0, "delta": {"content": "Inkling-2-Mini."}, "finish_reason": None}]}
                    )
                chunks.append(
                    {"choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}], "usage": usage}
                )
                for chunk in chunks:
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return

            message: dict[str, Any] = {
                "role": "assistant",
                "content": content_out,
                "reasoning": reasoning_str,
            }
            if tool_calls_out is not None:
                message["tool_calls"] = tool_calls_out

            chat_choice: dict[str, Any] = {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
            payload_out: dict[str, Any] = {
                "id": "chatcmpl-mock",
                "object": "chat.completion",
                "model": state.model_id,
                "choices": [chat_choice],
                "usage": usage,
            }
            if body.get("return_token_ids"):
                chat_choice["token_ids"] = [1234, EOS_TOKEN_ID]
                chat_choice["stop_reason"] = EOS_TOKEN_ID
                payload_out["prompt_token_ids"] = state.tokenize_str(json.dumps(messages))

            self._send_json(200, payload_out)

    return _MockHandler


@pytest.fixture()
def mock_server():
    state = _MockServerState()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(state))
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{port}"
        yield KernClient(base_url=base_url, trace_url=base_url), state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_selftest_health_and_models(mock_server: tuple[KernClient, _MockServerState]) -> None:
    cli, _ = mock_server
    status, _ = cli.health()
    assert status == 200
    models = cli.models()
    card = models["data"][0]
    assert card["id"] == "Inkling-2-Mini"
    assert card["owned_by"] == "vllm-frontend-rs"
    assert card["max_model_len"] == MAX_MODEL_LEN


def test_selftest_tokenize_detokenize_roundtrip(mock_server: tuple[KernClient, _MockServerState]) -> None:
    cli, _ = mock_server
    tok = cli.tokenize(prompt="Hello Inkling", return_token_strs=True)
    assert tok["count"] == len(tok["tokens"])
    assert len(tok["token_strs"]) == len(tok["tokens"])
    assert_valid_token_ids(tok["tokens"])
    detok = cli.detokenize(tok["tokens"])
    assert isinstance(detok["prompt"], str) and detok["prompt"]


def test_selftest_completions_and_sse_parity(mock_server: tuple[KernClient, _MockServerState]) -> None:
    cli, _ = mock_server
    non_stream = cli.completions("The capital of France is", max_tokens=8, temperature=0, return_token_ids=True)
    streamed = cli.completions_stream("The capital of France is", max_tokens=8, temperature=0, return_token_ids=True)
    assert streamed.done is True
    assert streamed.content == non_stream["choices"][0]["text"]
    assert streamed.finish_reason == "stop"
    assert streamed.token_ids == non_stream["choices"][0]["token_ids"]
    assert streamed.ttft_s is not None and streamed.ttft_s >= 0.0
    assert streamed.elapsed_s is not None and streamed.elapsed_s >= streamed.ttft_s
    stripped = strip_terminal_stop_token(non_stream["choices"][0]["token_ids"], "stop")
    assert stripped == [1234, 5678]


def test_selftest_chat_reasoning_field_and_stream(mock_server: tuple[KernClient, _MockServerState]) -> None:
    cli, _ = mock_server
    messages = [{"role": "system", "content": "Be concise."}, {"role": "user", "content": "Hi"}]
    resp = cli.chat_completions(messages, reasoning_effort="high", return_token_ids=True)
    msg = resp["choices"][0]["message"]
    assert msg["reasoning"] == "Let me check step by step."
    assert "reasoning_content" not in msg
    assert resp["choices"][0]["token_ids"] == [1234, EOS_TOKEN_ID]
    assert resp["usage"]["completion_tokens_details"]["reasoning_tokens"] > 0

    sse = cli.chat_completions_stream(messages, reasoning_effort="high")
    assert sse.done is True
    assert sse.reasoning == msg["reasoning"]
    assert sse.content == msg["content"]
    assert sse.ttft_s is not None and sse.ttft_s >= 0.0


def test_selftest_chat_tool_calls_streaming_assembly(mock_server: tuple[KernClient, _MockServerState]) -> None:
    cli, _ = mock_server
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Lookup weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}},
            },
        }
    ]
    sse = cli.chat_completions_stream(
        [{"role": "user", "content": "Weather in Tokyo?"}],
        tools=tools,
    )
    assert sse.done is True
    assert sse.finish_reason == "tool_calls"
    assert len(sse.tool_calls) == 1
    assert sse.tool_calls[0]["function"]["name"] == "get_weather"
    parsed_args = json.loads(sse.tool_calls[0]["function"]["arguments"])
    assert parsed_args == {"city": "Tokyo"}


def test_selftest_midstream_abort_leaves_server_healthy(
    mock_server: tuple[KernClient, _MockServerState],
) -> None:
    cli, _ = mock_server
    got = cli.raw_post_abort(
        "/v1/completions",
        {"model": "Inkling-2-Mini", "prompt": "Stream abort test", "max_tokens": 2048, "temperature": 0},
        chunks_before_close=3,
    )
    assert got == 3
    follow_up = cli.completions("The capital of France is", max_tokens=8, temperature=0)
    assert follow_up["choices"][0]["text"] == " Paris."


def test_selftest_midstream_sse_error_event(mock_server: tuple[KernClient, _MockServerState]) -> None:
    cli, _ = mock_server
    sse = cli.completions_stream("__TRIGGER_SSE_ERROR__", max_tokens=8, temperature=0)
    assert sse.done is True
    assert len(sse.errors) == 1
    assert sse.errors[0]["type"] == "InternalServerError"
    assert sse.errors[0]["code"] == 500


def test_selftest_openmetrics_prometheus_delta_and_mtp_math(
    mock_server: tuple[KernClient, _MockServerState],
) -> None:
    cli, _ = mock_server
    p1_ids = cli.tokenize(prompt="Repeat prefix cache check")["tokens"]
    suffix_ids = cli.tokenize(prompt=" next")["tokens"]

    before = cli.metrics()
    # 1. Repeating the exact same /v1/completions prompt must miss (stateful finish() keys on prompt + output)
    r1 = cli.completions(p1_ids, max_tokens=16, temperature=0, return_token_ids=True)
    r_same = cli.completions(p1_ids, max_tokens=16, temperature=0, return_token_ids=True)
    assert r_same["usage"]["prompt_tokens_details"]["cached_tokens"] == 0

    # 2. max_tokens=1 triggers "not kept" (MTP overshoot) -> continuation misses
    short_ids = cli.tokenize(prompt="Short max_tokens=1 check")["tokens"]
    r_short = cli.completions(short_ids, max_tokens=1, temperature=0, return_token_ids=True)
    r_short_cont = cli.completions(
        short_ids + r_short["choices"][0]["token_ids"][:-1] + suffix_ids,
        max_tokens=16,
        temperature=0,
    )
    assert r_short_cont["usage"]["prompt_tokens_details"]["cached_tokens"] == 0

    # 3. Full-history Turn 2 continuation hits r1's finish() snapshot
    t2_ids = p1_ids + r1["choices"][0]["token_ids"][:-1] + suffix_ids
    r2 = cli.completions(t2_ids, max_tokens=16, temperature=0, return_token_ids=True)
    after = cli.metrics()

    assert r2["usage"]["prompt_tokens_details"]["cached_tokens"] > 0
    assert metric_delta(before, after, "vllm:prefix_cache_queries_total") == 5.0
    assert metric_delta(before, after, "vllm:prefix_cache_hits_total") == 1.0
    mtp = compute_mtp_stats(before, after)
    assert mtp["drafts"] == 10.0
    assert mtp["accepted_tokens"] == 30.0
    assert mtp["tokens_per_round"] == pytest.approx((30.0 + 10.0) / 10.0)
    assert mtp["accept_rate"] == pytest.approx(30.0 / 70.0)

    samples = parse_openmetrics_samples(cli.metrics_raw())
    assert any(
        s.name == "vllm:spec_decode_num_accepted_tokens_per_pos_total" and s.labels.get("position") == "0"
        for s in samples
    )
    trace_m = cli.trace_metrics()
    assert trace_m["kern_steps_total"] == 5.0
    assert cli.trace_rotate()["closed"] == ["trace_0001.json"]
    parsed_nan = parse_openmetrics("metric_nan NaN\nmetric_ok 3.5\n")
    assert parsed_nan == {"metric_ok": 3.5}


def test_selftest_negative_errors_vision_logprobs_overflow_bridge(
    mock_server: tuple[KernClient, _MockServerState],
) -> None:
    cli, _ = mock_server
    for bad_msgs in (
        build_vision_image_url_messages(),
        build_vision_base64_messages(),
        build_audio_messages(),
    ):
        with pytest.raises(HttpError) as exc:
            cli.chat_completions(bad_msgs, max_tokens=8)
        assert exc.value.status == 400
        assert exc.value.payload is not None
        assert exc.value.payload["error"]["type"] == "BadRequestError"

    with pytest.raises(HttpError) as exc_lp:
        cli.completions("hello", max_tokens=4, logprobs=True)
    assert exc_lp.value.status == 400

    with pytest.raises(HttpError) as exc_ovf:
        cli.completions([42] * MAX_MODEL_LEN, max_tokens=1)
    assert exc_ovf.value.status == 400
    assert exc_ovf.value.payload["error"]["param"] == "input_tokens"

    with pytest.raises(HttpError) as exc_unk:
        cli.completions("hello", model="nonexistent-model", max_tokens=4)
    assert exc_unk.value.status == 404

    with pytest.raises(HttpError) as exc_bridge:
        cli.completions("hello", max_tokens=4, min_tokens=1)
    assert exc_bridge.value.status == 500

    status_msg, _, _ = cli.raw_request("POST", "/v1/messages", body={"model": "Inkling-2-Mini"})
    assert status_msg == 501


def test_selftest_parquet_recording_and_questions_validation(tmp_path: Path) -> None:
    rec_path = tmp_path / "baseline_recording.parquet"
    rec_schema = pa.schema(
        [
            pa.field("source", pa.string()),
            pa.field("prompt", pa.list_(pa.int32())),
            pa.field("output", pa.list_(pa.int32())),
            pa.field("top_ids", pa.list_(pa.list_(pa.int32()))),
            pa.field("top_logprobs", pa.list_(pa.list_(pa.float32()))),
            pa.field("noise", pa.list_(pa.float64())),
        ],
        metadata={b"producer": b"vllm-mock", b"temperature": b"0.0"},
    )
    rec_table = pa.table(
        {
            "source": pa.array(["q1", "q2"], type=pa.string()),
            "prompt": pa.array([[10, 20], [30, 40, 50]], type=pa.list_(pa.int32())),
            "output": pa.array([[100, EOS_TOKEN_ID], [200]], type=pa.list_(pa.int32())),
            "top_ids": pa.array(
                [[[100, 101], [EOS_TOKEN_ID, 102]], [[200, 201]]],
                type=pa.list_(pa.list_(pa.int32())),
            ),
            "top_logprobs": pa.array(
                [[[-0.1, -2.5], [-0.2, -3.1]], [[-0.05, -4.0]]],
                type=pa.list_(pa.list_(pa.float32())),
            ),
            "noise": pa.array([[0.001, 0.002], [0.0015]], type=pa.list_(pa.float64())),
        },
        schema=rec_schema,
    )
    pq.write_table(rec_table, rec_path)

    q_path = tmp_path / "questions.parquet"
    q_table = pa.table(
        {
            "source": pa.array(["q1"], type=pa.string()),
            "messages": pa.array(['[{"role":"user","content":"2+2?"}]'], type=pa.string()),
        }
    )
    pq.write_table(q_table, q_path)

    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(
        json.dumps({"tokens_per_round": 3.25, "end_to_end_tokens_per_second": 140.5}),
        encoding="utf-8",
    )

    info = validate_recording_parquet(rec_path)
    assert info["rows"] == 2
    assert info["has_noise"] is True
    assert info["metadata"]["producer"] == "vllm-mock"
    assert validate_questions_parquet(q_path)["rows"] == 1

    discovered = discover_baselines(tmp_path)
    assert discovered.recording_parquet == rec_path
    assert discovered.questions_parquet == q_path
    assert discovered.metrics_json == metrics_path

    # Bad schema must fail validation
    bad_path = tmp_path / "bad_recording.parquet"
    pq.write_table(pa.table({"source": ["x"], "output": [[1]]}), bad_path)
    with pytest.raises(AssertionError):
        validate_recording_parquet(bad_path)


def test_selftest_vocab_bounds_and_near_tie_ulp_helpers() -> None:
    assert_valid_token_ids([0, 42, EOS_TOKEN_ID, UNPADDED_VOCAB_SIZE - 1])
    with pytest.raises(AssertionError):
        assert_valid_token_ids([UNPADDED_VOCAB_SIZE])
    with pytest.raises(AssertionError):
        assert_valid_token_ids([-1])
    assert ulp_of_bf16(1.0) == pytest.approx(2.0**-7)
    assert is_near_tie_bf16(10.0, 9.9, max_ulps=4.0) is True
    assert is_near_tie_bf16(10.0, 5.0, max_ulps=4.0) is False


def test_selftest_weights_config_validation(tmp_path: Path) -> None:
    # Real checkpoint config validation if weights exist on host
    real_weights = Path("/workspace/weights/Inkling-2-Mini")
    if real_weights.exists():
        res = validate_weights_configs(real_weights)
        assert res["present"] is True
        assert res["model_max_length"] == MAX_MODEL_LEN

    # Synthetic weights config with max_position_embeddings overlay
    wdir = tmp_path / "weights"
    wdir.mkdir()
    (wdir / "config.json").write_text(
        json.dumps(
            {
                "model_type": "inkling2_loop",
                "eos_token_id": EOS_TOKEN_ID,
                "max_position_embeddings": MAX_MODEL_LEN,
                "text_config": {
                    "model_max_length": MAX_MODEL_LEN,
                    "unpadded_vocab_size": UNPADDED_VOCAB_SIZE,
                    "vocab_size": 201024,
                    "sconv_kernel_size": 4,
                    "sliding_window_size": 1024,
                    "log_scaling_n_floor": 8192,
                },
                "mtp_config": {"block_size": 8},
            }
        ),
        encoding="utf-8",
    )
    (wdir / "generation_config.json").write_text(json.dumps({"eos_token_id": EOS_TOKEN_ID}), encoding="utf-8")
    (wdir / "tokenizer_config.json").write_text(json.dumps({"model_max_length": MAX_MODEL_LEN}), encoding="utf-8")
    synth = validate_weights_configs(wdir)
    assert synth["has_max_position_embeddings"] is True
    assert synth["warnings"] == []


def test_selftest_run_e2e_dry_run_cli(tmp_path: Path) -> None:
    if os.environ.get("KERN_E2E_IN_DRY_RUN") == "1":
        pytest.skip("Already running inside run_e2e.py --dry-run")
    json_rep = tmp_path / "dryrun.json"
    junit_rep = tmp_path / "dryrun.xml"
    cmd = [
        sys.executable,
        str(_E2E_DIR / "run_e2e.py"),
        "--dry-run",
        "--json-report",
        str(json_rep),
        "--junit-xml",
        str(junit_rep),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"run_e2e.py --dry-run failed:\n{proc.stdout}\n{proc.stderr}"
    payload = json.loads(json_rep.read_text(encoding="utf-8"))
    assert payload["ok"] is True
    assert payload["collected_counts"]["tier1"] >= 65
    assert payload["collected_counts"]["tier2"] >= 65
    assert payload["collected_counts"]["tier3"] >= 13
    assert payload["collected_counts"]["tier4"] >= 7
    assert junit_rep.exists()


def test_selftest_openmetrics_braces_in_labels_and_exemplars() -> None:
    text = "\n".join(
        [
            'http_requests_total{route="/v1/{model}",code="200"} 42',
            r'm{route="/a/\"{x}\"/b\\",ok="1"} 7.5 1700000000.123 # {trace_id="t1"} 1.0',
            'plain 12 # {trace="x"} 1.0',
            'bad{a="1" 5',
        ]
    )
    samples = parse_openmetrics_samples(text)
    by_name = {s.name: s for s in samples}
    assert set(by_name) == {"http_requests_total", "m", "plain"}
    assert by_name["http_requests_total"].labels == {"route": "/v1/{model}", "code": "200"}
    assert by_name["http_requests_total"].value == 42.0
    assert by_name["m"].labels == {"route": '/a/"{x}"/b\\', "ok": "1"}
    assert by_name["m"].value == 7.5
    assert by_name["plain"].labels == {}
    assert by_name["plain"].value == 12.0
    assert parse_openmetrics('http_requests_total{route="/v1/{model}"} 42\n') == {"http_requests_total": 42.0}


def test_selftest_sse_error_variants_and_choice_prompt_ids() -> None:
    lines = [
        b'data: {"error": "plain string error"}\n',
        b'data: {"object": "error", "message": "vllm error", "code": 500}\n',
        b'data: {"error": {"message": "OOM"}}\n',
        b'data: {"error": null, "choices": [{"index": 0, "text": "", "prompt_token_ids": [10, 20, 30]}]}\n',
        b'data: {"error": null, "choices": [{"index": 0, "text": "ok", "finish_reason": "stop"}]}\n',
        b"data: [DONE]\n",
    ]
    res = _assemble_sse_lines(lines)
    assert res.done is True
    assert len(res.errors) == 3
    assert res.errors[0] == {"message": "plain string error"}
    assert res.errors[1]["object"] == "error"
    assert res.errors[2] == {"message": "OOM"}
    assert res.prompt_token_ids == [10, 20, 30]
    assert res.content == "ok"
    assert res.finish_reason == "stop"
    assert res.ttft_s is not None and res.ttft_s >= 0.0
    assert res.elapsed_s is not None and res.elapsed_s >= res.ttft_s


def test_selftest_kern_cli_capacity_and_missing_generated_ids(tmp_path: Path) -> None:
    fake_kern = tmp_path / "fake_kern.sh"
    fake_kern.write_text(
        """#!/usr/bin/env python3
import os, sys
args = sys.argv[1:]
mode = os.environ.get("FAKE_KERN_MODE", "ok")
if mode == "ok":
    assert "--capacity" in args
    idx = args.index("--capacity")
    assert args[idx + 1] == "4096"
    print("generated ids: [101, 202]", file=sys.stderr)
    sys.exit(0)
elif mode == "empty_ids":
    print("generated ids: []", file=sys.stderr)
    sys.exit(0)
elif mode == "missing_ids":
    print("prefill done without generated ids log", file=sys.stderr)
    sys.exit(0)
else:
    print("boom on gpu", file=sys.stderr)
    sys.exit(1)
""",
        encoding="utf-8",
    )
    fake_kern.chmod(0o755)
    cli = KernCLI(fake_kern)
    manifest = tmp_path / "model.json"
    weights = tmp_path / "weights"
    manifest.write_text("{}", encoding="utf-8")
    weights.mkdir()

    old_mode = os.environ.get("FAKE_KERN_MODE")
    try:
        os.environ["FAKE_KERN_MODE"] = "ok"
        proc, ids = cli.kern_run(manifest, weights, prompt_ids=[1, 2], steps=2, capacity=4096)
        assert proc.returncode == 0
        assert ids == [101, 202]

        os.environ["FAKE_KERN_MODE"] = "empty_ids"
        proc_empty, ids_empty = cli.kern_run(manifest, weights, prompt_ids=[1, 2], steps=2)
        assert proc_empty.returncode == 0
        assert ids_empty == []

        os.environ["FAKE_KERN_MODE"] = "missing_ids"
        with pytest.raises(RuntimeError, match="without 'generated ids:'"):
            cli.kern_run(manifest, weights, prompt_ids=[1, 2], steps=2)

        os.environ["FAKE_KERN_MODE"] = "fail"
        with pytest.raises(RuntimeError, match="kern run failed"):
            cli.kern_run(manifest, weights, prompt_ids=[1, 2], steps=2)
    finally:
        if old_mode is None:
            os.environ.pop("FAKE_KERN_MODE", None)
        else:
            os.environ["FAKE_KERN_MODE"] = old_mode


def test_selftest_corrupt_parquet_and_explicit_baseline_errors(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.parquet"
    corrupt.write_bytes(b"NOT_PARQUET")
    empty_pq = tmp_path / "empty.parquet"
    empty_pq.write_bytes(b"")

    for bad in (corrupt, empty_pq):
        with pytest.raises(pa.ArrowException):
            validate_recording_parquet(bad)
        with pytest.raises(pa.ArrowException):
            validate_questions_parquet(bad)

    with pytest.raises(FileNotFoundError):
        discover_baselines(tmp_path, explicit_parquet=tmp_path / "does_not_exist.parquet")
    with pytest.raises(pa.ArrowException):
        discover_baselines(tmp_path, explicit_parquet=corrupt)
    with pytest.raises(FileNotFoundError):
        discover_baselines(tmp_path, explicit_metrics=tmp_path / "missing_metrics.json")
    bad_metrics = tmp_path / "bad_metrics.json"
    bad_metrics.write_text('{"unrelated": 1}', encoding="utf-8")
    with pytest.raises(ValueError):
        discover_baselines(tmp_path, explicit_metrics=bad_metrics)

    valid_pq = tmp_path / "valid_recording.parquet"
    schema = pa.schema(
        [
            pa.field("source", pa.string()),
            pa.field("prompt", pa.list_(pa.int32())),
            pa.field("output", pa.list_(pa.int32())),
            pa.field("top_ids", pa.list_(pa.list_(pa.int32()))),
            pa.field("top_logprobs", pa.list_(pa.list_(pa.float32()))),
        ]
    )
    table = pa.table(
        {
            "source": pa.array(["q1"], type=pa.string()),
            "prompt": pa.array([[10, 20]], type=pa.list_(pa.int32())),
            "output": pa.array([[100]], type=pa.list_(pa.int32())),
            "top_ids": pa.array([[[100, 101]]], type=pa.list_(pa.list_(pa.int32()))),
            "top_logprobs": pa.array([[[-0.1, -2.0]]], type=pa.list_(pa.list_(pa.float32()))),
        },
        schema=schema,
    )
    pq.write_table(table, valid_pq)

    discovered = discover_baselines(tmp_path)
    assert discovered.recording_parquet == valid_pq
    assert any("corrupt" in w.lower() or "ignoring" in w.lower() for w in discovered.warnings)


def test_selftest_extract_baseline_accept_pct_scaling() -> None:
    m_pct = extract_baseline_mtp_metrics({"accept_pct": 1.0})
    assert m_pct["tokens_per_round"] == pytest.approx(1.0 + 7.0 * 0.01)
    m_rate = extract_baseline_mtp_metrics({"acceptance_rate": 0.5})
    assert m_rate["tokens_per_round"] == pytest.approx(1.0 + 7.0 * 0.5)

    m_bench = extract_baseline_mtp_metrics(
        {
            "entries": [
                {
                    "concurrency": 1,
                    "output_throughput_tok_s": 184.6189,
                    "ttft_mean_ms": 48.827,
                    "ttft_p50_ms": 49.322,
                    "mtp_mean_acceptance_length": 2.9649,
                    "mtp_acceptance_rate": 0.2807,
                    "requests": [{"ttft_s": 0.0449}],
                }
            ]
        }
    )
    assert m_bench["tokens_per_round"] == pytest.approx(2.9649)
    assert m_bench["accept_pct"] == pytest.approx(0.2807)
    assert m_bench["tokens_per_second"] == pytest.approx(184.6189)
    assert m_bench["ttft_s"] == pytest.approx(0.049322)


def test_selftest_nonempty_token_ids_and_near_tie_probe_verifier(tmp_path: Path) -> None:
    with pytest.raises(AssertionError):
        assert_valid_token_ids([])
    assert_valid_token_ids([], require_non_empty=False)

    manifest_f32 = tmp_path / "m_f32.json"
    manifest_f32.write_text(
        json.dumps({"constants": {"V": 8}, "buffers": {"logits": {"dtype": "f32", "shape": ["seqs", "V"]}}}),
        encoding="utf-8",
    )
    assert manifest_logits_spec(manifest_f32) == ("f32", 8)

    # f32 probe dump: tok 1 = 10.0, tok 2 = 9.9 (near tie <= 4 BF16 ULPs), tok 3 = 5.0 (confident flip)
    f32_vals = [0.0, 10.0, 9.9, 5.0, 0.0, 0.0, 0.0, 0.0]
    f32_path = tmp_path / "chunk.logits.bin"
    f32_path.write_bytes(struct.pack("<8f", *f32_vals))
    assert read_probe_logits_row0(f32_path, dtype="f32", vocab_size=8) == pytest.approx(f32_vals)

    # bf16 probe dump (high 2 bytes of IEEE float32)
    bf16_raw = b"".join(struct.pack("<f", v)[2:4] for v in f32_vals)
    bf16_path = tmp_path / "bf16.logits.bin"
    bf16_path.write_bytes(bf16_raw)
    bf16_read = read_probe_logits_row0(bf16_path, dtype="bf16", vocab_size=8)
    assert bf16_read[1] == pytest.approx(10.0)
    assert bf16_read[2] == pytest.approx(9.9, abs=0.1)

    tok_path = tmp_path / "chunk.tokens.bin"
    tok_path.write_bytes(struct.pack("<q", 1))
    assert 1 in read_probe_first_token(tok_path)

    class _FakeProbeCLI:
        def kern_probe(self, **kwargs: Any) -> subprocess.CompletedProcess[str]:
            pdir = Path(kwargs["probe_dir"])
            pdir.mkdir(parents=True, exist_ok=True)
            (pdir / "chunk.logits.bin").write_bytes(struct.pack("<8f", *f32_vals))
            (pdir / "chunk.tokens.bin").write_bytes(struct.pack("<q", 1))
            return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

        def kern_run(self, **kwargs: Any) -> tuple[subprocess.CompletedProcess[str], list[int]]:
            return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr=""), []

    weights = tmp_path / "weights"
    weights.mkdir()
    # Near-tie (tok 2 vs tok 1) passes
    verify_greedy_parity_with_near_tie(
        kern_cli=_FakeProbeCLI(),
        manifest=manifest_f32,
        weights=weights,
        prompt_ids=[10, 20],
        serve_ids=[2],
        run_ids=[1],
        steps=1,
    )
    # Confident flip (tok 3 vs tok 1) fails
    with pytest.raises(AssertionError, match="Confident greedy parity flip"):
        verify_greedy_parity_with_near_tie(
            kern_cli=_FakeProbeCLI(),
            manifest=manifest_f32,
            weights=weights,
            prompt_ids=[10, 20],
            serve_ids=[3],
            run_ids=[1],
            steps=1,
        )


def test_selftest_concurrent_map_barrier(mock_server: tuple[KernClient, _MockServerState]) -> None:
    cli, _ = mock_server
    lock = threading.Lock()
    active = 0
    peak_active = 0

    def _work(item: int) -> int:
        nonlocal active, peak_active
        with lock:
            active += 1
            peak_active = max(peak_active, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return item * 2

    out = cli.concurrent_map(_work, list(range(8)), max_workers=4, synchronize=True, barrier_timeout=10.0)
    assert out == [i * 2 for i in range(8)]
    assert peak_active == 4


def test_selftest_run_e2e_mode_partitions_and_tier4_skip_gate(tmp_path: Path) -> None:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{_REPO_ROOT}{os.pathsep}{env['PYTHONPATH']}" if env.get("PYTHONPATH") else str(_REPO_ROOT)
    tier_files = [
        str(_E2E_DIR / "test_tier1_features.py"),
        str(_E2E_DIR / "test_tier2_boundaries.py"),
        str(_E2E_DIR / "test_tier3_combinations.py"),
        str(_E2E_DIR / "test_tier4_workloads.py"),
    ]

    def _collect_expr(expr: str) -> list[str]:
        cmd = [
            sys.executable,
            "-m",
            "pytest",
            *tier_files,
            "-m",
            expr,
            "--collect-only",
            "-q",
            "-c",
            str(_E2E_DIR / "pytest.ini"),
        ]
        p = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(_E2E_DIR))
        assert p.returncode == 0, f"collect-only {expr!r} failed:\n{p.stdout}\n{p.stderr}"
        return [line.strip() for line in p.stdout.splitlines() if "::" in line and not line.startswith("=")]

    http_nodes = _collect_expr("not cli")
    cli_nodes = _collect_expr("cli and not hybrid")
    hybrid_nodes = _collect_expr("hybrid")
    assert len(http_nodes) == 142
    assert len(cli_nodes) == 8
    assert len(hybrid_nodes) == 9
    assert len(http_nodes) + len(cli_nodes) + len(hybrid_nodes) == 159

    # Verify _run_live fails if any Tier 4 test skips even when pytest returns 0
    json_out = tmp_path / "live_report.json"
    args = types.SimpleNamespace(
        base_url="http://127.0.0.1:8000",
        trace_url="",
        tier="4",
        mode="all",
        baseline_dir=str(tmp_path),
        baseline_parquet="",
        baseline_metrics="",
        manifest=str(tmp_path / "m.json"),
        kernels=str(tmp_path / "k"),
        weights=str(tmp_path / "w"),
        kern_bin=str(tmp_path / "kern"),
        model_id="",
        kl_limit="",
        max_ctx_probe=65536,
        gpu=0,
        junit_xml="",
        json_report=str(json_out),
    )

    def _fake_subprocess_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        j_idx = cmd.index("--junitxml") + 1
        j_path = Path(cmd[j_idx])
        j_path.write_text(
            '<testsuite name="pytest" tests="8" failures="0" errors="0" skipped="1">'
            '<testcase classname="tests.e2e.test_tier4_workloads" name="test_t4_w01_e2e_logits_gate">'
            '<skipped message="missing baseline"/>'
            "</testcase>"
            "</testsuite>",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    with mock.patch.object(run_e2e.subprocess, "run", side_effect=_fake_subprocess_run):
        rc = run_e2e._run_live(args, [])

    assert rc == 1
    rep = json.loads(json_out.read_text(encoding="utf-8"))
    assert rep["ok"] is False
    assert rep["returncode"] == 1
    assert rep["pytest_returncode"] == 0
    assert rep["tier4_skipped"] == 1
