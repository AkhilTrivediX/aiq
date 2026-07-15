# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for the semantic_ontology_query NAT registration."""

import json
import sys
import types
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import httpx
import pytest
from pydantic import BaseModel
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# Stub heavy optional dependencies before importing the module under test.
# ---------------------------------------------------------------------------

# Stub langchain_core so the import doesn't require the full package.
_lc_callbacks = types.ModuleType("langchain_core.callbacks")
_lc_callbacks.adispatch_custom_event = AsyncMock(return_value=None)
sys.modules.setdefault("langchain_core", types.ModuleType("langchain_core"))
sys.modules["langchain_core.callbacks"] = _lc_callbacks

# Stub nat packages.
for _mod in [
    "nat",
    "nat.builder",
    "nat.builder.builder",
    "nat.builder.context",
    "nat.builder.function_info",
    "nat.cli",
    "nat.cli.register_workflow",
    "nat.data_models",
    "nat.data_models.function",
    "nat.data_models.intermediate_step",
]:
    sys.modules.setdefault(_mod, types.ModuleType(_mod))

_nat_builder = sys.modules["nat.builder.builder"]
_nat_builder.Builder = object

_fi = types.ModuleType("nat.builder.function_info")


class _FunctionInfo:
    @staticmethod
    def from_fn(fn, *, description=None):
        return fn


_fi.FunctionInfo = _FunctionInfo
sys.modules["nat.builder.function_info"] = _fi

_rw = types.ModuleType("nat.cli.register_workflow")
_rw.register_function = lambda config_type: lambda fn: fn
sys.modules["nat.cli.register_workflow"] = _rw

_fn = types.ModuleType("nat.data_models.function")


class _FunctionBaseConfig(BaseModel):
    def __init_subclass__(cls, name: str = "", **kwargs):
        super().__init_subclass__(**kwargs)


_fn.FunctionBaseConfig = _FunctionBaseConfig
sys.modules["nat.data_models.function"] = _fn

# Stub aiq_agent.auth before register imports it at call-time.
_aiq_agent = types.ModuleType("aiq_agent")
_aiq_auth = types.ModuleType("aiq_agent.auth")
_aiq_auth.get_auth_token = MagicMock(return_value="test-token")
sys.modules.setdefault("aiq_agent", _aiq_agent)
sys.modules["aiq_agent.auth"] = _aiq_auth

from semantic_ontology_query.register import SemanticOntologyQueryToolConfig  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sse_lines(*events: dict) -> list[str]:
    """Build SSE line lists from event dicts."""
    lines = []
    for ev in events:
        lines.append(f"data: {json.dumps(ev)}")
    lines.append("data: [DONE]")
    return lines


def _make_config(base_url: str = "https://so.example.com", timeout: float = 5.0) -> SemanticOntologyQueryToolConfig:
    return SemanticOntologyQueryToolConfig(base_url=base_url, timeout_seconds=timeout)


async def _run_query(config: SemanticOntologyQueryToolConfig, question: str = "test question") -> str:
    """Drive the async generator and invoke the inner function."""
    gen = semantic_ontology_query(config, builder=None)
    fn = await gen.__anext__()
    try:
        await gen.__anext__()
    except StopAsyncIteration:
        pass
    return await fn(question)


# Import after stubs are in place.
from semantic_ontology_query.register import semantic_ontology_query  # noqa: E402

# ---------------------------------------------------------------------------
# Config validation tests
# ---------------------------------------------------------------------------


class TestSemanticOntologyQueryToolConfigValidation:
    def test_https_url_accepted(self):
        cfg = SemanticOntologyQueryToolConfig(base_url="https://so.example.com")
        assert cfg.base_url == "https://so.example.com"

    def test_http_url_rejected(self):
        with pytest.raises(ValidationError, match="HTTPS"):
            SemanticOntologyQueryToolConfig(base_url="http://so.example.com")

    def test_empty_url_accepted(self):
        cfg = SemanticOntologyQueryToolConfig(base_url="")
        assert cfg.base_url == ""

    def test_env_var_http_rejected(self, monkeypatch):
        monkeypatch.setenv("SEMANTIC_ONTOLOGY_BASE_URL", "http://bad.example.com")
        with pytest.raises(ValidationError, match="HTTPS"):
            SemanticOntologyQueryToolConfig()

    def test_env_var_https_accepted(self, monkeypatch):
        monkeypatch.setenv("SEMANTIC_ONTOLOGY_BASE_URL", "https://good.example.com")
        cfg = SemanticOntologyQueryToolConfig()
        assert cfg.base_url == "https://good.example.com"


# ---------------------------------------------------------------------------
# Fixtures for HTTP-layer tests
# ---------------------------------------------------------------------------


def _mock_stream_response(status_code: int, sse_lines: list[str]):
    """Build a mock async context manager that yields SSE lines via aiter_lines."""
    response = MagicMock()
    response.status_code = status_code
    response.headers = {}

    async def _aiter_lines():
        for line in sse_lines:
            yield line

    response.aiter_lines = _aiter_lines
    response.aclose = AsyncMock()
    response.aread = AsyncMock(return_value=b"")

    ctx = AsyncMock()
    ctx.__aenter__ = AsyncMock(return_value=response)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx, response


def _mock_client(stream_ctx):
    client = MagicMock()
    client.stream = MagicMock(return_value=stream_ctx)
    client_ctx = AsyncMock()
    client_ctx.__aenter__ = AsyncMock(return_value=client)
    client_ctx.__aexit__ = AsyncMock(return_value=False)
    return client_ctx, client


# ---------------------------------------------------------------------------
# SSE parsing tests
# ---------------------------------------------------------------------------


class TestSSEParsing:
    @pytest.fixture(autouse=True)
    def _patch_emit_status(self):
        with patch("semantic_ontology_query.register._emit_status", new_callable=AsyncMock) as mock:
            self._emit_status = mock
            yield

    async def test_result_event_returns_answer(self):
        lines = _make_sse_lines({"type": "result", "answer": {"response": "42 widgets", "sql_code": None}})
        stream_ctx, _ = _mock_stream_response(200, lines)
        client_ctx, _ = _mock_client(stream_ctx)
        with patch("httpx.AsyncClient", return_value=client_ctx):
            result = await _run_query(_make_config())
        assert result == "42 widgets"

    async def test_result_with_sql_code_appended(self):
        lines = _make_sse_lines(
            {
                "type": "result",
                "answer": {"response": "The count is 5", "sql_code": "SELECT COUNT(*) FROM foo"},
            }
        )
        stream_ctx, _ = _mock_stream_response(200, lines)
        client_ctx, _ = _mock_client(stream_ctx)
        with patch("httpx.AsyncClient", return_value=client_ctx):
            result = await _run_query(_make_config())
        assert "The count is 5" in result
        assert "<sql>" in result
        assert "SELECT COUNT(*) FROM foo" in result

    async def test_result_with_sql_result_appended(self):
        lines = _make_sse_lines(
            {
                "type": "result",
                "answer": {
                    "response": "Zero rows",
                    "sql_code": "SELECT 1",
                    "sql_response_from_db": [],
                },
            }
        )
        stream_ctx, _ = _mock_stream_response(200, lines)
        client_ctx, _ = _mock_client(stream_ctx)
        with patch("httpx.AsyncClient", return_value=client_ctx):
            result = await _run_query(_make_config())
        assert "<sql_result>" in result
        assert "[]" in result

    async def test_step_event_emits_status(self):
        lines = _make_sse_lines(
            {"type": "step", "label": "extracting entities", "node": "entity_extractor"},
            {"type": "result", "answer": {"response": "done", "sql_code": None}},
        )
        stream_ctx, _ = _mock_stream_response(200, lines)
        client_ctx, _ = _mock_client(stream_ctx)
        with patch("httpx.AsyncClient", return_value=client_ctx):
            await _run_query(_make_config())
        self._emit_status.assert_awaited_once_with("extracting entities")

    async def test_error_event_returns_error_message(self):
        lines = _make_sse_lines({"type": "error", "message": "internal DB error"})
        stream_ctx, _ = _mock_stream_response(200, lines)
        client_ctx, _ = _mock_client(stream_ctx)
        with patch("httpx.AsyncClient", return_value=client_ctx):
            result = await _run_query(_make_config())
        assert "internal DB error" in result

    async def test_non_json_sse_lines_are_skipped(self):
        lines = [": ping", "", "data: not-json", "data: [DONE]"]
        stream_ctx, _ = _mock_stream_response(200, lines)
        client_ctx, _ = _mock_client(stream_ctx)
        with patch("httpx.AsyncClient", return_value=client_ctx):
            result = await _run_query(_make_config())
        assert "no answer" in result.lower()

    async def test_no_result_event_returns_no_answer(self):
        lines = _make_sse_lines({"type": "step", "label": "working"})
        stream_ctx, _ = _mock_stream_response(200, lines)
        client_ctx, _ = _mock_client(stream_ctx)
        with patch("httpx.AsyncClient", return_value=client_ctx):
            result = await _run_query(_make_config())
        assert "no answer" in result.lower()


# ---------------------------------------------------------------------------
# HTTP error handling tests
# ---------------------------------------------------------------------------


class TestHTTPErrorHandling:
    async def test_401_returns_unauthorized_message(self):
        stream_ctx, response = _mock_stream_response(401, [])
        response.aread = AsyncMock(return_value=b"Unauthorized")
        response.headers = {"www-authenticate": "Bearer realm=test"}
        client_ctx, _ = _mock_client(stream_ctx)
        with patch("httpx.AsyncClient", return_value=client_ctx):
            result = await _run_query(_make_config())
        assert "401" in result or "Unauthorized" in result

    async def test_unexpected_4xx_returns_error(self):
        stream_ctx, response = _mock_stream_response(503, [])
        response.aread = AsyncMock(return_value=b"Service Unavailable")
        client_ctx, _ = _mock_client(stream_ctx)
        with patch("httpx.AsyncClient", return_value=client_ctx):
            result = await _run_query(_make_config())
        assert "503" in result

    async def test_timeout_exception_returns_timeout_message(self):
        client_ctx = MagicMock()
        client_ctx.__aenter__ = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
        client_ctx.__aexit__ = AsyncMock(return_value=False)
        with patch("httpx.AsyncClient", return_value=client_ctx):
            result = await _run_query(_make_config())
        assert "timed out" in result.lower()


# ---------------------------------------------------------------------------
# 409 retry and deadline tests
# ---------------------------------------------------------------------------


class TestRetryAndDeadline:
    async def test_409_retries_and_eventually_returns_busy(self):
        """All attempts return 409 → final "busy" message (no real sleep)."""
        stream_ctx, _ = _mock_stream_response(409, [])
        client_ctx, client = _mock_client(stream_ctx)
        with patch("httpx.AsyncClient", return_value=client_ctx), patch("asyncio.sleep", new_callable=AsyncMock):
            result = await _run_query(_make_config())
        assert "busy" in result.lower()

    async def test_409_then_success_returns_answer(self):
        """First attempt returns 409, second returns a valid result."""
        result_lines = _make_sse_lines({"type": "result", "answer": {"response": "got it", "sql_code": None}})

        call_count = 0

        def _stream_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                ctx, _ = _mock_stream_response(409, [])
                return ctx
            ctx, _ = _mock_stream_response(200, result_lines)
            return ctx

        client = MagicMock()
        client.stream = MagicMock(side_effect=_stream_side_effect)
        client_ctx = AsyncMock()
        client_ctx.__aenter__ = AsyncMock(return_value=client)
        client_ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("httpx.AsyncClient", return_value=client_ctx), patch("asyncio.sleep", new_callable=AsyncMock):
            result = await _run_query(_make_config())
        assert result == "got it"
        assert call_count == 2

    async def test_overall_deadline_short_circuits_retries(self):
        """When deadline is already past, the loop should not retry."""
        stream_ctx, _ = _mock_stream_response(409, [])
        client_ctx, _ = _mock_client(stream_ctx)

        # Make time.monotonic() always return a value past the deadline.
        import time as _time

        original_monotonic = _time.monotonic
        base = original_monotonic()
        call_count = 0

        def _fast_clock():
            nonlocal call_count
            call_count += 1
            # After the first call (which sets overall_deadline), return deadline+1.
            if call_count == 1:
                return base
            return base + 1000.0

        with (
            patch("httpx.AsyncClient", return_value=client_ctx),
            patch("semantic_ontology_query.register.time") as mock_time,
            patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep,
        ):
            mock_time.monotonic = _fast_clock
            result = await _run_query(_make_config(timeout=0.001))

        assert "timed out" in result.lower()
        mock_sleep.assert_not_called()

    async def test_missing_base_url_returns_config_error(self):
        cfg = SemanticOntologyQueryToolConfig(base_url="")
        result = await _run_query(cfg)
        assert "not configured" in result.lower() or "base url" in result.lower()

    async def test_missing_auth_token_returns_auth_error(self):
        with patch("aiq_agent.auth.get_auth_token", return_value=None):
            result = await _run_query(_make_config())
        assert "authentication" in result.lower() or "token" in result.lower()
