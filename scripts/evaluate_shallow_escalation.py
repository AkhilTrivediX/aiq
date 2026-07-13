# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the production shallow-escalation assessor against the committed quality set.

This is an opt-in, assessment-only evaluation. It makes exactly one tool-free
model invocation per case and never starts the shallow tool loop, clarifier, or
deep-research workflow.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from datetime import UTC
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Literal

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from pydantic import ValidationError

from aiq_agent.agents.shallow_researcher.escalation import ShallowEscalationAssessor
from aiq_agent.agents.shallow_researcher.models import ShallowEscalationAssessment
from aiq_agent.common import extract_json
from aiq_agent.evaluation.shallow_escalation import EscalationCase
from aiq_agent.evaluation.shallow_escalation import acceptance_passed
from aiq_agent.evaluation.shallow_escalation import build_summary
from aiq_agent.evaluation.shallow_escalation import load_cases

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FIXTURE = REPO_ROOT / "tests/aiq_agent/agents/shallow_researcher/fixtures/escalation_cases.json"
DEFAULT_OUTPUT = Path("/tmp/aiq-shallow-escalation-eval.json")
DEFAULT_MODEL = "nvidia/nemotron-3-super-120b-a12b"
DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"


def _route(status: str) -> Literal["shallow", "escalate"]:
    return "shallow" if status == "sufficient" else "escalate"


def inspect_raw_assessment(text: str) -> tuple[bool, str | None]:
    """Report whether raw model output independently satisfies the production schema."""
    parsed = extract_json(text)
    if parsed is None:
        return False, None
    try:
        assessment = ShallowEscalationAssessment.model_validate(parsed)
    except ValidationError:
        return False, parsed.get("status") if isinstance(parsed, dict) else None
    return True, assessment.status


def _message_text(message: Any) -> str:
    text = getattr(message, "text", None)
    if isinstance(text, str):
        return text
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(block.get("text", "")) for block in content if isinstance(block, dict) and block.get("type") == "text"
        )
    return str(content or "")


def _usage_from_mapping(mapping: dict[str, Any] | None) -> tuple[int | None, int | None]:
    if not mapping:
        return None, None
    input_tokens = mapping.get("input_tokens", mapping.get("prompt_tokens"))
    output_tokens = mapping.get("output_tokens", mapping.get("completion_tokens"))
    return (
        int(input_tokens) if input_tokens is not None else None,
        int(output_tokens) if output_tokens is not None else None,
    )


class AssessmentCaptureCallback(BaseCallbackHandler):
    """Capture one assessor invocation's raw output and provider usage metadata."""

    def __init__(self) -> None:
        self._run_ids: set[str] = set()
        self._anonymous_starts = 0
        self.raw_output = ""
        self.input_tokens: int | None = None
        self.output_tokens: int | None = None
        self.model_name: str | None = None
        self.error_type: str | None = None

    @property
    def call_count(self) -> int:
        return len(self._run_ids) + self._anonymous_starts

    def _record_start(self, run_id: Any = None) -> None:
        if run_id is None:
            self._anonymous_starts += 1
        else:
            self._run_ids.add(str(run_id))

    def on_chat_model_start(self, serialized: dict[str, Any], messages: list[list[Any]], **kwargs: Any) -> None:
        del serialized, messages
        self._record_start(kwargs.get("run_id"))

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> None:
        del serialized, prompts
        self._record_start(kwargs.get("run_id"))

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        del kwargs
        generation = response.generations[0][0] if response.generations and response.generations[0] else None
        message = getattr(generation, "message", None)
        self.raw_output = _message_text(message) if message is not None else str(getattr(generation, "text", ""))

        usage = getattr(message, "usage_metadata", None) if message is not None else None
        self.input_tokens, self.output_tokens = _usage_from_mapping(usage)

        response_metadata = getattr(message, "response_metadata", {}) if message is not None else {}
        if self.input_tokens is None or self.output_tokens is None:
            token_usage = response_metadata.get("token_usage", {}) if isinstance(response_metadata, dict) else {}
            self.input_tokens, self.output_tokens = _usage_from_mapping(token_usage)

        llm_output = response.llm_output or {}
        if self.input_tokens is None or self.output_tokens is None:
            self.input_tokens, self.output_tokens = _usage_from_mapping(llm_output.get("token_usage", {}))
        if isinstance(response_metadata, dict):
            self.model_name = response_metadata.get("model_name") or response_metadata.get("model")
        self.model_name = self.model_name or llm_output.get("model_name")

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        del kwargs
        self.error_type = type(error).__name__


async def evaluate_case(case: EscalationCase, llm: Any) -> dict[str, Any]:
    """Make exactly one production-assessor invocation for one fixture case."""
    capture = AssessmentCaptureCallback()
    assessor = ShallowEscalationAssessor(llm, callbacks=[capture])
    started = time.perf_counter()
    assessment = await assessor.assess(
        query=case.query,
        answer=case.answer,
        source_count=case.source_count,
        tool_budget_exhausted=case.tool_budget_exhausted,
    )
    latency_seconds = time.perf_counter() - started
    parse_success, raw_status = inspect_raw_assessment(capture.raw_output)

    return {
        "id": case.id,
        "query": case.query,
        "answer": case.answer,
        "source_count": case.source_count,
        "tool_budget_exhausted": case.tool_budget_exhausted,
        "expected_status": case.expected_status,
        "predicted_status": assessment.status,
        "raw_status": raw_status,
        "expected_route": _route(case.expected_status),
        "predicted_route": _route(assessment.status),
        "exact_status_correct": assessment.status == case.expected_status,
        "route_correct": _route(assessment.status) == _route(case.expected_status),
        "parse_success": parse_success,
        "logical_model_calls": capture.call_count,
        "input_tokens": capture.input_tokens,
        "output_tokens": capture.output_tokens,
        "latency_seconds": round(latency_seconds, 4),
        "provider_model_name": capture.model_name,
        "model_error_type": capture.error_type,
        "raw_output": capture.raw_output,
        "assessment": assessment.model_dump(),
    }


async def run_evaluation(cases: list[EscalationCase], llm: Any) -> list[dict[str, Any]]:
    """Evaluate sequentially so one fixture maps to one isolated model call."""
    results: list[dict[str, Any]] = []
    for case in cases:
        result = await evaluate_case(case, llm)
        results.append(result)
        print(
            f"{case.id}: expected={case.expected_status} predicted={result['predicted_status']} "
            f"parse={result['parse_success']} calls={result['logical_model_calls']}"
        )
    return results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=os.environ.get("NVIDIA_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--min-escalation-correct", type=int, default=8)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    cases = load_cases(args.fixture)

    # Imported only by the opt-in executable; deterministic unit tests do not
    # initialize a client or require network credentials.
    from langchain_nvidia_ai_endpoints import ChatNVIDIA

    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "base_url": args.base_url,
        "temperature": 0,
    }
    if args.seed is not None:
        llm_kwargs["seed"] = args.seed
    llm = ChatNVIDIA(**llm_kwargs)

    results = asyncio.run(run_evaluation(cases, llm))
    summary = build_summary(results)
    passed = acceptance_passed(summary, min_escalation_correct=args.min_escalation_correct)
    report = {
        "metadata": {
            "generated_at": datetime.now(UTC).isoformat(),
            "fixture": str(args.fixture.resolve()),
            "model": args.model,
            "base_url": args.base_url,
            "seed": args.seed,
            "tool_calls": 0,
            "deep_workflow_calls": 0,
        },
        "acceptance": {
            "passed": passed,
            "required_shallow_route_correct": 10,
            "required_escalation_route_correct": args.min_escalation_correct,
            "required_parse_success": 20,
            "required_exactly_one_call_per_case": True,
        },
        "summary": summary,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "acceptance": report["acceptance"], "summary": summary}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
