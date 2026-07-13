# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the bounded shallow-escalation assessment."""

from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage

from aiq_agent.agents.chat_researcher.agent import _should_escalate_shallow
from aiq_agent.agents.shallow_researcher.escalation import ShallowEscalationAssessment
from aiq_agent.agents.shallow_researcher.escalation import ShallowEscalationAssessor
from aiq_agent.agents.shallow_researcher.escalation import parse_escalation_assessment
from aiq_agent.evaluation.shallow_escalation import EscalationCase
from aiq_agent.evaluation.shallow_escalation import acceptance_passed
from aiq_agent.evaluation.shallow_escalation import build_summary
from aiq_agent.evaluation.shallow_escalation import load_cases

FIXTURE_PATH = Path(__file__).parent / "fixtures/escalation_cases.json"
ESCALATION_CASES = {case.id: case for case in load_cases(FIXTURE_PATH)}


def test_fixture_has_ten_shallow_and_ten_escalation_cases() -> None:
    """The reusable quality set retains its intended balanced acceptance boundary."""
    statuses = [case.expected_status for case in ESCALATION_CASES.values()]

    assert len(ESCALATION_CASES) == 20
    assert statuses.count("sufficient") == 10
    assert statuses.count("material_gap") + statuses.count("material_conflict") == 10


@pytest.mark.parametrize(
    "case_id,case",
    ESCALATION_CASES.items(),
    ids=ESCALATION_CASES.keys(),
)
def test_expected_assessments_parse_and_route(case_id: str, case: EscalationCase) -> None:
    """Fixture labels provide valid examples for deterministic parser/router coverage.

    This does not evaluate model semantics; the opt-in live evaluator supplies
    each query and answer to the production assessor for that purpose.
    """
    assessment = parse_escalation_assessment(case.expected_assessment_json())

    assert assessment.status == case.expected_status, case_id
    assert _should_escalate_shallow(assessment, enabled=True) is (case.expected_status != "sufficient")


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "not json",
        "{}",
        '{"status":"unknown"}',
        '{"status":"material_gap","unresolved_requirement":"Missing cost",'
        '"material_conflict":null,"deep_research_strategy":null}',
        '{"status":"material_conflict","unresolved_requirement":null,'
        '"material_conflict":"Sources disagree","deep_research_strategy":null}',
        '{"status":"sufficient","unresolved_requirement":"Something is missing",'
        '"material_conflict":null,"deep_research_strategy":null}',
    ],
)
def test_parse_escalation_assessment_fails_closed(raw: str) -> None:
    """Malformed output and invalid field combinations never initiate deep research."""
    assessment = parse_escalation_assessment(raw)

    assert assessment == ShallowEscalationAssessment.sufficient()
    assert _should_escalate_shallow(assessment, enabled=True) is False


def test_parse_escalation_assessment_accepts_json_code_fence() -> None:
    """The shared JSON extractor's tolerated code-fence form remains valid."""
    assessment = parse_escalation_assessment(
        """```json
        {
          "status": "material_gap",
          "unresolved_requirement": "The requested cost comparison is unsupported.",
          "material_conflict": null,
          "deep_research_strategy": "Research current pricing for both products."
        }
        ```"""
    )

    assert assessment.status == "material_gap"


def test_routing_policy_honors_disabled_escalation() -> None:
    """Configuration remains authoritative even for a material assessment."""
    assessment = parse_escalation_assessment(ESCALATION_CASES["D1"].expected_assessment_json())

    assert _should_escalate_shallow(assessment, enabled=False) is False


@pytest.mark.asyncio
async def test_assessor_makes_one_bounded_tool_free_call() -> None:
    """Assessment is exactly one bounded model call and never binds or invokes tools."""
    response = AIMessage(content=ESCALATION_CASES["D3"].expected_assessment_json())
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=response)
    llm.bind_tools = MagicMock()
    assessor = ShallowEscalationAssessor(llm)

    assessment = await assessor.assess(
        query=ESCALATION_CASES["D3"].query,
        answer=ESCALATION_CASES["D3"].answer,
        source_count=ESCALATION_CASES["D3"].source_count,
        tool_budget_exhausted=ESCALATION_CASES["D3"].tool_budget_exhausted,
    )

    assert assessment.status == "material_gap"
    llm.ainvoke.assert_awaited_once()
    invoke_kwargs = llm.ainvoke.await_args.kwargs
    assert invoke_kwargs["temperature"] == 0
    assert invoke_kwargs["max_tokens"] == 256
    assert invoke_kwargs["config"]["run_name"] == "shallow_escalation_assessment"
    assert invoke_kwargs["config"]["metadata"] == {
        "aiq_phase": "shallow_escalation_assessment",
        "tool_free": True,
    }
    llm.bind_tools.assert_not_called()


@pytest.mark.asyncio
async def test_assessor_disables_chat_nvidia_thinking() -> None:
    """ChatNVIDIA spends the bounded output budget on JSON rather than reasoning."""

    class FakeChatNVIDIA:
        __module__ = "langchain_nvidia_ai_endpoints.chat_models"

        def __init__(self) -> None:
            self.ainvoke = AsyncMock(return_value=AIMessage(content=ESCALATION_CASES["S1"].expected_assessment_json()))

    llm = FakeChatNVIDIA()
    assessor = ShallowEscalationAssessor(llm)

    await assessor.assess(
        query=ESCALATION_CASES["S1"].query,
        answer=ESCALATION_CASES["S1"].answer,
        source_count=1,
        tool_budget_exhausted=False,
    )

    invoke_kwargs = llm.ainvoke.await_args.kwargs
    assert invoke_kwargs["temperature"] == 0
    assert invoke_kwargs["max_tokens"] == 256
    assert invoke_kwargs["thinking_mode"] is False
    assert invoke_kwargs["config"]["run_name"] == "shallow_escalation_assessment"


@pytest.mark.asyncio
async def test_assessor_parses_text_content_blocks() -> None:
    """Providers may return JSON in LangChain text content blocks."""
    llm = MagicMock()
    llm.ainvoke = AsyncMock(
        return_value=AIMessage(
            content=[
                {"type": "reasoning", "reasoning": "internal"},
                {"type": "text", "text": ESCALATION_CASES["D2"].expected_assessment_json()},
            ]
        )
    )
    assessor = ShallowEscalationAssessor(llm)

    assessment = await assessor.assess(
        query=ESCALATION_CASES["D2"].query,
        answer=ESCALATION_CASES["D2"].answer,
        source_count=2,
        tool_budget_exhausted=False,
    )

    assert assessment.status == "material_conflict"


@pytest.mark.asyncio
async def test_assessor_failure_fails_closed() -> None:
    """A provider failure cannot replace a usable shallow answer with escalation."""
    llm = MagicMock()
    llm.ainvoke = AsyncMock(side_effect=TimeoutError("provider timeout"))
    assessor = ShallowEscalationAssessor(llm)

    assessment = await assessor.assess(
        query="What is CUDA?",
        answer="CUDA is a parallel-computing platform [1].",
        source_count=1,
        tool_budget_exhausted=False,
    )

    assert assessment == ShallowEscalationAssessment.sufficient()


@pytest.mark.asyncio
async def test_missing_assessment_prompt_fails_closed_without_model_call() -> None:
    """A packaging defect is visible but cannot add a useless paid invocation."""
    llm = MagicMock()
    llm.ainvoke = AsyncMock()
    assessor = ShallowEscalationAssessor(llm)
    assessor.prompt = None

    assessment = await assessor.assess(
        query="What is CUDA?",
        answer="CUDA is a parallel-computing platform [1].",
        source_count=1,
        tool_budget_exhausted=False,
    )

    assert assessment == ShallowEscalationAssessment.sufficient()
    llm.ainvoke.assert_not_awaited()


@pytest.mark.asyncio
async def test_human_outreach_is_not_a_deep_research_strategy() -> None:
    """The deep workflow cannot call people, so outreach cannot justify escalation."""
    llm = MagicMock()
    llm.ainvoke = AsyncMock(
        return_value=AIMessage(
            content=(
                '{"status":"material_gap","unresolved_requirement":"Vendor confirmation is absent",'
                '"material_conflict":null,"deep_research_strategy":'
                '"Contact the vendor support representative for confirmation"}'
            )
        )
    )
    assessor = ShallowEscalationAssessor(llm)

    assessment = await assessor.assess(
        query="Does product X support feature Y?",
        answer="Public documentation does not confirm feature Y [1].",
        source_count=1,
        tool_budget_exhausted=False,
    )

    assert assessment == ShallowEscalationAssessment.sufficient()


@pytest.mark.asyncio
async def test_assessor_bounds_answer_context() -> None:
    """Long answers retain their beginning and conclusion without sending the full body."""
    response = AIMessage(content=ESCALATION_CASES["S1"].expected_assessment_json())
    llm = MagicMock()
    llm.ainvoke = AsyncMock(return_value=response)
    assessor = ShallowEscalationAssessor(llm)
    answer = f"BEGIN-OF-ANSWER\n{'x' * 20_000}\nEND-OF-ANSWER"

    await assessor.assess(
        query="Summarize the result.",
        answer=answer,
        source_count=1,
        tool_budget_exhausted=False,
    )

    messages = llm.ainvoke.await_args.args[0]
    prompt = "\n".join(str(message.content) for message in messages)
    assert "BEGIN-OF-ANSWER" in prompt
    assert "END-OF-ANSWER" in prompt
    assert len(prompt) < len(answer)


def _evaluation_result(
    case: EscalationCase,
    *,
    predicted_status: str | None = None,
    parse_success: bool = True,
    logical_model_calls: int = 1,
    latency_seconds: float = 1.0,
) -> dict[str, object]:
    predicted = predicted_status or case.expected_status
    expected_route = "shallow" if case.expected_status == "sufficient" else "escalate"
    predicted_route = "shallow" if predicted == "sufficient" else "escalate"
    return {
        "expected_status": case.expected_status,
        "predicted_status": predicted,
        "parse_success": parse_success,
        "logical_model_calls": logical_model_calls,
        "exact_status_correct": predicted == case.expected_status,
        "route_correct": predicted_route == expected_route,
        "input_tokens": 100,
        "output_tokens": 20,
        "latency_seconds": latency_seconds,
    }


def test_live_evaluator_aggregation_accepts_perfect_routing() -> None:
    """Pure aggregation enforces the documented 20-call acceptance boundary."""
    results = [
        _evaluation_result(case, latency_seconds=float(index))
        for index, case in enumerate(ESCALATION_CASES.values(), start=1)
    ]

    summary = build_summary(results)

    assert summary["case_count"] == 20
    assert summary["logical_model_call_count"] == 20
    assert summary["cases_with_exactly_one_call"] == 20
    assert summary["parse_success_count"] == 20
    assert summary["shallow_route_correct"] == 10
    assert summary["escalation_route_correct"] == 10
    assert summary["input_tokens"] == 2_000
    assert summary["output_tokens"] == 400
    assert summary["p50_latency_seconds"] == 10.5
    assert summary["p95_latency_seconds"] == 19.05
    assert acceptance_passed(summary)


def test_live_evaluator_aggregation_reports_confusion_and_call_failure() -> None:
    """Aggregation exposes category confusion and rejects extra or missing calls."""
    results = [_evaluation_result(case) for case in ESCALATION_CASES.values()]
    results[10] = {
        **results[10],
        "predicted_status": "sufficient",
        "exact_status_correct": False,
        "route_correct": False,
        "logical_model_calls": 0,
    }

    summary = build_summary(results)

    assert summary["confusion_matrix"]["material_gap"]["sufficient"] == 1
    assert summary["escalation_route_correct"] == 9
    assert summary["cases_with_exactly_one_call"] == 19
    assert not acceptance_passed(summary)
