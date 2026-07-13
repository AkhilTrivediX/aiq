# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused integration tests for post-shallow escalation routing."""

from unittest.mock import AsyncMock
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage

from aiq_agent.agents.chat_researcher.agent import ChatResearcherAgent
from aiq_agent.agents.chat_researcher.agent import _should_escalate_shallow
from aiq_agent.agents.chat_researcher.models import ChatResearcherState
from aiq_agent.agents.chat_researcher.models import DepthDecision
from aiq_agent.agents.chat_researcher.models import IntentResult
from aiq_agent.agents.chat_researcher.models import ShallowResult
from aiq_agent.agents.clarifier.models import ClarifierResult
from aiq_agent.agents.shallow_researcher.escalation import ShallowEscalationAssessment
from aiq_agent.agents.shallow_researcher.models import ShallowResearchAgentState
from aiq_agent.common.citation_verification import EmptySourceRegistryError
from aiq_api.auth.errors import AuthError


async def _shallow_intent(_state: ChatResearcherState) -> dict[str, object]:
    return {
        "user_intent": IntentResult(intent="research", raw=None),
        "depth_decision": DepthDecision(decision="shallow", raw_reasoning="bounded lookup"),
    }


def _assessment(status: str) -> ShallowEscalationAssessment:
    if status == "material_gap":
        return ShallowEscalationAssessment(
            status="material_gap",
            unresolved_requirement="The explicitly requested independent benchmark is unsupported.",
            deep_research_strategy="Search independent benchmarks and validate comparable workloads.",
        )
    if status == "material_conflict":
        return ShallowEscalationAssessment(
            status="material_conflict",
            material_conflict="Two primary sources disagree on the conclusion-critical value.",
            deep_research_strategy="Trace document revisions and reconcile the conflicting definitions.",
        )
    return ShallowEscalationAssessment.sufficient()


def _build_agent(
    *,
    shallow_research_fn,
    enable_escalation: bool = True,
    enable_clarifier: bool = True,
    validate_deep_research_tools_fn=None,
) -> tuple[ChatResearcherAgent, AsyncMock, AsyncMock]:
    clarifier = AsyncMock(return_value=ClarifierResult(clarifier_log="No additional context required."))

    async def deep_research(state):
        result = MagicMock()
        result.messages = list(state.messages) + [AIMessage(content="# Deep report")]
        return result

    deep = AsyncMock(side_effect=deep_research)
    agent = ChatResearcherAgent(
        intent_classifier_fn=_shallow_intent,
        shallow_research_fn=shallow_research_fn,
        deep_research_fn=deep,
        clarifier_fn=clarifier,
        enable_escalation=enable_escalation,
        enable_clarifier=enable_clarifier,
        validate_deep_research_tools_fn=validate_deep_research_tools_fn,
    )
    return agent, clarifier, deep


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status,should_escalate",
    [
        ("sufficient", False),
        ("material_gap", True),
        ("material_conflict", True),
    ],
)
async def test_graph_routes_only_material_assessments_through_clarifier(
    status: str,
    should_escalate: bool,
) -> None:
    """Only validated material outcomes continue through clarifier and deep research."""

    async def shallow(state: ShallowResearchAgentState) -> ShallowResearchAgentState:
        return state.model_copy(
            update={
                "messages": list(state.messages) + [AIMessage(content="Shallow answer [1].")],
                "escalation_assessment": _assessment(status),
            }
        )

    agent, clarifier, deep = _build_agent(shallow_research_fn=shallow)
    result = await agent.run(
        ChatResearcherState(messages=[HumanMessage(content="Research this question.")]),
        thread_id=f"route-{status}",
    )

    if should_escalate:
        clarifier.assert_awaited_once()
        deep.assert_awaited_once()
        assert result["messages"][-1].content == "# Deep report"
    else:
        clarifier.assert_not_awaited()
        deep.assert_not_awaited()
        assert result["messages"][-1].content == "Shallow answer [1]."


@pytest.mark.asyncio
async def test_legacy_keyword_does_not_escalate() -> None:
    """Ordinary answer prose no longer acts as an implicit routing protocol."""

    async def shallow(state: ShallowResearchAgentState) -> ShallowResearchAgentState:
        return state.model_copy(
            update={
                "messages": list(state.messages)
                + [AIMessage(content="APT reports 'unable to find package' when no enabled index has that name.")],
                "escalation_assessment": ShallowEscalationAssessment.sufficient(),
            }
        )

    agent, clarifier, deep = _build_agent(shallow_research_fn=shallow)
    result = await agent.run(
        ChatResearcherState(messages=[HumanMessage(content="Explain apt's unable to find package error.")]),
        thread_id="legacy-keyword",
    )

    clarifier.assert_not_awaited()
    deep.assert_not_awaited()
    assert "unable to find package" in result["messages"][-1].content


@pytest.mark.asyncio
async def test_budget_exhaustion_alone_does_not_escalate() -> None:
    """The hard shallow budget remains a stopping condition, not an escalation trigger."""

    async def shallow(state: ShallowResearchAgentState) -> ShallowResearchAgentState:
        return state.model_copy(
            update={
                "messages": list(state.messages) + [AIMessage(content="The requested comparison is complete [1].")],
                "tool_iterations": 5,
                "escalation_assessment": ShallowEscalationAssessment.sufficient(),
            }
        )

    agent, clarifier, deep = _build_agent(shallow_research_fn=shallow)
    await agent.run(
        ChatResearcherState(messages=[HumanMessage(content="Compare lists and tuples.")]),
        thread_id="budget-sufficient",
    )

    clarifier.assert_not_awaited()
    deep.assert_not_awaited()


@pytest.mark.asyncio
async def test_escalation_disabled_does_not_request_assessment() -> None:
    """Disabled escalation preserves the standalone shallow path with no assessor work."""
    observed_assess_flags: list[bool] = []

    async def shallow(state: ShallowResearchAgentState) -> ShallowResearchAgentState:
        observed_assess_flags.append(state.assess_escalation)
        return state.model_copy(update={"messages": list(state.messages) + [AIMessage(content="Shallow answer [1].")]})

    agent, clarifier, deep = _build_agent(
        shallow_research_fn=shallow,
        enable_escalation=False,
    )
    await agent.run(
        ChatResearcherState(messages=[HumanMessage(content="What is CUDA?")]),
        thread_id="escalation-disabled",
    )

    assert observed_assess_flags == [False]
    clarifier.assert_not_awaited()
    deep.assert_not_awaited()


@pytest.mark.asyncio
async def test_unavailable_deep_tools_skip_assessment_and_preserve_shallow_answer() -> None:
    """Do not pay for an assessment when deep research cannot execute."""
    observed_assess_flags: list[bool] = []
    validate_deep_tools = MagicMock(return_value=(False, "Deep research tools are unavailable."))

    async def shallow(state: ShallowResearchAgentState) -> ShallowResearchAgentState:
        observed_assess_flags.append(state.assess_escalation)
        return state.model_copy(update={"messages": list(state.messages) + [AIMessage(content="Shallow answer [1].")]})

    agent, clarifier, deep = _build_agent(
        shallow_research_fn=shallow,
        validate_deep_research_tools_fn=validate_deep_tools,
    )
    result = await agent.run(
        ChatResearcherState(messages=[HumanMessage(content="What is CUDA?")]),
        thread_id="deep-tools-unavailable",
    )

    validate_deep_tools.assert_called_once_with(None)
    assert observed_assess_flags == [False]
    clarifier.assert_not_awaited()
    deep.assert_not_awaited()
    assert result["messages"][-1].content == "Shallow answer [1]."


def test_explicit_legacy_escalation_request_remains_compatible() -> None:
    """A legacy explicit true result remains authoritative when escalation is enabled."""
    explicit_result = ShallowResult(
        answer="The central requirement is unresolved.",
        confidence="low",
        escalate_to_deep=True,
        escalation_reason="A deeper source set is required.",
    )

    assert _should_escalate_shallow(None, enabled=True, shallow_result=explicit_result) is True
    assert _should_escalate_shallow(None, enabled=False, shallow_result=explicit_result) is False


def test_explicit_error_result_does_not_escalate() -> None:
    """Legacy result objects only route when their explicit flag is true."""
    error_result = ShallowResult(
        answer="The shallow provider failed.",
        confidence="high",
        escalate_to_deep=False,
    )

    assert _should_escalate_shallow(None, enabled=True, shallow_result=error_result) is False


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_kind", ["empty_sources", "auth", "exception", "no_messages"])
async def test_shallow_failures_do_not_escalate(failure_kind: str) -> None:
    """Research and infrastructure failures terminate without invoking deep research."""

    async def shallow(state: ShallowResearchAgentState):
        assert state.assess_escalation is True
        if failure_kind == "empty_sources":
            raise EmptySourceRegistryError("shallow research")
        if failure_kind == "auth":
            raise AuthError("Authentication is required for this source.")
        if failure_kind == "exception":
            raise RuntimeError("provider unavailable")
        return state.model_copy(update={"messages": []})

    agent, clarifier, deep = _build_agent(shallow_research_fn=shallow)
    result = await agent.run(
        ChatResearcherState(messages=[HumanMessage(content="Research this.")]),
        thread_id=f"failure-{failure_kind}",
    )

    clarifier.assert_not_awaited()
    deep.assert_not_awaited()
    assert result["messages"][-1].content
    if failure_kind == "no_messages":
        assert result["messages"][-1].content == "An error occurred during shallow research."
