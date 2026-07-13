# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded, tool-free assessment of whether a shallow answer needs deep research."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import ValidationError

from aiq_agent.common import extract_json
from aiq_agent.common import load_prompt

from .models import ShallowEscalationAssessment

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"
_MAX_QUERY_JSON_CHARS = 3_000
_MAX_ANSWER_JSON_CHARS = 12_000
_MAX_SERIALIZED_PAYLOAD_CHARS = 16_000
_MAX_OUTPUT_TOKENS = 256
_ASSESSMENT_TIMEOUT_SECONDS = 30
_TRUNCATION_MARKER = "\n\n[... content truncated ...]\n\n"
_ASSESSMENT_RUN_NAME = "shallow_escalation_assessment"
_ASSESSMENT_TAGS = ["aiq", "shallow-escalation-assessment"]
_UNAVAILABLE_OUTREACH_STRATEGY = re.compile(
    r"\b(?:ask|call|contact|email|interview|reach out to)\b.{0,60}"
    r"\b(?:account manager|expert|person|representative|sales|support|vendor)\b",
    re.IGNORECASE,
)


def _json_string_size(value: str) -> int:
    """Return the serialized JSON size for a string, including escaping and quotes."""
    return len(json.dumps(value, ensure_ascii=False))


def _bound_text(value: str, *, max_json_chars: int) -> str:
    """Retain a text's opening and conclusion within an escaped JSON budget."""
    if _json_string_size(value) <= max_json_chars:
        return value

    low = 0
    high = len(value)
    best = _TRUNCATION_MARKER
    while low <= high:
        retained_chars = (low + high) // 2
        head_chars = retained_chars * 2 // 3
        tail_chars = retained_chars - head_chars
        tail = value[-tail_chars:] if tail_chars else ""
        candidate = f"{value[:head_chars]}{_TRUNCATION_MARKER}{tail}"
        if _json_string_size(candidate) <= max_json_chars:
            best = candidate
            low = retained_chars + 1
        else:
            high = retained_chars - 1
    return best


def _serialize_assessment_payload(
    *,
    query: str,
    answer: str,
    source_count: int,
    tool_budget_exhausted: bool,
) -> str:
    """Serialize the complete assessor input within a fixed character budget."""
    payload = json.dumps(
        {
            "user_query": _bound_text(query, max_json_chars=_MAX_QUERY_JSON_CHARS),
            "shallow_answer": _bound_text(answer, max_json_chars=_MAX_ANSWER_JSON_CHARS),
            "retrieved_source_count": source_count,
            "tool_budget_exhausted": tool_budget_exhausted,
        },
        ensure_ascii=False,
    )
    if len(payload) > _MAX_SERIALIZED_PAYLOAD_CHARS:
        # The independent escaped-string budgets leave ample room for keys and
        # scalar fields. Keep this defensive guard so future payload additions
        # cannot silently make the assessment unbounded.
        logger.warning(
            "Shallow escalation assessment fallback status=sufficient reason=payload_budget_exceeded payload_chars=%d",
            len(payload),
        )
        return ""
    return payload


def _is_nvidia_chat_model(llm: BaseChatModel) -> bool:
    """Return whether the model supports ChatNVIDIA's provider-neutral thinking switch."""
    return type(llm).__module__.startswith("langchain_nvidia_ai_endpoints")


def _parse_escalation_assessment(text: str) -> tuple[ShallowEscalationAssessment, str | None]:
    """Return a validated assessment and its fallback reason, if any."""
    parsed = extract_json(text)
    if parsed is None:
        return ShallowEscalationAssessment.sufficient(), "malformed_output"
    try:
        return ShallowEscalationAssessment.model_validate(parsed), None
    except ValidationError:
        return ShallowEscalationAssessment.sufficient(), "schema_validation"


def parse_escalation_assessment(text: str) -> ShallowEscalationAssessment:
    """Parse and validate an assessment, defaulting conservatively on any invalid output."""
    assessment, fallback_reason = _parse_escalation_assessment(text)
    if fallback_reason is not None:
        logger.warning(
            "Shallow escalation assessment fallback status=sufficient reason=%s",
            fallback_reason,
        )
    return assessment


class ShallowEscalationAssessor:
    """Use one bounded LLM call to assess a completed shallow answer."""

    def __init__(self, llm: BaseChatModel, callbacks: list[Any] | None = None) -> None:
        self.llm = llm
        self.callbacks = callbacks or []
        self.prompt = self._load_prompt()

    @staticmethod
    def _load_prompt() -> str | None:
        """Load the assessment rubric."""
        try:
            return load_prompt(_PROMPTS_DIR, "escalation_assessment")
        except Exception:
            logger.warning(
                "Shallow escalation assessment unavailable reason=prompt_load_error",
                exc_info=True,
            )
            return None

    async def assess(
        self,
        *,
        query: str,
        answer: str,
        source_count: int,
        tool_budget_exhausted: bool,
    ) -> ShallowEscalationAssessment:
        """Assess a completed answer without invoking tools or another agent workflow."""
        if self.prompt is None:
            return ShallowEscalationAssessment.sufficient()

        payload = _serialize_assessment_payload(
            query=query,
            answer=answer,
            source_count=source_count,
            tool_budget_exhausted=tool_budget_exhausted,
        )
        if not payload:
            return ShallowEscalationAssessment.sufficient()

        invoke_config: RunnableConfig = {
            "run_name": _ASSESSMENT_RUN_NAME,
            "tags": _ASSESSMENT_TAGS,
            "metadata": {
                "aiq_phase": _ASSESSMENT_RUN_NAME,
                "tool_free": True,
            },
        }
        if self.callbacks:
            invoke_config["callbacks"] = self.callbacks
        invoke_kwargs: dict[str, Any] = {
            "temperature": 0,
            "max_tokens": _MAX_OUTPUT_TOKENS,
            "config": invoke_config,
        }
        if _is_nvidia_chat_model(self.llm):
            invoke_kwargs["thinking_mode"] = False
        try:
            response = await asyncio.wait_for(
                self.llm.ainvoke(
                    [SystemMessage(content=self.prompt), HumanMessage(content=payload)],
                    **invoke_kwargs,
                ),
                timeout=_ASSESSMENT_TIMEOUT_SECONDS,
            )
        except TimeoutError:
            logger.warning(
                "Shallow escalation assessment fallback status=sufficient reason=timeout timeout_seconds=%d",
                _ASSESSMENT_TIMEOUT_SECONDS,
            )
            return ShallowEscalationAssessment.sufficient()
        except Exception as exc:
            logger.warning(
                "Shallow escalation assessment fallback status=sufficient reason=model_error error_type=%s",
                type(exc).__name__,
            )
            return ShallowEscalationAssessment.sufficient()

        if hasattr(response, "text"):
            response_text = str(response.text)
        else:
            content = response.content if hasattr(response, "content") else response
            response_text = content if isinstance(content, str) else str(content or "")
        assessment, fallback_reason = _parse_escalation_assessment(response_text)
        if fallback_reason is not None:
            logger.warning(
                "Shallow escalation assessment fallback status=sufficient reason=%s",
                fallback_reason,
            )
            return assessment
        if assessment.deep_research_strategy and _UNAVAILABLE_OUTREACH_STRATEGY.search(
            assessment.deep_research_strategy
        ):
            logger.warning(
                "Shallow escalation assessment fallback status=sufficient reason=unavailable_outreach_strategy"
            )
            return ShallowEscalationAssessment.sufficient()
        if assessment.status == "material_conflict" and source_count < 2:
            logger.warning(
                "Shallow escalation assessment fallback status=sufficient "
                "reason=insufficient_sources_for_conflict retrieved_source_count=%d",
                source_count,
            )
            return ShallowEscalationAssessment.sufficient()

        logger.info(
            "Shallow escalation assessment status=%s fallback=false retrieved_source_count=%d tool_budget_exhausted=%s",
            assessment.status,
            source_count,
            tool_budget_exhausted,
        )
        return assessment
