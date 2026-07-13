# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure fixture and aggregation helpers for shallow-escalation evaluation."""

from __future__ import annotations

import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from typing import Literal

Status = Literal["sufficient", "material_gap", "material_conflict"]
STATUSES: tuple[Status, ...] = ("sufficient", "material_gap", "material_conflict")


@dataclass(frozen=True)
class EscalationCase:
    """One completed shallow answer and its expected assessment outcome."""

    id: str
    query: str
    answer: str
    expected_status: Status
    expected_detail: str | None
    expected_strategy: str | None
    source_count: int
    tool_budget_exhausted: bool

    def expected_assessment_json(self) -> str:
        """Build valid structured output for deterministic parser and router tests."""
        return json.dumps(
            {
                "status": self.expected_status,
                "unresolved_requirement": self.expected_detail if self.expected_status == "material_gap" else None,
                "material_conflict": self.expected_detail if self.expected_status == "material_conflict" else None,
                "deep_research_strategy": self.expected_strategy,
            }
        )


def load_cases(path: Path) -> list[EscalationCase]:
    """Load and minimally validate a committed assessment quality set."""
    raw_cases = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_cases, list):
        raise ValueError("Escalation fixture must contain a JSON list")
    cases = [EscalationCase(**raw_case) for raw_case in raw_cases]
    ids = [case.id for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Escalation fixture case IDs must be unique")
    if any(case.expected_status not in STATUSES for case in cases):
        raise ValueError("Escalation fixture contains an unsupported expected status")
    return cases


def percentile(values: list[float], percentile_value: float) -> float:
    """Return a linearly interpolated percentile without an optional dependency."""
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile_value
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def build_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate routing quality, schema validity, usage, latency, and confusion."""
    confusion = {expected: {predicted: 0 for predicted in STATUSES} for expected in STATUSES}
    for result in results:
        confusion[result["expected_status"]][result["predicted_status"]] += 1

    shallow = [result for result in results if result["expected_status"] == "sufficient"]
    escalation = [result for result in results if result["expected_status"] != "sufficient"]
    latencies = [float(result["latency_seconds"]) for result in results]
    usage_complete = [
        result
        for result in results
        if result.get("input_tokens") is not None and result.get("output_tokens") is not None
    ]

    return {
        "case_count": len(results),
        "logical_model_call_count": sum(int(result["logical_model_calls"]) for result in results),
        "cases_with_exactly_one_call": sum(result["logical_model_calls"] == 1 for result in results),
        "parse_success_count": sum(bool(result["parse_success"]) for result in results),
        "exact_status_correct": sum(bool(result["exact_status_correct"]) for result in results),
        "route_correct": sum(bool(result["route_correct"]) for result in results),
        "shallow_route_correct": sum(bool(result["route_correct"]) for result in shallow),
        "shallow_case_count": len(shallow),
        "escalation_route_correct": sum(bool(result["route_correct"]) for result in escalation),
        "escalation_case_count": len(escalation),
        "input_tokens": sum(int(result["input_tokens"]) for result in usage_complete),
        "output_tokens": sum(int(result["output_tokens"]) for result in usage_complete),
        "cases_with_token_usage": len(usage_complete),
        "p50_latency_seconds": round(statistics.median(latencies), 4) if latencies else 0.0,
        "p95_latency_seconds": round(percentile(latencies, 0.95), 4),
        "confusion_matrix": confusion,
    }


def acceptance_passed(summary: dict[str, Any], *, min_escalation_correct: int = 8) -> bool:
    """Apply the documented conservative acceptance thresholds."""
    return bool(
        summary["case_count"] == 20
        and summary["shallow_case_count"] == 10
        and summary["escalation_case_count"] == 10
        and summary["shallow_route_correct"] == 10
        and summary["escalation_route_correct"] >= min_escalation_correct
        and summary["parse_success_count"] == 20
        and summary["cases_with_exactly_one_call"] == 20
        and summary["logical_model_call_count"] == 20
    )
