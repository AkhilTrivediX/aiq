# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured assessment for conservative shallow-to-deep escalation."""

from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import field_validator
from pydantic import model_validator


class ShallowEscalationAssessment(BaseModel):
    """A narrow recommendation for whether shallow research needs deeper investigation."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["sufficient", "material_gap", "material_conflict"]
    unresolved_requirement: str | None = None
    material_conflict: str | None = None
    deep_research_strategy: str | None = None

    @field_validator("unresolved_requirement", "material_conflict", "deep_research_strategy", mode="before")
    @classmethod
    def _normalize_optional_text(cls, value: object) -> object:
        """Normalize blank model-generated detail fields to ``None``."""
        if isinstance(value, str):
            return value.strip() or None
        return value

    @model_validator(mode="after")
    def _validate_status_fields(self) -> "ShallowEscalationAssessment":
        """Require exactly the evidence appropriate for the selected status."""
        if self.status == "sufficient":
            if self.unresolved_requirement or self.material_conflict or self.deep_research_strategy:
                raise ValueError("sufficient assessments cannot include escalation details")
            return self

        if not self.deep_research_strategy:
            raise ValueError("escalation assessments require a concrete deep research strategy")

        if self.status == "material_gap":
            if not self.unresolved_requirement or self.material_conflict:
                raise ValueError("material_gap requires only an unresolved requirement and strategy")
            return self

        if not self.material_conflict or self.unresolved_requirement:
            raise ValueError("material_conflict requires only a material conflict and strategy")
        return self

    @classmethod
    def sufficient(cls) -> "ShallowEscalationAssessment":
        """Return the fail-closed assessment used when evaluation is unavailable."""
        return cls(status="sufficient")

    @property
    def recommends_escalation(self) -> bool:
        """Return whether the validated assessment recommends deep research."""
        return self.status in {"material_gap", "material_conflict"}
