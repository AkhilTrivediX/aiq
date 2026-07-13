# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Smoke tests for runtime assets distributed with the AI-Q package."""

import tomllib
from importlib.resources import files
from pathlib import Path


def test_shallow_researcher_prompts_have_an_explicit_package_contract() -> None:
    """Keep both shallow prompt assets available to installed-package code."""
    repository_root = Path(__file__).parents[2]
    with (repository_root / "pyproject.toml").open("rb") as config_file:
        project_config = tomllib.load(config_file)

    package_data = project_config["tool"]["setuptools"]["package-data"]
    assert "prompts/*.j2" in package_data["aiq_agent.agents.shallow_researcher"]

    prompts = files("aiq_agent.agents.shallow_researcher").joinpath("prompts")
    assert prompts.joinpath("researcher.j2").read_text().strip()
    assessment_prompt = prompts.joinpath("escalation_assessment.j2").read_text()
    assert '"material_gap"' in assessment_prompt
    assert '"material_conflict"' in assessment_prompt
