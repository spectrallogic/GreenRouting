"""Shared test fixtures for GreenRouting tests."""

import pytest

from greenrouting.core.model_profile import ModelProfile
from greenrouting.core.registry import ModelRegistry


@pytest.fixture
def sample_profiles() -> list[ModelProfile]:
    """A realistic set of model profiles for testing."""
    return [
        ModelProfile(
            name="gpt-4o",
            provider="openai",
            estimated_params_b=200,
            cost_per_1k_input=0.005,
            cost_per_1k_output=0.015,
            avg_latency_ms=800,
            energy_per_query_wh=0.2,
            tags={"general", "code", "reasoning"},
        ),
        ModelProfile(
            name="gpt-4o-mini",
            provider="openai",
            estimated_params_b=8,
            cost_per_1k_input=0.00015,
            cost_per_1k_output=0.0006,
            avg_latency_ms=200,
            energy_per_query_wh=0.008,
            tags={"general"},
        ),
        ModelProfile(
            name="claude-sonnet",
            provider="anthropic",
            estimated_params_b=70,
            cost_per_1k_input=0.003,
            cost_per_1k_output=0.015,
            avg_latency_ms=600,
            energy_per_query_wh=0.07,
            tags={"general", "code", "reasoning"},
        ),
    ]


@pytest.fixture
def registry(sample_profiles: list[ModelProfile]) -> ModelRegistry:
    """A pre-populated registry with sample models."""
    reg = ModelRegistry()
    for p in sample_profiles:
        reg.register(p)
    return reg
