"""Tests for the benchmark-based model matcher."""

import pytest

from greenrouting.core.matcher import BenchmarkMatcher
from greenrouting.core.model_profile import ModelProfile
from greenrouting.core.registry import ModelRegistry
from greenrouting.core.taxonomy import Capability, QueryProfile


@pytest.fixture
def benchmark_registry() -> ModelRegistry:
    """Registry with models that have benchmark scores."""
    reg = ModelRegistry()
    reg.register(
        ModelProfile(
            name="big-model",
            provider="test",
            estimated_params_b=200,
            cost_per_1k_input=0.01,
            energy_per_query_wh=0.2,
            benchmark_scores={
                "mmlu": 0.90,
                "gsm8k": 0.95,
                "humaneval": 0.92,
                "arc_challenge": 0.96,
            },
        )
    )
    reg.register(
        ModelProfile(
            name="medium-model",
            provider="test",
            estimated_params_b=70,
            cost_per_1k_input=0.003,
            energy_per_query_wh=0.07,
            benchmark_scores={
                "mmlu": 0.82,
                "gsm8k": 0.85,
                "humaneval": 0.80,
                "arc_challenge": 0.88,
            },
        )
    )
    reg.register(
        ModelProfile(
            name="small-model",
            provider="test",
            estimated_params_b=8,
            cost_per_1k_input=0.0001,
            energy_per_query_wh=0.005,
            benchmark_scores={
                "mmlu": 0.68,
                "gsm8k": 0.70,
                "humaneval": 0.60,
                "arc_challenge": 0.73,
            },
        )
    )
    return reg


class TestBenchmarkMatcher:
    def test_easy_query_routes_to_small_model(self, benchmark_registry):
        """Easy queries should go to the smallest/greenest model."""
        matcher = BenchmarkMatcher(benchmark_registry)
        profile = QueryProfile.single(Capability.KNOWLEDGE, difficulty=1)
        decision = matcher.match(profile)
        # Small model is cheapest and meets the low threshold (0.2)
        assert decision.selected_model == "small-model"
        assert decision.energy_estimate_wh < 0.01

    def test_hard_query_routes_to_capable_model(self, benchmark_registry):
        """Hard queries need a strong model — small won't meet threshold."""
        matcher = BenchmarkMatcher(benchmark_registry)
        profile = QueryProfile.single(Capability.MATH, difficulty=5)
        decision = matcher.match(profile)
        # Only big-model has gsm8k >= 0.9 threshold
        assert decision.selected_model == "big-model"

    def test_medium_query_finds_efficient_middle(self, benchmark_registry):
        """Medium difficulty should pick medium model — efficient enough."""
        matcher = BenchmarkMatcher(benchmark_registry)
        profile = QueryProfile.single(Capability.KNOWLEDGE, difficulty=3)
        decision = matcher.match(profile)
        # Medium model meets threshold and is cheaper than big
        assert decision.selected_model in ["medium-model", "small-model"]
        assert decision.energy_estimate_wh < 0.1

    def test_simple_capability_picks_cheapest(self, benchmark_registry):
        """SIMPLE queries should always go to the cheapest model."""
        matcher = BenchmarkMatcher(benchmark_registry)
        profile = QueryProfile.single(Capability.SIMPLE, difficulty=1)
        decision = matcher.match(profile)
        assert decision.selected_model == "small-model"

    def test_empty_registry_raises(self):
        matcher = BenchmarkMatcher(ModelRegistry())
        profile = QueryProfile.single(Capability.CODE, difficulty=3)
        with pytest.raises(ValueError, match="empty"):
            matcher.match(profile)

    def test_decision_has_reasoning(self, benchmark_registry):
        matcher = BenchmarkMatcher(benchmark_registry)
        profile = QueryProfile.single(Capability.CODE, difficulty=2)
        decision = matcher.match(profile)
        assert decision.reasoning is not None
        assert "code" in decision.reasoning

    def test_fallback_when_no_benchmark_data(self):
        """Models without benchmark data should use parameter-based estimation."""
        reg = ModelRegistry()
        reg.register(ModelProfile(name="unknown-big", provider="x", estimated_params_b=200, energy_per_query_wh=0.2))
        reg.register(ModelProfile(name="unknown-small", provider="x", estimated_params_b=7, energy_per_query_wh=0.005))

        matcher = BenchmarkMatcher(reg)
        profile = QueryProfile.single(Capability.SIMPLE, difficulty=1)
        decision = matcher.match(profile)
        assert decision.selected_model == "unknown-small"

    def test_mixed_capability_matching(self, benchmark_registry):
        """Mixed-capability queries should use weighted benchmark scores."""
        matcher = BenchmarkMatcher(benchmark_registry)
        # Code + math query — needs both humaneval and gsm8k
        profile = QueryProfile(
            capability_weights={"code": 0.5, "math": 0.35, "reasoning": 0.15},
            difficulty=4,
        )
        decision = matcher.match(profile)
        # Should pick a model strong in both code and math
        assert decision.selected_model in ["big-model", "medium-model"]
