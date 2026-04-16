"""Integration tests — end-to-end routing pipeline."""

import pytest

from greenrouting.core.compression import get_compression_hint
from greenrouting.core.matcher import BenchmarkMatcher
from greenrouting.core.registry import ModelRegistry
from greenrouting.core.taxonomy import Capability, QueryProfile
from greenrouting.energy.green_score import GreenScorer
from greenrouting.energy.profiles import get_known_profiles
from greenrouting.energy.tracker import EnergyTracker


@pytest.fixture
def full_registry() -> ModelRegistry:
    """Registry populated with all known model profiles."""
    reg = ModelRegistry()
    for profile in get_known_profiles().values():
        reg.register(profile)
    return reg


class TestEndToEnd:
    def test_simple_query_routes_efficiently(self, full_registry):
        """A trivial query should route to the cheapest model."""
        matcher = BenchmarkMatcher(full_registry)
        profile = QueryProfile.single(Capability.SIMPLE, difficulty=1)
        decision = matcher.match(profile)

        # Should pick one of the small/cheap models
        assert decision.energy_estimate_wh < 0.02
        assert decision.selected_model in [
            "gpt-4o-mini",
            "claude-haiku",
            "llama-3.1-8b",
            "gemini-2.0-flash",
            "mistral-small",
        ]

    def test_hard_math_routes_to_strong_model(self, full_registry):
        """A hard math problem should route to a strong model."""
        matcher = BenchmarkMatcher(full_registry)
        profile = QueryProfile.single(Capability.MATH, difficulty=5)
        decision = matcher.match(profile)

        # Should pick a model that meets the high threshold
        assert decision.quality_estimate > 0.9
        assert decision.energy_estimate_wh < decision.all_scores["claude-opus"].energy_estimate_wh

    def test_energy_tracker_records_savings(self, full_registry):
        """Tracker should accumulate energy savings across queries."""
        matcher = BenchmarkMatcher(full_registry)
        tracker = EnergyTracker()

        queries = [
            QueryProfile.single(Capability.SIMPLE, difficulty=1),
            QueryProfile.single(Capability.SIMPLE, difficulty=2),
            QueryProfile.single(Capability.CODE, difficulty=3),
            QueryProfile.single(Capability.MATH, difficulty=5),
        ]

        for profile in queries:
            decision = matcher.match(profile)
            max_energy = max(s.energy_estimate_wh for s in decision.all_scores.values())
            max_cost = max(s.cost_estimate for s in decision.all_scores.values())
            tracker.record(
                energy_wh=decision.energy_estimate_wh,
                max_energy_wh=max_energy,
                cost=decision.cost_estimate,
                max_cost=max_cost,
            )

        report = tracker.report()
        assert report.total_queries == 4
        assert report.energy_saved_pct > 0
        assert report.total_energy_wh < report.total_energy_if_max_wh

    def test_compression_hint_integration(self):
        """Compression hints should vary by query difficulty."""
        easy = QueryProfile.single(Capability.SIMPLE, difficulty=1)
        hard = QueryProfile.single(Capability.REASONING, difficulty=5)

        easy_hint = get_compression_hint(easy)
        hard_hint = get_compression_hint(hard)

        assert easy_hint.should_compress
        assert not hard_hint.should_compress
        assert easy_hint.estimated_token_savings_pct > hard_hint.estimated_token_savings_pct

    def test_green_score_presets_change_behavior(self, full_registry):
        """Different Green Score presets should yield different routing decisions."""
        profile = QueryProfile.single(Capability.KNOWLEDGE, difficulty=3)

        quality_scorer = GreenScorer.from_config({"green_score": {"preset": "quality_first"}})
        green_scorer = GreenScorer.from_config({"green_score": {"preset": "maximum_green"}})

        quality_matcher = BenchmarkMatcher(full_registry, quality_scorer)
        green_matcher = BenchmarkMatcher(full_registry, green_scorer)

        quality_decision = quality_matcher.match(profile)
        green_decision = green_matcher.match(profile)

        assert green_decision.energy_estimate_wh <= quality_decision.energy_estimate_wh

    def test_known_profiles_all_have_benchmarks(self):
        """All known profiles should have benchmark data."""
        profiles = get_known_profiles()
        assert len(profiles) >= 10

        for name, profile in profiles.items():
            assert len(profile.benchmark_scores) > 0, f"{name} has no benchmarks"
            assert profile.energy_per_query_wh is not None, f"{name} has no energy data"

    def test_mixed_query_routing(self, full_registry):
        """Mixed-capability queries should route based on weighted needs."""
        matcher = BenchmarkMatcher(full_registry)
        # Code + math — like "write a script to solve a differential equation"
        profile = QueryProfile(
            capability_weights={"code": 0.5, "math": 0.35, "reasoning": 0.15},
            difficulty=4,
        )
        decision = matcher.match(profile)
        assert decision.selected_model is not None
        assert decision.energy_estimate_wh > 0
        assert "code" in decision.reasoning
