"""Tests for Router base class and RandomRouter."""

import pytest

from greenrouting.core.registry import ModelRegistry
from greenrouting.routers.random_router import RandomRouter


class TestRandomRouter:
    def test_route_returns_valid_decision(self, registry):
        router = RandomRouter(registry, seed=42)
        decision = router.route("What is 2+2?")

        assert decision.selected_model in registry.model_names
        assert decision.green_score is not None
        assert decision.quality_estimate >= 0
        assert decision.energy_estimate_wh >= 0
        assert decision.cost_estimate >= 0
        assert len(decision.all_scores) == 3

    def test_route_deterministic_with_seed(self, registry):
        r1 = RandomRouter(registry, seed=123)
        r2 = RandomRouter(registry, seed=123)

        d1 = r1.route("test query")
        d2 = r2.route("test query")
        assert d1.selected_model == d2.selected_model

    def test_route_empty_registry_raises(self):
        reg = ModelRegistry()
        router = RandomRouter(reg, seed=42)
        with pytest.raises(ValueError, match="empty"):
            router.route("test")

    def test_score_models_uniform(self, registry):
        router = RandomRouter(registry, seed=42)
        scores = router.score_models("test")

        assert len(scores) == 3
        expected = 1.0 / 3
        for score in scores.values():
            assert abs(score - expected) < 1e-9

    def test_energy_savings_property(self, registry):
        router = RandomRouter(registry, seed=42)
        decision = router.route("test query")

        savings = decision.energy_savings_vs_max
        assert 0.0 <= savings <= 1.0

    def test_all_scores_present(self, registry):
        router = RandomRouter(registry, seed=42)
        decision = router.route("test")

        for name in registry.model_names:
            assert name in decision.all_scores
            score = decision.all_scores[name]
            assert score.model_name == name
