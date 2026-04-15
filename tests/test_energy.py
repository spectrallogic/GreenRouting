"""Tests for GreenScorer and energy estimation."""

from greenrouting.core.model_profile import ModelProfile
from greenrouting.core.registry import ModelRegistry
from greenrouting.energy.green_score import PRESETS, GreenScorer


class TestGreenScorer:
    def test_default_weights(self):
        scorer = GreenScorer()
        assert scorer.alpha == 0.6
        assert scorer.beta == 0.25
        assert scorer.gamma == 0.15

    def test_from_config_preset(self):
        scorer = GreenScorer.from_config({"green_score": {"preset": "maximum_green"}})
        alpha, beta, gamma = PRESETS["maximum_green"]
        assert scorer.alpha == alpha
        assert scorer.beta == beta

    def test_from_config_custom(self):
        scorer = GreenScorer.from_config({"green_score": {"alpha": 0.9, "beta": 0.05, "gamma": 0.05}})
        assert scorer.alpha == 0.9

    def test_score_all_basic(self, registry):
        scorer = GreenScorer()
        quality = {"gpt-4o": 0.95, "gpt-4o-mini": 0.7, "claude-sonnet": 0.9}
        scores = scorer.score_all(registry, quality)

        assert len(scores) == 3
        assert scores["gpt-4o"].quality_score == 0.95
        assert scores["gpt-4o-mini"].energy_estimate_wh == 0.008

    def test_higher_quality_increases_green_score(self):
        """When energy/cost are equal, higher quality → higher green score."""
        reg = ModelRegistry()
        reg.register(ModelProfile(name="a", provider="x", energy_per_query_wh=0.1, cost_per_1k_input=0.01))
        reg.register(ModelProfile(name="b", provider="x", energy_per_query_wh=0.1, cost_per_1k_input=0.01))

        scorer = GreenScorer()
        scores = scorer.score_all(reg, {"a": 0.9, "b": 0.5})
        assert scores["a"].green_score > scores["b"].green_score

    def test_lower_energy_increases_green_score(self):
        """When quality is equal, lower energy → higher green score."""
        reg = ModelRegistry()
        reg.register(ModelProfile(name="big", provider="x", energy_per_query_wh=1.0))
        reg.register(ModelProfile(name="small", provider="x", energy_per_query_wh=0.01))

        scorer = GreenScorer()
        scores = scorer.score_all(reg, {"big": 0.8, "small": 0.8})
        assert scores["small"].green_score > scores["big"].green_score

    def test_select_picks_best_green_score(self, registry):
        scorer = GreenScorer()
        # Give gpt-4o-mini decent quality — it should win on green score
        quality = {"gpt-4o": 0.9, "gpt-4o-mini": 0.85, "claude-sonnet": 0.88}
        decision = scorer.select(registry, quality)

        # gpt-4o-mini has much lower energy/cost, quality close enough
        assert decision.selected_model == "gpt-4o-mini"
        assert decision.green_score > 0

    def test_select_respects_min_quality(self, registry):
        scorer = GreenScorer()
        quality = {"gpt-4o": 0.9, "gpt-4o-mini": 0.3, "claude-sonnet": 0.88}
        decision = scorer.select(registry, quality, min_quality=0.5)

        # gpt-4o-mini excluded due to low quality
        assert decision.selected_model != "gpt-4o-mini"

    def test_energy_estimation_tiers(self):
        scorer = GreenScorer()

        # Tier 1: direct measurement
        p1 = ModelProfile(name="a", provider="x", energy_per_query_wh=0.42)
        assert scorer._estimate_energy(p1) == 0.42

        # Tier 2: parameter-based
        p2 = ModelProfile(name="b", provider="x", estimated_params_b=70)
        assert scorer._estimate_energy(p2) == 0.07

        # Tier 3: latency proxy
        p3 = ModelProfile(name="c", provider="x", avg_latency_ms=1000)
        assert scorer._estimate_energy(p3) == 0.05

        # Tier 4: default fallback
        p4 = ModelProfile(name="d", provider="x")
        assert scorer._estimate_energy(p4) == scorer.default_energy_wh

    def test_empty_registry(self):
        scorer = GreenScorer()
        reg = ModelRegistry()
        scores = scorer.score_all(reg, {})
        assert scores == {}

    def test_maximum_green_prefers_efficiency(self, registry):
        """Maximum green preset should favor efficient models more aggressively."""
        quality = {"gpt-4o": 0.95, "gpt-4o-mini": 0.7, "claude-sonnet": 0.9}

        balanced = GreenScorer.from_config({"green_score": {"preset": "balanced"}})
        green = GreenScorer.from_config({"green_score": {"preset": "maximum_green"}})

        balanced_decision = balanced.select(registry, quality)
        green_decision = green.select(registry, quality)

        # maximum_green should pick a more efficient model
        balanced_energy = balanced_decision.energy_estimate_wh
        green_energy = green_decision.energy_estimate_wh
        assert green_energy <= balanced_energy
