"""GreenScorer — composite scoring that balances quality, energy, and cost."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from greenrouting.core.decision import ModelScore, RoutingDecision
from greenrouting.core.model_profile import ModelProfile
from greenrouting.core.registry import ModelRegistry


# Preset weight configurations
PRESETS: dict[str, tuple[float, float, float]] = {
    "quality_first": (1.0, 0.1, 0.1),
    "balanced": (0.6, 0.25, 0.15),
    "maximum_green": (0.3, 0.5, 0.2),
}


@dataclass
class GreenScorer:
    """Computes the Green Score for each model given predicted quality scores.

    green_score = alpha * quality - beta * norm_energy - gamma * norm_cost

    The scorer normalizes energy and cost across the current model pool so that
    the Green Score is comparable regardless of absolute values.
    """

    alpha: float = 0.6  # quality weight
    beta: float = 0.25  # energy penalty weight
    gamma: float = 0.15  # cost penalty weight

    # Default energy estimate when a model has no energy data (Wh)
    default_energy_wh: float = 0.05
    # Default tokens assumed per query for cost estimation
    default_query_tokens: int = 500

    @classmethod
    def from_quality(cls, quality: float) -> GreenScorer:
        """Create a scorer from a single quality dial (0.0 to 1.0).

        This is the simplest way to control the quality vs energy trade-off:
            0.0 = maximum energy savings (picks smallest capable model)
            0.5 = balanced (default)
            1.0 = maximum quality (picks the smartest model available)

        The dial smoothly interpolates the internal weights:
            alpha (quality weight):  0.3 at q=0  ->  1.0 at q=1
            beta  (energy penalty):  0.5 at q=0  ->  0.1 at q=1
            gamma (cost penalty):    0.2 at q=0  ->  0.1 at q=1
        """
        q = max(0.0, min(1.0, quality))
        alpha = 0.3 + 0.7 * q
        beta = 0.5 - 0.4 * q
        gamma = 0.2 - 0.1 * q
        return cls(alpha=alpha, beta=beta, gamma=gamma)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> GreenScorer:
        green_cfg = config.get("green_score", {})
        # Single quality dial takes priority
        quality = green_cfg.get("quality")
        if quality is not None:
            return cls.from_quality(float(quality))
        preset = green_cfg.get("preset")
        if preset and preset in PRESETS:
            alpha, beta, gamma = PRESETS[preset]
            return cls(alpha=alpha, beta=beta, gamma=gamma)
        return cls(
            alpha=green_cfg.get("alpha", 0.6),
            beta=green_cfg.get("beta", 0.25),
            gamma=green_cfg.get("gamma", 0.15),
            default_energy_wh=green_cfg.get("default_energy_wh", 0.05),
            default_query_tokens=green_cfg.get("default_query_tokens", 500),
        )

    def _estimate_energy(self, profile: ModelProfile) -> float:
        """Estimate energy (Wh) for a single query on this model."""
        if profile.energy_per_query_wh is not None:
            return profile.energy_per_query_wh
        if profile.estimated_params_b is not None:
            # Rough heuristic: larger models use more energy
            # ~0.001 Wh per billion parameters per query (calibratable)
            return profile.estimated_params_b * 0.001
        if profile.avg_latency_ms is not None:
            # Latency proxy: ~0.00005 Wh per ms (rough GPU estimate)
            return profile.avg_latency_ms * 0.00005
        return self.default_energy_wh

    def _estimate_cost(self, profile: ModelProfile) -> float:
        """Estimate cost (USD) for a single query on this model."""
        tokens = self.default_query_tokens
        input_cost = (tokens / 1000) * profile.cost_per_1k_input
        output_cost = (tokens / 1000) * profile.cost_per_1k_output
        return input_cost + output_cost

    def score_all(
        self,
        registry: ModelRegistry,
        quality_scores: dict[str, float],
    ) -> dict[str, ModelScore]:
        """Compute Green Scores for all models in the registry.

        Args:
            registry: The model pool.
            quality_scores: Predicted quality per model (from the router).

        Returns:
            Dict mapping model name → ModelScore with all components.
        """
        models = registry.list_models()
        if not models:
            return {}

        # Compute raw energy and cost for each model
        raw: dict[str, tuple[float, float]] = {}
        for m in models:
            raw[m.name] = (self._estimate_energy(m), self._estimate_cost(m))

        # Normalize energy and cost to [0, 1] across the pool
        energies = [e for e, _ in raw.values()]
        costs = [c for _, c in raw.values()]
        max_energy = max(energies) if energies else 1.0
        max_cost = max(costs) if costs else 1.0
        # Avoid division by zero
        max_energy = max_energy if max_energy > 0 else 1.0
        max_cost = max_cost if max_cost > 0 else 1.0

        scores: dict[str, ModelScore] = {}
        for m in models:
            energy, cost = raw[m.name]
            norm_energy = energy / max_energy
            norm_cost = cost / max_cost
            quality = quality_scores.get(m.name, 0.0)

            green_score = (
                self.alpha * quality - self.beta * norm_energy - self.gamma * norm_cost
            )

            scores[m.name] = ModelScore(
                model_name=m.name,
                quality_score=quality,
                energy_estimate_wh=energy,
                cost_estimate=cost,
                green_score=green_score,
            )

        return scores

    def select(
        self,
        registry: ModelRegistry,
        quality_scores: dict[str, float],
        min_quality: float = 0.0,
    ) -> RoutingDecision:
        """Score all models and select the one with the highest Green Score.

        Args:
            registry: The model pool.
            quality_scores: Predicted quality per model.
            min_quality: Minimum quality threshold — models below this are excluded.

        Returns:
            RoutingDecision with the best model selected.
        """
        all_scores = self.score_all(registry, quality_scores)

        # Filter by minimum quality
        candidates = {
            name: score
            for name, score in all_scores.items()
            if score.quality_score >= min_quality
        }

        if not candidates:
            # Fallback: pick the highest quality model regardless
            candidates = all_scores

        best_name = max(candidates, key=lambda n: candidates[n].green_score)
        best = candidates[best_name]

        return RoutingDecision(
            selected_model=best_name,
            green_score=best.green_score,
            quality_estimate=best.quality_score,
            energy_estimate_wh=best.energy_estimate_wh,
            cost_estimate=best.cost_estimate,
            all_scores=all_scores,
        )
