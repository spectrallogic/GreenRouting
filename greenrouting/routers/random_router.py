"""RandomRouter — uniform random baseline for benchmarking."""

from __future__ import annotations

import random
from typing import Any

from greenrouting.core.decision import RoutingDecision
from greenrouting.core.registry import ModelRegistry
from greenrouting.core.router import Router
from greenrouting.energy.green_score import GreenScorer


class RandomRouter(Router):
    """Selects a model uniformly at random from the pool.

    Used as a lower-bound baseline: any real router should beat this.
    """

    def __init__(
        self,
        registry: ModelRegistry,
        config: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(registry, config)
        self._rng = random.Random(seed)
        self._scorer = GreenScorer.from_config(config or {})

    def score_models(self, query: str) -> dict[str, float]:
        """Return uniform quality scores (all models equal)."""
        models = self.registry.list_models()
        if not models:
            return {}
        score = 1.0 / len(models)
        return {m.name: score for m in models}

    def route(self, query: str, context: dict[str, Any] | None = None) -> RoutingDecision:
        """Pick a random model and compute its Green Score."""
        models = self.registry.list_models()
        if not models:
            raise ValueError("Cannot route: model registry is empty")

        quality_scores = self.score_models(query)
        decision = self._scorer.select(self.registry, quality_scores)
        # Override with random selection (baseline ignores green score ranking)
        selected = self._rng.choice(models)
        selected_score = decision.all_scores[selected.name]

        return RoutingDecision(
            selected_model=selected.name,
            green_score=selected_score.green_score,
            quality_estimate=selected_score.quality_score,
            energy_estimate_wh=selected_score.energy_estimate_wh,
            cost_estimate=selected_score.cost_estimate,
            all_scores=decision.all_scores,
            reasoning="Random baseline selection",
        )
