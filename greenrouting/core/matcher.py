"""Benchmark-based model matcher — selects the greenest model that meets query needs.

This is the core of the dynamic routing: the classifier says what a query NEEDS
(as weighted capabilities), and the matcher finds the cheapest/greenest model in
the pool that can deliver across ALL required dimensions.

Adding a new model = providing its benchmark scores. No retraining required.
"""

from __future__ import annotations

import math

from greenrouting.core.decision import RoutingDecision
from greenrouting.core.model_profile import ModelProfile
from greenrouting.core.registry import ModelRegistry
from greenrouting.core.taxonomy import CAPABILITY_BENCHMARKS, Capability, QueryProfile
from greenrouting.energy.green_score import GreenScorer


class BenchmarkMatcher:
    """Matches a QueryProfile to the best model using weighted benchmark scores.

    For a multi-capability query like {code: 0.5, math: 0.35, reasoning: 0.15}:
    1. For each capability, look up which benchmarks measure it
    2. For each model, compute its weighted average across all required benchmarks
    3. Filter out models below the difficulty-based threshold
    4. Among qualifying models, pick the one with the best Green Score
    """

    def __init__(self, registry: ModelRegistry, scorer: GreenScorer | None = None) -> None:
        self.registry = registry
        self.scorer = scorer or GreenScorer()

    def match(self, profile: QueryProfile) -> RoutingDecision:
        """Find the greenest model that can handle this query profile."""
        models = self.registry.list_models()
        if not models:
            raise ValueError("Cannot match: model registry is empty")

        threshold = profile.min_benchmark_threshold()

        # Score each model's fitness for this query (weighted across capabilities)
        fitness: dict[str, float] = {}
        for model in models:
            score = self._compute_weighted_fitness(model, profile.capability_weights)
            fitness[model.name] = score

        # Find models that meet the threshold
        qualified = {name: score for name, score in fitness.items() if score >= threshold}

        # If no model meets the threshold, relax: use all models
        if not qualified:
            qualified = fitness

        # Use Green Score to pick the greenest among qualified models
        max_fitness = max(qualified.values()) if qualified else 1.0
        quality_scores = {
            name: score / max_fitness if max_fitness > 0 else 1.0
            for name, score in qualified.items()
        }

        decision = self.scorer.select(self.registry, quality_scores)

        # Build reasoning string
        selected = self.registry.get(decision.selected_model)
        cap_str = ", ".join(
            f"{cap} ({w:.0%})"
            for cap, w in sorted(
                profile.capability_weights.items(),
                key=lambda x: -x[1],
            )
            if w >= 0.1
        )
        decision.reasoning = (
            f"Query needs [{cap_str}] "
            f"(difficulty {profile.difficulty}/5). "
            f"Selected {selected.name} — fitness {fitness[selected.name]:.2f} "
            f"(threshold {threshold:.2f}), lowest energy footprint."
        )

        return decision

    def _compute_weighted_fitness(
        self,
        model: ModelProfile,
        capability_weights: dict[str, float],
    ) -> float:
        """Compute a model's weighted fitness across multiple capabilities.

        For each capability in the query, looks up relevant benchmarks and
        computes the model's average score. Then combines across capabilities
        using the query's weights.

        Example: query needs {code: 0.5, math: 0.3, reasoning: 0.2}
        Model scores: code=0.9, math=0.7, reasoning=0.85
        Weighted fitness = 0.5*0.9 + 0.3*0.7 + 0.2*0.85 = 0.83
        """
        if not capability_weights:
            return 0.9  # No specific capability needed → any model works

        total_weight = 0.0
        weighted_score = 0.0

        for cap_name, weight in capability_weights.items():
            if weight < 0.01:  # Skip negligible weights
                continue

            try:
                cap = Capability(cap_name)
            except ValueError:
                continue

            cap_score = self._score_capability(model, cap)
            weighted_score += weight * cap_score
            total_weight += weight

        if total_weight == 0:
            return 0.5  # Fallback

        return weighted_score / total_weight

    def _score_capability(self, model: ModelProfile, capability: Capability) -> float:
        """Score a model on a single capability using its benchmark data."""
        benchmarks = CAPABILITY_BENCHMARKS.get(capability, [])

        if not benchmarks:
            # SIMPLE capability — any model works
            return 0.9

        scores = []
        for bench in benchmarks:
            if bench in model.benchmark_scores:
                scores.append(model.benchmark_scores[bench])

        if scores:
            return sum(scores) / len(scores)

        # Fallback: estimate from parameter count
        if model.estimated_params_b is not None:
            # Capability scales roughly with log(params)
            # 1B → ~0.3, 8B → ~0.5, 70B → ~0.7, 200B → ~0.8, 1000B → ~0.9
            return min(0.9, 0.15 * math.log2(max(model.estimated_params_b, 0.5)) + 0.3)

        # No data at all
        return 0.5
