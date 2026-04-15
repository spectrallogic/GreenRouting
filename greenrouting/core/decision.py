"""RoutingDecision — the result of a routing operation."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelScore:
    """Scores for a single candidate model."""

    model_name: str
    quality_score: float  # Predicted quality (0–1)
    energy_estimate_wh: float  # Estimated energy in Watt-hours
    cost_estimate: float  # Estimated cost in USD
    green_score: float  # Composite Green Score


@dataclass
class RoutingDecision:
    """The output of a routing operation — which model to use and why."""

    selected_model: str
    green_score: float
    quality_estimate: float
    energy_estimate_wh: float
    cost_estimate: float
    all_scores: dict[str, ModelScore] = field(default_factory=dict)
    reasoning: str | None = None

    @property
    def energy_savings_vs_max(self) -> float:
        """Energy saved compared to the most expensive model in the pool."""
        if not self.all_scores:
            return 0.0
        max_energy = max(s.energy_estimate_wh for s in self.all_scores.values())
        if max_energy == 0:
            return 0.0
        return (max_energy - self.energy_estimate_wh) / max_energy
