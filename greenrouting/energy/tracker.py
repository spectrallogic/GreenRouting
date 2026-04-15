"""Energy savings tracker — tracks cumulative impact of routing decisions."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EnergyReport:
    """Summary of energy savings from routing."""

    total_queries: int
    total_energy_wh: float
    total_energy_if_max_wh: float
    total_cost: float
    total_cost_if_max: float

    @property
    def energy_saved_wh(self) -> float:
        return self.total_energy_if_max_wh - self.total_energy_wh

    @property
    def energy_saved_pct(self) -> float:
        if self.total_energy_if_max_wh == 0:
            return 0.0
        return self.energy_saved_wh / self.total_energy_if_max_wh * 100

    @property
    def cost_saved(self) -> float:
        return self.total_cost_if_max - self.total_cost

    @property
    def cost_saved_pct(self) -> float:
        if self.total_cost_if_max == 0:
            return 0.0
        return self.cost_saved / self.total_cost_if_max * 100

    def __str__(self) -> str:
        return (
            f"GreenRouting Impact Report\n"
            f"{'=' * 40}\n"
            f"Queries routed:    {self.total_queries}\n"
            f"Energy used:       {self.total_energy_wh:.4f} Wh\n"
            f"Energy saved:      {self.energy_saved_wh:.4f} Wh ({self.energy_saved_pct:.1f}%)\n"
            f"Cost:              ${self.total_cost:.4f}\n"
            f"Cost saved:        ${self.cost_saved:.4f} ({self.cost_saved_pct:.1f}%)\n"
        )


@dataclass
class EnergyTracker:
    """Accumulates energy/cost data across routing decisions."""

    _energy_used: list[float] = field(default_factory=list)
    _energy_max: list[float] = field(default_factory=list)
    _cost_used: list[float] = field(default_factory=list)
    _cost_max: list[float] = field(default_factory=list)

    def record(
        self,
        energy_wh: float,
        max_energy_wh: float,
        cost: float,
        max_cost: float,
    ) -> None:
        """Record a single routing decision's energy and cost impact."""
        self._energy_used.append(energy_wh)
        self._energy_max.append(max_energy_wh)
        self._cost_used.append(cost)
        self._cost_max.append(max_cost)

    def report(self) -> EnergyReport:
        """Generate an impact report."""
        return EnergyReport(
            total_queries=len(self._energy_used),
            total_energy_wh=sum(self._energy_used),
            total_energy_if_max_wh=sum(self._energy_max),
            total_cost=sum(self._cost_used),
            total_cost_if_max=sum(self._cost_max),
        )

    def reset(self) -> None:
        """Clear all recorded data."""
        self._energy_used.clear()
        self._energy_max.clear()
        self._cost_used.clear()
        self._cost_max.clear()
