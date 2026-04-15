"""Abstract Router — base class for all routing strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from greenrouting.core.decision import RoutingDecision
from greenrouting.core.registry import ModelRegistry


class Router(ABC):
    """Base class for all GreenRouting routers.

    A router takes a query string and returns a RoutingDecision indicating
    which model from the registry should handle the query. The router predicts
    quality scores; the Green Score layer combines those with energy/cost data.
    """

    def __init__(self, registry: ModelRegistry, config: dict[str, Any] | None = None) -> None:
        self.registry = registry
        self.config = config or {}

    @abstractmethod
    def route(self, query: str, context: dict[str, Any] | None = None) -> RoutingDecision:
        """Select the best model for this query.

        Args:
            query: The user's input query.
            context: Optional metadata (e.g. conversation history length, domain hint).

        Returns:
            A RoutingDecision with the selected model and scores.
        """
        ...

    @abstractmethod
    def score_models(self, query: str) -> dict[str, float]:
        """Return predicted quality scores for each model in the pool.

        Args:
            query: The user's input query.

        Returns:
            Dict mapping model name → predicted quality score (0–1).
        """
        ...

    def save(self, path: str | Path) -> None:
        """Persist router state to disk. Override in subclasses with learned weights."""
        raise NotImplementedError(f"{type(self).__name__} does not support save()")

    @classmethod
    def load(cls, path: str | Path, registry: ModelRegistry) -> Router:
        """Load a router from disk. Override in subclasses with learned weights."""
        raise NotImplementedError(f"{cls.__name__} does not support load()")
