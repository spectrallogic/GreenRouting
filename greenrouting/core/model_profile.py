"""ModelProfile — metadata descriptor for any LLM in the routing pool."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ModelProfile:
    """Describes a model's identity, capabilities, and energy/cost characteristics.

    This is the foundation of model-agnosticism: the router never sees hardcoded
    model names — it sees capability vectors, energy estimates, and cost metadata.
    """

    # Identity
    name: str
    provider: str  # e.g. "openai", "anthropic", "local"
    model_id: str | None = None  # Provider-specific identifier (defaults to name)

    # Learned capability representation (populated during profiling/training)
    capability_vector: np.ndarray | None = field(default=None, repr=False)

    # Energy and cost metadata
    estimated_params_b: float | None = None  # Estimated parameter count in billions
    cost_per_1k_input: float = 0.0  # USD per 1K input tokens
    cost_per_1k_output: float = 0.0  # USD per 1K output tokens
    avg_latency_ms: float | None = None  # Average latency in milliseconds
    energy_per_query_wh: float | None = None  # Estimated Wh per average query

    # Provider configuration
    api_base: str | None = None
    api_key_env: str | None = None  # Environment variable name for API key

    # Benchmark scores (populated during profiling)
    benchmark_scores: dict[str, float] = field(default_factory=dict)

    # Tags for coarse pre-filtering
    tags: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        if self.model_id is None:
            self.model_id = self.name

    @property
    def has_capability_vector(self) -> bool:
        return self.capability_vector is not None

    @property
    def has_energy_data(self) -> bool:
        return self.energy_per_query_wh is not None or self.estimated_params_b is not None

    def to_dict(self) -> dict:
        """Serialize to a JSON-safe dictionary (excludes numpy arrays)."""
        return {
            "name": self.name,
            "provider": self.provider,
            "model_id": self.model_id,
            "estimated_params_b": self.estimated_params_b,
            "cost_per_1k_input": self.cost_per_1k_input,
            "cost_per_1k_output": self.cost_per_1k_output,
            "avg_latency_ms": self.avg_latency_ms,
            "energy_per_query_wh": self.energy_per_query_wh,
            "api_base": self.api_base,
            "api_key_env": self.api_key_env,
            "benchmark_scores": self.benchmark_scores,
            "tags": sorted(self.tags),
        }

    @classmethod
    def from_dict(cls, data: dict) -> ModelProfile:
        """Deserialize from a dictionary."""
        data = dict(data)  # shallow copy
        if "tags" in data:
            data["tags"] = set(data["tags"])
        return cls(**data)
