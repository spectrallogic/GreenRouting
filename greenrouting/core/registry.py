"""ModelRegistry — manages the pool of available models."""

from __future__ import annotations

from greenrouting.core.model_profile import ModelProfile


class ModelRegistry:
    """Thread-safe registry for managing models in the routing pool.

    Models can be added, removed, and queried at runtime. The registry
    is the foundation of model-agnosticism — the router only sees what's
    registered, never hardcoded model names.
    """

    def __init__(self) -> None:
        self._models: dict[str, ModelProfile] = {}

    def register(self, profile: ModelProfile) -> None:
        """Add a model to the pool."""
        self._models[profile.name] = profile

    def unregister(self, name: str) -> ModelProfile:
        """Remove a model from the pool. Raises KeyError if not found."""
        return self._models.pop(name)

    def get(self, name: str) -> ModelProfile:
        """Get a model profile by name. Raises KeyError if not found."""
        return self._models[name]

    def list_models(self, tags: set[str] | None = None) -> list[ModelProfile]:
        """List all models, optionally filtered by tags (intersection)."""
        models = list(self._models.values())
        if tags:
            models = [m for m in models if tags <= m.tags]
        return models

    @property
    def model_names(self) -> list[str]:
        return list(self._models.keys())

    def __len__(self) -> int:
        return len(self._models)

    def __contains__(self, name: str) -> bool:
        return name in self._models

    @classmethod
    def from_config(cls, config: dict) -> ModelRegistry:
        """Build a registry from a config dictionary.

        Expected format:
            models:
              - name: gpt-4o
                provider: openai
                cost_per_1k_input: 0.005
                ...
        """
        registry = cls()
        for model_cfg in config.get("models", []):
            registry.register(ModelProfile.from_dict(model_cfg))
        return registry

    def to_config(self) -> dict:
        """Export registry to a serializable dictionary."""
        return {"models": [m.to_dict() for m in self._models.values()]}
