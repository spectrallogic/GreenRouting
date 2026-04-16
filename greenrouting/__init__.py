"""GreenRouting — Smart model routing for sustainable AI."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from greenrouting.core.compression import CompressionHint, get_compression_hint
from greenrouting.core.decision import ModelScore, RoutingDecision
from greenrouting.core.matcher import BenchmarkMatcher
from greenrouting.core.model_profile import ModelProfile
from greenrouting.core.registry import ModelRegistry
from greenrouting.core.router import Router
from greenrouting.core.taxonomy import Capability, QueryProfile
from greenrouting.energy.green_score import GreenScorer
from greenrouting.energy.profiles import get_known_profiles
from greenrouting.energy.tracker import EnergyTracker
from greenrouting.routers.classifier_router import ClassifierRouter
from greenrouting.routers.random_router import RandomRouter

if TYPE_CHECKING:
    from typing import Any

__version__ = "0.2.0"

# Default location for the shipped pre-trained model
_PRETRAINED_DIR = Path(__file__).parent / "pretrained"


def load_pretrained(
    model_dir: str | Path | None = None,
    registry: ModelRegistry | None = None,
    scorer_config: dict[str, Any] | None = None,
    quality: float | None = None,
) -> ClassifierRouter:
    """Load a pre-trained GreenRouting classifier, ready to route queries.

    This is the simplest way to get started — one function call gives you a
    fully functional router with the shipped model weights and 11 pre-built
    model profiles.

    Args:
        model_dir: Path to a saved classifier directory. If None, uses the
            bundled pre-trained model that ships with the package.
        registry: ModelRegistry with your model pool. If None, uses the
            built-in pool of 11 popular models (GPT-4o, Claude, Llama, etc.).
        scorer_config: Optional config dict for the GreenScorer preset.
            Defaults to ``{"green_score": {"preset": "balanced"}}``.
        quality: Float from 0.0 to 1.0 controlling the quality vs energy
            trade-off. 0.0 = maximum energy savings, 0.5 = balanced (default),
            1.0 = maximum quality. Overrides scorer_config if both are given.

    Returns:
        A ClassifierRouter ready to call ``.route(query)`` on any string.

    Example::

        from greenrouting import load_pretrained

        router = load_pretrained()
        decision = router.route("What is 2+2?")
        print(decision.selected_model)        # "llama-3.1-8b"
        print(decision.energy_savings_vs_max) # 0.99

        # Want smarter models? Turn up the quality dial:
        router = load_pretrained(quality=0.8)

    Raises:
        FileNotFoundError: If no pre-trained model is found at the given path.
    """
    path = Path(model_dir) if model_dir else _PRETRAINED_DIR

    if not (path / "classifier_head.pt").exists():
        raise FileNotFoundError(
            f"No pre-trained model found at {path}. "
            "Run `python -m greenrouting.train` or "
            "`python examples/train_and_save.py` to train one first."
        )

    if registry is None:
        registry = ModelRegistry()
        for profile in get_known_profiles().values():
            registry.register(profile)

    router = ClassifierRouter.load(path, registry)

    # quality parameter takes priority over scorer_config
    if quality is not None:
        router.scorer = GreenScorer.from_quality(quality)
        router.matcher = BenchmarkMatcher(registry, router.scorer)
    elif scorer_config:
        router.scorer = GreenScorer.from_config(scorer_config)
        router.matcher = BenchmarkMatcher(registry, router.scorer)

    return router


__all__ = [
    "BenchmarkMatcher",
    "Capability",
    "ClassifierRouter",
    "CompressionHint",
    "EnergyTracker",
    "GreenScorer",
    "ModelProfile",
    "ModelRegistry",
    "ModelScore",
    "QueryProfile",
    "RandomRouter",
    "RoutingDecision",
    "Router",
    "get_compression_hint",
    "get_known_profiles",
    "load_pretrained",
]


# Lazy import for serving module (requires optional litellm dependency)
def __getattr__(name: str) -> Any:
    if name == "GreenRoutingClient":
        from greenrouting.serving.client import GreenRoutingClient

        return GreenRoutingClient
    if name == "ModelConfig":
        from greenrouting.serving.client import ModelConfig

        return ModelConfig
    raise AttributeError(f"module 'greenrouting' has no attribute {name!r}")
