"""ClassifierRouter — the core intelligent router.

A lightweight neural classifier that predicts what a query NEEDS (capability +
difficulty), then matches against model benchmark scores to find the greenest
model. This is the heart of GreenRouting.

Architecture:
    Query → Sentence Embedding → MLP → (capability, difficulty, output_length)
    Then: QueryProfile → BenchmarkMatcher → RoutingDecision
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

from greenrouting.core.decision import RoutingDecision
from greenrouting.core.matcher import BenchmarkMatcher
from greenrouting.core.registry import ModelRegistry
from greenrouting.core.router import Router
from greenrouting.core.taxonomy import Capability, QueryProfile
from greenrouting.energy.green_score import GreenScorer


# All capability labels in a fixed order for classification
CAPABILITY_LABELS: list[str] = sorted([c.value for c in Capability])
CAPABILITY_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(CAPABILITY_LABELS)}
IDX_TO_CAPABILITY: dict[int, str] = {i: c for c, i in CAPABILITY_TO_IDX.items()}

OUTPUT_LENGTH_LABELS = ["short", "medium", "long"]
OUTPUT_LENGTH_TO_IDX = {l: i for i, l in enumerate(OUTPUT_LENGTH_LABELS)}


class QueryClassifierHead(nn.Module):
    """MLP classification head that predicts capability, difficulty, and output length.

    Takes a sentence embedding as input and produces three predictions:
    - capability: which capability category (8-class classification)
    - difficulty: difficulty level (regression, 1-5)
    - output_length: expected output length (3-class classification)
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.capability_head = nn.Linear(hidden_dim, len(CAPABILITY_LABELS))
        self.difficulty_head = nn.Linear(hidden_dim, 1)  # regression
        self.output_length_head = nn.Linear(hidden_dim, len(OUTPUT_LENGTH_LABELS))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shared = self.shared(x)
        capability_logits = self.capability_head(shared)
        difficulty = self.difficulty_head(shared).squeeze(-1)
        output_length_logits = self.output_length_head(shared)
        return capability_logits, difficulty, output_length_logits


class ClassifierRouter(Router):
    """Routes queries by classifying what they need, then matching to the greenest model.

    This router is model-agnostic: it predicts query requirements (capability +
    difficulty), not model names. New models can be added to the pool at any time
    by providing their benchmark scores — no retraining needed.
    """

    DEFAULT_ENCODER = "all-MiniLM-L6-v2"  # 22M params, fast

    def __init__(
        self,
        registry: ModelRegistry,
        config: dict[str, Any] | None = None,
        encoder: SentenceTransformer | None = None,
        head: QueryClassifierHead | None = None,
        device: str | None = None,
    ) -> None:
        super().__init__(registry, config)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Sentence encoder (frozen — we only train the head)
        encoder_name = (config or {}).get("encoder_model", self.DEFAULT_ENCODER)
        self.encoder = encoder or SentenceTransformer(encoder_name, device=self.device)

        # Get embedding dimension from encoder
        embedding_dim = self.encoder.get_embedding_dimension()

        # Classification head (lightweight, trainable)
        hidden_dim = (config or {}).get("hidden_dim", 256)
        self.head = head or QueryClassifierHead(embedding_dim, hidden_dim)
        self.head = self.head.to(self.device)

        # Scorer and matcher
        self.scorer = GreenScorer.from_config(config or {})
        self.matcher = BenchmarkMatcher(registry, self.scorer)

    def classify_query(self, query: str) -> QueryProfile:
        """Classify a query into a QueryProfile (what it needs)."""
        self.head.eval()
        with torch.no_grad():
            embedding = self.encoder.encode(
                query, convert_to_tensor=True, device=self.device
            )
            if embedding.dim() == 1:
                embedding = embedding.unsqueeze(0)

            cap_logits, diff_pred, len_logits = self.head(embedding)

            # Capability: softmax probabilities as weights (multi-label)
            cap_probs = torch.softmax(cap_logits, dim=-1)[0]
            cap_idx = cap_probs.argmax().item()
            confidence = cap_probs[cap_idx].item()

            # Build weighted capability dict — keep weights >= 0.1
            capability_weights: dict[str, float] = {}
            for i, prob in enumerate(cap_probs.tolist()):
                if prob >= 0.1:
                    capability_weights[IDX_TO_CAPABILITY[i]] = prob
            # Normalize weights to sum to 1.0
            total = sum(capability_weights.values())
            if total > 0:
                capability_weights = {k: v / total for k, v in capability_weights.items()}
            else:
                # Fallback: use argmax
                capability_weights = {IDX_TO_CAPABILITY[cap_idx]: 1.0}

            # Difficulty: clamp regression output to 1-5
            difficulty = int(round(max(1.0, min(5.0, diff_pred.item()))))

            # Output length: argmax
            len_idx = len_logits.argmax(dim=-1).item()

        return QueryProfile(
            capability_weights=capability_weights,
            difficulty=difficulty,
            expected_output_length=OUTPUT_LENGTH_LABELS[len_idx],
            confidence=confidence,
        )

    def score_models(self, query: str) -> dict[str, float]:
        """Score models based on query classification + benchmark matching."""
        profile = self.classify_query(query)
        decision = self.matcher.match(profile)
        return {name: score.quality_score for name, score in decision.all_scores.items()}

    def route(self, query: str, context: dict[str, Any] | None = None) -> RoutingDecision:
        """Classify the query, then find the greenest model that can handle it."""
        profile = self.classify_query(query)
        decision = self.matcher.match(profile)

        # Enrich reasoning with classification details
        decision.reasoning = (
            f"Classified as {profile.primary_capability.value} "
            f"(difficulty {profile.difficulty}/5, confidence {profile.confidence:.2f}). "
            f"{decision.reasoning}"
        )

        return decision

    def save(self, path: str | Path) -> None:
        """Save the classifier head weights and config."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.head.state_dict(), path / "classifier_head.pt")
        config = {
            "encoder_model": self.config.get("encoder_model", self.DEFAULT_ENCODER),
            "hidden_dim": self.config.get("hidden_dim", 256),
            "embedding_dim": self.encoder.get_embedding_dimension(),
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

    @classmethod
    def load(cls, path: str | Path, registry: ModelRegistry) -> ClassifierRouter:
        """Load a trained classifier from disk."""
        path = Path(path)
        with open(path / "config.json") as f:
            config = json.load(f)

        router = cls(registry, config=config)

        state_dict = torch.load(
            path / "classifier_head.pt",
            map_location=router.device,
            weights_only=True,
        )
        router.head.load_state_dict(state_dict)
        return router
