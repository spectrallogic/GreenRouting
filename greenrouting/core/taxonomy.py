"""Capability taxonomy — defines what capabilities a query might need.

The classifier predicts a QueryProfile (what the query NEEDS), not which model
to use. This decouples routing from specific models — when a new model appears,
you just provide its benchmark scores and the router instantly knows how to use it.

Queries can require MULTIPLE capabilities with different weights. For example:
"Write a Python script that solves this differential equation" needs:
  code: 0.5, math: 0.35, reasoning: 0.15
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Capability(str, Enum):
    """Core capability dimensions, aligned with standard LLM benchmarks."""

    REASONING = "reasoning"  # Logic, analysis, multi-step thinking (ARC, BBH)
    MATH = "math"  # Mathematical problem solving (GSM8K, MATH)
    CODE = "code"  # Code generation and understanding (HumanEval, MBPP)
    KNOWLEDGE = "knowledge"  # Factual knowledge retrieval (MMLU, TriviaQA)
    CREATIVE = "creative"  # Creative writing, brainstorming (MT-Bench creative)
    INSTRUCTION = "instruction"  # Following complex instructions (IFEval)
    MULTILINGUAL = "multilingual"  # Non-English language tasks
    SIMPLE = "simple"  # Trivial tasks any model can handle


# Maps each capability to well-known benchmarks that measure it
CAPABILITY_BENCHMARKS: dict[Capability, list[str]] = {
    Capability.REASONING: ["arc_challenge", "bbh", "gpqa"],
    Capability.MATH: ["gsm8k", "math", "minerva_math"],
    Capability.CODE: ["humaneval", "mbpp", "livecodebench"],
    Capability.KNOWLEDGE: ["mmlu", "triviaqa", "naturalqa"],
    Capability.CREATIVE: ["mt_bench_creative", "alpaca_eval"],
    Capability.INSTRUCTION: ["ifeval", "mt_bench"],
    Capability.MULTILINGUAL: ["mgsm", "flores"],
    Capability.SIMPLE: [],  # Any model handles these
}

# Ordered list for classifier output indices
ALL_CAPABILITIES: list[Capability] = sorted(Capability, key=lambda c: c.value)
CAPABILITY_TO_IDX: dict[str, int] = {c.value: i for i, c in enumerate(ALL_CAPABILITIES)}
NUM_CAPABILITIES: int = len(ALL_CAPABILITIES)


@dataclass
class QueryProfile:
    """What a query needs — predicted by the classifier.

    Unlike single-label classification, this supports WEIGHTED multi-capability
    requirements. A query like "Write Python code to solve a PDE" produces:
        capability_weights = {"code": 0.5, "math": 0.35, "reasoning": 0.15}

    The matcher uses these weights to find the model with the best WEIGHTED
    average across the required benchmarks.
    """

    # Weighted capability requirements (capability → weight, sums to ~1.0)
    # e.g. {"code": 0.6, "math": 0.3, "reasoning": 0.1}
    capability_weights: dict[str, float]

    # Difficulty level: 1 (trivial) to 5 (expert-level)
    difficulty: int

    # Estimated output length: "short" (<50 tokens), "medium" (50-200), "long" (200+)
    expected_output_length: str = "medium"

    # Confidence of the classifier's prediction (0-1)
    confidence: float = 1.0

    def __post_init__(self) -> None:
        self.difficulty = max(1, min(5, self.difficulty))

    @property
    def primary_capability(self) -> Capability:
        """The dominant capability needed (highest weight)."""
        if not self.capability_weights:
            return Capability.SIMPLE
        top = max(self.capability_weights, key=lambda k: self.capability_weights[k])
        return Capability(top)

    @property
    def is_mixed(self) -> bool:
        """Whether this query requires multiple capabilities."""
        significant = [w for w in self.capability_weights.values() if w >= 0.15]
        return len(significant) > 1

    @property
    def needs_strong_model(self) -> bool:
        """Whether this query likely needs a frontier model."""
        return self.difficulty >= 4

    @property
    def is_trivial(self) -> bool:
        """Whether any model in the pool could handle this."""
        return self.difficulty <= 2 and self.primary_capability == Capability.SIMPLE and not self.is_mixed

    def min_benchmark_threshold(self) -> float:
        """Minimum benchmark score a model should have for this query.

        Maps difficulty 1-5 to a threshold on the relevant benchmark.
        Difficulty 1 → 0.2 (almost any model), Difficulty 5 → 0.9 (frontier only).
        """
        return 0.2 + (self.difficulty - 1) * 0.175  # 0.2, 0.375, 0.55, 0.725, 0.9

    @classmethod
    def single(
        cls,
        capability: Capability,
        difficulty: int,
        expected_output_length: str = "medium",
    ) -> QueryProfile:
        """Convenience: create a single-capability profile."""
        return cls(
            capability_weights={capability.value: 1.0},
            difficulty=difficulty,
            expected_output_length=expected_output_length,
        )
