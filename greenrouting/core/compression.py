"""Caveman-inspired output compression hints.

Inspired by the Caveman repo (github.com/JuliusBrussee/caveman), which saves
~65% output tokens via terse communication. GreenRouting integrates this as an
optional layer: after routing to the right model, we can also suggest output
compression to further reduce energy.

The compression level is based on the query difficulty:
- Trivial queries (difficulty 1-2): aggressive compression — be brief
- Medium queries (difficulty 3): moderate compression
- Complex queries (difficulty 4-5): no compression — full detail needed
"""

from __future__ import annotations

from dataclasses import dataclass

from greenrouting.core.taxonomy import QueryProfile


@dataclass
class CompressionHint:
    """Suggestion for how to compress the model's output."""

    level: str  # "none", "moderate", "aggressive"
    system_prompt_addon: str  # Text to append to the system prompt
    estimated_token_savings_pct: float  # Estimated % of output tokens saved

    @property
    def should_compress(self) -> bool:
        return self.level != "none"


# System prompt addons inspired by Caveman's approach
_AGGRESSIVE = (
    "Be extremely concise. Use fragments, not full sentences. "
    "Drop articles and filler words. Give the answer directly. "
    "No pleasantries, no preamble, no summary."
)

_MODERATE = "Be concise and direct. Skip unnecessary preamble. Focus on the answer, not the explanation unless asked."


def get_compression_hint(profile: QueryProfile) -> CompressionHint:
    """Determine the appropriate compression level for a query.

    This is the bridge between routing and caveman-style output compression.
    Simple queries get aggressive compression; complex ones get full output.
    """
    if profile.is_trivial or profile.difficulty <= 2:
        return CompressionHint(
            level="aggressive",
            system_prompt_addon=_AGGRESSIVE,
            estimated_token_savings_pct=0.65,
        )

    if profile.difficulty == 3:
        return CompressionHint(
            level="moderate",
            system_prompt_addon=_MODERATE,
            estimated_token_savings_pct=0.30,
        )

    # Difficulty 4-5: complex queries need full, detailed responses
    return CompressionHint(
        level="none",
        system_prompt_addon="",
        estimated_token_savings_pct=0.0,
    )
