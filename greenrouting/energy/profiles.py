"""Pre-built energy and benchmark profiles for popular models.

Users can use these as a starting point — just register the ones they have
API access to. Benchmark scores are from public leaderboards (approximate).
"""

from __future__ import annotations

from greenrouting.core.model_profile import ModelProfile


def get_known_profiles() -> dict[str, ModelProfile]:
    """Return pre-built profiles for well-known models.

    Benchmark scores are normalized to 0-1 scale. Sources:
    - LMSYS Chatbot Arena ELO (converted to 0-1)
    - Open LLM Leaderboard
    - Published papers and model cards
    """
    return {p.name: p for p in [
        # ── OpenAI ────────────────────────────────────────────────────
        ModelProfile(
            name="gpt-4o",
            provider="openai",
            model_id="gpt-4o",
            estimated_params_b=200,
            cost_per_1k_input=0.005,
            cost_per_1k_output=0.015,
            avg_latency_ms=800,
            energy_per_query_wh=0.2,
            benchmark_scores={
                "mmlu": 0.887, "gsm8k": 0.95, "humaneval": 0.90,
                "arc_challenge": 0.96, "mt_bench": 0.91, "ifeval": 0.87,
            },
            tags={"general", "code", "reasoning", "math"},
        ),
        ModelProfile(
            name="gpt-4o-mini",
            provider="openai",
            model_id="gpt-4o-mini",
            estimated_params_b=8,
            cost_per_1k_input=0.00015,
            cost_per_1k_output=0.0006,
            avg_latency_ms=200,
            energy_per_query_wh=0.008,
            benchmark_scores={
                "mmlu": 0.82, "gsm8k": 0.87, "humaneval": 0.87,
                "arc_challenge": 0.85, "mt_bench": 0.82, "ifeval": 0.80,
            },
            tags={"general", "code"},
        ),
        ModelProfile(
            name="gpt-3.5-turbo",
            provider="openai",
            model_id="gpt-3.5-turbo",
            estimated_params_b=20,
            cost_per_1k_input=0.0005,
            cost_per_1k_output=0.0015,
            avg_latency_ms=300,
            energy_per_query_wh=0.02,
            benchmark_scores={
                "mmlu": 0.70, "gsm8k": 0.57, "humaneval": 0.48,
                "arc_challenge": 0.78, "mt_bench": 0.72, "ifeval": 0.65,
            },
            tags={"general"},
        ),
        # ── Anthropic ─────────────────────────────────────────────────
        ModelProfile(
            name="claude-opus",
            provider="anthropic",
            model_id="claude-opus-4-20250514",
            estimated_params_b=300,
            cost_per_1k_input=0.015,
            cost_per_1k_output=0.075,
            avg_latency_ms=1200,
            energy_per_query_wh=0.3,
            benchmark_scores={
                "mmlu": 0.90, "gsm8k": 0.96, "humaneval": 0.93,
                "arc_challenge": 0.97, "mt_bench": 0.94, "ifeval": 0.90,
                "gpqa": 0.65,
            },
            tags={"general", "code", "reasoning", "math"},
        ),
        ModelProfile(
            name="claude-sonnet",
            provider="anthropic",
            model_id="claude-sonnet-4-20250514",
            estimated_params_b=70,
            cost_per_1k_input=0.003,
            cost_per_1k_output=0.015,
            avg_latency_ms=600,
            energy_per_query_wh=0.07,
            benchmark_scores={
                "mmlu": 0.887, "gsm8k": 0.92, "humaneval": 0.93,
                "arc_challenge": 0.93, "mt_bench": 0.90, "ifeval": 0.86,
            },
            tags={"general", "code", "reasoning"},
        ),
        ModelProfile(
            name="claude-haiku",
            provider="anthropic",
            model_id="claude-haiku-4-20250514",
            estimated_params_b=8,
            cost_per_1k_input=0.0008,
            cost_per_1k_output=0.004,
            avg_latency_ms=200,
            energy_per_query_wh=0.008,
            benchmark_scores={
                "mmlu": 0.80, "gsm8k": 0.85, "humaneval": 0.80,
                "arc_challenge": 0.83, "mt_bench": 0.80, "ifeval": 0.77,
            },
            tags={"general"},
        ),
        # ── Meta (Open Source) ────────────────────────────────────────
        ModelProfile(
            name="llama-3.1-405b",
            provider="meta",
            estimated_params_b=405,
            cost_per_1k_input=0.003,
            cost_per_1k_output=0.003,
            avg_latency_ms=1500,
            energy_per_query_wh=0.4,
            benchmark_scores={
                "mmlu": 0.87, "gsm8k": 0.96, "humaneval": 0.89,
                "arc_challenge": 0.95, "mt_bench": 0.88, "ifeval": 0.86,
            },
            tags={"general", "code", "reasoning", "math"},
        ),
        ModelProfile(
            name="llama-3.1-70b",
            provider="meta",
            estimated_params_b=70,
            cost_per_1k_input=0.0009,
            cost_per_1k_output=0.0009,
            avg_latency_ms=500,
            energy_per_query_wh=0.07,
            benchmark_scores={
                "mmlu": 0.82, "gsm8k": 0.91, "humaneval": 0.80,
                "arc_challenge": 0.88, "mt_bench": 0.84, "ifeval": 0.80,
            },
            tags={"general", "code", "reasoning"},
        ),
        ModelProfile(
            name="llama-3.1-8b",
            provider="meta",
            estimated_params_b=8,
            cost_per_1k_input=0.0001,
            cost_per_1k_output=0.0001,
            avg_latency_ms=100,
            energy_per_query_wh=0.005,
            benchmark_scores={
                "mmlu": 0.68, "gsm8k": 0.76, "humaneval": 0.62,
                "arc_challenge": 0.73, "mt_bench": 0.68, "ifeval": 0.62,
            },
            tags={"general"},
        ),
        # ── Google ────────────────────────────────────────────────────
        ModelProfile(
            name="gemini-2.0-flash",
            provider="google",
            model_id="gemini-2.0-flash",
            estimated_params_b=30,
            cost_per_1k_input=0.0001,
            cost_per_1k_output=0.0004,
            avg_latency_ms=250,
            energy_per_query_wh=0.015,
            benchmark_scores={
                "mmlu": 0.83, "gsm8k": 0.90, "humaneval": 0.82,
                "arc_challenge": 0.86, "mt_bench": 0.84, "ifeval": 0.82,
            },
            tags={"general", "code"},
        ),
        # ── Mistral ───────────────────────────────────────────────────
        ModelProfile(
            name="mistral-small",
            provider="mistral",
            model_id="mistral-small-latest",
            estimated_params_b=22,
            cost_per_1k_input=0.0002,
            cost_per_1k_output=0.0006,
            avg_latency_ms=200,
            energy_per_query_wh=0.012,
            benchmark_scores={
                "mmlu": 0.75, "gsm8k": 0.80, "humaneval": 0.70,
                "arc_challenge": 0.80, "mt_bench": 0.76, "ifeval": 0.72,
            },
            tags={"general"},
        ),
    ]}
