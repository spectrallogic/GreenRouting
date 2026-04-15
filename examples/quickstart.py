"""GreenRouting Quickstart — see intelligent routing in action.

This example shows the full pipeline:
1. Register models (or use pre-built profiles)
2. Classify queries by what they NEED (single or multi-capability)
3. Route to the greenest model that can handle them
4. Track energy savings
"""

from greenrouting import (
    BenchmarkMatcher,
    Capability,
    EnergyTracker,
    GreenScorer,
    ModelRegistry,
    QueryProfile,
    get_compression_hint,
    get_known_profiles,
)


def main() -> None:
    # ── 1. Set up the model pool ──────────────────────────────────────
    registry = ModelRegistry()
    for profile in get_known_profiles().values():
        registry.register(profile)

    print(f"Registered {len(registry)} models: {registry.model_names}\n")

    # ── 2. Configure the Green Score ──────────────────────────────────
    scorer = GreenScorer.from_config({"green_score": {"preset": "balanced"}})
    matcher = BenchmarkMatcher(registry, scorer)
    tracker = EnergyTracker()

    # ── 3. Route some queries ─────────────────────────────────────────
    # Simulate what the trained classifier would predict for each query
    test_queries = [
        ("What is 2+2?",
         QueryProfile.single(Capability.SIMPLE, difficulty=1)),
        ("Explain quantum entanglement",
         QueryProfile.single(Capability.KNOWLEDGE, difficulty=3)),
        ("Write a binary search in Python",
         QueryProfile.single(Capability.CODE, difficulty=3)),
        ("Prove the Cauchy-Schwarz inequality",
         QueryProfile.single(Capability.MATH, difficulty=5)),
        ("Translate 'hello' to Spanish",
         QueryProfile.single(Capability.SIMPLE, difficulty=1)),
        # Mixed-capability: code + math + reasoning
        ("Write a Python script that solves a differential equation using RK4",
         QueryProfile(capability_weights={"code": 0.5, "math": 0.35, "reasoning": 0.15}, difficulty=4)),
    ]

    print("=" * 70)
    print("ROUTING DECISIONS")
    print("=" * 70)

    for query, profile in test_queries:
        decision = matcher.match(profile)
        hint = get_compression_hint(profile)

        # Track energy
        max_energy = max(s.energy_estimate_wh for s in decision.all_scores.values())
        max_cost = max(s.cost_estimate for s in decision.all_scores.values())
        tracker.record(decision.energy_estimate_wh, max_energy, decision.cost_estimate, max_cost)

        cap_str = profile.primary_capability.value
        if profile.is_mixed:
            caps = ", ".join(f"{k} {v:.0%}" for k, v in
                            sorted(profile.capability_weights.items(), key=lambda x: -x[1]))
            cap_str = f"[{caps}]"

        print(f"\nQuery: \"{query}\"")
        print(f"  Needs:    {cap_str} (difficulty {profile.difficulty}/5)")
        print(f"  Routed:   {decision.selected_model}")
        print(f"  Energy:   {decision.energy_estimate_wh:.4f} Wh")
        print(f"  Savings:  {decision.energy_savings_vs_max:.0%} vs most expensive model")
        if hint.should_compress:
            print(f"  Compress: {hint.level} (~{hint.estimated_token_savings_pct:.0%} output token savings)")

    # ── 4. Impact report ──────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print(tracker.report())


if __name__ == "__main__":
    main()
