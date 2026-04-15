"""Load a pre-trained GreenRouting model and route queries.

This is the simplest way to use GreenRouting after training:

    python examples/train_and_evaluate.py   # train and save the model
    python examples/load_and_route.py       # load and route queries

Or use the one-liner in your own code:

    from greenrouting import load_pretrained
    router = load_pretrained()
    decision = router.route("your query here")
"""

from __future__ import annotations

from greenrouting import load_pretrained, get_compression_hint


def main() -> None:
    # Load the pre-trained classifier (uses bundled weights + 11 model profiles)
    print("Loading pre-trained GreenRouting classifier...")
    router = load_pretrained()
    print(f"Loaded! {len(router.registry)} models in pool.\n")

    # Route some queries
    queries = [
        "What is 2+2?",
        "Explain how photosynthesis works",
        "Write a Python function to sort a list using merge sort",
        "Prove that there are infinitely many prime numbers",
        "Translate 'good morning' to French",
        "Write a short poem about the ocean",
        "Write a Python script that solves a system of linear equations",
    ]

    print(f"{'Query':<55} {'Model':<20} {'Energy':>10} {'Savings':>8}")
    print("-" * 97)

    for query in queries:
        decision = router.route(query)
        profile = router.classify_query(query)
        hint = get_compression_hint(profile)

        compress = " +compress" if hint.should_compress else ""
        print(
            f"{query:<55} "
            f"{decision.selected_model + compress:<20} "
            f"{decision.energy_estimate_wh:>8.4f}Wh "
            f"{decision.energy_savings_vs_max:>7.0%}"
        )

    # Show how to use with custom model pools and presets
    print("\n--- Custom configurations ---\n")

    # Maximum green: prioritize energy savings over quality
    from greenrouting import ModelRegistry, GreenScorer, BenchmarkMatcher, get_known_profiles

    router_green = load_pretrained(
        scorer_config={"green_score": {"preset": "maximum_green"}}
    )
    decision = router_green.route("Explain quantum computing")
    print(f"maximum_green preset: {decision.selected_model} ({decision.energy_savings_vs_max:.0%} savings)")

    # Quality first: prioritize quality, still save some energy
    router_quality = load_pretrained(
        scorer_config={"green_score": {"preset": "quality_first"}}
    )
    decision = router_quality.route("Explain quantum computing")
    print(f"quality_first preset: {decision.selected_model} ({decision.energy_savings_vs_max:.0%} savings)")


if __name__ == "__main__":
    main()
