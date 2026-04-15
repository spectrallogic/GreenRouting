"""Train the GreenRouting classifier and evaluate it end-to-end.

This script:
1. Generates 10k+ synthetic training examples
2. Splits into train/val sets
3. Trains the classifier head (~30s on CPU)
4. Evaluates on held-out validation data
5. Runs the full pipeline: raw query → classifier → routing decision
"""

from __future__ import annotations

import random
import time
from pathlib import Path

from rich.console import Console
from rich.table import Table

from greenrouting import (
    BenchmarkMatcher,
    ClassifierRouter,
    EnergyTracker,
    GreenScorer,
    ModelRegistry,
    get_compression_hint,
    get_known_profiles,
)
from greenrouting.training.synthetic_data import generate_dataset
from greenrouting.training.trainer import train_router

console = Console()


def main() -> None:
    # ── 1. Generate training data ─────────────────────────────────────
    console.rule("[bold blue]Step 1: Generating synthetic training data")

    all_examples = generate_dataset(n_per_category=150, seed=42)
    console.print(f"Generated [bold]{len(all_examples)}[/bold] training examples")

    # Count by category
    cap_counts: dict[str, int] = {}
    mixed_count = 0
    for ex in all_examples:
        if len(ex.capability_weights) > 1:
            mixed_count += 1
        top = max(ex.capability_weights, key=lambda k: ex.capability_weights[k])
        cap_counts[top] = cap_counts.get(top, 0) + 1

    table = Table(title="Dataset Distribution")
    table.add_column("Capability", style="cyan")
    table.add_column("Count", justify="right")
    for cap in sorted(cap_counts):
        table.add_row(cap, str(cap_counts[cap]))
    table.add_row("[bold]Mixed-capability", f"[bold]{mixed_count}")
    table.add_row("[bold]Total", f"[bold]{len(all_examples)}")
    console.print(table)

    # ── 2. Split train/val ────────────────────────────────────────────
    console.rule("[bold blue]Step 2: Train/validation split")

    rng = random.Random(42)
    shuffled = list(all_examples)
    rng.shuffle(shuffled)
    split = int(len(shuffled) * 0.85)
    train_data = shuffled[:split]
    val_data = shuffled[split:]
    console.print(f"Train: [bold]{len(train_data)}[/bold] | Val: [bold]{len(val_data)}[/bold]")

    # ── 3. Set up registry and router ─────────────────────────────────
    console.rule("[bold blue]Step 3: Initializing router and model pool")

    registry = ModelRegistry()
    for profile in get_known_profiles().values():
        registry.register(profile)
    console.print(f"Registered [bold]{len(registry)}[/bold] models: {registry.model_names}")

    router = ClassifierRouter(registry, config={"encoder_model": "all-MiniLM-L6-v2"})
    console.print(f"Encoder: all-MiniLM-L6-v2 | Device: {router.device}")

    # ── 4. Train ──────────────────────────────────────────────────────
    console.rule("[bold blue]Step 4: Training classifier")
    start = time.time()

    result = train_router(
        router=router,
        train_examples=train_data,
        val_examples=val_data,
        epochs=25,
        lr=1e-3,
        batch_size=64,
    )

    elapsed = time.time() - start
    console.print(f"\nTraining complete in [bold]{elapsed:.1f}s[/bold]")
    console.print(f"  Capability accuracy: [bold green]{result.capability_accuracy:.1%}[/bold green]")
    console.print(f"  Difficulty MAE:      [bold]{result.difficulty_mae:.2f}[/bold] (out of 5)")
    console.print(f"  Output length acc:   [bold]{result.output_length_accuracy:.1%}[/bold]")
    console.print(f"  Final loss:          {result.final_loss:.4f}")

    # Save the trained model
    save_dir = Path(__file__).resolve().parent.parent / "greenrouting" / "pretrained"
    router.save(save_dir)
    console.print(f"\nModel saved to [bold]{save_dir}[/bold]")

    # ── 5. Evaluate on hand-crafted test queries ──────────────────────
    console.rule("[bold blue]Step 5: End-to-end evaluation on real queries")

    test_queries = [
        # (query, expected_primary_capability, expected_difficulty_range)
        ("What is 2+2?", "simple", (1, 2)),
        ("What color is the sky?", "simple", (1, 2)),
        ("How many legs does a dog have?", "simple", (1, 1)),
        ("Explain how vaccines work", "knowledge", (2, 3)),
        ("What are the key differences between TCP and UDP?", "knowledge", (3, 4)),
        ("Explain quantum entanglement in simple terms", "knowledge", (3, 4)),
        ("Describe the Standard Model of particle physics", "knowledge", (4, 5)),
        ("Write a function to check if a string is a palindrome", "code", (2, 3)),
        ("Implement binary search in Python", "code", (2, 3)),
        ("Implement an LRU cache with O(1) operations", "code", (4, 5)),
        ("Design a distributed consensus algorithm", "code", (5, 5)),
        ("Solve for x: 3x + 7 = 22", "math", (1, 2)),
        ("Find the derivative of f(x) = 3x^3 + 2x^2", "math", (3, 4)),
        ("Prove that the square root of 2 is irrational", "math", (4, 5)),
        ("If all cats are animals and all animals breathe, do all cats breathe?", "reasoning", (2, 3)),
        ("You have 12 balls, one is heavier. Find it in 3 weighings.", "reasoning", (4, 5)),
        ("Write a haiku about nature", "creative", (1, 2)),
        ("Write a short story with a twist ending about time travel", "creative", (3, 4)),
        ("Translate 'hello' to Spanish", "multilingual", (1, 2)),
        ("List exactly 3 benefits of exercise, numbered 1-3", "instruction", (2, 3)),
        # Mixed-capability queries
        ("Write a Python function to solve a quadratic equation", "code", (2, 4)),  # code + math
        ("Explain the trolley problem and analyze it from utilitarian and deontological perspectives", "reasoning", (3, 5)),  # reasoning + knowledge
        ("Write a historically accurate short story set during the French Revolution", "creative", (3, 4)),  # creative + knowledge
    ]

    # Classify each query and show results
    results_table = Table(title="Classifier Predictions vs Expected")
    results_table.add_column("Query", style="white", max_width=55)
    results_table.add_column("Expected", style="cyan")
    results_table.add_column("Predicted", style="green")
    results_table.add_column("Diff", justify="center")
    results_table.add_column("Match", justify="center")

    correct_cap = 0
    correct_diff = 0
    total = len(test_queries)

    for query, expected_cap, expected_diff_range in test_queries:
        profile = router.classify_query(query)
        predicted_cap = profile.primary_capability.value
        predicted_diff = profile.difficulty

        cap_match = predicted_cap == expected_cap
        diff_match = expected_diff_range[0] <= predicted_diff <= expected_diff_range[1]

        if cap_match:
            correct_cap += 1
        if diff_match:
            correct_diff += 1

        cap_str = f"[green]{predicted_cap}[/green]" if cap_match else f"[red]{predicted_cap}[/red]"
        diff_str = f"[green]{predicted_diff}[/green]" if diff_match else f"[red]{predicted_diff}[/red]"
        match_str = "[green]OK[/green]" if (cap_match and diff_match) else "[red]MISS[/red]"

        # Show top weights for mixed queries
        weights_str = ""
        if profile.is_mixed:
            top_caps = sorted(profile.capability_weights.items(), key=lambda x: -x[1])[:3]
            weights_str = " (" + ", ".join(f"{k}:{v:.0%}" for k, v in top_caps) + ")"

        results_table.add_row(
            query[:55],
            f"{expected_cap} d{expected_diff_range[0]}-{expected_diff_range[1]}",
            f"{cap_str}{weights_str}",
            diff_str,
            match_str,
        )

    console.print(results_table)
    console.print(f"\nCapability accuracy: [bold]{correct_cap}/{total} ({correct_cap/total:.0%})[/bold]")
    console.print(f"Difficulty accuracy: [bold]{correct_diff}/{total} ({correct_diff/total:.0%})[/bold]")

    # ── 6. Full routing pipeline demo ─────────────────────────────────
    console.rule("[bold blue]Step 6: Full routing pipeline (query string -> model selection)")

    scorer = GreenScorer.from_config({"green_score": {"preset": "balanced"}})
    router.matcher = BenchmarkMatcher(registry, scorer)
    tracker = EnergyTracker()

    demo_queries = [
        "What is the capital of France?",
        "Write a Python function to reverse a linked list",
        "Prove there are infinitely many prime numbers",
        "Explain how transformer models work in machine learning",
        "Write a haiku about the ocean",
        "Write a Python script that numerically integrates a function using Simpson's rule",
        "Translate 'I love programming' to Japanese",
        "Is 17 a prime number?",
    ]

    route_table = Table(title="Full Pipeline: Query -> Model")
    route_table.add_column("Query", style="white", max_width=60)
    route_table.add_column("Classified As", style="cyan")
    route_table.add_column("Routed To", style="green")
    route_table.add_column("Energy", justify="right")
    route_table.add_column("Savings", justify="right", style="bold green")

    for query in demo_queries:
        decision = router.route(query)
        profile = router.classify_query(query)
        hint = get_compression_hint(profile)

        max_energy = max(s.energy_estimate_wh for s in decision.all_scores.values())
        max_cost = max(s.cost_estimate for s in decision.all_scores.values())
        tracker.record(decision.energy_estimate_wh, max_energy, decision.cost_estimate, max_cost)

        cap_str = profile.primary_capability.value
        if profile.is_mixed:
            top = sorted(profile.capability_weights.items(), key=lambda x: -x[1])[:2]
            cap_str = "+".join(k for k, _ in top)
        cap_str += f" d{profile.difficulty}"

        compress_str = ""
        if hint.should_compress:
            compress_str = f" +compress"

        route_table.add_row(
            query[:60],
            cap_str,
            decision.selected_model + compress_str,
            f"{decision.energy_estimate_wh:.4f} Wh",
            f"{decision.energy_savings_vs_max:.0%}",
        )

    console.print(route_table)

    # Impact report
    report = tracker.report()
    console.print(f"\n[bold]Impact Report[/bold]")
    console.print(f"  Queries routed:  {report.total_queries}")
    console.print(f"  Energy used:     {report.total_energy_wh:.4f} Wh")
    console.print(f"  Energy saved:    {report.energy_saved_wh:.4f} Wh ([bold green]{report.energy_saved_pct:.1f}%[/bold green])")
    console.print(f"  Cost:            ${report.total_cost:.4f}")
    console.print(f"  Cost saved:      ${report.cost_saved:.4f} ([bold green]{report.cost_saved_pct:.1f}%[/bold green])")


if __name__ == "__main__":
    main()
