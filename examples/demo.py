"""GreenRouting Interactive Demo — see intelligent model routing in action.

No API keys needed. This demo shows how GreenRouting classifies queries and
routes them to the most energy-efficient model that can handle the task.

    pip install greenrouting
    python examples/demo.py

Or:
    python -m greenrouting.demo
"""

from __future__ import annotations

import logging
import os
import warnings

# Silence HF/tqdm noise before importing greenrouting (which imports transformers)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from rich import box  # noqa: E402

# Rich is already a dependency of greenrouting
from rich.console import Console  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.table import Table  # noqa: E402

from greenrouting import GreenScorer, get_compression_hint, load_pretrained  # noqa: E402

console = Console()


def _quiet_load(**kwargs):
    """Load pretrained model with all loading noise suppressed."""
    logging.disable(logging.WARNING)
    try:
        from transformers.utils import logging as hf_logging

        hf_logging.disable_progress_bar()
        hf_logging.set_verbosity_error()
    except Exception:
        pass
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return load_pretrained(**kwargs)
        finally:
            logging.disable(logging.NOTSET)


# ── Demo queries spanning all 8 capability types ─────────────────────────

DEMO_QUERIES = [
    # (query, description of what it tests)
    ("What color is the sky?", "Simple factual"),
    ("What is 2+2?", "Trivial math"),
    ("Explain how vaccines create immunity in the body", "Knowledge retrieval"),
    ("Write a Python function that implements binary search", "Code generation"),
    ("Prove that the square root of 2 is irrational", "Advanced math"),
    (
        "Analyze the trade-offs between microservices and monolithic architecture for a startup with 5 engineers",
        "Complex reasoning",
    ),
    ("Write a haiku about autumn leaves", "Creative writing"),
    ("Translate 'The meeting is at 3pm tomorrow' to Japanese", "Multilingual"),
    (
        "Write a Python script that uses numerical methods to solve a system of differential equations",
        "Mixed: code + math + reasoning",
    ),
    (
        "Summarize this text in one sentence: AI is transforming industries.",
        "Instruction following",
    ),
]


def run_demo() -> None:
    console.print()
    console.print(
        Panel(
            "[bold]GreenRouting Demo[/bold]\n"
            "Intelligent model routing for sustainable AI\n\n"
            "Watch how the same query gets routed to different models\n"
            "depending on whether you prioritize quality or energy savings.",
            border_style="green",
            width=70,
        )
    )

    # Load the pre-trained classifier
    console.print("\n[dim]Loading pre-trained classifier...[/dim]")
    router = _quiet_load()
    console.print(f"[green]Ready![/green] {len(router.registry)} models in pool.\n")

    # ── Part 1: Classification ───────────────────────────────────────────

    console.rule("[bold]1. Query Classification[/bold]")
    console.print(
        "\nGreenRouting's neural classifier predicts what capabilities each\n"
        "query needs -- without knowing which models exist.\n"
    )

    class_table = Table(box=box.SIMPLE_HEAVY, show_edge=False)
    class_table.add_column("Query", style="cyan", max_width=50)
    class_table.add_column("Capabilities", style="yellow")
    class_table.add_column("Difficulty", justify="center")

    for query, _ in DEMO_QUERIES:
        profile = router.classify_query(query)

        # Format capability weights
        caps = sorted(profile.capability_weights.items(), key=lambda x: x[1], reverse=True)
        cap_str = ", ".join(f"{c} {w:.0%}" for c, w in caps if w >= 0.05)

        # Difficulty bar
        diff_bar = (
            "[red]" + ("*" * profile.difficulty) + "[/red]" + ("[dim]" + ("." * (5 - profile.difficulty)) + "[/dim]")
        )

        # Truncate query for display
        display_q = query if len(query) <= 50 else query[:47] + "..."
        class_table.add_row(display_q, cap_str, diff_bar)

    console.print(class_table)

    # ── Part 2: Quality dial ────────────────────────────────────────────

    console.rule("[bold]2. The Quality Dial[/bold]")
    console.print(
        "\nOne parameter controls everything: [bold]quality[/bold] (0.0 to 1.0)\n"
        "  0.0 = maximum energy savings (smallest capable model)\n"
        "  0.5 = balanced (default)\n"
        "  1.0 = maximum quality (smartest model available)\n"
    )

    # Load routers at different quality levels
    quality_levels = [0.0, 0.5, 1.0]
    q_labels = {0.0: "q=0 (green)", 0.5: "q=0.5 (balanced)", 1.0: "q=1 (quality)"}
    q_routers = {}
    for q in quality_levels:
        q_routers[q] = _quiet_load(quality=q)

    route_table = Table(box=box.SIMPLE_HEAVY, show_edge=False)
    route_table.add_column("Query", style="cyan", max_width=35)
    route_table.add_column(q_labels[0.0], style="yellow")
    route_table.add_column(q_labels[0.5], style="green")
    route_table.add_column(q_labels[1.0], style="blue")

    for query, label in DEMO_QUERIES:
        display_q = query if len(query) <= 35 else query[:32] + "..."

        cells = []
        for q in quality_levels:
            decision = q_routers[q].route(query)
            cells.append(decision.selected_model)

        route_table.add_row(display_q, *cells)

    console.print(route_table)

    console.print("\n[dim]Usage:  router = load_pretrained(quality=0.7)[/dim]\n")

    # ── Part 3: Energy savings detail ────────────────────────────────────

    console.rule("[bold]3. Energy Savings Detail[/bold]")
    console.print("\nFor each query, we show the energy used vs. always picking the largest model.\n")

    balanced_router = q_routers[0.5]

    # Find the most expensive model for reference
    all_models = balanced_router.registry.list_models()
    scorer = GreenScorer()
    max_energy_model = max(all_models, key=lambda m: scorer._estimate_energy(m))
    max_energy_wh = scorer._estimate_energy(max_energy_model)

    energy_table = Table(box=box.SIMPLE_HEAVY, show_edge=False)
    energy_table.add_column("Query", style="cyan", max_width=35)
    energy_table.add_column("Routed To", style="green")
    energy_table.add_column("Energy", justify="right")
    energy_table.add_column(f"vs {max_energy_model.name}", justify="right", style="yellow")
    energy_table.add_column("Cost", justify="right")

    total_routed_energy = 0.0
    total_max_energy = 0.0
    total_routed_cost = 0.0
    total_max_cost = 0.0

    for query, _ in DEMO_QUERIES:
        decision = balanced_router.route(query)
        savings = decision.energy_savings_vs_max

        display_q = query if len(query) <= 35 else query[:32] + "..."

        energy_table.add_row(
            display_q,
            decision.selected_model,
            f"{decision.energy_estimate_wh:.4f} Wh",
            f"[bold]-{savings:.0%}[/bold]" if savings > 0 else "0%",
            f"${decision.cost_estimate:.5f}",
        )

        total_routed_energy += decision.energy_estimate_wh
        total_max_energy += max_energy_wh
        total_routed_cost += decision.cost_estimate
        # Get cost of max model from the scores
        if decision.all_scores and max_energy_model.name in decision.all_scores:
            total_max_cost += decision.all_scores[max_energy_model.name].cost_estimate

    energy_table.add_section()
    overall_savings = (total_max_energy - total_routed_energy) / total_max_energy if total_max_energy > 0 else 0
    cost_savings = (total_max_cost - total_routed_cost) / total_max_cost if total_max_cost > 0 else 0
    energy_table.add_row(
        "[bold]TOTAL (10 queries)[/bold]",
        "",
        f"[bold]{total_routed_energy:.4f} Wh[/bold]",
        f"[bold green]-{overall_savings:.0%}[/bold green]",
        f"[bold]${total_routed_cost:.5f}[/bold]",
    )

    console.print(energy_table)

    # ── Part 4: Compression hints ────────────────────────────────────────

    console.rule("[bold]4. Output Compression Hints[/bold]")
    console.print(
        "\nFor simple queries, GreenRouting suggests compressed output to save\n"
        "even more energy (inspired by the Caveman approach).\n"
    )

    comp_table = Table(box=box.SIMPLE_HEAVY, show_edge=False)
    comp_table.add_column("Query", style="cyan", max_width=40)
    comp_table.add_column("Compression", justify="center")
    comp_table.add_column("Token Savings")

    for query, _ in DEMO_QUERIES:
        profile = router.classify_query(query)
        hint = get_compression_hint(profile)

        display_q = query if len(query) <= 40 else query[:37] + "..."

        if hint.level == "aggressive":
            level_str = "[bold red]aggressive[/bold red]"
        elif hint.level == "moderate":
            level_str = "[yellow]moderate[/yellow]"
        else:
            level_str = "[dim]none[/dim]"

        savings_str = f"~{hint.estimated_token_savings_pct:.0%}" if hint.should_compress else "-"

        comp_table.add_row(display_q, level_str, savings_str)

    console.print(comp_table)

    # ── Summary ──────────────────────────────────────────────────────────

    console.print()
    console.print(
        Panel(
            f"[bold green]Summary[/bold green]\n\n"
            f"Across {len(DEMO_QUERIES)} queries with the [bold]balanced[/bold] preset:\n"
            f"  Energy saved:  [bold]{overall_savings:.0%}[/bold] vs always using {max_energy_model.name}\n"
            f"  Cost saved:    [bold]{cost_savings:.0%}[/bold]\n\n"
            f"At scale (1M queries/day), that's ~[bold]"
            f"{(total_max_energy - total_routed_energy) / len(DEMO_QUERIES) * 1_000_000:.0f} Wh"
            f"[/bold] saved daily.\n\n"
            f"[dim]No API keys were used. GreenRouting routes based on query\n"
            f"classification and model benchmarks, not live API calls.[/dim]",
            border_style="green",
            width=70,
        )
    )
    console.print()


if __name__ == "__main__":
    run_demo()
