"""Interactive REPL for GreenRouting — type queries, see routing decisions.

Usage:
    python -m greenrouting.repl

Commands:
    /quality <0.0-1.0>   Change the quality dial (reloads the router)
    /help                Show commands
    /quit                Exit (also: Ctrl-D, Ctrl-C)
"""

from __future__ import annotations

import logging
import os
import warnings

# Silence HF/tqdm noise before importing greenrouting (which imports transformers)
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from rich.console import Console  # noqa: E402

from greenrouting import get_compression_hint, load_pretrained  # noqa: E402

console = Console()


def _quiet_load(quality: float):
    """Load pretrained model with loading noise suppressed."""
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
            return load_pretrained(quality=quality)
        finally:
            logging.disable(logging.NOTSET)


def _print_help() -> None:
    console.print(
        "\n[bold]Commands[/bold]\n"
        "  [cyan]/quality <0.0-1.0>[/cyan]  change the quality dial\n"
        "  [cyan]/help[/cyan]               show this help\n"
        "  [cyan]/quit[/cyan]               exit\n"
    )


def _route_and_print(router, query: str) -> None:
    decision = router.route(query)
    profile = router.classify_query(query)
    hint = get_compression_hint(profile)

    caps = sorted(profile.capability_weights.items(), key=lambda x: x[1], reverse=True)
    cap_str = ", ".join(f"{c} {w:.0%}" for c, w in caps if w >= 0.10)

    console.print(
        f"  [green]->[/green] [bold]{decision.selected_model}[/bold]   "
        f"energy=[yellow]{decision.energy_estimate_wh:.4f} Wh[/yellow]   "
        f"savings=[green]{decision.energy_savings_vs_max:.0%}[/green]   "
        f"cost=[cyan]${decision.cost_estimate:.5f}[/cyan]"
    )
    console.print(
        f"     [dim]caps: {cap_str}  |  difficulty: {profile.difficulty}/5  |  compress: {hint.level}[/dim]\n"
    )


def main() -> None:
    quality = 0.5
    console.print("\n[bold green]GreenRouting REPL[/bold green]  [dim](type /help, /quit)[/dim]")
    console.print(f"[dim]Loading pretrained classifier... (quality={quality})[/dim]")
    router = _quiet_load(quality)
    console.print(f"[dim]Ready. {len(router.registry)} models in pool.[/dim]\n")

    while True:
        try:
            line = console.input(f"[bold](q={quality})[/bold] > ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye[/dim]")
            return

        if not line:
            continue

        if line in ("/quit", "/exit", "/q"):
            console.print("[dim]bye[/dim]")
            return

        if line in ("/help", "/?"):
            _print_help()
            continue

        if line.startswith("/quality"):
            parts = line.split()
            if len(parts) != 2:
                console.print("[red]usage: /quality <0.0-1.0>[/red]")
                continue
            try:
                new_q = max(0.0, min(1.0, float(parts[1])))
            except ValueError:
                console.print("[red]quality must be a float between 0.0 and 1.0[/red]")
                continue
            quality = new_q
            console.print(f"[dim]Reloading at quality={quality}...[/dim]")
            router = _quiet_load(quality)
            console.print("[dim]Ready.[/dim]\n")
            continue

        if line.startswith("/"):
            console.print(f"[red]unknown command: {line}[/red]  [dim](try /help)[/dim]")
            continue

        _route_and_print(router, line)


if __name__ == "__main__":
    main()
