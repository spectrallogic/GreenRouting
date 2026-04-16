"""Run the GreenRouting interactive demo.

Usage:
    python -m greenrouting.demo

No API keys required. Shows routing decisions, energy savings, and
compression hints for a variety of queries.
"""

from __future__ import annotations


def main() -> None:
    import sys
    from pathlib import Path

    # When running from the repo, examples/ is at the repo root
    repo_root = Path(__file__).resolve().parent.parent
    examples_dir = repo_root / "examples"

    if examples_dir.exists() and (examples_dir / "demo.py").exists():
        # Running from repo — import the examples module
        sys.path.insert(0, str(repo_root))
        from examples.demo import run_demo

        run_demo()
    else:
        # Installed via pip — examples aren't available
        print("GreenRouting Interactive Demo")
        print("=" * 40)
        print()
        print("The full interactive demo requires the examples/ directory.")
        print("Clone the repo to run it:")
        print()
        print("  git clone https://github.com/spectrallogic/GreenRouting.git")
        print("  cd GreenRouting")
        print("  pip install -e .")
        print("  python examples/demo.py")
        print()
        print("Quick routing test:")
        print()

        from greenrouting import load_pretrained

        router = load_pretrained()
        queries = [
            "What color is the sky?",
            "Write a Python quicksort function",
            "Prove sqrt(2) is irrational",
        ]
        for q in queries:
            d = router.route(q)
            print(f"  '{q}'")
            print(f"    -> {d.selected_model} (saves {d.energy_savings_vs_max:.0%} energy)")
            print()


if __name__ == "__main__":
    main()
