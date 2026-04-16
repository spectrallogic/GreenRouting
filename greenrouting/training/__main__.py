"""Train the GreenRouting classifier and save it for use.

Usage:
    python -m greenrouting.training                    # train and save to greenrouting/pretrained/
    python -m greenrouting.training --output ./my_model  # save to custom path
    python -m greenrouting.training --epochs 50          # more epochs
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path

from greenrouting import ClassifierRouter, ModelRegistry, get_known_profiles
from greenrouting.training.synthetic_data import generate_dataset
from greenrouting.training.trainer import train_router


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the GreenRouting classifier on synthetic data.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=str(Path(__file__).parent.parent / "pretrained"),
        help="Directory to save the trained model (default: greenrouting/pretrained/)",
    )
    parser.add_argument("--epochs", type=int, default=25, help="Training epochs (default: 25)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument(
        "--n-per-category",
        type=int,
        default=150,
        help="Training examples per category (default: 150, produces ~6600 total)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    args = parser.parse_args()

    print("GreenRouting Classifier Training")
    print("=" * 50)

    # Generate synthetic training data
    print(f"\n[1/4] Generating synthetic data ({args.n_per_category} per category)...")
    all_examples = generate_dataset(n_per_category=args.n_per_category, seed=args.seed)
    print(f"  Generated {len(all_examples)} training examples")

    # Split train/val
    rng = random.Random(args.seed)
    shuffled = list(all_examples)
    rng.shuffle(shuffled)
    split = int(len(shuffled) * 0.85)
    train_data = shuffled[:split]
    val_data = shuffled[split:]
    print(f"  Train: {len(train_data)} | Val: {len(val_data)}")

    # Set up registry and router
    print("\n[2/4] Initializing model registry and router...")
    registry = ModelRegistry()
    for profile in get_known_profiles().values():
        registry.register(profile)
    print(f"  Registered {len(registry)} models")

    router = ClassifierRouter(registry, config={"encoder_model": "all-MiniLM-L6-v2"})
    print(f"  Encoder: all-MiniLM-L6-v2 | Device: {router.device}")

    # Train
    print(f"\n[3/4] Training classifier ({args.epochs} epochs, batch_size={args.batch_size})...")
    start = time.time()
    result = train_router(
        router=router,
        train_examples=train_data,
        val_examples=val_data,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        save_path=args.output,
    )
    elapsed = time.time() - start

    print(f"\n  Training complete in {elapsed:.1f}s")
    print(f"  Capability accuracy: {result.capability_accuracy:.1%}")
    print(f"  Difficulty MAE:      {result.difficulty_mae:.2f}")
    print(f"  Output length acc:   {result.output_length_accuracy:.1%}")

    # Save
    print(f"\n[4/4] Saving model to {args.output}")
    router.save(args.output)
    print("  Saved: classifier_head.pt + config.json")

    # Show how to use it
    print(f"\n{'=' * 50}")
    print("Done! Load your trained model with:\n")
    print("  from greenrouting import load_pretrained")
    print(f'  router = load_pretrained("{args.output}")')
    print('  decision = router.route("What is 2+2?")')
    print("  print(decision.selected_model)")


if __name__ == "__main__":
    main()
