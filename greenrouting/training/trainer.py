"""Trainer — trains the query classifier on synthetic or real data."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from rich.progress import Progress, SpinnerColumn, TextColumn
from torch.utils.data import DataLoader, Dataset

from greenrouting.core.taxonomy import NUM_CAPABILITIES
from greenrouting.routers.classifier_router import (
    CAPABILITY_TO_IDX,
    OUTPUT_LENGTH_TO_IDX,
    ClassifierRouter,
)
from greenrouting.training.synthetic_data import TrainingExample


class QueryDataset(Dataset):
    """PyTorch dataset wrapping TrainingExamples with pre-computed embeddings.

    Supports both single-capability and multi-capability (weighted) examples.
    For multi-capability examples, the target is a soft distribution over
    capabilities rather than a hard one-hot label.
    """

    def __init__(
        self,
        examples: list[TrainingExample],
        router: ClassifierRouter,
    ) -> None:
        self.examples = examples
        # Pre-compute all embeddings for efficiency
        queries = [ex.query for ex in examples]
        self.embeddings = router.encoder.encode(
            queries, convert_to_tensor=True, show_progress_bar=True, device=router.device
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, float, int]:
        ex = self.examples[idx]
        embedding = self.embeddings[idx]

        # Build soft target distribution from capability_weights
        cap_target = torch.zeros(NUM_CAPABILITIES)
        for cap_name, weight in ex.capability_weights.items():
            if cap_name in CAPABILITY_TO_IDX:
                cap_target[CAPABILITY_TO_IDX[cap_name]] = weight

        difficulty = float(ex.difficulty)
        len_idx = OUTPUT_LENGTH_TO_IDX[ex.expected_output_length]
        return embedding, cap_target, difficulty, len_idx


@dataclass
class TrainingResult:
    """Result of a training run."""

    epochs_completed: int
    final_loss: float
    capability_accuracy: float
    difficulty_mae: float
    output_length_accuracy: float


def train_router(
    router: ClassifierRouter,
    train_examples: list[TrainingExample],
    val_examples: list[TrainingExample] | None = None,
    epochs: int = 20,
    lr: float = 1e-3,
    batch_size: int = 32,
    save_path: str | Path | None = None,
) -> TrainingResult:
    """Train the classifier head on labeled query data.

    The sentence encoder is frozen — only the MLP head is trained.
    This makes training fast (~seconds on CPU for synthetic data).

    Args:
        router: ClassifierRouter instance to train.
        train_examples: Labeled training data.
        val_examples: Optional validation data.
        epochs: Number of training epochs.
        lr: Learning rate.
        batch_size: Batch size.
        save_path: If provided, save the trained model here.

    Returns:
        TrainingResult with final metrics.
    """
    # Build datasets
    train_ds = QueryDataset(train_examples, router)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_loader = None
    if val_examples:
        val_ds = QueryDataset(val_examples, router)
        val_loader = DataLoader(val_ds, batch_size=batch_size)

    # Loss functions — soft CrossEntropy for capabilities (supports multi-label weights)
    diff_loss_fn = nn.MSELoss()
    len_loss_fn = nn.CrossEntropyLoss()

    # Only train the head, not the encoder
    optimizer = torch.optim.Adam(router.head.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    router.head.train()
    best_val_loss = float("inf")

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        TextColumn("{task.fields[metrics]}"),
    ) as progress:
        task = progress.add_task("Training", total=epochs, metrics="")

        for epoch in range(epochs):
            total_loss = 0.0
            n_batches = 0

            for embeddings, cap_targets, diff_targets, len_targets in train_loader:
                embeddings = embeddings.to(router.device)
                cap_targets = cap_targets.to(router.device).float()
                diff_targets = diff_targets.to(router.device).float()
                len_targets = len_targets.to(router.device)

                cap_logits, diff_preds, len_logits = router.head(embeddings)

                # Soft cross-entropy: works for both single-label and multi-label
                cap_log_probs = torch.nn.functional.log_softmax(cap_logits, dim=-1)
                cap_loss = -(cap_targets * cap_log_probs).sum(dim=-1).mean()

                loss = (
                    cap_loss
                    + 0.5 * diff_loss_fn(diff_preds, diff_targets)
                    + 0.3 * len_loss_fn(len_logits, len_targets)
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg_loss = total_loss / max(n_batches, 1)

            # Validation
            metrics_str = f"epoch {epoch + 1}/{epochs}, loss: {avg_loss:.4f}"
            if val_loader:
                val_metrics = _evaluate(router, val_loader)
                metrics_str += (
                    f" | val_cap_acc: {val_metrics['cap_acc']:.3f}, "
                    f"val_diff_mae: {val_metrics['diff_mae']:.3f}"
                )
                val_loss = val_metrics["loss"]
                if val_loss < best_val_loss and save_path:
                    best_val_loss = val_loss
                    router.save(save_path)

            progress.update(task, advance=1, metrics=metrics_str)

    # Final evaluation
    eval_loader = val_loader or train_loader
    final_metrics = _evaluate(router, eval_loader)

    if save_path and not val_loader:
        router.save(save_path)

    return TrainingResult(
        epochs_completed=epochs,
        final_loss=final_metrics["loss"],
        capability_accuracy=final_metrics["cap_acc"],
        difficulty_mae=final_metrics["diff_mae"],
        output_length_accuracy=final_metrics["len_acc"],
    )


def _evaluate(
    router: ClassifierRouter,
    loader: DataLoader,
) -> dict[str, float]:
    """Evaluate the classifier on a data loader."""
    router.head.eval()

    diff_loss_fn = nn.MSELoss()
    len_loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    cap_correct = 0
    len_correct = 0
    diff_abs_error = 0.0
    total = 0

    with torch.no_grad():
        for embeddings, cap_targets, diff_targets, len_targets in loader:
            embeddings = embeddings.to(router.device)
            cap_targets = cap_targets.to(router.device).float()
            diff_targets = diff_targets.to(router.device).float()
            len_targets = len_targets.to(router.device)

            cap_logits, diff_preds, len_logits = router.head(embeddings)

            cap_log_probs = torch.nn.functional.log_softmax(cap_logits, dim=-1)
            cap_loss = -(cap_targets * cap_log_probs).sum(dim=-1).mean()

            loss = (
                cap_loss
                + 0.5 * diff_loss_fn(diff_preds, diff_targets)
                + 0.3 * len_loss_fn(len_logits, len_targets)
            )
            total_loss += loss.item()

            # Accuracy: compare argmax of prediction vs argmax of target
            cap_preds = cap_logits.argmax(dim=-1)
            cap_true = cap_targets.argmax(dim=-1)
            cap_correct += (cap_preds == cap_true).sum().item()

            len_preds = len_logits.argmax(dim=-1)
            len_correct += (len_preds == len_targets).sum().item()

            diff_abs_error += (diff_preds - diff_targets).abs().sum().item()
            total += len(embeddings)

    router.head.train()

    n_batches = len(loader)
    return {
        "loss": total_loss / max(n_batches, 1),
        "cap_acc": cap_correct / max(total, 1),
        "diff_mae": diff_abs_error / max(total, 1),
        "len_acc": len_correct / max(total, 1),
    }
