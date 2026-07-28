"""Shared training scaffolding for the STL-10 mini-Inception scripts.

`scripts/train_stl.py` (BatchNorm) and `scripts/train_stl_bnfree.py` (norm-free)
differ only in the optimizer, scheduler, `use_bn`, and a couple of default
hyperparameters. Everything else — the STL loaders, evaluation, snapshotting,
the shared CLI flags, and the epoch loop — lives here so it exists once.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import STL10
from tqdm import tqdm

from olt.neuron_tracking import NeuronTracker, add_tracking_args
from olt.tfms import stl_fancy_train_transform, stl_train_transform, stl_transform

STEM_STRIDE = 2  # 96x96 input; must match what stl_inception(ckpt_path=...) loads with


def make_loaders(
    data_root: Path, batch_size: int, num_workers: int, train_transform
) -> tuple[DataLoader, DataLoader]:
    train_ds = STL10(
        root=str(data_root), split="train", download=True, transform=train_transform
    )
    test_ds = STL10(
        root=str(data_root), split="test", download=True, transform=stl_transform
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, test_loader


@torch.no_grad()
def evaluate(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    for x, y in tqdm(loader, desc="eval"):
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    model.train()
    return correct / total


def save_snapshot(
    model,
    ckpt_dir: Path,
    step: int,
    epoch: int,
    test_acc: float,
    extra: dict | None = None,
) -> None:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"step_{step:06d}.pt"
    snapshot = {
        "model": model.state_dict(),
        "step": step,
        "epoch": epoch,
        "test_acc": test_acc,
    }
    if extra:
        snapshot.update(extra)  # e.g. {"use_bn": False} marker for the norm-free net
    torch.save(snapshot, path)
    print(f"  snapshot -> {path} (step={step}, epoch={epoch}, test_acc={test_acc:.4f})")


def add_shared_args(parser: argparse.ArgumentParser) -> None:
    """Flags common to both STL training scripts. Each script still adds its own
    `--epochs` / `--lr` (different defaults) and any optimizer-specific flags."""
    parser.add_argument("--ckpt-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/stl-raw"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--fancy-aug",
        action="store_true",
        help="use stl_fancy_train_transform (random-mode pad crop + color jitter "
        "+ random grayscale) to fight black-border/color shortcuts",
    )
    parser.add_argument(
        "--snapshot-every-steps",
        type=int,
        default=None,
        help="snapshot every N global steps (mutually exclusive with --snapshot-steps)",
    )
    parser.add_argument(
        "--snapshot-steps",
        type=int,
        nargs="+",
        default=None,
        help="explicit global steps at which to snapshot",
    )
    add_tracking_args(parser)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")


def resolve_train_transform(fancy_aug: bool):
    transform = stl_fancy_train_transform if fancy_aug else stl_train_transform
    print(f"train transform: {'fancy' if fancy_aug else 'plain'}")
    return transform


def make_snapshot_predicate(
    every_steps: int | None, explicit_steps: list[int] | None
) -> Callable[[int], bool]:
    """A step -> bool predicate. Defaults to every 500 steps when neither is set."""
    if every_steps is None and explicit_steps is None:
        every_steps = 500  # sensible default schedule
    explicit = set(explicit_steps or [])

    def should_snapshot(step: int) -> bool:
        if step in explicit:
            return True
        return every_steps is not None and step % every_steps == 0

    return should_snapshot


def train_loop(
    *,
    model,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device,
    optimizer,
    scheduler,
    criterion,
    epochs: int,
    ckpt_dir: Path,
    should_snapshot: Callable[[int], bool],
    snapshot_extra: dict | None = None,
    grad_clip: float | None = None,
    tracker: NeuronTracker | None = None,
) -> None:
    """The shared epoch loop. `grad_clip` (max grad norm; <=0 or None disables)
    and `tracker` are the only per-script variations; the baseline neuron record
    is taken by `build_tracker` before this is called."""
    global_step = 0
    for epoch in range(epochs):
        running = 0.0
        for x, y in tqdm(train_loader, desc=f"epoch {epoch}"):
            # snapshot BEFORE the step so step 0 captures the init (random) model
            if should_snapshot(global_step):
                acc = evaluate(model, test_loader, device)
                save_snapshot(model, ckpt_dir, global_step, epoch, acc, snapshot_extra)

            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            if grad_clip and grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            running += loss.item()
            global_step += 1
            if tracker is not None:
                tracker.record(model, step=global_step)

        acc = evaluate(model, test_loader, device)
        print(f"epoch {epoch}: loss={running / len(train_loader):.4f} test_acc={acc:.4f}")

    # always snapshot the final model
    acc = evaluate(model, test_loader, device)
    save_snapshot(model, ckpt_dir, global_step, epochs, acc, snapshot_extra)

    if tracker is not None:
        tracker.save()
