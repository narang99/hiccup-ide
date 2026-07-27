"""Train the CIFAR mini-Inception, saving snapshots during training.

The point of this model is to run the act_ranges / clustering pipeline over
MULTIPLE snapshots of the SAME model taken at different points in training, to
watch how the studied 1x1-conv circuits form. So training's main job, beyond
producing a decent model, is to checkpoint at a chosen schedule.

Each snapshot is written to <ckpt-dir>/step_<global_step>.pt as
    {"model": state_dict, "step": int, "epoch": int, "test_acc": float}
which `olt.models.cifar_inception.cifar_inception(ckpt_path=...)` loads
directly. Run the pipeline per snapshot with a per-snapshot output root (see
olt/CLAUDE.md notes on multi-snapshot orchestration).

Usage:
    uv run scripts/train_cifar.py --ckpt-dir data/cifar/checkpoints \
        --epochs 40 --snapshot-every-steps 500

    # or an explicit (log-spaced-ish) schedule of global steps:
    uv run scripts/train_cifar.py --ckpt-dir data/cifar/checkpoints \
        --epochs 40 --snapshot-steps 0 100 250 500 1000 2000 5000 10000
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tqdm import tqdm

from olt.models.cifar_inception import cifar_inception
from olt.tfms import cifar_train_transform, cifar_transform


def _make_loaders(data_root: Path, batch_size: int, num_workers: int):
    train_ds = CIFAR10(
        root=str(data_root), train=True, download=True, transform=cifar_train_transform
    )
    test_ds = CIFAR10(
        root=str(data_root), train=False, download=True, transform=cifar_transform
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
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    model.train()
    return correct / total


def save_snapshot(model, ckpt_dir: Path, step: int, epoch: int, test_acc: float):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"step_{step:06d}.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "step": step,
            "epoch": epoch,
            "test_acc": test_acc,
        },
        path,
    )
    print(f"  snapshot -> {path} (step={step}, epoch={epoch}, test_acc={test_acc:.4f})")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt-dir", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=Path("data/cifar-raw"))
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)  # Adam-appropriate
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--snapshot-every-steps",
        type=int,
        default=None,
        help="snapshot every N global steps (mutually exclusive with --snapshot-steps)",
    )
    p.add_argument(
        "--snapshot-steps",
        type=int,
        nargs="+",
        default=None,
        help="explicit global steps at which to snapshot",
    )
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if args.snapshot_every_steps is None and args.snapshot_steps is None:
        args.snapshot_every_steps = 500  # sensible default schedule

    explicit_steps = set(args.snapshot_steps or [])
    device = torch.device(args.device)

    train_loader, test_loader = _make_loaders(
        args.data_root, args.batch_size, args.num_workers
    )

    model = cifar_inception(redirected_relu=False, eval_mode=False).to(device)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"CifarInception: {n_params:,} params, device={device}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    total_steps = args.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps
    )

    def should_snapshot(step: int) -> bool:
        if step in explicit_steps:
            return True
        if args.snapshot_every_steps is not None:
            return step % args.snapshot_every_steps == 0
        return False

    global_step = 0
    for epoch in range(args.epochs):
        running = 0.0
        for x, y in tqdm(train_loader, desc=f"epoch {epoch}"):
            # snapshot BEFORE the step so step 0 captures the init (random) model
            if should_snapshot(global_step):
                acc = evaluate(model, test_loader, device)
                save_snapshot(model, args.ckpt_dir, global_step, epoch, acc)

            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            scheduler.step()
            running += loss.item()
            global_step += 1

        acc = evaluate(model, test_loader, device)
        print(f"epoch {epoch}: loss={running / len(train_loader):.4f} test_acc={acc:.4f}")

    # always snapshot the final model
    acc = evaluate(model, test_loader, device)
    save_snapshot(model, args.ckpt_dir, global_step, args.epochs, acc)


if __name__ == "__main__":
    main()
