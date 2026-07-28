"""Train the NORM-FREE mini-Inception on STL-10 (96x96) — Route B schedule.

Companion to `train_stl.py` (the BN version). Builds `stl_inception(use_bn=False)`
and trains it with the recipe that replaces each job BatchNorm was doing, so a
from-scratch un-normalized net still reaches good accuracy:

  * Kaiming init         -- correct activation scale at step 0 (in the model's
                            _init_weights; BN would otherwise re-standardize away
                            any init mistake every layer).
  * LR warmup            -- ramp LR up over the first --warmup-steps so a big
                            early step can't fling the random net into a dead
                            region it can't recover from without BN.
  * cosine decay         -- high LR early (explore / regularize), annealed low
                            late (settle into the minimum).
  * lower max LR + AdamW -- norm-free nets have a narrower stable-LR band; the
                            decoupled weight decay keeps weight/activation scale
                            from drifting mid-training (and regularizes).
  * gradient clipping    -- hard rail so one large-gradient batch can't blow the
                            activations out (BN's implicit gradient control).
  * more epochs          -- norm-free converges slower; budget for it.
  * crop+flip aug        -- replaces the mild regularization BN's batch noise gave.

The resulting checkpoint has conv biases and NO bn/running-stat keys, so load it
with `stl_inception(ckpt_path=..., use_bn=False, stem_stride=2)`. Its conv output
IS the true pre-ReLU activation (no BN to hook past or fold) — the whole reason
for going norm-free.

Usage:
    uv run scripts/train_stl_bnfree.py --ckpt-dir data/stl-bnfree/checkpoints \
        --epochs 80 --device mps
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import STL10
from tqdm import tqdm

from olt.models.stl_inception import stl_inception
from olt.tfms import stl_fancy_train_transform, stl_train_transform, stl_transform

STEM_STRIDE = 2  # 96x96 input; must match what stl_inception(ckpt_path=...) loads with


def _make_loaders(data_root: Path, batch_size: int, num_workers: int, train_transform):
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


def save_snapshot(model, ckpt_dir: Path, step: int, epoch: int, test_acc: float):
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"step_{step:06d}.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "step": step,
            "epoch": epoch,
            "test_acc": test_acc,
            "use_bn": False,  # marker: load with stl_inception(..., use_bn=False)
        },
        path,
    )
    print(f"  snapshot -> {path} (step={step}, epoch={epoch}, test_acc={test_acc:.4f})")


def _make_scheduler(optimizer, total_steps: int, warmup_steps: int):
    """Linear warmup for `warmup_steps`, then cosine decay over the rest."""
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warmup_steps
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - warmup_steps)
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[warmup_steps]
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt-dir", type=Path, required=True)
    p.add_argument("--data-root", type=Path, default=Path("data/stl-raw"))
    p.add_argument("--epochs", type=int, default=80)  # norm-free converges slower
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)  # lower max LR than the BN run
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--grad-clip", type=float, default=1.0, help="max grad norm; <=0 disables")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument(
        "--fancy-aug",
        action="store_true",
        help="use stl_fancy_train_transform (random-mode pad crop + color jitter "
        "+ random grayscale) to fight black-border/color shortcuts",
    )
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

    train_transform = stl_fancy_train_transform if args.fancy_aug else stl_train_transform
    print(f"train transform: {'fancy' if args.fancy_aug else 'plain'}")
    train_loader, test_loader = _make_loaders(
        args.data_root, args.batch_size, args.num_workers, train_transform
    )

    # use_bn=False: the whole point of this script (Kaiming init lives in the model)
    model = stl_inception(
        redirected_relu=False, eval_mode=False, stem_stride=STEM_STRIDE, use_bn=False
    ).to(device)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"StlInception norm-free (stem_stride={STEM_STRIDE}): {n_params:,} params, device={device}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    total_steps = args.epochs * len(train_loader)
    scheduler = _make_scheduler(optimizer, total_steps, args.warmup_steps)

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
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
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
