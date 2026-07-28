"""Train the mini-Inception on STL-10 (96x96), saving snapshots during training.

STL-10 is the higher-resolution (96x96, 10-class) drop-in for CIFAR — sharper
images make the downstream feature-viz / patch reports legible. This mirrors
`train_cifar.py`; the differences are: the STL10 dataset (5000 labeled train /
8000 test images, `split=` API), STL transforms, and `stem_stride=2` so the
96x96 input downsamples to sane block grids (block_a/b at 24x24, block_c/d at
12x12). The model, checkpoint format, and act_ranges constants are identical.

This is the BatchNorm variant; see `train_stl_bnfree.py` for the norm-free net.
The shared loaders/eval/snapshot/loop live in `olt.stl_training`.

Each snapshot is written to <ckpt-dir>/step_<global_step>.pt as
    {"model": state_dict, "step": int, "epoch": int, "test_acc": float}
Load it with
    stl_inception(ckpt_path=..., stem_stride=2)   # stem_stride must match!

Usage:
    uv run scripts/train_stl.py --ckpt-dir data/stl/checkpoints --epochs 40 \
        --snapshot-every-steps 500

    # final model only (no intermediate snapshots), on Apple Silicon:
    uv run scripts/train_stl.py --ckpt-dir data/stl/checkpoints --epochs 40 \
        --snapshot-steps -1 --device mps
"""

import argparse

import torch
import torch.nn as nn

from olt.models.stl_inception import stl_inception
from olt.neuron_tracking import build_tracker
from olt.stl_training import (
    STEM_STRIDE,
    add_shared_args,
    make_loaders,
    make_snapshot_predicate,
    resolve_train_transform,
    train_loop,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    add_shared_args(p)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)  # Adam-appropriate
    args = p.parse_args()

    device = torch.device(args.device)
    train_transform = resolve_train_transform(args.fancy_aug)
    train_loader, test_loader = make_loaders(
        args.data_root, args.batch_size, args.num_workers, train_transform
    )

    model = stl_inception(
        redirected_relu=False, eval_mode=False, stem_stride=STEM_STRIDE
    ).to(device)
    model.train()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"StlInception (stem_stride={STEM_STRIDE}): {n_params:,} params, device={device}")

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

    tracker = build_tracker(
        args.track_neuron, args.track_metric, args.track_out, args.ckpt_dir, model
    )

    train_loop(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        epochs=args.epochs,
        ckpt_dir=args.ckpt_dir,
        should_snapshot=make_snapshot_predicate(
            args.snapshot_every_steps, args.snapshot_steps
        ),
        tracker=tracker,
    )


if __name__ == "__main__":
    main()
