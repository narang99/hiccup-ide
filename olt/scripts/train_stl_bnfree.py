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

    # track how specific neurons drift each gradient step:
    uv run scripts/train_stl_bnfree.py --ckpt-dir data/stl-bnfree/checkpoints \
        --track-neuron block_c.branch_1x1_pre_relu_conv:0
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
    add_shared_args(p)
    p.add_argument("--epochs", type=int, default=80)  # norm-free converges slower
    p.add_argument("--lr", type=float, default=3e-4)  # lower max LR than the BN run
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--grad-clip", type=float, default=1.0, help="max grad norm; <=0 disables")
    args = p.parse_args()

    device = torch.device(args.device)
    train_transform = resolve_train_transform(args.fancy_aug)
    train_loader, test_loader = make_loaders(
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
        snapshot_extra={"use_bn": False},  # marker: load with stl_inception(..., use_bn=False)
        grad_clip=args.grad_clip,
        tracker=tracker,
    )


if __name__ == "__main__":
    main()
