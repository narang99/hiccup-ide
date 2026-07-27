"""A small, InceptionV1-shaped CNN for STL-10 (96x96), WITH BatchNorm.

Same architecture family as `cifar_inception.CifarInception`, but two things
differ, both driven by training STL-10 from scratch:

1. **BatchNorm.** Each `*_pre_relu_conv` is followed by `nn.BatchNorm2d` and then
   the ReLU: the path is `conv -> bn -> relu`. A from-scratch, un-normalized
   Inception-shaped net underfits badly (CIFAR's BN-free variant plateaued ~46%
   on STL); BN fixes the optimization. Convs use `bias=False` because BN's affine
   subsumes the bias.

2. **stem_stride=2 by default.** STL inputs are 96x96, so the stem conv strides by
   2 (96 -> 48), then the stem maxpool halves again (-> 24). block_a/block_b run
   at 24x24, downpool -> 12x12, block_c/block_d at 12x12, global avg pool -> 1x1.

Layer names match the CIFAR model's nested `get_submodule` paths
(`block_b.branch_1x1_pre_relu_conv`, ...), so the act_ranges constants and the
FlattenedChannelMap demux carry over unchanged (`BLOCK_BRANCHES` /
`ANALYSABLE_1X1_LAYERS` below are byte-identical to the CIFAR ones — the channel
widths are the same). They stay disjoint from lucent InceptionV1's flat
`mixed*` / `conv2d*` names, so any hardcoded inception constant still fails loudly.

**Interp note — BN and the analyser.** The conv named `*_pre_relu_conv` is NO
LONGER the pre-ReLU activation: the true pre-ReLU tensor is the BN output
(`*_bn`). At eval BN is a fixed per-channel affine `act = (γ/σ)(W·patch + b − μ) + β`,
so:
  - ranking which upstream channels explain a neuron is UNCHANGED (γ/σ is a
    single per-output-channel scalar; it doesn't reorder input-channel
    contributions), so the parent/cluster-finding flow is safe; but
  - any absolute-activation reader (s1 threshold, activation ranges, report
    weight·patch magnitudes) must either HOOK the `*_bn` output or FOLD BN into
    effective (W', b') = ((γ/σ)W, (γ/σ)(b − μ) + β). That fold is the remaining
    work when wiring the analysis pass for this model.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from lucent.modelzoo.inceptionv1 import helper_layers

NUM_CLASSES = 10


class StlInceptionBlock(nn.Module):
    """One Inception block with BatchNorm: four branches (1x1, 3x3, 5x5,
    pool_reduce) run on the same input and concatenated in that order. Each conv
    path is `conv -> bn -> relu`. Constructor args follow torchvision's GoogLeNet
    `Inception` convention:
    (in_channels, ch1x1, ch3x3_reduce, ch3x3, ch5x5_reduce, ch5x5, pool_proj).
    """

    def __init__(
        self,
        in_channels: int,
        ch1x1: int,
        ch3x3_reduce: int,
        ch3x3: int,
        ch5x5_reduce: int,
        ch5x5: int,
        pool_proj: int,
        relu_cls: type[nn.Module],
    ):
        super().__init__()
        self.out_channels = ch1x1 + ch3x3 + ch5x5 + pool_proj

        # 1x1 branch
        self.branch_1x1_pre_relu_conv = nn.Conv2d(in_channels, ch1x1, 1, bias=False)
        self.branch_1x1_bn = nn.BatchNorm2d(ch1x1)
        self.branch_1x1 = relu_cls()

        # 3x3 branch: 1x1 bottleneck -> 3x3 conv (padding=1 keeps spatial size)
        self.branch_3x3_bottleneck_pre_relu_conv = nn.Conv2d(
            in_channels, ch3x3_reduce, 1, bias=False
        )
        self.branch_3x3_bottleneck_bn = nn.BatchNorm2d(ch3x3_reduce)
        self.branch_3x3_bottleneck = relu_cls()
        self.branch_3x3_pre_relu_conv = nn.Conv2d(
            ch3x3_reduce, ch3x3, 3, padding=1, bias=False
        )
        self.branch_3x3_bn = nn.BatchNorm2d(ch3x3)
        self.branch_3x3 = relu_cls()

        # 5x5 branch: 1x1 bottleneck -> 5x5 conv (padding=2 keeps spatial size)
        self.branch_5x5_bottleneck_pre_relu_conv = nn.Conv2d(
            in_channels, ch5x5_reduce, 1, bias=False
        )
        self.branch_5x5_bottleneck_bn = nn.BatchNorm2d(ch5x5_reduce)
        self.branch_5x5_bottleneck = relu_cls()
        self.branch_5x5_pre_relu_conv = nn.Conv2d(
            ch5x5_reduce, ch5x5, 5, padding=2, bias=False
        )
        self.branch_5x5_bn = nn.BatchNorm2d(ch5x5)
        self.branch_5x5 = relu_cls()

        # pool branch: 3x3/stride-1 maxpool (padding=1 keeps size) -> 1x1 reduce
        self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch_pool_reduce_pre_relu_conv = nn.Conv2d(
            in_channels, pool_proj, 1, bias=False
        )
        self.branch_pool_reduce_bn = nn.BatchNorm2d(pool_proj)
        self.branch_pool_reduce = relu_cls()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch_1x1(self.branch_1x1_bn(self.branch_1x1_pre_relu_conv(x)))

        b3 = self.branch_3x3_bottleneck(
            self.branch_3x3_bottleneck_bn(self.branch_3x3_bottleneck_pre_relu_conv(x))
        )
        b3 = self.branch_3x3(self.branch_3x3_bn(self.branch_3x3_pre_relu_conv(b3)))

        b5 = self.branch_5x5_bottleneck(
            self.branch_5x5_bottleneck_bn(self.branch_5x5_bottleneck_pre_relu_conv(x))
        )
        b5 = self.branch_5x5(self.branch_5x5_bn(self.branch_5x5_pre_relu_conv(b5)))

        p = self.branch_pool(x)
        p = self.branch_pool_reduce(
            self.branch_pool_reduce_bn(self.branch_pool_reduce_pre_relu_conv(p))
        )

        return torch.cat((b1, b3, b5, p), dim=1)


class StlInception(nn.Module):
    """See module docstring. Set `redirected_relu=True` for feature
    visualization (lucent), `False` (default) for training and analysis.

    Spatial trace (stem_stride=2, STL 96x96): stem conv -> 48x48, stem maxpool
    -> 24x24; block_a/block_b at 24x24; downpool -> 12x12; block_c/block_d at
    12x12; global avg pool -> 1x1.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        redirected_relu: bool = False,
        stem_stride: int = 2,
    ):
        super().__init__()
        # See cifar_inception for the full redirected-ReLU rationale: a normal
        # forward but a faked backward that leaks ~10% gradient through the
        # negative region, wanted ONLY for lucent feature-viz optimization.
        # Training and act_ranges analysis use redirected_relu=False.
        relu_cls = (
            helper_layers.RedirectedReluLayer
            if redirected_relu
            else helper_layers.ReluLayer
        )

        # stem: 3x3 conv (stride=stem_stride, padding=1) -> bn -> relu, then
        # 3x3/stride-2 maxpool. stem_stride=2 -> 96x96 conv to 48x48, pool to 24x24.
        self.stem_pre_relu_conv = nn.Conv2d(
            3, 64, kernel_size=3, stride=stem_stride, padding=1, bias=False
        )
        self.stem_bn = nn.BatchNorm2d(64)
        self.stem = relu_cls()
        self.stem_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # (in, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj)
        self.block_a = StlInceptionBlock(64, 32, 48, 64, 8, 16, 16, relu_cls)   # out 128
        self.block_b = StlInceptionBlock(128, 64, 64, 96, 16, 32, 32, relu_cls)  # out 224
        self.block_c = StlInceptionBlock(224, 96, 64, 128, 16, 32, 32, relu_cls)  # out 288
        self.block_d = StlInceptionBlock(288, 112, 72, 144, 16, 48, 48, relu_cls)  # out 352

        # downsample between the block_a/b resolution and the block_c/d resolution
        self.downpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(self.block_d.out_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem_pool(self.stem(self.stem_bn(self.stem_pre_relu_conv(x))))

        x = self.block_a(x)
        x = self.block_b(x)

        x = self.downpool(x)  # halve resolution before block_c/d
        x = self.block_c(x)
        x = self.block_d(x)

        x = self.flatten(self.avgpool(x))
        return self.fc(x)


def stl_inception(
    ckpt_path: str | None = None,
    redirected_relu: bool = False,
    map_location: str = "cpu",
    eval_mode: bool = True,
    stem_stride: int = 2,
) -> StlInception:
    """Factory mirroring `lucent.modelzoo.inceptionv1`'s call style. Loads a
    training snapshot's weights when `ckpt_path` is given.

    Pass `redirected_relu=True` when using the model for lucent feature
    visualization; leave it False for training and act_ranges analysis.
    `stem_stride` must match whatever the snapshot was trained with (2 for STL).
    """
    model = StlInception(redirected_relu=redirected_relu, stem_stride=stem_stride)
    if ckpt_path is not None:
        state = torch.load(ckpt_path, map_location=map_location)
        # accept either a bare state_dict or a {"model": state_dict, ...} snapshot
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state)
    if eval_mode:
        model.eval()
    return model


# --- act_ranges metadata (used by the not-yet-wired analysis pass) -----------
# Byte-identical to cifar_inception's tables: the channel widths and cat order
# are the same, and the layer names match. Kept here (not imported) so this
# model file is self-contained. Cat order per block, as (full_layer_name,
# n_channels), matching torch.cat((b1, b3, b5, p), 1) in StlInceptionBlock.forward.
BLOCK_BRANCHES: dict[str, list[tuple[str, int]]] = {
    "block_a": [
        ("block_a.branch_1x1_pre_relu_conv", 32),
        ("block_a.branch_3x3_pre_relu_conv", 64),
        ("block_a.branch_5x5_pre_relu_conv", 16),
        ("block_a.branch_pool_reduce_pre_relu_conv", 16),
    ],
    "block_b": [
        ("block_b.branch_1x1_pre_relu_conv", 64),
        ("block_b.branch_3x3_pre_relu_conv", 96),
        ("block_b.branch_5x5_pre_relu_conv", 32),
        ("block_b.branch_pool_reduce_pre_relu_conv", 32),
    ],
    "block_c": [
        ("block_c.branch_1x1_pre_relu_conv", 96),
        ("block_c.branch_3x3_pre_relu_conv", 128),
        ("block_c.branch_5x5_pre_relu_conv", 32),
        ("block_c.branch_pool_reduce_pre_relu_conv", 32),
    ],
    "block_d": [
        ("block_d.branch_1x1_pre_relu_conv", 112),
        ("block_d.branch_3x3_pre_relu_conv", 144),
        ("block_d.branch_5x5_pre_relu_conv", 48),
        ("block_d.branch_pool_reduce_pre_relu_conv", 48),
    ],
}

# current_layer_name -> (dependency_block, F.pad between the dep block's concat
# and this conv). Only the 1x1 branch of a block that reads the previous
# same-resolution block directly. block_c is NOT here: its input is
# downsample-maxpool'd, which the single-offset dep_coords logic can't translate.
ANALYSABLE_1X1_LAYERS: dict[str, tuple[str, tuple[int, int]]] = {
    "block_b.branch_1x1_pre_relu_conv": ("block_a", (0, 0)),
    "block_d.branch_1x1_pre_relu_conv": ("block_c", (0, 0)),
}
