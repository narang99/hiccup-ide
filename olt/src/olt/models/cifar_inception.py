"""A small, InceptionV1-shaped CNN for CIFAR-10.

This is the "clean custom arch" that the act_ranges / clustering pipeline was
originally built around lucent's `InceptionV1` for. It is deliberately much
smaller (4 Inception blocks, ~450k params, 32x32 input). Written as a plain,
explicit torch model: an `InceptionBlock` module with named submodules and a
straight-line forward, and the top model holding four blocks as named
attributes. No metaprogramming.

Layer names are the natural nested `get_submodule` paths, e.g.
`block_b.branch_1x1_pre_relu_conv` or `block_a.branch_5x5_pre_relu_conv`. The
analysis code (`olt.act` hooks, `get_layer_params`, FlattenedChannelMap) only
ever needs the caller to pass the full name, so nesting is fine — and it makes
these names naturally disjoint from lucent InceptionV1's flat `mixed*` /
`conv2d*` names, so any inception layer name/constant that got hardcoded into a
pipeline path fails loudly (get_submodule AttributeError / dict KeyError)
instead of silently resolving to a same-named but differently-shaped layer.

Two properties the analysis machinery relies on are kept:
1. Each pre-ReLU conv is its own named `nn.Conv2d` submodule, so a forward hook
   on it captures the conv's input, and `get_layer_params` reads
   `.weight/.kernel_size/.stride/.padding` off it.
2. Consecutive same-resolution blocks let a later block's 1x1 conv read the
   previous block's concatenated output DIRECTLY (no pad, no pool between) — the
   case `NeuronParentAnalyser` supports. `block_b.branch_1x1_pre_relu_conv`
   reads `block_a`'s output, and `block_d.branch_1x1_pre_relu_conv` reads
   `block_c`'s — the CIFAR analogues of InceptionV1's supported
   `mixed4e_1x1_pre_relu_conv` (which reads `mixed4d`).

The relu layers are reused from lucent's `helper_layers` so the redirected-ReLU
trick (needed for lucent feature visualization) can be swapped in exactly like
`InceptionV1`. Everything else is a real nn.Module (convs with folded `padding=`,
`nn.MaxPool2d`, `nn.AdaptiveAvgPool2d`) so a forward hook can capture it — the
only functional op left is `torch.cat`, and you don't need to hook it separately
since a block's output IS that concat (hook the block module).

Convs carry their own `padding=` rather than a manual `F.pad` in forward: the
forward is identical, but a hook on the conv then captures its true (unpadded)
input, and the receptive-field code (`layer_utils.receptive_block` /
`get_layer_params`) already reads `layer.padding` to account for it.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from lucent.modelzoo.inceptionv1 import helper_layers

NUM_CLASSES = 10


class InceptionBlock(nn.Module):
    """One Inception block: four branches (1x1, 3x3, 5x5, pool_reduce) run on
    the same input and concatenated in that order. Constructor args follow
    torchvision's GoogLeNet `Inception` convention:
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
        self.branch_1x1_pre_relu_conv = nn.Conv2d(in_channels, ch1x1, 1)
        self.branch_1x1 = relu_cls()

        # 3x3 branch: 1x1 bottleneck -> 3x3 conv (padding=1 keeps spatial size)
        self.branch_3x3_bottleneck_pre_relu_conv = nn.Conv2d(in_channels, ch3x3_reduce, 1)
        self.branch_3x3_bottleneck = relu_cls()
        self.branch_3x3_pre_relu_conv = nn.Conv2d(ch3x3_reduce, ch3x3, 3, padding=1)
        self.branch_3x3 = relu_cls()

        # 5x5 branch: 1x1 bottleneck -> 5x5 conv (padding=2 keeps spatial size)
        self.branch_5x5_bottleneck_pre_relu_conv = nn.Conv2d(in_channels, ch5x5_reduce, 1)
        self.branch_5x5_bottleneck = relu_cls()
        self.branch_5x5_pre_relu_conv = nn.Conv2d(ch5x5_reduce, ch5x5, 5, padding=2)
        self.branch_5x5 = relu_cls()

        # pool branch: 3x3/stride-1 maxpool (padding=1 keeps size) -> 1x1 reduce
        self.branch_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch_pool_reduce_pre_relu_conv = nn.Conv2d(in_channels, pool_proj, 1)
        self.branch_pool_reduce = relu_cls()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b1 = self.branch_1x1(self.branch_1x1_pre_relu_conv(x))

        b3 = self.branch_3x3_bottleneck(self.branch_3x3_bottleneck_pre_relu_conv(x))
        b3 = self.branch_3x3(self.branch_3x3_pre_relu_conv(b3))

        b5 = self.branch_5x5_bottleneck(self.branch_5x5_bottleneck_pre_relu_conv(x))
        b5 = self.branch_5x5(self.branch_5x5_pre_relu_conv(b5))

        p = self.branch_pool(x)
        p = self.branch_pool_reduce(self.branch_pool_reduce_pre_relu_conv(p))

        return torch.cat((b1, b3, b5, p), dim=1)


class CifarInception(nn.Module):
    """See module docstring. Set `redirected_relu=True` for feature
    visualization (lucent), `False` (default) for training and analysis.

    `stem_stride` sets the stem conv's stride and is the ONLY input-size knob:
    the classifier is a global avg-pool so any input size works, this just keeps
    the block grids sane for larger inputs. It does not affect the analysed 1x1
    blocks or any act_ranges constant (those are within-resolution and read
    `layer.padding`, not the absolute grid).

    Spatial trace, stem_stride=1 (CIFAR 32x32): stem conv keeps 32x32, stem
    maxpool -> 16x16; block_a/block_b at 16x16; downpool -> 8x8;
    block_c/block_d at 8x8; global avg pool -> 1x1.

    Spatial trace, stem_stride=2 (STL-10 96x96): stem conv -> 48x48, stem
    maxpool -> 24x24; block_a/block_b at 24x24; downpool -> 12x12;
    block_c/block_d at 12x12; global avg pool -> 1x1.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        redirected_relu: bool = False,
        stem_stride: int = 1,
    ):
        super().__init__()
        # Why the ReLU is a swappable module rather than an inline F.relu:
        #
        # `redirected_relu` picks between a normal ReLU (`ReluLayer`) and lucent's
        # `RedirectedReluLayer` — a ReLU with a NORMAL forward but a deliberately
        # FAKED backward (see lucent helper_layers.RedirectedReLU): where the input
        # was negative, instead of passing 0 gradient it leaks ~10% through.
        #
        # This matters only for FEATURE VISUALIZATION, which optimizes an input
        # image (from random noise) to maximize a neuron, with weights frozen. A
        # random init often drives the units on the path to the target negative
        # everywhere; a real ReLU then gives that image exactly 0 gradient, so
        # optimization is dead on arrival. The leaked gradient gives the optimizer
        # a direction to push those pre-activations back positive and escape the
        # dead zone. It is intentionally "wrong" (true gradient there is 0), so it
        # is wanted ONLY during feature-viz optimization.
        #
        # Training and act_ranges analysis need the true ReLU gradient (a fake one
        # would corrupt training and attributions), so they use redirected_relu=
        # False (the default). Because it's a parameter-free module, the same
        # trained weights load into either variant — build with redirected_relu=
        # True to get a feature-viz-ready copy, exactly like lucent's
        # InceptionV1(redirected_ReLU=...).
        relu_cls = (
            helper_layers.RedirectedReluLayer
            if redirected_relu
            else helper_layers.ReluLayer
        )

        # stem: 3x3 conv (padding=1, stride=stem_stride), then 3x3/stride-2 maxpool.
        # stem_stride=1 keeps 32x32 (CIFAR); stem_stride=2 -> 48x48 (STL-10 96x96).
        self.stem_pre_relu_conv = nn.Conv2d(
            3, 64, kernel_size=3, stride=stem_stride, padding=1
        )
        self.stem = relu_cls()
        self.stem_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # (in, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj)
        self.block_a = InceptionBlock(64, 32, 48, 64, 8, 16, 16, relu_cls)   # out 128
        self.block_b = InceptionBlock(128, 64, 64, 96, 16, 32, 32, relu_cls)  # out 224
        self.block_c = InceptionBlock(224, 96, 64, 128, 16, 32, 32, relu_cls)  # out 288
        self.block_d = InceptionBlock(288, 112, 72, 144, 16, 48, 48, relu_cls)  # out 352

        # downsample between the 16x16 blocks and the 8x8 blocks
        self.downpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(self.block_d.out_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem_pool(self.stem(self.stem_pre_relu_conv(x)))  # /4 (stem_stride*pool)

        x = self.block_a(x)
        x = self.block_b(x)

        x = self.downpool(x)  # halve resolution before block_c/d
        x = self.block_c(x)
        x = self.block_d(x)

        x = self.flatten(self.avgpool(x))
        return self.fc(x)


def cifar_inception(
    ckpt_path: str | None = None,
    redirected_relu: bool = False,
    map_location: str = "cpu",
    eval_mode: bool = True,
    stem_stride: int = 1,
) -> CifarInception:
    """Factory mirroring `lucent.modelzoo.inceptionv1`'s call style. Loads a
    training snapshot's weights when `ckpt_path` is given.

    Pass `redirected_relu=True` when using the model for lucent feature
    visualization; leave it False for training and act_ranges analysis. Use
    `stem_stride=2` for STL-10 (96x96), `stem_stride=1` (default) for CIFAR
    (32x32) — must match whatever the snapshot was trained with.
    """
    model = CifarInception(redirected_relu=redirected_relu, stem_stride=stem_stride)
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
# Cat order per block, as (full_layer_name, n_channels), matching the
# torch.cat((b1, b3, b5, p), 1) order in InceptionBlock.forward. This is what a
# FlattenedChannelMap (act_ranges/layer_utils.py) needs to map a flattened
# channel in a block's concatenated output back to (layer, channel). Kept as
# explicit literals; must stay in sync with the block channel widths above.
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
# same-resolution block directly — the analysable layers, analogous to
# InceptionV1's mixed4e_1x1_pre_relu_conv. block_c is NOT here: its input is
# downsample-maxpool'd, which the single-offset dep_coords logic can't translate
# through (same reason mixed4e_pool_reduce is unsupported).
ANALYSABLE_1X1_LAYERS: dict[str, tuple[str, tuple[int, int]]] = {
    "block_b.branch_1x1_pre_relu_conv": ("block_a", (0, 0)),
    "block_d.branch_1x1_pre_relu_conv": ("block_c", (0, 0)),
}
