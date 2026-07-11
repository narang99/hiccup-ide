# Channel layout of the mixed4d Inception block's 4 branches, in the order they
# are torch.cat'd together in lucent's InceptionV1.forward (see
# mixed4d = self.mixed4d((mixed4d_1x1, mixed4d_3x3, mixed4d_5x5, mixed4d_pool_reduce), 1)).
# get_layer_and_chan_from_flattened (layer_utils.py) walks this list to map a flattened
# channel index in the 528-channel concatenated `mixed4d` tensor back to the
# (layer_name, channel) that produced it.
#
# When adding support for a new current_layer whose dependency is a different
# concatenated block, add a new ordered branch list here (mirroring the cat() order
# in the model's forward()) rather than hardcoding channel boundaries inline.
MIXED4D_BRANCHES = [
    ("mixed4d_1x1_pre_relu_conv", 112),
    ("mixed4d_3x3_pre_relu_conv", 288),
    ("mixed4d_5x5_pre_relu_conv", 64),
    ("mixed4d_pool_reduce_pre_relu_conv", 64),
]

# mixed5b_5x5_pre_relu_conv's dependency is not a concatenated multi-branch tensor —
# it's a single upstream layer, mixed5b_5x5_bottleneck_pre_relu_conv (48 channels),
# reached through exactly one manual F.pad (see constants/paddings.py). A single-entry
# list here lets FlattenedChannelMap degenerate to an identity channel mapping instead
# of needing separate handling for the concat vs. non-concat case.
MIXED5B_5X5_DEP_BRANCHES = [
    ("mixed5b_5x5_bottleneck_pre_relu_conv", 48),
]

# current_layer_name -> its dependency's branch list (see FlattenedChannelMap in
# layer_utils.py), one entry per current_layer_name NeuronParentAnalyser supports
# (mirrors F_PAD_MANUAL_BY_CURRENT_LAYER in constants/paddings.py) — lets callers look
# up the right branch list by current_layer_name instead of hardcoding which constant
# goes with which layer.
BRANCHES_BY_CURRENT_LAYER = {
    "mixed4e_1x1_pre_relu_conv": MIXED4D_BRANCHES,
    "mixed5b_5x5_pre_relu_conv": MIXED5B_5X5_DEP_BRANCHES,
}
