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
