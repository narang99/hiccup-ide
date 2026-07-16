# f_pad_manual_before_between_us_and_dep reference values, keyed by current_layer_name.
#
# Background: in lucent's InceptionV1, a dependency block's branches are conv'd (+relu),
# concatenated, and then some paths apply a manual F.pad directly on the concatenated
# tensor before current_layer's own conv. NeuronParentAnalyser.dep_coords needs that pad
# amount to translate a receptive-field coordinate in current_layer's (post-pad) captured
# "input" tensor back to the dependency's raw (pre-pad) output coordinate.
#
# When adding support for a new current_layer, trace its forward() path back to the
# dependency block's concat and record the manual F.pad amount here (or (0, 0) if the
# conv reads the concatenated tensor directly, e.g. a bare 1x1 conv with no pad).
F_PAD_MANUAL_BY_CURRENT_LAYER = {
    # mixed4e_1x1_pre_relu_conv convolves `mixed4d` directly (1x1 kernel, no F.pad).
    "mixed4e_1x1_pre_relu_conv": (0, 0),
    # mixed5b_5x5_pre_relu_conv convolves mixed5b_5x5_bottleneck_pre_relu_conv (relu'd)
    # through exactly one manual F.pad((2, 2, 2, 2)) before the 5x5 conv.
    "mixed5b_5x5_pre_relu_conv": (2, 2),
}

# current_layer_names known NOT to be safely handled by NeuronParentAnalyser's current
# coordinate-mapping logic (single-scalar pad cut via dep_coords), because there's an
# extra hop between the dependency's manual F.pad and current_layer's own conv that
# isn't accounted for.
#
# mixed4e_pool_reduce_pre_relu_conv: mixed4d is F.pad'd, then passed through a 3x3
# maxpool (stride 1), and only then convolved. The maxpool has its own receptive field
# on top of the pad, which dep_coords does not currently translate through.
UNSUPPORTED_CURRENT_LAYERS = {
    "mixed4e_pool_reduce_pre_relu_conv",
}
