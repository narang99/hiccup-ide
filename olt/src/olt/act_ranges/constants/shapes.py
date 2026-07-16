# Grid shape to reshape a flattened per-branch weight*patch vector back into for
# visualization (see report_assets.py, feature_viz.py). Only layers analysed so far
# have an entry; add one here when adding support for a new current_layer/dep_layer.
#
# This dict's keys also double as the default `all_layers` list passed into
# NeuronParentAnalyser (via `list(LAYER_NAME_BY_SHAPE.keys())` — see the notebook's
# write_cluster_report), so a current_layer_name and every one of its dependency
# layer names must both have an entry here even though the current_layer_name's own
# shape value is unused (it's never itself a dep_layer_name today).
LAYER_NAME_BY_SHAPE = {
    "mixed4d_1x1_pre_relu_conv": (16, 32),
    "mixed4d_3x3_pre_relu_conv": (36, 36),
    "mixed4d_pool_reduce_pre_relu_conv": (16, 32),
    "mixed4d_5x5_pre_relu_conv": (25, 32),
    "mixed4e_1x1_pre_relu_conv": (22, 24),
    "mixed5b_5x5_bottleneck_pre_relu_conv": (26, 32),
    "mixed5b_5x5_pre_relu_conv": (30, 40),
}
