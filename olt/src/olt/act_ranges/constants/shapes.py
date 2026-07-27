# Grid shape to reshape a flattened per-branch weight*patch vector back into for
# visualization (see report_assets.py, feature_viz.py). Only layers analysed so far
# have an entry; add one here when adding support for a new current_layer/dep_layer.
#
# NOTE: this is purely a reshape-for-viz lookup, keyed by dep_layer_name. Do NOT
# use `list(LAYER_NAME_BY_SHAPE.keys())` as the analyser's `all_layers` — that
# lumps every analysed layer of every model together (InceptionV1 `mixed*` and the
# CIFAR model's `cifar_*`), so it would try to hook the wrong model's submodules.
# Use constants.layers_to_hook(current_layer_name) instead (branches.py), which
# derives the exact {current + dep-branch} layers to hook for one model.
LAYER_NAME_BY_SHAPE = {
    "mixed4d_1x1_pre_relu_conv": (16, 32),
    "mixed4d_3x3_pre_relu_conv": (36, 36),
    "mixed4d_pool_reduce_pre_relu_conv": (16, 32),
    "mixed4d_5x5_pre_relu_conv": (25, 32),
    "mixed4e_1x1_pre_relu_conv": (22, 24),
    "mixed5b_5x5_bottleneck_pre_relu_conv": (26, 32),
    "mixed5b_5x5_pre_relu_conv": (30, 40),
}
