# Grid shape to reshape a flattened per-branch weight*patch vector back into for
# visualization (see plot_clusters in analyser.py). Only layers analysed so far
# have an entry; add one here when adding support for a new current_layer/dep_layer.
LAYER_NAME_BY_SHAPE = {
    "mixed4d_1x1_pre_relu_conv": (16, 32),
    "mixed4d_3x3_pre_relu_conv": (36, 36),
    "mixed4d_pool_reduce_pre_relu_conv": (16, 32),
    "mixed4d_5x5_pre_relu_conv": (25, 32),
    "mixed4e_1x1_pre_relu_conv": (22, 24),
}
