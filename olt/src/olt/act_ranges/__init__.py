from olt.act_ranges.analyser import DependencyMatch, NeuronParentAnalyser
from olt.act_ranges.constants import (
    CLUSTER_PATCH_SET_DIR,
    F_PAD_MANUAL_BY_CURRENT_LAYER,
    KELLY_COLORS,
    LAYER_NAME_BY_SHAPE,
    MIXED4D_BRANCHES,
    UNSUPPORTED_CURRENT_LAYERS,
)
from olt.act_ranges.filters import NoiseRatioRangeFilter
from olt.act_ranges.layer_utils import (
    FlattenedChannelMap,
    get_layer_params,
    receptive_block,
)
from olt.act_ranges.plotting import save_combined_scatter_png
from olt.act_ranges.quarto_report import (
    check_at_most_one_firing_per_origin,
    check_same_sign,
    compute_dep_order,
    compute_firing_stats,
    compute_image_act_sums,
    dedupe_to_one_origin_per_image,
    dump_cluster_asset,
    dump_overview_assets,
    print_report_for_neuron,
    render_activation_bar,
    render_image_tab_body,
    render_image_tab_placeholder_body,
    render_neuron_tabset_card,
    render_overview_tab_body,
    split_dep_order_by_frequency,
)
from olt.act_ranges.reports import crop_top, get_cluster_photo
from olt.act_ranges.similarity import (
    closest_pw,
    get_neuron_closest_cluster,
    mean_cosine_similarity,
    min_euclidean_distance,
)
from olt.act_ranges.stats import (
    get_labels_above_noise_range,
    get_noise_range,
    indices_for_percentage,
    shorth,
)

__all__ = [
    "CLUSTER_PATCH_SET_DIR",
    "LAYER_NAME_BY_SHAPE",
    "KELLY_COLORS",
    "MIXED4D_BRANCHES",
    "F_PAD_MANUAL_BY_CURRENT_LAYER",
    "UNSUPPORTED_CURRENT_LAYERS",
    "get_layer_params",
    "receptive_block",
    "FlattenedChannelMap",
    "indices_for_percentage",
    "shorth",
    "get_noise_range",
    "get_labels_above_noise_range",
    "mean_cosine_similarity",
    "min_euclidean_distance",
    "closest_pw",
    "get_neuron_closest_cluster",
    "get_cluster_photo",
    "crop_top",
    "dump_cluster_asset",
    "dump_overview_assets",
    "render_activation_bar",
    "compute_dep_order",
    "compute_firing_stats",
    "compute_image_act_sums",
    "check_at_most_one_firing_per_origin",
    "split_dep_order_by_frequency",
    "check_same_sign",
    "dedupe_to_one_origin_per_image",
    "render_overview_tab_body",
    "render_image_tab_body",
    "render_image_tab_placeholder_body",
    "render_neuron_tabset_card",
    "save_combined_scatter_png",
    "print_report_for_neuron",
    "NoiseRatioRangeFilter",
    "NeuronParentAnalyser",
    "DependencyMatch",
]
