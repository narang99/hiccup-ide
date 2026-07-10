from olt.act_ranges.analyser import NeuronParentAnalyser
from olt.act_ranges.constants import (
    CLUSTER_PATCH_SET_DIR,
    F_PAD_MANUAL_BY_CURRENT_LAYER,
    KELLY_COLORS,
    LAYER_NAME_BY_SHAPE,
    MIXED4D_BRANCHES,
    UNSUPPORTED_CURRENT_LAYERS,
)
from olt.act_ranges.dependency_match import DependencyMatch
from olt.act_ranges.layer_utils import (
    FlattenedChannelMap,
    get_layer_params,
    receptive_block,
)
from olt.act_ranges.plotting import (
    save_combined_scatter_jpeg,
    save_concentration_sparkline_jpeg,
)
from olt.act_ranges.pw_samples import merge_pw_samples_into
from olt.act_ranges.quarto_report import print_report_for_neuron
from olt.act_ranges.report_assets import (
    dump_cluster_asset,
    dump_concentration_asset,
    dump_overview_assets,
)
from olt.act_ranges.report_config import FeatureVizConfig, PwSamplesConfig, ReportConfig
from olt.act_ranges.report_render import (
    render_neuron_card,
    render_notes_summary,
    render_origin_cluster_header,
    render_overview_tab_body,
    render_summary,
)
from olt.act_ranges.report_stats import (
    check_at_most_one_firing_per_origin,
    compute_concentration_curves,
    compute_concentration_values,
    compute_dep_order,
    compute_firing_stats,
    split_dep_order_by_frequency,
)
from olt.act_ranges.reports import crop_top, get_cluster_photo
from olt.act_ranges.similarity import (
    closest_patch_index,
    closest_pw,
    get_neuron_closest_cluster,
    max_cosine_similarity,
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
    "max_cosine_similarity",
    "min_euclidean_distance",
    "closest_pw",
    "closest_patch_index",
    "get_neuron_closest_cluster",
    "get_cluster_photo",
    "crop_top",
    "dump_cluster_asset",
    "dump_overview_assets",
    "dump_concentration_asset",
    "compute_dep_order",
    "compute_firing_stats",
    "compute_concentration_curves",
    "compute_concentration_values",
    "check_at_most_one_firing_per_origin",
    "split_dep_order_by_frequency",
    "render_overview_tab_body",
    "render_neuron_card",
    "render_summary",
    "render_notes_summary",
    "render_origin_cluster_header",
    "save_combined_scatter_jpeg",
    "save_concentration_sparkline_jpeg",
    "print_report_for_neuron",
    "NeuronParentAnalyser",
    "DependencyMatch",
    "merge_pw_samples_into",
    "ReportConfig",
    "PwSamplesConfig",
    "FeatureVizConfig",
]
