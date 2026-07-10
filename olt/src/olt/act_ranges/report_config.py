from dataclasses import dataclass
from typing import Optional


@dataclass
class PwSamplesConfig:
    """Wild pointwise-multiplication sample grid per matched cluster
    (report_assets.dump_pw_sample_asset). Set ReportConfig.pw_samples=None to
    disable this section — also disables feature_viz, since it draws its
    clusters from the same samples.

    layer_by_channel_by_cid_by_pw_samples: dict[dep_layer, dict[str(dep_channel),
    dict[str(dep_cid), list[{"dep_patch", "dep_w", "image_key", "dep_y",
    "dep_x"}]]]] — accumulated across NeuronParentAnalyser.collect_cluster_stats_df
    calls for every input image (see pw_samples.merge_pw_samples_into).
    """
    layer_by_channel_by_cid_by_pw_samples: dict
    max_samples: int = 5


@dataclass
class FeatureVizConfig:
    """Feature-viz reconstruction of each matched cluster's wild pw samples
    (feature_viz.dump_feature_viz_assets), added as an extra row alongside
    PwSamplesConfig's wild/match grid — one image per wild sample (up to
    PwSamplesConfig.max_samples), not jointly optimized. Runs on whatever
    device `model` is already on (CPU by default) with modest defaults, since
    it's one render_vis call per sample. Set ReportConfig.feature_viz=None to
    disable. Raises if set while ReportConfig.pw_samples is None — feature-viz
    has no samples to reconstruct without it.

    neurons: optional set of (dep_layer_name, dep_channel) pairs to restrict
    feature-viz to (e.g. {("mixed4d_3x3_pre_relu_conv", 12)}). render_vis is
    expensive (one gradient-based optimization per wild sample per matched
    cluster), so on a report with many dependency neurons this lets you scope
    a run to just the ones you're currently looking at instead of paying for
    all of them. None (the default) runs feature-viz for every matched,
    non-outlier cluster.
    """
    model: object
    image_size: int = 64
    thresholds: tuple = (128,)
    neurons: Optional[frozenset] = None


@dataclass
class ReportConfig:
    """Tunables and feature toggles for print_report_for_neuron, as opposed to
    the report's actual data (stats_df, noise/cluster dicts, output paths)
    which stay as direct function args."""

    max_points_per_cluster: int = 50
    outlier_ratio_threshold: float = 0.1
    relative_strength_method: str = "median_sum"  # or "median_per_image_share"
    pw_samples: Optional[PwSamplesConfig] = None
    feature_viz: Optional[FeatureVizConfig] = None

    def __post_init__(self):
        if self.feature_viz is not None and self.pw_samples is None:
            raise ValueError(
                "ReportConfig.feature_viz is set but pw_samples is None — feature-viz "
                "reconstructs each matched cluster's wild pw samples, so it has nothing "
                "to reconstruct without a PwSamplesConfig."
            )
