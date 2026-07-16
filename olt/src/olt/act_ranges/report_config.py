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
    which stay as direct function args.

    cluster_notes: dict[(dep_layer_name, dep_channel, dep_cid), str] —
    free-text notes for specific dependency clusters, or None to skip both
    note sections entirely. Rendered twice: collected into one collapsible
    callout at the top of the report (report_render.render_notes_summary, so
    every note is visible without hunting through cards), and again inside
    that cluster's own "show cluster photo" <details> section in its card
    (report_render.render_cluster_breakdown), as the first thing shown
    there, in the same callout-note style — so a note is easy to spot both
    from a single skim at the top and in context when you land on that
    cluster.

    sort_order: which order the "frequently firing" cards are displayed in
    (report_stats.sort_card_order) - "median" (default: descending |median
    contribution|, i.e. the order compute_dep_order already produces) or
    "firing_frequency" (descending firing count - see
    report_stats.compute_firing_stats - with "median" order as the
    tie-breaker). Only affects card display order; the concentration
    sparkline's own ranking (report_stats.compute_concentration_curves) is
    always by descending |value| regardless of this setting.

    fraction_above_threshold: noise-median distance (noise-radius units,
    signed - e.g. -0.8 for a cutoff a bit below noise_med) used by
    report_stats.compute_fraction_above_threshold_by_example to build the
    per-origin-example "what fraction of this example's firings ran hot"
    histogram (report_assets.dump_fraction_above_threshold_histogram_asset,
    report_render.render_report_histograms) — grouped by (input_image_key,
    origin_y, origin_x), not by image alone, since one image can be probed
    at multiple positions. None (the default) skips this section of the
    report entirely.
    """

    max_points_per_cluster: int = 50
    outlier_ratio_threshold: float = 0.1
    concentration_metric: str = "median"  # or "weighted" — see report_stats.compute_concentration_values
    sort_order: str = "median"  # or "firing_frequency" — see report_stats.sort_card_order
    pw_samples: Optional[PwSamplesConfig] = None
    feature_viz: Optional[FeatureVizConfig] = None
    cluster_notes: Optional[dict] = None
    histogram_bins: int = 40  # bin count for the report-level output-activation-vs-noise histogram, see report_assets.dump_output_activation_histogram_asset
    fraction_above_threshold: Optional[float] = None  # noise-radius-unit threshold, or None to skip — see report_stats.compute_fraction_above_threshold_by_example

    def __post_init__(self):
        if self.feature_viz is not None and self.pw_samples is None:
            raise ValueError(
                "ReportConfig.feature_viz is set but pw_samples is None — feature-viz "
                "reconstructs each matched cluster's wild pw samples, so it has nothing "
                "to reconstruct without a PwSamplesConfig."
            )
