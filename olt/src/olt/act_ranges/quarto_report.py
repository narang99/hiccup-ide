from pathlib import Path

from tqdm import tqdm

from olt.act_ranges.report_card import CardConfig, render_neuron_card
from olt.act_ranges.report_data import build_report_data


def _render_section(title, keys, data, config, desc):
    if not keys:
        return None
    cards = "\n\n".join(render_neuron_card(*key, data, config) for key in tqdm(keys, desc=desc))
    return f"## {title}\n\n{cards}"


def print_report_for_neuron(
    stats_df,
    base_report_dir,
    assets_dump_dir,
    assets_ref_dir,
    layer_by_channel_by_noise,
    layer_by_channel_by_label_by_points,
    config,
    output_path=None,
):
    """
    Prints the full Quarto markdown to stdout — copy it into a .qmd file to
    render/test. If output_path is given, also writes it there.

    The report is one card per dependency neuron (see report_stats.compute_dep_order
    for ordering), split into "Frequently firing" and "One-off / low frequency"
    sections (see report_stats.split_dep_order_by_frequency). Each card
    (report_render.render_neuron_card) holds just the Overview content:
    median-contribution bar + combined activation/noise scatter plot + per-cluster
    firing breakdown, aggregated across every row of stats_df for that dep
    neuron. No per-image tabs/panel-tabset.

    stats_df: the concatenation of NeuronParentAnalyser.collect_cluster_stats_df
    outputs across multiple input images, for a single neuron — must have exactly
    one (origin_layer, origin_channel) pair and an "input_image_key" column.

    layer_by_channel_by_noise: same dict passed into NeuronParentAnalyser, used
    for the Overview tab's raw noise-sample scatter plots.
    layer_by_channel_by_label_by_points: same dict passed into
    NeuronParentAnalyser as layer_by_channel_by_label_by_points (aka "ACTS_DICT"
    in the exploration notebooks) — every cluster's full activation
    population, used for the Overview tab's per-cluster scatter (see
    report_stats.select_cluster_points, plotting.save_combined_scatter_png).

    config: report_config.ReportConfig — every tunable and feature toggle for
    this report:
    - max_points_per_cluster: cap on how many points from each cluster's
      population get plotted in the Overview tab's per-cluster scatter (see
      report_stats.select_cluster_points) — clusters can otherwise hold far
      more points than is legible or fast to render.
    - outlier_ratio_threshold: a matched cluster whose firing share falls below
      this (see report_stats.compute_cluster_breakdown's is_outlier) is flagged
      red in the "Firing by cluster" breakdown and excluded from the per-cluster
      scatter's matched/Kelly-colored set (folded into "unmatched" there instead)
      — too small a share of firings to call a real match.
    - relative_strength_method: which calculation feeds each card's Overview
      contribution-strength bar — "median_sum" (report_stats.compute_relative_strength_median_sum,
      default: this neuron's median contribution as a fraction of the sum
      of every dep neuron's median) or "median_per_image_share"
      (report_stats.compute_relative_strength_median_per_image_share: median,
      across images, of this neuron's own per-image share). The two are
      interchangeable; swap this to compare them without touching the rest of
      the report.
    - pw_samples: report_config.PwSamplesConfig, or None to skip the
      pointwise-multiplication-sample section for every cluster entirely. When
      set, up to pw_samples.max_samples wild samples per matched cluster are
      rendered, each paired with its own closest match from that cluster's
      population (see report_assets.dump_pw_sample_asset).
    - feature_viz: report_config.FeatureVizConfig, or None to skip feature-viz
      entirely (each wild sample is its own gradient-based optimization loop,
      so this can be slow). Requires pw_samples to be set — see
      ReportConfig.__post_init__. When set, every matched (non-outlier)
      cluster across every card that has pw_samples and a dep_layer_name in
      feature_viz.SUPPORTED_DEP_LAYER_NAMES gets a feature-viz reconstruction
      (see feature_viz.dump_feature_viz_assets) of its wild
      pointwise-multiplication samples — one render_vis call per sample (not
      jointly optimized), added as an extra row alongside
      dump_pw_sample_asset's wild/match grid. FeatureVizConfig.neurons can
      narrow this to a specific set of (dep_layer_name, dep_channel) pairs
      instead of every matched cluster, to bound the cost on a report with
      many dependency neurons. Collected across the *whole* report via
      report_data.build_report_data rather than resolved per-card.
    """
    data = build_report_data(stats_df, assets_dump_dir, assets_ref_dir, config)
    card_config = CardConfig(
        base_report_dir=base_report_dir,
        assets_dump_dir=assets_dump_dir,
        assets_ref_dir=assets_ref_dir,
        layer_by_channel_by_noise=layer_by_channel_by_noise,
        layer_by_channel_by_label_by_points=layer_by_channel_by_label_by_points,
        report=config,
    )

    sections = [
        s
        for s in (
            _render_section("Frequently firing", data.frequent, data, card_config, "Frequently firing cards"),
            _render_section("One-off / low frequency", data.one_off, data, card_config, "One-off / low frequency cards"),
        )
        if s is not None
    ]
    content = "\n\n".join(sections)
    print(content)
    if output_path is not None:
        Path(output_path).write_text(content)
