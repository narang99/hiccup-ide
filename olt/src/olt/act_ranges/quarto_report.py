from pathlib import Path

from tqdm import tqdm

from olt.act_ranges.report_assets import (
    dump_cluster_asset,
    dump_firing_frequency_histogram_asset,
    dump_output_activation_histogram_asset,
)
from olt.act_ranges.report_card import CardConfig, render_neuron_card
from olt.act_ranges.report_data import build_report_data
from olt.act_ranges.report_render import (
    render_notes_summary,
    render_origin_cluster_header,
    render_report_histograms,
    render_summary,
)
from olt.act_ranges.report_stats import (
    compute_firing_frequency_ratios,
    compute_output_activation_noise_max_distances,
)


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
    origin_cluster_label=None,
):
    """
    Writes the full Quarto markdown to output_path if given; otherwise prints
    it to stdout (copy it into a .qmd file to render/test) — never both, so
    looping this over many clusters/reports (each with its own output_path)
    doesn't spam stdout with every report's full content.

    The report opens with a "# Report: Dependencies of {origin_layer}:{origin_channel}[{origin_cluster_label}]"
    title (the "[{origin_cluster_label}]" suffix is omitted if
    origin_cluster_label is None) and, if origin_cluster_label is given, that
    cluster's own photo (report_render.render_origin_cluster_header) —
    origin_layer/origin_channel come from stats_df itself; origin_cluster_label
    identifies which of that neuron's clusters this report's stats_df was
    built for (e.g. the LABEL/cluster_label used to filter the input rows
    passed into collect_cluster_stats_df). Next is a side-by-side pair of
    report-level histograms (report_render.render_report_histograms): the
    output-activation-vs-noise histogram (report_stats.compute_output_activation_noise_max_distances,
    config.histogram_bins) — every "frequent" dep neuron's firings pooled
    into one distribution, expressed in noise-radius units so neurons with
    different raw activation scales are directly comparable — and the firing-
    frequency histogram (report_stats.compute_firing_frequency_ratios), one
    value per "frequent" dep neuron (firing_count / total_examples). Both
    exclude one-off/low-frequency dep neurons, same population as the cards
    below. Then a one-line
    summary (report_render.render_summary) stating how many dependency
    neurons were excluded as one-off/low-frequency
    firers (see report_stats.split_dep_order_by_frequency) and the firing-count
    threshold used — those neurons are not rendered as cards at all — followed,
    if config.cluster_notes is set, by a collapsible "Cluster notes" callout
    (report_render.render_notes_summary) listing every note in one place. The
    rest is one card per remaining (frequently-firing) dependency neuron (see
    report_stats.compute_dep_order for ordering). Each card
    (report_render.render_neuron_card) has a header with the neuron's name, a
    plain "Fired: {percent}%" figure, and its tiny positive/negative
    concentration sparkline with ticker-style cumulative-share labels (see
    report_stats.compute_concentration_curves) — and a body with just the
    combined activation/noise scatter plot + per-cluster firing breakdown,
    aggregated across every row of stats_df for that dep neuron. No
    per-image tabs/panel-tabset, and no separate contribution/firing-frequency
    bars — the header figures replace both.

    stats_df: the concatenation of NeuronParentAnalyser.collect_cluster_stats_df
    outputs across multiple input images, for a single neuron — must have exactly
    one (origin_layer, origin_channel) pair and an "input_image_key" column.

    layer_by_channel_by_noise: same dict passed into NeuronParentAnalyser, used
    for the Overview tab's raw noise-sample scatter plots.
    layer_by_channel_by_label_by_points: same dict passed into
    NeuronParentAnalyser as layer_by_channel_by_label_by_points (aka "ACTS_DICT"
    in the exploration notebooks) — every cluster's full activation
    population, used for the Overview tab's per-cluster scatter (see
    report_stats.select_cluster_points, plotting.save_combined_scatter_jpeg).

    config: report_config.ReportConfig — every tunable and feature toggle for
    this report:
    - max_points_per_cluster: cap on how many points from each cluster's
      population get plotted in the Overview tab's per-cluster scatter (see
      report_stats.select_cluster_points) — clusters can otherwise hold far
      more points than is legible or fast to render.
    - outlier_ratio_threshold: a matched cluster whose firing share falls below
      this (see report_stats.compute_cluster_breakdown's is_outlier) is flagged
      yellow in the "Firing by cluster" breakdown and excluded from the per-cluster
      scatter's matched/Kelly-colored set (folded into "unmatched" there instead)
      — too small a share of firings to call a real match. (Not to be confused
      with the one-off/low-frequency dep neurons excluded up front by the
      report's opening summary — this is about clusters within a single
      neuron's own firings.)
    - concentration_metric: which per-dep-neuron value feeds the header's
      concentration sparkline/ticker labels — "median" (default: the raw
      median contribution) or "weighted" (median scaled by firing rate, so
      an infrequently-firing neuron's contribution doesn't overquantify its
      apparent share — see report_stats.compute_concentration_values).
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
    - cluster_notes: dict[(dep_layer_name, dep_channel, dep_cid), str], or
      None to skip. Free-text notes for specific dependency clusters,
      rendered both in the top-of-report "Cluster notes" callout
      (render_notes_summary) and inside that cluster's own "show cluster
      photo" section in its card (render_cluster_breakdown), in a matching
      callout-note block.
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

    origin_layer = stats_df["origin_layer"].iloc[0]
    origin_channel = stats_df["origin_channel"].iloc[0]
    origin_cluster_photo_ref = None
    if origin_cluster_label is not None:
        dump_path = dump_cluster_asset(
            assets_dump_dir, base_report_dir, origin_layer, origin_channel, origin_cluster_label
        )
        if dump_path is not None:
            origin_cluster_photo_ref = f"{assets_ref_dir}/{origin_layer}/{origin_channel}/cluster_{origin_cluster_label}.jpeg"
    origin_header = render_origin_cluster_header(
        origin_layer, origin_channel, origin_cluster_label, origin_cluster_photo_ref
    )

    histogram_distances = compute_output_activation_noise_max_distances(
        stats_df, layer_by_channel_by_noise, data.frequent
    )
    activation_histogram_ref_path = f"{assets_ref_dir}/output_activation_histogram.jpeg"
    dump_output_activation_histogram_asset(assets_dump_dir, histogram_distances, bins=config.histogram_bins)

    firing_frequency_ratios = compute_firing_frequency_ratios(
        data.firing_counts, data.total_examples, data.frequent
    )
    firing_frequency_histogram_ref_path = f"{assets_ref_dir}/firing_frequency_histogram.jpeg"
    dump_firing_frequency_histogram_asset(assets_dump_dir, firing_frequency_ratios, bins=config.histogram_bins)

    histogram_section = render_report_histograms(
        activation_histogram_ref_path, firing_frequency_histogram_ref_path
    )

    summary = render_summary(len(data.one_off), data.one_off_threshold, data.total_examples)
    notes_summary = render_notes_summary(config.cluster_notes)
    frequent_section = _render_section(
        "Frequently firing", data.frequent, data, card_config, "Frequently firing cards"
    )
    sections = [s for s in (origin_header, histogram_section, summary, notes_summary, frequent_section) if s]
    content = "\n\n".join(sections)
    if output_path is not None:
        Path(output_path).write_text(content)
    else:
        print(content)
