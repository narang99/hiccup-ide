"""
Precomputes everything print_report_for_neuron's cards need up front (rather
than each card recomputing its own slice), and collects feature-viz clusters
across the whole report so they can be optimized in one batched call. See
report_card.py for how these are consumed per-card.
"""

from dataclasses import dataclass

from olt.act_ranges.feature_viz import SUPPORTED_DEP_LAYER_NAMES, dump_feature_viz_assets
from olt.act_ranges.report_stats import (
    compute_cluster_breakdown,
    compute_concentration_curves,
    compute_concentration_values,
    compute_dep_order,
    compute_firing_stats,
    split_dep_order_by_frequency,
)


@dataclass
class ReportData:
    """Everything computed once per report (not per-card) that render_neuron_card needs."""

    dep_order: list
    full_groups: object
    medians: dict
    firing_counts: dict
    total_examples: int
    frequent: list
    one_off: list
    one_off_threshold: int
    cluster_stats_by_key: dict
    feature_viz_ref_by_key: dict
    pos_curve: list
    neg_curve: list
    marker_by_key: dict


def _validate_single_origin(stats_df):
    origin_layers = stats_df["origin_layer"].unique()
    origin_channels = stats_df["origin_channel"].unique()
    if len(origin_layers) != 1 or len(origin_channels) != 1:
        raise ValueError(
            "stats_df must contain exactly one (origin_layer, origin_channel) pair, got "
            f"origin_layers={list(origin_layers)}, origin_channels={list(origin_channels)}"
        )


def _collect_feature_viz_clusters(cluster_stats_by_key, layer_by_channel_by_cid_by_pw_samples, neurons=None):
    clusters = []
    for (dep_layer_name, dep_channel), cluster_stats in cluster_stats_by_key.items():
        if dep_layer_name not in SUPPORTED_DEP_LAYER_NAMES:
            continue
        if neurons is not None and (dep_layer_name, dep_channel) not in neurons:
            continue
        dep_pw_samples = layer_by_channel_by_cid_by_pw_samples.get(dep_layer_name, {}).get(str(dep_channel), {})
        for row in cluster_stats:
            if row["is_outlier"]:
                continue
            samples = dep_pw_samples.get(str(row["dep_cid"]))
            if samples:
                clusters.append(
                    {
                        "dep_layer_name": dep_layer_name,
                        "dep_channel": dep_channel,
                        "dep_cid": row["dep_cid"],
                        "pw_samples": samples,
                    }
                )
    return clusters


def _build_feature_viz_refs(assets_dump_dir, assets_ref_dir, cluster_stats_by_key, pw_samples_config, feature_viz_config):
    """Collects every matched, non-outlier cluster across the whole report
    (optionally narrowed to feature_viz_config.neurons) into a single
    dump_feature_viz_assets call, and returns dict[(dep_layer_name,
    dep_channel, dep_cid), ref_path] for whichever ones produced an asset."""
    if feature_viz_config is None:
        return {}
    clusters = _collect_feature_viz_clusters(
        cluster_stats_by_key,
        pw_samples_config.layer_by_channel_by_cid_by_pw_samples,
        neurons=feature_viz_config.neurons,
    )
    if not clusters:
        return {}
    paths_by_key = dump_feature_viz_assets(
        assets_dump_dir,
        feature_viz_config.model,
        clusters,
        max_samples=pw_samples_config.max_samples,
        image_size=feature_viz_config.image_size,
        thresholds=feature_viz_config.thresholds,
    )
    return {
        key: f"{assets_ref_dir}/{key[0]}/{key[1]}/featureviz_{key[2]}.jpeg"
        for key, path in paths_by_key.items()
        if path is not None
    }


def build_report_data(stats_df, assets_dump_dir, assets_ref_dir, config):
    _validate_single_origin(stats_df)

    dep_order = compute_dep_order(stats_df)

    full_groups = stats_df.groupby(["dep_layer", "dep_channel"])
    groups = full_groups["contribution"]
    medians = {key: groups.get_group(key).median() for key in dep_order}

    firing_counts, total_examples = compute_firing_stats(stats_df)
    frequent, one_off, one_off_threshold = split_dep_order_by_frequency(
        dep_order, firing_counts, total_examples
    )

    # Concentration curves deliberately exclude one_off/outlier neurons — the
    # curves are meant to reflect the actual concentration among neurons
    # someone would look at; folding in rare one-off firers would mask that
    # with mass that doesn't really co-occur with the rest in practice.
    concentration_values = compute_concentration_values(
        medians, firing_counts, total_examples, config.concentration_metric
    )
    pos_curve, neg_curve, marker_by_key = compute_concentration_curves(concentration_values, frequent)

    # Only for `frequent` — one_off neurons aren't rendered as cards (see
    # print_report_for_neuron's summary), so there's no need to pay for their
    # cluster breakdown or (potentially expensive) feature-viz assets.
    cluster_stats_by_key = {
        key: compute_cluster_breakdown(full_groups.get_group(key), config.outlier_ratio_threshold)
        for key in frequent
    }

    feature_viz_ref_by_key = _build_feature_viz_refs(
        assets_dump_dir, assets_ref_dir, cluster_stats_by_key, config.pw_samples, config.feature_viz
    )

    return ReportData(
        dep_order=dep_order,
        full_groups=full_groups,
        medians=medians,
        firing_counts=firing_counts,
        total_examples=total_examples,
        frequent=frequent,
        one_off=one_off,
        one_off_threshold=one_off_threshold,
        cluster_stats_by_key=cluster_stats_by_key,
        feature_viz_ref_by_key=feature_viz_ref_by_key,
        pos_curve=pos_curve,
        neg_curve=neg_curve,
        marker_by_key=marker_by_key,
    )
