"""
Precomputes everything print_report_for_neuron's cards need up front (rather
than each card recomputing its own slice), and collects feature-viz clusters
across the whole report so they can be optimized in one batched call. See
report_card.py for how these are consumed per-card.
"""

from dataclasses import dataclass

from olt.act_ranges.feature_viz import SUPPORTED_DEP_LAYER_NAMES, dump_feature_viz_assets
from olt.act_ranges.report_stats import (
    check_same_sign,
    compute_cluster_breakdown,
    compute_dep_order,
    compute_firing_stats,
    compute_image_act_sums,
    compute_per_image_shares,
    dedupe_to_one_origin_per_image,
    split_dep_order_by_frequency,
)


@dataclass
class ReportData:
    """Everything computed once per report (not per-card) that render_neuron_card needs."""

    dep_order: list
    image_keys: list
    full_groups: object
    medians: dict
    median_sum: float
    firing_counts: dict
    total_examples: int
    frequent: list
    one_off: list
    image_act_sums: dict
    row_by_dep_and_image: dict
    per_image_shares: dict
    cluster_stats_by_key: dict
    feature_viz_ref_by_key: dict


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
    deduped_stats_df = dedupe_to_one_origin_per_image(stats_df)
    image_keys = deduped_stats_df["input_image_key"].unique()[: config.max_input_keys]

    full_groups = stats_df.groupby(["dep_layer", "dep_channel"])
    groups = full_groups["output_activation"]
    medians = {key: groups.get_group(key).median() for key in dep_order}
    check_same_sign(medians.values())
    median_sum = sum(medians.values())

    firing_counts, total_examples = compute_firing_stats(stats_df)
    frequent, one_off = split_dep_order_by_frequency(dep_order, firing_counts, total_examples)

    image_act_sums = compute_image_act_sums(deduped_stats_df)
    row_by_dep_and_image = {
        (row["dep_layer"], row["dep_channel"], row["input_image_key"]): row
        for _, row in deduped_stats_df.iterrows()
    }
    per_image_shares = compute_per_image_shares(deduped_stats_df, image_act_sums)

    cluster_stats_by_key = {
        key: compute_cluster_breakdown(full_groups.get_group(key), config.outlier_ratio_threshold)
        for key in dep_order
    }

    feature_viz_ref_by_key = _build_feature_viz_refs(
        assets_dump_dir, assets_ref_dir, cluster_stats_by_key, config.pw_samples, config.feature_viz
    )

    return ReportData(
        dep_order=dep_order,
        image_keys=image_keys,
        full_groups=full_groups,
        medians=medians,
        median_sum=median_sum,
        firing_counts=firing_counts,
        total_examples=total_examples,
        frequent=frequent,
        one_off=one_off,
        image_act_sums=image_act_sums,
        row_by_dep_and_image=row_by_dep_and_image,
        per_image_shares=per_image_shares,
        cluster_stats_by_key=cluster_stats_by_key,
        feature_viz_ref_by_key=feature_viz_ref_by_key,
    )
