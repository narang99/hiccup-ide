"""Renders one neuron card (see report_render.render_neuron_tabset_card) from
a ReportData bundle (report_data.build_report_data) plus this report's fixed
config — the per-card counterpart to report_data.py's once-per-report work."""

from dataclasses import dataclass

from olt.act_ranges.report_assets import dump_cluster_asset, dump_overview_assets, dump_pw_sample_asset
from olt.act_ranges.report_render import (
    render_image_tab_body,
    render_image_tab_placeholder_body,
    render_neuron_tabset_card,
    render_overview_tab_body,
)
from olt.act_ranges.report_stats import (
    compute_relative_strength_median_per_image_share,
    compute_relative_strength_median_sum,
)


@dataclass
class CardConfig:
    """Fixed for the whole report — see print_report_for_neuron for field meanings.
    report holds the user-facing tunables/toggles (report_config.ReportConfig);
    the rest are paths and per-layer data dicts that aren't report_config's concern."""

    base_report_dir: object
    assets_dump_dir: object
    assets_ref_dir: str
    layer_by_channel_by_noise: dict
    layer_by_channel_by_label_by_points: dict
    report: object  # report_config.ReportConfig


def _relative_strength(key, data, config):
    method = config.report.relative_strength_method
    if method == "median_sum":
        return compute_relative_strength_median_sum(key, data.medians, data.median_sum)
    if method == "median_per_image_share":
        return compute_relative_strength_median_per_image_share(key, data.per_image_shares)
    raise ValueError(
        f"unknown relative_strength_method: {method!r}, expected \"median_sum\" or \"median_per_image_share\""
    )


def _dep_pw_samples(config, dep_layer_name, dep_channel):
    pw_samples_config = config.report.pw_samples
    if pw_samples_config is None:
        return {}
    return pw_samples_config.layer_by_channel_by_cid_by_pw_samples.get(dep_layer_name, {}).get(str(dep_channel), {})


def _build_overview_body(dep_layer_name, dep_channel, data, config, cluster_stats, firing_counts):
    key = (dep_layer_name, dep_channel)
    dep_full_rows = data.full_groups.get_group(key)
    noise_samples = config.layer_by_channel_by_noise.get(dep_layer_name, {}).get(str(dep_channel), [])
    label_by_points = config.layer_by_channel_by_label_by_points.get(dep_layer_name, {}).get(str(dep_channel), {})
    matched_cids = {row["dep_cid"] for row in cluster_stats if not row["is_outlier"]}

    dump_overview_assets(
        config.assets_dump_dir,
        dep_layer_name,
        dep_channel,
        dep_full_rows,
        noise_samples,
        label_by_points,
        matched_cids,
        config.report.max_points_per_cluster,
    )
    scatter_ref_path = f"{config.assets_ref_dir}/{dep_layer_name}/{dep_channel}/overview_scatter.png"

    cluster_photo_ref_by_cid, pw_sample_ref_by_cid, feature_viz_ref_by_cid = _cluster_asset_refs(
        dep_layer_name, dep_channel, data, config, cluster_stats
    )

    return render_overview_tab_body(
        _relative_strength(key, data, config),
        data.medians[key],
        scatter_ref_path,
        firing_counts.get(key, 0),
        data.total_examples,
        cluster_stats,
        cluster_photo_ref_by_cid,
        pw_sample_ref_by_cid,
        feature_viz_ref_by_cid,
    )


def _cluster_asset_refs(dep_layer_name, dep_channel, data, config, cluster_stats):
    cluster_photo_ref_by_cid = {}
    pw_sample_ref_by_cid = {}
    feature_viz_ref_by_cid = {}
    pw_samples_config = config.report.pw_samples
    dep_pw_samples = _dep_pw_samples(config, dep_layer_name, dep_channel)

    for row in cluster_stats:
        dep_cid = row["dep_cid"]
        dump_path = dump_cluster_asset(
            config.assets_dump_dir, config.base_report_dir, dep_layer_name, dep_channel, dep_cid
        )
        if dump_path is not None:
            cluster_photo_ref_by_cid[dep_cid] = f"{config.assets_ref_dir}/{dep_layer_name}/{dep_channel}/cluster_{dep_cid}.jpeg"

        samples = dep_pw_samples.get(str(dep_cid))
        if samples and pw_samples_config is not None:
            pw_dump_path = dump_pw_sample_asset(
                config.assets_dump_dir,
                dep_layer_name,
                dep_channel,
                dep_cid,
                samples,
                max_samples=pw_samples_config.max_samples,
            )
            if pw_dump_path is not None:
                pw_sample_ref_by_cid[dep_cid] = f"{config.assets_ref_dir}/{dep_layer_name}/{dep_channel}/pw_{dep_cid}.jpeg"

        feature_viz_ref = data.feature_viz_ref_by_key.get((dep_layer_name, dep_channel, dep_cid))
        if feature_viz_ref is not None:
            feature_viz_ref_by_cid[dep_cid] = feature_viz_ref

    return cluster_photo_ref_by_cid, pw_sample_ref_by_cid, feature_viz_ref_by_cid


def _build_image_tabs(dep_layer_name, dep_channel, data, config):
    tabs = []
    for i, image_key in enumerate(data.image_keys):
        row = data.row_by_dep_and_image.get((dep_layer_name, dep_channel, image_key))
        dump_path = None
        if row is not None:
            dump_path = dump_cluster_asset(
                config.assets_dump_dir, config.base_report_dir, dep_layer_name, dep_channel, row["dep_cid"]
            )
        if row is None or dump_path is None:
            tabs.append((i, render_image_tab_placeholder_body()))
            continue

        act_sum = data.image_act_sums[image_key]
        relative_strength = row["output_activation"] / act_sum if act_sum != 0 else 0.0
        ref_path = f"{config.assets_ref_dir}/{dep_layer_name}/{dep_channel}/cluster_{row['dep_cid']}.jpeg"
        tabs.append(
            (
                i,
                render_image_tab_body(
                    row["dep_cid"],
                    row["noise_distance"],
                    row["output_activation"],
                    row["similarity"],
                    ref_path,
                    relative_strength,
                ),
            )
        )
    return tabs


def render_neuron_card(dep_layer_name, dep_channel, data, config):
    """One card: an Overview tab (_build_overview_body) plus one tab per input
    image (_build_image_tabs), wrapped by report_render.render_neuron_tabset_card."""
    cluster_stats = data.cluster_stats_by_key[(dep_layer_name, dep_channel)]
    overview_body = _build_overview_body(dep_layer_name, dep_channel, data, config, cluster_stats, data.firing_counts)
    image_tabs = _build_image_tabs(dep_layer_name, dep_channel, data, config)
    return render_neuron_tabset_card(dep_layer_name, dep_channel, overview_body, image_tabs)
