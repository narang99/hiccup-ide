"""Renders one neuron card (see report_render.render_neuron_card) from
a ReportData bundle (report_data.build_report_data) plus this report's fixed
config — the per-card counterpart to report_data.py's once-per-report work."""

from dataclasses import dataclass

from olt.act_ranges.report_assets import (
    dump_cluster_asset,
    dump_concentration_asset,
    dump_overview_assets,
    dump_pw_sample_asset,
)
from olt.act_ranges.report_render import render_neuron_card as render_neuron_card_shell
from olt.act_ranges.report_render import render_overview_tab_body
from olt.act_ranges.report_stats import compute_noise_radius_table


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


def _dep_pw_samples(config, dep_layer_name, dep_channel):
    pw_samples_config = config.report.pw_samples
    if pw_samples_config is None:
        return {}
    return pw_samples_config.layer_by_channel_by_cid_by_pw_samples.get(dep_layer_name, {}).get(str(dep_channel), {})


def _dep_cluster_notes(config, dep_layer_name, dep_channel):
    """This neuron's slice of config.report.cluster_notes (dict[(dep_layer_name,
    dep_channel, dep_cid), str]), reduced to dict[dep_cid, str] — the shape
    render_cluster_breakdown's note_by_cid expects, same convention as
    cluster_photo_ref_by_cid/pw_sample_ref_by_cid/feature_viz_ref_by_cid."""
    cluster_notes = config.report.cluster_notes
    if not cluster_notes:
        return {}
    return {
        dep_cid: note
        for (layer_name, channel, dep_cid), note in cluster_notes.items()
        if layer_name == dep_layer_name and channel == dep_channel
    }


def _build_overview_body(dep_layer_name, dep_channel, data, config, cluster_stats):
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
    scatter_ref_path = f"{config.assets_ref_dir}/{dep_layer_name}/{dep_channel}/overview_scatter.jpeg"

    cluster_photo_ref_by_cid, pw_sample_ref_by_cid, feature_viz_ref_by_cid = _cluster_asset_refs(
        dep_layer_name, dep_channel, data, config, cluster_stats
    )
    note_by_cid = _dep_cluster_notes(config, dep_layer_name, dep_channel)
    noise_radius_table_rows = compute_noise_radius_table(
        dep_full_rows, noise_samples, label_by_points, cluster_stats
    )

    return render_overview_tab_body(
        scatter_ref_path,
        cluster_stats,
        cluster_photo_ref_by_cid,
        pw_sample_ref_by_cid,
        feature_viz_ref_by_cid,
        note_by_cid,
        noise_radius_table_rows,
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


def render_neuron_card(dep_layer_name, dep_channel, data, config):
    """One card: just the Overview content (_build_overview_body), wrapped by
    report_render.render_neuron_card. No per-image tabs/panel-tabset — the
    card is the aggregate view across every row of stats_df for this dep
    neuron."""
    key = (dep_layer_name, dep_channel)
    cluster_stats = data.cluster_stats_by_key[key]
    overview_body = _build_overview_body(dep_layer_name, dep_channel, data, config, cluster_stats)

    pos_marker, neg_marker = data.marker_by_key[key]
    dump_concentration_asset(
        config.assets_dump_dir, dep_layer_name, dep_channel, data.pos_curve, data.neg_curve, pos_marker, neg_marker
    )
    concentration_ref_path = f"{config.assets_ref_dir}/{dep_layer_name}/{dep_channel}/concentration.jpeg"

    firing_pct = data.firing_counts.get(key, 0) / data.total_examples if data.total_examples else 0.0

    return render_neuron_card_shell(
        dep_layer_name, dep_channel, overview_body, concentration_ref_path, pos_marker, neg_marker, firing_pct
    )
