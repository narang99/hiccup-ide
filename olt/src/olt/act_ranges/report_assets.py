from olt.act_ranges.plotting import save_combined_scatter_png
from olt.act_ranges.report_stats import select_cluster_points
from olt.act_ranges.reports import get_cluster_photo


def dump_cluster_asset(assets_dump_dir, base_report_dir, dep_layer_name, dep_channel, dep_cid):
    """
    Writes (or reuses, if already present) the heatmap photo for one dependency
    cluster under {assets_dump_dir}/{dep_layer_name}/{dep_channel}/cluster_{dep_cid}.jpeg
    — this directory is the neuron's asset dir; the `cluster_` filename prefix is
    reserved for this purpose so other per-neuron assets can live alongside it
    without colliding. Returns the dumped Path, or None if the source photo doesn't exist.
    """
    neuron_dir = assets_dump_dir / dep_layer_name / str(dep_channel)
    neuron_dir.mkdir(parents=True, exist_ok=True)
    dump_path = neuron_dir / f"cluster_{dep_cid}.jpeg"
    if not dump_path.exists():
        photo = get_cluster_photo(
            base_report_dir, dep_layer_name, dep_channel, dep_cid, "combined", crop_max_height=470
        )
        if photo is None:
            return None
        photo.save(dump_path)
    return dump_path


def dump_overview_assets(
    assets_dump_dir,
    dep_layer_name,
    dep_channel,
    dep_full_rows,
    noise_samples,
    label_by_points,
    matched_cids,
    max_points_per_cluster=50,
):
    """
    Always (re)writes one combined scatter PNG for one dep neuron's Overview
    card, under {assets_dump_dir}/{dep_layer_name}/{dep_channel}/. Reserved
    filename (parallel to the cluster_ prefix reserved by dump_cluster_asset):
    overview_scatter.png. Unlike dump_cluster_asset, this is always
    regenerated (no exists-check) — cheap to recompute from stats_df each run.

    dep_full_rows: the full stats_df rows for this (dep_layer, dep_channel) —
    needs "output_activation".
    label_by_points: this (dep_layer, dep_channel)'s slice of
    layer_by_channel_by_label_by_points (dict[str(cid), list[float]]) — used
    via select_cluster_points to plot every one of this dep neuron's clusters
    (not just matched ones) alongside the activation/noise scatter, each
    subsampled to at most max_points_per_cluster points (see
    save_combined_scatter_png).
    matched_cids: the dep_cid values to Kelly-color/give their own legend
    entry — the caller excludes outlier clusters (see
    compute_cluster_breakdown's is_outlier) so a cluster that only accounts
    for a tiny share of firings doesn't get treated as a real match here.

    Returns the dumped Path.
    """
    neuron_dir = assets_dump_dir / dep_layer_name / str(dep_channel)
    scatter_path = neuron_dir / "overview_scatter.png"
    cluster_points_by_cid = select_cluster_points(label_by_points, max_points_per_cluster)
    save_combined_scatter_png(
        list(dep_full_rows["output_activation"]),
        list(noise_samples),
        cluster_points_by_cid,
        matched_cids,
        scatter_path,
    )
    return scatter_path
