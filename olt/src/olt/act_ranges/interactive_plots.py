"""
Notebook-only plotting for NeuronParentAnalyser.plot_clusters: renders the
act-range scatter, cluster heatmap, and pointwise-multiplication comparison
for a single dependency match via plt.show(), as opposed to report_assets.py
(which dumps the equivalent visuals headlessly to disk for the Quarto report).
"""

import matplotlib.pyplot as plt

from olt.act_ranges.constants import KELLY_COLORS, LAYER_NAME_BY_SHAPE
from olt.act_ranges.reports import get_cluster_photo
from olt.act_ranges.stats import get_labels_above_noise_range
from olt.show import show_single_channel_red_green_black as S

_COLORS = KELLY_COLORS[1:]


def _color_map_for(labels):
    return {label: _COLORS[i % len(_COLORS)] for i, label in enumerate(labels)}


def _print_match_summary(dep_layer_name, dep_channel, n_labels, match):
    print(
        f"############### {dep_layer_name}:{dep_channel}, {n_labels}, "
        f"cid={match.best_cid}, sim={match.best_sim.item()}, "
        f"ratio={match.ratio:.3f} dist={match.point_dist:.3f}"
        f"#########################"
    )


def show_act_range_plot(match, l0, color_map, patch, cur_chan, cur_rel_y, cur_rel_x):
    """Scatter of each above-noise cluster's activation population (l0, see
    stats.get_labels_above_noise_range) next to the raw noise sample scatter,
    with this firing's own value marked as a dashed line."""
    _, axes = plt.subplots(1, 2, sharey=True, figsize=(15, 4))
    for label, ys in l0.items():
        axes[0].scatter(range(len(ys)), ys, color=color_map[label], label=label)

    axes[0].axhline(
        patch[cur_chan, cur_rel_y, cur_rel_x].item(), color="white", linestyle="--"
    )
    axes[1].scatter(range(len(match.noise)), match.noise)

    axes[0].legend()
    plt.show()


def show_cluster_heatmap(base_report_dir, dep_layer_name, dep_channel, match):
    """The matched cluster's combined heatmap photo, if one was generated for it
    (see reports.get_cluster_photo — absent for singleton clusters)."""
    cluster_photo = get_cluster_photo(
        base_report_dir,
        dep_layer_name,
        dep_channel,
        match.best_cid,
        "combined",
        crop_max_height=470,
    )
    if cluster_photo is not None:
        plt.imshow(cluster_photo)
        plt.show()


def show_pw_plot(dep_layer_name, match):
    """This firing's own dep_w * dep_patch next to the same product for up to
    three of the matched cluster's stored patches, for a side-by-side compare."""
    best_pws = [match.dep_w * p for p in match.best_patches[:3]]
    dep_pw = match.dep_w * match.dep_patch
    all_pws = [dep_pw] + best_pws
    all_pws = [p.reshape(LAYER_NAME_BY_SHAPE[dep_layer_name]) for p in all_pws]

    S(all_pws, 10, 4)
    plt.show()


def show_cluster_match(
    base_report_dir,
    dep_layer_name,
    dep_channel,
    match,
    patch,
    cur_chan,
    cur_rel_y,
    cur_rel_x,
    stuff_to_show,
    max_percent_points_allowed_in_noise_for_one_cluster,
):
    """Renders whichever of {"act_range", "heatmap", "pw"} are requested in
    stuff_to_show for one accepted dependency match, matching plot_clusters'
    original inline behavior."""
    l0 = get_labels_above_noise_range(
        match.label_by_points, match.noise, max_percent_points_allowed_in_noise_for_one_cluster
    )
    labels = list(l0.keys())
    color_map = _color_map_for(labels)

    _print_match_summary(dep_layer_name, dep_channel, len(labels), match)

    if "act_range" in stuff_to_show:
        show_act_range_plot(match, l0, color_map, patch, cur_chan, cur_rel_y, cur_rel_x)

    if len(labels) > 0:
        if "heatmap" in stuff_to_show:
            show_cluster_heatmap(base_report_dir, dep_layer_name, dep_channel, match)
        if "pw" in stuff_to_show:
            show_pw_plot(dep_layer_name, match)
