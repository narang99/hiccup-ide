import matplotlib.pyplot as plt
import numpy as np

from olt.act_ranges.constants import KELLY_COLORS

# KELLY_COLORS[0] is white, reserved as a background/reference color elsewhere
# (see NeuronParentAnalyser.plot_clusters) — excluded here too so no matched
# cluster gets a color easily confused with the noise/unmatched colors below.
_CLUSTER_COLORS = KELLY_COLORS[1:]
_NOISE_COLOR = "white"
_UNMATCHED_COLOR = "#888888"


def _style_dark_axis(fig, ax):
    fig.patch.set_facecolor("#212529")
    ax.set_facecolor("#212529")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")


def _jittered_x(n, rng):
    """
    Continuous random x in [0, n) for n points, instead of the integer
    sequence range(n) — plotting several same-length series (e.g. many
    same-size subsampled clusters) each against their own range(n) stacks
    points from every series onto the same handful of integer x positions,
    reading as solid vertical lines/columns rather than a scatter. Continuous
    jitter spreads them out so overlapping series are visually distinguishable.
    rng is caller-supplied so results are reproducible across a single figure.
    """
    return rng.uniform(0, max(n, 1), size=n)


def save_combined_scatter_png(
    activations, noise_samples, cluster_points_by_cid, matched_cids, output_path
):
    """
    Saves a dark-theme-friendly figure with two side-by-side scatter plots
    (sharey=True, so both read off the same activation-value y-axis):

    - Left: noise samples (white) and activation values (red) overlaid, each
      plotted against its own 0..len(series)-1 x-index (matching the
      axes[1].scatter(range(len(match.noise)), match.noise) idiom used in
      NeuronParentAnalyser.plot_clusters), so the two series aren't forced
      onto a shared x-axis meaning.
    - Right: one scatter series per matched dependency cluster (cluster_points_by_cid
      — sourced from layer_by_channel_by_label_by_points / "ACTS_DICT" via
      select_cluster_points, the same per-cluster point collection
      plot_clusters uses for its axes[0], subsampled per cluster) plus one
      pooled series covering every other (unmatched) cluster this dep neuron
      has — a dep neuron typically has far more clusters than the ones it
      fired against. Clusters in matched_cids get their own Kelly color and
      their own legend entry (cid=...); every other cluster's points are
      pooled into a single neutral gray "unmatched" series (one legend entry
      total), plotted first so matched ones render on top. Every series' x is
      continuous jitter (_jittered_x), not a bare range(len(points)) index —
      several equal-length series sharing the same integer x positions would
      otherwise stack into solid vertical lines instead of reading as a
      scatter (only y, via sharey, carries meaning here, same as the left
      plot).

    cluster_points_by_cid: dict[dep_cid, list[float]] — every cluster this dep
    neuron has (matched or not).
    matched_cids: set of dep_cid values this neuron actually matched — the
    subset of cluster_points_by_cid.keys() that gets Kelly-colored.
    Headless (savefig, no plt.show) since this is always regenerated during
    report building rather than viewed inline in a notebook.
    """
    fig, (ax_combined, ax_clusters) = plt.subplots(1, 2, figsize=(9, 3), sharey=True)
    for ax in (ax_combined, ax_clusters):
        _style_dark_axis(fig, ax)

    if len(noise_samples) > 0:
        ax_combined.scatter(
            range(len(noise_samples)), noise_samples, s=12, color=_NOISE_COLOR, label="noise"
        )
    ax_combined.scatter(
        range(len(activations)), activations, s=12, color="red", label="output_activation"
    )
    legend = ax_combined.legend(facecolor="#212529", labelcolor="white")
    legend.get_frame().set_edgecolor("white")

    rng = np.random.default_rng(0)
    matched_order = sorted(matched_cids)
    color_by_matched_cid = {
        cid: _CLUSTER_COLORS[i % len(_CLUSTER_COLORS)] for i, cid in enumerate(matched_order)
    }
    unmatched_points = [
        point
        for cid, points in cluster_points_by_cid.items()
        if cid not in matched_cids
        for point in points
    ]

    if unmatched_points:
        ax_clusters.scatter(
            _jittered_x(len(unmatched_points), rng),
            unmatched_points,
            s=8,
            color=_UNMATCHED_COLOR,
            label="unmatched",
        )
    for cid in matched_order:
        points = cluster_points_by_cid.get(cid, [])
        ax_clusters.scatter(
            _jittered_x(len(points), rng),
            points,
            s=14,
            color=color_by_matched_cid[cid],
            label=f"cid={cid}",
            zorder=3,
        )
    if cluster_points_by_cid:
        cluster_legend = ax_clusters.legend(
            facecolor="#212529", labelcolor="white", fontsize=7, loc="best"
        )
        cluster_legend.get_frame().set_edgecolor("white")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=fig.get_facecolor())
    plt.close(fig)
