import matplotlib.pyplot as plt
import numpy as np

from olt.act_ranges.constants import KELLY_COLORS

# KELLY_COLORS[0] is white, reserved as the noise scatter's color below —
# excluded here too so no matched cluster gets a color easily confused with
# the noise/unmatched colors below.
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


_POS_FILL = "#8fd19e"  # light green
_NEG_FILL = "#f4a3a3"  # light red
_POS_MARKER = "#1b7a34"  # dark green
_NEG_MARKER = "#a11f1f"  # dark red
_HIST_COLOR = "#4dabf7"  # light blue


def save_output_activation_histogram_jpeg(distances, output_path, bins=40):
    """
    Report-level (not per-card) histogram: distance of every collected
    "frequent" dep neuron's firings from its own noise_max, in noise-radius
    units (see report_stats.compute_output_activation_noise_max_distances) —
    pooled across every dependency neuron and every origin example in this
    report, since the per-neuron noise-radius normalization is what makes
    pooling different neurons' activation scales into one histogram
    meaningful. A dashed white line at x=0 marks "right at this neuron's own
    noise ceiling" — mass to the right is firings clearly above their own
    neuron's noise; mass at/left of it is firings that, relative to their own
    neuron's noise, look unremarkable. bins is caller-configurable (see
    report_config.ReportConfig.histogram_bins) since the right resolution
    depends on how many firings/dep neurons a given report has.

    Two additional solid white lines mark the min and max of distances
    (the extremes of the pooled distribution), each labeled with its own
    numeric value — so the plotted range's edges are readable, not just
    eyeballed off the x-axis. A text annotation in the top corners reports
    how many pooled values fall below/above the x=0 noise-ceiling line
    (n<0 / n>0), a coarser, exact-count summary of the same mass-left-vs-
    mass-right read the dashed line is meant to convey visually.
    """
    distances = np.asarray(distances)
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    _style_dark_axis(fig, ax)
    ax.hist(distances, bins=bins, color=_HIST_COLOR, edgecolor="none")
    ax.axvline(x=0, color="white", linestyle="--", linewidth=1)

    if len(distances) > 0:
        dmin, dmax = distances.min(), distances.max()
        ax.axvline(x=dmin, color="white", linestyle="-", linewidth=1)
        ax.axvline(x=dmax, color="white", linestyle="-", linewidth=1)
        ymax = ax.get_ylim()[1]
        ax.text(dmin, ymax, f"{dmin:.2f}", color="white", ha="left", va="bottom", fontsize=8)
        ax.text(dmax, ymax, f"{dmax:.2f}", color="white", ha="right", va="bottom", fontsize=8)

        n_below = int((distances < 0).sum())
        n_above = int((distances > 0).sum())
        ax.text(
            0.01, 0.95, f"n<0: {n_below}", color="white", fontsize=8,
            ha="left", va="top", transform=ax.transAxes,
        )
        ax.text(
            0.99, 0.95, f"n>0: {n_above}", color="white", fontsize=8,
            ha="right", va="top", transform=ax.transAxes,
        )

    ax.set_xlabel("distance from noise_max (noise-radius units)", color="white")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="jpeg", facecolor=fig.get_facecolor(), pil_kwargs={"quality": 90})
    plt.close(fig)


def save_firing_frequency_histogram_jpeg(ratios, output_path, bins=20, total_neurons=None):
    """
    Report-level histogram of firing frequency: one value per "frequent" dep
    neuron (firing_count / total_examples, see
    report_stats.compute_firing_frequency_ratios) — how large a share of
    origin examples each dep neuron actually fired in. Same exclusion as
    save_output_activation_histogram_jpeg: one_off/outlier neurons (see
    report_stats.split_dep_order_by_frequency) are never included, since
    they were already dropped from `frequent` upstream before this is
    called — kept in sync deliberately, not incidentally.

    total_neurons, if given, is annotated in the bottom-right corner (same
    convention as save_output_activation_histogram_jpeg) — here it's exactly
    len(ratios), since this histogram has one entry per neuron rather than
    one entry per firing.
    """
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    _style_dark_axis(fig, ax)
    ax.hist(ratios, bins=bins, color=_HIST_COLOR, edgecolor="none")

    if total_neurons is not None:
        ax.text(
            0.99, 0.02, f"n neurons: {total_neurons}", color="white", fontsize=8,
            ha="right", va="bottom", transform=ax.transAxes,
        )

    ax.set_xlabel("firing frequency (fraction of examples)", color="white")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="jpeg", facecolor=fig.get_facecolor(), pil_kwargs={"quality": 90})
    plt.close(fig)


def save_concentration_sparkline_jpeg(pos_curve, neg_curve, pos_marker, neg_marker, output_path):
    """
    Tiny two-panel sparkline showing the raw cumulative "concentration" curve
    (report_stats.compute_concentration_curves — plain cumulative sums of
    |median contribution|, no normalization) for positive contributions
    (left, light green fill) and negative contributions (right, light red
    fill) across the whole report: x is rank (1st, 2nd, ... neuron by
    |median contribution| within that sign), y is cumulative |median
    contribution| magnitude. Meant to counter over-weighting a single
    bright/wide bar — the filled area shows at a glance how many neurons and
    how much magnitude it actually took to reach a given height.

    sharey=True (not a fixed range) is what makes the two sides comparable:
    matplotlib scales both panels to the same y-axis, sized to whichever
    side's cumulative total is larger, so a real difference in scale between
    positive and negative shows up as an honest height difference instead of
    both being squashed into an identical [0, 1] range. bottom is pinned to
    0 explicitly (see ylim below) so an empty/near-empty side reads as
    "barely any height" rather than autoscaling to its own tiny range and
    looking full.

    pos_marker/neg_marker: each either None (no line drawn — this card's own
    neuron isn't on that side, see compute_concentration_curves) or a tuple
    whose first element is rank — used to draw a full-height dark vertical
    line at that x position; the position is the only thing used here; the
    cumulative-share/delta values in the same tuple are for
    report_render's printed ticker text next to the image, a separate
    calculation unrelated to what's plotted (see compute_concentration_curves).
    Exactly one of the two is ever non-None for a given card.
    No axis ticks/labels/legend — this is a glanceable sparkline, not a
    standalone chart, meant to sit inline in a card header. Rendered
    oversized relative to its final display size (high dpi) so it stays
    crisp when scaled down.
    """
    fig, (ax_pos, ax_neg) = plt.subplots(1, 2, figsize=(3.2, 1.0), sharey=True)
    fig.patch.set_facecolor("#212529")
    top = max(pos_curve[-1] if pos_curve else 0.0, neg_curve[-1] if neg_curve else 0.0)
    for ax, curve, fill, marker, marker_color in (
        (ax_pos, pos_curve, _POS_FILL, pos_marker, _POS_MARKER),
        (ax_neg, neg_curve, _NEG_FILL, neg_marker, _NEG_MARKER),
    ):
        ax.set_facecolor("#212529")
        ax.set_ylim(0, top * 1.05 if top > 0 else 1.0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        if curve:
            xs = range(1, len(curve) + 1)
            ax.fill_between(xs, curve, color=fill, alpha=0.9, linewidth=0)
            ax.plot(xs, curve, color=fill, linewidth=1.2)
            ax.set_xlim(1, len(curve))
        if marker is not None:
            rank = marker[0]
            ax.axvline(x=rank, color=marker_color, linewidth=2.2, zorder=5)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, wspace=0.12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path, format="jpeg", facecolor=fig.get_facecolor(), dpi=300, pil_kwargs={"quality": 90}
    )
    plt.close(fig)


def save_combined_scatter_jpeg(
    activations, noise_samples, cluster_points_by_cid, matched_cids, output_path
):
    """
    Saves a dark-theme-friendly figure with two side-by-side scatter plots
    (sharey=True, so both read off the same activation-value y-axis):

    - Left: noise samples (white) and activation values (red) overlaid, each
      plotted against its own 0..len(series)-1 x-index, so the two series
      aren't forced onto a shared x-axis meaning.
    - Right: one scatter series per matched dependency cluster (cluster_points_by_cid
      — sourced from layer_by_channel_by_label_by_points / "ACTS_DICT" via
      select_cluster_points, subsampled per cluster) plus one pooled series
      covering every other (unmatched) cluster this dep neuron
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
    fig.savefig(output_path, format="jpeg", facecolor=fig.get_facecolor(), pil_kwargs={"quality": 90})
    plt.close(fig)
