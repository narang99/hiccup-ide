import math
from pathlib import Path

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


def _draw_output_activation_histogram(ax, distances, bins=40):
    """
    Draws one output-activation-vs-noise histogram onto ax: distance of every
    given firing from its own dep neuron's noise_med (median), in noise-radius units
    (see report_stats.compute_output_activation_noise_median_distances). A
    dashed white line at x=0 marks "right at this neuron's own noise
    median" — mass to the right is firings clearly above their own neuron's
    typical noise; mass at/left of it is firings that, relative to their own
    neuron's noise, look unremarkable. Two additional solid white lines mark
    the min and max of distances (the extremes of this distribution) — no
    in-plot text (counts/values are stated once in
    report_render.render_report_stats_summary instead, since text
    annotations inside a small figure get cramped/overlapping).

    Shared by save_output_activation_histogram_jpeg (headless, one panel per
    JPEG) and show_output_activation_histograms (interactive, three panels
    in one figure) so both draw identically.
    """
    distances = np.asarray(distances)
    ax.hist(distances, bins=bins, color=_HIST_COLOR, edgecolor="none")
    ax.axvline(x=0, color="white", linestyle="--", linewidth=1)

    if len(distances) > 0:
        ax.axvline(x=distances.min(), color="white", linestyle="-", linewidth=1)
        ax.axvline(x=distances.max(), color="white", linestyle="-", linewidth=1)

    ax.set_xlabel("distance from noise_med (noise-radius units)", color="white")


def save_output_activation_histogram_jpeg(distances, output_path, bins=40):
    """
    Report-level (not per-card) histogram: distance of every collected
    "frequent" dep neuron's firings from its own noise_med (median), in noise-radius
    units (see report_stats.compute_output_activation_noise_median_distances) —
    pooled across every dependency neuron and every origin example in this
    report, since the per-neuron noise-radius normalization is what makes
    pooling different neurons' activation scales into one histogram
    meaningful. bins is caller-configurable (see
    report_config.ReportConfig.histogram_bins) since the right resolution
    depends on how many firings/dep neurons a given report has. See
    _draw_output_activation_histogram for the actual plot.
    """
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    _style_dark_axis(fig, ax)
    _draw_output_activation_histogram(ax, distances, bins=bins)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="jpeg", facecolor=fig.get_facecolor(), pil_kwargs={"quality": 90})
    plt.close(fig)


def show_output_activation_histograms(
    all_distances, adding_distances, inhibiting_distances, bins=40, figsize=(14, 3.5)
):
    """
    Interactive (non-headless) sibling of save_output_activation_histogram_jpeg
    — the same three output-activation-vs-noise histograms
    print_report_for_neuron renders in its report row (see
    report_render.render_report_histograms:
    all-pooled / adding (contribution > 0) / inhibiting (contribution <= 0),
    see report_stats.compute_output_activation_noise_median_distances and
    report_stats.split_output_activation_distances_by_contribution_sign for
    how to produce these three lists), shown live in one matplotlib figure
    via plt.show() instead of dumped to JPEGs — for quick notebook
    exploration of a stats_df's noise distances directly, without touching
    print_report_for_neuron/build_report_data's asset-dumping or
    card-rendering machinery at all.

    Returns (fig, axs).
    """
    fig, axs = plt.subplots(1, 3, figsize=figsize)
    for ax, distances, title in zip(
        axs,
        (all_distances, adding_distances, inhibiting_distances),
        ("all", "adding (contribution > 0)", "inhibiting (contribution <= 0)"),
    ):
        _style_dark_axis(fig, ax)
        _draw_output_activation_histogram(ax, distances, bins=bins)
        ax.set_title(title, color="white")
    fig.tight_layout()
    return fig, axs


def show_label_points_grid(
    label_by_points, n_rows, n_cols, colors=None, figsize=None, sharey=True, max_points_per_label=200, rng=None
):
    """
    Interactive grid of per-label scatter panels — one panel of overlaid
    scatters per chunk of labels, labels split evenly (ceil(n_labels /
    (n_rows*n_cols)) per panel) across n_rows*n_cols panels, e.g. for eyeballing
    every cluster's raw point population (layer_by_channel_by_label_by_points)
    at once instead of one cluster at a time.

    label_by_points: dict[label, list[float]] — the full point population per
    label; each label is subsampled to at most max_points_per_label points
    (without replacement — a label already at or under the cap is plotted
    as-is rather than resampled with possible duplicates).

    Each label within a panel is plotted with jittered (not bare range(n)) x
    positions (see _jittered_x) — several labels capped to the same
    max_points_per_label length would otherwise share identical integer x
    positions and stack into vertical stripes instead of reading as a
    scatter (the same fix save_combined_scatter_jpeg uses for its overview
    scatter).

    colors: cycled (via modulo) across labels if there are more labels than
    colors — defaults to constants.KELLY_COLORS[1:] (index 0 is reserved
    elsewhere in this module for a "noise" series color, see
    _CLUSTER_COLORS).

    rng: np.random.Generator for subsampling/jitter reproducibility — a
    fresh np.random.default_rng() is used if not given.

    Panels beyond however many are actually needed to place every label are
    turned off (axis hidden) rather than left as empty ticked boxes — this
    is computed per-panel from panel_labels directly (not sliced off the end
    by label count), since a chunk size > 1 per panel means "panels with no
    labels" and "labels used up" aren't the same index.

    Returns (fig, axes) — axes flattened to 1D regardless of n_rows/n_cols.
    """
    rng = np.random.default_rng() if rng is None else rng
    if colors is None:
        colors = KELLY_COLORS[1:]

    label_by_points = {
        label: rng.choice(points, size=min(len(points), max_points_per_label), replace=False)
        for label, points in label_by_points.items()
    }
    labels = list(label_by_points.keys())
    color_map = {label: colors[i % len(colors)] for i, label in enumerate(labels)}

    if figsize is None:
        figsize = (7 * n_cols, 5 * n_rows)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharey=sharey)
    axes = np.array(axes).reshape(-1)

    n_panels = n_rows * n_cols
    labels_per_panel = -(-len(labels) // n_panels) if n_panels > 0 else 0

    for i, ax in enumerate(axes):
        panel_labels = labels[i * labels_per_panel : (i + 1) * labels_per_panel]
        for label in panel_labels:
            ys = label_by_points[label]
            xs = _jittered_x(len(ys), rng)
            ax.scatter(xs, ys, color=color_map[label], label=label)
        if panel_labels:
            ax.legend()
        else:
            ax.axis("off")

    fig.tight_layout()
    return fig, axes


def grid_dims_for_labels(n_labels, clusters_per_panel=8):
    """Heuristic (n_rows, n_cols) for laying out per-label scatter panels: pack
    ~`clusters_per_panel` labels into each panel, then arrange the resulting
    panels into the nearest-square grid (cols = ceil(sqrt(n_panels))). Used by
    save_label_points_grid_jpeg so the caller doesn't have to hand-pick a grid
    for however many clusters a neuron happens to have. n_labels==0 collapses to
    a single (1, 1) panel (show_label_points_grid just turns it off)."""
    n_panels = max(1, math.ceil(n_labels / clusters_per_panel))
    n_cols = math.ceil(math.sqrt(n_panels))
    n_rows = math.ceil(n_panels / n_cols)
    return n_rows, n_cols


def save_label_points_grid_jpeg(
    label_by_points,
    output_path,
    clusters_per_panel=8,
    colors=None,
    figsize=None,
    max_points_per_label=200,
    rng=None,
):
    """Headless (savefig, no plt.show) sibling of show_label_points_grid: dumps
    the per-cluster activation-range scatter grid to a dark-themed JPEG, sizing
    the grid from `clusters_per_panel` via grid_dims_for_labels so the caller
    only supplies the label->points map. Styled to match the rest of the report
    (dark axes + dark legends) so it reads cleanly as the first tab of
    html_report.generate_html_report. label_by_points: dict[label, list[float]]
    (e.g. {cluster_label: [output activations]} for one dep neuron, optionally
    with a "noise" baseline series)."""
    n_rows, n_cols = grid_dims_for_labels(len(label_by_points), clusters_per_panel)
    fig, axes = show_label_points_grid(
        label_by_points,
        n_rows,
        n_cols,
        colors=colors,
        figsize=figsize,
        max_points_per_label=max_points_per_label,
        rng=rng,
    )
    for ax in axes:
        if not ax.axison:  # empty panels are turned off by show_label_points_grid
            continue
        _style_dark_axis(fig, ax)
        legend = ax.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor("#212529")
            legend.get_frame().set_edgecolor("white")
            for text in legend.get_texts():
                text.set_color("white")
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path, format="jpeg", facecolor=fig.get_facecolor(), pil_kwargs={"quality": 90}
    )
    plt.close(fig)


def save_firing_frequency_histogram_jpeg(ratios, output_path, bins=20):
    """
    Report-level histogram of firing frequency: one value per "frequent" dep
    neuron (firing_count / total_examples, see
    report_stats.compute_firing_frequency_ratios) — how large a share of
    origin examples each dep neuron actually fired in. Same exclusion as
    save_output_activation_histogram_jpeg: one_off/outlier neurons (see
    report_stats.split_dep_order_by_frequency) are never included, since
    they were already dropped from `frequent` upstream before this is
    called — kept in sync deliberately, not incidentally. The neuron count
    itself is stated in report_render.render_report_stats_summary rather
    than as in-plot text.
    """
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    _style_dark_axis(fig, ax)
    ax.hist(ratios, bins=bins, color=_HIST_COLOR, edgecolor="none")

    ax.set_xlabel("firing frequency (fraction of examples)", color="white")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="jpeg", facecolor=fig.get_facecolor(), pil_kwargs={"quality": 90})
    plt.close(fig)


def save_fraction_above_threshold_histogram_jpeg(fractions, output_path, threshold, bins=20):
    """
    Report-level histogram: one value per probed origin instance/example — a
    single (input_image_key, origin_y, origin_x), not one whole image, since
    an image can be probed at multiple positions (see
    report_stats.compute_fraction_above_threshold_by_example) — what fraction
    of that example's "frequent" dep-neuron firings sit above `threshold`
    noise-radius units from their own dep neuron's noise_med. threshold is shown in
    the x-axis label (rather than as a vertical line — unlike
    _draw_output_activation_histogram's x=0 marker, there's no fixed
    reference point here since threshold itself is the caller's configurable
    constant, see report_config.ReportConfig.fraction_above_threshold) so a
    reader knows exactly which cutoff produced this distribution without
    checking the report's config separately.
    """
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    _style_dark_axis(fig, ax)
    ax.hist(fractions, bins=bins, color=_HIST_COLOR, edgecolor="none")

    ax.set_xlabel(f"fraction of firings with distance > {threshold:g} (noise-radius units)", color="white")
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
