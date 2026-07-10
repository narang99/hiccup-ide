import math

import numpy as np


def check_at_most_one_firing_per_origin(stats_df):
    """
    Raises if any (dep_layer, dep_channel) appears more than once for the same
    origin instance (origin_layer, origin_channel, origin_y, origin_x,
    input_image_key). current_layer_name is always a 1x1 conv (see
    NeuronParentAnalyser class docstring), so receptive_block collapses to a
    single spatial point and a given dep channel can only be a top contributor
    once per origin — a violation means stats_df has duplicate/corrupted rows
    upstream (e.g. the same (image, y, x) probe collected into stats_df more
    than once), not a real second firing, and firing-frequency counts would
    silently be wrong if we didn't catch it here.
    """
    origin_cols = ["origin_layer", "origin_channel", "origin_y", "origin_x", "input_image_key"]
    dupe_key_cols = ["dep_layer", "dep_channel"] + origin_cols
    counts = stats_df.groupby(dupe_key_cols).size()
    violations = counts[counts > 1]
    if len(violations) > 0:
        sample = violations.head(5).reset_index(name="row_count")
        raise ValueError(
            "compute_firing_stats: found dep neuron(s) appearing more than once for "
            "the same origin instance, which should be impossible for a 1x1 conv "
            "current_layer (each dep channel can only be a top contributor once per "
            "origin). This means stats_df has duplicate rows for that (dep_layer, "
            "dep_channel, origin_layer, origin_channel, origin_y, origin_x, "
            "input_image_key) — check how stats_df was assembled (e.g. the same "
            "image/probe collected twice). "
            f"{len(violations)} violating combination(s), showing up to 5:\n{sample}"
        )


def compute_firing_stats(stats_df):
    """
    Counts how many distinct origin instances each dep neuron fired in, across
    EVERY row of stats_df. Assumes (and validates via
    check_at_most_one_firing_per_origin) that a dep neuron appears at most once
    per origin instance, so firing_count <= total_examples always holds.

    total_examples: number of distinct (origin_layer, origin_channel,
    origin_y, origin_x, input_image_key) tuples in stats_df — the denominator
    for firing ratios. input_image_key is included so that two different
    images which happen to probe the same (origin_layer, origin_channel,
    origin_y, origin_x) are counted as separate examples, not collapsed into
    one.

    Returns (firing_counts: dict[(dep_layer, dep_channel), int], total_examples: int).
    """
    check_at_most_one_firing_per_origin(stats_df)
    origin_cols = ["origin_layer", "origin_channel", "origin_y", "origin_x", "input_image_key"]
    total_examples = stats_df[origin_cols].drop_duplicates().shape[0]
    firing_counts = stats_df.groupby(["dep_layer", "dep_channel"]).size().to_dict()
    return firing_counts, total_examples


def split_dep_order_by_frequency(
    dep_order, firing_counts, total_examples, min_ratio=0.1, min_count=2
):
    """
    Splits dep_order into (frequent, one_off) based on each dep neuron's
    firing count relative to total_examples: a neuron is "frequent" if it
    fired in at least max(min_count, ceil(min_ratio * total_examples))
    examples, otherwise it's a one-off/outlier neuron. Order within each
    sub-list is preserved from dep_order (already sorted by descending
    |median contribution|, with signed median contribution as tie-breaker,
    via compute_dep_order).

    Returns (frequent, one_off, threshold) — threshold is the computed
    minimum firing count, surfaced so callers (see render_summary) can state
    the actual criteria used rather than just the counts.
    """
    threshold = max(min_count, math.ceil(min_ratio * total_examples))
    frequent, one_off = [], []
    for key in dep_order:
        (frequent if firing_counts.get(key, 0) >= threshold else one_off).append(key)
    return frequent, one_off, threshold


def compute_dep_order(stats_df):
    """
    Unique (dep_layer, dep_channel) pairs across the whole stats_df, ordered by
    descending |median contribution| (this dep neuron's own patch*weight
    contribution to the origin neuron — see analyser.top_contributing_indices —
    not firing frequency; a neuron that fires rarely but with a huge
    contribution each time should surface before one that fires often but
    weakly), with descending median contribution (signed) as a tie-breaker —
    used so every neuron card lists dependency neurons in the same order (and
    so section placement in split_dep_order_by_frequency is stable).
    """
    medians = stats_df.groupby(["dep_layer", "dep_channel"])["contribution"].median()
    return sorted(
        medians.index.tolist(),
        key=lambda key: (-abs(medians[key]), -medians[key]),
    )


CONCENTRATION_METRICS = ("median", "weighted")


def compute_concentration_values(medians, firing_counts, total_examples, metric="median"):
    """
    Per-dep-neuron scalar value compute_concentration_curves builds the
    concentration sparkline/ticker labels from — kept swappable via `metric`
    (report_config.ReportConfig.concentration_metric) so different weighting
    strategies can be compared side by side rather than committing to one:

    - "median": medians[key] as-is — this dep neuron's median contribution
      across only the examples it actually fired in. Simple, but can
      "overquantify" a neuron that fires rarely: a large-but-rare per-firing
      contribution counts exactly the same in the cumulative curve as one
      from a neuron that fires in every example, even though the rare one
      contributes far less to the report's examples overall.
    - "weighted": medians[key] * firing_counts.get(key, 0) / total_examples —
      scales the median down by how often this neuron actually fires (see
      compute_firing_stats), so an infrequently-firing neuron's rare large
      contributions no longer inflate its apparent share of the total.

    Returns dict[(dep_layer, dep_channel), float], same keys as medians,
    signed (sign is preserved/unaffected — "weighted" only scales magnitude).
    """
    if metric == "median":
        return dict(medians)
    if metric == "weighted":
        return {
            key: median * firing_counts.get(key, 0) / total_examples if total_examples else 0.0
            for key, median in medians.items()
        }
    raise ValueError(
        f"unknown concentration_metric: {metric!r}, expected one of {CONCENTRATION_METRICS}"
    )


def compute_concentration_curves(values, dep_order):
    """
    Builds the data behind each card's tiny positive/negative "concentration"
    sparkline (report_assets.dump_concentration_asset): a Lorenz-style
    cumulative curve per sign, so a viewer isn't misled into treating the
    single brightest/widest bar as representative — the sparkline shows how
    much magnitude it actually took to get there, and how many more neurons
    make up the rest.

    values: dict[(dep_layer, dep_channel), float], signed, one scalar per
    dep neuron — see compute_concentration_values for how this is built
    (swappable "median" vs "weighted" metric); this function itself doesn't
    care which metric produced it, it just splits by sign and cumulates.

    dep_order neurons are split by sign of their value (0 is excluded from
    both — neither positive nor negative) and, within each sign, ranked by
    descending |value| (dep_order is already sorted that way overall, so a
    stable filter preserves per-sign rank order).

    pos_curve/neg_curve are RAW cumulative sums of |value| — not
    normalized/divided by anything. The plot (save_concentration_sparkline_jpeg)
    is what makes the two sides comparable, via matplotlib's sharey=True: the
    shared y-axis autoscales to whichever side's cumulative sum is larger, so
    a real magnitude difference between the two sides still shows up as a
    height difference, without needing a shared denominator baked into the
    numbers themselves. This is deliberately decoupled from the printed
    "{cumulative}% (^{delta}%)" ticker label (report_render._format_ticker_label)
    next to the sparkline — that label is its own percentage calculation (see
    marker_at below) and has no bearing on what's plotted; the plot is just
    the values.

    Returns (pos_curve, neg_curve, marker_by_key):
    - pos_curve, neg_curve: list[float], raw cumulative |value| after each
      rank (index 0 is rank 1), one per positive/negative neuron
      respectively — these are what's actually plotted.
    - marker_by_key: dict[(dep_layer, dep_channel), (pos_marker, neg_marker)],
      one entry per key in dep_order. pos_marker/neg_marker are each either
      None (that sign hasn't appeared yet as of this point in dep_order) or
      (rank, cumulative_share, delta), where cumulative_share/delta are
      fractions of THAT SIDE'S OWN total |value| — not a combined total
      across both signs — so cumulative_share reaches 100% at each side's
      own last-ranked neuron, independently. Used only for the printed
      ticker label, not for anything plotted. rank is still used by the plot
      (as an x position for the vertical marker line), but the share/delta
      values are not. delta is that single rank's own marginal share of its
      side's total (cumulative_share minus the previous rank's, or
      cumulative_share itself at rank 1). For a key that's itself positive,
      pos_marker is exactly its own point (it just updated the running
      position); neg_marker is whatever negative neuron was most recently
      seen above it, frozen in place — so scrolling through cards keeps both
      curves' "where are we" markers current even on cards belonging to the
      other sign.
    """
    pos_keys = [k for k in dep_order if values[k] > 0]
    neg_keys = [k for k in dep_order if values[k] < 0]

    def raw_curve(keys):
        cum = 0.0
        points = []
        for k in keys:
            cum += abs(values[k])
            points.append(cum)
        return points

    pos_curve = raw_curve(pos_keys)
    neg_curve = raw_curve(neg_keys)
    pos_total = pos_curve[-1] if pos_curve else 0.0
    neg_total = neg_curve[-1] if neg_curve else 0.0
    pos_rank = {k: i for i, k in enumerate(pos_keys)}
    neg_rank = {k: i for i, k in enumerate(neg_keys)}

    def marker_at(curve, i, total):
        cum = curve[i]
        prev = curve[i - 1] if i > 0 else 0.0
        share = cum / total if total else 0.0
        delta = (cum - prev) / total if total else 0.0
        return (i + 1, share, delta)

    marker_by_key = {}
    last_pos, last_neg = None, None
    for k in dep_order:
        if k in pos_rank:
            last_pos = marker_at(pos_curve, pos_rank[k], pos_total)
        elif k in neg_rank:
            last_neg = marker_at(neg_curve, neg_rank[k], neg_total)
        marker_by_key[k] = (last_pos, last_neg)

    return pos_curve, neg_curve, marker_by_key


def compute_cluster_breakdown(dep_full_rows, outlier_ratio_threshold=0.1):
    """
    Splits one dep neuron's firings (dep_full_rows: the full stats_df rows for a
    single (dep_layer, dep_channel), one row per origin instance it fired in — see
    check_at_most_one_firing_per_origin) by which dependency cluster (dep_cid) each
    firing was closest to. For each dep_cid: how many firings landed there (count),
    that as a share of this neuron's total firings (ratio), the median
    "similarity" column value among those firings (how closely, on average, the
    firing point matched that cluster), and whether it's an "outlier" — a
    cluster that only accounts for a small slice (ratio < outlier_ratio_threshold)
    of this neuron's firings, and therefore not representative enough to call a
    real match (see render_cluster_breakdown, which flags these, and
    save_combined_scatter_jpeg, which folds them into the "unmatched" bucket
    instead of giving them their own Kelly color/legend entry).

    Returns a list of dicts sorted by descending count (ties broken by descending
    median_similarity): [{"dep_cid", "count", "ratio", "median_similarity",
    "is_outlier"}, ...].
    """
    total = len(dep_full_rows)
    rows = []
    for cid, group in dep_full_rows.groupby("dep_cid"):
        ratio = len(group) / total if total else 0.0
        rows.append(
            {
                "dep_cid": cid,
                "count": len(group),
                "ratio": ratio,
                "median_similarity": group["similarity"].median(),
                "is_outlier": ratio < outlier_ratio_threshold,
            }
        )
    rows.sort(key=lambda r: (-r["count"], -r["median_similarity"]))
    return rows


def select_cluster_points(label_by_points, max_points_per_cluster=50):
    """
    Subsamples every cluster in label_by_points — dict[str(cid), list[float]],
    i.e. layer_by_channel_by_label_by_points for one (dep_layer, dep_channel)
    (aka "ACTS_DICT" in the exploration notebooks), the same per-cluster point
    collection NeuronParentAnalyser.plot_clusters uses for its axes[0] — down
    to at most max_points_per_cluster points each. A dep neuron can have far
    more clusters (and far more points per cluster) than the ones it actually
    matched, and the Overview scatter now plots every cluster, not just
    matched ones (see save_combined_scatter_jpeg), so capping keeps rendering
    fast and the plot legible. Sampling uses a fixed seed so re-running report
    generation on unchanged input data reproduces the same plot.

    Returns dict[dep_cid, list[float]], keyed by int(cid) to match
    dep_full_rows["dep_cid"] values, for save_combined_scatter_jpeg.
    """
    rng = np.random.default_rng(0)
    sampled = {}
    for cid_str, points in label_by_points.items():
        points = list(points)
        if len(points) > max_points_per_cluster:
            idx = rng.choice(len(points), size=max_points_per_cluster, replace=False)
            points = [points[i] for i in idx]
        sampled[int(cid_str)] = points
    return sampled


