import math

import numpy as np

from olt.act_ranges.stats import get_noise_range, noise_stats


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


def compute_output_activation_noise_max_distances(stats_df, layer_by_channel_by_noise, frequent):
    """
    Report-level (not per-card) pool of (output_activation - noise_max) /
    noise_radius across every firing row belonging to a "frequent" dep neuron
    (see split_dep_order_by_frequency) — the same population that gets its
    own card, so the histogram this feeds (report_assets.dump_output_activation_histogram_asset)
    matches what a reader sees below it. One-off/low-frequency dep neurons
    (excluded from cards entirely) are excluded here too.

    Each firing is normalized against its OWN (dep_layer, dep_channel)'s
    noise stats (see stats.noise_stats) — noise_max/noise_radius vary
    per dep neuron, since different neurons can have wildly different raw
    activation scales, so pooling raw activations across neurons would be
    meaningless. Expressing every firing as "how many noise-radii above (or
    below) this neuron's own noise ceiling" makes them directly comparable in
    one histogram. noise_stats is computed once per (dep_layer, dep_channel)
    group, not once per row, since it involves an O(n^2) shorth call.

    Returns a flat list[float], one per included row, in no particular order.
    """
    frequent = set(frequent)
    distances = []
    for (dep_layer_name, dep_channel), group in stats_df.groupby(["dep_layer", "dep_channel"]):
        if (dep_layer_name, dep_channel) not in frequent:
            continue
        noise = layer_by_channel_by_noise[dep_layer_name][str(dep_channel)]
        _, _, noise_max, noise_radius = noise_stats(noise)
        distances.extend((group["output_activation"] - noise_max) / noise_radius)
    return distances


def compute_firing_frequency_ratios(firing_counts, total_examples, frequent):
    """
    Report-level pool of firing_count / total_examples, one value per
    "frequent" dep neuron (see split_dep_order_by_frequency) — the same
    population whose firings feed compute_output_activation_noise_max_distances
    and get their own card, so this histogram's exclusion matches that one's:
    one-off/low-frequency dep neurons are excluded here too.

    Returns a flat list[float] of length len(frequent), in no particular
    order.
    """
    return [firing_counts[key] / total_examples for key in frequent]


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
      one entry per key in dep_order. Exactly one of pos_marker/neg_marker is
      ever non-None for a given key — whichever side that neuron's own value
      falls on — and the other is always None; a key never shows a marker for
      a sign it didn't itself contribute to. The non-None one is (rank,
      cumulative_share, delta), where cumulative_share/delta are fractions of
      THAT SIDE'S OWN total |value| — not a combined total across both signs
      — so cumulative_share reaches 100% at each side's own last-ranked
      neuron, independently. delta is that single rank's own marginal share
      of its side's total (cumulative_share minus the previous rank's, or
      cumulative_share itself at rank 1). Used for both the printed ticker
      label and the plot's vertical marker line (rank) — deliberately *not*
      "frozen" from a previous card of the other sign: the ticker is meant to
      read as "what did this neuron itself change," so a card must never show
      a stale value for a sign it isn't part of.
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
    for k in dep_order:
        if k in pos_rank:
            marker_by_key[k] = (marker_at(pos_curve, pos_rank[k], pos_total), None)
        elif k in neg_rank:
            marker_by_key[k] = (None, marker_at(neg_curve, neg_rank[k], neg_total))
        else:
            marker_by_key[k] = (None, None)

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
    (aka "ACTS_DICT" in the exploration notebooks) — down to at most
    max_points_per_cluster points each. A dep neuron can have far
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


def compute_noise_radius_table(dep_full_rows, noise, label_by_points, cluster_stats):
    """
    One row per (dep_layer, dep_channel) card's "distance from noise" table:
    how far a representative value and that row's own shorth-based low end
    sit from this neuron's own noise_max, expressed in units of noise_radius
    (see stats.noise_stats — 1 unit = 1 noise_radius, so e.g. 2.0 means "two
    noise-radii above noise_max").

    Row 1 is always "output activation": dep_full_rows["output_activation"]
    — every firing collected for this dep neuron in this report. One further
    row per matched, non-outlier cluster in cluster_stats (see
    compute_cluster_breakdown — outliers are excluded, same convention as
    save_combined_scatter_jpeg's Kelly-colored set): label_by_points[str(dep_cid)]
    — that cluster's own full population, not just the subset of its points
    that happened to match a firing here.

    Each row has:
    - "med-noise_max": (median(values) - noise_max) / noise_radius
    - "min-noise_max": (values_min - noise_max) / noise_radius, where
      values_min is that row's own shorth-based lower bound — same params as
      noise's own shorth-based range (see stats.get_noise_range, frac=0.9).
    noise_max/noise_radius are the same for every row here, since they all
    belong to one dep neuron's own noise sample.

    Returns list[dict]: [{"label": "output activation", "med-noise_max": float,
    "min-noise_max": float}, {"label": "cid=<dep_cid>", ...}, ...], in the
    same order as cluster_stats (already sorted by descending firing count).
    """
    _, _, noise_max, noise_radius = noise_stats(noise)

    def _row(label, values):
        values = np.asarray(values)
        value_med = np.median(values)
        value_min, _ = get_noise_range(values)
        return {
            "label": label,
            "med-noise_max": (value_med - noise_max) / noise_radius,
            "min-noise_max": (value_min - noise_max) / noise_radius,
        }

    rows = [_row("output activation", dep_full_rows["output_activation"])]
    for cluster_row in cluster_stats:
        if cluster_row["is_outlier"]:
            continue
        dep_cid = cluster_row["dep_cid"]
        rows.append(_row(f"cid={dep_cid}", label_by_points[str(dep_cid)]))
    return rows


