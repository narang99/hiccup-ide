import math

import numpy as np

from olt.act_ranges.plotting import show_output_activation_histograms
from olt.act_ranges.stats import noise_stats


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
    minimum firing count, surfaced so callers (see render_report_stats_summary)
    can state the actual criteria used rather than just the counts.
    """
    threshold = max(min_count, math.ceil(min_ratio * total_examples))
    frequent, one_off = [], []
    for key in dep_order:
        (frequent if firing_counts.get(key, 0) >= threshold else one_off).append(key)
    return frequent, one_off, threshold


SORT_ORDERS = ("median", "firing_frequency")


def sort_card_order(frequent, firing_counts, sort_order="median"):
    """
    Orders the "frequent" dep neurons (see split_dep_order_by_frequency) for
    card display — report_config.ReportConfig.sort_order picks between:

    - "median" (default): `frequent`'s incoming order as-is, i.e. descending
      |median contribution| (see compute_dep_order) — unchanged from before
      this function existed.
    - "firing_frequency": descending firing_counts (see compute_firing_stats),
      so the dep neurons that fire in the most examples lead the report
      instead of the ones with the single largest contribution. Python's sort
      is stable, so neurons tied on firing count keep their relative "median"
      order as a tie-breaker, rather than an arbitrary one.

    Deliberately separate from dep_order/compute_dep_order itself: dep_order's
    descending-|median| ordering is also relied on by
    compute_concentration_curves (its Lorenz-style cumulative curves assume
    each sign's keys already arrive ranked by descending magnitude) — reusing
    that same list for card display, rather than re-sorting it in place, keeps
    the concentration sparkline's own ranking/shape independent of whatever
    order cards happen to be displayed in.

    Returns a list, same elements as `frequent`, reordered.
    """
    if sort_order == "median":
        return list(frequent)
    if sort_order == "firing_frequency":
        return sorted(frequent, key=lambda key: -firing_counts.get(key, 0))
    raise ValueError(f"unknown sort_order: {sort_order!r}, expected one of {SORT_ORDERS}")


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


def _frequent_distances_with_contribution(stats_df, layer_by_channel_by_noise, frequent):
    """
    Shared groupby/noise-normalization pass behind
    compute_output_activation_noise_median_distances,
    split_output_activation_distances_by_contribution_sign, and
    compute_fraction_above_threshold_by_example — every firing row belonging
    to a "frequent" dep neuron (see split_dep_order_by_frequency), tupled as
    (distance, contribution, input_image_key, origin_y, origin_x):

    - distance: (output_activation - noise_med) / noise_radius, this firing's
      own dep neuron's noise stats (see stats.noise_stats) — noise_med/
      noise_radius vary per dep neuron, since different neurons can have
      wildly different raw activation scales, so pooling raw activations
      across neurons would be meaningless. Expressing every firing as "how
      many noise-radii above (or below) this neuron's own noise median"
      makes them directly comparable in one histogram. Distance from the
      plain median (not noise_max, the shorth-based upper edge of the noise
      band) since "N units above/below noise_max" reads unintuitively once N
      goes negative (e.g. a firing sitting a bit below noise_max but still
      well above typical noise) — median is a single, always-meaningful
      reference point in both directions.
    - contribution: this firing's own patch*weight value at the origin
      neuron's contributing index (see
      NeuronParentAnalyser.collect_cluster_stats_df's "contribution" column)
      — positive means this firing added to the origin neuron's signal,
      negative/zero means it inhibited it. Distinct from the sign of
      `distance` itself (how far this dep neuron's own activation sits from
      its own noise) — a firing can be clearly above its own noise median
      while still inhibiting the origin neuron, or vice versa.
    - input_image_key, origin_y, origin_x: together identify which origin
      instance (single probed (image, y, x) position — origin_layer/
      origin_channel are constant across a single stats_df, see
      _validate_single_origin) this firing came from — unused by the two
      distance-pooling callers, but needed by
      compute_fraction_above_threshold_by_example to group firings back by
      origin instance rather than by image alone (an image can be probed at
      more than one (y, x) position, and those are distinct examples, not
      one).

    noise_stats is computed once per (dep_layer, dep_channel) group, not once
    per row, since it involves an O(n^2) shorth call.

    Returns a flat list[tuple[float, float, object, object, object]], one per
    included row, in no particular order.
    """
    frequent = set(frequent)
    rows = []
    for (dep_layer_name, dep_channel), group in stats_df.groupby(["dep_layer", "dep_channel"]):
        if (dep_layer_name, dep_channel) not in frequent:
            continue
        noise = layer_by_channel_by_noise[dep_layer_name][str(dep_channel)]
        _, noise_med, _, noise_radius = noise_stats(noise)
        distance = (group["output_activation"] - noise_med) / noise_radius
        rows.extend(
            zip(distance, group["contribution"], group["input_image_key"], group["origin_y"], group["origin_x"])
        )
    return rows


def compute_output_activation_noise_median_distances(stats_df, layer_by_channel_by_noise, frequent):
    """
    Report-level (not per-card) pool of (output_activation - noise_med) /
    noise_radius across every firing row belonging to a "frequent" dep neuron
    — the same population that gets its own card, so the histogram this
    feeds (report_assets.dump_output_activation_histogram_asset) matches
    what a reader sees below it. See _frequent_distances_with_contribution
    for the per-row computation.

    Returns a flat list[float], one per included row, in no particular order.
    """
    return [
        distance
        for distance, *_ in _frequent_distances_with_contribution(stats_df, layer_by_channel_by_noise, frequent)
    ]


def split_output_activation_distances_by_contribution_sign(stats_df, layer_by_channel_by_noise, frequent):
    """
    Splits compute_output_activation_noise_median_distances's pooled population
    by the sign of each firing's own "contribution" (patch*weight value, see
    _frequent_distances_with_contribution) instead of pooling every firing
    together — contribution > 0 means this firing was adding to the origin
    neuron's signal, contribution <= 0 means it was inhibiting it. This lets
    a reader compare "how far above/below noise do adding firings sit" against
    "how far above/below noise do inhibiting firings sit" separately, rather
    than one combined histogram where both kinds of firing are indistinguishable
    (see report_render.render_report_histograms).

    Returns (adding_distances, inhibiting_distances), each a flat list[float]
    in noise-radius units, split from the same pool
    compute_output_activation_noise_median_distances would return combined.
    """
    rows = _frequent_distances_with_contribution(stats_df, layer_by_channel_by_noise, frequent)
    adding = [distance for distance, contribution, *_ in rows if contribution > 0]
    inhibiting = [distance for distance, contribution, *_ in rows if contribution <= 0]
    return adding, inhibiting


def _fraction_above_threshold_by_example(rows, threshold):
    """
    Shared per-origin-instance aggregation behind
    compute_fraction_above_threshold_by_example and
    split_fraction_above_threshold_by_contribution_sign: given a flat list of
    (distance, contribution, input_image_key, origin_y, origin_x) rows (see
    _frequent_distances_with_contribution — contribution is unused here, the
    caller has already filtered/split rows by its sign if it wants that),
    groups by (input_image_key, origin_y, origin_x) — one probed origin
    instance/example, not one whole image, since a single image can be probed
    at more than one (y, x) position and those shouldn't be pooled together
    — and returns one fraction per example: the share of that example's rows
    whose distance exceeds `threshold`.

    Returns a flat list[float], one per distinct (input_image_key, origin_y,
    origin_x) present in rows, in no particular order. An example with no
    rows in the input simply has no entry (nothing to divide by), rather than
    a spurious 0.0.
    """
    distances_by_example = {}
    for distance, _, image_key, origin_y, origin_x in rows:
        distances_by_example.setdefault((image_key, origin_y, origin_x), []).append(distance)
    return [
        sum(1 for d in distances if d > threshold) / len(distances)
        for distances in distances_by_example.values()
    ]


def compute_fraction_above_threshold_by_example(stats_df, layer_by_channel_by_noise, frequent, threshold):
    """
    Report-level pool of one value per probed origin instance/example (a
    single (input_image_key, origin_y, origin_x) — not one whole image, since
    an image can be probed at multiple positions and those are distinct
    examples): the fraction of that example's "frequent"-dep-neuron firings
    (same population as compute_output_activation_noise_median_distances) whose
    noise-median distance exceeds `threshold` (in noise-radius units — same
    units as that function's output; threshold can be negative, e.g. -0.8,
    since it's a signed offset from noise_med, not from a one-sided ceiling
    — a reader may as easily want to ask about a cutoff sitting a bit below
    the median as one above it).
    Feeds report_assets.dump_fraction_above_threshold_histogram_asset's
    histogram — one bar per example showing what share of its firings ran hot
    (or, depending on threshold's sign, merely "not clearly quiet") relative
    to their own dep neuron's noise, so a reader can spot whether that's
    evenly spread across examples or concentrated in a few.

    Returns a flat list[float] of length equal to the number of distinct
    (input_image_key, origin_y, origin_x) examples with at least one
    frequent-dep-neuron firing, in no particular order.
    """
    rows = _frequent_distances_with_contribution(stats_df, layer_by_channel_by_noise, frequent)
    return _fraction_above_threshold_by_example(rows, threshold)


def split_fraction_above_threshold_by_contribution_sign(stats_df, layer_by_channel_by_noise, frequent, threshold):
    """
    Splits compute_fraction_above_threshold_by_example's per-example
    population by the sign of each firing's own "contribution" (patch*weight
    value, see _frequent_distances_with_contribution), mirroring
    split_output_activation_distances_by_contribution_sign for the fraction-
    above-threshold histogram: contribution > 0 firings ("adding") and
    contribution <= 0 firings ("inhibiting") are grouped and fractioned
    independently, so an example's "adding" fraction is out of that example's
    own adding-firing count, not out of its total firing count.

    An example that has rows of only one sign contributes a fraction for that
    sign's list only — same "no entry rather than a spurious 0.0" rule as
    _fraction_above_threshold_by_example, applied per sign.

    Returns (adding_fractions, inhibiting_fractions), each a flat list[float].
    """
    rows = _frequent_distances_with_contribution(stats_df, layer_by_channel_by_noise, frequent)
    adding_rows = [row for row in rows if row[1] > 0]
    inhibiting_rows = [row for row in rows if row[1] <= 0]
    return (
        _fraction_above_threshold_by_example(adding_rows, threshold),
        _fraction_above_threshold_by_example(inhibiting_rows, threshold),
    )


def show_output_activation_noise_histograms(stats_df, layer_by_channel_by_noise, bins=40, figsize=(14, 3.5)):
    """
    One-call interactive plot of the three output-activation-vs-noise
    histograms print_report_for_neuron renders in its report row (see
    report_render.render_report_histograms) — the all/adding/inhibiting
    distance splits (compute_output_activation_noise_median_distances,
    split_output_activation_distances_by_contribution_sign) computed
    directly from stats_df + layer_by_channel_by_noise, then handed to
    plotting.show_output_activation_histograms. For quick notebook
    exploration of one stats_df's noise distances only — does not touch
    print_report_for_neuron/build_report_data, so no assets are dumped and
    no cards are built.

    Deliberately does NOT filter to "frequent" dep neurons the way the full
    report does (compute_firing_stats, split_dep_order_by_frequency) — every
    (dep_layer, dep_channel) pair in stats_df is included. compute_firing_stats
    assumes (via check_at_most_one_firing_per_origin) that a dep neuron fires
    at most once per origin instance, which only holds for a 1x1-kernel
    current_layer (e.g. mixed4e_1x1_pre_relu_conv) — for a current_layer with
    a larger kernel (e.g. mixed5b's 5x5 branch), the same dep channel can
    legitimately be a top contributor at more than one spatial position
    within a single origin instance, so that check would reject perfectly
    valid data. Since this function only cares about the noise-distance
    distributions (not firing-frequency filtering), it skips that machinery
    entirely rather than requiring an invariant it doesn't need.

    stats_df: see NeuronParentAnalyser.collect_cluster_stats_df — same shape
    print_report_for_neuron expects.
    layer_by_channel_by_noise: same dict passed into NeuronParentAnalyser.

    Returns (fig, axs).
    """
    dep_neurons = compute_dep_order(stats_df)

    all_distances = compute_output_activation_noise_median_distances(stats_df, layer_by_channel_by_noise, dep_neurons)
    adding_distances, inhibiting_distances = split_output_activation_distances_by_contribution_sign(
        stats_df, layer_by_channel_by_noise, dep_neurons
    )
    return show_output_activation_histograms(
        all_distances, adding_distances, inhibiting_distances, bins=bins, figsize=figsize
    )


def compute_firing_frequency_ratios(firing_counts, total_examples, frequent):
    """
    Report-level pool of firing_count / total_examples, one value per
    "frequent" dep neuron (see split_dep_order_by_frequency) — the same
    population whose firings feed compute_output_activation_noise_median_distances
    and get their own card, so this histogram's exclusion matches that one's:
    one-off/low-frequency dep neurons are excluded here too.

    Returns a flat list[float] of length len(frequent), in no particular
    order.

    Not currently wired into print_report_for_neuron's output (the firing-
    frequency histogram was pulled from the report row to make room for the
    positive/negative output-activation split) — kept here since it may come
    back in some form later.
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


