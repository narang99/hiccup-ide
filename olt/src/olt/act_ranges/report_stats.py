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
    examples, otherwise it's a one-off/noise neuron. Order within each
    sub-list is preserved from dep_order (already sorted by descending
    firing_count, with median output_activation as tie-breaker, via
    compute_dep_order).
    """
    threshold = max(min_count, math.ceil(min_ratio * total_examples))
    frequent, one_off = [], []
    for key in dep_order:
        (frequent if firing_counts.get(key, 0) >= threshold else one_off).append(key)
    return frequent, one_off


def dedupe_to_one_origin_per_image(stats_df):
    """
    Keeps only rows matching the first (origin_y, origin_x) seen for each
    input_image_key (in row order) — a given image should contribute one unique
    probe position, but upstream data can end up with more than one, which would
    otherwise show the same dep neuron repeated in a tab.
    """
    first_origin_per_image = stats_df.drop_duplicates(subset="input_image_key", keep="first")[
        ["input_image_key", "origin_y", "origin_x"]
    ]
    return stats_df.merge(first_origin_per_image, on=["input_image_key", "origin_y", "origin_x"])


def compute_per_image_shares(deduped_stats_df, image_act_sums):
    """
    For each row in deduped_stats_df (one firing per dep neuron per image, see
    dedupe_to_one_origin_per_image), computes that neuron's output_activation as a
    share of the image's total (output_activation / image_act_sums[input_image_key],
    see compute_image_act_sums) — the same per-image share render_image_tab_body's
    bar shows for one image, collected here across every image so
    compute_relative_strength_median_per_image_share can take a median over them.

    Returns dict[(dep_layer, dep_channel), list[float]], one share per image the
    neuron fired in.
    """
    shares_by_dep = {}
    for _, row in deduped_stats_df.iterrows():
        act_sum = image_act_sums[row["input_image_key"]]
        share = row["output_activation"] / act_sum if act_sum != 0 else 0.0
        shares_by_dep.setdefault((row["dep_layer"], row["dep_channel"]), []).append(share)
    return shares_by_dep


def compute_relative_strength_median_sum(key, medians, median_sum):
    """
    Overview relative-strength method A ("share of the median"): this dep neuron's
    median output_activation (across every firing in stats_df, see `medians` in
    print_report_for_neuron) as a fraction of median_sum (the sum of every dep
    neuron's median output_activation for this origin neuron). Swappable with
    compute_relative_strength_median_per_image_share via print_report_for_neuron's
    relative_strength_method param — this one first sums then divides, the other
    first divides (per image) then takes a median, and the two can disagree when a
    neuron's activation or the image's total varies a lot across images.
    """
    return medians[key] / median_sum if median_sum != 0 else 0.0


def compute_relative_strength_median_per_image_share(key, per_image_shares):
    """
    Overview relative-strength method B ("median of the shares"): median, across
    every image this dep neuron fired in, of its per-image share of that image's
    total output_activation (see compute_per_image_shares) — i.e. the same quantity
    render_image_tab_body's per-image bar shows, aggregated with a median instead of
    picking one image's tab. Swappable with compute_relative_strength_median_sum via
    print_report_for_neuron's relative_strength_method param.
    """
    shares = per_image_shares.get(key, [])
    return float(np.median(shares)) if shares else 0.0


def compute_dep_order(stats_df):
    """
    Unique (dep_layer, dep_channel) pairs across the whole stats_df, ordered by
    descending firing_count (see compute_firing_stats — how many distinct origin
    instances a dep neuron fired in), with descending median output_activation as
    a tie-breaker — used so every neuron card lists dependency neurons in the
    same order (and so section placement in split_dep_order_by_frequency is
    stable).
    """
    firing_counts, _ = compute_firing_stats(stats_df)
    medians = stats_df.groupby(["dep_layer", "dep_channel"])["output_activation"].median()
    return sorted(
        medians.index.tolist(),
        key=lambda key: (-firing_counts.get(key, 0), -medians[key]),
    )


def check_same_sign(values):
    """
    Raises if `values` contains both a strictly positive and a strictly negative
    entry — 0 is compatible with either sign. Used because output_activation
    fractions-of-sum only make sense (as a "share of total") when every firing dep
    neuron for an image pushed the origin neuron the same direction.
    """
    has_positive = any(v > 0 for v in values)
    has_negative = any(v < 0 for v in values)
    if has_positive and has_negative:
        raise ValueError(
            f"output_activation values must all share the same sign (0 allowed on either side), "
            f"got mixed signs: {list(values)}"
        )


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
    save_combined_scatter_png, which folds them into the "unmatched" bucket
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
    matched ones (see save_combined_scatter_png), so capping keeps rendering
    fast and the plot legible. Sampling uses a fixed seed so re-running report
    generation on unchanged input data reproduces the same plot.

    Returns dict[dep_cid, list[float]], keyed by int(cid) to match
    dep_full_rows["dep_cid"] values, for save_combined_scatter_png.
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


def compute_image_act_sums(deduped_stats_df):
    """
    Sum of output_activation across all dep neurons firing for each input image
    (same-sign-checked per image via check_same_sign), keyed by input_image_key —
    the denominator for each neuron's per-image relative_strength bar. Computed
    once up front so every neuron's card can look up its image's sum, instead of
    each neuron recomputing it from a per-image slice.
    """
    sums = {}
    for image_key, group in deduped_stats_df.groupby("input_image_key")["output_activation"]:
        check_same_sign(group)
        sums[image_key] = group.sum()
    return sums
