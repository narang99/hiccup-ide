import math
from pathlib import Path

from olt.act_ranges.plotting import save_combined_scatter_png
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


def render_activation_bar(relative_strength, output_activation):
    """
    A thin progress bar reflecting relative_strength (float in [0, 1]): this dep
    neuron's output_activation as a fraction of the sum of output_activation across
    all dep neurons firing for this image (sum-norm, not max-abs-norm) — i.e. its
    share of the total. No floor: 0 renders as an empty bar, since 0 really means no
    contribution here. Color encodes sign: green for output_activation >= 0, red for
    < 0. A grayed "{percent}% | {output_activation}" label sits to the right.
    """
    width_pct = relative_strength * 100
    color_class = "bg-success" if output_activation >= 0 else "bg-danger"
    label = f"{relative_strength * 100:.0f}% | {output_activation:.2f}"
    return (
        f'<div class="d-flex align-items-center gap-2">'
        f'<div class="progress flex-grow-1" style="height: 6px;" role="progressbar" '
        f'aria-valuenow="{width_pct:.0f}" aria-valuemin="0" aria-valuemax="100">'
        f'<div class="progress-bar {color_class}" style="width: {width_pct:.0f}%;"></div>'
        f"</div>"
        f'<span class="text-body-secondary small">{label}</span>'
        f"</div>"
    )


def render_overview_tab_body(
    relative_strength, median_output_activation, scatter_ref_path, firing_count, total_examples
):
    """Content of a neuron card's "Overview" tab (the default tab, see
    render_neuron_tabset_card): median-activation bar + combined activation/noise
    scatter plot + a firing-ratio caption. No card/heading wrapper — the card and
    its "### Overview" tab heading are added by render_neuron_tabset_card."""
    bar = render_activation_bar(relative_strength, median_output_activation)
    firing_ratio = firing_count / total_examples if total_examples else 0.0
    caption = f"fired in {firing_count}/{total_examples} examples ({firing_ratio:.0%})"
    return (
        f"{bar}\n\n"
        f"![output_activation (red) vs noise (gray)]({scatter_ref_path})\n\n{caption}"
    )


def render_image_tab_body(dep_cid, noise_distance, output_activation, similarity, ref_path, relative_strength):
    """Content of a neuron card's per-image tab when the neuron fired for that
    image: per-image relative-strength bar + its cluster heatmap. No card/heading
    wrapper — see render_neuron_tabset_card."""
    caption = (
        f"cid={dep_cid} noise_dist={noise_distance:.2f} "
        f"act={output_activation:.2f} sim={similarity:.2f}"
    )
    bar = render_activation_bar(relative_strength, output_activation)
    return f"{bar}\n\n![{caption}]({ref_path})\n\n{caption}"


def render_image_tab_placeholder_body():
    """Content of a neuron card's per-image tab when the neuron has no row for
    that particular image — i.e. it didn't fire for it, even though it does for
    others in the report."""
    return '<span class="text-body-secondary">Did not fire</span>'


def render_neuron_tabset_card(dep_layer_name, dep_channel, overview_tab_body, image_tab_bodies):
    """
    One card per (dep_layer, dep_channel) neuron, containing a Quarto
    panel-tabset scoped to this single card: "Overview" (median stats + scatter
    plot) is the default tab, followed by one tab per input image (labelled by
    index, matching the image_keys order used to build image_tab_bodies). Because
    the tabset lives inside one small card rather than spanning the whole page,
    any scroll/focus jump Bootstrap's tab.js causes on switch is negligible —
    you're already looking at this card, so it doesn't disrupt comparing views of
    the same neuron.

    image_tab_bodies: list of (label, body_markdown) pairs, one per input image
    tab, in display order — body_markdown from render_image_tab_body or
    render_image_tab_placeholder_body.
    """
    tabs = [f"### Overview\n\n{overview_tab_body}"]
    tabs += [f"### {label}\n\n{body}" for label, body in image_tab_bodies]
    tabset_body = "\n\n".join(tabs)
    return (
        f'::: {{.card .mb-2 .shadow-sm}}\n'
        f'::: {{.card-header .text-body-secondary .small}}\n'
        f"{dep_layer_name}:{dep_channel}\n"
        f":::\n\n"
        f'::: {{.card-body}}\n'
        f'::: {{.panel-tabset}}\n\n'
        f"{tabset_body}\n\n"
        f":::\n"
        f":::\n"
        f":::\n"
    )


def dump_overview_assets(assets_dump_dir, dep_layer_name, dep_channel, activations, noise_samples):
    """
    Always (re)writes one combined scatter PNG for one dep neuron's Overview
    card, under {assets_dump_dir}/{dep_layer_name}/{dep_channel}/. Reserved
    filename (parallel to the cluster_ prefix reserved by dump_cluster_asset):
    overview_scatter.png. Unlike dump_cluster_asset, this is always
    regenerated (no exists-check) — cheap to recompute from stats_df each run.
    Returns the dumped Path.
    """
    neuron_dir = assets_dump_dir / dep_layer_name / str(dep_channel)
    scatter_path = neuron_dir / "overview_scatter.png"
    save_combined_scatter_png(list(activations), list(noise_samples), scatter_path)
    return scatter_path


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


def print_report_for_neuron(
    stats_df,
    base_report_dir,
    assets_dump_dir,
    assets_ref_dir,
    layer_by_channel_by_noise,
    max_input_keys,
    output_path=None,
):
    """
    Prints the full Quarto markdown to stdout — copy it into a .qmd file to
    render/test. If output_path is given, also writes it there.

    The report is one card per dependency neuron (see compute_dep_order for
    ordering), split into "Frequently firing" and "One-off / low frequency"
    sections (see split_dep_order_by_frequency). Each card holds its own
    panel-tabset (render_neuron_tabset_card): an "Overview" tab (median-activation
    bar + combined activation/noise scatter plot, aggregated across every row of
    stats_df regardless of max_input_keys or per-image dedup) as the default tab,
    followed by one tab per input image (labelled by index rather than the
    potentially long/unwieldy input_image_key), showing that neuron's per-image
    relative-strength bar and cluster heatmap, or a "Did not fire" placeholder.
    Scoping the tabset to each card (rather than one tabset for the whole page)
    means switching tabs to compare a neuron across images doesn't jump you
    elsewhere on the page.

    stats_df: the concatenation of NeuronParentAnalyser.collect_cluster_stats_df
    outputs across multiple input images, for a single neuron — must have exactly
    one (origin_layer, origin_channel) pair and an "input_image_key" column.
    Only the first `max_input_keys` distinct input_image_key values (in the order
    they first appear) get their own tab, even if stats_df has more.

    layer_by_channel_by_noise: same dict passed into NeuronParentAnalyser, used
    for the Overview tab's raw noise-sample scatter plots.
    """
    origin_layers = stats_df["origin_layer"].unique()
    origin_channels = stats_df["origin_channel"].unique()
    if len(origin_layers) != 1 or len(origin_channels) != 1:
        raise ValueError(
            "stats_df must contain exactly one (origin_layer, origin_channel) pair, got "
            f"origin_layers={list(origin_layers)}, origin_channels={list(origin_channels)}"
        )
    dep_order = compute_dep_order(stats_df)
    deduped_stats_df = dedupe_to_one_origin_per_image(stats_df)
    image_keys = deduped_stats_df["input_image_key"].unique()[:max_input_keys]

    groups = stats_df.groupby(["dep_layer", "dep_channel"])["output_activation"]
    medians = {key: groups.get_group(key).median() for key in dep_order}
    check_same_sign(medians.values())
    median_sum = sum(medians.values())

    firing_counts, total_examples = compute_firing_stats(stats_df)
    frequent, one_off = split_dep_order_by_frequency(dep_order, firing_counts, total_examples)

    image_act_sums = compute_image_act_sums(deduped_stats_df)
    row_by_dep_and_image = {
        (row["dep_layer"], row["dep_channel"], row["input_image_key"]): row
        for _, row in deduped_stats_df.iterrows()
    }

    def render_card(dep_layer_name, dep_channel):
        dep_rows = groups.get_group((dep_layer_name, dep_channel))
        median_output_activation = medians[(dep_layer_name, dep_channel)]
        overview_relative_strength = (
            median_output_activation / median_sum if median_sum != 0 else 0.0
        )
        noise_samples = layer_by_channel_by_noise.get(dep_layer_name, {}).get(
            str(dep_channel), []
        )
        dump_overview_assets(assets_dump_dir, dep_layer_name, dep_channel, dep_rows, noise_samples)
        scatter_ref_path = (
            f"{assets_ref_dir}/{dep_layer_name}/{dep_channel}/overview_scatter.png"
        )
        overview_body = render_overview_tab_body(
            overview_relative_strength,
            median_output_activation,
            scatter_ref_path,
            firing_counts.get((dep_layer_name, dep_channel), 0),
            total_examples,
        )

        image_tabs = []
        for i, image_key in enumerate(image_keys):
            row = row_by_dep_and_image.get((dep_layer_name, dep_channel, image_key))
            dump_path = None
            if row is not None:
                dump_path = dump_cluster_asset(
                    assets_dump_dir, base_report_dir, dep_layer_name, dep_channel, row["dep_cid"]
                )
            if row is None or dump_path is None:
                image_tabs.append((i, render_image_tab_placeholder_body()))
                continue

            act_sum = image_act_sums[image_key]
            relative_strength = row["output_activation"] / act_sum if act_sum != 0 else 0.0
            ref_path = f"{assets_ref_dir}/{dep_layer_name}/{dep_channel}/cluster_{row['dep_cid']}.jpeg"
            image_tabs.append((
                i,
                render_image_tab_body(
                    row["dep_cid"],
                    row["noise_distance"],
                    row["output_activation"],
                    row["similarity"],
                    ref_path,
                    relative_strength,
                ),
            ))

        return render_neuron_tabset_card(dep_layer_name, dep_channel, overview_body, image_tabs)

    sections = []
    if frequent:
        cards = "\n\n".join(render_card(*key) for key in frequent)
        sections.append(f"## Frequently firing\n\n{cards}")
    if one_off:
        cards = "\n\n".join(render_card(*key) for key in one_off)
        sections.append(f"## One-off / low frequency\n\n{cards}")
    content = "\n\n".join(sections)
    print(content)
    if output_path is not None:
        Path(output_path).write_text(content)
