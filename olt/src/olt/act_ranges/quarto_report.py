from pathlib import Path

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


def render_dep_neuron_card(
    dep_layer_name,
    dep_channel,
    dep_cid,
    noise_distance,
    output_activation,
    similarity,
    ref_path,
    relative_strength,
):
    caption = (
        f"cid={dep_cid} noise_dist={noise_distance:.2f} "
        f"act={output_activation:.2f} sim={similarity:.2f}"
    )
    bar = render_activation_bar(relative_strength, output_activation)
    return (
        f'::: {{.card .mb-2 .shadow-sm}}\n'
        f'::: {{.card-header .text-body-secondary .small}}\n'
        f"{dep_layer_name}:{dep_channel}\n\n"
        f"{bar}\n"
        f":::\n\n"
        f'::: {{.card-body}}\n'
        f"![{caption}]({ref_path})\n\n{caption}\n"
        f":::\n"
        f":::\n"
    )


def render_dep_neuron_placeholder_card(dep_layer_name, dep_channel):
    """Card for a dep neuron that has no row for this particular image — i.e. it
    didn't fire for this image, even though it does for others in the report."""
    return (
        f'::: {{.card .mb-2}}\n'
        f'::: {{.card-header .text-body-secondary .small}}\n'
        f"{dep_layer_name}:{dep_channel}\n"
        f":::\n\n"
        f'::: {{.card-body .text-body-secondary}}\n'
        f"Did not fire\n"
        f":::\n"
        f":::\n"
    )


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
    descending median output_activation — used so every image tab lists dependency
    neurons in the same order, including ones that didn't fire for that image.
    """
    return (
        stats_df.groupby(["dep_layer", "dep_channel"])["output_activation"]
        .median()
        .sort_values(ascending=False)
        .index.tolist()
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


def render_neuron_block(dep_order, image_stats_df, base_report_dir, assets_dump_dir, assets_ref_dir):
    """
    Renders one card per (dep_layer, dep_channel) in `dep_order` (see
    compute_dep_order), using `image_stats_df` (one image's rows, as produced by
    NeuronParentAnalyser.collect_cluster_stats_df) to fill each card in — a dep
    neuron with no row here gets a "Did not fire" placeholder card instead of a
    heatmap. No wrapping card for the origin (current) neuron — the report writer
    adds a header for that manually, since it's the same for the whole report.

    assets_dump_dir: real filesystem directory to write asset files under.
    assets_ref_dir: the path/prefix string used inside the generated markdown image
    links — may differ from assets_dump_dir (e.g. once this report is copied
    elsewhere with a different relative asset path).
    """
    row_by_dep = {
        (row["dep_layer"], row["dep_channel"]): row for _, row in image_stats_df.iterrows()
    }
    check_same_sign(image_stats_df["output_activation"])
    act_sum = image_stats_df["output_activation"].sum()

    cards = []
    for dep_layer_name, dep_channel in dep_order:
        row = row_by_dep.get((dep_layer_name, dep_channel))
        if row is None:
            cards.append(render_dep_neuron_placeholder_card(dep_layer_name, dep_channel))
            continue

        dep_cid = row["dep_cid"]
        dump_path = dump_cluster_asset(
            assets_dump_dir, base_report_dir, dep_layer_name, dep_channel, dep_cid
        )
        if dump_path is None:
            cards.append(render_dep_neuron_placeholder_card(dep_layer_name, dep_channel))
            continue

        relative_strength = row["output_activation"] / act_sum if act_sum != 0 else 0.0
        ref_path = f"{assets_ref_dir}/{dep_layer_name}/{dep_channel}/cluster_{dep_cid}.jpeg"
        cards.append(
            render_dep_neuron_card(
                dep_layer_name,
                dep_channel,
                dep_cid,
                row["noise_distance"],
                row["output_activation"],
                row["similarity"],
                ref_path,
                relative_strength,
            )
        )

    return "\n\n".join(cards)


def render_image_tab(tab_label, neuron_blocks):
    body = "\n\n".join(neuron_blocks)
    return (
        f"## {tab_label}\n\n"
        f'::: {{style="max-height: 85vh; overflow-y: auto;"}}\n\n'
        f"{body}\n\n"
        f":::\n"
    )


def render_report(image_tabs):
    body = "\n\n".join(image_tabs)
    return f"::: {{.panel-tabset}}\n\n{body}\n\n:::\n"


def print_report_for_neuron(
    stats_df, base_report_dir, assets_dump_dir, assets_ref_dir, max_input_keys, output_path=None
):
    """
    Prints the full Quarto markdown (one tab per input image, labelled by index
    rather than the (potentially long/unwieldy) input_image_key) to stdout — copy
    it into a .qmd file to render/test. If output_path is given, also writes it there.

    stats_df: the concatenation of NeuronParentAnalyser.collect_cluster_stats_df
    outputs across multiple input images, for a single neuron — must have exactly
    one (origin_layer, origin_channel) pair and an "input_image_key" column.
    Only the first `max_input_keys` distinct input_image_key values (in the order
    they first appear) are rendered, even if stats_df has more.
    """
    origin_layers = stats_df["origin_layer"].unique()
    origin_channels = stats_df["origin_channel"].unique()
    if len(origin_layers) != 1 or len(origin_channels) != 1:
        raise ValueError(
            "stats_df must contain exactly one (origin_layer, origin_channel) pair, got "
            f"origin_layers={list(origin_layers)}, origin_channels={list(origin_channels)}"
        )
    stats_df = dedupe_to_one_origin_per_image(stats_df)
    dep_order = compute_dep_order(stats_df)
    image_keys = stats_df["input_image_key"].unique()[:max_input_keys]

    image_tabs = [
        render_image_tab(
            i,
            [
                render_neuron_block(
                    dep_order,
                    stats_df[stats_df["input_image_key"] == image_key],
                    base_report_dir,
                    assets_dump_dir,
                    assets_ref_dir,
                )
            ],
        )
        for i, image_key in enumerate(image_keys)
    ]
    content = render_report(image_tabs)
    print(content)
    if output_path is not None:
        Path(output_path).write_text(content)
