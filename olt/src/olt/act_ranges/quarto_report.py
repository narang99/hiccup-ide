from pathlib import Path

from olt.act_ranges.report_assets import dump_cluster_asset, dump_overview_assets
from olt.act_ranges.report_render import (
    render_image_tab_body,
    render_image_tab_placeholder_body,
    render_neuron_tabset_card,
    render_overview_tab_body,
)
from olt.act_ranges.report_stats import (
    check_same_sign,
    compute_dep_order,
    compute_firing_stats,
    compute_image_act_sums,
    dedupe_to_one_origin_per_image,
    split_dep_order_by_frequency,
)


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
