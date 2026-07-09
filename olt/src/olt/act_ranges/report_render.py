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
