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


def render_frequency_bar(firing_count, total_examples):
    """
    A thin progress bar showing how often this dep neuron fired: firing_count out
    of total_examples origin instances (see compute_firing_stats). Same visual
    style as render_activation_bar, but neutral blue (bg-info) throughout since
    firing frequency isn't signed the way output_activation is. A grayed
    "{percent}% | fired {count}/{total}" label sits to the right.
    """
    ratio = firing_count / total_examples if total_examples else 0.0
    width_pct = ratio * 100
    label = f"{ratio * 100:.0f}% | fired {firing_count}/{total_examples}"
    return (
        f'<div class="d-flex align-items-center gap-2">'
        f'<div class="progress flex-grow-1" style="height: 6px;" role="progressbar" '
        f'aria-valuenow="{width_pct:.0f}" aria-valuemin="0" aria-valuemax="100">'
        f'<div class="progress-bar bg-info" style="width: {width_pct:.0f}%;"></div>'
        f"</div>"
        f'<span class="text-body-secondary small">{label}</span>'
        f"</div>"
    )


def render_cluster_breakdown(
    cluster_stats,
    cluster_photo_ref_by_cid=None,
    pw_sample_ref_by_cid=None,
    feature_viz_ref_by_cid=None,
):
    """
    Below the Overview scatter: one thin bar per dependency cluster (dep_cid) this
    neuron's firings were matched to (see compute_cluster_breakdown), ordered by
    descending firing count. Bar width encodes that cluster's share of this
    neuron's total firings — unlike render_activation_bar's relative_strength,
    these ratios sum to ~100% across the full list of bars, since every firing is
    attributed to exactly one cluster. Bars are neutral blue (bg-info), except
    "outlier" clusters (is_outlier, i.e. below compute_cluster_breakdown's
    outlier_ratio_threshold — too small a share of firings to call a real match)
    which get red (bg-danger) instead, with " · outlier" appended to the label,
    matching how save_combined_scatter_png folds these same clusters into its
    "unmatched" bucket rather than giving them a Kelly color. Each bar's label
    carries the cluster id, its raw count/share, and its median "similarity"
    score (how tightly, on average, firings landing in that cluster matched it)
    so both "which cluster does this neuron mostly fire with" and "how
    confidently" are visible at a glance without a separate table.

    Each bar is followed by a single collapsed `<details>` section (closed by
    default, so it doesn't blow up page length across every card/cluster at
    once) titled "show cluster photo", revealing on expand the cluster's
    heatmap photo followed by up to 5 "wild" pointwise-multiplication samples
    for this cluster (real firings that matched it), each paired with its own
    closest match from the cluster itself — see dump_pw_sample_asset.
    cluster_photo_ref_by_cid: dict[dep_cid, ref_path]
    (ref_path relative to the .qmd, as used elsewhere for image markdown, e.g.
    dump_cluster_asset's output path rebased under assets_ref_dir) — a cluster
    missing from this dict (photo unavailable, typically a singleton cluster
    built from only one image, see get_cluster_photo) gets an explanatory
    placeholder in place of the photo. pw_sample_ref_by_cid: dict[dep_cid,
    ref_path], same ref_path convention; a cluster missing from this dict (no
    firing recorded a sample for it, e.g. its only firings came from a
    stats_df built without pw_samples) simply omits the pointwise-
    multiplication image, with no placeholder needed. feature_viz_ref_by_cid:
    dict[dep_cid, ref_path], same ref_path convention — see
    feature_viz.dump_feature_viz_assets; a cluster missing from this dict
    (print_report_for_neuron was called without `model`, or this cluster's
    dep_layer_name isn't in feature_viz.SUPPORTED_DEP_LAYER_NAMES) simply omits
    the row, same as pw_sample_ref_by_cid. The `<details>` is omitted entirely
    only when none of a photo, pw sample, or feature-viz image is available.
    """
    if not cluster_stats:
        return ""
    cluster_photo_ref_by_cid = cluster_photo_ref_by_cid or {}
    pw_sample_ref_by_cid = pw_sample_ref_by_cid or {}
    feature_viz_ref_by_cid = feature_viz_ref_by_cid or {}
    bars = []
    for row in cluster_stats:
        width_pct = row["ratio"] * 100
        color_class = "bg-danger" if row["is_outlier"] else "bg-info"
        label = (
            f"cid={row['dep_cid']} · {row['count']} ({width_pct:.0f}%) · "
            f"median_sim={row['median_similarity']:.2f}"
        )
        if row["is_outlier"]:
            label += " · outlier"
        ref_path = cluster_photo_ref_by_cid.get(row["dep_cid"])
        pw_ref_path = pw_sample_ref_by_cid.get(row["dep_cid"])
        feature_viz_ref_path = feature_viz_ref_by_cid.get(row["dep_cid"])
        if ref_path is None and pw_ref_path is None and feature_viz_ref_path is None:
            details = (
                '<span class="text-body-secondary small">no cluster photo — this cluster was '
                "built from samples in only a single image, so its combined photo wasn't "
                "generated</span>"
            )
        else:
            if ref_path is not None:
                body = f"![cid={row['dep_cid']}]({ref_path})\n\n"
            else:
                body = (
                    '<span class="text-body-secondary small">no cluster photo — this cluster was '
                    "built from samples in only a single image, so its combined photo wasn't "
                    "generated</span>\n\n"
                )
            if pw_ref_path is not None:
                body += (
                    f"![cid={row['dep_cid']} wild samples vs. each one's closest cluster match]({pw_ref_path})\n\n"
                )
            if feature_viz_ref_path is not None:
                body += (
                    f"![cid={row['dep_cid']} feature-viz reconstruction of the wild samples above]"
                    f"({feature_viz_ref_path})\n\n"
                )
            details = (
                "<details><summary class=\"text-body-secondary small\">show cluster photo</summary>\n\n"
                f"{body}"
                "</details>"
            )
        bars.append(
            '<div class="mb-2">'
            '<div class="d-flex align-items-center gap-2 mb-1">'
            f'<div class="progress flex-grow-1" style="height: 6px;" role="progressbar" '
            f'aria-valuenow="{width_pct:.0f}" aria-valuemin="0" aria-valuemax="100">'
            f'<div class="progress-bar {color_class}" style="width: {width_pct:.0f}%;"></div>'
            f"</div>"
            f'<span class="text-body-secondary small">{label}</span>'
            f"</div>\n\n{details}\n\n</div>"
        )
    return '<div class="text-body-secondary small mb-1">Firing by cluster</div>' + "\n\n".join(bars)


def render_overview_tab_body(
    relative_strength,
    median_output_activation,
    scatter_ref_path,
    firing_count,
    total_examples,
    cluster_stats,
    cluster_photo_ref_by_cid=None,
    pw_sample_ref_by_cid=None,
    feature_viz_ref_by_cid=None,
):
    """Content of a neuron card's "Overview" tab (the default tab, see
    render_neuron_tabset_card): an activation-strength bar + a firing-frequency
    bar (render_frequency_bar) + combined activation/noise scatter plot + a
    per-cluster firing breakdown (see render_cluster_breakdown), each with a
    single collapsed section showing the cluster photo followed by a
    pointwise-multiplication sample, if available. No card/heading wrapper —
    the card and its "### Overview" tab heading are added by
    render_neuron_tabset_card."""
    activation_bar = render_activation_bar(relative_strength, median_output_activation)
    frequency_bar = render_frequency_bar(firing_count, total_examples)
    breakdown = render_cluster_breakdown(
        cluster_stats, cluster_photo_ref_by_cid, pw_sample_ref_by_cid, feature_viz_ref_by_cid
    )
    return (
        f"{activation_bar}\n\n{frequency_bar}\n\n"
        f"![output_activation (red) vs noise (gray)]({scatter_ref_path})\n\n"
        f"{breakdown}"
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
