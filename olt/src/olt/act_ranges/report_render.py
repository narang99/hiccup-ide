def render_origin_cluster_header(dep_layer_name, dep_channel, origin_cluster_label, cluster_photo_ref_path):
    """
    Document title + the origin neuron's own cluster photo, placed before
    everything else in the report (see print_report_for_neuron) — the rest
    of the report is all about which dependency neurons explain this
    cluster's firings, so showing what that cluster actually looks like
    up front gives a viewer the "what am I looking at" context before diving
    into cards.

    origin_cluster_label is the (dep_layer_name, dep_channel)'s own cluster
    id the whole report was built for (e.g. the LABEL/cluster_label used to
    filter collect_cluster_stats_df's input rows) — None if the caller didn't
    pass one (print_report_for_neuron's origin_cluster_label param), in which
    case only the heading is shown, no photo section. cluster_photo_ref_path
    is the already-dumped path (report_assets.dump_cluster_asset, reusing the
    same per-cluster photo mechanism dependency clusters use) — None means a
    label was given but no photo was found (e.g. a singleton cluster), so an
    explanatory placeholder is shown instead of an image.
    """
    heading = f"# Report: Dependencies of {dep_layer_name}:{dep_channel}"
    if origin_cluster_label is None:
        return f"{heading}\n"
    if cluster_photo_ref_path is not None:
        body = f"![cid={origin_cluster_label}]({cluster_photo_ref_path})\n"
    else:
        body = (
            '<span class="text-body-secondary small">no cluster photo — this cluster was '
            "built from samples in only a single image, so its combined photo wasn't "
            "generated</span>\n"
        )
    return f"{heading}\n\n{body}"


def render_summary(one_off_count, one_off_threshold, total_examples):
    """
    A short markdown block placed before every card (see print_report_for_neuron):
    states how many dependency neurons were excluded as one-off/low-frequency
    firers (see report_stats.split_dep_order_by_frequency) and the criteria
    used — firing in fewer than one_off_threshold of the total_examples input
    images collected. These excluded neurons aren't rendered as cards at all,
    so this is the only place their existence is surfaced.
    """
    return (
        f'::: {{.text-body-secondary .small .mb-3}}\n'
        f"**{one_off_count}** dependency neuron(s) fired in fewer than "
        f"**{one_off_threshold}** of the **{total_examples}** collected input images "
        f"and are treated as one-off/outliers — not shown below.\n"
        f":::\n"
    )


def render_notes_summary(cluster_notes):
    """
    Collapsible callout placed near the top of the report (see
    print_report_for_neuron), listing every entry in cluster_notes
    (report_config.ReportConfig.cluster_notes: dict[(dep_layer_name,
    dep_channel, dep_cid), str]) in one place, so a note is visible on a
    single skim of the report instead of only when you happen to expand that
    specific cluster's own "show cluster photo" section — where the same
    note is repeated verbatim (see render_cluster_breakdown), so the two
    don't drift apart. Returns "" if cluster_notes is empty/None (nothing to
    show, so no empty callout is emitted).
    """
    if not cluster_notes:
        return ""
    items = "\n".join(
        f"- **{dep_layer_name}:{dep_channel} (cid={dep_cid})** — {note}"
        for (dep_layer_name, dep_channel, dep_cid), note in cluster_notes.items()
    )
    return (
        f'::: {{.callout-note collapse="true"}}\n'
        f"## Cluster notes\n\n"
        f"{items}\n"
        f":::\n"
    )


def render_cluster_breakdown(
    cluster_stats,
    cluster_photo_ref_by_cid=None,
    pw_sample_ref_by_cid=None,
    feature_viz_ref_by_cid=None,
    note_by_cid=None,
):
    """
    Below the Overview scatter: one thin bar per dependency cluster (dep_cid) this
    neuron's firings were matched to (see compute_cluster_breakdown), ordered by
    descending firing count. Bar width encodes that cluster's share of this
    neuron's total firings — these ratios sum to ~100% across the full list of
    bars, since every firing is attributed to exactly one cluster. Bars are
    neutral blue (bg-info), except "outlier" clusters (is_outlier, i.e. below
    compute_cluster_breakdown's outlier_ratio_threshold — too small a share
    of firings to call a real match) which get yellow (bg-warning) instead —
    red is reserved elsewhere in the report for negative-contribution
    coloring (see report_render's ticker labels / plotting's concentration
    sparkline), so outlier clusters need a distinct color to avoid being
    misread as "negative" — with " · outlier" appended to the label, matching
    how save_combined_scatter_jpeg folds these same clusters into its
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
    the row, same as pw_sample_ref_by_cid. note_by_cid: dict[dep_cid, str] —
    this neuron's slice of report_config.ReportConfig.cluster_notes (see
    render_notes_summary for the same notes collected once at the top of the
    report); a cluster with a note gets it rendered first inside the
    `<details>`, in a `{.callout-note}` block for visual consistency with the
    top-of-report summary, and "· has note" appended to the summary line so
    it's spottable without expanding. The `<details>` is omitted entirely
    only when none of a photo, pw sample, feature-viz image, or note is
    available.
    """
    if not cluster_stats:
        return ""
    cluster_photo_ref_by_cid = cluster_photo_ref_by_cid or {}
    pw_sample_ref_by_cid = pw_sample_ref_by_cid or {}
    feature_viz_ref_by_cid = feature_viz_ref_by_cid or {}
    note_by_cid = note_by_cid or {}
    bars = []
    for row in cluster_stats:
        width_pct = row["ratio"] * 100
        color_class = "bg-warning" if row["is_outlier"] else "bg-info"
        label = (
            f"cid={row['dep_cid']} · {row['count']} ({width_pct:.1f}%) · "
            f"median_sim={row['median_similarity']:.2f}"
        )
        if row["is_outlier"]:
            label += " · outlier"
        ref_path = cluster_photo_ref_by_cid.get(row["dep_cid"])
        pw_ref_path = pw_sample_ref_by_cid.get(row["dep_cid"])
        feature_viz_ref_path = feature_viz_ref_by_cid.get(row["dep_cid"])
        note = note_by_cid.get(row["dep_cid"])
        summary_text = "show cluster photo"
        if note is not None:
            summary_text += " · has note"
        if ref_path is None and pw_ref_path is None and feature_viz_ref_path is None and note is None:
            details = (
                '<span class="text-body-secondary small">no cluster photo — this cluster was '
                "built from samples in only a single image, so its combined photo wasn't "
                "generated</span>"
            )
        else:
            body = ""
            if note is not None:
                body += f'::: {{.callout-note}}\n{note}\n:::\n\n'
            if ref_path is not None:
                body += f"![cid={row['dep_cid']}]({ref_path})\n\n"
            else:
                body += (
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
                f'<details><summary class="text-body-secondary small">{summary_text}</summary>\n\n'
                f"{body}"
                "</details>"
            )
        bars.append(
            '<div class="mb-2">'
            '<div class="d-flex align-items-center gap-2 mb-1">'
            f'<div class="progress flex-grow-1" style="height: 6px;" role="progressbar" '
            f'aria-valuenow="{width_pct:.1f}" aria-valuemin="0" aria-valuemax="100">'
            f'<div class="progress-bar {color_class}" style="width: {width_pct:.1f}%;"></div>'
            f"</div>"
            f'<span class="text-body-secondary small">{label}</span>'
            f"</div>\n\n{details}\n\n</div>"
        )
    return '<div class="text-body-secondary small mb-1">Firing by cluster</div>' + "\n\n".join(bars)


def render_overview_tab_body(
    scatter_ref_path,
    cluster_stats,
    cluster_photo_ref_by_cid=None,
    pw_sample_ref_by_cid=None,
    feature_viz_ref_by_cid=None,
    note_by_cid=None,
):
    """Content of a neuron card's body: the combined activation/noise scatter
    plot + a per-cluster firing breakdown (see render_cluster_breakdown), each
    with a single collapsed section showing the cluster photo followed by a
    pointwise-multiplication sample and any cluster_notes entry for that
    cluster, if available. No card wrapper — the card is added by
    render_neuron_card. The contribution-strength and firing-frequency bars
    this used to lead with are gone: the header's concentration sparkline +
    ticker labels already cover contribution
    (report_stats.compute_concentration_curves), and firing frequency is now
    a plain "Fired: {percent}%" figure in the card header (see
    render_neuron_card's firing_pct param) instead of its own bar."""
    breakdown = render_cluster_breakdown(
        cluster_stats, cluster_photo_ref_by_cid, pw_sample_ref_by_cid, feature_viz_ref_by_cid, note_by_cid
    )
    return f"![]({scatter_ref_path})\n\n{breakdown}"


_POS_TEXT_COLOR = "#1b7a34"  # matches plotting._POS_MARKER
_NEG_TEXT_COLOR = "#a11f1f"  # matches plotting._NEG_MARKER


def _format_ticker_label(marker):
    """
    Stock-ticker-style label for one concentration-sparkline panel's marker
    (report_stats.compute_concentration_curves's (rank, cumulative_share,
    delta)): the main number is the cumulative share reached by this rank,
    the parenthetical is this single neuron's own marginal share of the
    total — always shown with an up arrow since cumulative share only ever
    increases. Returns None if marker is None (nothing to show yet for that
    panel — see compute_concentration_curves).
    """
    if marker is None:
        return None
    _rank, cumulative_share, delta = marker
    return f"{cumulative_share * 100:.1f}% (↑{delta * 100:.1f}%)"


def render_neuron_card(
    dep_layer_name,
    dep_channel,
    overview_body,
    concentration_ref_path=None,
    pos_marker=None,
    neg_marker=None,
    firing_pct=None,
):
    """
    One card per (dep_layer, dep_channel) neuron: header naming the neuron —
    plus, if firing_pct is given, a plain "Fired: {percent}%" figure right
    next to the name (replaces the old firing-frequency bar with a single
    number, since a bar was overkill for one scalar) — and, if
    concentration_ref_path is given, that neuron's tiny positive/negative
    concentration sparkline — see
    report_assets.dump_concentration_asset/report_stats.compute_concentration_curves
    — inline next to the name, small enough to sit in the gray header strip
    rather than taking card-body space, flanked by ticker-style text labels —
    _format_ticker_label(pos_marker) in green to its left, negative in red to
    its right, mirroring the sparkline's own positive-left/negative-right
    layout — instead of printing the numbers on the plot itself. Body holds
    the Overview content (render_overview_tab_body) — scatter plot and
    per-cluster firing breakdown, aggregated across every row of stats_df for
    this dep neuron. No per-image breakdown/tabset; that scoping is
    unnecessary now that the card only shows the aggregate view.
    """
    title = f"{dep_layer_name}:{dep_channel}"
    if firing_pct is not None:
        title += f" · Fired: {firing_pct * 100:.1f}%"
    if concentration_ref_path is not None:
        pos_label = _format_ticker_label(pos_marker)
        neg_label = _format_ticker_label(neg_marker)
        pos_span = f'[{pos_label}]{{style="color: {_POS_TEXT_COLOR};"}} ' if pos_label else ""
        neg_span = f' [{neg_label}]{{style="color: {_NEG_TEXT_COLOR};"}}' if neg_label else ""
        # Title and the sparkline group are separate paragraphs (blank line
        # between) so pandoc emits them as two sibling <p> elements inside
        # the header div — only then are they actually two flex items,
        # letting .justify-content-between push the group to the right edge.
        # (A single run-on line renders as one <p>, so flex has nothing to
        # push against — that's why it previously sat wherever the text
        # happened to end instead of the corner.) Within the group itself,
        # the label spans and image are plain inline content, so they just
        # flow left-to-right in source order — no extra flex nesting needed.
        header_body = (
            f"{title}\n\n"
            f"{pos_span}![]({concentration_ref_path}){{height=60px}}{neg_span}"
        )
        header_classes = ".card-header .text-body-secondary .small .d-flex .align-items-center .justify-content-between"
    else:
        header_body = title
        header_classes = ".card-header .text-body-secondary .small"
    return (
        f'::: {{.card .mb-2 .shadow-sm}}\n'
        f'::: {{{header_classes}}}\n'
        f"{header_body}\n"
        f":::\n\n"
        f'::: {{.card-body}}\n\n'
        f"{overview_body}\n\n"
        f":::\n"
        f":::\n"
    )
