import math

import torch
from PIL import Image, ImageDraw

from olt.act_ranges.constants import CLUSTER_PATCH_SET_DIR, LAYER_NAME_BY_SHAPE
from olt.act_ranges.plotting import save_combined_scatter_png
from olt.act_ranges.report_stats import select_cluster_points
from olt.act_ranges.reports import get_cluster_photo
from olt.act_ranges.similarity import closest_patch_index, load_cluster_patches
from olt.html_report import _fit_and_pad, apply_cmap
from olt.show import get_local_image_limits, rd_bk_gn


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


def render_pw_grid_jpeg(images, titles, out_path, ncols, cell_px=200, pad=4, bg="#2c2e33"):
    """
    Headless PIL grid renderer for dump_pw_sample_asset's asset: images is a
    flat row-major list of length nrows*ncols (2 rows here: wild samples then
    their per-sample closest matches, same column = same wild sample), where
    None stands in for a blank cell — used to pad a cluster's row out to
    ncols when it recorded fewer than ncols wild samples, keeping every
    cluster's grid the same width.

    Reuses html_report.apply_cmap/_fit_and_pad — the same PIL-only path
    html_report.py already uses for its "third" per-cluster overlay view —
    instead of matplotlib's figure/axes machinery
    (show.save_single_channel_red_green_black), since this asset is dumped
    once per (neuron, matched cluster) and the plotting overhead dominates at
    that volume.
    """
    nrows = math.ceil(len(images) / ncols)
    title_h = 20
    total_w = ncols * cell_px + (ncols - 1) * pad
    total_h = nrows * (cell_px + title_h) + (nrows - 1) * pad
    canvas = Image.new("RGB", (total_w, total_h), color="#141517")
    draw = ImageDraw.Draw(canvas)
    for i, img in enumerate(images):
        row, col = divmod(i, ncols)
        x = col * (cell_px + pad)
        y = row * (cell_px + title_h + pad)
        if img is None:
            cell = Image.new("RGB", (cell_px, cell_px), color=bg)
        else:
            arr = img.detach().cpu().numpy() if isinstance(img, torch.Tensor) else img
            vmin, vmax = get_local_image_limits(arr)
            oh, ow = arr.shape[:2]
            scale = min(cell_px / ow, cell_px / oh)
            fitted = (int(ow * scale), int(oh * scale))
            cell = apply_cmap(
                arr, rd_bk_gn, vmin=vmin, vmax=vmax, size=fitted, interpolation=Image.NEAREST
            )
            cell = _fit_and_pad(cell, cell_px, cell_px, bg=bg)
        canvas.paste(cell, (x, y))
        if titles and i < len(titles) and titles[i]:
            draw.text(
                (x + cell_px // 2, y + cell_px + title_h // 2),
                titles[i],
                fill="#ffffff",
                anchor="mm",
            )
    canvas.save(out_path, format="JPEG", quality=95)


def dump_pw_sample_asset(
    assets_dump_dir,
    dep_layer_name,
    dep_channel,
    dep_cid,
    pw_samples,
    cluster_patch_set_dir=CLUSTER_PATCH_SET_DIR,
    max_samples=5,
):
    """
    Writes (or reuses, if already present) one JPEG grid with up to
    max_samples "wild" pointwise-multiplication samples (pw_samples: entries
    from layer_by_channel_by_cid_by_pw_samples[dep_layer_name][str(dep_channel)][str(dep_cid)],
    see NeuronParentAnalyser.collect_cluster_stats_df) in the top row, each
    paired directly below it (same column) with its own closest match from
    the matched cluster's stored patch population — the per-wild-sample
    nearest neighbor by w-weighted cosine similarity
    (similarity.closest_patch_index), not a shared top-k list taken in
    storage order (which is what this used to show, and isn't actually
    "closest" to anything). Columns beyond len(pw_samples) are left blank
    rather than omitted, so every cluster's grid is max_samples wide.

    best_patches (the matched cluster's full patch population) isn't carried
    on pw_samples — it's loaded fresh here via similarity.load_cluster_patches,
    since it's identical for every sample sharing this dep_cid and shouldn't
    be duplicated per firing.

    Written under {assets_dump_dir}/{dep_layer_name}/{dep_channel}/pw_{dep_cid}.jpeg
    — parallel to dump_cluster_asset's cluster_{cid}.jpeg. Returns the dumped
    Path.
    """
    neuron_dir = assets_dump_dir / dep_layer_name / str(dep_channel)
    neuron_dir.mkdir(parents=True, exist_ok=True)
    dump_path = neuron_dir / f"pw_{dep_cid}.jpeg"
    if dump_path.exists():
        return dump_path

    best_patches = load_cluster_patches(
        dep_layer_name, dep_channel, int(dep_cid), cluster_patch_set_dir
    )
    shape = LAYER_NAME_BY_SHAPE[dep_layer_name]

    samples = pw_samples[:max_samples]
    wild_images, match_images, wild_titles, match_titles = [], [], [], []
    for i in range(max_samples):
        if i < len(samples):
            dep_w = samples[i]["dep_w"]
            dep_patch = samples[i]["dep_patch"]
            match_idx = closest_patch_index(dep_w, best_patches, dep_patch)
            wild_images.append((dep_w * dep_patch).reshape(shape))
            match_images.append((dep_w * best_patches[match_idx]).reshape(shape))
            wild_titles.append(f"wild #{i + 1}")
            match_titles.append(f"match #{i + 1}")
        else:
            wild_images.append(None)
            match_images.append(None)
            wild_titles.append(None)
            match_titles.append(None)

    render_pw_grid_jpeg(
        wild_images + match_images, wild_titles + match_titles, dump_path, ncols=max_samples
    )
    return dump_path


def dump_overview_assets(
    assets_dump_dir,
    dep_layer_name,
    dep_channel,
    dep_full_rows,
    noise_samples,
    label_by_points,
    matched_cids,
    max_points_per_cluster=50,
):
    """
    Always (re)writes one combined scatter PNG for one dep neuron's Overview
    card, under {assets_dump_dir}/{dep_layer_name}/{dep_channel}/. Reserved
    filename (parallel to the cluster_ prefix reserved by dump_cluster_asset):
    overview_scatter.png. Unlike dump_cluster_asset, this is always
    regenerated (no exists-check) — cheap to recompute from stats_df each run.

    dep_full_rows: the full stats_df rows for this (dep_layer, dep_channel) —
    needs "output_activation".
    label_by_points: this (dep_layer, dep_channel)'s slice of
    layer_by_channel_by_label_by_points (dict[str(cid), list[float]]) — used
    via select_cluster_points to plot every one of this dep neuron's clusters
    (not just matched ones) alongside the activation/noise scatter, each
    subsampled to at most max_points_per_cluster points (see
    save_combined_scatter_png).
    matched_cids: the dep_cid values to Kelly-color/give their own legend
    entry — the caller excludes outlier clusters (see
    compute_cluster_breakdown's is_outlier) so a cluster that only accounts
    for a tiny share of firings doesn't get treated as a real match here.

    Returns the dumped Path.
    """
    neuron_dir = assets_dump_dir / dep_layer_name / str(dep_channel)
    scatter_path = neuron_dir / "overview_scatter.png"
    cluster_points_by_cid = select_cluster_points(label_by_points, max_points_per_cluster)
    save_combined_scatter_png(
        list(dep_full_rows["output_activation"]),
        list(noise_samples),
        cluster_points_by_cid,
        matched_cids,
        scatter_path,
    )
    return scatter_path
