"""
Feature visualization for a dependency neuron's own pointwise-multiplication
("wild") samples — see dump_pw_sample_asset in report_assets.py, which shows
dep_w * dep_patch (the dependency neuron's own conv weight against its own
receptive-field input) for a handful of real firings. This module answers
"what does feature-viz think the model is thinking" for those same firings:
via torch-lucent's render_vis, it optimizes an image from noise so that, when
run through the real model, the dependency neuron's own pointwise-
multiplication response at that firing's position matches the real one — a
reconstruction, not a crop. One render_vis call per wild sample (CPU, no
batching) — see dump_feature_viz_asset.

Scope: only supports dep_layer_name values that are branches of the mixed4d
block (see constants/branches.py's MIXED4D_BRANCHES), i.e. the dependencies of
current_layer_name="mixed4e_1x1_pre_relu_conv" — the only current_layer
NeuronParentAnalyser itself supports today (see olt/CLAUDE.md). Unlike
NeuronParentAnalyser.dep_coords (which inverts coordinates backwards through
an unknown number of ops and gives up at a maxpool — see
constants/paddings.py), this module only ever replays ops *forward*, since we
already know the exact pad/pool sequence from lucent's InceptionV1.forward.
That means it isn't blocked by the maxpool branch that blocks dep_coords —
see _DEP_LAYER_HOOK_SOURCE below, which covers all four mixed4d branches.
"""

import torch.nn.functional as F
from lucent.optvis import objectives, param, render
from PIL import Image, ImageDraw
from tqdm import tqdm

from olt.act_ranges.layer_utils import (
    ReceptiveFieldOutOfBounds,
    get_layer_params,
    receptive_block,
)


def _pad_zero(pad):
    return lambda x: F.pad(x, pad)


def _pad_then_maxpool(pad, ksize, stride):
    def prep(x):
        x = F.pad(x, pad, value=float("-inf"))
        return F.max_pool2d(x, kernel_size=ksize, stride=stride, padding=0)

    return prep


# dep_layer_name -> (hook_layer_name, prep_fn). hook_layer_name is the lucent-hookable
# submodule whose *output* is (before prep_fn) the tensor dep_layer_name's own conv
# reads; prep_fn replays whatever pad/pool sits between that output and the conv
# (None if the conv reads it directly). Traced from lucent's InceptionV1.forward:
#
#   mixed4d_1x1_pre_relu_conv           = self.mixed4d_1x1_pre_relu_conv(mixed4c)
#   mixed4d_3x3_bottleneck_pre_relu_conv = self.mixed4d_3x3_bottleneck_pre_relu_conv(mixed4c)
#   ...bottleneck... -> relu -> F.pad(x, (1,1,1,1)) -> mixed4d_3x3_pre_relu_conv
#   ...bottleneck... -> relu -> F.pad(x, (2,2,2,2)) -> mixed4d_5x5_pre_relu_conv
#   mixed4d_pool_pad = F.pad(mixed4c, (1,1,1,1), value=-inf)
#   mixed4d_pool = self.mixed4d_pool(mixed4d_pool_pad, kernel_size=3, stride=1, padding=0)
#   mixed4d_pool_reduce_pre_relu_conv = self.mixed4d_pool_reduce_pre_relu_conv(mixed4d_pool)
_DEP_LAYER_HOOK_SOURCE = {
    "mixed4d_1x1_pre_relu_conv": ("mixed4c", None),
    "mixed4d_3x3_pre_relu_conv": ("mixed4d_3x3_bottleneck", _pad_zero((1, 1, 1, 1))),
    "mixed4d_5x5_pre_relu_conv": ("mixed4d_5x5_bottleneck", _pad_zero((2, 2, 2, 2))),
    "mixed4d_pool_reduce_pre_relu_conv": (
        "mixed4c",
        _pad_then_maxpool((1, 1, 1, 1), ksize=3, stride=1),
    ),
}

# dep_layer_name values dump_feature_viz_assets/dump_feature_viz_asset can handle —
# callers (e.g. report_data.py) should filter to this set before building a
# `clusters` list, rather than relying on the ValueError, so an unsupported branch
# just silently gets no feature-viz asset instead of failing the whole report.
SUPPORTED_DEP_LAYER_NAMES = frozenset(_DEP_LAYER_HOOK_SOURCE.keys())


@objectives.wrap_objective()
def _pw_objective(
    hook_layer_name,
    prep_fn,
    dep_weight,
    pw,
    position,
    ksize,
    stride,
    padding,
    batch=None,
):
    """One render_vis call's objective: drives the image parameterization so
    that dep_layer_name's own pointwise-multiplication response at `position`
    matches `pw` (a single wild sample's dep_w * dep_patch)."""

    @objectives.handle_batch(batch)
    def inner(model):
        o = model(hook_layer_name)
        if prep_fn is not None:
            o = prep_fn(o)
        y0, y1 = receptive_block(
            position[0], ksize[0], stride[0], padding[0], input_size=o.shape[-2]
        )
        x0, x1 = receptive_block(
            position[1], ksize[1], stride[1], padding[1], input_size=o.shape[-1]
        )
        block = o[:1, :, y0:y1, x0:x1].reshape(-1) * dep_weight
        return F.mse_loss(block, pw)

    return inner


def _dump_path_for(assets_dump_dir, dep_layer_name, dep_channel, dep_cid):
    neuron_dir = assets_dump_dir / dep_layer_name / str(dep_channel)
    neuron_dir.mkdir(parents=True, exist_ok=True)
    return neuron_dir / f"featureviz_{dep_cid}.jpeg"


def _render_one(
    model,
    hook_layer_name,
    prep_fn,
    dep_weight,
    sample,
    ksize,
    stride,
    padding,
    image_size,
    thresholds,
):
    pw = (sample["dep_w"] * sample["dep_patch"]).reshape(-1)
    position = (sample["dep_y"], sample["dep_x"])
    objective = _pw_objective(
        hook_layer_name, prep_fn, dep_weight, pw, position, ksize, stride, padding
    )
    param_f = lambda: param.image(image_size, batch=1)
    rendered = render.render_vis(
        model,
        objective,
        param_f=param_f,
        thresholds=thresholds,
        show_image=False,
        progress=False,
    )
    return rendered[-1][0]  # [H, W, C] float in [0, 1]; drop the batch dim


def _render_row_jpeg(images, titles, out_path, cell_px=128, pad=4, bg="#141517"):
    """Pastes images (RGB, [H, W, C] float in [0, 1]; None for a blank cell)
    side by side in a single row, each with a caption below it."""
    ncols = len(images)
    title_h = 20
    total_w = ncols * cell_px + (ncols - 1) * pad
    total_h = cell_px + title_h
    canvas = Image.new("RGB", (total_w, total_h), color=bg)
    draw = ImageDraw.Draw(canvas)
    for i, image in enumerate(images):
        x = i * (cell_px + pad)
        if image is None:
            cell = Image.new("RGB", (cell_px, cell_px), color=bg)
        else:
            cell = Image.fromarray((image * 255).astype("uint8")).resize(
                (cell_px, cell_px)
            )
        canvas.paste(cell, (x, 0))
        if titles and titles[i]:
            draw.text(
                (x + cell_px // 2, cell_px + title_h // 2),
                titles[i],
                fill="#ffffff",
                anchor="mm",
            )
    canvas.save(out_path, format="JPEG", quality=95)


def dump_feature_viz_asset(
    assets_dump_dir,
    model,
    dep_layer_name,
    dep_channel,
    dep_cid,
    pw_samples,
    max_samples=5,
    image_size=64,
    thresholds=(128,),
):
    """
    Writes (or reuses, if already present) one JPEG row for a matched
    dependency cluster: up to max_samples feature-viz reconstructions, one
    render_vis call per wild sample (see _render_one) — not jointly optimized
    — so each column reconstructs that one sample's own dep_w * dep_patch
    response, the same wild sample dump_pw_sample_asset's row shows for real
    crops. Columns beyond len(pw_samples) are left blank, so every cluster's
    row is max_samples wide, matching dump_pw_sample_asset's grid. A sample
    whose receptive field falls outside dep_layer_name's own hooked tensor
    (layer_utils.ReceptiveFieldOutOfBounds — a boundary position, not
    expected for the one current_layer this module supports) is left blank
    the same way, rather than failing the whole cluster's row.

    Expensive (max_samples full gradient-based optimization loops), so
    deliberately cached like dump_cluster_asset/dump_pw_sample_asset: if the
    output file already exists, it's reused rather than regenerated.

    pw_samples: entries from layer_by_channel_by_cid_by_pw_samples[dep_layer_name]
    [str(dep_channel)][str(dep_cid)] (dicts with "dep_patch", "dep_w", "dep_y",
    "dep_x" — see NeuronParentAnalyser.collect_cluster_stats_df).

    Runs on whatever device `model` is already on — CPU by default, no
    batching, and deliberately modest defaults (image_size, thresholds) since
    this now runs one render per sample instead of one per cluster.

    Raises ValueError if dep_layer_name isn't one of mixed4d's branches this
    module knows how to replay the pad/pool for (see _DEP_LAYER_HOOK_SOURCE) —
    i.e. only supports dependencies of current_layer_name=
    "mixed4e_1x1_pre_relu_conv" today, mirroring NeuronParentAnalyser's own
    single-current-layer scope (see olt/CLAUDE.md).
    """
    if dep_layer_name not in _DEP_LAYER_HOOK_SOURCE:
        raise ValueError(
            f"dump_feature_viz_asset does not know how to reconstruct dep_layer_name={dep_layer_name!r}'s "
            "own conv input — see _DEP_LAYER_HOOK_SOURCE in feature_viz.py."
        )

    dump_path = _dump_path_for(assets_dump_dir, dep_layer_name, dep_channel, dep_cid)
    if dump_path.exists():
        return dump_path

    hook_layer_name, prep_fn = _DEP_LAYER_HOOK_SOURCE[dep_layer_name]
    w, ksize, stride, padding = get_layer_params(model, dep_layer_name, dep_channel)
    dep_weight = w.reshape(-1)

    samples = pw_samples[:max_samples]
    images, titles = [], []
    for i in range(max_samples):
        image, title = None, None
        if i < len(samples):
            try:
                image = _render_one(
                    model,
                    hook_layer_name,
                    prep_fn,
                    dep_weight,
                    samples[i],
                    ksize,
                    stride,
                    padding,
                    image_size,
                    thresholds,
                )
                title = f"fv #{i + 1}"
            except ReceptiveFieldOutOfBounds as e:
                print(
                    f"WARN: skipping feature-viz sample #{i + 1} for {dep_layer_name}:{dep_channel} "
                    f"cid={dep_cid}: {e}"
                )
        images.append(image)
        titles.append(title)

    _render_row_jpeg(images, titles, dump_path)
    return dump_path


def dump_feature_viz_assets(
    assets_dump_dir,
    model,
    clusters,
    max_samples=5,
    image_size=64,
    thresholds=(128,),
):
    """
    Thin loop over dump_feature_viz_asset: writes one feature-viz JPEG row per
    entry in `clusters`. Kept as the entry point report_data.py calls (so it
    can hand over every matched cluster on a report page at once) without
    needing every caller to loop and juggle dep_layer_name/dep_channel keys
    itself.

    clusters: list of dicts, one per matched dependency cluster:
    {"dep_layer_name", "dep_channel", "dep_cid", "pw_samples"} — pw_samples is
    the same list dump_pw_sample_asset takes (dicts with "dep_patch", "dep_w",
    "dep_y", "dep_x").

    Returns dict[(dep_layer_name, dep_channel, dep_cid), Path] for every
    cluster passed in, in the same order as `clusters`. Cached clusters
    (dump_path already exists) are skipped by dump_feature_viz_asset itself,
    but still advance the tqdm bar below — each tick is one cluster done
    (cached or freshly rendered), not one gradient step.
    """
    results = {}
    for c in tqdm(clusters, desc="Feature-viz clusters"):
        dep_layer_name, dep_channel, dep_cid = (
            c["dep_layer_name"],
            c["dep_channel"],
            c["dep_cid"],
        )
        dump_path = dump_feature_viz_asset(
            assets_dump_dir,
            model,
            dep_layer_name,
            dep_channel,
            dep_cid,
            c["pw_samples"],
            max_samples=max_samples,
            image_size=image_size,
            thresholds=thresholds,
        )
        results[(dep_layer_name, dep_channel, dep_cid)] = dump_path
    return results
