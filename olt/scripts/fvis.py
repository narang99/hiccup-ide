import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd
from lucent.modelzoo import inceptionv1
from lucent.optvis import objectives, render
from PIL import Image
from torch.nn import functional as F
from tqdm import tqdm

from olt.act import InputOutputModelSnapshot
from olt.tfms import transform

TARGET_THUMBNAIL_H = 300

TARGET_THUMBNAIL_W = 300

FLAT_IMAGE_DIR = Path(
    "/Users/hariomnarang/Desktop/personal/hiccup-ide/olt/notebooks/this-and-prev/flat-images"
)

device = "cpu"
model = inceptionv1(pretrained=True)
model = model.to(device)
model = model.eval()


@objectives.wrap_objective()
def patch_across_channel_with_ksize(
    weight, layer, patches, positions, ksize, batch=None
):
    @objectives.handle_batch(batch)
    def inner(model):
        o = model(layer)
        mse = 0
        for patch, pos in zip(patches, positions):
            y, x = pos
            y1 = y + ksize[0]
            x1 = x + ksize[1]
            # print(patch.shape, o[:, :, y:y1, x:x1].shape, weight.shape)
            # pw0 = o[:, :, y:y1, x:x1] * weight
            # pw1 = patch * weight
            pw0 = o[:, :, y:y1, x:x1]
            pw1 = patch
            mse += F.mse_loss(pw0, pw1)
        return mse

    return inner


def _get_uniq_imagenet_labels(current_cluster_df, n_samples):
    uniq_imagenet_labels = current_cluster_df.imagenet_label.unique()
    uniq_imagenet_labels = np.random.choice(
        uniq_imagenet_labels, min(len(uniq_imagenet_labels), n_samples)
    )
    return uniq_imagenet_labels


def _receptive_block(i, ksize, stride, padding, input_size=None):
    """
    Returns [start, end) input indices (end=exclusive) that influence
    output position i of a conv layer.
    """
    start = i * stride - padding
    end = start + ksize

    if input_size is not None:
        start = max(start, 0)
        end = min(end, input_size)

    return start, end


def _to_int_image(arr):
    arr = np.asarray(arr, dtype=np.float32)
    arr -= arr.min()
    if arr.max() > 0:
        arr /= arr.max()
    arr = (arr * 255).astype(np.uint8)
    return arr


def _get_patch(pil, model, layer_name, y, x, ksize, stride, padding):
    timg = transform(pil)[None]
    acts = InputOutputModelSnapshot.get_activations(timg, model, [layer_name])
    our_act = acts[layer_name]["input"]

    start_y, end_y = _receptive_block(y, ksize[0], stride[0], padding[0])
    start_x, end_x = _receptive_block(x, ksize[1], stride[1], padding[1])
    one_patch = our_act[:, :, start_y:end_y, start_x:end_x]

    return one_patch, (start_y, end_y), (start_x, end_x)


def _do_one_row(row, model, prev_layer_name, ksize, stride, padding, thresholds=(256,)):
    ikey = row.input_image_key
    pil = Image.open(FLAT_IMAGE_DIR / f"{ikey}.jpeg")

    one_patch, (start_y, _), (start_x, _) = _get_patch(
        pil,
        model,
        row.layer_name,
        row.y_position,
        row.x_position,
        ksize,
        stride,
        padding,
    )

    weight = model.get_submodule(row.layer_name).weight[row.channel].detach().cpu()
    print(weight.shape)
    obj = patch_across_channel_with_ksize(
        weight, prev_layer_name, [one_patch], [(start_y, start_x)], ksize
    )

    svizs = render.render_vis(model, obj, thresholds=thresholds, show_image=False)
    return Image.fromarray(_to_int_image(svizs[-1][0])).convert("RGB")


def _make_feature_viz(
    df,
    layer_name,
    channel,
    cluster_label,
    model,
    ksize,
    stride,
    padding,
    prev_layer_name,
    out_image_path,
    n_samples=4,
):
    current_cluster_df = df[
        (df.layer_name == layer_name)
        & (df.channel == channel)
        & (df.cluster_label == cluster_label)
    ]
    uniq_imagenet_labels = _get_uniq_imagenet_labels(current_cluster_df, n_samples)
    images = []
    for imagenet_label in uniq_imagenet_labels:
        row = current_cluster_df[
            current_cluster_df.imagenet_label == imagenet_label
        ].iloc[0]
        im = _do_one_row(row, model, prev_layer_name, ksize, stride, padding)
        im.thumbnail((TARGET_THUMBNAIL_W, TARGET_THUMBNAIL_H))
        images.append(im)

    canvas = Image.new(
        "RGB", (TARGET_THUMBNAIL_W * len(images), TARGET_THUMBNAIL_H), "black"
    )
    for i, im in enumerate(images):
        x_off = (TARGET_THUMBNAIL_W - im.width) // 2
        y_off = (TARGET_THUMBNAIL_H - im.height) // 2
        canvas.paste(im, (i * TARGET_THUMBNAIL_W + x_off, y_off))
    canvas.save(out_image_path)
    print("saved to", out_image_path)


layer_by_params = {
    "mixed4d_3x3_pre_relu_conv": {
        "padding": (1, 1),
        "stride": (1, 1),
        "ksize": (3, 3),
        "prev_layer_name": "mixed4d_3x3_bottleneck",
    },
    "mixed4d_5x5_pre_relu_conv": {
        "padding": (2, 2),
        "stride": (1, 1),
        "ksize": (5, 5),
        "prev_layer_name": "mixed4d_5x5_bottleneck",
    },
    "mixed4d_1x1_pre_relu_conv": {
        "padding": (0, 0),
        "stride": (1, 1),
        "ksize": (1, 1),
        "prev_layer_name": "mixed4c",
    },
    "mixed4d_pool_reduce_pre_relu_conv": {
        "padding": (0, 0),
        "stride": (1, 1),
        "ksize": (1, 1),
        "prev_layer_name": "mixed4d_pool",
    },
}


def make_feature_viz(
    df,
    layer_name,
    channel,
    cluster_label,
    model,
    out_image_path,
    n_samples=5,
):
    ps = layer_by_params[layer_name]
    _make_feature_viz(
        df,
        layer_name,
        channel,
        cluster_label,
        model,
        ps["ksize"],
        ps["stride"],
        ps["padding"],
        ps["prev_layer_name"],
        out_image_path,
        n_samples,
    )


if __name__ == "__main__":
    import sys

    main_df = pd.read_csv(sys.argv[1])
    inp_path = sys.argv[2]
    out_dir = Path(sys.argv[3])
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"read df from {sys.argv[1]}, inp_path={inp_path}, out_dir={out_dir}")
    with open(inp_path) as f:
        key_by_count = json.load(f)
    for k in tqdm(list(key_by_count.keys())):
        try:
            layer_name, channel, cluster_label = ast.literal_eval(k)
            d = out_dir / layer_name / str(channel) / str(cluster_label)
            d.mkdir(parents=True, exist_ok=True)
            make_feature_viz(
                main_df,
                layer_name,
                channel,
                cluster_label,
                model,
                d / "vis.jpeg",
                n_samples=5,
            )
        except Exception as ex:
            print(f"FATAL: failed to render visualisation for {k}, ex={ex}")
            print("moving on")
