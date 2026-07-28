"""Activation-range data collection (the pre-analysis dump step).

This module holds the workhorses that used to live inline in
`notebooks/this-and-prev/actiavtion-ranges-data-collection.ipynb`. They take a
set of cluster reports (the s0->s4 `report.csv`s, already downloaded), replay the
model over the firing images, and dump four artifacts the analysis notebooks
(`activation-ranges-analysis`, `fox-or-cat`) consume:

  1. `build_main_df`                -> the flattened per-firing table
  2. `collect_acts_for_all_image`   -> layer/channel/cluster -> [scalar acts]
  3. `collect_all_noise_activations`-> layer/channel -> [baseline acts]
  4. `get_neuron_cluster_patches`   -> layer/channel/cluster -> [rf input patches]

**We hook the pre-relu conv itself** (`["output"]` for the neuron's own scalar,
`["input"]` for the receptive-field patch). That is the raw response of the
convolutional *kernel* — deliberately BEFORE any BatchNorm. So these collectors
are model-agnostic and need no BN handling: the only per-model/per-dataset knobs
are the `transform` (ImageNet vs `stl_transform`) and the `device`, both passed
in. InceptionV1 and `StlInception` share the exact same code path.
"""

from collections import defaultdict
import gc
import pickle
from itertools import batched
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from olt.act import InputOutputModelSnapshot
from olt.act_ranges.layer_utils import receptive_block
from olt.shards import raw_iter_shards, read_image_shard


# --- 1. main_df from downloaded reports --------------------------------------
def build_main_df(base_report_dir):
    """Concat every `report.csv` under `base_report_dir` (the extracted s0->s4
    reports you downloaded from S3) into one table, dropping unclustered firings
    (`cluster_label == -1`). Columns used downstream: input_image_key,
    layer_name, channel, cluster_label, y_position, x_position."""
    base_report_dir = Path(base_report_dir)
    dfs = []
    for f in tqdm(list(base_report_dir.rglob("report.csv")), desc="report.csv"):
        df = pd.read_csv(f)
        df = df[df.cluster_label != -1].reset_index()
        dfs.append(df)
    return pd.concat(dfs)


# --- 2. per-cluster neuron activations (the conv-output scalar) ---------------
def collect_acts_for_all_image(df, model, all_layers, device, flat_image_dir, transform):
    """For every firing in `df`, record the neuron's own scalar activation
    (the conv output at (channel, y, x)) grouped by cluster label."""
    flat_image_dir = Path(flat_image_dir)
    df = df.set_index("input_image_key", drop=False)
    all_image_keys = df.input_image_key.unique()
    result = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for image_key in tqdm(all_image_keys):
        batch = transform(Image.open(flat_image_dir / f"{image_key}.jpeg"))[None].to(device)
        idf = df.loc[[image_key]]
        with torch.no_grad():
            acts = InputOutputModelSnapshot.get_activations(batch, model, list(all_layers))
        for tup in idf.itertuples():
            result[tup.layer_name][str(tup.channel)][str(tup.cluster_label)].append(
                acts[tup.layer_name]["output"][
                    0, tup.channel, tup.y_position, tup.x_position
                ].item()
            )
    return result


def dump_layer_by_chan_by_cid_by_act(result, checkpoint_path):
    """Materialize the nested defaultdict to plain dict and pickle it."""
    plain = {}
    for layer_name, by_channel in result.items():
        plain[layer_name] = {}
        for channel, by_cid in by_channel.items():
            plain[layer_name][channel] = {cid: acts for cid, acts in by_cid.items()}
    with open(checkpoint_path, "wb") as f:
        pickle.dump(plain, f)


# --- 3. noise / baseline activations -----------------------------------------
def get_all_tfmed_images_as_batch(shards_base_dir, transform, per_dir=1):
    """Transformed sample images for the baseline ("noise") pool the per-channel
    scatter is drawn against: `per_dir` images from each class shard dir under
    `shards_base_dir`. With few classes (e.g. STL's 10) bump `per_dir` so the
    pool isn't tiny -- the noise sample count is `len(class_dirs) * per_dir *
    samples_per_image` (see get_random_sampled_activations)."""
    shards_base_dir = Path(shards_base_dir)
    label_dirs = [p for p in shards_base_dir.glob("*") if p.is_dir()]
    test_images = []
    for d in tqdm(label_dirs):
        shards = raw_iter_shards(d)
        if not shards:
            continue
        keys, images = next(read_image_shard(shards[0], per_dir, {}))
        test_images.extend(transform(img) for img in images)
    return test_images


def get_random_sampled_activations(
    model,
    test_images,
    layer_name,
    channel,
    device,
    batch_size=8,
    samples_per_image=16,
    disable_tqdm=False,
):
    """`samples_per_image` random spatial-location activations per test image for
    `channel` (drawn without replacement per image, capped at the number of
    spatial positions). One-per-image was the old behavior and starves the noise
    distribution when there are few images -- raise it for a denser sample."""
    noise_acts = []
    for batch in tqdm(list(batched(test_images, batch_size)), disable=disable_tqdm):
        batch = torch.stack(batch).to(device)
        act = InputOutputModelSnapshot.get_activations(batch, model, [layer_name])[
            layer_name
        ]["output"]
        bs = act.shape[0]
        act = act[:, channel, :, :].reshape(bs, -1)  # (bs, H*W)
        n_pos = act.shape[1]
        k = min(samples_per_image, n_pos)
        idx = torch.stack(
            [torch.randperm(n_pos, device=act.device)[:k] for _ in range(bs)]
        )
        noise_acts.append(torch.gather(act, 1, idx).reshape(-1).cpu())
    return torch.cat(noise_acts)


def collect_all_noise_activations(
    model, all_layers, test_images, batch_size, device, samples_per_image=16
):
    """Per (layer, channel) baseline activations over `test_images`."""
    result = defaultdict(dict)
    for layer_name in all_layers:
        n_chans = model.get_submodule(layer_name).weight.shape[0]
        for chan in tqdm(range(n_chans), desc=layer_name):
            result[layer_name][str(chan)] = get_random_sampled_activations(
                model,
                test_images,
                layer_name,
                chan,
                device,
                batch_size=batch_size,
                samples_per_image=samples_per_image,
                disable_tqdm=True,
            )
    return result


# --- 4. per-cluster receptive-field input patches ----------------------------
def _build_lookup_structures(df, model):
    layer_names = df.layer_name.unique()
    ikeys = df.input_image_key.unique()
    groups = dict(tuple(df.groupby("input_image_key")))
    layer_cache = {ln: model.get_submodule(ln) for ln in layer_names}
    return layer_names, ikeys, groups, layer_cache


def _load_batch(batched_ikeys, flat_image_dir, transform, device):
    return torch.stack(
        [transform(Image.open(flat_image_dir / f"{ikey}.jpeg")) for ikey in batched_ikeys]
    ).to(device)


def _extract_patches_for_image(rdf, acts, layer_cache, j, result, max_per_cid):
    for tup in rdf.itertuples():
        layer = layer_cache[tup.layer_name]
        y0, y1 = receptive_block(
            tup.y_position, layer.kernel_size[0], layer.stride[0], layer.padding[0]
        )
        x0, x1 = receptive_block(
            tup.x_position, layer.kernel_size[1], layer.stride[1], layer.padding[1]
        )
        patch = acts[tup.layer_name]["input"][j, :, y0:y1, x0:x1].clone()
        if len(result[tup.layer_name][tup.channel][tup.cluster_label]) < max_per_cid:
            result[tup.layer_name][tup.channel][tup.cluster_label].append(patch)


def dump_checkpoint(result, checkpoint_path):
    """Stack each cluster's patch list and torch.save the plain dict."""
    plain = {}
    for layer_name, by_channel in result.items():
        plain[layer_name] = {}
        for channel, by_cid in by_channel.items():
            plain[layer_name][channel] = {
                cid: torch.stack(patches) for cid, patches in by_cid.items()
            }
    torch.save(plain, checkpoint_path)


def load_checkpoint(checkpoint_path):
    """Inverse of `dump_checkpoint`: unstack back into lists so accumulation can
    resume."""
    plain = torch.load(checkpoint_path)
    result = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for layer_name, by_channel in plain.items():
        for channel, by_cid in by_channel.items():
            for cid, stacked in by_cid.items():
                result[layer_name][channel][cid] = list(stacked.unbind(0))
    return result


def get_neuron_cluster_patches(
    model,
    df,
    flat_image_dir,
    transform,
    device,
    checkpoint_path=None,
    checkpoint_every=100,
    max_per_cid=100,
    batch_size=32,
):
    """For each firing, slice the receptive-field patch out of the conv's INPUT
    tensor and bucket it by cluster label. Checkpoints periodically if
    `checkpoint_path` is given."""
    flat_image_dir = Path(flat_image_dir)
    layer_names, ikeys, groups, layer_cache = _build_lookup_structures(df, model)
    result = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    batches = list(batched(ikeys, batch_size))
    with torch.no_grad():
        for i, batched_ikeys in enumerate(tqdm(batches)):
            batch = _load_batch(batched_ikeys, flat_image_dir, transform, device)
            acts = InputOutputModelSnapshot.get_activations(batch, model, list(layer_names))
            for j, ikey in enumerate(batched_ikeys):
                _extract_patches_for_image(
                    groups[ikey], acts, layer_cache, j, result, max_per_cid
                )
            if checkpoint_path is not None and (i + 1) % checkpoint_every == 0:
                dump_checkpoint(result, checkpoint_path)
    if checkpoint_path is not None:
        dump_checkpoint(result, checkpoint_path)
        gc.collect()
    return result
