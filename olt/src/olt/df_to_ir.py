import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from captum.attr import NeuronDeepLift
from PIL import Image
from sklearn.preprocessing import normalize
from torch import nn
from tqdm import tqdm

from olt.act import get_layer_activations
from olt.path import RemotePath
from olt.shards import raw_iter_shards, read_image_shard
from olt.stages.s2_collect_patches import patches_of_single_batch_with_indices
from olt.tfms import inverse_transform, transform


def extract_images_to_flat_folder(images_shard_dir, flat_dest_dir):
    for d in tqdm(list(images_shard_dir.glob("*"))):
        shards = raw_iter_shards(d)
        for shard in shards:
            for keys, images in read_image_shard(shard, 64, {}):
                for k, i in zip(keys, images):
                    # print(k, i)
                    i.save(flat_dest_dir / f"{k}.jpeg")


def labels_with_unique_inp_keys_below_threshold(df, threshold):
    counts = df.groupby("cluster_label")["input_image_key"].nunique()
    return counts[counts < threshold].index.tolist()


def stratified_sample_for_cluster_label(df, layer_name, cluster_label, n_total):
    # for a given cluster label, sample images stratified on imagenet_label
    # i think we should add layer_name also since thats the main combo

    # 1. Filter for the specific cluster
    mask = (df["cluster_label"] == cluster_label) & (df["layer_name"] == layer_name)
    cluster_df = df[mask]

    # 2. Sample safely by capping the sample size at len(x)
    sampled_indices = (
        cluster_df.groupby("imagenet_label", group_keys=False)
        .apply(
            lambda x: x.sample(
                min(len(x), max(1, round(len(x) / len(cluster_df) * n_total)))
            ),
            include_groups=False,
        )
        .index
    )

    # 3. Fetch and return the complete rows
    return df.loc[sampled_indices]


def get_neuron_selectors_patches_and_labels(rows, timg, model, layer_name, device):
    layer = model.get_submodule(layer_name)
    layer_weight = layer.weight.detach().cpu().numpy()

    input_activation = get_layer_activations(timg, model, [layer_name])[layer_name]

    all_cluster_labels, all_pws, all_indices = [], [], []
    for chan in rows["channel"].unique():
        channel_rows = rows[rows["channel"] == chan]
        channels = channel_rows["channel"]
        y_pos = channel_rows["y_position"]
        x_pos = channel_rows["x_position"]
        indices = [[0, chan, y, x] for (chan, y, x) in zip(channels, y_pos, x_pos)]
        indices = torch.tensor(indices)
        patches = patches_of_single_batch_with_indices(input_activation, layer, indices)
        # patches = torch.stack(patches)
        pws = patches.numpy() * layer_weight[chan].reshape(-1)
        pws = normalize(pws, "l2")

        # convert back to lists for homogenous api
        cluster_labels = channel_rows["cluster_label"]

        indices = [i for i in indices]
        pws = [pw for pw in pws]

        if len(cluster_labels) != len(pws) or len(cluster_labels) != len(indices):
            raise Exception(
                f"fatal error. labels, indices and pws length do not match, labels={len(cluster_labels)} indices={len(indices)} pws={len(pws)}"
            )

        all_cluster_labels.extend(cluster_labels)
        all_pws.extend(pws)
        all_indices.extend(indices)

    return all_cluster_labels, all_pws, all_indices


def get_neuron_attribution_for_points(
    timg,
    model,
    layer_name,
    device,
    indices,
):
    ng = NeuronDeepLift(model, model.get_submodule(layer_name))
    # ng = NeuronIntegratedGradients(model, model.get_submodule(layer_name))
    timg = timg.to(device)
    attributions = []
    for i in indices:
        _, c, h, w = i
        neuron_att = (
            ng.attribute(
                timg,
                (c, h, w),
            )
            .detach()
            .cpu()
        )
        # neuron_att = ng.attribute(
        #     timg, (c, h, w), n_steps=64, internal_batch_size=64
        # ).detach().cpu()
        attributions.append(neuron_att)
    return attributions


def dump_single_images_report_elements(
    image_key: str,
    imagenet_label: int,
    out_dir: RemotePath,
    timg: torch.Tensor,
    inverse_transform_fn,
    cluster_labels: list[int],
    pws: list[torch.Tensor],
    neuron_attrs: list[torch.Tensor],
):
    dest_dir = Path(out_dir) / str(imagenet_label) / str(image_key)
    dest_dir.mkdir(parents=True, exist_ok=True)
    inv_img = inverse_transform_fn(timg)[0].cpu().permute(1, 2, 0).numpy()
    plt.imsave(dest_dir / "image.jpg", inv_img)

    for i in range(len(cluster_labels)):
        cluster_label = cluster_labels[i]
        pw = pws[i]
        neuron_attr = neuron_attrs[i]

        att_dest_dir = dest_dir / str(cluster_label) / "neuron_attributions"
        att_dest_dir.mkdir(parents=True, exist_ok=True)
        torch.save(neuron_attr, att_dest_dir / f"{i}.pth")

        activation_dest_dir = dest_dir / str(cluster_label) / "input_activations"
        activation_dest_dir.mkdir(parents=True, exist_ok=True)
        torch.save(pw, activation_dest_dir / f"{i}.pth")


def create_report_dir_for_one_input(
    report_df,
    image_key: str,
    model: nn.Module,
    layer_name: str,
    flat_images_dir: RemotePath,
    out_dir: RemotePath,
    input_transform_fn,
    inverse_transform_fn,
    device,
):
    # outdir / imagenet_label / key
    # .    image.jpg
    #     <cluser_label>
    # .       neuron_attributions
    #           0.pth
    #        input_activations
    #           0.pth
    # this structure is not nice at all though
    # we'll need to improve it later, but for now, the report generator
    # expects this

    mask = (report_df["input_image_key"] == image_key) & (
        report_df["layer_name"] == layer_name
    )
    our_report_df = report_df[mask]
    if len(our_report_df) == 0:
        print(
            f"WARN: did not create any report outputs for image_key={image_key} layer_name={layer_name}. the dataframe is empty for these"
        )
        return
    uniq_imagenet_labels = our_report_df["imagenet_label"].unique()
    if len(uniq_imagenet_labels) != 1:
        print(
            "SKIP: key has multiple imagenet labels; key =",
            image_key,
            "len labels =",
            len(uniq_imagenet_labels),
        )
        return
        # raise Exception(
        #     "imagenet labels in the filtered df is not of size 1 "
        #     f"(it should be since we only have one inputs df, length={len(uniq_imagenet_labels)} key={image_key}"
        # )
    imagenet_label = int(uniq_imagenet_labels[0])

    image_path = flat_images_dir / f"{image_key}.jpeg"
    timg = input_transform_fn(Image.open(image_path))[None].to(device)
    cluster_labels, pws, indices = get_neuron_selectors_patches_and_labels(
        our_report_df, timg, model, layer_name, device
    )
    neuron_attrs = get_neuron_attribution_for_points(
        timg, model, layer_name, device, indices
    )
    dump_single_images_report_elements(
        image_key,
        imagenet_label,
        out_dir,
        timg,
        inverse_transform_fn,
        cluster_labels,
        pws,
        neuron_attrs,
    )


def prepare_reports_dir_after_sampling(
    df, report_out_dir, model, flat_images_base, device
):
    labels = labels_with_unique_inp_keys_below_threshold(df, 2)
    filtered_df = df[~df["cluster_label"].isin(labels)]

    combinations = filtered_df[["layer_name", "channel"]].drop_duplicates()
    samples_per_cluster_label = 100

    for _, row in combinations.iterrows():
        # for each laeyr name and channel, we want a different dir
        layer_name = row["layer_name"]
        channel = row["channel"]
        out_dir = report_out_dir / layer_name / str(channel)
        shutil.rmtree(out_dir, ignore_errors=True)

        mask = (filtered_df["layer_name"] == layer_name) & (
            filtered_df["channel"] == channel
        )
        this_neurons_filtered_df = filtered_df[mask]
        dfs = []
        uniq_cluster_labels = this_neurons_filtered_df["cluster_label"].unique()
        print("uniq lables", uniq_cluster_labels)
        for cluster_label in uniq_cluster_labels:
            sampled = stratified_sample_for_cluster_label(
                this_neurons_filtered_df,
                layer_name,
                cluster_label,
                samples_per_cluster_label,
            )
            dfs.append(sampled)
        result_df = pd.concat(dfs, ignore_index=True)

    print(
        f"layer_name: {layer_name} channel: {channel} sampled_df_size={len(result_df)} total_cluster_labels={len(uniq_cluster_labels)}"
    )
    for input_image_key in tqdm(result_df["input_image_key"].unique()):
        create_report_dir_for_one_input(
            result_df,
            input_image_key,
            model,
            layer_name,
            flat_images_base,
            out_dir,
            transform,
            inverse_transform,
            device,
        )
