import gc
import json
import pickle
import time

import numpy as np
import pandas as pd
import torch
from torch import nn

from olt.path import RemotePath, remote_mkdir
from olt.stages.s2_collect_patches import read_all_patches

from .run_hdbscan import TrainedModel, sweep_train_hdbscan


def train_models_for_layer(
    model: nn.Module,
    layer_name: str,
    channels: list[int],
    model_store_dir: RemotePath,
    patches_base_dir: RemotePath,
    hdbscan_module,
    min_cluster_sizes: list[int],
):
    for channel in channels:
        if _is_done(model_store_dir / layer_name / str(channel) / "meta.json"):
            print(f"SKIP: channel={channel}; already done")
            continue

        start = time.time()
        best, _, meta, indices, input_keys, imagenet_labels = (
            train_clusterer_for_single_neuron(
                model,
                layer_name,
                channel,
                patches_base_dir,
                hdbscan_module,
                min_cluster_sizes,
            )
        )
        _persist_clusterer_and_meta(
            model_store_dir,
            layer_name,
            channel,
            best,
            meta,
            indices,
            input_keys,
            imagenet_labels,
        )
        gc.collect()
        print(f"train time: {time.time() - start} seconds")


def _is_done(meta_file_path: RemotePath):
    if not meta_file_path.exists():
        return False
    with meta_file_path.open("r") as f:
        content = json.load(f)
    return content.get("done", False)


def _persist_clusterer_and_meta(
    model_store_dir: RemotePath,
    layer_name: str,
    channel: int,
    best,
    meta,
    indices: torch.Tensor,
    input_keys: list[str],
    imagenet_labels: list[int],
):
    # we would need to store it too now. we only store best
    dest_dir = model_store_dir / layer_name / str(channel)
    remote_mkdir(dest_dir)
    with (dest_dir / "model.pkl").open("wb") as f:
        pickle.dump(best, f)
    with (dest_dir / "meta.json").open("w") as f:
        json.dump(meta, f)

    channels = [int(i[1].item()) for i in indices]
    y_positions = [int(i[2].item()) for i in indices]
    x_positions = [int(i[3].item()) for i in indices]
    layer_names = [layer_name for _ in range(len(indices))]
    cluster_labels = best["clusterer"].labels_

    df = pd.DataFrame(
        {
            "channel": channels,
            "y_position": y_positions,
            "x_position": x_positions,
            "cluster_label": cluster_labels,
            "imagenet_label": imagenet_labels,
            "input_image_key": input_keys,
            "layer_name": layer_names,
        }
    )
    with (dest_dir / "report.csv").open("w") as f:
        df.to_csv(f)


def train_clusterer_for_single_neuron(
    model: nn.Module,
    layer_name: str,
    channel: int,
    patches_base_dir: RemotePath,
    hdbscan_module,
    min_cluster_sizes: list[int],
):
    patches, indices, input_keys, imagenet_labels = read_all_patches(
        patches_base_dir, layer_name, channel
    )
    layer = model.get_submodule(layer_name)
    layer_weight = layer.weight[channel].reshape(-1).detach().cpu()  # ty: ignore
    pws = patches * layer_weight

    print(f"############################ {channel} #############################")
    print(f"layer weight shape: {layer_weight.shape}")
    print(f"pws shape: {pws.shape}")
    best, rest = sweep_train_hdbscan(pws.numpy(), hdbscan_module, min_cluster_sizes)

    meta = {
        "done": True,
        "best": model_to_meta(best),
        "rest": [model_to_meta(r) for r in rest],
    }

    del pws, patches

    return best, rest, meta, indices, input_keys, imagenet_labels


def model_to_meta(trained_model: TrainedModel):
    clusterer = trained_model["clusterer"]
    score = trained_model["dbcv"]
    labels = clusterer.labels_
    unique, counts = np.unique(labels, return_counts=True)
    return {
        "min_cluster_size": int(clusterer.min_cluster_size),
        "dbcv": float(score),
        "labels": {int(k): int(v) for k, v in zip(unique, counts)},
    }
