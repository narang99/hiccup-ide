import gc
import json
import pickle

import numpy as np
from torch import nn

from olt.path import RemotePath, remote_mkdir
from olt.stages.s2_collect_patches import read_all_patches

from .run_hdbscan import TrainedModelWithoutMedoids, sweep_train_hdbscan


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
        best, _, meta = train_clusterer_for_single_neuron(
            model,
            layer_name,
            channel,
            patches_base_dir,
            hdbscan_module,
            min_cluster_sizes,
        )
        _persist_clusterer_and_meta(model_store_dir, layer_name, channel, best, meta)
        gc.collect()


def _persist_clusterer_and_meta(
    model_store_dir: RemotePath, layer_name: str, channel: int, best, meta
):
    # we would need to store it too now. we only store best
    dest_dir = model_store_dir / layer_name / str(channel)
    remote_mkdir(dest_dir)
    with (dest_dir / "model.pkl").open("wb") as f:
        pickle.dump(best, f)
    with (dest_dir / "meta.json").open("w") as f:
        json.dump(meta, f)


def train_clusterer_for_single_neuron(
    model: nn.Module,
    layer_name: str,
    channel: int,
    patches_base_dir: RemotePath,
    hdbscan_module,
    min_cluster_sizes: list[int],
):
    patches = read_all_patches(patches_base_dir, layer_name, channel)
    layer = model.get_submodule(layer_name)
    layer_weight = layer.weight[channel].reshape(-1).detach().cpu()  # ty: ignore
    pws = patches * layer_weight

    print(f"############################ {channel} #############################")
    print(f"layer weight shape: {layer_weight.shape}")
    print(f"pws shape: {pws.shape}")
    best, rest = sweep_train_hdbscan(pws.numpy(), hdbscan_module, min_cluster_sizes)

    meta = {
        "best": model_to_meta(best),
        "rest": [model_to_meta(r) for r in rest],
    }

    del pws, patches

    return best, rest, meta


def model_to_meta(trained_model: TrainedModelWithoutMedoids):
    clusterer = trained_model["clusterer"]
    score = trained_model["dbcv"]
    labels = clusterer.labels_
    unique, counts = np.unique(labels, return_counts=True)
    return {
        "min_cluster_size": int(clusterer.min_cluster_size),
        "dbcv": float(score),
        "labels": {int(k): int(v) for k, v in zip(unique, counts)},
    }
