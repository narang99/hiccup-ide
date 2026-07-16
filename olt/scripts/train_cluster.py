import json
import time

import numpy as np
import torch

from olt.path import RemotePath

try:
    import cupy as cp
except ImportError:
    cp = np

try:
    from cuml.cluster import hdbscan
    from cuml.preprocessing import normalize
except ImportError:
    import hdbscan
    from sklearn.preprocessing import normalize


def get_normalized_dataset(model, layer_name, channel, patches_base):
    start = time.time()
    patches, indices, input_keys, imagenet_labels, pws = get_full_dataset(
        patches_base, model, layer_name, channel
    )
    if isinstance(pws, torch.Tensor):
        pws = cp.asarray(pws.numpy())
    pws = normalize(pws)

    print("dataset prep:", time.time() - start, "seconds")
    return patches, indices, input_keys, imagenet_labels, pws


def train_clusterer(
    pws,
    max_samples_for_train,
    train_kwargs=None,
):
    start = time.time()

    sampled_idxs = cp.random.permutation(len(pws))[:max_samples_for_train]
    sampled_pws = pws[sampled_idxs]
    # print("dataset prep:", time.time() - start, "seconds")
    # start train
    if train_kwargs is None:
        train_kwargs = {
            "prediction_data": True,
            "min_cluster_size": 20,
            "min_samples": 5,
            "cluster_selection_method": "leaf",
        }
    print(f"starting train. samples={len(sampled_pws)} train_kwargs={train_kwargs}")
    clusterer = hdbscan.HDBSCAN(**train_kwargs)
    clusterer.fit(sampled_pws)

    print("cluster fit:", time.time() - start)
    start = time.time()

    # inference
    labels = hdbscan.approximate_predict(clusterer, pws)[
        0
    ].get()  # cupy needs calls to get for retrieving numpy array
    print("cluster inference:", time.time() - start)

    return labels


def train_clusterer_and_dump_labels(
    model,
    layer_name,
    channel,
    patches_base,
    out_json_path: RemotePath,
    max_samples_for_train=100_000,
    train_kwargs=None,
):
    patches, indices, input_keys, imagenet_labels, pws = get_normalized_dataset(
        patches_base, model, layer_name, channel
    )
    labels = train_clusterer(pws, max_samples_for_train, train_kwargs)
    labels = labels.tolist()

    with out_json_path.open("w") as f:
        json.dump(labels, f)


def train_one_neuron(
    model,
    layer_name,
    channel,
    patches_base,
    flat_images_base,
    report_dest,
    image_shape,
    max_samples_for_train=100_000,
    device="cuda",
    train_kwargs=None,
):

    patches, indices, input_keys, imagenet_labels, pws = get_normalized_dataset(
        patches_base, model, layer_name, channel
    )
    labels = train_clusterer(pws, max_samples_for_train, train_kwargs)

    # report generation
    report_dest.mkdir(exist_ok=True, parents=True)
    _prepare_reports(
        layer_name,
        channel,
        labels,
        indices,
        input_keys,
        imagenet_labels,
        report_dest,
        model,
        flat_images_base,
        device,
        image_shape,
    )
    print("report generation:", time.time() - start)
