from typing import Any, TypedDict

import numpy as np
import torch
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import normalize


class TrainedModel(TypedDict):
    "This class is pickled, its therefore kept lean (we dont attach methods to it)"

    clusterer: Any
    dbcv: float


def predict_labels(X, hdbscan_module, trained_model: TrainedModel):
    X = normalize(X, "l2")
    return _predict_labels(X, hdbscan_module, trained_model)


def predict_labels_and_recons(X, hdbscan_module, trained_model: TrainedModel):
    X = normalize(X, "l2")
    labels = _predict_labels(X, hdbscan_module, trained_model)
    recons = _predict_recons(X, labels, trained_model)
    return labels, recons


def _predict_labels(X, hdbscan_module, trained_model: TrainedModel):
    # does not normalize
    labels, _ = hdbscan_module.approximate_predict(trained_model["clusterer"], X)
    return labels


def _predict_recons(X, labels, medoids):
    noise_label = -1
    reconstructed = np.empty_like(X)
    for i, label in enumerate(labels):
        if label == noise_label or label not in medoids:
            reconstructed[i] = np.random.randn(X.shape[1]).astype(X.dtype)
        else:
            reconstructed[i] = medoids[label]
    return reconstructed


def sweep_train_hdbscan(
    data,
    hdbscan_module,
    min_cluster_sizes: list[int],
    train_kwargs=None,
):
    if not min_cluster_sizes:
        raise Exception(
            f"passed empty min cluster sizes to clusterer: {min_cluster_sizes}"
        )
    data = normalize(data, "l2")
    trained_models = []
    for min_cluster_size in min_cluster_sizes:
        trained_model = train_hdbscan_model(
            data, hdbscan_module, min_cluster_size, train_kwargs
        )
        trained_models.append(trained_model)
    trained_models = sorted(trained_models, key=lambda m: m["dbcv"], reverse=True)
    best = trained_models[0]
    best = {
        "clusterer": best["clusterer"],
        "dbcv": best["dbcv"],
    }

    rest = [
        {"clusterer": m["clusterer"], "dbcv": m["dbcv"]} for m in trained_models[1:]
    ]
    return best, rest


def train_hdbscan_model(data, hdbscan_module, min_cluster_size, train_kwargs=None):
    """assumes we get L2 normalized data already, hdbscan_module is either CUML version of stock version. you should use sweep version generally"""
    if train_kwargs is None:
        train_kwargs = {}
    train_kwargs = {
        **train_kwargs,
        "prediction_data": True,
        "gen_min_span_tree": True,
        "min_cluster_size": min_cluster_size,
    }
    clusterer = hdbscan_module.HDBSCAN(**train_kwargs)
    clusterer.fit(data)
    dbcv_score = DBCV(clusterer.minimum_spanning_tree_, clusterer.labels_)

    return {
        "clusterer": clusterer,
        "dbcv": dbcv_score,
    }


def _get_medoids(labels, data):
    unique_labels = sorted(set(labels) - {-1})
    medoids = {}
    for label in unique_labels:
        mask = labels == label
        cluster_points = data[mask]
        centroid = cluster_points.mean(axis=0, keepdims=True)
        dists = euclidean_distances(cluster_points, centroid).squeeze()
        medoids[label] = cluster_points[np.argmin(dists)]

    return medoids


def _as_numpy(X):
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    return X


# reference: https://github.com/rapidsai/cuml/issues/5945#issuecomment-2217297472
def DBCV(minimum_spanning_tree, labels):
    sizes = np.bincount(labels + 1)
    noise_size = sizes[0]
    cluster_size = sizes[1:]
    total = noise_size + np.sum(cluster_size)
    num_clusters = len(cluster_size)
    DSC = np.zeros(num_clusters)
    min_outlier_sep = np.inf  # only required if num_clusters = 1
    correction_const = 2  # only required if num_clusters = 1

    # Unltimately, for each Ci, we only require the
    # minimum of DSPC(Ci, Cj) over all Cj != Ci.
    # So let's call this value DSPC_wrt(Ci), i.e.
    # density separation 'with respect to' Ci.
    DSPC_wrt = np.ones(num_clusters) * np.inf
    max_distance = 0

    mst_df = minimum_spanning_tree.to_pandas()

    for edge in mst_df.iterrows():
        label1 = labels[int(edge[1]["from"])]
        label2 = labels[int(edge[1]["to"])]
        length = edge[1]["distance"]

        max_distance = max(max_distance, length)

        if label1 == -1 and label2 == -1:
            continue
        elif label1 == -1 or label2 == -1:
            # If exactly one of the points is noise
            min_outlier_sep = min(min_outlier_sep, length)
            continue

        if label1 == label2:
            # Set the density sparseness of the cluster
            # to the sparsest value seen so far.
            DSC[label1] = max(length, DSC[label1])
        else:
            # Check whether density separations with
            # respect to each of these clusters can
            # be reduced.
            DSPC_wrt[label1] = min(length, DSPC_wrt[label1])
            DSPC_wrt[label2] = min(length, DSPC_wrt[label2])

    # In case min_outlier_sep is still np.inf, we assign a new value to it.
    # This only makes sense if num_clusters = 1 since it has turned out
    # that the MR-MST has no edges between a noise point and a core point.
    min_outlier_sep = max_distance if min_outlier_sep == np.inf else min_outlier_sep

    # DSPC_wrt[Ci] might be infinite if the connected component for Ci is
    # an "island" in the MR-MST. Whereas for other clusters Cj and Ck, the
    # MR-MST might contain an edge with one point in Cj and ther other one
    # in Ck. Here, we replace the infinite density separation of Ci by
    # another large enough value.
    #
    # TODO: Think of a better yet efficient way to handle this.
    correction = correction_const * (
        max_distance if num_clusters > 1 else min_outlier_sep
    )
    DSPC_wrt[np.where(DSPC_wrt == np.inf)] = correction

    V_index = [
        (DSPC_wrt[i] - DSC[i]) / max(DSPC_wrt[i], DSC[i]) for i in range(num_clusters)
    ]
    score = np.sum(
        [(cluster_size[i] * V_index[i]) / total for i in range(num_clusters)]
    )

    return score
