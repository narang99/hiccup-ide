from dataclasses import dataclass
from itertools import batched

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity
from torch import nn

# from pt_to_api.utils import *
from pt_to_api.utils import (
    show_single_channel_red_green_black as S,
)

from .core import SingleRun


def show_closest_component_of_W_for_each_component(
    components, W_true, image_shape, figsize=(5, 2), comps_per_row=3, mode="light"
):
    """
    Given two arrays of numpy vectors of same shapes
    for every component in `components`, this function shows the array in `W_true`
    which has the maximum cosine similarity with the component
    """
    sims = np.abs(cosine_similarity(components, W_true))
    pairs = []
    for i in range(len(components)):
        j = np.argmax(sims[i])
        pairs.append((i, j, sims[i][j]))
    for batch in batched(pairs, comps_per_row):
        ax_titles, comps, scores = [], [], []

        for i, j, score in batch:
            comps.extend(
                [components[i].reshape(image_shape), W_true[j].reshape(image_shape)]
            )
            ax_titles.extend([f"component\n{score:.5f}", "ground_truth"])
            # scores.append(score)
        S(
            comps,
            figsize,
            2 * comps_per_row,
            mode=mode,
            # suptitle=f"similarity score={score}",
            ax_titles=ax_titles,
            viztype="local",
        )
        plt.show()


def evaluate_recovery(W_learned, W_true, threshold=0.95):
    """
    W_learned: (n_atoms, patch_dim)
    W_true: (n_atoms, patch_dim)

    for each true atom, finds the best matching learned atom by cosine similarity
    returns fraction of true atoms recovered above threshold
    """
    W_l = W_learned / (np.linalg.norm(W_learned, axis=1, keepdims=True) + 1e-8)
    W_t = W_true / (np.linalg.norm(W_true, axis=1, keepdims=True) + 1e-8)

    sim = np.abs(W_l @ W_t.T)  # (n_atoms, n_atoms), abs because sign is arbitrary
    best_match = sim.max(
        axis=0
    )  # for each true atom, best cosine with any learned atom

    recovered = (best_match >= threshold).mean()
    print(f"Mean best cosine similarity: {best_match.mean():.4f}")
    print(f"Fraction recovered (>{threshold}): {recovered:.4f}")
    return best_match, recovered


def match_atoms(D1: np.ndarray, D2: np.ndarray):
    """
    Match atoms of D1 to atoms of D2 using the Hungarian algorithm
    on cosine distances. Assumes square dictionaries (same n_components).

    D1, D2: shape (n_components, n_features) — sklearn's components_ layout.

    Returns:
        row_ind, col_ind: matched index arrays
        matched_similarities: per-pair cosine similarities
        mean_sim: mean cosine similarity across matched pairs
    """
    # Guard against dead atoms (zero-norm rows produce NaN cosine distances)
    norms_1 = np.linalg.norm(D1, axis=1, keepdims=True)
    norms_2 = np.linalg.norm(D2, axis=1, keepdims=True)
    if np.any(norms_1 == 0) or np.any(norms_2 == 0):
        raise ValueError(
            "One or more atoms have zero norm. "
            "Remove or replace dead atoms before matching."
        )

    # cost = cosine_distances(D1, D2)          # shape (n_components, n_components), values in [0, 2]
    cost = np.abs(cosine_similarity(D1, D2))
    cost = 1 - cost

    row_ind, col_ind = linear_sum_assignment(cost)
    matched_similarities = 1 - cost[row_ind, col_ind]
    mean_sim = float(matched_similarities.mean())
    return row_ind, col_ind, matched_similarities, mean_sim


def get_live(components):
    dead = find_dead_atoms(components).numpy()
    live = np.array([i for i in range(components.shape[0]) if i not in dead])
    return components[live]


def hungarian_match(all_components: list[np.ndarray]):
    """
    Pairwise similarity matching across runs using the Hungarian algorithm.

    Args:
        all_components: list of arrays, each shape (n_components, n_features).
                        All arrays must have the same shape.

    Returns:
        upper: 1-D array of pairwise similarities for all unique pairs
        stability_score: mean of upper
        best_run_idx: index of the run most similar to all others
        pairwise_sims: (n_runs, n_runs) symmetric similarity matrix, diagonal = 1
    """
    n_runs = len(all_components)

    if n_runs < 2:
        raise ValueError("Need at least 2 runs to compute pairwise similarity.")

    shapes = [d.shape for d in all_components]
    if len(set(shapes)) != 1:
        raise ValueError(f"All dictionaries must have the same shape. Got: {shapes}")

    pairwise_sims = np.ones((n_runs, n_runs))
    for i in range(n_runs):
        for j in range(i + 1, n_runs):
            _, _, _, mean_sim = match_atoms(all_components[i], all_components[j])
            pairwise_sims[i, j] = mean_sim
            pairwise_sims[j, i] = mean_sim

    upper = pairwise_sims[np.triu_indices(n_runs, k=1)]
    stability_score = float(upper.mean())

    # Exclude self-similarity (diagonal=1) when ranking runs
    np.fill_diagonal(pairwise_sims, 0)
    mean_sim_per_run = pairwise_sims.sum(axis=1) / (n_runs - 1)
    best_run_idx = int(np.argmax(mean_sim_per_run))
    np.fill_diagonal(pairwise_sims, 1)  # restore diagonal

    return upper, stability_score, best_run_idx, pairwise_sims


def find_dead_atoms(W, threshold=0.1):
    if not isinstance(W, torch.Tensor):
        W = torch.tensor(W)

    peak = W.abs().max(dim=1).values
    peak_normalised = peak / peak.max()

    return torch.where(peak_normalised < threshold)[0]


def std_ratios(v1, v2):
    rat = min(v1, v2) / max(v1, v2)
    if isinstance(rat, torch.Tensor):
        return rat.item()
    return rat


def get_metrics_from_run(run: SingleRun, W_true=None, support_overlap_threshold=0.01):
    if W_true is not None:
        _, _, sim_vector, mean_sim = match_atoms(W_true, run.components)
    else:
        sim_vector, mean_sim = None, None

    # decoder max vals matrix
    decoder_maxes = run.components.max(axis=1)[0]

    return {
        "mse": run.loss,
        "support_overlap": support_overlap(run.components, support_overlap_threshold),
        "decoder_maxes_mean_ratio": std_ratios(
            decoder_maxes.mean().item(), run.hyperparameters["sigma_0"]
        ),
        "decoder_sigma_ratio": std_ratios(
            run.components.std(), run.hyperparameters["sigma_0"]
        ),
        "decoder_sigma_per_col_ratio": std_ratios(
            run.components.sum(axis=0).std(), run.hyperparameters["sigma_0"]
        ),
        "encoder_sigma_ratio": std_ratios(
            run.encoder.std(), run.hyperparameters["sigma_enc"]
        ),
        "mean_sim": mean_sim,
        "vec_sim": sim_vector,
        # "gram": gram_orthogonality_error(run.components.T),
    }


def aggregate_metrics(dicts):
    keys = dicts[0].keys()
    result = {}
    for key in keys:
        vals = [d[key] for d in dicts]
        if key == "vec_sim":
            result[key] = np.stack(vals).mean(axis=0)
        elif key == "mse":
            result[key] = sum(v.item() for v in vals) / len(vals)
        else:
            result[key] = sum(vals) / len(vals)
    return result


def print_summary(X, W_true, codes_true, run: SingleRun):
    print("standard devications")
    print("\tX:", X.std())
    print("\tW_true:", W_true.std())
    print("\tcodes_true:", codes_true.std())
    print("\tcodes:", run.codes.std(), "sigma_s", run.hyperparameters["sigma_s"])
    print("\tcomponents:", run.components.std())
    print("\trecon:", run.recon.std())
    print(
        "\tdecoder:",
        run.components.std(),
        "sigma_0:",
        run.hyperparameters["sigma_0"],
    )
    print(
        "\tencoder:",
        run.encoder.std(),
        "sigma_enc:",
        run.hyperparameters["sigma_enc"],
    )
    print(
        "\tMSE:", (X - run.recon).std(), "sigma_eps:", run.hyperparameters["sigma_eps"]
    )


def get_device(dim):
    if dim < 100:
        return "cpu"
    else:
        return "mps"


def support_overlap_matrix_batched(W, threshold=0.05):
    # [B, n-components, dims]
    W = torch.tensor(W)

    # [B, n-components, dims]
    abs_W = W.abs()

    # [B, n-components, 1]
    # do i need different maxes though? idts. use the useful one only
    maxvals = abs_W.max(dim=2, keepdim=True).values

    # [B, n-components, dims]
    support = abs_W > threshold * maxvals
    support_f = support.float()

    # [B, n-components, dims], [B, n-components, dims] -> [B, n-components, n-components]
    intersection = torch.einsum("bid,bjd->bij", support_f, support_f)

    # [B, n-components]
    support_sizes = support.sum(dim=2).float()

    # [B, n-components, n-components]
    min_sizes = torch.min(support_sizes[:, :, None], support_sizes[:, None, :])

    overlap = intersection / min_sizes
    return overlap.mean(dim=0)


def support_overlap_matrix(W, threshold=0.01):
    W = torch.tensor(W)
    abs_W = W.abs()
    maxvals = abs_W.max(dim=1, keepdim=True).values  # (n_components, 1)
    support = abs_W > threshold * maxvals  # (n_components, n_dims) binary

    # pairwise intersection over union (or just intersection)
    support_f = support.float()
    intersection = support_f @ support_f.T  # (n_components, n_components)
    support_sizes = support.sum(dim=1).float()  # (n_components,)
    min_sizes = torch.min(support_sizes.unsqueeze(1), support_sizes.unsqueeze(0))

    overlap = intersection / min_sizes  # normalized: 0=disjoint, 1=fully overlapping
    return overlap


def support_overlap(W, threshold=0.01):
    overlap = support_overlap_matrix(W, threshold)
    return overlap.fill_diagonal_(0).mean()
