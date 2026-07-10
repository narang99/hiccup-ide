import torch
from torch.nn import functional as F

from olt.act_ranges.constants import CLUSTER_PATCH_SET_DIR
from olt.act_ranges.layer_utils import ReceptiveFieldOutOfBounds, receptive_block


def max_cosine_similarity(batch_tensor, ref_tensor):
    # batch_tensor: [B, C, K, K]
    # ref_tensor: [C, K, K]
    B = batch_tensor.shape[0]
    x = batch_tensor.reshape(B, -1)  # [B, C*K*K]
    r = ref_tensor.reshape(1, -1)  # [1, C*K*K]
    sims = F.cosine_similarity(x, r, dim=1)  # [B]
    return sims.max()


def min_euclidean_distance(batch_tensor, ref_tensor):
    # batch_tensor: [B, C, K, K]
    # ref_tensor: [C, K, K]
    B = batch_tensor.shape[0]
    x = batch_tensor.reshape(B, -1)  # [B, C*K*K]
    r = ref_tensor.reshape(1, -1)  # [1, C*K*K]
    dists = torch.norm(x - r, dim=1)  # [B]
    return dists.min()


def _load_cid_by_patches(
    layer_name, channel, cluster_patch_set_dir=CLUSTER_PATCH_SET_DIR
):
    # dict[cid, shape[b, ip-c, k, k]]
    return torch.load(
        cluster_patch_set_dir / layer_name / str(channel) / "cid_by_patches.pt",
        weights_only=False,
    )


def load_cluster_patches(
    layer_name, channel, cid, cluster_patch_set_dir=CLUSTER_PATCH_SET_DIR
):
    """The stored patch population (shape [B, C, K, K]) for one cluster, e.g. for
    rendering it against a wild pointwise-multiplication sample at report time."""
    cid_by_patches = _load_cid_by_patches(layer_name, channel, cluster_patch_set_dir)
    return cid_by_patches[cid]


# HDBSCAN's label for unclustered points (see stages/s3_cluster/run_hdbscan.py).
# cid_by_patches.pt is not expected to contain this key today, but "noise"
# isn't a coherent pattern even if it ever did show up here, so it's excluded
# defensively rather than trusted to just be absent.
NOISE_CID = -1


def closest_pw(
    patch,
    w,
    layer_name,
    channel,
    cluster_patch_set_dir=CLUSTER_PATCH_SET_DIR,
    min_similarity=0.0,
):
    """
    Finds the cluster (excluding NOISE_CID, see above) whose w-weighted patch
    population is most cosine-similar to w * patch. A cluster is only accepted
    as a match if its similarity is > min_similarity — a merely-least-negative
    "best" isn't actually similar to anything, it's just the least-dissimilar
    option among a bad field, so treating it as a match would be misleading.

    Returns (best_cid, best_sim, best_patches), or (None, None, None) if no
    cluster has similarity > min_similarity — callers must treat that as "no
    similar cluster found" and skip this patch rather than substituting an
    arbitrary cluster (this used to silently fall back to cid=-1, see git
    history — a crash if -1 isn't a real key, silent misattribution if it is).
    """
    cid_by_patches = _load_cid_by_patches(layer_name, channel, cluster_patch_set_dir)
    cid_by_sim = {
        cid: max_cosine_similarity(w * patches, w * patch)
        for cid, patches in cid_by_patches.items()
        if cid != NOISE_CID
    }

    best_cid, best_sim = None, min_similarity
    for cid, sim in cid_by_sim.items():
        if sim > best_sim:
            best_sim = sim
            best_cid = cid

    if best_cid is None:
        return None, None, None
    return best_cid, best_sim, cid_by_patches[best_cid]


def closest_patch_index(w, patches, patch):
    """Index into patches (shape [B, C, K, K], one cluster's stored patch
    population) whose w-weighted cosine similarity to patch (a single [C, K, K]
    sample, itself weighted by w) is highest — the per-sample nearest neighbor
    within an already-matched cluster, e.g. for pairing each wild
    pointwise-multiplication sample with its own closest cluster member at
    report time (see report_assets.dump_pw_sample_asset). Unlike closest_pw,
    which picks the best-matching cluster (best_cid) across all clusters, this
    picks the best-matching patch within one already-chosen cluster's population."""
    x = (w * patches).reshape(patches.shape[0], -1)
    r = (w * patch).reshape(1, -1)
    sims = F.cosine_similarity(x, r, dim=1)
    return int(sims.argmax())


def get_neuron_closest_cluster(
    model,
    layer_name,
    b,
    channel,
    y,
    x,
    captured_acts,
    cluster_patch_set_dir=CLUSTER_PATCH_SET_DIR,
):
    """Returns (None, None, None, None, None) if (layer_name, channel)'s own
    receptive field at (y, x) falls outside its captured input tensor (see
    layer_utils.ReceptiveFieldOutOfBounds) — callers already treat an all-None
    result the same as "no confident cluster match" (see
    analyser.resolve_dependency_match/get_cluster_above_noise_ratio) and skip
    this dependency firing, rather than the whole run failing on one
    boundary position."""
    layer = model.get_submodule(layer_name)
    input_tensor = captured_acts[layer_name]["input"]
    try:
        y0, y1 = receptive_block(
            y,
            layer.kernel_size[0],
            layer.stride[0],
            layer.padding[0],
            input_size=input_tensor.shape[-2],
        )
        x0, x1 = receptive_block(
            x,
            layer.kernel_size[1],
            layer.stride[1],
            layer.padding[1],
            input_size=input_tensor.shape[-1],
        )
    except ReceptiveFieldOutOfBounds as e:
        print(
            f"WARN: skipping dependency firing {layer_name}:{channel} at (y={y}, x={x}): {e}"
        )
        return None, None, None, None, None

    patch = input_tensor[b, :, y0:y1, x0:x1]  # [C, k, k]
    w = layer.weight[channel].detach().cpu().clone()  # [C, k, k]

    best_cid, max_sim, all_patches = closest_pw(
        patch, w, layer_name, channel, cluster_patch_set_dir
    )

    return best_cid, max_sim, patch, all_patches, w
