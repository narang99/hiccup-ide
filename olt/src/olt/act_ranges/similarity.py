import torch
from torch.nn import functional as F

from olt.act_ranges.constants import CLUSTER_PATCH_SET_DIR
from olt.act_ranges.layer_utils import receptive_block


def mean_cosine_similarity(batch_tensor, ref_tensor):
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


def closest_pw(
    patch, w, layer_name, channel, cluster_patch_set_dir=CLUSTER_PATCH_SET_DIR
):
    # get the neuron's acts first
    # dict[cid, shape[b, ip-c, k, k]]
    cid_by_patches = torch.load(
        cluster_patch_set_dir / layer_name / str(channel) / "cid_by_patches.pt",
        weights_only=False,
    )
    cid_by_sim = {}
    for cid, patches in cid_by_patches.items():
        # multiply weight for pw similarity
        # patches: [B, C, K, K]
        # patch: [C, K, K]
        # w: [C, K, K]
        cid_by_sim[cid] = mean_cosine_similarity(w * patches, w * patch)

    max_sim = 0
    best_cid = -1
    for cid, sim in cid_by_sim.items():
        if max_sim < sim:
            max_sim = sim
            best_cid = cid
    return best_cid, max_sim, cid_by_patches[best_cid]


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
    layer = model.get_submodule(layer_name)
    y0, y1 = receptive_block(
        y, layer.kernel_size[0], layer.stride[0], layer.padding[0]
    )
    x0, x1 = receptive_block(
        x, layer.kernel_size[1], layer.stride[1], layer.padding[1]
    )

    patch = captured_acts[layer_name]["input"][b, :, y0:y1, x0:x1]  # [C, k, k]
    w = layer.weight[channel].detach().cpu().clone()  # [C, k, k]

    best_cid, max_sim, all_patches = closest_pw(
        patch, w, layer_name, channel, cluster_patch_set_dir
    )

    return best_cid, max_sim, patch, all_patches, w
