from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from olt.act import InputOutputModelSnapshot
from olt.show import show_single_channel_red_green_black as S
from olt.tfms import transform

CLUSTER_PATCH_SET_DIR = Path("cluster-patches-set")

LAYER_NAME_BY_SHAPE = {
    "mixed4d_1x1_pre_relu_conv": (16, 32),
    "mixed4d_3x3_pre_relu_conv": (36, 36),
    "mixed4d_pool_reduce_pre_relu_conv": (16, 32),
    "mixed4d_5x5_pre_relu_conv": (25, 32),
    "mixed4e_1x1_pre_relu_conv": (22, 24),
}

KELLY_COLORS = [
    "#F2F3F4",  # white (skip as bg reference)
    "#F3C300",
    "#875692",
    "#F38400",
    "#A1CAF1",
    "#BE0032",
    "#C2B280",
    "#848482",
    "#008856",
    "#E68FAC",
    "#0067A5",
    "#F99379",
    "#604E97",
    "#F6A600",
    "#B3446C",
    "#DCD300",
    "#882D17",
    "#8DB600",
    "#654522",
    "#E25822",
    "#2B3D26",
]


def _get_layer_params(model, current_layer_name, current_channel):
    layer = model.get_submodule(current_layer_name)
    padding = layer.padding
    ksize, stride = layer.kernel_size, layer.stride
    w = layer.weight[current_channel].detach().cpu()
    return w, ksize, stride, padding


def indices_for_percentage(arr, pct):
    arr = np.array(arr)
    shape = arr.shape
    flat = arr.ravel()

    pos_idx = np.where(flat > 0)[0]
    neg_idx = np.where(flat < 0)[0]

    def top_indices(idx, values, target_pct):
        order = idx[np.argsort(-np.abs(values[idx]))]
        total = np.abs(values[order]).sum()
        target = total * target_pct
        cumsum = np.cumsum(np.abs(values[order]))
        cutoff = min(np.searchsorted(cumsum, target) + 1, len(order))
        frac_done = cumsum[:cutoff] / total
        return order[:cutoff], frac_done

    pos_flat, pos_frac = top_indices(pos_idx, flat, pct)
    neg_flat, neg_frac = top_indices(neg_idx, flat, pct)

    pos_result = np.stack(np.unravel_index(pos_flat, shape), axis=-1)
    neg_result = np.stack(np.unravel_index(neg_flat, shape), axis=-1)

    return pos_result, pos_frac, neg_result, neg_frac


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


def _get_layer_and_chan_from_flattened(flattened_chan):
    if flattened_chan < 112:
        return "mixed4d_1x1_pre_relu_conv", flattened_chan
    elif flattened_chan < 400:
        return "mixed4d_3x3_pre_relu_conv", flattened_chan - 112
    elif flattened_chan < 464:
        return "mixed4d_5x5_pre_relu_conv", flattened_chan - 400
    else:
        return "mixed4d_pool_reduce_pre_relu_conv", flattened_chan - 464


def shorth(data, frac=0.5):
    data_sorted = np.sort(data)
    n = len(data_sorted)
    window = int(np.ceil(frac * n))
    min_width = np.inf
    best_start = 0
    for i in range(n - window + 1):
        width = data_sorted[i + window - 1] - data_sorted[i]
        if width < min_width:
            min_width = width
            best_start = i
    return data_sorted[best_start], data_sorted[best_start + window - 1]


def get_noise_range(data, frac=0.9):
    return shorth(data, frac)


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
    y0, y1 = _receptive_block(
        y, layer.kernel_size[0], layer.stride[0], layer.padding[0]
    )
    x0, x1 = _receptive_block(
        x, layer.kernel_size[1], layer.stride[1], layer.padding[1]
    )

    patch = captured_acts[layer_name]["input"][b, :, y0:y1, x0:x1]  # [C, k, k]
    w = layer.weight[channel].detach().cpu().clone()  # [C, k, k]

    best_cid, max_sim, all_patches = closest_pw(
        patch, w, layer_name, channel, cluster_patch_set_dir
    )

    return best_cid, max_sim, patch, all_patches, w


class NoiseRatioRangeFilter:
    """
    Filters clusters by their noise-above ratio, and caps how many
    get plotted. Once `max_clusters` have been accepted, all further
    calls return False (no more plotting).
    """

    def __init__(self, min_d, max_d, max_clusters, kind="ratio"):
        self.min_d = min_d
        self.max_d = max_d
        self.max_clusters = max_clusters
        self.count = 0
        self.kind = kind

    def __call__(self, ratio, point_dist):
        d = ratio if self.kind == "ratio" else point_dist
        if self.count >= self.max_clusters:
            return False
        if not (self.min_d <= d <= self.max_d):
            return False
        self.count += 1
        return True

    @property
    def is_full(self):
        return self.count >= self.max_clusters


class NeuronParentAnalyser:
    def __init__(
        self,
        model,
        all_layers,
        layer_by_channel_by_label_by_points,
        layer_by_channel_by_noise,
        cluster_patch_set_dir,
        base_report_dir,
        flat_image_dir,
        f_pad_manual_before_between_us_and_dep,
    ):
        self.model = model
        self.all_layers = all_layers
        self.layer_by_channel_by_label_by_points = layer_by_channel_by_label_by_points
        self.layer_by_channel_by_noise = layer_by_channel_by_noise
        self.cluster_patch_set_dir = cluster_patch_set_dir
        self.flat_image_dir = flat_image_dir
        self.f_pad_manual_before_between_us_and_dep = (
            f_pad_manual_before_between_us_and_dep
        )
        self.base_report_dir = base_report_dir

    def get_activations_for_image(self, image_key):
        """Run the forward pass for an image; caller passes the result into the other methods."""
        timg = transform(Image.open(self.flat_image_dir / f"{image_key}.jpeg"))[None]
        return InputOutputModelSnapshot.get_activations(
            timg, self.model, self.all_layers
        )

    def _top_contributing_indices(
        self, layer_name, channel, y, x, current_act, sum_upto_percent, kind
    ):
        """
        Shared by collect_cluster_above_noise_ratios and plot_clusters:
        finds the flattened-input indices that account for `sum_upto_percent`
        of the patch*weight contribution at (layer_name, channel, y, x).
        """
        w, ksize, stride, padding = _get_layer_params(self.model, layer_name, channel)

        y0, y1 = _receptive_block(y, ksize[0], stride[0], padding[0])
        x0, x1 = _receptive_block(x, ksize[1], stride[1], padding[1])
        patch = current_act[layer_name]["input"][0, :, y0:y1, x0:x1]
        pw = patch * w
        pos_indices, pos_fracs, neg_indices, neg_fracs = indices_for_percentage(
            pw, sum_upto_percent
        )
        indices = pos_indices if kind == "positive" else neg_indices
        return y0, x0, patch, indices

    def _dep_coords(self, y0, x0, cur_rel_y, cur_rel_x):
        dep_y = y0 + cur_rel_y - self.f_pad_manual_before_between_us_and_dep[0]
        dep_x = x0 + cur_rel_x - self.f_pad_manual_before_between_us_and_dep[1]
        return dep_y, dep_x

    def _closest_cluster_info(self, layer_name, channel, y, x, current_act):
        """
        Finds the cluster whose patches are most similar (weighted by conv
        weight) to the activation patch at (layer_name, channel, y, x).
        """
        best_cid, best_sim, dep_patch, best_patches, w = get_neuron_closest_cluster(
            self.model,
            layer_name,
            0,
            channel,
            y,
            x,
            current_act,
            self.cluster_patch_set_dir,
        )
        return best_cid, best_sim, dep_patch, best_patches, w

    @staticmethod
    def _ratio_for_cluster(label_by_points, noise, cid):
        _, noise_max = get_noise_range(noise)
        dep_points = np.array(label_by_points[str(cid)])
        return (dep_points > noise_max).sum() / len(dep_points)

    def _get_noise_stats(self, noise):
        noise_min, noise_max = get_noise_range(noise)
        noise_med = np.median(noise)

        tol = 1e-6
        noise_radius = noise_max - noise_med + tol

        return noise_min, noise_med, noise_max, noise_radius

    def _get_activation_distance_from_noise(
        self, layer_name, channel, y, x, current_act
    ):
        op_act = current_act[layer_name]["output"][0, channel, y, x]
        noise = self.layer_by_channel_by_noise[layer_name][str(channel)]
        _, noise_med, _, noise_radius = self._get_noise_stats(noise)
        return (op_act - noise_med) / noise_radius

    def _get_cluster_distance_from_noise(self, cluster_points, noise):
        cluster_points = np.array(cluster_points)
        cluster_min, cluster_max = shorth(cluster_points)
        cluster_med = np.median(cluster_points)

        _, noise_med, _, noise_radius = self._get_noise_stats(noise)
        cluster_med_dist = (cluster_med - noise_med) / noise_radius
        cluster_max_dist = (cluster_max - noise_med) / noise_radius
        cluster_min_dist = (cluster_min - noise_med) / noise_radius

        return cluster_min_dist, cluster_med_dist, cluster_max_dist

    def _get_cluster_above_noise_ratio(self, layer_name, channel, y, x, current_act):
        best_cid, _, _, _, _ = self._closest_cluster_info(
            layer_name, channel, y, x, current_act
        )
        noise = self.layer_by_channel_by_noise[layer_name][str(channel)]
        label_by_points = self.layer_by_channel_by_label_by_points[layer_name][
            str(channel)
        ]
        return self._ratio_for_cluster(label_by_points, noise, best_cid)

    def collect_cluster_distances(
        self,
        current_layer_name,
        current_channel,
        y,
        x,
        current_act,
        sum_upto_percent=0.9,
        kind="positive",
    ):
        y0, x0, patch, indices = self._top_contributing_indices(
            current_layer_name,
            current_channel,
            y,
            x,
            current_act,
            sum_upto_percent,
            kind,
        )

        dists = []
        for cur_chan, cur_rel_y, cur_rel_x in indices:
            dep_layer_name, dep_channel = _get_layer_and_chan_from_flattened(cur_chan)
            dep_y, dep_x = self._dep_coords(y0, x0, cur_rel_y, cur_rel_x)

            r = self._get_activation_distance_from_noise(
                dep_layer_name, dep_channel, dep_y, dep_x, current_act
            )
            dists.append(r)
        return dists

    def collect_cluster_above_noise_ratios(
        self,
        current_layer_name,
        current_channel,
        y,
        x,
        current_act,
        sum_upto_percent=0.9,
        kind="positive",
    ):
        y0, x0, patch, indices = self._top_contributing_indices(
            current_layer_name,
            current_channel,
            y,
            x,
            current_act,
            sum_upto_percent,
            kind,
        )

        noise_ratios = []
        for cur_chan, cur_rel_y, cur_rel_x in indices:
            dep_layer_name, dep_channel = _get_layer_and_chan_from_flattened(cur_chan)
            dep_y, dep_x = self._dep_coords(y0, x0, cur_rel_y, cur_rel_x)

            r = self._get_cluster_above_noise_ratio(
                dep_layer_name, dep_channel, dep_y, dep_x, current_act
            )
            noise_ratios.append(r)
        return noise_ratios

    def plot_clusters(
        self,
        current_layer_name,
        current_channel,
        y,
        x,
        current_act,
        filter_fn,
        sum_upto_percent=0.9,
        kind="positive",
        max_percent_points_allowed_in_noise_for_one_cluster=25,
        stuff_to_show=None,
    ):
        # filter_fn(ratio) -> bool: decides whether this cluster gets plotted
        if stuff_to_show is None:
            stuff_to_show = ["heatmap", "pw", "act_range"]
        y0, x0, patch, indices = self._top_contributing_indices(
            current_layer_name,
            current_channel,
            y,
            x,
            current_act,
            sum_upto_percent,
            kind,
        )
        colors = KELLY_COLORS[1:]

        for cur_chan, cur_rel_y, cur_rel_x in indices:
            if getattr(filter_fn, "is_full", False):
                break  # filter won't accept any more, stop doing lookups

            dep_layer_name, dep_channel = _get_layer_and_chan_from_flattened(cur_chan)
            dep_y, dep_x = self._dep_coords(y0, x0, cur_rel_y, cur_rel_x)

            best_cid, best_sim, dep_patch, best_patches, dep_w = (
                self._closest_cluster_info(
                    dep_layer_name, dep_channel, dep_y, dep_x, current_act
                )
            )

            noise = self.layer_by_channel_by_noise[dep_layer_name][str(dep_channel)]
            label_by_points = self.layer_by_channel_by_label_by_points[dep_layer_name][
                str(dep_channel)
            ]

            ratio = self._ratio_for_cluster(label_by_points, noise, best_cid)
            point_dist = self._get_activation_distance_from_noise(
                dep_layer_name, dep_channel, dep_y, dep_x, current_act
            )

            if not filter_fn(ratio, point_dist):
                continue

            l0 = get_labels_above_noise_range(
                label_by_points,
                noise,
                max_percent_points_allowed_in_noise_for_one_cluster,
            )
            labels = list(l0.keys())
            color_map = {l: colors[i % len(colors)] for i, l in enumerate(labels)}

            print(
                f"############### {dep_layer_name}:{dep_channel}, {len(labels)}, "
                f"cid={best_cid}, sim={best_sim.item()}, ratio={ratio:.3f} dist={point_dist:.3f}"
                f"#########################"
            )
            if "act_range" in stuff_to_show:
                _, axes = plt.subplots(1, 2, sharey=True, figsize=(15, 4))
                for label in labels:
                    ys = l0[label]
                    axes[0].scatter(
                        range(len(ys)), ys, color=color_map[label], label=label
                    )

                axes[0].axhline(
                    patch[cur_chan, cur_rel_y, cur_rel_x].item(),
                    color="white",
                    linestyle="--",
                )
                axes[1].scatter(range(len(noise)), noise)

                axes[0].legend()
                plt.show()

            if len(labels) > 0:
                if "heatmap" in stuff_to_show:
                    cluster_photo = get_cluster_photo(
                        self.base_report_dir,
                        dep_layer_name,
                        dep_channel,
                        best_cid,
                        "combined",
                        crop_max_height=470,
                    )
                    if cluster_photo is not None:
                        plt.imshow(cluster_photo)
                        plt.show()

                if "pw" in stuff_to_show:
                    best_pws = [dep_w * p for p in best_patches[:3]]
                    dep_pw = dep_w * dep_patch
                    all_pws = [dep_pw] + best_pws
                    all_pws = [
                        p.reshape(LAYER_NAME_BY_SHAPE[dep_layer_name]) for p in all_pws
                    ]

                    S(all_pws, 10, 4)
                    plt.show()

    def collect_cluster_stats_df(
        self,
        current_layer_name,
        current_channel,
        y,
        x,
        current_act,
        sum_upto_percent=0.9,
        kind="positive",
        csv_path=None,
        append=False,
    ):
        """
        Same traversal as plot_clusters, but instead of plotting, collects
        one row per contributing (dep_layer, dep_channel) with: origin
        neuron info, similarity, noise distance, above-noise ratio, kind,
        dep_layer, dep_channel, dep_cid. Returns a DataFrame; optionally
        writes/appends to csv_path.
        """
        y0, x0, patch, indices = self._top_contributing_indices(
            current_layer_name,
            current_channel,
            y,
            x,
            current_act,
            sum_upto_percent,
            kind,
        )

        rows = []
        for cur_chan, cur_rel_y, cur_rel_x in indices:
            dep_layer_name, dep_channel = _get_layer_and_chan_from_flattened(cur_chan)
            dep_y, dep_x = self._dep_coords(y0, x0, cur_rel_y, cur_rel_x)

            best_cid, best_sim, dep_patch, best_patches, dep_w = (
                self._closest_cluster_info(
                    dep_layer_name, dep_channel, dep_y, dep_x, current_act
                )
            )

            noise = self.layer_by_channel_by_noise[dep_layer_name][str(dep_channel)]
            label_by_points = self.layer_by_channel_by_label_by_points[dep_layer_name][
                str(dep_channel)
            ]

            ratio = self._ratio_for_cluster(label_by_points, noise, best_cid)
            point_dist = self._get_activation_distance_from_noise(
                dep_layer_name, dep_channel, dep_y, dep_x, current_act
            )

            cluster_min, cluster_med, cluster_max = (
                self._get_cluster_distance_from_noise(
                    label_by_points[str(best_cid)], noise
                )
            )

            rows.append(
                {
                    "origin_layer": current_layer_name,
                    "origin_channel": current_channel,
                    "origin_y": y,
                    "origin_x": x,
                    "similarity": best_sim.item()
                    if hasattr(best_sim, "item")
                    else best_sim,
                    "noise_distance": point_dist.item()
                    if hasattr(point_dist, "item")
                    else point_dist,
                    "above_noise_ratio": ratio,
                    "kind": kind,
                    "dep_layer": dep_layer_name,
                    "dep_channel": dep_channel,
                    "dep_cid": best_cid,
                    "best_cluster_min": cluster_min,
                    "best_cluster_med": cluster_med,
                    "best_cluster_max": cluster_max,
                }
            )

        df = pd.DataFrame(rows)

        if csv_path is not None:
            write_header = not (append and Path(csv_path).exists())
            df.to_csv(
                csv_path, mode="a" if append else "w", header=write_header, index=False
            )

        return df
