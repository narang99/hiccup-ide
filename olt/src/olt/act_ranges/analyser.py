from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from olt.act import InputOutputModelSnapshot
from olt.act_ranges.constants import UNSUPPORTED_CURRENT_LAYERS
from olt.act_ranges.dependency_match import DependencyMatch
from olt.act_ranges.interactive_plots import show_cluster_match
from olt.act_ranges.layer_utils import get_layer_params, receptive_block
from olt.act_ranges.similarity import get_neuron_closest_cluster
from olt.act_ranges.stats import get_noise_range, indices_for_percentage, shorth
from olt.tfms import transform


def _scalar(value):
    return value.item() if hasattr(value, "item") else value


def _stats_row(current_layer_name, current_channel, y, x, kind, dep_layer_name, dep_channel, match, cluster_dist):
    cluster_min, cluster_med, cluster_max = cluster_dist
    return {
        "origin_layer": current_layer_name,
        "origin_channel": current_channel,
        "origin_y": y,
        "origin_x": x,
        "similarity": _scalar(match.best_sim),
        "noise_distance": _scalar(match.point_dist),
        "output_activation": _scalar(match.op_act),
        "above_noise_ratio": match.ratio,
        "kind": kind,
        "dep_layer": dep_layer_name,
        "dep_channel": dep_channel,
        "dep_cid": match.best_cid,
        "best_cluster_min": cluster_min,
        "best_cluster_med": cluster_med,
        "best_cluster_max": cluster_max,
    }


def _add_pw_sample(pw_samples, dep_layer_name, dep_channel, match, image_key, dep_y, dep_x):
    cid_samples = (
        pw_samples.setdefault(dep_layer_name, {})
        .setdefault(str(dep_channel), {})
        .setdefault(str(match.best_cid), [])
    )
    cid_samples.append(
        {
            "dep_patch": match.dep_patch,
            "dep_w": match.dep_w,
            "image_key": image_key,
            "dep_y": dep_y,
            "dep_x": dep_x,
        }
    )


class NeuronParentAnalyser:
    """
    Analyses dependencies for a single (current_layer_name, current_channel)
    neuron. f_pad_manual_before_between_us_and_dep and flattened_channel_map
    are only valid for that one current_layer_name, so it's fixed at
    construction; y/x/current_act still vary per call, since the same
    current_layer/channel is typically probed at many spatial positions
    across many images.
    """

    def __init__(
        self,
        model,
        current_layer_name,
        current_channel,
        all_layers,
        layer_by_channel_by_label_by_points,
        layer_by_channel_by_noise,
        cluster_patch_set_dir,
        base_report_dir,
        flat_image_dir,
        f_pad_manual_before_between_us_and_dep,
        flattened_channel_map,
    ):
        if current_layer_name in UNSUPPORTED_CURRENT_LAYERS:
            raise ValueError(
                f"NeuronParentAnalyser does not support current_layer_name={current_layer_name!r}: "
                "see olt/act_ranges/constants/paddings.py for why."
            )
        self.model = model
        self.current_layer_name = current_layer_name
        self.current_channel = current_channel
        self.all_layers = all_layers
        self.layer_by_channel_by_label_by_points = layer_by_channel_by_label_by_points
        self.layer_by_channel_by_noise = layer_by_channel_by_noise
        self.cluster_patch_set_dir = cluster_patch_set_dir
        self.flat_image_dir = flat_image_dir
        self.f_pad_manual_before_between_us_and_dep = (
            f_pad_manual_before_between_us_and_dep
        )
        self.base_report_dir = base_report_dir
        self.flattened_channel_map = flattened_channel_map

    def get_activations_for_image(self, image_key, device="cpu"):
        """Run the forward pass for an image; caller passes the result into the other methods."""
        timg = transform(Image.open(self.flat_image_dir / f"{image_key}.jpeg"))[
            None
        ].to(device)
        return InputOutputModelSnapshot.get_activations(
            timg, self.model, self.all_layers
        )

    def top_contributing_indices(self, y, x, current_act, sum_upto_percent, kind):
        """
        Shared by collect_cluster_above_noise_ratios and plot_clusters:
        finds the flattened-input indices that account for `sum_upto_percent`
        of the patch*weight contribution at (current_layer_name, current_channel, y, x).
        """
        w, ksize, stride, padding = get_layer_params(
            self.model, self.current_layer_name, self.current_channel
        )

        y0, y1 = receptive_block(y, ksize[0], stride[0], padding[0])
        x0, x1 = receptive_block(x, ksize[1], stride[1], padding[1])
        patch = current_act[self.current_layer_name]["input"][0, :, y0:y1, x0:x1]
        pw = patch * w
        pos_indices, pos_fracs, neg_indices, neg_fracs = indices_for_percentage(
            pw, sum_upto_percent
        )
        indices = pos_indices if kind == "positive" else neg_indices
        return y0, x0, patch, indices

    def dep_coords(self, y0, x0, cur_rel_y, cur_rel_x):
        dep_y = y0 + cur_rel_y - self.f_pad_manual_before_between_us_and_dep[0]
        dep_x = x0 + cur_rel_x - self.f_pad_manual_before_between_us_and_dep[1]
        return dep_y, dep_x

    def iter_dependency_coords(self, y0, x0, indices):
        """Shared by every traversal below: maps each flattened contributing
        index to its (dep_layer_name, dep_channel, dep_y, dep_x) location."""
        for cur_chan, cur_rel_y, cur_rel_x in indices:
            dep_layer_name, dep_channel = (
                self.flattened_channel_map.get_layer_and_chan_from_flattened(cur_chan)
            )
            dep_y, dep_x = self.dep_coords(y0, x0, cur_rel_y, cur_rel_x)
            yield dep_layer_name, dep_channel, dep_y, dep_x

    def closest_cluster_info(self, layer_name, channel, y, x, current_act):
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
    def ratio_for_cluster(label_by_points, noise, cid):
        _, noise_max = get_noise_range(noise)
        dep_points = np.array(label_by_points[str(cid)])
        return (dep_points > noise_max).sum() / len(dep_points)

    def get_noise_stats(self, noise):
        noise_min, noise_max = get_noise_range(noise)
        noise_med = np.median(noise)

        tol = 1e-6
        noise_radius = noise_max - noise_med + tol

        return noise_min, noise_med, noise_max, noise_radius

    def get_activation_value(self, layer_name, channel, y, x, current_act):
        return current_act[layer_name]["output"][0, channel, y, x]

    def get_activation_distance_from_noise(
        self, layer_name, channel, y, x, current_act
    ):
        op_act = self.get_activation_value(layer_name, channel, y, x, current_act)
        noise = self.layer_by_channel_by_noise[layer_name][str(channel)]
        _, noise_med, _, noise_radius = self.get_noise_stats(noise)
        return (op_act - noise_med) / noise_radius

    def get_cluster_distance_from_noise(self, cluster_points, noise):
        cluster_points = np.array(cluster_points)
        cluster_min, cluster_max = shorth(cluster_points)
        cluster_med = np.median(cluster_points)

        _, noise_med, _, noise_radius = self.get_noise_stats(noise)
        cluster_med_dist = (cluster_med - noise_med) / noise_radius
        cluster_max_dist = (cluster_max - noise_med) / noise_radius
        cluster_min_dist = (cluster_min - noise_med) / noise_radius

        return cluster_min_dist, cluster_med_dist, cluster_max_dist

    def get_cluster_above_noise_ratio(self, layer_name, channel, y, x, current_act):
        best_cid, _, _, _, _ = self.closest_cluster_info(
            layer_name, channel, y, x, current_act
        )
        noise = self.layer_by_channel_by_noise[layer_name][str(channel)]
        label_by_points = self.layer_by_channel_by_label_by_points[layer_name][
            str(channel)
        ]
        return self.ratio_for_cluster(label_by_points, noise, best_cid)

    def resolve_dependency_match(
        self, dep_layer_name, dep_channel, dep_y, dep_x, current_act
    ):
        """
        Shared by plot_clusters and collect_cluster_stats_df: resolves the
        closest cluster, noise stats, and above-noise ratio for a dependency
        neuron at (dep_layer_name, dep_channel, dep_y, dep_x).
        """
        best_cid, best_sim, dep_patch, best_patches, dep_w = self.closest_cluster_info(
            dep_layer_name, dep_channel, dep_y, dep_x, current_act
        )

        noise = self.layer_by_channel_by_noise[dep_layer_name][str(dep_channel)]
        label_by_points = self.layer_by_channel_by_label_by_points[dep_layer_name][
            str(dep_channel)
        ]

        ratio = self.ratio_for_cluster(label_by_points, noise, best_cid)
        point_dist = self.get_activation_distance_from_noise(
            dep_layer_name, dep_channel, dep_y, dep_x, current_act
        )
        op_act = self.get_activation_value(
            dep_layer_name, dep_channel, dep_y, dep_x, current_act
        )

        return DependencyMatch(
            dep_layer_name=dep_layer_name,
            dep_channel=dep_channel,
            dep_y=dep_y,
            dep_x=dep_x,
            best_cid=best_cid,
            best_sim=best_sim,
            dep_patch=dep_patch,
            best_patches=best_patches,
            dep_w=dep_w,
            noise=noise,
            label_by_points=label_by_points,
            ratio=ratio,
            point_dist=point_dist,
            op_act=op_act,
        )

    def collect_cluster_distances(
        self,
        y,
        x,
        current_act,
        sum_upto_percent=0.9,
        kind="positive",
    ):
        y0, x0, patch, indices = self.top_contributing_indices(
            y, x, current_act, sum_upto_percent, kind
        )

        dists = []
        for dep_layer_name, dep_channel, dep_y, dep_x in self.iter_dependency_coords(
            y0, x0, indices
        ):
            r = self.get_activation_distance_from_noise(
                dep_layer_name, dep_channel, dep_y, dep_x, current_act
            )
            dists.append(r)
        return dists

    def collect_cluster_above_noise_ratios(
        self,
        y,
        x,
        current_act,
        sum_upto_percent=0.9,
        kind="positive",
    ):
        y0, x0, patch, indices = self.top_contributing_indices(
            y, x, current_act, sum_upto_percent, kind
        )

        noise_ratios = []
        for dep_layer_name, dep_channel, dep_y, dep_x in self.iter_dependency_coords(
            y0, x0, indices
        ):
            r = self.get_cluster_above_noise_ratio(
                dep_layer_name, dep_channel, dep_y, dep_x, current_act
            )
            noise_ratios.append(r)
        return noise_ratios

    def plot_clusters(
        self,
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
        y0, x0, patch, indices = self.top_contributing_indices(
            y, x, current_act, sum_upto_percent, kind
        )

        for cur_chan, cur_rel_y, cur_rel_x in indices:
            if getattr(filter_fn, "is_full", False):
                break  # filter won't accept any more, stop doing lookups

            dep_layer_name, dep_channel = (
                self.flattened_channel_map.get_layer_and_chan_from_flattened(cur_chan)
            )
            dep_y, dep_x = self.dep_coords(y0, x0, cur_rel_y, cur_rel_x)

            match = self.resolve_dependency_match(
                dep_layer_name, dep_channel, dep_y, dep_x, current_act
            )

            if not filter_fn(match.ratio, match.point_dist):
                continue

            show_cluster_match(
                self.base_report_dir,
                dep_layer_name,
                dep_channel,
                match,
                patch,
                cur_chan,
                cur_rel_y,
                cur_rel_x,
                stuff_to_show,
                max_percent_points_allowed_in_noise_for_one_cluster,
            )

    def collect_cluster_stats_df(
        self,
        y,
        x,
        current_act,
        image_key=None,
        sum_upto_percent=0.9,
        kind="positive",
        csv_path=None,
        append=False,
    ):
        """
        Same traversal as plot_clusters, but instead of plotting, collects
        one row per contributing (dep_layer, dep_channel) with: origin
        neuron info, similarity, noise distance, raw output activation,
        above-noise ratio, kind, dep_layer, dep_channel, dep_cid. Returns
        (df, pw_samples):

        - df: a DataFrame, scalars only (CSV-safe); optionally written/appended
          to csv_path.
        - pw_samples: dict[dep_layer_name, dict[str(dep_channel), dict[str(dep_cid),
          list[{"dep_patch", "dep_w", "image_key", "dep_y", "dep_x"}]]]] — one
          entry per firing, appended to the list for whichever cluster it
          matched. dep_patch/dep_w are the raw tensors behind this firing's
          pointwise-multiplication sample (see similarity.get_neuron_closest_cluster);
          kept out of df so they never need to survive a CSV round-trip.
          image_key identifies which input image this firing came from — pass
          it in so callers merging pw_samples across many images (accumulating
          into one dict, mirroring how layer_by_channel_by_label_by_points is
          built up) can tell samples apart. best_patches (the cluster's own
          patch population) is deliberately NOT stored here — it's identical
          for every firing sharing a dep_cid, so storing it per-firing would
          duplicate the whole cluster population once per firing; report-time
          rendering should load it once per matched cluster instead (see
          similarity.load_cluster_patches).
        """
        y0, x0, patch, indices = self.top_contributing_indices(
            y, x, current_act, sum_upto_percent, kind
        )

        rows = []
        pw_samples = {}
        for dep_layer_name, dep_channel, dep_y, dep_x in self.iter_dependency_coords(
            y0, x0, indices
        ):
            match = self.resolve_dependency_match(
                dep_layer_name, dep_channel, dep_y, dep_x, current_act
            )

            cluster_dist = self.get_cluster_distance_from_noise(
                match.label_by_points[str(match.best_cid)], match.noise
            )
            rows.append(
                _stats_row(
                    self.current_layer_name,
                    self.current_channel,
                    y,
                    x,
                    kind,
                    dep_layer_name,
                    dep_channel,
                    match,
                    cluster_dist,
                )
            )
            _add_pw_sample(pw_samples, dep_layer_name, dep_channel, match, image_key, dep_y, dep_x)

        df = pd.DataFrame(rows)

        if csv_path is not None:
            write_header = not (append and Path(csv_path).exists())
            df.to_csv(
                csv_path, mode="a" if append else "w", header=write_header, index=False
            )

        return df, pw_samples
