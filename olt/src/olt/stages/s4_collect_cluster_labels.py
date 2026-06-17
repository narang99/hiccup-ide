import typing
from typing import TypedDict

# from lucent.modelzoo import inceptionv1
import torch
from PIL import Image
from tqdm import tqdm

from ..attr import get_layer_attributions
from ..path import RemotePath
from ..shards import raw_iter_shards, read_image_shard
from .s2_collect_patches import (
    get_indices_and_patches_for_batch,
)
from .s3_cluster.run_hdbscan import TrainedModel, predict_labels


class ClusterConfig(TypedDict):
    clusterer: TrainedModel


class LayerClusterConfig(TypedDict):
    positive_thresholds: torch.Tensor  # [C, 1, 1]
    negative_thresholds: torch.Tensor  # [C, 1, 1]
    channel_by_clusterer: dict[int, ClusterConfig]


LayerByClusterConfig = dict[str, LayerClusterConfig]


class InstrumentedRow(TypedDict):
    cluster_label: int
    imagenet_label: int
    layer_name: str
    channel: int
    input_image_key: str
    location: tuple[int, int]


def _should_continue_processing(processing_count: int, total_count: int | None):
    if total_count is None:
        return True
    return processing_count < total_count


class ModelInstrumenter:
    """
    - Go through the input shards
    - get attribution for all layers which are registered
    - Threshold them
    - Call appropriate clusterer for registered channels
    - Prepare a CSV in the end

    A single row of this CSV:
    {
        "cluster_label": "...",
        "imagenet_label": "...",
        "layer_name": "...",
        "channel": "...",
        "input_image_key": "...", // image shard key
        "location": "..." // [h,w]
    }
    """

    def __init__(
        self,
        layer_by_cluster_config: LayerByClusterConfig,
        input_transform_fn,
        model,
        hdbscan_module,
        device="cpu",
    ):
        self.layer_by_cluster_config = layer_by_cluster_config
        self.input_transform_fn = input_transform_fn
        self.device = device
        self.model = model
        self.hdbscan_module = hdbscan_module
        self.layer_by_channel_by_weight = {}
        for layer_name in self.layer_by_cluster_config:
            cc = self.layer_by_cluster_config[layer_name]
            self.layer_by_channel_by_weight[layer_name] = {}
            for channel in cc["channel_by_clusterer"]:
                layer_weight = self.model.get_submodule(layer_name).weight[channel]
                layer_weight = layer_weight.detach().cpu().numpy().reshape(-1)
                self.layer_by_channel_by_weight[layer_name][channel] = layer_weight

    def process_all_in_image_base(
        self,
        images_base: RemotePath,
        shard_read_batch_size: int = 64,
        images_to_process_per_imagenet_label: int | None = None,
    ) -> list[InstrumentedRow]:
        all_labels = [int(p.name) for p in images_base.glob("*")]
        result = []
        for imagenet_label in tqdm(all_labels):
            result.extend(
                self.process_single_imagenet_label_dir(
                    images_base,
                    imagenet_label,
                    shard_read_batch_size,
                    images_to_process_per_imagenet_label,
                )
            )
        return result

    def process_single_imagenet_label_dir(
        self,
        images_base,
        imagenet_label,
        shard_read_batch_size,
        images_to_process_per_imagenet_label: int | None = None,
    ) -> list[InstrumentedRow]:
        result = []
        input_shards = raw_iter_shards(images_base / str(imagenet_label))
        processed_count = 0
        for shard in input_shards:
            for keys, images in read_image_shard(shard, shard_read_batch_size, {}):
                for key, image in zip(keys, images):
                    result.extend(self.process_single_input(key, image, imagenet_label))
                    processed_count += 1
                    if not _should_continue_processing(
                        processed_count, images_to_process_per_imagenet_label
                    ):
                        return result
        return result

    def process_single_input(
        self, key: str, image: Image.Image, imagenet_label: int
    ) -> list[InstrumentedRow]:
        result: list[InstrumentedRow] = []
        for layer in self.layer_by_cluster_config:
            result.extend(self.process_single_layer(key, image, layer, imagenet_label))
        return result

    def channels_to_keep(self, layer_name: str):
        cluster_config = self.layer_by_cluster_config[layer_name]
        return list(cluster_config["channel_by_clusterer"].keys())

    def process_single_layer(
        self, key: str, input_image: Image.Image, layer_name: str, imagenet_label: int
    ) -> list[InstrumentedRow]:
        # [B, C, H, W] (same as `timg`)
        timg = self.input_transform_fn(input_image)[None, ...].to(self.device)
        attributions = get_layer_attributions(
            timg, self.model, layer_name, imagenet_label, method="deeplift"
        )
        attributions = [att for att in attributions]
        cluster_config = self.layer_by_cluster_config[layer_name]
        indices, patches = get_indices_and_patches_for_batch(
            [input_image],
            attributions,
            cluster_config["positive_thresholds"],
            cluster_config["negative_thresholds"],
            self.model,
            layer_name,
            layer_name,
            self.input_transform_fn,
            device=self.device,
        )
        result: list[InstrumentedRow] = []
        if indices is not None and patches is not None:
            for ind, patch in zip(indices, patches):
                patch_res = self.process_single_patch(
                    ind, patch, layer_name, key, imagenet_label
                )
                if patch_res is not None:
                    result.append(patch_res)
        return result

    def process_single_patch(
        self,
        ind: torch.Tensor,
        patch: torch.Tensor,
        layer_name: str,
        key: str,
        imagenet_label: int,
    ) -> InstrumentedRow | None:
        cluster_config = self.layer_by_cluster_config[layer_name]
        channel = typing.cast(int, ind[0].item())
        if channel in self.channels_to_keep(layer_name):
            weight = self.layer_by_channel_by_weight[layer_name][channel]
            patch = patch[None].numpy()
            pws = patch * weight
            labels = predict_labels(
                pws,
                self.hdbscan_module,
                cluster_config["channel_by_clusterer"][channel]["clusterer"],
            )
            return {
                "cluster_label": labels[0],
                "imagenet_label": imagenet_label,
                "layer_name": layer_name,
                "channel": channel,
                "input_image_key": key,
                "location": (
                    typing.cast(int, ind[1].item()),
                    typing.cast(int, ind[2].item()),
                ),
            }
        else:
            return None
