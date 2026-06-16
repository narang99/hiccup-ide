import gc
import io
import json
import shutil
import tempfile
import time
import typing
from collections import defaultdict
from pathlib import Path
from uuid import uuid4

import torch
import torch.nn.functional as F
import webdataset as wds
from PIL import Image
from torch import nn
from tqdm import tqdm


def tensor_to_bytes(t):
    buf = io.BytesIO()
    torch.save(t, buf)
    return buf.getvalue()


class PatchesMeta:
    """
    We want to track progress of which patches are done. This job has ended up being quite complicated right now because of the size of data
    We first extract the patches by doing a forward pass in the model for all inputs.
    We might not be able to store all of it on the disk, so we take a "channels_to_do" input. this is our main unit of work
    We keep track of which channels are done. End to end (everything generated, thats it).

    We keep track in out_dir/layer_name/meta.json
    It simply contains `channels_done` key which is a dict of all done channels
    Call this after you have done one full pass for storing activations
    """

    FILENAME = "meta.json"

    def __init__(
        self,
        out_dir: RemotePath,
        layer_name: str,
    ):
        self.meta_path = out_dir / layer_name / self.FILENAME

    def _read(self) -> dict:
        if self.meta_path.exists():
            with self.meta_path.open("r") as f:
                return json.load(f)
        return {"channels_done": {}}

    def are_all_channels_done(self, channels: list[int]) -> bool:
        done = self._read()["channels_done"]
        return all(c in done for c in channels)

    def mark_all_channels_done(self, channels: list[int]) -> None:
        meta = self._read()
        for c in channels:
            meta["channels_done"][c] = True
        with self.meta_path.open("w") as f:
            json.dump(meta, f)


class MultiShardWriter:
    """
    Utility class for maintaining shard writers for patch writing
    Patches are split across channels. So we keep a ShardWriter per channel
    This class provides a simple context manager for that.
    """

    def __init__(
        self,
        out_dir: RemotePath,
        layer_name: str,
        channels: list[int],
        max_objects_in_one_shard: int,
    ):
        self.channels = channels
        self.out_dir = out_dir
        self._writers = {}
        self._layer_name = layer_name
        self.max_objects_in_one_shard = max_objects_in_one_shard

    def __enter__(self):
        for channel in self.channels:
            pattern = self.out_dir / self._layer_name / str(channel) / "%06d.tar"
            self._writers[channel] = wds.ShardWriter(
                pattern, maxcount=self.max_objects_in_one_shard
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for writer in self._writers.values():
            writer.__exit__(exc_type, exc_val, exc_tb)
        return False

    def __getitem__(self, channel) -> wds.ShardWriter:
        return self._writers[channel]


def remote_mkdir(p: RemotePath):
    if isinstance(p, Path):
        p.mkdir(exist_ok=True, parents=True)


class IndexAndPatch(typing.TypedDict):
    index: torch.Tensor
    patch: torch.Tensor


class PatchExtractor:
    """
    For given input, attributions and thresholds, get all the patches where attribution > threshold and persist them in disk
    This is used for generating training data for the clusterer.
    """

    def __init__(
        self,
        all_labels: list[int],
        image_shards_base_dir: RemotePath,
        attribution_shards_base_dir: RemotePath,
        model: nn.Module,
        input_layer_name: str,
        current_layer_name: str,
        input_transform_fn,
        pos_thresholds: torch.Tensor,
        neg_thresholds: torch.Tensor,
        out_dir: RemotePath,
        device="cpu",
        input_shard_reader_bs=64,
        output_shard_writer_bs=32,
    ):
        self.all_labels = all_labels
        self.images_shards_base_dir = image_shards_base_dir
        self.attribution_shards_base_dir = attribution_shards_base_dir
        self.model = model
        self.input_layer_name = input_layer_name
        self.current_layer_name = current_layer_name
        self.input_transform_fn = input_transform_fn
        self.pos_thresholds = pos_thresholds
        self.neg_thresholds = neg_thresholds
        self.out_dir = out_dir
        self.device = device
        self.input_shard_reader_bs = input_shard_reader_bs
        self.output_shard_writer_bs = output_shard_writer_bs
        self.patches_meta = PatchesMeta(self.out_dir, self.current_layer_name)

    def extract(self, channels_to_keep: list[int]):
        if self.patches_meta.are_all_channels_done(channels_to_keep):
            print(f"SKIP: all channels are done for layer={self.current_layer_name}")
            return
        with MultiShardWriter(
            self.out_dir,
            self.current_layer_name,
            channels_to_keep,
            self.output_shard_writer_bs,
        ) as writers:
            for label_idx, label in tqdm(
                enumerate(self.all_labels), total=len(self.all_labels)
            ):
                if label_idx == 4:
                    print("hack, skipping 4, for perf chekcing")
                    continue
                self._extract_single_label(label, channels_to_keep, writers)
        self.patches_meta.mark_all_channels_done(channels_to_keep)

    def _extract_single_label(
        self,
        label: int,
        channels_to_keep: list[int],
        multi_shard_writer: MultiShardWriter,
    ):
        input_shards = raw_iter_shards(self.images_shards_base_dir / str(label))
        attribution_shards = raw_iter_shards(
            self.attribution_shards_base_dir / str(label) / self.current_layer_name
        )
        attribution_shard_name_to_path = {p.name: p for p in attribution_shards}
        with torch.no_grad():
            for input_shard in input_shards:
                attribution_shard = attribution_shard_name_to_path[input_shard.name]
                self._extract_for_one_shard(
                    input_shard, attribution_shard, multi_shard_writer, channels_to_keep
                )
                gc.collect()

    def _extract_for_one_shard(
        self,
        input_shard: RemotePath,
        attribution_shard: RemotePath,
        multi_shard_writer: MultiShardWriter,
        channels_to_keep: list[int],
    ):
        input_reader = read_image_shard(input_shard, self.input_shard_reader_bs, {})
        attribution_reader = read_attribution_shard(
            attribution_shard, self.input_shard_reader_bs, {}
        )
        for input_it_val, attribution_it_val in zip(input_reader, attribution_reader):
            (input_keys, input_list) = input_it_val
            (attr_keys, attributions) = attribution_it_val
            if input_keys != attr_keys:
                raise Exception(
                    f"found different keys in input and attribution while finding patches\n\tinput_keys={input_keys}\n\tattr_keys={attr_keys}"
                )
            indices, patches = self._get_indices_and_patches(input_list, attributions)
            self._persist_patches(
                indices, patches, channels_to_keep, multi_shard_writer
            )

    def _persist_patches(
        self,
        indices: torch.Tensor | None,
        patches: list[torch.Tensor] | None,
        channels_to_keep: list[int],
        multi_shard_writer: MultiShardWriter,
    ):
        if indices is not None and patches is not None:
            # indices are [B, C, H, W]
            # we only get the patches for which the index channel is in channels to keep
            # and write to the corresponding channel
            for i in range(len(indices)):
                ind, patch = indices[i], patches[i]
                chan = ind[1]
                if chan in channels_to_keep:
                    multi_shard_writer[chan].write(
                        {"__key__": str(uuid4()), "patch.pth": tensor_to_bytes(patch)}
                    )

    def _get_indices_and_patches(self, input_list, attributions):
        return get_indices_and_patches_for_batch(
            input_list,
            attributions,  # list[[C, H, W]]
            self.pos_thresholds,  # [C, 1, 1]
            self.neg_thresholds,  # [C, 1, 1]
            self.model,
            self.input_layer_name,
            self.current_layer_name,
            self.input_transform_fn,
            self.device,
        )


def get_indices_and_patches_for_batch(
    input_list: list[Image.Image],
    attributions: list[torch.Tensor],  # list[[C, H, W]]
    pos_thresh: torch.Tensor,  # [C, 1, 1]
    neg_thresh: torch.Tensor,  # [C, 1, 1]
    model: nn.Module,
    input_layer_name: str,
    current_layer_name: str,
    input_transform_fn,
    device: str = "cpu",
) -> tuple[torch.Tensor | None, list[torch.Tensor] | None]:
    input_tensor_list = [input_transform_fn(ip) for ip in input_list]
    input_batch = torch.stack(input_tensor_list).detach()
    # [B, C, H, W]
    attribution_batch = torch.stack(attributions).detach().to("cpu")
    # each index is 4 length, [B, C, H, W] coordinates
    indices = get_indices_of_patches_to_extract(
        attribution_batch, pos_thresh, neg_thresh
    )
    if len(indices) == 0:
        return None, None
    model = model.to(device)
    input_batch = input_batch.to(device)
    model_activations = get_layer_activations(input_batch, model, [input_layer_name])
    activation_batch = model_activations[input_layer_name]
    patches = patches_of_single_batch_with_indices(
        activation_batch, model.get_submodule(current_layer_name), indices
    )
    return indices, patches


def separate_indices_and_patches_for_each_input_key(
    indices: torch.Tensor | None, patches: list[torch.Tensor] | None, total_keys: int
) -> list[list[IndexAndPatch]]:
    """
    indices will be of the form [[B, C, H, W]] (a 2d tensor, the inner tensor always contains 4 integers)
    patches and indices are assumed to be the same length. total_keys is the number of input keys

    if the first dimension of the index is b, then that is the bth element from `total_keys`
    All we do is create a list of size total_keys
    And put all indices and patches of leading dimension `b` to list index `b`

    Index is now of shape [C, H, W]
    patches are usual, no changes in shape.
    """

    if indices is not None and patches is not None:
        # indices: [B, C, H, W]
        # patches: list[activation_shape] (i think, i need to check)
        # we return [(key, payload)]
        element_idx_by_payload: dict[int, list] = defaultdict(list)
        for i in range(len(indices)):
            ind, patch = indices[i], patches[i]
            element_idx = typing.cast(int, ind[0].item())
            if element_idx >= total_keys:
                raise Exception(
                    f"found index whose value was larger than the number of total keys passed, element_idx={element_idx}, total_keys={total_keys}"
                )
            element_idx_by_payload[element_idx].append(
                {"index": ind[1:], "patch": patch}
            )
        return [element_idx_by_payload[i] for i in range(total_keys)]
    elif indices is None and patches is None:
        res = []
        # empty list
        for _ in range(total_keys):
            res.append([])
        return res
    elif indices is None and patches is not None:
        raise Exception(
            f"invalid inputs, indices is None but patches is not, patches={patches}"
        )
    elif indices is not None and patches is None:
        raise Exception(
            f"invalid inputs, patches is None but indices is not, indices={indices}"
        )
    else:
        raise Exception("unhandled case, indices and patches, impossible")


def _separate_single_patch_shard_into_channels(
    indices_and_patches: list[IndexAndPatch],
    channels_to_keep: list[int],
) -> dict[int, list[IndexAndPatch]]:
    """
    indices_and_patches has index of shape [[C, H, W]]
    basically all the channels are mixed.
    We separate the channels (just like how we separated the examples before this step)

    This step then, removes the channel dim too.

    For index with channel `c`, it would be put in the dict with key `c`.
    index is now of shape [H, W]
    """
    channel_by_patches: dict[int, list[IndexAndPatch]] = defaultdict(list)
    for patch_payload in indices_and_patches:
        channel = typing.cast(int, patch_payload["index"][0].item())
        if channel in channels_to_keep:
            channel_by_patches[channel].append(
                {
                    "index": patch_payload["index"][1:],  # remove channel
                    "patch": patch_payload["patch"],
                }
            )
    return channel_by_patches


def get_indices_of_patches_to_extract(
    attribution: torch.Tensor, pos_thresh: torch.Tensor, neg_thresh: torch.Tensor
):
    pos_inds = torch.argwhere(attribution >= pos_thresh)
    neg_inds = torch.argwhere(attribution <= neg_thresh)
    return torch.cat([pos_inds, neg_inds]).detach().cpu()


def patches_of_single_batch_with_indices(
    input_act_of_batch: torch.Tensor, layer, indices: torch.Tensor
) -> list[torch.Tensor]:
    op_shape = get_output_shape(
        input_act_of_batch.shape,
        layer.kernel_size,
        layer.stride,
        layer.padding,
        layer.dilation,
    )
    op_r, op_c = op_shape[-2], op_shape[-1]
    b = input_act_of_batch.shape[0]

    patches = F.unfold(
        input_act_of_batch,
        layer.kernel_size,
        layer.dilation,
        layer.padding,
        layer.stride,
    ).reshape(b, -1, op_r, op_c)
    patches = patches.detach().cpu()

    return [patches[ind[0], :, ind[-2], ind[-1]] for ind in indices]


def get_output_shape(
    input_shape: tuple, ksize: tuple, stride: tuple, padding: tuple, dilation: tuple
):
    B, C, H, W = input_shape
    Ho = (H + 2 * padding[0] - dilation[0] * (ksize[0] - 1) - 1) // stride[0] + 1
    Wo = (W + 2 * padding[1] - dilation[1] * (ksize[1] - 1) - 1) // stride[1] + 1
    return (B, C, Ho, Wo)


# def read_patches_shard(
#     tar_path: RemotePath, batch_size: int
# ) -> typing.Iterator[tuple[list[str], list[list[dict]]]]:
#     """Read the shard at tar_path, and return a stream of [list[key], list[attribution-tensor]]

#     The returned tensor would be of shape [C, H, W] (no batch dimension)
#     key is the filename of the input from which this attribution was calculated
#     """
#     return read_torch_tensor_shard(tar_path, batch_size, "patches.pth", {})
