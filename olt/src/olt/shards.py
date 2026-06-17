import io
from typing import Iterator

import torch
import webdataset as wds
from PIL import Image

from .path import RemotePath


def read_image_shard(
    tar_path: RemotePath, batch_size: int, wds_cache_kwargs: dict
) -> Iterator[tuple[list[str], list[Image.Image]]]:
    dataset = (
        wds.WebDataset(str(tar_path), shardshuffle=False, **wds_cache_kwargs)  # ty: ignore
        .decode("pil")
        .to_tuple("__key__", "jpg")
        .batched(batch_size, collation_fn=lambda samples: list(zip(*samples)))
    )
    for keys, images in dataset:
        yield list(keys), list(images)


def raw_iter_shards(d: RemotePath):
    return list(d.glob("*.tar"))


def tensor_to_bytes(t):
    buf = io.BytesIO()
    torch.save(t, buf)
    return buf.getvalue()


def pth_decoder(key, data):
    if key.endswith(".pth"):
        return torch.load(io.BytesIO(data))
    return None


def read_torch_tensor_shard(
    tar_path: RemotePath, batch_size: int, tensor_key: str, wds_cache_kwargs: dict
) -> Iterator[tuple[list[str], list[torch.Tensor]]]:
    """Utility for decoding shards which contain torch tensors
    The shards shouild contain data of type {__key__: <file-name-of-input>, {tensor_key}: bytes of tensor which can be decoded using pth_decoder}
    """
    dataset = (
        wds.WebDataset(str(tar_path), shardshuffle=False, **wds_cache_kwargs)  # ty: ignore
        .decode(pth_decoder)
        .to_tuple("__key__", tensor_key)
        .batched(batch_size, collation_fn=lambda samples: list(zip(*samples)))
    )
    for keys, attributions in dataset:
        yield list(keys), list(attributions)


def read_attribution_shard(
    tar_path: RemotePath, batch_size: int, wds_cache_kwargs: dict
) -> Iterator[tuple[list[str], list[torch.Tensor]]]:
    """Read the shard at tar_path, and return a stream of [list[key], list[attribution-tensor]]

    The returned tensor would be of shape [C, H, W] (no batch dimension)
    key is the filename of the input from which this attribution was calculated
    """
    return read_torch_tensor_shard(
        tar_path, batch_size, "attribution.pth", wds_cache_kwargs
    )


def read_patches_shard(
    tar_path: RemotePath, batch_size: int
) -> Iterator[
    tuple[list[str], list[torch.Tensor], list[torch.Tensor], list[list[str]]]
]:
    """Read the shard at tar_path, and return a stream of
    [list[key], list[patch-tensor], list[index-tensor], list[list[input-key]]]

    patch-tensor and index-tensor are stacked tensors.
    patch at index i has index at index tensor position i, with input corresponding to input-keys[i]

    The returned tensor would be of shape [C, H, W] (no batch dimension)
    key is the filename of the input from which this attribution was calculated
    """
    dataset = (
        wds.WebDataset(str(tar_path), shardshuffle=False)  # ty: ignore
        .decode(pth_decoder)
        .to_tuple("__key__", "patch.pth", "indices.pth", "input_keys.json")
        .batched(batch_size, collation_fn=lambda samples: list(zip(*samples)))
    )
    for keys, patches, indices, input_keys in dataset:
        yield list(keys), list(patches), list(indices), list(input_keys)
