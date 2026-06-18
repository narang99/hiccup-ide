from pathlib import Path

import torch
import webdataset as wds
from torch import nn
from tqdm import tqdm

from ..attr import get_layer_attributions
from ..path import RemotePath
from ..shards import raw_iter_shards, read_image_shard, tensor_to_bytes
from ..tfms import transform


def collect_attributions(
    all_labels,
    layer_name,
    image_shards_base_path: RemotePath,
    attribution_shards_base_path: RemotePath,
    model,
    read_shard_batch_size: int = 64,
    device="cpu",
    method="deeplift",
    n_steps=128,
):
    for label in tqdm(all_labels):
        run_attribution_and_write(
            image_shards_base_path,
            int(label),
            transform,
            model,
            layer_name,
            attribution_shards_base_path,
            read_shard_batch_size,
            show_progress=False,
            device=device,
            method=method,
            n_steps=n_steps,
        )


def run_attribution_and_write(
    base_images_shard_dir: RemotePath,
    imagenet_label: int,
    input_transform_fn,
    model: nn.Module,
    layer_name: str,
    out_dir: RemotePath,
    batch_size: int,
    device="cpu",
    n_steps: int = 128,
    internal_batch_size: int | None = None,
    show_progress=True,
    method="deeplift",
):
    """For a given target imagenet label, a model and a layer (using layer name), find the layer attribution of that layer for all inputs of that imagenet label

    We assume the input shards are at base_images_shard_dir / imagenet_label
    The attribution shards would be written to out_dir / imagenet_label / layer_name
    The shards would contain {__key__: "file-name", "attribution.pth": tensor of the attribution of the provided layer for the input of "file-name"}

    attribution shape would be [C, H, W], where C is the number of output channels of the layer. H,W is the shape of the output activation of the layer
    """
    model = model.to(device)
    out_dir = Path(out_dir) / layer_name / str(imagenet_label)
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_it = raw_iter_shards(base_images_shard_dir / str(imagenet_label))
    if show_progress:
        shard_it = tqdm(shard_it)

    for tar_path in shard_it:
        with (out_dir / tar_path.name).open("wb") as f:
            with wds.TarWriter(f) as sink:  # ty: ignore
                for keys, images in read_image_shard(tar_path, batch_size, {}):
                    # [B, 3, H, W]
                    batch = torch.stack(list(map(input_transform_fn, images)))
                    batch = batch.to(device)
                    # [B, C, H, W]
                    attributions = get_layer_attributions(
                        batch,
                        model,
                        layer_name,
                        int(imagenet_label),
                        n_steps,
                        internal_batch_size,
                        method=method,
                    )
                    for key, attr in zip(keys, attributions):
                        sink.write(
                            {
                                "__key__": key,
                                # [C, H, W]
                                "attribution.pth": tensor_to_bytes(attr.to("cpu")),
                            }
                        )
