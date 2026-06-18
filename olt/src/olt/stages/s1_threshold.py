import gc
import json
import shutil
import tempfile
import typing
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict
from uuid import uuid4

import torch
from matplotlib.figure import Figure
from tqdm import tqdm

from ..path import RemotePath, remote_mkdir
from ..shards import raw_iter_shards, read_attribution_shard


class SingleLabelThreshold(TypedDict):
    positive: float
    negative: float
    pos_above_elbow_ratio: float
    neg_above_elbow_ratio: float


ChannelThresholdFileData = dict[int, SingleLabelThreshold]


LabelByThresholds = dict[int, torch.Tensor]


def get_collected_thresholds_for_layer(
    layer_name: str, total_channels: int, threshold_base_dir: RemotePath
) -> tuple[LabelByThresholds, LabelByThresholds]:
    """
    we used to return the collected thresholds as a tensor of shape [total_channels, 1, 1] for a given layer

    Now we return dict[label, the-tensor-described-above]
    label is imagenet label, which is present inside thresholds.json
    """
    label_by_pos_thresholds = defaultdict(list)
    label_by_neg_thresholds = defaultdict(list)

    for channel in range(total_channels):
        threshold_file = (
            threshold_base_dir / layer_name / str(channel) / "thresholds.json"
        )
        with open(threshold_file) as f:
            label_by_thresh_result: ChannelThresholdFileData = json.load(f)

            for label, thresh_result in label_by_thresh_result.items():
                label_by_pos_thresholds[label].append(thresh_result["positive"])
                label_by_neg_thresholds[label].append(thresh_result["negative"])

    label_by_pos_thresholds = {
        # [C, 1, 1]
        label: torch.tensor(pos_thresholds)[:, None, None]
        for label, pos_thresholds in label_by_pos_thresholds.items()
    }
    label_by_neg_thresholds = {
        # [C, 1, 1]
        label: torch.tensor(neg_thresholds)[:, None, None]
        for label, neg_thresholds in label_by_neg_thresholds.items()
    }

    return label_by_pos_thresholds, label_by_neg_thresholds


def calculate_thresholds_for_layer_attributions(
    base_attr_dir: RemotePath,
    threshold_store_dir: RemotePath,
    imagenet_labels: list[int],
    layer_name: str,
    total_channels: int,
    n_channels_in_one_batch: int = 16,
    plot_thresholds=False,
) -> None:

    # now how many channels is the question no? that would require us to read one of the attributions lol
    # total_channels = _get_num_channels_in_the_first_attribution(
    #     base_attr_dir, imagenet_labels[0], layer_name, wds_cache_kwargs
    # )
    print("starting threshold calculation, total-channels =", total_channels)
    if total_channels is None:
        raise Exception(
            f"could not find the number of channels for base={base_attr_dir} layer_name={layer_name}"
        )
    if plot_thresholds:
        fig = Figure(figsize=(10, 5))
        axes = fig.subplots(1, 2)
    else:
        fig, axes = None, None

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for channel_start in range(0, total_channels, n_channels_in_one_batch):
            channel_end = min(channel_start + n_channels_in_one_batch, total_channels)
            calculate_thresholds_for_layer_attributions_for_channels(
                base_attr_dir,
                threshold_store_dir,
                imagenet_labels,
                layer_name,
                channel_start,
                channel_end,
                tmp,
                fig,
                axes,
            )


def calculate_thresholds_for_layer_attributions_for_channels(
    base_attr_dir: RemotePath,
    threshold_store_dir: RemotePath,
    imagenet_labels: list[int],
    layer_name: str,
    channel_start: int,
    channel_end: int,
    tmp_path: Path,
    fig,
    axes,
) -> None:
    # note: this can be made faster
    # we dont need to aggregate labels_by_attrs, its wasteful (in terms of memory also)
    # we can simply do the calculation per label inside the main loop
    # for now skipping as i dont want to change a lot of code, but this might be useful speedup
    """calculate the thresholds for each neuron.
    currently we calculate one positive and negative scaler value for each neuron

    Now we want to actually make thresholds per label (2 scalers per imagenet label)
    The problem is cat and cars are not coming if you dont keep separate thresholds
    """
    print(
        f"threshold calculation: {channel_start}:{channel_end} num-labels={len(imagenet_labels)}"
    )
    n_channels = channel_end - channel_start
    layer_threshold_store = threshold_store_dir / layer_name

    # collect all attrs for the given channels
    label_by_attrs = {}
    for label in tqdm(imagenet_labels, desc="stage: collection"):
        # [B, C, H, W]
        labels_attributions = get_all_stacked_attributions(
            base_attr_dir,
            label,
            layer_name,
            channel_start,
            channel_end,
        )
        # [B, n_channels, H, W]
        label_by_attrs[label] = labels_attributions

    gc.collect()
    for chan in tqdm(range(n_channels), desc="stage: plotting and calc"):
        label_by_threshold_result = (
            _get_threshold_calc_result_across_labels_for_one_channel(
                label_by_attrs, chan, tmp_path, fig, axes
            )
        )
        actual_channel = channel_start + chan
        _store_single_channel_threshold_result_to_remote_path(
            layer_threshold_store, actual_channel, label_by_threshold_result
        )

    gc.collect()


def get_all_stacked_attributions(
    base_attr_dir: RemotePath,
    imagenet_label: int,
    layer_name: str,
    channel_start: int,
    channel_end: int,
) -> torch.Tensor:
    """Download the attributions for the whole layer for a given target imagenet label, collect them in a single stacked tensor

    This will return [B, C, H, W], a single tensor containing the attributions of all samples for a given imagenet label and layer
    """
    attribution_shards = raw_iter_shards(
        base_attr_dir / layer_name / str(imagenet_label)
    )

    cat_res = []
    if len(attribution_shards) == 0:
        print(
            "WARN: shard count of label",
            len(attribution_shards),
            "label =",
            imagenet_label,
        )

    for shard in attribution_shards:
        # attributions: list[torch.Tensor], the tensor is of shape [C, H, W]
        for _, attributions in read_attribution_shard(shard, 16, {}):
            for att in attributions:
                cat_res.append(att[channel_start:channel_end])
    return torch.stack(cat_res).detach().clone().cpu()


def find_elbow_index_in_sorted_data(data: torch.Tensor) -> int:
    n = len(data)
    x = torch.arange(n, dtype=data.dtype, device=data.device)
    y = data

    x1, y1 = x[0], y[0]
    x2, y2 = x[-1], y[-1]

    numerator = ((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1).abs()
    denominator = torch.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    distances = numerator / denominator

    return typing.cast(int, torch.argmax(distances).item())


def get_sorted_pos_and_neg_partitions(
    tens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Takes an arbitrary tensor, flattens it, returns the positive and negative values separately. sorted.
    All values are absolute values (the negative value tensor has absolute values, its positive)

    Return shape: (n,) (n is the number of elements in tens)
    """
    pos_vals = tens.clone()
    pos_vals[pos_vals < 0] = 0
    pos_vals = pos_vals.reshape(-1).abs()

    neg_vals = tens.clone()
    neg_vals[neg_vals > 0] = 0
    neg_vals = neg_vals.reshape(-1).abs()

    return torch.sort(pos_vals).values, torch.sort(neg_vals).values


@dataclass
class ThresholdCalculationResult:
    positive_threshold: float
    negative_threshold: float
    pos_above_elbow_ratio: float
    neg_above_elbow_ratio: float
    plot: Path | None


def _get_threshold_calc_result_across_labels_for_one_channel(
    label_by_attrs: dict[int, torch.Tensor], channel: int, tmp_path: Path, fig, axes
) -> dict[int, ThresholdCalculationResult]:
    return {
        label: _get_threshold_calc_result_for_one_channel_and_label(
            attrs, channel, tmp_path, fig, axes
        )
        for label, attrs in label_by_attrs.items()
    }


def _get_threshold_calc_result_for_one_channel_and_label(
    all_attrs: torch.Tensor,
    channel: int,
    tmp_path: Path,
    fig,
    axes,
) -> ThresholdCalculationResult:
    """
    all_attrs is [B, n_channels, H, W]

    `channel` is assumed to be an index inside all_attrs

    Given the tensor of attributions for a given channel, calculate the thresholds along with the plots so that user can verify later
    """

    channels_attrs = all_attrs[:, channel, :, :]
    pos_vals, neg_vals = get_sorted_pos_and_neg_partitions(channels_attrs.clone())
    pos_elbow_idx = find_elbow_index_in_sorted_data(pos_vals)
    neg_elbow_idx = find_elbow_index_in_sorted_data(neg_vals)

    pos_threshold = pos_vals[pos_elbow_idx]
    neg_threshold = neg_vals[neg_elbow_idx]

    pos_above_elbow_ratio = (pos_vals >= pos_threshold).sum() / pos_vals.numel()

    neg_above_elbow_ratio = (neg_vals >= neg_threshold).sum() / neg_vals.numel()

    if fig is None or axes is None:
        image_path = None
    else:
        axes[0].clear()
        axes[0].plot(pos_vals)
        axes[0].axvline(x=pos_elbow_idx, color="red", linestyle="--")

        axes[1].clear()
        axes[1].plot(neg_vals)
        axes[1].axvline(x=neg_elbow_idx, color="red", linestyle="--")

        image_path = tmp_path / f"{uuid4()}.jpeg"
        fig.savefig(image_path, format="jpeg")

    return ThresholdCalculationResult(
        positive_threshold=pos_threshold.item(),
        negative_threshold=-neg_threshold.item(),
        pos_above_elbow_ratio=pos_above_elbow_ratio.item(),
        neg_above_elbow_ratio=neg_above_elbow_ratio.item(),
        plot=image_path,
    )


def _store_single_channel_threshold_result_to_remote_path(
    layer_threshold_store: RemotePath,
    channel: int,
    label_by_thresh_result: dict[int, ThresholdCalculationResult],
) -> None:
    channel_store = layer_threshold_store / str(channel)
    remote_mkdir(channel_store)
    with (channel_store / "thresholds.json").open("w") as f:
        label_by_store: ChannelThresholdFileData = {
            label: {
                "positive": thresh_result.positive_threshold,
                "negative": thresh_result.negative_threshold,
                "pos_above_elbow_ratio": thresh_result.pos_above_elbow_ratio,
                "neg_above_elbow_ratio": thresh_result.neg_above_elbow_ratio,
            }
            for label, thresh_result in label_by_thresh_result.items()
        }
        json.dump(label_by_store, f)

    for label, thresh_result in label_by_thresh_result.items():
        if thresh_result.plot is not None:
            shutil.copy2(thresh_result.plot, channel_store / f"{label}_thresholds.jpeg")
