"""Track how a single neuron's weights drift over training.

A "neuron" is one output channel of a conv layer, selected as
"layer_name:channel" (e.g. "block_c.branch_1x1_pre_relu_conv:0"): the first part
is the argument to `model.get_submodule`, the second is the output channel. Its
"version" at a given step is the flattened weight slice `conv.weight[channel]`
with the bias scalar appended. After each gradient step we measure how far that
vector moved from the previous step under a configurable metric, giving a
per-neuron drift curve over training.

Model-agnostic: works on any `nn.Conv2d` reachable via `get_submodule`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


def _neuron_vector(model: nn.Module, layer_name: str, channel: int) -> torch.Tensor:
    """The parameter vector for output channel `channel` of `layer_name`:
    weight[channel].flatten() with bias[channel] appended (if the conv has a bias).
    Detached, float, on CPU."""
    conv = model.get_submodule(layer_name)
    # clone/detach so the stored vector owns its memory and can't alias the live
    # parameter (which the optimizer mutates in place between steps).
    parts = [conv.weight[channel].detach().reshape(-1).float()]
    if conv.bias is not None:
        parts.append(conv.bias[channel].detach().reshape(1).float())
    return torch.cat(parts).clone().cpu()


def _euclidean(cur: torch.Tensor, prev: torch.Tensor) -> float:
    return torch.linalg.vector_norm(cur - prev).item()


def _cosine(cur: torch.Tensor, prev: torch.Tensor) -> float:
    # cosine DISTANCE: 0 when unchanged in direction, up to 2 when flipped.
    return (1.0 - F.cosine_similarity(cur, prev, dim=0)).item()


METRICS = {"euclidean": _euclidean, "cosine": _cosine}


def parse_selector(sel: str) -> tuple[str, int]:
    layer, channel = sel.split(":")
    return layer, int(channel)


class NeuronTracker:
    """Records, per tracked neuron, the metric distance between its parameter
    vector at consecutive steps. `record` is called once at init (baseline, no
    distance yet) and once after every optimizer step."""

    def __init__(
        self, selectors: list[tuple[str, int]], metric_name: str, out_dir: Path
    ):
        self.selectors = selectors
        self.metric_name = metric_name
        self.metric = METRICS[metric_name]
        self.out_dir = out_dir
        self.keys = [f"{layer}:{ch}" for layer, ch in selectors]
        self.prev: dict[str, torch.Tensor] = {}
        self.steps: list[int] = []
        self.series: dict[str, list[float]] = {k: [] for k in self.keys}

    def validate(self, model: nn.Module) -> None:
        for layer, ch in self.selectors:
            conv = model.get_submodule(layer)  # raises if the path is wrong
            if not isinstance(conv, nn.Conv2d):
                raise ValueError(f"{layer!r} is a {type(conv).__name__}, not a conv")
            n_out = conv.weight.shape[0]
            if not 0 <= ch < n_out:
                raise ValueError(f"channel {ch} out of range for {layer!r} (0..{n_out - 1})")

    @torch.no_grad()
    def record(self, model: nn.Module, step: int) -> None:
        cur = {
            f"{layer}:{ch}": _neuron_vector(model, layer, ch)
            for layer, ch in self.selectors
        }
        if self.prev:
            self.steps.append(step)
            for key in self.keys:
                self.series[key].append(self.metric(cur[key], self.prev[key]))
        self.prev = cur

    def save(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        data = {"metric": self.metric_name, "steps": self.steps, "series": self.series}
        json_path = self.out_dir / f"neuron_tracks_{self.metric_name}.json"
        json_path.write_text(json.dumps(data, indent=2))
        print(f"  neuron tracks -> {json_path}")
        self._plot()

    def _plot(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5))
        for key in self.keys:
            ax.plot(self.steps, self.series[key], label=key, linewidth=1)
        ax.set_xlabel("gradient descent step")
        ax.set_ylabel(f"{self.metric_name} distance from previous step")
        ax.set_title("Per-neuron parameter drift over training")
        ax.legend(fontsize="small")
        fig.tight_layout()
        png_path = self.out_dir / f"neuron_tracks_{self.metric_name}.png"
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
        print(f"  neuron tracks -> {png_path}")


def add_tracking_args(parser: argparse.ArgumentParser) -> None:
    """Add --track-neuron / --track-metric / --track-out to a training parser."""
    parser.add_argument(
        "--track-neuron",
        type=str,
        nargs="+",
        default=None,
        metavar="LAYER:CHANNEL",
        help="one or more neuron selectors 'layer_name:channel' (arg to "
        "model.get_submodule + output channel) whose parameter drift to record "
        "each gradient step, e.g. block_c.branch_1x1_pre_relu_conv:0",
    )
    parser.add_argument(
        "--track-metric",
        default="euclidean",
        choices=sorted(METRICS),
        help="distance between a neuron's version at consecutive steps",
    )
    parser.add_argument(
        "--track-out",
        type=Path,
        default=None,
        help="dir for neuron-track json/png (default: <ckpt-dir>/neuron_tracks)",
    )


def build_tracker(
    track_neuron: list[str] | None,
    track_metric: str,
    track_out: Path | None,
    ckpt_dir: Path,
    model: nn.Module,
) -> NeuronTracker | None:
    """Build a NeuronTracker from parsed CLI values (or None if no neuron given),
    validating the selectors and recording the init baseline."""
    if not track_neuron:
        return None
    selectors = [parse_selector(s) for s in track_neuron]
    out_dir = track_out or ckpt_dir / "neuron_tracks"
    tracker = NeuronTracker(selectors, track_metric, out_dir)
    tracker.validate(model)
    tracker.record(model, step=0)  # baseline at init
    print(
        f"tracking {len(selectors)} neuron(s) with metric '{track_metric}': "
        f"{', '.join(tracker.keys)}"
    )
    return tracker
