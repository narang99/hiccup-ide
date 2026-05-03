import torch
from typing import Any, Optional


def get_layer_type(layer_name: str, model=None) -> str:
    """Determine the layer type from the model structure."""
    if layer_name == "x":
        return "Input"
    if model is None:
        return "Unknown"
    for name, module in model.named_modules():
        if name == layer_name:
            return module.__class__.__name__
    return "Unknown"


def generate_contrib_coordinates(
    total_contribs: dict[str, torch.Tensor],
    sample_idx: int = 0,
    model=None,
) -> dict[str, dict]:
    """
    Convert the total_contribs dict returned by get_contribs_for_inp_vectorized
    into a coordinate-keyed dictionary that mirrors the format used in
    activation_processor.py / generate_coordinates.

    All tensors in total_contribs have a batch dimension (dim 0) of size N.
    We extract a single sample via sample_idx.

    Coordinate scheme (matching activation_processor conventions):
    ─────────────────────────────────────────────────────────────
    Conv / ReLU spatial tensors  →  shape [N, C, H, W]
        per-channel:              "<key>.out_<c>"
            data  : 2-D list [H, W]
            shape : [H, W]

    Slice tensors                →  shape [N, C_out, C_in, H, W]
        per (out, in) pair:       "<key>.out_<c_out>.in_<c_in>"
            data  : 2-D list [H, W]
            shape : [H, W]

    Linear / flat tensors        →  shape [N, D]
        per neuron:               "<key>.out_<d>"
            data  : scalar float
            shape : []

    Input saliency               →  shape [N, 1, H, W]  (treated as C=1 channel)
        "x.out_0"
            data  : 2-D list [H, W]
            shape : [H, W]
    """
    coords: dict[str, dict] = {}

    # layers.4 is the flat [N, 784] intermediate that is immediately re-shaped
    # into layers.3 [N, 16, 7, 7] — skip it to avoid 784 redundant scalar entries.
    SKIP_KEYS = {"layers.4"}

    for key, tensor in total_contribs.items():
        if tensor is None or key in SKIP_KEYS:
            continue

        # Pull out the single sample we care about
        t = tensor[sample_idx]  # drop the batch dim

        ndim = t.ndim
        layer_type = get_layer_type(key, model)

        # ── Slice tensors: [C_out, C_in, H, W] ──────────────────────────────
        if ndim == 4 and key.endswith(".slice"):
            layer_name = key[: -len(".slice")]  # e.g. "layers.2" from "layers.2.slice"
            c_out, c_in, h, w = t.shape
            for out_ch in range(c_out):
                for in_ch in range(c_in):
                    coord = f"{layer_name}.out_{out_ch}.in_{in_ch}"
                    coords[coord] = {
                        "layer_name": layer_name,
                        "data": t[out_ch, in_ch].numpy().tolist(),
                        "shape": [h, w],
                        "coordinate_type": "input_output_channel",
                        "data_type": "contrib",
                        "layer_type": "Conv2d",
                    }
            continue

        # ── Spatial tensors (Conv / ReLU): [C, H, W] ─────────────────────────
        if ndim == 3:
            c, h, w = t.shape
            for ch in range(c):
                coord = f"{key}.out_{ch}"
                layer_name = key
                coords[coord] = {
                    "layer_name": layer_name,
                    "data": t[ch].numpy().tolist(),
                    "shape": [h, w],
                    "coordinate_type": "output_channel",
                    "data_type": "contrib",
                    "layer_type": layer_type,
                }
            continue

        # ── Flat / linear tensors: [D] ───────────────────────────────────────
        if ndim == 1:
            d = t.shape[0]
            for neuron in range(d):
                coord = f"{key}.out_{neuron}"
                layer_name = key
                coords[coord] = {
                    "layer_name": layer_name,
                    "data": float(t[neuron].item()),
                    "shape": [],
                    "coordinate_type": "neuron",
                    "data_type": "contrib",
                    "layer_type": layer_type,
                }
            continue

        # ── Anything else: skip ───────────────────────────────────────────────
        # (e.g. the flat [N, 784] tensor for layers.4 before it is viewed as
        #  [N, 16, 7, 7] — that view is already stored under "layers.3")

    return coords


def process_contribs_to_coordinates(
    total_contribs: dict[str, torch.Tensor],
    sample_idx: int = 0,
    model=None,
) -> dict[str, dict]:
    """
    Public entry point. Converts total_contribs (from
    get_contribs_for_inp_vectorized) to a coordinate-keyed dictionary.

    Parameters
    ----------
    total_contribs : dict
        The dict returned by get_contribs_for_inp_vectorized.
    sample_idx : int
        Which sample in the batch to extract (default: 0).
    model : torch.nn.Module, optional
        The model to extract layer types from.

    Returns
    -------
    dict
        Keys are coordinate strings, values are dicts with
        {"data", "shape", "coordinate_type", "data_type"}.
    """
    return generate_contrib_coordinates(total_contribs, sample_idx, model)


def get_contrib_coordinate_data(
    coordinate: str,
    coord_data: dict[str, dict],
) -> Optional[dict[str, Any]]:
    """
    Look up a single coordinate in the processed contrib data.
    Returns None if not found.
    """
    return coord_data.get(coordinate)
