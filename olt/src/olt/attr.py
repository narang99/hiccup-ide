import torch
from captum.attr import LayerDeepLift, LayerIntegratedGradients
from torch import nn


# "mixed4e_1x1_pre_relu_conv:55"
def get_layer_attributions(
    batch: torch.Tensor,
    model: nn.Module,
    layer_name: str,
    label: int,
    n_steps: int = 128,
    internal_batch_size: int | None = None,
    method="deeplift",
):
    if internal_batch_size is None:
        internal_batch_size = 128
    # batch: [B, C, H, W] (C is the input's channels, 3 for inceptionv1)
    # we return [B, c, h, w], where c is the layer's output channels (number of kernels in the layer)
    # h,w is the activation outputs dimensions

    if method == "deeplift":
        ig = LayerDeepLift(model, model.get_submodule(layer_name))
        layer_attribution = ig.attribute(
            batch,
            target=label,
        )
    elif method == "ig":
        ig = LayerIntegratedGradients(model, model.get_submodule(layer_name))
        layer_attribution = ig.attribute(
            batch,
            n_steps=n_steps,
            target=label,
            internal_batch_size=internal_batch_size,
        )
    else:
        raise Exception(f"invalid method {method}")

    return layer_attribution
