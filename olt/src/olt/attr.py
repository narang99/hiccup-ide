import torch
from captum.attr import LayerDeepLift
from torch import nn


# "mixed4e_1x1_pre_relu_conv:55"
def get_layer_attributions(
    batch: torch.Tensor,
    model: nn.Module,
    layer_name: str,
    label: int,
    n_steps: int = 128,
    internal_batch_size: int | None = None,
):
    if internal_batch_size is None:
        internal_batch_size = 128
    # batch: [B, C, H, W] (C is the input's channels, 3 for inceptionv1)
    # we return [B, c, h, w], where c is the layer's output channels (number of kernels in the layer)
    # h,w is the activation outputs dimensions
    # ig = LayerIntegratedGradients(model, model.get_submodule(layer_name))
    # layer_attribution = ig.attribute(
    #     batch, n_steps=n_steps, target=label, internal_batch_size=internal_batch_size
    # )

    ig = LayerDeepLift(model, model.get_submodule(layer_name))
    layer_attribution = ig.attribute(
        batch,
        target=label,
    )

    return layer_attribution
