import torch
from torch import nn


class ModelSnapshot:
    # TODO: add context manager and test, for now we are manually. calling remove
    def __init__(self, model: nn.Module, layer_names: list[str]):
        self.model = model
        self.activations = {}
        self.parameters = {}  # Dictionary to store weights and biases
        self.layer_names = layer_names
        self.hooks = []

        # Register hooks for dynamic activations
        self._register_hooks(self.layer_names)

    def _register_hooks(self, layer_names: list[str]):
        for layer_name in layer_names:
            module = self.model.get_submodule(layer_name)
            hook = module.register_forward_hook(self._get_hook(layer_name))
            self.hooks.append(hook)

    def _get_hook(self, name: str):
        def hook_fn(module, input, output):
            # Store the main activation output
            self.activations[name] = input[0].detach().cpu()
            # self.activations[name] = output.detach().cpu()

        return hook_fn

    def remove(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def get_layer_activations(
    batch: torch.Tensor, model: nn.Module, layer_names: list[str]
):
    # batch: [B, C, H, W] (C is the input's channels, 3 for inceptionv1)snapshot = ModelSnapshot(model, layer_names)
    # we return [B, c, h, w], where c is the layer's output channels (number of kernels in the layer)
    # contract is similar to get_layer_attributions
    snapshot = ModelSnapshot(model, layer_names)
    with torch.no_grad():
        model(batch)
    activations = {
        layer_name: a.detach().cpu().clone()
        for layer_name, a in snapshot.activations.items()
    }
    snapshot.remove()
    del snapshot
    return activations
