import torch
from pt_to_api.capture import get_model_internals
import random
from collections import defaultdict
from fastai.basics import (
    get_image_files,
    parent_label,
    CategoryBlock,
    DataLoaders,
    GrandparentSplitter,
    DataBlock,
)
from pt_to_api.utils import it_chain
from torch import nn
from fastai.vision.all import (
    untar_data,
    Resize,
    Normalize,
    PILImageBW,
    Learner,
    CrossEntropyLossFlat,
    accuracy,
    URLs,
    ProgressCallback,
    ImageBlock,
)
from pt_to_api.contribs import v1, v2
from pt_to_api.utils import to_device, detach_all


# 2. Define the Model
class SimpleMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            # First Conv block: 1 channel in -> 8 filters out
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),  # Output: 14x14
            nn.ReLU(),
            # Second Conv block: 8 in -> 16 filters out
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),  # Output: 7x7
            nn.ReLU(),
            # Flatten to 1D vector: 16 filters * 7 * 7 = 784
            nn.Flatten(),
            # Final Linear layer for 10 classes
            nn.Linear(16 * 7 * 7, 10),
        )

    def forward(self, x):
        return self.layers(x)


def get_learner(device="cpu"):
    dls = get_mnist_dataloader()
    model = SimpleMNIST()
    learn = Learner(dls, model, loss_func=CrossEntropyLossFlat(), metrics=accuracy)
    learn.remove_cb(ProgressCallback)
    learn.model = learn.model.to(device)  # ty: ignore
    return learn


def stratified_subset(folder, fraction=0.2, seed=42):
    files = get_image_files(folder)
    by_class = defaultdict(list)
    for f in files:
        by_class[f.parent.name].append(f)
    samples = it_chain(
        [random.sample(v, int(fraction * len(v))) for v in by_class.values()]
    )
    return samples


def _mnist_stratified_subset(folder, fraction=0.2, seed=42):
    train_files = stratified_subset(folder / "training", fraction, seed)
    valid_files = stratified_subset(folder / "testing", fraction, seed)
    return train_files + valid_files


def get_mnist_dataloader(fraction=1.0, **kwargs):
    path = untar_data(URLs.MNIST)
    vocab = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    splitter = GrandparentSplitter(train_name="training", valid_name="testing")
    item_tfms = [Resize(28)]
    batch_tfms = [Normalize()]

    def _get_items(f):
        return _mnist_stratified_subset(f, fraction)

    dblock = DataBlock(
        blocks=[
            ImageBlock(PILImageBW),  # ty: ignore
            CategoryBlock(vocab=vocab),
        ],
        get_items=_get_items,
        splitter=splitter,
        get_y=parent_label,
        item_tfms=item_tfms,
        batch_tfms=batch_tfms,
    )
    return DataLoaders.from_dblock(dblock, path, path=path, **kwargs)


def get_contribs_for_inp_vectorized(batch_inp_tens, model, last_layer_contribs, last_layer_key, device):
    acts, parameters = get_model_internals(model, batch_inp_tens)
    acts["x"] = batch_inp_tens

    acts = to_device(acts, device)

    # Initialize backprop controller
    all_layer_keys = ["layers.0", "layers.1", "layers.2", "layers.3", "layers.4", "layers.5"]
    controller = LayerBackpropController(all_layer_keys, last_layer_key)

    total_contribs = _empty_initialise(last_layer_key, acts, controller)
    total_contribs[last_layer_key] = last_layer_contribs

    # Backpropagate through layers based on controller logic
    if controller.should_backprop("layers.4"):
        total_contribs["layers.4"] = v2.linear_calculate_contribs_for_all(
            model.get_submodule("layers.5"),
            acts["layers.4"],
            total_contribs["layers.5"],
            device,
        )
    
    if controller.should_backprop("layers.3"):
        # flatten
        total_contribs["layers.3"] = total_contribs["layers.4"].view(acts["layers.3"].shape)

    if controller.should_backprop("layers.2"):
        total_contribs["layers.2"] = v1.relu_calculate_contribs(total_contribs["layers.3"])

    if controller.should_backprop("layers.1"):
        total_contribs["layers.1"], slice_contrib = (
            v2.conv_calculate_contribs_for_all(
                acts["layers.1"],
                model.get_submodule("layers.2"),
                total_contribs["layers.2"],
            )
        )
        if controller.should_backprop("layers.2.slice"):
            total_contribs["layers.2.slice"] = slice_contrib
    
    if controller.should_backprop("layers.0"):
        total_contribs["layers.0"] = v1.relu_calculate_contribs(total_contribs["layers.1"])

    if controller.should_backprop("x"):
        total_contribs["x"], slice_contrib = (
            v2.conv_calculate_contribs_for_all(
                acts["x"],
                model.get_submodule("layers.0"),
                total_contribs["layers.0"],
            )
        )
        if controller.should_backprop("layers.0.slice"):
            total_contribs["layers.0.slice"] = slice_contrib

    total_contribs = detach_all(to_device(total_contribs, "cpu"))
    acts = detach_all(to_device(acts, "cpu"))
    parameters = detach_all(to_device(parameters, "cpu"))
    return total_contribs, acts, parameters


class LayerBackpropController:
    def __init__(self, layer_names, last_layer_key):
        self.layer_names = layer_names
        self.last_layer_key = last_layer_key
        self.last_layer_index = self.layer_names.index(last_layer_key)
        
        # Map slice layers to their dependency layers
        self.slice_dependencies = {
            "layers.0.slice": "x",
            "layers.2.slice": "layers.1"
        }
    
    def should_backprop(self, layer_name):
        """Returns True if the layer should be backpropagated, False if it should be set to zero/input"""
        # Handle slice layers
        if layer_name in self.slice_dependencies:
            dependency = self.slice_dependencies[layer_name]
            return self.should_backprop(dependency)
        
        # Handle regular layers
        try:
            layer_index = self.layer_names.index(layer_name)
            return layer_index < self.last_layer_index
        except ValueError:
            # Layer name not in our list, assume it should be backpropagated
            return True


def _empty_initialise(last_layer_key: str, acts: dict, controller: LayerBackpropController):
    all_layer_keys = ["layers.0", "layers.1", "layers.2", "layers.3", "layers.4", "layers.5"]
    slice_layer_keys = ["layers.0.slice", None, "layers.2.slice", None, None, None]

    total_contribs = {}
    
    for i, layer_key in enumerate(all_layer_keys):
        if layer_key in acts and not controller.should_backprop(layer_key):
            total_contribs[layer_key] = acts[layer_key] * 0
            # Handle slice contributions for layers that don't participate
            if slice_layer_keys[i] is not None:
                # Get previous layer for slice shape calculation
                prev_layer_key = all_layer_keys[i-1] if i > 0 else "x"
                if prev_layer_key in acts:
                    total_contribs[slice_layer_keys[i]] = _get_0_slice_contribs(
                        acts[layer_key], acts[prev_layer_key]
                    )
    return total_contribs

def _get_0_slice_contribs(curr_act: torch.Tensor, prev_act: torch.Tensor):
    # [b, chan_out, chan_in, out_h, out_w]
    b, c_out, out_h, out_w = curr_act.shape
    _, c_in, _, _ = prev_act.shape
    return torch.zeros((b, c_out, c_in, out_h, out_w)).float()


# def get_contribs_for_inp(inp_tens, model, input_ratios):
#     acts, parameters = get_model_internals(model, inp_tens)
#     total_contribs = {}

#     total_contribs["layers.5"] = input_ratios

#     total_contribs["layers.4"] = v1.linear_calculate_contribs_for_all(
#         model.get_submodule("layers.5"),
#         acts["layers.4"],
#         acts["layers.5"],
#         total_contribs["layers.5"],
#     )

#     # flatten
#     total_contribs["layers.3"] = total_contribs["layers.4"].view([1, 16, 7, 7])

#     total_contribs["layers.2"] = v1.relu_calculate_contribs(total_contribs["layers.3"])

#     total_contribs["layers.1"] = v1.conv_calculate_contribs_for_all(
#         model.get_submodule("layers.2"),
#         acts["layers.1"],
#         acts["layers.2"],
#         total_contribs["layers.2"],
#     )
#     total_contribs["layers.0"] = v1.relu_calculate_contribs(total_contribs["layers.1"])
#     total_contribs["inputs"] = v1.conv_calculate_contribs_for_all(
#         model.get_submodule("layers.0"),
#         inp_tens,
#         acts["layers.0"],
#         total_contribs["layers.0"],
#     )
#     acts["inputs"] = inp_tens
#     return total_contribs, acts, parameters
