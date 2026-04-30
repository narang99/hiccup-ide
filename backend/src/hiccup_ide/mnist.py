from hiccup_ide.capture import get_model_internals
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
from hiccup_ide.utils import it_chain
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
from hiccup_ide.contribs import v1, v2
from hiccup_ide.utils import to_device, detach_all


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


def get_contribs_for_inp_vectorized(batch_inp_tens, model, batch_input_ratios, device):
    acts, parameters = get_model_internals(model, batch_inp_tens)
    acts["inputs"] = batch_inp_tens

    acts = to_device(acts, device)

    total_contribs = {}
    total_contribs["layers.5"] = batch_input_ratios

    total_contribs["layers.4"] = v2.linear_calculate_contribs_for_all(
        model.get_submodule("layers.5"),
        acts["layers.4"],
        total_contribs["layers.5"],
        device,
    )
    # flatten
    total_contribs["layers.3"] = total_contribs["layers.4"].view(acts["layers.3"].shape)

    total_contribs["layers.2"] = v1.relu_calculate_contribs(total_contribs["layers.3"])

    total_contribs["layers.1"], total_contribs["layers.2.slice"] = (
        v2.conv_calculate_contribs_for_all(
            acts["layers.1"],
            model.get_submodule("layers.2"),
            total_contribs["layers.2"],
        )
    )
    total_contribs["layers.0"] = v1.relu_calculate_contribs(total_contribs["layers.1"])

    total_contribs["inputs"], total_contribs["layers.0.slice"] = (
        v2.conv_calculate_contribs_for_all(
            acts["inputs"],
            model.get_submodule("layers.0"),
            total_contribs["layers.0"],
        )
    )
    total_contribs = detach_all(to_device(total_contribs, "cpu"))
    acts = detach_all(to_device(acts, "cpu"))
    parameters = detach_all(to_device(parameters, "cpu"))
    return total_contribs, acts, parameters


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
