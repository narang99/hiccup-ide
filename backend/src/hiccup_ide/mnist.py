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
import numpy as np


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

    def _get_items(f): return _mnist_stratified_subset(f, fraction)

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

