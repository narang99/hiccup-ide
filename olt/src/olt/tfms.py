import random

import torch
from torchvision import transforms

# transform = transforms.Compose(
#     [
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
#     ]
# )

# inception uses this
transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x * 255 - 117),
    ]
)


class InverseTransform:
    def __call__(self, tensor):
        return ((tensor + 117) / 255).clamp(0, 1)


# --- CIFAR-10 (for olt.models.cifar_inception) -------------------------------
# The CIFAR mini-Inception is trained from scratch, so we control preprocessing.
# Images are already 32x32, so there is no resize/crop at eval/analysis time —
# unlike the ImageNet 256->224 pipeline above. Keep `cifar_transform` byte-exact
# with whatever the training script uses for the *eval* branch, and keep
# `cifar_inverse_transform` its exact inverse (report rendering / feature-viz
# rely on round-tripping normalized tensors back to [0, 1] RGB).
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)

cifar_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD),
    ]
)

# Training-time augmentation (random crop + flip); eval/analysis uses
# `cifar_transform` above with no augmentation.
cifar_train_transform = transforms.Compose(
    [
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=CIFAR_MEAN, std=CIFAR_STD),
    ]
)


class CifarInverseTransform:
    def __init__(self, device):
        mean = torch.tensor(CIFAR_MEAN).to(device).view(-1, 1, 1)
        std = torch.tensor(CIFAR_STD).to(device).view(-1, 1, 1)
        self.mean, self.std = mean, std

    def __call__(self, tensor):
        return (tensor * self.std + self.mean).clamp(0, 1)


# cifar_inverse_transform = CifarInverseTransform()


# --- STL-10 (96x96, for olt.models.cifar_inception at stem_stride=2) ----------
# Same story as CIFAR (trained from scratch, we own preprocessing), but 96x96
# images -> sharper feature-viz / patch crops than CIFAR's 32x32. No resize at
# eval/analysis time; `stl_transform` must stay byte-exact with training's eval
# branch and `stl_inverse_transform` its exact inverse.
STL10_MEAN = (0.4467, 0.4398, 0.4066)
STL10_STD = (0.2603, 0.2566, 0.2713)

stl_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=STL10_MEAN, std=STL10_STD),
    ]
)

stl_train_transform = transforms.Compose(
    [
        transforms.RandomCrop(96, padding=12),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=STL10_MEAN, std=STL10_STD),
    ]
)


class StlInverseTransform:
    def __init__(self, device):
        self.mean = torch.tensor(STL10_MEAN).to(device).view(-1, 1, 1)
        self.std = torch.tensor(STL10_STD).to(device).view(-1, 1, 1)

    def __call__(self, tensor):
        return (tensor * self.std + self.mean).clamp(0, 1)


# stl_inverse_transform = StlInverseTransform()


# --- STL-10 "fancy" anti-shortcut augmentation -------------------------------
# The plain `stl_train_transform` uses a fixed constant(=black) padded RandomCrop,
# which bakes black borders into a big fraction of training crops -> the net
# learns black-border detectors. And STL backgrounds correlate with classes
# (ship->water, plane->sky), so it also leans on color (e.g. blue). The transforms
# here counter both, WITHOUT touching the originals (keep those for reproducing
# existing checkpoints). Opt in via the training scripts' --fancy-aug flag.
class RandomPaddingModeCrop:
    """A RandomCrop whose `padding_mode` is picked at random each call, so the
    net sometimes sees black (constant/zero) borders and sometimes reflected
    ones instead of always black. This keeps borders in the training
    distribution (so the net stays robust to them) but stops any single border
    style from being a reliable class-independent cue. `modes` are sampled
    uniformly; `fill` applies only to the "constant" mode."""

    def __init__(self, size, padding, modes=("constant", "reflect"), fill=0):
        self.size = size
        self.padding = padding
        self.modes = list(modes)
        self.fill = fill

    def __call__(self, img):
        mode = random.choice(self.modes)
        return transforms.RandomCrop(
            self.size, padding=self.padding, padding_mode=mode, fill=self.fill
        )(img)


stl_fancy_train_transform = transforms.Compose(
    [
        RandomPaddingModeCrop(96, padding=12, modes=("constant", "reflect")),
        transforms.RandomHorizontalFlip(),
        # weaken color shortcuts: jitter color, and sometimes drop color entirely
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=STL10_MEAN, std=STL10_STD),
    ]
)


# class InverseTransform:
#     def __init__(self):
#         self.normalize = transforms.Normalize(
#             mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
#             std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
#         )

#     def __call__(self, tensor):
#         return (self.normalize(tensor) * 255).clamp(0, 255).byte()


inverse_transform = InverseTransform()
