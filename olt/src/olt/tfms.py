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
    def __init__(self):
        mean = torch.tensor(CIFAR_MEAN).view(-1, 1, 1)
        std = torch.tensor(CIFAR_STD).view(-1, 1, 1)
        self.mean, self.std = mean, std

    def __call__(self, tensor):
        return (tensor * self.std + self.mean).clamp(0, 1)


cifar_inverse_transform = CifarInverseTransform()


# class InverseTransform:
#     def __init__(self):
#         self.normalize = transforms.Normalize(
#             mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
#             std=[1 / 0.229, 1 / 0.224, 1 / 0.225],
#         )

#     def __call__(self, tensor):
#         return (self.normalize(tensor) * 255).clamp(0, 255).byte()


inverse_transform = InverseTransform()
