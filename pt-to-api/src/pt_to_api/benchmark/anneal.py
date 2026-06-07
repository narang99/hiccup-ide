import math
from dataclasses import dataclass


@dataclass
class ConstantReconError:
    def get_multiplier(self, current_epoch: int, total_epochs):
        return 1


@dataclass
class CosineAnnealReconError:
    max_factor: float
    min_factor: float = 1.0
    hold_frac: float = 0.2

    def get_multiplier(self, current_epoch: int, total_epochs):
        return cosine_anneal(
            self.min_factor,
            self.max_factor,
            current_epoch,
            total_epochs,
            self.hold_frac,
        )


@dataclass
class CosineIncreaseReconError(CosineAnnealReconError):
    # same as the parent, but we pass total-current, gives a mirror image of increasing value
    def get_multiplier(self, current_epoch: int, total_epochs):
        return cosine_anneal(
            self.min_factor,
            self.max_factor,
            total_epochs - current_epoch,
            total_epochs,
            self.hold_frac,
        )


def cosine_anneal(min_val, max_val, epoch, total_epochs, hold_frac=0.2):
    decay_epochs = int(total_epochs * (1 - hold_frac))
    if epoch >= decay_epochs:
        return min_val
    return min_val + 0.5 * (max_val - min_val) * (
        1 + math.cos(math.pi * epoch / decay_epochs)
    )


ReconErrSchedule = (
    CosineAnnealReconError | ConstantReconError | CosineIncreaseReconError
)
