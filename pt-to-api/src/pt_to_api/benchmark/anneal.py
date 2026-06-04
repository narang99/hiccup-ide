from dataclasses import dataclass
import math


@dataclass
class ConstantReconError:
    def get_multiplier(self, current_epoch: int, total_epochs):
        return 1


@dataclass
class CosineAnnealReconError:
    max_factor: int
    min_factor: int = 1

    def get_multiplier(self, current_epoch: int, total_epochs):
        return cosine_anneal(
            self.min_factor, self.max_factor, current_epoch, total_epochs
        )


def cosine_anneal(min_val, max_val, epoch, total_epochs, hold_frac=0.2):
    decay_epochs = int(total_epochs * (1 - hold_frac))
    if epoch >= decay_epochs:
        return min_val
    return min_val + 0.5 * (max_val - min_val) * (
        1 + math.cos(math.pi * epoch / decay_epochs)
    )


ReconErrSchedule = CosineAnnealReconError | ConstantReconError
