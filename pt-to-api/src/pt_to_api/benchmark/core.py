import numpy as np
from dataclasses import dataclass
from typing import Any

@dataclass
class SingleRun:
    model: Any
    codes: np.ndarray
    components: np.ndarray
    recon: np.ndarray
    loss: float
    hyperparameters: dict