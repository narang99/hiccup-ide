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
    baseline_loss: float | None = None


@dataclass(frozen=True)
class RunId:
    n_components: int
    seed: int

@dataclass(frozen=True)
class CompId:
    run_id: RunId
    idx_inside_run: int

@dataclass(frozen=True)
class IdAndComp:
    comp_id: CompId
    comp: np.ndarray


C2R = dict[RunId, SingleRun]