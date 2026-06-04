import numpy as np
from dataclasses import dataclass
from typing import Any


@dataclass
class SingleRun:
    # coefficients. shape: [n_samples, n_components]
    codes: np.ndarray
    # Shape [n_components, dimensions], same as decoder/components shape
    # JAX and Torch use different shapes for their models, so we keep the same shapes which dont depend on them
    encoder: np.ndarray
    # this is the decoder, not changing for backwards compat. Shape [n_components, dimensions]
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
