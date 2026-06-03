from typing import Literal, Callable
from dataclasses import dataclass
import numpy as np
from sklearn.mixture import GaussianMixture

@dataclass
class NonNegative:
    threshold: float

@dataclass
class NonPositive:
    threshold: float

@dataclass
class MixedSign:
    t_pos: float
    t_neg: float
    pos_max: float
    neg_max: float

@dataclass
class Ambiguous:
    t_pos: float
    t_neg: float
    pos_max: float
    neg_max: float

SignResult = NonNegative | NonPositive | MixedSign | Ambiguous

def find_threshold_using_gmm(values, conservatism=0):
    values = np.array(values)
    gmm = GaussianMixture(n_components=2, random_state=0)
    gmm.fit(values.reshape(-1, 1))
    
    garbage_idx = np.argmax(gmm.means_)
    real_idx = 1 - garbage_idx
    
    midpoint = (gmm.means_[garbage_idx] + gmm.means_[real_idx]) / 2
    gap = gmm.means_[real_idx] - gmm.means_[garbage_idx]  # signed, points away from garbage
    
    return (midpoint - conservatism * gap).item()

def threshold_assuming_noise_at_0_with_only_one_side_active(values: np.ndarray, threshold_fn: Callable = find_threshold_using_gmm) -> SignResult:
    """
    Determines the sign structure of the data and returns the appropriate threshold.

    The threshold_fn is called on absolute values. The sign is inferred by checking
    whether each side's max falls within the other side's garbage threshold:
    - If neg_max < pos_threshold: data is non-negative (negative values are noise)
    - If pos_max < neg_threshold: data is non-positive (positive values are noise)
    - If both: ambiguous (everything is near zero)
    - If neither: genuinely mixed sign

    Args:
        values: raw (signed) attribution values
        threshold_fn: callable that takes absolute values and returns a threshold (also in absolute value space)

    Returns:
        one of NonNegative, NonPositive, MixedSign, Ambiguous
    """
    values = np.array(values)
    pos = values[values > 0]
    neg = values[values < 0]

    t_neg = threshold_fn(np.abs(neg)) if len(neg) > 0 else 0
    t_pos = threshold_fn(pos) if len(pos) > 0 else 0

    pos_max = pos.max() if len(pos) > 0 else 0
    neg_max = abs(neg.min()) if len(neg) > 0 else 0

    is_non_neg = neg_max < t_pos
    is_non_pos = pos_max < t_neg

    if is_non_neg and is_non_pos:
        return Ambiguous(t_pos=t_pos, t_neg=t_neg, pos_max=pos_max, neg_max=neg_max)
    elif is_non_neg:
        return NonNegative(threshold=t_pos)
    elif is_non_pos:
        return NonPositive(threshold=-t_neg)
    else:
        return MixedSign(t_pos=t_pos, t_neg=t_neg, pos_max=pos_max, neg_max=neg_max)


class ThresholdFailureException(Exception):
    def __init__(self, message, sign_result: SignResult):
        self.message = message
        self.sign_result = sign_result