import numpy as np


def indices_for_percentage(arr, pct):
    """
    Ranks every index in arr by |value| (positive and negative together, not
    separately) and keeps the smallest prefix whose cumulative |value| covers
    `pct` of the total |value| sum. Combining signs into one ranking (rather
    than independently taking the top `pct` of the positive total and the top
    `pct` of the negative total) means a sign that contributes little to the
    overall magnitude naturally contributes few or no indices, instead of
    being padded out to its own `pct` regardless of how small its total is.

    Returns (indices, frac_done): indices is an array of shape
    (k, arr.ndim), one row per kept index, in descending |value| order.
    frac_done[i] is the cumulative |value| fraction of the total after
    including indices[i].
    """
    arr = np.array(arr)
    shape = arr.shape
    flat = arr.ravel()

    order = np.argsort(-np.abs(flat))
    total = np.abs(flat).sum()
    target = total * pct
    cumsum = np.cumsum(np.abs(flat[order]))
    cutoff = min(np.searchsorted(cumsum, target) + 1, len(order))
    frac_done = cumsum[:cutoff] / total

    result = np.stack(np.unravel_index(order[:cutoff], shape), axis=-1)
    return result, frac_done


def shorth(data, frac=0.5):
    data_sorted = np.sort(data)
    n = len(data_sorted)
    window = int(np.ceil(frac * n))
    min_width = np.inf
    best_start = 0
    for i in range(n - window + 1):
        width = data_sorted[i + window - 1] - data_sorted[i]
        if width < min_width:
            min_width = width
            best_start = i
    return data_sorted[best_start], data_sorted[best_start + window - 1]


def get_noise_range(data, frac=0.9):
    return shorth(data, frac)


def noise_stats(noise, tol=1e-6):
    """
    (noise_min, noise_med, noise_max, noise_radius) for one dep neuron's raw
    noise samples — noise_min/noise_max via get_noise_range (shorth), noise_med
    the plain median, and noise_radius = noise_max - noise_med + tol (the
    "1 unit" used everywhere distances are expressed in noise-radius units,
    e.g. analyser.get_activation_distance_from_noise,
    report_stats.compute_output_activation_noise_max_distances) — tol avoids
    a division by zero on a degenerate (single-valued) noise sample.
    """
    noise_min, noise_max = get_noise_range(noise)
    noise_med = np.median(noise)
    noise_radius = noise_max - noise_med + tol
    return noise_min, noise_med, noise_max, noise_radius


def get_labels_above_noise_range(label_by_points, noise, percentile):
    mn, mx = get_noise_range(noise)

    def filter_fn(p):
        return len(p) > 0 and np.percentile(p, percentile) > mx

    high_labels = [label for label, p in label_by_points.items() if filter_fn(p)]
    return {label: label_by_points[label] for label in high_labels}
