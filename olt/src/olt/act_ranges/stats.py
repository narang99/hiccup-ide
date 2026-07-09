import numpy as np


def indices_for_percentage(arr, pct):
    arr = np.array(arr)
    shape = arr.shape
    flat = arr.ravel()

    pos_idx = np.where(flat > 0)[0]
    neg_idx = np.where(flat < 0)[0]

    def top_indices(idx, values, target_pct):
        order = idx[np.argsort(-np.abs(values[idx]))]
        total = np.abs(values[order]).sum()
        target = total * target_pct
        cumsum = np.cumsum(np.abs(values[order]))
        cutoff = min(np.searchsorted(cumsum, target) + 1, len(order))
        frac_done = cumsum[:cutoff] / total
        return order[:cutoff], frac_done

    pos_flat, pos_frac = top_indices(pos_idx, flat, pct)
    neg_flat, neg_frac = top_indices(neg_idx, flat, pct)

    pos_result = np.stack(np.unravel_index(pos_flat, shape), axis=-1)
    neg_result = np.stack(np.unravel_index(neg_flat, shape), axis=-1)

    return pos_result, pos_frac, neg_result, neg_frac


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


def get_labels_above_noise_range(label_by_points, noise, percentile):
    mn, mx = get_noise_range(noise)

    def filter_fn(p):
        return len(p) > 0 and np.percentile(p, percentile) > mx

    high_labels = [label for label, p in label_by_points.items() if filter_fn(p)]
    return {label: label_by_points[label] for label in high_labels}
