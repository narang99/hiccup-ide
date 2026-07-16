import itertools
import math

import matplotlib.pyplot as plt
import numpy as np

from olt.show import show_single_channel_red_green_black
from olt.stages.s3_cluster import predict_labels_and_recons


def show_comps_of_varied_losses(best, data, hdbscan_mod, image_shape=(3, 3)):
    codes, recon = predict_labels_and_recons(data, hdbscan_mod, best)
    # codes, recon = model.transform(data)
    mse = ((data - recon) ** 2).sum(axis=1)

    idxs = np.argsort(mse)

    lows = idxs[:20]
    highs = idxs[-40:]
    mid_length = len(idxs) // 2
    mids = idxs[mid_length : mid_length + 20]

    plt.hist(mse)
    plt.title("loss histogram")
    p50 = np.percentile(mse, 50)
    p75 = np.percentile(mse, 75)
    p90 = np.percentile(mse, 90)
    plt.axvline(p50, color="green", linestyle="--", label=f"Median (p50): {p50:.4f}")
    plt.axvline(p75, color="orange", linestyle="--", label=f"p75: {p75:.4f}")
    plt.axvline(p90, color="red", linestyle="--", label=f"p90: {p90:.4f}")
    plt.legend()
    plt.show()
    plt.hist(codes)
    plt.show()

    def _get_title_tup(pref, ns):
        return list(
            itertools.chain.from_iterable(
                (f"{pref}/{mse[i]:.4f}", f"{pref}/{mse[i]:.4f}") for i in ns
            )
        )

    ts = (
        [(data[i], recon[i]) for i in highs]
        + [(data[i], recon[i]) for i in mids]
        + [(data[i], recon[i]) for i in lows]
    )
    ax_titles = (
        _get_title_tup("high", highs)
        + _get_title_tup("mid", mids)
        + _get_title_tup("low", lows)
    )
    _show_comps(
        list(itertools.chain.from_iterable(ts)),
        image_shape,
        show_kwargs={"ax_titles": ax_titles, "viztype": "local"},
    )


def _show_comps(components, image_shape, row_sz=3, col_sz=3, show_kwargs=None):
    if show_kwargs is None:
        show_kwargs = {}
    ncols = min(len(components), 8)
    nrows = math.ceil(len(components) / ncols)
    figsize = (ncols * col_sz, nrows * row_sz)
    show_single_channel_red_green_black(
        [c.reshape(image_shape) for c in components],
        figsize,
        ncols,
        **show_kwargs,
    )
    plt.show()
