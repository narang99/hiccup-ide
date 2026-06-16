import math

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

colors_dark_v2 = ["#FF3131", "#333333", "#39FF14"]
rd_bk_gn = mcolors.LinearSegmentedColormap.from_list("RdBkGn", colors_dark_v2)
rd_bk_gn.set_bad("yellow")

colors_white = ["#B22222", "#D3D3D3", "#00A550"]
rd_wht_gn = mcolors.LinearSegmentedColormap.from_list("RdBkGn", colors_white)
rd_wht_gn.set_bad("yellow")


def show_single_channel_red_green_black(
    images,
    figsize=None,
    ncols=2,
    axis="on",
    viztype="global",
    mode="light",
    suptitle="",
    ax_titles=None,
):
    if len(images) == 1:
        ncols = 1

    fig, axs, v_limit = _plot_single_channel_red_green_black(
        images, figsize, ncols, axis, viztype, mode, suptitle, ax_titles
    )
    return axs


def _plot_single_channel_red_green_black(
    images,
    figsize=None,
    ncols=2,
    axis="on",
    viztype="global",
    mode="dark",
    suptitle="",
    ax_titles=None,
):
    if not images:
        return None, None, None

    if images[0].dtype == np.uint8:
        images = [img.astype(np.float32) for img in images]

    all_min = min(img.min() for img in images)
    all_max = max(img.max() for img in images)
    v_limit = max(abs(all_min), abs(all_max))

    images = list(images)
    rows = math.ceil(len(images) / ncols)
    if figsize is None:
        figsize = (5 * rows, 5 * rows)
    if figsize is not None and isinstance(figsize, int):
        figsize = (figsize, figsize)

    fig, axs = plt.subplots(rows, ncols, figsize=figsize)
    fig.suptitle(suptitle)
    if len(images) > 1:
        axs = axs.flatten()
    else:
        axs = [axs]

    for i, img in enumerate(images):
        params = {}
        if viztype == "gray":
            params["cmap"] = "gray"
        else:
            cmap = rd_bk_gn if mode == "dark" else rd_wht_gn
            params["cmap"] = cmap
            if viztype == "global":
                params["vmin"], params["vmax"] = -v_limit, v_limit
            elif viztype == "local":
                params["vmin"], params["vmax"] = get_local_image_limits(img)
            else:
                raise Exception(f"invalid viztype {viztype}")

        axs[i].imshow(img, **params)
        axs[i].axis(axis)

    if ax_titles is not None:
        for i in range(len(axs)):
            if i < len(ax_titles):
                axs[i].set_title(ax_titles[i])

    plt.tight_layout()
    return fig, axs, v_limit


def get_local_image_limits(img):
    # return (img.min(), img.max())
    mx, mn = img.max(), img.min()
    mx = max(abs(mx), abs(mn))
    lim = (-mx, mx)
    return lim
