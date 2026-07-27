import math
from pathlib import Path

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
    mode="dark",
    suptitle="",
    ax_titles=None,
):
    if len(images) == 1:
        ncols = 1

    fig, axs, v_limit = _plot_single_channel_red_green_black(
        images, figsize, ncols, axis, viztype, mode, suptitle, ax_titles
    )
    return axs


def save_single_channel_red_green_black(
    images,
    output_path,
    figsize=None,
    ncols=2,
    axis="on",
    viztype="global",
    mode="dark",
    suptitle="",
    ax_titles=None,
):
    """Headless sibling of show_single_channel_red_green_black: savefig + close
    instead of returning axes, for dumping report assets rather than notebook display."""
    if len(images) == 1:
        ncols = 1
    fig, _, _ = _plot_single_channel_red_green_black(
        images, figsize, ncols, axis, viztype, mode, suptitle, ax_titles
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=fig.get_facecolor())
    plt.close(fig)


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


def show_grid(image_list, rows, cols, ax_titles=None):
    # Initialize the figure layout
    if ax_titles is None:
        ax_titles = []
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

    # Flatten axes array for easy 1D iteration
    axes = axes.flatten()

    for i, img in enumerate(image_list):
        if i < len(axes):
            axes[i].imshow(img)
            axes[i].axis("off")  # Hide the X/Y coordinate ticks
            if len(ax_titles) > i:
                axes[i].set_title(ax_titles[i])

    # Hide any remaining empty subplots if image_list is shorter than rows * cols
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
