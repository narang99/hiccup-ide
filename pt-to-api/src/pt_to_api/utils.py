import itertools
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import numpy as np
import math
import torch

# the std one has too less contrast
# std_dark_colors = ["red", "black", "green"]
colors_dark_v2 = ["#FF3131", "#333333", "#39FF14"]
rd_bk_gn = mcolors.LinearSegmentedColormap.from_list("RdBkGn", colors_dark_v2)

# Crimson -> Gainsboro (Very light grey) -> Dark Green
colors_white = ["#B22222", "#D3D3D3", "#00A550"]
rd_wht_gn = mcolors.LinearSegmentedColormap.from_list("RdBkGn", colors_white)


def it_chain(iterator):
    return list(itertools.chain.from_iterable(iterator))


def to_device(batch, device):
    """
    Recursively moves a nested structure of tensors to a specified device.
    Works with dicts, lists, tuples, and torch.Tensors.
    """
    if isinstance(batch, torch.Tensor):
        return batch.to(device)

    elif isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}

    elif isinstance(batch, (list, tuple)):
        # Returns the same type (list or tuple)
        return type(batch)(to_device(v, device) for v in batch)

    # Return as-is if it's a string, int, or other non-tensor leaf
    return batch


def detach_all(batch):
    """
    Recursively detaches a nested structure of tensors
    Works with dicts, lists, tuples, and torch.Tensors.
    """
    if isinstance(batch, torch.Tensor):
        return batch.detach()

    elif isinstance(batch, dict):
        return {k: detach_all(v) for k, v in batch.items()}

    elif isinstance(batch, (list, tuple)):
        # Returns the same type (list or tuple)
        return type(batch)(detach_all(v) for v in batch)

    # Return as-is if it's a string, int, or other non-tensor leaf
    return batch


def zeros_with_1_at(length, idx_of_1):
    res = torch.zeros(length).to(torch.float32).unsqueeze(0).cpu()
    res[0][idx_of_1] = 1.0
    return res


def show_single_channel_red_green_black(
    images, figsize=None, ncols=2, axis="on", viztype="global", mode="dark"
):
    if len(images) == 1:
        ncols = 1
    if viztype == "gray":
        show(images, figsize=figsize, ncols=ncols, axis=axis, cmap="gray")

    fig, axs, v_limit = _plot_single_channel_red_green_black(
        images, figsize, ncols, axis, viztype, mode
    )
    return axs


def _plot_single_channel_red_green_black(
    images, figsize=None, ncols=2, axis="on", viztype="global", mode="dark"
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

    plt.tight_layout()
    return fig, axs, v_limit


def get_local_image_limits(img):
    # return (img.min(), img.max())
    mx, mn = img.max(), img.min()
    mx = max(abs(mx), abs(mn))
    lim = (-mx, mx)
    return lim


def _plot(crops, figsize=None, ncols=2, axis="on", cmap=None, titles=None):
    crops = list(crops)
    rows = math.ceil(len(crops) / ncols)
    if titles is not None:
        titles = list(titles)
    if figsize is None:
        figsize = (5 * rows, 5 * rows)
    if figsize is not None and isinstance(figsize, int):
        figsize = (figsize, figsize)
    fig, axs = plt.subplots(rows, ncols, figsize=figsize)
    if len(crops) > 1:
        axs = axs.flatten()
    else:
        axs = [axs]
    for i, c in enumerate(crops):
        title = None
        if titles is not None and i < len(titles):
            title = titles[i]
        if cmap is not None:
            axs[i].imshow(c, cmap=cmap)
        else:
            axs[i].imshow(c)
        if title is not None:
            axs[i].set_title(title)
        axs[i].axis(axis)
    plt.tight_layout()
    return fig, axs


def show(crops, figsize=None, ncols=2, axis="on", cmap=None, titles=None):
    if len(crops) == 1:
        ncols = 1
    fig, axs = _plot(crops, figsize, ncols, axis, cmap, titles)
    plt.show()


def to_show_list(tens):
    "creates a list of numpy arrays along the first dimension for viewing"
    return [t.detach().cpu().numpy() for t in tens]


def mk_rect_on_ax(ax, r, c, h, w, edgecolor="r"):
    linewidth = 0.5
    x = c - linewidth
    y = r - linewidth
    rect = patches.Rectangle(
        (x, y),
        h,
        w,
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor="none",
    )
    ax.add_patch(rect)

def get_ratios_for_labels(labels):
    return torch.cat([zeros_with_1_at(10, lb) for lb in labels])