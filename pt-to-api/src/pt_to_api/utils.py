import itertools
import numpy as np
from sklearn.decomposition import PCA
import seaborn as sns
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
rd_bk_gn.set_bad("yellow")

# Crimson -> Gainsboro (Very light grey) -> Dark Green
colors_white = ["#B22222", "#D3D3D3", "#00A550"]
rd_wht_gn = mcolors.LinearSegmentedColormap.from_list("RdBkGn", colors_white)
rd_wht_gn.set_bad("yellow")


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
    images, figsize=None, ncols=2, axis="on", viztype="global", mode="light", suptitle="", ax_titles=None
):
    if len(images) == 1:
        ncols = 1
    if viztype == "gray":
        show(images, figsize=figsize, ncols=ncols, axis=axis, cmap="gray")
        return

    fig, axs, v_limit = _plot_single_channel_red_green_black(
        images, figsize, ncols, axis, viztype, mode, suptitle, ax_titles
    )
    return axs


def _plot_single_channel_red_green_black(
    images, figsize=None, ncols=2, axis="on", viztype="global", mode="dark", suptitle="", ax_titles=None
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
    if isinstance(tens, torch.Tensor):
        tens = tens.clone().detach().cpu().numpy()
    return [t for t in tens]


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

def scatter_plot_1d(numbers, suff=""):
    # 2. Create the visualization
    plt.figure(figsize=(20, 3))
    sns.stripplot(x=numbers, color='blue', alpha=0.5, jitter=True)

    plt.title('1D Clustering Visualization' + suff)
    plt.xlabel('Value')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.show()


def show_72_list(xs, **kwargs):
    res = list(itertools.chain.from_iterable([to_show_list(x.reshape(8,3,3)) for x in xs]))
    show_single_channel_red_green_black(res, (20, 4*len(xs)), 8, **kwargs)
    plt.show()

def show_72(x, **kwargs):
    show_single_channel_red_green_black(
        to_show_list(x.reshape(8, 3, 3)), 20, 8, **kwargs
    )
    plt.show()

# def get_receptive(y, x, ksize=3, stride=2, padding=1):
#     ys = y*stride - padding
#     xs = x*stride - padding
#     return (ys, xs), (ys+ksize, xs+ksize)

def get_receptive(y, x, ksize=(3,3), stride=(2,2), padding=(1,1), dilation=(1,1)):
    ys = y * stride[0] - padding[0]
    xs = x * stride[1] - padding[1]
    effective_ky = dilation[0] * (ksize[0] - 1) + 1
    effective_kx = dilation[1] * (ksize[1] - 1) + 1
    return (ys, xs), (ys + effective_ky, xs + effective_kx)


def otsu_threshold(data, bins=256):
    hist, bin_edges = np.histogram(data, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    total = hist.sum()
    total_mean = (hist * bin_centers).sum() / total
    
    best_thresh = 0
    best_variance = 0
    weight_bg = 0
    mean_bg = 0
    
    for i in range(len(hist)):
        weight_bg += hist[i] / total
        if weight_bg == 0:
            continue
        
        mean_bg += hist[i] * bin_centers[i] / total
        weight_fg = 1 - weight_bg
        
        if weight_fg == 0:
            break
        
        mean_fg = (total_mean - mean_bg) / weight_fg
        
        between_variance = weight_bg * weight_fg * (mean_bg / weight_bg - mean_fg) ** 2

        if between_variance > best_variance:
            best_variance = between_variance
            best_thresh = bin_centers[i]
    
    return best_thresh



def explain_variance_with_pca(X):
    pca = PCA().fit(X)
    cumvar = np.cumsum(pca.explained_variance_ratio_)

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, len(cumvar) + 1), cumvar, linewidth=1.5)
    plt.axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label='95%')
    plt.axhline(y=0.99, color='g', linestyle='--', alpha=0.5, label='99%')
    plt.xlabel("n_components")
    plt.ylabel("cumulative variance explained")
    plt.title("PCA variance explained")
    plt.legend()
    plt.tight_layout()
    plt.show()

    for threshold in [0.90, 0.95, 0.99]:
        n = np.searchsorted(cumvar, threshold) + 1
        print(f"{threshold:.0%} variance explained by {n} components")


def show_gram(W, title="", figsize=None):
    if not isinstance(W, torch.Tensor):
        W = torch.tensor(W)
    W_norm = W / (W.norm(dim=0, keepdim=True) + 1e-8)
    gram = W_norm.T @ W_norm  # (n_components, n_components)
    if figsize is not None:
        plt.figure(figsize=figsize)
    plt.imshow(gram, cmap="gray")
    plt.title(title)
    plt.show()
    return gram

def gram_orthogonality_error(W):
    if not isinstance(W, torch.Tensor):
        W = torch.tensor(W)
    
    W_norm = W / (W.norm(dim=0, keepdim=True) + 1e-8)
    gram = W_norm.T @ W_norm  # (n_components, n_components)
    
    n = gram.shape[0]
    identity = torch.eye(n, device=gram.device, dtype=gram.dtype)
    
    # Normalise gram to [-1, 1] before computing error
    gram_normalised = gram / (gram.abs().max() + 1e-8)
    identity_normalised = identity / (identity.abs().max() + 1e-8)
    
    error = (gram_normalised - identity_normalised).pow(2).mean().sqrt()
    return error.item()

def run_single_test(dims_list, atoms_ratio, noise_std_set, term3_set):
    for dim in dims_list:
        for a in atoms_ratio:
            for noise_std in noise_std_set:
                for term3 in term3_set:
                
                    print("#######", dim, a, noise_std, term3)
                    gc.collect()
                    atoms = math.ceil(dim*a)
                    # keep all active
                    k = atoms
                    n_samples = 100*atoms
                    # active_dims = math.ceil(dim*sp_rat)
        
                    device = get_device(dim)
                    X, W_true, codes_true, dim_partition = gen_fn(dim, atoms, k, n_samples=n_samples, noise_std=noise_std)
            
                    scaler = MeanPerDimGlobalStdScaler().fit(X)
                    X_scaled = scaler.transform(X)
    
                    mets = []
                    for run_idx in range(NUM_RUNS_PER_TEST):
                        print("RUN:", run_idx)
                        kwargs = {kwarg_key: term3}
                        run = train(X_scaled, atoms, 1e-3, epochs=4000, device=device, **kwargs)
                        mets.append(get_metrics_from_run(run, W_true))
                        gc.collect()
                    
                    metrics[(dim, a, noise_std, term3)] = aggregate_metrics(mets)