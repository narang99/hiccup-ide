import math
import random
import tarfile
import time
from collections import defaultdict
from contextlib import contextmanager
from multiprocessing import Pool
from pathlib import Path

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from captum.attr import NeuronDeepLift
from PIL import Image, ImageDraw
from tqdm import tqdm

from olt.show import get_local_image_limits, rd_bk_gn


def _get_attribution_pth_files(d, cluster_label):
    return list(d.glob(f"*/{cluster_label}/neuron_attributions/*.pth"))


def sample_for_label(samples_for_each: int, src_dir: Path):
    # src_dir / imagenet_label / cluster_label / neuron_attributions / *.pth
    cluster_label_by_imagenet_label_by_samples = defaultdict(
        lambda: defaultdict(lambda: {"total": 0, "samples": []})
    )

    for imagenet_label_dir in tqdm(list(src_dir.glob("*")), desc="file system scan"):
        if not imagenet_label_dir.is_dir():
            continue
        cluster_label_dirs = list(
            (d for d in imagenet_label_dir.glob("*/*") if d.is_dir())
        )
        # print("clusteR_label dirs", cluster_label_dirs)
        for cluster_label_dir in cluster_label_dirs:
            if not cluster_label_dir.is_dir():
                continue
            # print("checking", cluster_label_dir)
            files = list((cluster_label_dir / "neuron_attributions").glob("*.pth"))
            # sampled_files = random.sample(files, k=min(len(files), samples_for_each))
            if len(files) > 0:
                val = cluster_label_by_imagenet_label_by_samples[
                    cluster_label_dir.name
                ][imagenet_label_dir.name]
                val["samples"].extend(files)
                # val["total"] += len(files)
    return {
        cluster_label: dict(imagenet_map)
        for cluster_label, imagenet_map in cluster_label_by_imagenet_label_by_samples.items()
    }
    # return cluster_label_by_imagenet_label_by_samples


# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

_HEAD = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Attribution Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
<link href="https://fonts.googleapis.com/css2?family=Source+Sans+3:wght@400;600;700&display=swap" rel="stylesheet"/>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: #141517;
    color: #c1c2c5;
    font-family: "Source Sans 3", "Source Sans Pro", system-ui, -apple-system, sans-serif;
    font-size: 16px;
    line-height: 1.5;
  }

  /* ---------- tab bar ---------- */
  .tab-bar-wrap {
    position: sticky;
    top: 0;
    z-index: 100;
    background: #141517;
    border-bottom: 2px solid #2c2e33;
    overflow-x: auto;
    scrollbar-width: thin;
    scrollbar-color: #4dabf7 #1e1f22;
  }

  .tab-bar {
    display: flex;
    gap: 0;
    min-width: max-content;
    padding: 0 1.5rem;
  }

  .tab-btn {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.1rem;
    padding: 0.65rem 1.1rem;
    background: none;
    border: none;
    border-bottom: 3px solid transparent;
    color: #868e96;
    font-family: inherit;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    cursor: pointer;
    white-space: nowrap;
    transition: color 0.15s, border-color 0.15s;
    margin-bottom: -2px; /* sit on top of wrap border */
  }

  .tab-btn .tab-count {
    font-size: 0.68rem;
    font-weight: 400;
    color: #495057;
    transition: color 0.15s;
  }

  .tab-btn:hover { color: #c1c2c5; }
  .tab-btn:hover .tab-count { color: #868e96; }

  .tab-btn.active {
    color: #4dabf7;
    border-bottom-color: #4dabf7;
  }

  .tab-btn.active .tab-count { color: #74c0fc; }

  /* ---------- content ---------- */
  .container {
    max-width: 1000px;
    margin: 0 auto;
    padding: 2.5rem 1.5rem;
  }

  h1.report-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: #e9ecef;
    border-bottom: 2px solid #4dabf7;
    padding-bottom: 0.5rem;
    margin-bottom: 2rem;
  }

  .label-section {
    display: none; /* hidden until tab activated */
    background: #1e1f22;
    border: 1px solid #2c2e33;
    border-radius: 6px;
    padding: 1.25rem 1.5rem;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.4);
  }

  .label-section.active { display: block; }

  .label-header {
    font-size: 0.95rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #4dabf7;
    margin-bottom: 0.9rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #2c2e33;
  }

  .grid-block { margin-bottom: 0.5rem; }

  .grid-meta {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 0.75rem;
  }

  .tag {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    background: #1971c2;
    color: #e9ecef;
    padding: 0.18rem 0.65rem;
    border-radius: 999px;
  }

  .total { font-size: 0.83rem; color: #868e96; }

  .toggle-btn {
    margin-left: auto;
    background: #1e1f22;
    border: 1px solid #4dabf7;
    color: #4dabf7;
    font-family: inherit;
    font-size: 0.82rem;
    font-weight: 600;
    padding: 0.28rem 0.9rem;
    cursor: pointer;
    border-radius: 4px;
    transition: background 0.15s, color 0.15s;
  }

  .toggle-btn:hover { background: #1971c2; color: #e9ecef; border-color: #1971c2; }

  .view {
    width: 100%;
    display: block;
    border: 1px solid #2c2e33;
    border-radius: 4px;
  }

  /* lazy load fade-in */
  .view { opacity: 0; transition: opacity 0.3s ease; }
  .view.loaded { opacity: 1; }
</style>
</head>
<body>
"""

_TAIL = """\
<script>
  function activateTab(tabId) {
    // deactivate all
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.label-section').forEach(s => s.classList.remove('active'));

    // activate target
    const btn = document.querySelector(`.tab-btn[data-tab="${tabId}"]`);
    const section = document.getElementById(`section-${tabId}`);
    if (!btn || !section) return;
    btn.classList.add('active');
    section.classList.add('active');

    // lazy load: move data-src -> src for images in this section
    section.querySelectorAll('img[data-src]').forEach(img => {
      img.src = img.dataset.src;
      delete img.dataset.src;
      img.addEventListener('load', () => img.classList.add('loaded'), { once: true });
    });
  }

  function toggleView(btn) {
    const block = btn.closest('.grid-block');
    const combined = block.querySelector('.combined');
    const third    = block.querySelector('.third');
    if (btn.dataset.state === 'combined') {
      combined.style.display = 'none';
      third.style.display    = 'block';
      btn.textContent        = 'Show Overlays';
      btn.dataset.state      = 'third';
    } else {
      third.style.display    = 'none';
      combined.style.display = 'block';
      btn.textContent        = 'Show Pointwise multiplications';
      btn.dataset.state      = 'combined';
    }
  }

  // activate first tab on load
  const firstTab = document.querySelector('.tab-btn');
  if (firstTab) activateTab(firstTab.dataset.tab);
</script>
</body>
</html>"""

# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def to_pil(img, size):
    """Convert a numpy/tensor image (H,W,3) or (3,H,W) to a resized PIL RGB image."""
    if isinstance(img, torch.Tensor):
        img = img.detach().cpu().numpy()
    if img.shape[0] == 3:  # (3,H,W) -> (H,W,3)
        img = img.transpose(1, 2, 0)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img).resize(size, Image.BILINEAR)  # ty: ignore


def apply_cmap(arr, cmap, vmin, vmax, size, interpolation=Image.NEAREST):  # ty: ignore
    """Apply a matplotlib colormap to a 2D array and return a PIL RGB image."""
    arr = np.clip((arr - vmin) / (vmax - vmin + 1e-8), 0, 1)
    rgba = (cmap(arr) * 255).astype(np.uint8)  # cmap returns (H,W,4)
    return Image.fromarray(rgba, mode="RGBA").convert("RGB").resize(size, interpolation)


def _fit_and_pad(
    img: Image.Image, cell_w: int, cell_h: int, bg: str = "2c2e33"
) -> Image.Image:
    """Resize PIL image to fit within (cell_w, cell_h) preserving aspect ratio, then pad."""
    img.thumbnail((cell_w, cell_h), Image.BILINEAR)  # ty: ignore
    canvas = Image.new("RGB", (cell_w, cell_h), color=bg)
    x = (cell_w - img.width) // 2
    y = (cell_h - img.height) // 2
    canvas.paste(img, (x, y))
    return canvas


def render_grid_to_jpeg(
    pairs,
    view,
    out_path,
    ncols,
    col_sz,
    row_sz,
    cmap=rd_bk_gn,
    alpha=0.8,
    suptitle="",
    titles=None,
    pad_color="silver",
):
    """Render a list of (inv_img, neuron_att, third) pairs and save as a JPEG file."""
    n = len(pairs)
    nrows = math.ceil(n / ncols)

    padding = 4  # pixels between cells

    cell_w, cell_h = col_sz * 100, row_sz * 100  # pixels per cell
    total_w = ncols * cell_w + (ncols - 1) * padding
    header_h = 40 if suptitle else 0
    title_h = 20 if titles else 0
    total_h = nrows * (cell_h + title_h) + header_h + (nrows - 1) * padding

    canvas = Image.new("RGB", (total_w, total_h), color=(20, 21, 23))
    draw = ImageDraw.Draw(canvas)

    if suptitle:
        draw.text((total_w // 2, header_h // 2), suptitle, fill="#ffffff", anchor="mm")

    for i, (inv_img, neuron_att, third) in enumerate(pairs):
        row, col = divmod(i, ncols)
        x = col * (cell_w + padding)
        y = header_h + row * (cell_h + title_h + padding)

        if view == "combined":
            overlay = neuron_att.detach().cpu().sum(dim=0).numpy()
            overlay = overlay / np.abs(overlay).max()
            base = to_pil(inv_img, size=(cell_w, cell_h))
            heat = apply_cmap(
                overlay,
                cmap,
                vmin=-1,
                vmax=1,
                size=(cell_w, cell_h),
                interpolation=Image.BILINEAR,  # ty: ignore
            )
            cell = Image.blend(base, heat, alpha=alpha)
        else:
            if isinstance(third, torch.Tensor):
                overlay = third.cpu().numpy()
            else:
                overlay = third
            vmin, vmax = get_local_image_limits(overlay)
            oh, ow = overlay.shape[:2]
            scale = min(cell_w / ow, cell_h / oh)
            fitted_w, fitted_h = int(ow * scale), int(oh * scale)

            cell = apply_cmap(
                overlay,
                cmap,
                vmin=vmin,
                vmax=vmax,
                size=(fitted_w, fitted_h),
                interpolation=Image.NEAREST,  # ty: ignore
            )
            cell = _fit_and_pad(cell, cell_w, cell_h, bg=pad_color)

        canvas.paste(cell, (x, y))

        if titles and i < len(titles):
            draw.text(
                (x + cell_w // 2, y + cell_h + title_h // 2),
                titles[i],
                fill="#ffffff",
                anchor="mm",
            )

    canvas.save(out_path, format="JPEG", quality=95)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _get_unique_input_key_count_for_cluster_and_imagenet_label_pair(
    df, cluster_label, imagenet_label
):
    return df[
        (df.cluster_label == int(cluster_label))
        & (df.imagenet_label == int(imagenet_label))
    ].input_image_key.nunique()


def get_reshape_to_2d_act_picker(image_shape):
    def rsh(act):
        return act.reshape(image_shape)

    return rsh


def _load_pairs(
    imagenet_map,
    samples_per_cluster,
    act_picker,
    cluster_label,
    df,
):
    """Flatten all samples across imagenet labels, shuffle, cap, then load tensors."""

    all_samples = [
        # (s, label, data["total"])
        (
            s,
            label,
            _get_unique_input_key_count_for_cluster_and_imagenet_label_pair(
                df, cluster_label, label
            ),
        )
        for label, data in imagenet_map.items()
        for s in data["samples"]
    ]
    random.shuffle(all_samples)
    all_samples = all_samples[:samples_per_cluster]

    pairs, titles = [], []
    for sample, imagenet_label, total in all_samples:
        img_path = sample.parent.parent.parent / "image.jpg"
        act_path = sample.parent.parent / "input_activations" / sample.name
        pairs.append(
            (
                plt.imread(img_path),
                torch.load(sample, weights_only=False, map_location="cpu")[0],
                act_picker(
                    torch.load(act_path, weights_only=False, map_location="cpu")
                ),
            )
        )
        titles.append(f"{imagenet_label} / {total}")
    return pairs, titles


# ---------------------------------------------------------------------------
# HTML section builder
# ---------------------------------------------------------------------------


def _section_html(block_id, cluster_label, n_samples, fname_combined, fname_third):
    return f"""
        <section class="label-section" id="section-{block_id}">
          <h2 class="label-header">Cluster {cluster_label}</h2>
          <div class="grid-block">
            <div class="grid-meta">
              <span class="tag">cluster {cluster_label}</span>
              <span class="total">{n_samples} unique images</span>
              <button class="toggle-btn" onclick="toggleView(this)" data-state="combined">
                Show Pointwise multiplications
              </button>
            </div>
            <img class="view combined" data-src="{fname_combined}" />
            <img class="view third"    data-src="{fname_third}"    style="display:none" />
          </div>
        </section>"""


def _build_tab_bar(results):
    """Build the sticky tab bar HTML from ordered results."""
    buttons = []
    for result in results:
        if result is None:
            continue
        block_id, cluster_label, n_samples, _, _ = result
        label = f"Cluster {cluster_label}"
        buttons.append(
            f'<button class="tab-btn" data-tab="{block_id}" onclick="activateTab({block_id})">'
            f'{label}<span class="tab-count">{n_samples} samples</span></button>'
        )
    return (
        '<div class="tab-bar-wrap"><div class="tab-bar">'
        + "".join(buttons)
        + "</div></div>"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _cluster_sort_key(item):
    """Sort clusters by descending sample count; cluster -1 always last."""
    cluster_label, imagenet_map = item
    total = sum(len(data["samples"]) for data in imagenet_map.values())
    is_noise = 1 if cluster_label == -1 else 0
    return (is_noise, -total)


@contextmanager
def agg_backend():
    import matplotlib

    original = matplotlib.get_backend()
    matplotlib.use("Agg")
    try:
        yield
    finally:
        matplotlib.use(original)


# ---------------------------------------------------------------------------
# Parallel worker  (must be top-level for multiprocessing)
# ---------------------------------------------------------------------------


def _render_cluster_worker(args):
    (
        cluster_label,
        imagenet_map,
        block_id,
        out,
        df,
        samples_per_cluster,
        act_picker,
        ncols,
        col_sz,
        row_sz,
        alpha,
        cmap,
    ) = args

    pairs, titles = _load_pairs(
        imagenet_map, samples_per_cluster, act_picker, cluster_label, df
    )
    if not pairs:
        return None

    cluster_df = df[df.cluster_label == int(cluster_label)]
    total_cluster_samples = len(cluster_df)
    uniq_images_in_cluster = cluster_df.input_image_key.nunique()

    suptitle = f"cluster {cluster_label} | samples: {total_cluster_samples} | unique input images: {uniq_images_in_cluster} "
    safe_label = (
        str(cluster_label).replace("-", "neg")
        if cluster_label == -1
        else str(cluster_label)
    )
    fname_combined = f"cluster_{safe_label}_combined.jpeg"
    fname_third = f"cluster_{safe_label}_third.jpeg"

    render_kwargs = dict(
        ncols=ncols,
        col_sz=col_sz,
        row_sz=row_sz,
        cmap=cmap,
        alpha=alpha,
        suptitle=suptitle,
        titles=titles,
    )
    with agg_backend():
        render_grid_to_jpeg(pairs, "combined", out / fname_combined, **render_kwargs)
        render_grid_to_jpeg(pairs, "third", out / fname_third, **render_kwargs)

    return (block_id, cluster_label, len(pairs), fname_combined, fname_third)


# ---------------------------------------------------------------------------
# Entry point  (replaces generate_html_report)
# ---------------------------------------------------------------------------


def _get_cluster_size(args):
    imagenet_map = args[1]
    return sum(len(d["samples"]) for d in imagenet_map.values())


def generate_html_report(
    output_dir,
    clustering_and_attr_src_dir,
    df,
    ncols=5,
    col_sz=2,
    row_sz=2,
    alpha=0.7,
    image_shape=(22, 24),
    samples_per_cluster=100,
    cmap=rd_bk_gn,
    n_workers=4,
    act_picker=None,  # todo: remove image_shape
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with (out / "report.csv").open("w") as f:
        df.to_csv(f)

    cluster_map = sample_for_label(50, clustering_and_attr_src_dir)
    print(f"num cluster labels: {len(cluster_map)}")
    ordered = sorted(cluster_map.items(), key=_cluster_sort_key)

    if act_picker is None:
        act_picker = get_reshape_to_2d_act_picker(image_shape)

    worker_args = [
        (
            cluster_label,
            imagenet_map,
            block_id,
            out,
            df,
            samples_per_cluster,
            act_picker,
            ncols,
            col_sz,
            row_sz,
            alpha,
            cmap,
        )
        for block_id, (cluster_label, imagenet_map) in enumerate(ordered)
    ]

    t0 = time.time()
    if n_workers > 1:
        with Pool(n_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(_render_cluster_worker, worker_args),
                    total=len(worker_args),
                    desc="rendering clusters",
                )
            )
        print(f"all clusters rendered in {time.time() - t0:.1f}s")
    else:
        results = []
        for worker_arg in tqdm(worker_args, desc="rendering clusters"):
            results.append(_render_cluster_worker(worker_arg))
        print(f"all clusters rendered in {time.time() - t0:.1f}s")

    # results come back in submission order, so block_id ordering is preserved
    sections = []
    for result in results:
        if result is None:
            continue
        block_id, cluster_label, n_samples, fname_combined, fname_third = result
        sections.append(
            _section_html(
                block_id, cluster_label, n_samples, fname_combined, fname_third
            )
        )

    tab_bar = _build_tab_bar(results)
    body = (
        '<div class="container">'
        '<h1 class="report-title">Attribution Report</h1>'
        + "\n".join(sections)
        + "</div>"
    )
    html = _HEAD + tab_bar + body + "\n" + _TAIL
    (out / "index.html").write_text(html, encoding="utf-8")
    print(f"Report saved to {out}")


def archive_report(output_dir: Path, tar_path: None | Path = None):
    if tar_path is None:
        tar_path = output_dir.parent / f"{output_dir.name}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(output_dir, arcname=output_dir.name)
    print(f"archived to {tar_path}")
    return tar_path


def make_overlay_heatmap(
    model,
    layer_name,
    neuron_selector,
    pil_img,
    input_transform_fn,
    inverse_transform_fn,
    size,
    device="cpu",
):
    # input_transform_fn / inverse_transform_fn: model-specific, required (no
    # default) so the overlay isn't silently built with the ImageNet transform.
    # size: (W, H) to render the base image + heatmap at before blending; pass
    # the model-input resolution to avoid resampling.
    timg = input_transform_fn(pil_img)[None].to(device)
    model = model.to(device)
    inv_img = inverse_transform_fn(timg)
    ndl = NeuronDeepLift(model, model.get_submodule(layer_name))
    attr_res = ndl.attribute(timg, neuron_selector)

    overlay = attr_res[0].detach().cpu().sum(dim=0).numpy()
    overlay = overlay / np.abs(overlay).max()

    base = to_pil(inv_img[0], size=size)
    heat = apply_cmap(
        overlay,
        rd_bk_gn,
        vmin=-1,
        vmax=1,
        size=size,
        interpolation=Image.BILINEAR,  # ty: ignore
    )
    cell = Image.blend(base, heat, alpha=0.8)
    return cell
