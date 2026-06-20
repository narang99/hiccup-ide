import math
import random
import tarfile
import time
from collections import defaultdict
from contextlib import contextmanager
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
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
    margin-bottom: 2rem;
    background: #1e1f22;
    border: 1px solid #2c2e33;
    border-radius: 6px;
    padding: 1.25rem 1.5rem;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.4);
  }

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
</style>
</head>
<body>
<div class="container">
<h1 class="report-title">Attribution Report</h1>
"""

_TAIL = """\
</div>
<script>
  function toggleView(btn) {
    const block = btn.closest('.grid-block');
    const combined = block.querySelector('.combined');
    const third    = block.querySelector('.third');
    if (btn.dataset.state === 'combined') {
      combined.style.display = 'none';
      third.style.display    = 'block';
      btn.textContent        = 'Show Combined';
      btn.dataset.state      = 'third';
    } else {
      third.style.display    = 'none';
      combined.style.display = 'block';
      btn.textContent        = 'Show Third';
      btn.dataset.state      = 'combined';
    }
  }
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


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
):
    """Render a list of (inv_img, neuron_att, third) pairs and save as a JPEG file."""
    n = len(pairs)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * col_sz, nrows * row_sz))
    axes = np.array(axes).reshape(-1)

    for i, (inv_img, neuron_att, third) in enumerate(pairs):
        ax = axes[i]
        if view == "combined":
            overlay = neuron_att.detach().cpu().sum(dim=0).numpy()
            overlay = overlay / np.abs(overlay).max()
            ax.imshow(inv_img)
            ax.imshow(overlay, cmap=cmap, alpha=alpha, vmin=-1, vmax=1)
        else:
            if isinstance(third, torch.Tensor):
                overlay = third.cpu().numpy()
            else:
                overlay = third
            vmin, vmax = get_local_image_limits(overlay)
            ax.imshow(overlay, cmap=cmap, vmin=vmin, vmax=vmax)

        if titles and i < len(titles):
            ax.set_title(titles[i], fontsize=6, color="#868e96", pad=2)
        ax.axis("off")

    for j in range(len(pairs), len(axes)):
        axes[j].axis("off")

    fig.patch.set_facecolor("#141517")
    plt.suptitle(suptitle, color="#c1c2c5", fontsize=10)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1, wspace=0.05)

    fig.savefig(out_path, format="jpeg", bbox_inches="tight", facecolor="#141517")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _load_pairs(imagenet_map, samples_per_cluster, image_shape, cluster_label, df):
    """Flatten all samples across imagenet labels, shuffle, cap, then load tensors."""

    all_samples = [
        # (s, label, data["total"])
        (
            s,
            label,
            len(
                df[
                    (df.cluster_label == int(cluster_label))
                    & (df.imagenet_label == int(label))
                ]
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
                torch.load(act_path, weights_only=False, map_location="cpu").reshape(
                    image_shape
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
        <section class="label-section">
          <h2 class="label-header">Cluster {cluster_label}</h2>
          <div class="grid-block">
            <div class="grid-meta">
              <span class="tag">cluster {cluster_label}</span>
              <span class="total">{n_samples} samples</span>
              <button class="toggle-btn" onclick="toggleView(this)" data-state="combined">
                Show Third
              </button>
            </div>
            <img class="view combined" id="img-{block_id}-combined" src="{fname_combined}" />
            <img class="view third"    id="img-{block_id}-third"    src="{fname_third}"    style="display:none" />
          </div>
        </section>"""


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
        image_shape,
        ncols,
        col_sz,
        row_sz,
        alpha,
        cmap,
    ) = args

    pairs, titles = _load_pairs(
        imagenet_map, samples_per_cluster, image_shape, cluster_label, df
    )
    if not pairs:
        return None

    suptitle = f"cluster {cluster_label} — {len(pairs)} samples"
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
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    with (output_dir / "report.csv").open("w") as f:
        df.to_csv(f)

    cluster_map = sample_for_label(50, clustering_and_attr_src_dir)
    print(f"num cluster labels: {len(cluster_map)}")
    ordered = sorted(cluster_map.items(), key=_cluster_sort_key)

    worker_args = [
        (
            cluster_label,
            imagenet_map,
            block_id,
            out,
            df,
            samples_per_cluster,
            image_shape,
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
        for worker_arg in tqdm(worker_args, desc="rendering clusters"):
            results = _render_cluster_worker(worker_arg)
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

    html = _HEAD + "\n".join(sections) + "\n" + _TAIL
    (out / "index.html").write_text(html, encoding="utf-8")
    print(f"Report saved to {out}")

    archive_report(out)


def archive_report(output_dir: Path, tar_path: None | Path = None):
    if tar_path is None:
        tar_path = output_dir.parent / f"{output_dir.name}.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(output_dir, arcname=output_dir.name)
    print(f"archived to {tar_path}")
    return tar_path
