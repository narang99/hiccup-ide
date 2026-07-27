"""Sanity-check the CIFAR webdataset shards produced by cifar_to_shards.py.

Reads shards through the SAME readers the pipeline uses (`olt.shards`), so it
verifies the real path, and shows a batch of images per label.

Shard layout (from cifar_to_shards.py):
    <base>/<class_id>/000000.tar   with samples {__key__, jpg}

Notebook use (the intended way — plt shows inline):
    from scripts.show_cifar_shards import describe_label, show_label_batch, show_all_labels
    show_label_batch("data/cifar/image-shards", label=3)   # grid for one class
    show_all_labels("data/cifar/image-shards")             # counts + a grid per class

As a script (dumps a montage PNG per label instead of showing):
    uv run scripts/show_cifar_shards.py --base data/cifar/image-shards --out-dir /tmp/cifar-check
"""

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt

from olt.shards import raw_iter_shards, read_image_shard


def read_label_batch(base_dir, label, batch_size=128):
    """First `batch_size` images of one label, from its first shard.
    Returns (keys, images) — images are PIL.Image, keys are their __key__."""
    label_dir = Path(base_dir) / str(label)
    shards = raw_iter_shards(label_dir)
    if not shards:
        raise FileNotFoundError(f"no .tar shards under {label_dir}")
    keys, images = next(read_image_shard(sorted(shards)[0], batch_size, {}))
    return keys, images


def describe_label(base_dir, label):
    """Count shards / total images / unique keys / image size for one label —
    the numbers to eyeball for correct sharding (e.g. 1 shard, 128 images,
    128 unique keys, 32x32 for the CIFAR test split at 128 imgs/label)."""
    label_dir = Path(base_dir) / str(label)
    shards = sorted(raw_iter_shards(label_dir))
    total, keys = 0, set()
    size = None
    for shard in shards:
        for ks, imgs in read_image_shard(shard, 256, {}):
            total += len(imgs)
            keys.update(ks)
            if size is None and imgs:
                size = imgs[0].size  # (W, H)
    info = {
        "label": label,
        "shards": len(shards),
        "images": total,
        "unique_keys": len(keys),
        "image_size": size,
    }
    print(
        f"label {info['label']}: shards={info['shards']} images={info['images']} "
        f"unique_keys={info['unique_keys']} image_size={info['image_size']}"
    )
    return info


def _grid(images, titles=None, ncols=16, title=None):
    n = len(images)
    if n == 0:
        print("(no images to show)")
        return None
    ncols = min(ncols, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 0.9, nrows * 0.9))
    axes = [axes] if n == 1 else axes.flatten()
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < n:
            ax.imshow(images[i])
            if titles is not None and i < len(titles):
                ax.set_title(str(titles[i]), fontsize=5)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def show_label_batch(base_dir, label, batch_size=128, ncols=16):
    """Show a grid of one label's images (for a notebook). Returns the figure."""
    keys, images = read_label_batch(base_dir, label, batch_size)
    return _grid(images, ncols=ncols, title=f"label {label} (n={len(images)})")


def show_all_labels(base_dir, per_label=32, ncols=16):
    """Describe every label and show a small grid per label (for a notebook)."""
    label_dirs = sorted(
        (p for p in Path(base_dir).iterdir() if p.is_dir()),
        key=lambda p: int(p.name) if p.name.isdigit() else p.name,
    )
    for label_dir in label_dirs:
        label = label_dir.name
        describe_label(base_dir, label)
        keys, images = read_label_batch(base_dir, label, per_label)
        _grid(images, ncols=ncols, title=f"label {label}")
    plt.show()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, required=True, help="shard base dir")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--ncols", type=int, default=16)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="if set, save a montage PNG per label here instead of showing",
    )
    args = p.parse_args()

    label_dirs = sorted(
        (d for d in args.base.iterdir() if d.is_dir()),
        key=lambda d: int(d.name) if d.name.isdigit() else d.name,
    )
    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    for label_dir in label_dirs:
        label = label_dir.name
        describe_label(args.base, label)
        keys, images = read_label_batch(args.base, label, args.batch_size)
        fig = _grid(images, ncols=args.ncols, title=f"label {label} (n={len(images)})")
        if fig is None:
            continue
        if args.out_dir is not None:
            out = args.out_dir / f"label_{label}.png"
            fig.savefig(out, dpi=150)
            plt.close(fig)
            print(f"  saved {out}")
    if args.out_dir is None:
        plt.show()


if __name__ == "__main__":
    main()
