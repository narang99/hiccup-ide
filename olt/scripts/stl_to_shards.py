"""Convert STL-10 into the webdataset .tar shard layout the pipeline expects.

STL-10 is the higher-resolution (96x96, 10-class) drop-in for CIFAR: sharper
feature-viz / patch crops for interp, same pipeline. This mirrors
`cifar_to_shards.py` exactly; only the torchvision API differs (STL10 uses
`split=` and `.labels`, CIFAR uses `train=` and `.targets`).

The whole pipeline (stages/s0..s4) groups inputs by a per-class directory and
reads image shards via `olt.shards.read_image_shard`, which decodes samples
with keys `__key__` and `jpg`. The class id is simply the STL class 0..9 — no
stage signatures change, only the data on disk.

Output layout (mirrors the ImageNet image-shard layout):

    <out_dir>/<class_id>/000000.tar
        __key__ = "<class_id>_<running_index>"
        jpg     = <96x96 RGB JPEG bytes>

Usage:
    uv run scripts/stl_to_shards.py --out-dir data/stl/image-shards --split test
    uv run scripts/stl_to_shards.py --out-dir data/stl/image-shards --split train

The `test` split (8000 labeled images) is the natural analogue of the ImageNet
inputs the pipeline was run over; use `train` (5000 labeled) if you also want to
probe training images. `--flat-image-dir` optionally also dumps every image as
`<flat>/<key>.jpeg`, which is what `NeuronParentAnalyser.get_activations_for_image`
and the report renderers load individual inputs from.
"""

import argparse
import io
from pathlib import Path

import webdataset as wds
from torchvision.datasets import STL10
from tqdm import tqdm


def _jpeg_bytes(pil_img) -> bytes:
    buf = io.BytesIO()
    pil_img.convert("RGB").save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def convert(
    out_dir: Path,
    split: str,
    data_root: Path,
    shard_maxcount: int,
    flat_image_dir: Path | None,
    max_per_label: int | None = None,
) -> None:
    ds = STL10(root=str(data_root), split=split, download=True)

    # group sample indices by class so each class gets its own shard dir
    by_class: dict[int, list[int]] = {}
    for idx, label in enumerate(ds.labels):
        by_class.setdefault(int(label), []).append(idx)

    if flat_image_dir is not None:
        flat_image_dir.mkdir(parents=True, exist_ok=True)

    for class_id, idxs in tqdm(sorted(by_class.items()), desc=f"stl-{split}"):
        if max_per_label is not None:
            idxs = idxs[:max_per_label]  # cap images per class
        class_dir = out_dir / str(class_id)
        class_dir.mkdir(parents=True, exist_ok=True)
        # webdataset splits into 000000.tar, 000001.tar, ... at shard_maxcount
        pattern = str(class_dir / "%06d.tar")
        with wds.ShardWriter(pattern, maxcount=shard_maxcount) as sink:  # ty: ignore
            for running, idx in enumerate(idxs):
                img, _ = ds[idx]
                key = f"{class_id}_{running:05d}"
                jpg = _jpeg_bytes(img)
                sink.write({"__key__": key, "jpg": jpg})
                if flat_image_dir is not None:
                    (flat_image_dir / f"{key}.jpeg").write_bytes(jpg)

    print(f"wrote shards for {len(by_class)} classes to {out_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--split", choices=["train", "test"], default="test")
    p.add_argument("--data-root", type=Path, default=Path("data/stl-raw"))
    p.add_argument(
        "--shard-maxcount",
        type=int,
        default=1000,
        help="max samples per .tar shard",
    )
    p.add_argument(
        "--flat-image-dir",
        type=Path,
        default=None,
        help="also dump each image as <dir>/<key>.jpeg (used by the analyser/reports)",
    )
    p.add_argument(
        "--max-per-label",
        type=int,
        default=None,
        help="cap images per class (e.g. 128); default writes all of them",
    )
    args = p.parse_args()
    convert(
        args.out_dir,
        args.split,
        args.data_root,
        args.shard_maxcount,
        args.flat_image_dir,
        args.max_per_label,
    )


if __name__ == "__main__":
    main()
