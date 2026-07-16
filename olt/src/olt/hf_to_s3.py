"""
The helper functions are added to make imagenet-1k from HF port to our format in s3

the main code you need to run in colab, with appropriate helper imports.
I'm skipping adding this as its not used a lot.


```
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset
import shutil


ds = load_dataset(
    "ILSVRC/imagenet-1k",
    split="train",
    streaming=True,
)

push_every = 10_000

TMP_PATH = Path("tmp")
TMP_PATH.mkdir(parents=True, exist_ok=True)

for i, r in tqdm(enumerate(ds), total=ds.info.splits["train"].num_examples):
  if i % push_every == 0:
    print("Pushing to S3")
    ! aws s3 cp --quiet --recursive {TMP_PATH} s3://narang99-private/imagenet/
    ! rm -rf {TMP_PATH}
    ! mkdir {TMP_PATH}

  save_path = TMP_PATH / str(r["label"]) / get_image_cas_filename(r["image"])
  save_path.parent.mkdir(exist_ok=True, parents=True)
  with save_path.open("wb") as f:
    r["image"].save(f, format=r["image"].format)



for tag in tqdm(<all-labels>):
  download_data_from_s3(S3_BASE_DIR_KEY, BASE_WORKDIR, tag)
  write_image_shards(BASE_WORKDIR, str(tag), 512, DATASET_BASE / "images")
  shutil.rmtree(BASE_WORKDIR / str(tag))
```


"""

import hashlib
from io import BytesIO
from pathlib import Path

import webdataset as wds
from PIL import Image
from tqdm import tqdm

from olt.s3 import aws_s3_sync


def get_image_cas_filename(image: Image.Image, hash_algo=hashlib.sha256) -> str:
    """
    Generates a content-addressable filename for a PIL Image using a hash.

    Args:
        image: The PIL Image object.
        hash_algo: The hashing algorithm to use (default: hashlib.sha256).

    Returns:
        A string representing the content-addressable filename (hash + extension).
    """
    # Convert PIL image to bytes
    img_byte_arr = BytesIO()
    # Assume JPEG for now, you might want to infer or pass the format
    image.save(img_byte_arr, format="JPEG")
    img_byte_arr = img_byte_arr.getvalue()

    # Compute the hash of the image bytes
    image_hash = hash_algo(img_byte_arr).hexdigest()

    # Determine file extension (simplistic approach)
    # In a real scenario, you'd likely want to store/infer the original format
    extension = "jpg"  # Defaulting to jpg for now, but could be dynamic

    return f"{image_hash}.{extension}"


def download_data_from_s3(base_src_dir_key: str, base_workdir: Path, label: int | str):
    label = str(label)
    workdir = base_workdir / label
    workdir.mkdir(exist_ok=True, parents=True)

    src = f"{base_src_dir_key.strip('/')}/{label}".strip("/")
    dest = str(workdir).strip("/")

    print("calling s3", src, dest)
    aws_s3_sync(src, dest)


def write_image_shards(
    base_dir: Path, label: str, images_in_one_shard: int, shards_out_dir: Path
):
    # example: write_image_shards(BASE_WORKDIR, str(LABEL), 64,  Path("inputs"))
    # we write {__key__: filename, jpg: raw-file-bytes}
    image_paths = sorted((base_dir / label).glob("*.jpg"))

    out_dir = shards_out_dir / label
    out_dir.mkdir(exist_ok=True, parents=True)
    pattern = str(out_dir / "%06d.tar")

    with wds.ShardWriter(pattern, maxcount=images_in_one_shard) as sink:  # ty: ignore
        for p in tqdm(image_paths):
            with open(p, "rb") as f:
                jpg_bytes = f.read()

            sink.write(
                {
                    "__key__": p.stem,
                    "jpg": jpg_bytes,
                }
            )
