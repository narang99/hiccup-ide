"""Generate a simple row-based HTML report from an iterator of dicts.

Each dict must have keys: layer, channel, count, cluster_label, image.
The image may be a PIL Image, numpy ndarray (H,W,3) or (3,H,W), or torch.Tensor.

Output layout:
  output_dir/
    index.html
    row_0.jpeg
    row_1.jpeg
    ...
"""

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


_HEAD = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Row Report</title>
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
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
  }

  h1.report-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: #e9ecef;
    border-bottom: 2px solid #4dabf7;
    padding-bottom: 0.5rem;
    margin-bottom: 0.5rem;
  }

  .row-section {
    background: #1e1f22;
    border: 1px solid #2c2e33;
    border-radius: 6px;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.4);
    overflow: hidden;
  }

  .section-meta {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.6rem;
    padding: 0.85rem 1.25rem;
    border-bottom: 1px solid #2c2e33;
    background: #25262b;
  }

  .meta-item {
    display: flex;
    align-items: center;
    gap: 0.35rem;
  }

  .meta-key {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: #868e96;
  }

  .meta-val {
    font-size: 0.82rem;
    font-weight: 600;
    color: #e9ecef;
    background: #1971c2;
    padding: 0.18rem 0.65rem;
    border-radius: 999px;
  }

  .meta-val.cluster { background: #2f9e44; }
  .meta-val.count   { background: #862e9c; }

  .section-image {
    padding: 1.25rem;
    display: flex;
    justify-content: center;
  }

  .section-image img {
    max-width: 100%;
    border-radius: 4px;
    border: 1px solid #2c2e33;
  }
</style>
</head>
<body>
<div class="container">
<h1 class="report-title">Row Report</h1>
"""

_TAIL = """\
</div>
</body>
</html>"""


def _to_pil(image) -> Image.Image:
    try:
        import torch
        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()
    except ImportError:
        pass

    if isinstance(image, np.ndarray):
        if image.ndim == 3 and image.shape[0] == 3:
            image = image.transpose(1, 2, 0)
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)
        image = (image * 255).astype(np.uint8)
        return Image.fromarray(image)

    if isinstance(image, Image.Image):
        return image.convert("RGB")

    raise TypeError(f"Unsupported image type: {type(image)}")


def _save_jpeg(image, path: Path) -> None:
    _to_pil(image).save(path, format="JPEG", quality=95)


def _row_section_html(record: dict, fname: str) -> str:
    return f"""\
<section class="row-section">
  <div class="section-meta">
    <div class="meta-item">
      <span class="meta-key">layer</span>
      <span class="meta-val">{record["layer"]}</span>
    </div>
    <div class="meta-item">
      <span class="meta-key">channel</span>
      <span class="meta-val">{record["channel"]}</span>
    </div>
    <div class="meta-item">
      <span class="meta-key">count</span>
      <span class="meta-val count">{record["count"]}</span>
    </div>
    <div class="meta-item">
      <span class="meta-key">cluster</span>
      <span class="meta-val cluster">{record["cluster_label"]}</span>
    </div>
  </div>
  <div class="section-image">
    <img src="{fname}" alt="cluster {record['cluster_label']}" />
  </div>
</section>"""


def generate_row_report(records: Iterable[dict], output_dir: str | Path) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    sections = []
    for i, record in enumerate(records):
        fname = f"row_{i}.jpeg"
        _save_jpeg(record["image"], out / fname)
        sections.append(_row_section_html(record, fname))

    html = _HEAD + "\n".join(sections) + "\n" + _TAIL
    index = out / "index.html"
    index.write_text(html, encoding="utf-8")
    print(f"Report saved to {out} ({len(sections)} rows)")
    return index
