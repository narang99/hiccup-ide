"""
Tag neuron-cluster concepts using an LLM (Claude or Gemini).

For every `cluster_*_combined.jpeg` in a directory, sends the image to a
vision-capable model and asks it to name the visual concept the cluster
represents (the same task done manually earlier in this session).

Usage:
    python tag_concepts.py --provider claude --dir .
    python tag_concepts.py --provider gemini --dir . --output tags.json

Deps (install separately):
    pip install anthropic   # for --provider claude
    pip install google-genai  # for --provider gemini

Auth (either provider): set the relevant API key env var
    ANTHROPIC_API_KEY
    GEMINI_API_KEY
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
from pathlib import Path
from typing import Protocol

PROMPT = """\
This image is a grid of image patches from one channel/neuron of a CNN, \
each overlaid with a heatmap of that channel's activation.

Look at what the highlighted (bright) regions have in common across the \
patches, and name the single visual concept this channel is detecting.

Respond with:
1. A short tag (3-6 words).
2. One sentence explaining the shared visual pattern you saw.
"""


class ConceptTagger(Protocol):
    def tag(self, image_bytes: bytes, prompt: str) -> str: ...


class ClaudeTagger:
    def __init__(self, model: str = "claude-opus-4-8"):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model

    def tag(self, image_bytes: bytes, prompt: str) -> str:
        image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
        response = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_b64,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        return next(block.text for block in response.content if block.type == "text")


class GeminiTagger:
    def __init__(self, model: str = "gemini-2.5-pro"):
        from google import genai

        self.client = genai.Client()
        self.model = model

    def tag(self, image_bytes: bytes, prompt: str) -> str:
        from google.genai import types

        response = self.client.models.generate_content(
            model=self.model,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
                prompt,
            ],
        )
        return response.text


def build_tagger(provider: str, model: str | None) -> ConceptTagger:
    if provider == "claude":
        return ClaudeTagger(model) if model else ClaudeTagger()
    if provider == "gemini":
        return GeminiTagger(model) if model else GeminiTagger()
    raise ValueError(f"Unknown provider: {provider}")


def cluster_id_from_path(path: Path) -> str:
    match = re.match(r"cluster_(-?\d+)_combined", path.stem)
    return match.group(1) if match else path.stem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["claude", "gemini"], required=True)
    parser.add_argument("--model", default=None, help="Override the default model")
    parser.add_argument("--dir", default=".", help="Directory containing cluster images")
    parser.add_argument(
        "--pattern", default="cluster_*_combined.jpeg", help="Glob pattern for images"
    )
    parser.add_argument(
        "--skip-noise",
        action="store_true",
        help="Skip cluster_-1 (conventionally the noise/unclustered bucket)",
    )
    parser.add_argument("--output", default="concept_tags.json")
    args = parser.parse_args()

    directory = Path(args.dir)
    image_paths = sorted(directory.glob(args.pattern))
    if args.skip_noise:
        image_paths = [p for p in image_paths if cluster_id_from_path(p) != "-1"]

    tagger = build_tagger(args.provider, args.model)

    results: dict[str, str] = {}
    for path in image_paths:
        cluster_id = cluster_id_from_path(path)
        print(f"Tagging cluster {cluster_id} ({path.name})...")
        results[cluster_id] = tagger.tag(path.read_bytes(), PROMPT)
        print(results[cluster_id])
        print("-" * 40)

    output_path = directory / args.output
    output_path.write_text(json.dumps(results, indent=2))
    print(f"Wrote {len(results)} tags to {output_path}")


if __name__ == "__main__":
    main()
