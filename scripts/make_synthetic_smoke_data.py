#!/usr/bin/env python
"""Generate a tiny synthetic dataset in the FFAA MIDS JSON format.

Writes random face-sized images + ``train.json``/``val.json`` so the pipeline can be exercised
end-to-end without the real 1M dataset.  Only numpy + cv2 are needed (no torch).

    python scripts/make_synthetic_smoke_data.py --out-dir /tmp/mids_smoke --num-images 16
"""

from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np

DESC = "Image description: a synthetic face used for smoke testing."
REASON_REAL = "Forgery reasoning: textures and lighting look natural; no blending seams."
REASON_FAKE = "Forgery reasoning: visible blending boundary and irregular skin texture near the jaw."


def build_answers(cls_label: int) -> list:
    """1 neutral (claims the truth) + 2 hypothetical (real / fake).

    4-class label convention = 2*cls_label + (1 if claim=='fake' else 0).
    """
    truth = "real" if cls_label == 0 else "fake"
    triples = [truth, "real", "fake"]
    answers = []
    for result in triples:
        reason = REASON_REAL if result == "real" else REASON_FAKE
        label = 2 * cls_label + (1 if result == "fake" else 0)
        answers.append({"content": f"{DESC}\n{reason}", "result": result, "label": label})
    return answers


def make_split(out_dir: str, img_dir: str, n: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    items = []
    for i in range(n):
        cls_label = int(i % 2)
        img = rng.integers(0, 256, size=(128, 128, 3), dtype=np.uint8)
        path = os.path.join(img_dir, f"img_{seed}_{i:04d}.png")
        cv2.imwrite(path, img)
        items.append({"image": path, "cls_label": cls_label, "answers": build_answers(cls_label)})
    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-images", type=int, default=16)
    args = parser.parse_args()

    img_dir = os.path.join(args.out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    train = make_split(args.out_dir, img_dir, args.num_images, seed=1)
    val = make_split(args.out_dir, img_dir, max(4, args.num_images // 2), seed=2)
    with open(os.path.join(args.out_dir, "train.json"), "w") as f:
        json.dump(train, f, indent=2)
    with open(os.path.join(args.out_dir, "val.json"), "w") as f:
        json.dump(val, f, indent=2)
    print(f"Wrote {len(train)} train + {len(val)} val items to {args.out_dir}")


if __name__ == "__main__":
    main()
