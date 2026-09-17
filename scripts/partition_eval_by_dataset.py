#!/usr/bin/env python
"""Partition a MIDS-format eval JSON into per-source-dataset buckets (proxy for cross-dataset eval).

True per-dataset JSONs don't exist, so we bucket items by recognizable forgery-dataset tokens found
in their image paths. This is an intra-eval-set proxy for cross-domain robustness, not a true
held-out cross-dataset benchmark -- interpret sACC across these buckets accordingly.

Writes one JSON per bucket (>= --min-items, capped at --cap, missing images skipped) and prints a
space-separated ``name:path`` list for the orchestrator to consume.
"""

from __future__ import annotations

import argparse
import json
import os

# Ordered (bucket, tokens): first token found in the image path wins.
BUCKETS = [
    ("CelebDF",        ["CelebDF", "Celeb-synthesis", "/cdf/"]),
    ("DeeperForensics",["DeeperForensics"]),
    ("WFFD",           ["WFFD"]),
    ("ReplayMobile",   ["Replay-Mobile"]),
    ("ReplayAttack",   ["Replay-Attack"]),
    ("CeFA",           ["CeFA"]),
    ("DFMNIST",        ["DFMNIST"]),
    ("Oulu",           ["Oulu"]),
    ("SiW",            ["SiW"]),
    ("HiFiMask",       ["HiFiMask", "SuHiFiMask"]),
    ("DeepLiveCam",    ["deep_live_cam"]),
    ("StyleGAN",       ["StyleGAN", "styleGAN"]),
    ("MegaFS",         ["MegaFS"]),
    ("DeepFakeFace",   ["DeepFakeFace", "DeepfakeTIMIT", "DFFD"]),
    ("RealWebcam",     ["alive-images_1M", "alive_images", "from_nizar"]),
]


def bucket_of(path: str):
    for name, tokens in BUCKETS:
        if any(t in path for t in tokens):
            return name
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--min-items", type=int, default=150)
    p.add_argument("--cap", type=int, default=1500)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    data = json.load(open(args.data))
    groups = {}
    for it in data:
        if not os.path.exists(it["image"]):
            continue
        b = bucket_of(it["image"])
        if b is None:
            continue
        groups.setdefault(b, []).append(it)

    specs = []
    print("bucket            items   real   fake")
    for name, items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        if len(items) < args.min_items:
            continue
        items = items[: args.cap]
        nr = sum(1 for it in items if it["cls_label"] == 0)
        nf = len(items) - nr
        path = os.path.join(args.out_dir, f"{name}.json")
        json.dump(items, open(path, "w"))
        specs.append(f"{name}:{path}")
        print(f"{name:16s} {len(items):6d} {nr:6d} {nf:6d}")

    # machine-readable line the orchestrator parses
    print("SPECS=" + " ".join(specs))


if __name__ == "__main__":
    main()
