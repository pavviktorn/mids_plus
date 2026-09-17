#!/usr/bin/env python
"""Random subsets of large MIDS test JSONs (fixed seed), filtering missing images.

Both models are then evaluated on the SAME subset files for a fair comparison.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--per", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    specs = []
    for p in sorted(glob.glob(os.path.join(args.src_dir, "*.json"))):
        name = os.path.splitext(os.path.basename(p))[0]
        data = json.load(open(p))
        idxs = list(range(len(data)))
        random.Random(args.seed).shuffle(idxs)
        picked = []
        for i in idxs:
            if os.path.exists(data[i]["image"]):
                picked.append(data[i])
                if len(picked) >= args.per:
                    break
        out = os.path.join(args.out_dir, f"{name}.json")
        json.dump(picked, open(out, "w"))
        nr = sum(1 for x in picked if x["cls_label"] == 0)
        nf = len(picked) - nr
        specs.append(f"{name}:{out}")
        print(f"{name:14s} picked={len(picked):5d} (real {nr} / fake {nf}) from {len(data)}")
    print("SPECS=" + " ".join(specs))


if __name__ == "__main__":
    main()
