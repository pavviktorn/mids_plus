#!/usr/bin/env python
"""Evaluate MIDS with FIXED answer templates instead of MLLM-generated answers.

Replicates the idea in the user's inference_mids.py: feed MIDS the same 3 hand-written candidate
answers (identity-exchange / PAD / real) for every image, score, and decide. The image paths and
cls_label come from a labeled MIDS JSON; the JSON's MLLM answers are ignored. This lets us measure
the accuracy cost of dropping the MLLM, against the stored MLLM-answer numbers on identical images.

Deterministic: no cue-order shuffling (the candidate-order shuffle is a no-op for the selector, and
the cue-order shuffle only adds score noise). Makeup manipulation is treated as covered by the PAD
candidate (both are 'fake').
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from mids_plus.checkpoint import import_upstream_mids_checkpoint, load_checkpoint
from mids_plus.config import MidsPlusConfig
from mids_plus.data import CLIP_MEAN, CLIP_STD, build_image_transform
from mids_plus.metrics import binary_metrics, cross_set_summary
from mids_plus.model import build_model
from mids_plus.selector import make_decision, make_decision9
from mids_plus.tokenize import build_tokenizer


def build_fullimage_transform(size=336):
    """FULL-IMAGE transform: NO face crop, NO center crop. Letterbox-pad the whole frame to a
    square (preserves aspect, full field of view) then resize to size and CLIP-normalize."""
    import cv2
    mean = torch.tensor(CLIP_MEAN).view(3, 1, 1)
    std = torch.tensor(CLIP_STD).view(3, 1, 1)

    def _t(rgb):
        h, w = rgb.shape[:2]
        s = max(h, w)
        top, left = (s - h) // 2, (s - w) // 2
        canvas = np.zeros((s, s, 3), dtype=rgb.dtype)
        canvas[top:top + h, left:left + w] = rgb
        img = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        return (t - mean) / std

    return _t

# Exact templates from inference_mids.py (cue lists only; verdict kept separately for the selector).
FIXED = [
    ("fake", "misaligned features, smooth skin, inconsistent lighting, blending artifacts, "
             "unnatural integration between the face and body/the face and the background"),
    ("fake", "flat depth, uniform lighting, screen smoothing, moire patterns, screen borders, "
             "UI, reflections, glossy/plastic skin, rigid edges, hand-holding signs, recapture blur"),
    ("real", "consistent facial features, natural skin texture, consistent lighting, "
             "normal facial 3D depth, the absence of clear manipulation or visible artifacts"),
]
RESULTS = [a[0] for a in FIXED]
TEXTS = [a[1] for a in FIXED]
# 9-class claim types matching FIXED/TEXTS order: identity-exchange->deepfake(2), PAD->pad(1), real(0)
CLAIM_TYPES = [2, 1, 0]


class ImageOnlyDataset(Dataset):
    def __init__(self, path, cfg, transform):
        data = json.load(open(path))
        if getattr(cfg, "skip_missing_images", False):
            data = [it for it in data if os.path.exists(it["image"])]
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        import cv2

        it = self.data[i]
        bgr = cv2.imread(it["image"], cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return self.transform(rgb), int(it["cls_label"])


def collate(batch):
    imgs = torch.stack([b[0] for b in batch], 0)
    labels = [b[1] for b in batch]
    return imgs, labels


def parse_specs(specs):
    out = []
    for s in specs:
        if ":" in s and not os.path.exists(s):
            name, path = s.split(":", 1)
        else:
            name, path = os.path.splitext(os.path.basename(s))[0], s
        out.append((name, path))
    return out


@torch.no_grad()
def eval_one(model, tokenizer, transform, cfg, path, device, bs, nw, limit, label):
    ds = ImageOnlyDataset(path, cfg, transform)
    if limit and limit < len(ds.data):
        ds.data = ds.data[:limit]
    total = len(ds.data)
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw, collate_fn=collate)
    enc = tokenizer(TEXTS, return_tensors="pt", padding="longest", truncation=True, max_length=512)
    ids0, mask0 = enc["input_ids"], enc["attention_mask"]
    y_true, y_pred, y_score, n, t0 = [], [], [], 0, time.time()
    for imgs, labels in loader:
        b = imgs.size(0)
        imgs = imgs.to(device)
        ids = ids0.repeat(b, 1).to(device)        # (3b, L), b-major: [a0,a1,a2, a0,a1,a2, ...]
        mask = mask0.repeat(b, 1).to(device)
        out = model({"input_ids": ids, "attention_mask": mask}, imgs, None, b, 1, 1)
        scores = F.softmax(out["logits"].float(), dim=2)  # (b, 3, C) ; C=4 (binary) or 9 (3x3)
        nine = scores.size(2) == 9
        for j in range(b):
            if nine:
                _, pred, _, forgery, _ = make_decision9(CLAIM_TYPES, scores[j])
            else:
                _, pred, _, forgery = make_decision(RESULTS, scores[j])
            y_pred.append(pred); y_score.append(forgery)
        y_true.extend(labels); n += b
        if n >= 20000 and (n // 20000) != ((n - b) // 20000):
            print(f"  [{label}] {n}/{total} ({n/(time.time()-t0):.0f}/s)", flush=True)
    m = binary_metrics(y_true, y_pred, y_score)
    m["n"] = n
    m["n_real"] = sum(1 for t in y_true if t == 0)
    m["n_fake"] = sum(1 for t in y_true if t == 1)
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-type", choices=["mids_plus", "upstream"], required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--upstream-mids", default=None)
    p.add_argument("--image-model", default="models/clip-vit-large-patch14-336")
    p.add_argument("--text-model", default="models/t5-base")
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--limit-per-set", type=int, default=0)
    p.add_argument("--tag", default="fixed")
    p.add_argument("--out", default=None)
    p.add_argument("--full-image", action="store_true",
                   help="feed the FULL image (letterbox, no face/center crop) instead of CLIP center-crop")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.model_type == "upstream":
        cfg = MidsPlusConfig.from_dict(dict(
            image_model_path=args.image_model, text_model_path=args.text_model,
            clip_adapt="unfreeze", unfreeze_vision_last_layers=2, tune_layer_norm=False,
            artifact_enabled=False, skip_missing_images=True, num_workers=args.num_workers))
        model = build_model(cfg).to(device).eval()
        import_upstream_mids_checkpoint(model, args.upstream_mids)
    else:
        model, cfg = load_checkpoint(args.checkpoint, device=device, overrides={"skip_missing_images": True})
    tokenizer = build_tokenizer(cfg)
    transform = build_fullimage_transform() if args.full_image else build_image_transform(cfg, "val")
    if args.full_image:
        print("[full-image] letterbox (no face/center crop) transform", flush=True)

    per_set = {}
    for name, path in parse_specs(args.data):
        m = eval_one(model, tokenizer, transform, cfg, path, device, args.batch_size,
                     args.num_workers, args.limit_per_set, f"{args.tag}:{name}")
        per_set[name] = m
        print(f"[{args.tag}] {name:>16s}  n={m['n']:5d} (r{m['n_real']}/f{m['n_fake']})  "
              f"ACC={m['acc']*100:5.2f}  AUC={m['auc']*100:5.2f}  AP={m['ap']*100:5.2f}", flush=True)
    summary = cross_set_summary({k: v["acc"] for k, v in per_set.items()})
    print(f"[{args.tag}] MEAN ACC={summary['mean_acc']*100:.2f}  sACC={summary['sacc']:.2f}", flush=True)
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump({"tag": args.tag, "answers": "fixed", "per_set": per_set, "summary": summary}, open(args.out, "w"), indent=2)


if __name__ == "__main__":
    main()
