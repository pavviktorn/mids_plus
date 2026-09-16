#!/usr/bin/env python
"""Load an upstream FFAA mids.pth into MIDS++ `mids_original` mode and evaluate it.

Gives the apples-to-apples baseline that MIDS++ must beat, on your own data, using the exact
trained MIDS you ship. Example:

    python scripts/import_and_eval_upstream.py \
        --mids-pth checkpoints/mids_upstream.pth \
        --data /datasets/newout/vqa_info_2+13+4+3_fmt/mids_eval.json \
        --limit 2000 --batch-size 16
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mids_plus.checkpoint import import_upstream_mids_checkpoint
from mids_plus.config import MidsPlusConfig
from mids_plus.data import MidsAnswersDataset, build_image_transform, collate_fn
from mids_plus.metrics import binary_metrics
from mids_plus.model import build_model
from mids_plus.selector import make_decision_batch


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mids-pth", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--image-model", default="models/clip-vit-large-patch14-336")
    p.add_argument("--text-model", default="models/t5-base")
    p.add_argument("--limit", type=int, default=0, help="evaluate only the first N items (0 = all)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = MidsPlusConfig.from_dict(dict(
        image_model_path=args.image_model, text_model_path=args.text_model,
        clip_adapt="unfreeze", unfreeze_vision_last_layers=2, tune_layer_norm=False,
        artifact_enabled=False, samples_per_image=3, num_classes=4,
        skip_missing_images=True, num_workers=args.num_workers,
    ))

    print(f"device={device}; building mids_original model from {args.image_model} + {args.text_model}")
    model = build_model(cfg).to(device).eval()
    import_upstream_mids_checkpoint(model, args.mids_pth)

    from transformers import T5Tokenizer
    tokenizer = T5Tokenizer.from_pretrained(args.text_model, use_fast=False, legacy=False)
    transform = build_image_transform(cfg, "val")

    ds = MidsAnswersDataset(args.data, cfg, "val", transform)
    if args.limit and args.limit < len(ds.data):
        ds.data = ds.data[: args.limit]
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, collate_fn=collate_fn)

    s = cfg.samples_per_image
    y_true, y_pred, y_score, neutral_correct = [], [], [], 0
    t0 = time.time()
    n = 0
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            enc = tokenizer(batch["texts"], return_tensors="pt", padding="longest", truncation=True, max_length=512).to(device)
            b = images.size(0)
            out = model(enc, images, None, b, s - 1, 0)
            scores = F.softmax(out["logits"].float(), dim=2)[:, :s, :]
            _, preds, _, forgery = make_decision_batch(batch["answers_result"], scores, chunk_size=s)
            y_true.extend(batch["cls_label"].tolist())
            y_pred.extend(preds)
            y_score.extend(forgery)
            # neutral-answer baseline (answer 0's claim)
            for i in range(b):
                neutral = batch["answers_result"][i * s]
                truth = "fake" if batch["cls_label"][i].item() == 1 else "real"
                neutral_correct += int(neutral == truth)
            n += b
            if n % (args.batch_size * 10) == 0:
                print(f"  {n} items  ({n/(time.time()-t0):.1f}/s)")

    m = binary_metrics(y_true, y_pred, y_score)
    print("-" * 56)
    print(f"items evaluated : {n}")
    print(f"upstream MIDS   : ACC={m['acc']*100:.2f}  AUC={m['auc']*100:.2f}  AP={m['ap']*100:.2f}")
    print(f"neutral baseline: ACC={neutral_correct/max(1,n)*100:.2f}  (trust the neutral answer)")
    print(f"elapsed         : {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
