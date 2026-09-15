#!/usr/bin/env python
"""Unified evaluator: score a mids_plus checkpoint OR an upstream FFAA mids.pth across one or more
eval sets, reporting per-set ACC/AUC/AP plus the cross-set summary (mean ACC, sACC). Writes a JSON
results file so the orchestrator can assemble the final report.

Examples:
  # mids_plus checkpoint across dataset buckets
  python scripts/eval_models.py --model-type mids_plus --checkpoint runs/real/mids_pp/best.pt \
      --data CelebDF:buckets/CelebDF.json WFFD:buckets/WFFD.json --out results/mids_pp_buckets.json
  # upstream baseline
  python scripts/eval_models.py --model-type upstream --upstream-mids checkpoints/mids_upstream.pth \
      --data eval2000:runs/real/eval2000.json --out results/upstream_eval2000.json
"""

from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from mids_plus.checkpoint import import_upstream_mids_checkpoint, load_checkpoint
from mids_plus.config import MidsPlusConfig
from mids_plus.data import MidsAnswersDataset, build_image_transform, collate_fn
from mids_plus.metrics import binary_metrics, cross_set_summary
from mids_plus.model import build_model
from mids_plus.selector import make_decision_batch
from mids_plus.tokenize import build_tokenizer


def parse_specs(specs):
    out = []
    for spec in specs:
        if ":" in spec and not os.path.exists(spec):
            name, path = spec.split(":", 1)
        else:
            name, path = os.path.splitext(os.path.basename(spec))[0], spec
        out.append((name, path))
    return out


@torch.no_grad()
def eval_one(model, cfg, tokenizer, transform, path, device, batch_size, num_workers, limit, progress_label=""):
    ds = MidsAnswersDataset(path, cfg, "val", transform)
    if limit and limit < len(ds.data):
        ds.data = ds.data[:limit]
    total = len(ds.data)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    s = cfg.samples_per_image
    y_true, y_pred, y_score, neutral_correct, n = [], [], [], 0, 0
    t_start = time.time()
    next_mark = 20000
    for batch in loader:
        images = batch["image"].to(device)
        enc = tokenizer(batch["texts"], return_tensors="pt", padding="longest", truncation=True, max_length=512).to(device)
        b = images.size(0)
        out = model(enc, images, None, b, s - 1, 0)
        scores = F.softmax(out["logits"].float(), dim=2)[:, :s, :]
        _, preds, _, forgery = make_decision_batch(batch["answers_result"], scores, chunk_size=s)
        y_true.extend(batch["cls_label"].tolist()); y_pred.extend(preds); y_score.extend(forgery)
        for i in range(b):
            truth = "fake" if batch["cls_label"][i].item() == 1 else "real"
            neutral_correct += int(batch["answers_result"][i * s] == truth)
        n += b
        if progress_label and n >= next_mark:
            rate = n / max(1e-6, time.time() - t_start)
            print(f"  [{progress_label}] {n}/{total} ({rate:.0f}/s)", flush=True)
            next_mark += 20000
    m = binary_metrics(y_true, y_pred, y_score)
    m["n"] = n
    m["n_real"] = int(sum(1 for t in y_true if t == 0))
    m["n_fake"] = int(sum(1 for t in y_true if t == 1))
    m["neutral_acc"] = neutral_correct / max(1, n)
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-type", choices=["mids_plus", "upstream"], required=True)
    p.add_argument("--checkpoint", default=None, help="mids_plus best.pt (for model-type mids_plus)")
    p.add_argument("--upstream-mids", default=None, help="upstream mids.pth (for model-type upstream)")
    p.add_argument("--image-model", default="models/clip-vit-large-patch14-336")
    p.add_argument("--text-model", default="models/t5-base")
    p.add_argument("--data", nargs="+", required=True, help="[name:]path JSON eval sets")
    p.add_argument("--limit-per-set", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--tag", default="model")
    p.add_argument("--out", default=None)
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
    transform = build_image_transform(cfg, "val")

    per_set = {}
    t0 = time.time()
    for name, path in parse_specs(args.data):
        m = eval_one(model, cfg, tokenizer, transform, path, device, args.batch_size, args.num_workers,
                     args.limit_per_set, progress_label=f"{args.tag}:{name}")
        per_set[name] = m
        print(f"[{args.tag}] {name:>16s}  n={m['n']:5d} (r{m['n_real']}/f{m['n_fake']})  "
              f"ACC={m['acc']*100:5.2f}  AUC={m['auc']*100:5.2f}  AP={m['ap']*100:5.2f}  neutral={m['neutral_acc']*100:5.2f}", flush=True)
    summary = cross_set_summary({k: v["acc"] for k, v in per_set.items()})
    print(f"[{args.tag}] MEAN ACC={summary['mean_acc']*100:.2f}  sACC={summary['sacc']:.2f}  ({time.time()-t0:.0f}s)", flush=True)

    result = {"tag": args.tag, "model_type": args.model_type,
              "checkpoint": args.checkpoint or args.upstream_mids,
              "per_set": per_set, "summary": summary}
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        json.dump(result, open(args.out, "w"), indent=2)
        print(f"[{args.tag}] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
