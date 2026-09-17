#!/usr/bin/env python
"""Probe: how much does each model's 4-class output depend on the candidate ANSWER TEXT?

make_decision derives the binary verdict from the per-candidate 4-class scores. If those scores
don't move when we change the candidate text (keeping the image fixed), the decision is
image-dominated and MLLM-free inference is lossless by construction. We feed each image several
different candidate texts and measure the spread of the marginalized P(fake)=s[2]+s[3] across texts.
~0 spread => text-invariant (exactly MLLM-free-safe); larger => text-sensitive.
"""
from __future__ import annotations
import argparse, json, os, time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from mids_plus.checkpoint import import_upstream_mids_checkpoint, load_checkpoint
from mids_plus.config import MidsPlusConfig
from mids_plus.data import build_image_transform
from mids_plus.model import build_model
from mids_plus.tokenize import build_tokenizer

# A spread of candidate texts: the 3 fixed templates + 2 extra real/fake phrasings + an empty-ish one.
PROBE_TEXTS = [
    "misaligned features, smooth skin, inconsistent lighting, blending artifacts, unnatural integration between the face and body/the face and the background",
    "flat depth, uniform lighting, screen smoothing, moire patterns, screen borders, UI, reflections, glossy/plastic skin, rigid edges, hand-holding signs, recapture blur",
    "consistent facial features, natural skin texture, consistent lighting, normal facial 3D depth, the absence of clear manipulation or visible artifacts",
    "this is clearly a genuine authentic photograph of a real person with no manipulation",
    "this face is obviously a deepfake forgery with heavy manipulation artifacts everywhere",
    "a photo",
]


class DS(Dataset):
    def __init__(self, path, cfg, transform, limit):
        data = [it for it in json.load(open(path)) if os.path.exists(it["image"])]
        self.data = data[:limit] if limit else data
        self.transform = transform
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        import cv2
        it = self.data[i]
        bgr = cv2.imread(it["image"], cv2.IMREAD_COLOR)
        return self.transform(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)), int(it["cls_label"])


def collate(b):
    return torch.stack([x[0] for x in b], 0), [x[1] for x in b]


@torch.no_grad()
def run(tag, model, tokenizer, transform, cfg, path, device, bs, nw, limit):
    ds = DS(path, cfg, transform, limit)
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw, collate_fn=collate)
    enc = tokenizer(PROBE_TEXTS, return_tensors="pt", padding="longest", truncation=True, max_length=512)
    ids0, mask0 = enc["input_ids"], enc["attention_mask"]
    T = len(PROBE_TEXTS)
    spreads, pfakes_all = [], []
    n = 0
    for imgs, labels in loader:
        b = imgs.size(0)
        imgs = imgs.to(device)
        ids = ids0.repeat(b, 1).to(device)     # (T*b,L) b-major: [t0..t5, t0..t5, ...]
        mask = mask0.repeat(b, 1).to(device)
        out = model({"input_ids": ids, "attention_mask": mask}, imgs, None, b, 1, 1)
        s = F.softmax(out["logits"].float(), dim=2)        # (b, T, 4)
        pfake = (s[:, :, 2] + s[:, :, 3])                  # (b, T) marginal P(true=fake)
        spread = (pfake.max(dim=1).values - pfake.min(dim=1).values)  # per-image range across texts
        spreads.append(spread.cpu())
        pfakes_all.append(pfake.cpu())
        n += b
    spreads = torch.cat(spreads)
    pf = torch.cat(pfakes_all)                              # (N,T)
    # also: std across texts of P(fake), averaged over images
    std_across_text = pf.std(dim=1, unbiased=False).mean().item()
    return {
        "tag": tag, "n": int(spreads.numel()),
        "pfake_range_mean": spreads.mean().item(),       # avg over images of (max-min) P(fake) across texts
        "pfake_range_max": spreads.max().item(),
        "pfake_std_across_text_mean": std_across_text,
        "frac_images_range_gt_0.01": (spreads > 0.01).float().mean().item(),
        "frac_images_range_gt_0.05": (spreads > 0.05).float().mean().item(),
    }


def load_model(args, device):
    if args.model_type == "upstream":
        cfg = MidsPlusConfig.from_dict(dict(
            image_model_path=args.image_model, text_model_path=args.text_model,
            clip_adapt="unfreeze", unfreeze_vision_last_layers=2, tune_layer_norm=False,
            artifact_enabled=False, skip_missing_images=True))
        model = build_model(cfg).to(device).eval()
        import_upstream_mids_checkpoint(model, args.upstream_mids)
    else:
        model, cfg = load_checkpoint(args.checkpoint, device=device, overrides={"skip_missing_images": True})
    return model, cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-type", choices=["mids_plus", "upstream"], required=True)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--upstream-mids", default=None)
    p.add_argument("--image-model", default="models/clip-vit-large-patch14-336")
    p.add_argument("--text-model", default="models/t5-base")
    p.add_argument("--data", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--limit", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=8)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, cfg = load_model(args, device)
    tokenizer = build_tokenizer(cfg)
    transform = build_image_transform(cfg, "val")
    r = run(args.tag, model, tokenizer, transform, cfg, args.data, device,
            args.batch_size, args.num_workers, args.limit)
    print(json.dumps(r))


if __name__ == "__main__":
    main()
