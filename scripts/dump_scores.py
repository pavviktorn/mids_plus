#!/usr/bin/env python
"""Dump PER-SAMPLE softmax probabilities for a checkpoint over a set of bucket JSONs.

One GPU forward pass per (model, dataset); all threshold / fusion / stacking analysis is then done
offline on the dumped arrays (CPU). Fixed-answer, full-image (letterbox, no crop) -- same inference
path as eval_fixed_answers.py, but we keep the raw softmax instead of collapsing with make_decision.

For each bucket ``name`` writes ``{out}/{name}.npz`` with:
  probs : (N, S, C) float16   softmax over C classes for each of the S=3 fixed candidate answers
  cls   : (N,)      int8       binary cls_label (0 real / 1 fake) from the bucket JSON
  ids   : (N,)      <U         per-sample id (used to recover the source video for eval masking)
C is 4 (binary 2x2) or 9 (3x3). Candidate-answer order matches CLAIM_TYPES = [deepfake, pad, real].

Optional crop-free TTA (--tta): averages softmax over {orig, hflip[, zoomout, zoomout+hflip]}.
"""
from __future__ import annotations
import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from mids_plus.checkpoint import import_upstream_mids_checkpoint, load_checkpoint
from mids_plus.config import MidsPlusConfig
from mids_plus.data import CLIP_MEAN, CLIP_STD, build_image_transform
from mids_plus.model import build_model
from mids_plus.tokenize import build_tokenizer

# identity-exchange -> deepfake(2), PAD -> pad(1), real -> real(0)
TEXTS = [
    "misaligned features, smooth skin, inconsistent lighting, blending artifacts, "
    "unnatural integration between the face and body/the face and the background",
    "flat depth, uniform lighting, screen smoothing, moire patterns, screen borders, "
    "UI, reflections, glossy/plastic skin, rigid edges, hand-holding signs, recapture blur",
    "consistent facial features, natural skin texture, consistent lighting, "
    "normal facial 3D depth, the absence of clear manipulation or visible artifacts",
]
_MEAN = torch.tensor(CLIP_MEAN).view(3, 1, 1)
_STD = torch.tensor(CLIP_STD).view(3, 1, 1)


def _letterbox(rgb, size, margin=0.0):
    import cv2
    h, w = rgb.shape[:2]
    s = int(max(h, w) * (1.0 + margin))
    canvas = np.zeros((s, s, 3), dtype=rgb.dtype)
    top, left = (s - h) // 2, (s - w) // 2
    canvas[top:top + h, left:left + w] = rgb
    img = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    return (t - _MEAN) / _STD


def make_variants(rgb, size, tta):
    """Return list of (3,size,size) tensors. Crop-free only."""
    base = _letterbox(rgb, size, 0.0)
    out = [base]
    if tta in ("flip", "multi"):
        out.append(torch.flip(base, dims=[-1]))
    if tta == "multi":
        zo = _letterbox(rgb, size, 0.18)            # zoom-out (more background, no crop)
        out += [zo, torch.flip(zo, dims=[-1])]
    return out


class BucketDataset(Dataset):
    def __init__(self, path, transform_size, tta, full_image):
        self.data = [it for it in json.load(open(path)) if os.path.exists(it["image"])]
        self.size = transform_size
        self.tta = tta
        self.full_image = full_image
        self._std_tf = None

    def set_std_transform(self, tf):
        self._std_tf = tf

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        import cv2
        it = self.data[i]
        bgr = cv2.imread(it["image"], cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self.full_image:
            vs = make_variants(rgb, self.size, self.tta)
        else:
            vs = [self._std_tf(rgb)]
        return torch.stack(vs, 0), int(it["cls_label"]), str(it.get("id", i))


def collate(batch):
    imgs = torch.stack([b[0] for b in batch], 0)   # (B, V, 3, H, W)
    return imgs, [b[1] for b in batch], [b[2] for b in batch]


@torch.no_grad()
def dump_one(model, tok, cfg, path, device, bs, nw, tta, full_image, name):
    ds = BucketDataset(path, 336, tta, full_image)
    if not full_image:
        ds.set_std_transform(build_image_transform(cfg, "val"))
    loader = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=nw, collate_fn=collate)
    enc = tok(TEXTS, return_tensors="pt", padding="longest", truncation=True, max_length=512)
    ids0, mask0 = enc["input_ids"], enc["attention_mask"]
    P, CLS, IDS, t0 = [], [], [], time.time()
    for imgs, labels, sids in loader:
        B, V = imgs.size(0), imgs.size(1)
        flat = imgs.reshape(B * V, *imgs.shape[2:]).to(device)
        ids = ids0.repeat(B * V, 1).to(device)
        mask = mask0.repeat(B * V, 1).to(device)
        out = model({"input_ids": ids, "attention_mask": mask}, flat, None, B * V, 1, 1)
        sm = F.softmax(out["logits"].float(), dim=2)         # (B*V, S, C)
        sm = sm.reshape(B, V, sm.size(1), sm.size(2)).mean(1)  # TTA average -> (B, S, C)
        P.append(sm.cpu().numpy().astype(np.float16))
        CLS.extend(labels); IDS.extend(sids)
    probs = np.concatenate(P, 0) if P else np.zeros((0, 3, 4), np.float16)
    dt = time.time() - t0
    print(f"  [{name}] N={len(CLS):5d} C={probs.shape[-1]} ({len(CLS)/max(dt,1e-9):.0f}/s)", flush=True)
    return probs, np.asarray(CLS, np.int8), np.asarray(IDS)


def parse_specs(specs):
    out = []
    for s in specs:
        if ":" in s and not os.path.exists(s):
            name, path = s.split(":", 1)
        else:
            name, path = os.path.splitext(os.path.basename(s))[0], s
        out.append((name, path))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-type", choices=["mids_plus", "upstream"], default="mids_plus")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--upstream-mids", default=None)
    p.add_argument("--image-model", default="models/clip-vit-large-patch14-336")
    p.add_argument("--text-model", default="models/t5-base")
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--tta", choices=["none", "flip", "multi"], default="none")
    p.add_argument("--full-image", action="store_true")
    p.add_argument("--out", required=True, help="output directory for per-bucket .npz")
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
    tok = build_tokenizer(cfg)
    os.makedirs(args.out, exist_ok=True)
    print(f"[dump] tta={args.tta} full_image={args.full_image} -> {args.out}", flush=True)
    for name, path in parse_specs(args.data):
        op = f"{args.out}/{name}.npz"
        if os.path.exists(op):
            print(f"  [{name}] exists, skip", flush=True); continue
        probs, cls, ids = dump_one(model, tok, cfg, path, device, args.batch_size,
                                   args.num_workers, args.tta, args.full_image, name)
        np.savez_compressed(op, probs=probs, cls=cls, ids=ids)


if __name__ == "__main__":
    main()
