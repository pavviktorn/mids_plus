#!/usr/bin/env python
"""Build a balanced, genuinely-unseen, FULL-IMAGE (no face crop) test set in temp/ from user data.

Sources (all 0% overlap with training, verified):
  REAL  : axonlabs real videos (Reals_6ids_177video) + photo_reals jpgs       -> cls_label 0
  FAKE  : axonlabs PAD videos (Replay/Silicone/Textile/Latex/Wrapped3D/...)    -> cls_label 1 (pad)
          gasstation deepfake pngs                                             -> cls_label 1 (deepfake)

Videos: we extract K evenly-spaced FULL frames (downscaled to max-side 1280, NO crop) into temp.
Ready images (photo_reals, gasstation) are referenced in place. Output: per-source bucket JSONs +
balanced ALL/REAL/FAKE under runs/real/heldout2/. Deterministic (sorted lists + fixed strides).
"""
import cv2, glob, json, os

A = "/datasets/work/vLLM/data/axonlabs_data_1"
GS = "/datasets/work/vLLM/data_processing/down_gs/face_images/gasstation"
OUT = "runs/real/heldout2"
FR = f"{OUT}/frames"
os.makedirs(f"{OUT}/buckets", exist_ok=True)
os.makedirs(f"{FR}/real_video", exist_ok=True)
os.makedirs(f"{FR}/pad_video", exist_ok=True)
K_REAL, K_PAD, MAXSIDE = 8, 8, 1280

def vids(root):
    out = []
    for ext in ("*.mp4", "*.MOV", "*.mov", "*.MP4"):
        out += glob.glob(f"{root}/**/{ext}", recursive=True)
    return sorted(set(out))

def extract(vpath, outdir, k, tag):
    """Save k evenly-spaced full frames (downscaled, no crop). Returns saved abs paths."""
    cap = cv2.VideoCapture(vpath)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n <= 0:
        cap.release(); return []
    lo, hi = int(0.1 * n), int(0.9 * n) if n > 10 else n - 1
    hi = max(hi, lo)
    idxs = [int(lo + (hi - lo) * j / max(k - 1, 1)) for j in range(k)] if k > 1 else [n // 2]
    flat = tag + "__" + vpath.replace(A, "").replace(GS, "").strip("/").replace("/", "_").rsplit(".", 1)[0]
    saved = []
    for j, fi in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, fr = cap.read()
        if not ok or fr is None:
            continue
        h, w = fr.shape[:2]
        sc = MAXSIDE / max(h, w)
        if sc < 1:
            fr = cv2.resize(fr, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
        op = f"{outdir}/{flat}_f{j}.jpg"
        cv2.imwrite(op, fr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        saved.append(op)
    cap.release()
    return saved

# ---------- REAL ----------
real = []
rv = vids(f"{A}/real/Reals_6ids_177video")
print(f"real videos: {len(rv)}", flush=True)
for i, v in enumerate(rv):
    for p in extract(v, f"{FR}/real_video", K_REAL, "rv"):
        real.append({"id": f"rv{i}", "image": p, "cls_label": 0, "source": "real_video"})
    if i % 40 == 0: print(f"  real_video {i}/{len(rv)} -> {len(real)} frames", flush=True)
n_realvid = len(real)
photos = sorted(glob.glob(f"{A}/real/photo_reals_replay_mobile_460ids/**/*.jpg", recursive=True))
for i, p in enumerate(photos):
    real.append({"id": f"rp{i}", "image": p, "cls_label": 0, "source": "real_photo"})
print(f"REAL total = {len(real)} (video {n_realvid} + photo {len(photos)})", flush=True)

# ---------- FAKE: PAD videos (balanced across types) ----------
R = len(real)
pad_target = R // 2
pad_types = sorted([d for d in glob.glob(f"{A}/fake/*") if os.path.isdir(d)])
per_type_vids = {t: vids(t) for t in pad_types}
# round-robin pick videos across types until we have enough to hit pad_target at K_PAD frames each
need_vids = pad_target // K_PAD + 1
picked = []
ti = 0
cursor = {t: 0 for t in pad_types}
while len(picked) < need_vids and any(cursor[t] < len(per_type_vids[t]) for t in pad_types):
    t = pad_types[ti % len(pad_types)]; ti += 1
    if cursor[t] < len(per_type_vids[t]):
        picked.append(per_type_vids[t][cursor[t]]); cursor[t] += 1
fake = []
for i, v in enumerate(picked):
    for p in extract(v, f"{FR}/pad_video", K_PAD, "pad"):
        fake.append({"id": f"pad{i}", "image": p, "cls_label": 1, "source": "pad_video"})
    if i % 30 == 0: print(f"  pad_video {i}/{len(picked)} -> {len(fake)} frames", flush=True)
n_pad = len(fake)
# ---------- FAKE: gasstation deepfake (deterministic stride to balance) ----------
df_target = R - n_pad
gs = sorted(glob.glob(f"{GS}/**/*.png", recursive=True))
stride = max(1, len(gs) // df_target)
gs_sel = gs[::stride][:df_target]
for i, p in enumerate(gs_sel):
    fake.append({"id": f"df{i}", "image": p, "cls_label": 1, "source": "deepfake_img"})
print(f"FAKE total = {len(fake)} (pad {n_pad} + deepfake {len(gs_sel)})", flush=True)

# ---------- write ----------
def dump(name, items): json.dump(items, open(f"{OUT}/{name}.json", "w"))
def dumpb(name, items): json.dump(items, open(f"{OUT}/buckets/{name}.json", "w"))
alli = real + fake
dump("heldout2_ALL", alli); dump("heldout2_REAL", real); dump("heldout2_FAKE", fake)
import collections
bysrc = collections.defaultdict(list)
for it in alli: bysrc[it["source"]].append(it)
for s, items in bysrc.items(): dumpb(s, items)

print("\n=== SUMMARY ===")
print(f"{'source':14s} {'N':>6s} {'label':>6s}")
for s in sorted(bysrc): print(f"{s:14s} {len(bysrc[s]):6d} {bysrc[s][0]['cls_label']:>6d}")
print(f"{'REAL':14s} {len(real):6d}")
print(f"{'FAKE':14s} {len(fake):6d}")
print(f"{'ALL':14s} {len(alli):6d}")
open(f"{OUT}/DONE_BUILD", "w").close()
