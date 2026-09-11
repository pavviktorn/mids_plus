#!/usr/bin/env python
"""Build a 30K, many-subset, FULL-IMAGE (no crop) unseen test set per user spec.

REAL 10K     : photo_reals (460, referenced) + 177 axonlabs real videos x ~54 full frames.
               subsets = 6 ids + real_photo.
PAD 10K      : ALL 1728 axonlabs PAD videos x 6 full frames. subsets = 7 attack types.
DEEPFAKE 10K : gasstation pngs (referenced), ~equal per week folder across all 32 weeks.
               subsets = version_week (e.g. v3_2026W05).
Video frames are extracted in parallel (multiprocessing), downscaled to max-side 1280, NO crop.
Writes per-subset bucket JSONs + manifest.json {subset:{group,label}} under runs/real/heldout3/.
"""
import cv2, glob, json, math, os, collections
from multiprocessing import Pool

A = "/datasets/work/vLLM/data/axonlabs_data_1"
GS = "/datasets/work/vLLM/data_processing/down_gs/face_images/gasstation"
OUT = "runs/real/heldout3"
FR = f"{OUT}/frames"
MAXSIDE = 1280
REAL_TARGET = PAD_TARGET = DF_TARGET = 10000
os.makedirs(f"{OUT}/buckets", exist_ok=True)
os.makedirs(f"{FR}/real", exist_ok=True)
os.makedirs(f"{FR}/pad", exist_ok=True)

def vids(root):
    out = []
    for ext in ("*.mp4", "*.MOV", "*.mov", "*.MP4"):
        out += glob.glob(f"{root}/**/{ext}", recursive=True)
    return sorted(set(out))

def extract_job(args):
    vpath, k, outdir, subset, label, idp = args
    try:
        cap = cv2.VideoCapture(vpath)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n <= 0:
            cap.release(); return []
        lo, hi = int(0.05 * n), (int(0.95 * n) if n > 10 else n - 1)
        hi = max(hi, lo)
        idxs = sorted(set(int(lo + (hi - lo) * j / max(k - 1, 1)) for j in range(k))) if k > 1 else [n // 2]
        recs = []
        for j, fi in enumerate(idxs):
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, fr = cap.read()
            if not ok or fr is None:
                continue
            h, w = fr.shape[:2]
            sc = MAXSIDE / max(h, w)
            if sc < 1:
                fr = cv2.resize(fr, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
            op = f"{outdir}/{subset}__{idp}_f{j}.jpg"
            cv2.imwrite(op, fr, [cv2.IMWRITE_JPEG_QUALITY, 92])
            recs.append({"id": f"{idp}_f{j}", "image": op, "cls_label": label, "subset": subset})
        cap.release()
        return recs
    except Exception:
        return []

def main():
    manifest = {}
    buckets = collections.defaultdict(list)

    # ---------- REAL videos (per-id subsets) ----------
    ids = sorted(os.listdir(f"{A}/real/Reals_6ids_177video"))
    real_vid_jobs = []
    nrv = sum(len(vids(f"{A}/real/Reals_6ids_177video/{i}")) for i in ids)
    k_real = max(1, round((REAL_TARGET - 460) / max(nrv, 1)))  # frames/video to hit ~10k with photos
    for i in ids:
        vs = vids(f"{A}/real/Reals_6ids_177video/{i}")
        sub = f"real_{i}"
        manifest[sub] = {"group": "REAL", "label": 0}
        for vi, v in enumerate(vs):
            real_vid_jobs.append((v, k_real, f"{FR}/real", sub, 0, f"{i}_{vi}"))
    print(f"REAL: {nrv} videos, k={k_real} frames/video; ids={len(ids)}", flush=True)

    # ---------- PAD videos (per-type subsets), ALL videos ----------
    pad_types = sorted([os.path.basename(d) for d in glob.glob(f"{A}/fake/*") if os.path.isdir(d)])
    npv = sum(len(vids(f"{A}/fake/{t}")) for t in pad_types)
    k_pad = max(1, math.ceil(PAD_TARGET / max(npv, 1)))
    pad_jobs = []
    for t in pad_types:
        vs = vids(f"{A}/fake/{t}")
        sub = "pad_" + t.split("_")[0]  # Advanced/Replay/Silicone/Textile/Wrapped/latex/Replay
        sub = "pad_" + t.replace(" ", "")  # keep full distinct type name
        manifest[sub] = {"group": "PAD", "label": 1}
        for vi, v in enumerate(vs):
            pad_jobs.append((v, k_pad, f"{FR}/pad", sub, 1, f"{t.replace(' ','')}_{vi}"))
    print(f"PAD: {npv} videos, k={k_pad} frames/video; types={len(pad_types)}", flush=True)

    # ---------- parallel extraction ----------
    nproc = min(16, (os.cpu_count() or 8))
    print(f"extracting with {nproc} workers: {len(real_vid_jobs)} real + {len(pad_jobs)} pad videos", flush=True)
    with Pool(nproc) as p:
        for recs in p.imap_unordered(extract_job, real_vid_jobs, chunksize=4):
            for r in recs: buckets[r["subset"]].append(r)
        rv_done = sum(len(v) for v in buckets.values())
        print(f"  real_video frames extracted: {rv_done}", flush=True)
        for recs in p.imap_unordered(extract_job, pad_jobs, chunksize=8):
            for r in recs: buckets[r["subset"]].append(r)
    print(f"  total extracted (real+pad): {sum(len(v) for v in buckets.values())}", flush=True)

    # ---------- real photos (referenced) ----------
    photos = sorted(glob.glob(f"{A}/real/photo_reals_replay_mobile_460ids/**/*.jpg", recursive=True))
    manifest["real_photo"] = {"group": "REAL", "label": 0}
    for i, ph in enumerate(photos):
        buckets["real_photo"].append({"id": f"rp{i}", "image": ph, "cls_label": 0, "subset": "real_photo"})

    # ---------- deepfake (referenced), per-week subsets, ~equal per week ----------
    weeks = []
    for v in sorted(glob.glob(f"{GS}/gs-images-*")):
        for w in sorted(glob.glob(f"{v}/archives/*")):
            if os.path.isdir(w):
                weeks.append((os.path.basename(v).replace("gs-images-", ""), os.path.basename(w), w))
    per_week = math.ceil(DF_TARGET / max(len(weeks), 1))
    df_total = 0
    for ver, wk, wpath in weeks:
        pngs = sorted(glob.glob(f"{wpath}/**/*.png", recursive=True))
        if not pngs: continue
        stride = max(1, len(pngs) // per_week)
        sel = pngs[::stride][:per_week]
        sub = f"df_{ver}_{wk}"
        manifest[sub] = {"group": "DEEPFAKE", "label": 1}
        for i, p in enumerate(sel):
            buckets[sub].append({"id": f"{sub}_{i}", "image": p, "cls_label": 1, "subset": sub})
        df_total += len(sel)
    print(f"DEEPFAKE: {len(weeks)} weeks, ~{per_week}/week, total {df_total}", flush=True)

    # ---------- write ----------
    for sub, items in buckets.items():
        json.dump(items, open(f"{OUT}/buckets/{sub}.json", "w"))
    json.dump(manifest, open(f"{OUT}/manifest.json", "w"), indent=1)
    allitems = [x for items in buckets.values() for x in items]
    json.dump(allitems, open(f"{OUT}/heldout3_ALL.json", "w"))

    # summary
    grp = collections.Counter()
    for sub, items in buckets.items():
        grp[manifest[sub]["group"]] += len(items)
    print("\n=== SUMMARY ===")
    print(f"{'subset':28s} {'group':9s} {'label':>5s} {'N':>6s}")
    for sub in sorted(buckets, key=lambda s: (manifest[s]["group"], s)):
        m = manifest[sub]
        print(f"{sub:28s} {m['group']:9s} {m['label']:>5d} {len(buckets[sub]):>6d}")
    print(f"\nGROUP TOTALS: " + "  ".join(f"{g}={grp[g]}" for g in grp))
    print(f"ALL = {len(allitems)} ; subsets = {len(buckets)}")
    open(f"{OUT}/DONE_BUILD3", "w").close()

if __name__ == "__main__":
    main()
