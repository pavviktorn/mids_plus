#!/usr/bin/env python
"""Build a SOURCE-DISJOINT tuning set for threshold/fusion tuning.

Rules honored:
  * The 30K heldout3 EVAL set is never used to tune anything. This builds a SEPARATE set.
  * Disjoint from heldout at the VIDEO/IDENTITY level (not just frame level), because heldout
    extracted ~55 frames/real-video -> frame-disjoint alone would still be near-duplicate leakage.

How:
  REAL  : hold out a deterministic ~25% of each id's videos -> extract FRESH frames from them.
  PAD   : hold out a deterministic ~20% of each attack type's videos -> extract FRESH frames.
  DEEPF : gasstation pngs NOT selected by heldout3 (image-disjoint complement), per week.

Video indexing (vi) replicates heldout3's exact ``vids()`` ordering, so the recorded tune vidkeys
(``{id}_{vi}`` / ``{type}_{vi}``) align with heldout3's frame ids -> the analyzer can MASK these
videos OUT of the heldout EVAL to guarantee a clean video-disjoint evaluation.

Writes runs/real/tune/{buckets/*.json, manifest.json, tune_videos.json, DONE_BUILD_TUNE}.
"""
import cv2, glob, json, math, os, collections, random
from multiprocessing import Pool

A = "/datasets/work/vLLM/data/axonlabs_data_1"
GS = "/datasets/work/vLLM/data_processing/down_gs/face_images/gasstation"
HELD = "runs/real/heldout3"
OUT = "runs/real/tune"
FR = f"{OUT}/frames"
MAXSIDE = 1280
SEED = 1234
REAL_HOLD = 0.25          # fraction of real videos per id reserved for tuning
PAD_HOLD = 0.20           # fraction of pad videos per type reserved for tuning
K_REAL = 40               # frames per held-out real video
K_PAD = 15                # frames per held-out pad video
DF_PER_WEEK = 100         # deepfake tune images per week (from heldout's complement)

os.makedirs(f"{OUT}/buckets", exist_ok=True)
os.makedirs(f"{FR}/real", exist_ok=True)
os.makedirs(f"{FR}/pad", exist_ok=True)


def vids(root):
    """IDENTICAL to build_heldout3.vids so per-dir sort order (=> vi index) matches."""
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
    rng = random.Random(SEED)
    manifest = {}
    tune_videos = collections.defaultdict(list)   # subset -> [vidkey,...] held out for tuning
    buckets = collections.defaultdict(list)
    jobs = []

    # ---------------- REAL videos ----------------
    ids = sorted(os.listdir(f"{A}/real/Reals_6ids_177video"))
    for i in ids:
        vs = vids(f"{A}/real/Reals_6ids_177video/{i}")
        sub = f"real_{i}"
        manifest[sub] = {"group": "REAL", "label": 0}
        order = list(range(len(vs)))
        rng.shuffle(order)
        nhold = max(1, round(REAL_HOLD * len(vs)))
        hold = sorted(order[:nhold])
        for vi in hold:
            tune_videos[sub].append(f"{i}_{vi}")
            jobs.append((vs[vi], K_REAL, f"{FR}/real", sub, 0, f"{i}_{vi}"))
    print(f"REAL: {len(ids)} ids; held {sum(len(v) for k,v in tune_videos.items() if k.startswith('real_'))} videos", flush=True)

    # ---------------- PAD videos ----------------
    pad_types = sorted([os.path.basename(d) for d in glob.glob(f"{A}/fake/pad/*") if os.path.isdir(d)])
    for t in pad_types:
        vs = vids(f"{A}/fake/pad/{t}")
        sub = "pad_" + t.replace(" ", "")
        manifest[sub] = {"group": "PAD", "label": 1}
        order = list(range(len(vs)))
        rng.shuffle(order)
        nhold = max(1, round(PAD_HOLD * len(vs)))
        hold = sorted(order[:nhold])
        for vi in hold:
            tune_videos[sub].append(f"{t.replace(' ','')}_{vi}")
            jobs.append((vs[vi], K_PAD, f"{FR}/pad", sub, 1, f"{t.replace(' ','')}_{vi}"))
    print(f"PAD: {len(pad_types)} types; held {sum(len(v) for k,v in tune_videos.items() if k.startswith('pad_'))} videos", flush=True)

    # ---------------- extract real+pad frames in parallel ----------------
    nproc = min(16, (os.cpu_count() or 8))
    print(f"extracting {len(jobs)} held-out videos with {nproc} workers ...", flush=True)
    with Pool(nproc) as p:
        for recs in p.imap_unordered(extract_job, jobs, chunksize=4):
            for r in recs:
                buckets[r["subset"]].append(r)
    print(f"  real+pad tune frames: {sum(len(v) for v in buckets.values())}", flush=True)

    # ---------------- DEEPFAKE: heldout's complement, per week ----------------
    weeks = []
    for v in sorted(glob.glob(f"{GS}/gs-images-*")):
        for w in sorted(glob.glob(f"{v}/archives/*")):
            if os.path.isdir(w):
                weeks.append((os.path.basename(v).replace("gs-images-", ""), os.path.basename(w), w))
    DF_TARGET = 10000
    per_week_held = math.ceil(DF_TARGET / max(len(weeks), 1))   # == heldout3's per_week
    df_total = 0
    for ver, wk, wpath in weeks:
        pngs = sorted(glob.glob(f"{wpath}/**/*.png", recursive=True))
        if not pngs:
            continue
        stride = max(1, len(pngs) // per_week_held)
        held_sel = set(pngs[::stride][:per_week_held])          # what heldout3 used
        comp = [p for p in pngs if p not in held_sel]           # disjoint complement
        if not comp:
            continue
        st = max(1, len(comp) // DF_PER_WEEK)
        sel = comp[::st][:DF_PER_WEEK]
        sub = f"df_{ver}_{wk}"
        manifest[sub] = {"group": "DEEPFAKE", "label": 1}
        for k, ph in enumerate(sel):
            buckets[sub].append({"id": f"tune_{sub}_{k}", "image": ph, "cls_label": 1, "subset": sub})
        df_total += len(sel)
    print(f"DEEPFAKE: {len(weeks)} weeks, ~{DF_PER_WEEK}/week, total {df_total} (heldout complement)", flush=True)

    # ---------------- write ----------------
    for sub, items in buckets.items():
        json.dump(items, open(f"{OUT}/buckets/{sub}.json", "w"))
    json.dump(manifest, open(f"{OUT}/manifest.json", "w"), indent=1)
    json.dump({k: v for k, v in tune_videos.items()}, open(f"{OUT}/tune_videos.json", "w"), indent=1)
    allitems = [x for items in buckets.values() for x in items]
    json.dump(allitems, open(f"{OUT}/tune_ALL.json", "w"))

    grp = collections.Counter()
    for sub, items in buckets.items():
        grp[manifest[sub]["group"]] += len(items)
    print("\n=== TUNE SUMMARY ===")
    for sub in sorted(buckets, key=lambda s: (manifest[s]["group"], s)):
        print(f"  {sub:30s} {manifest[sub]['group']:9s} N={len(buckets[sub]):5d}")
    print("GROUP TOTALS: " + "  ".join(f"{g}={grp[g]}" for g in grp))
    print(f"ALL={len(allitems)}  subsets={len(buckets)}")
    open(f"{OUT}/DONE_BUILD_TUNE", "w").close()


if __name__ == "__main__":
    main()
