#!/usr/bin/env python
"""Apply the SAME face-quality filter used for the heldout EVAL real set to the TUNE real videos.

Tune real must match eval real: frontal, proper face size, not heavily posed/occluded. We re-extract
the held-out tune real videos (from tune_videos.json) and keep only frames passing the identical
SCRFD + 3D-landmark-pose filter as build_real_filtered.py. NO crop (kept frames stay full-image).
Overwrites runs/real/tune/buckets/real_*.json with the filtered frames; ids keep the {id}_{vi}_f{j}
scheme so video-disjoint masking against the heldout EVAL still aligns. PAD/DEEPFAKE tune untouched.
"""
import cv2, glob, json, os, shutil, collections, numpy as np
from multiprocessing import Pool

A = "/datasets/work/vLLM/data/axonlabs_data_1"
TUNE = "runs/real/tune"
FR = f"{TUNE}/frames/real"
MAXSIDE = 1280
CAND_PER_VIDEO = 150
KEEP_CAP_PER_VIDEO = 55
SCORE = 0.65
HMIN, HMAX = 0.15, 0.85
MINPX = 80
POSE = 28.0

_app = None
def _init():
    global _app
    from insightface.app import FaceAnalysis
    _app = FaceAnalysis(name="buffalo_l", allowed_modules=['detection', 'landmark_3d_68'],
                        providers=['CPUExecutionProvider'])
    _app.prepare(ctx_id=-1, det_size=(640, 640))

def _downscale(fr):
    h, w = fr.shape[:2]
    sc = MAXSIDE / max(h, w)
    return cv2.resize(fr, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA) if sc < 1 else fr

def _passes(img):
    h, w = img.shape[:2]
    faces = _app.get(img)
    if not faces:
        return False
    faces.sort(key=lambda f: -(f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    f = faces[0]
    x1, y1, x2, y2 = f.bbox
    fw, fh = x2 - x1, y2 - y1
    area = fw * fh
    if len(faces) > 1:
        f2 = faces[1]; a2 = (f2.bbox[2] - f2.bbox[0]) * (f2.bbox[3] - f2.bbox[1])
        if a2 > 0.5 * area:
            return False
    if f.det_score < SCORE:
        return False
    if not (HMIN <= fh / h <= HMAX):
        return False
    if min(fw, fh) < MINPX:
        return False
    pose = getattr(f, "pose", None)
    if pose is None or float(np.max(np.abs(pose))) > POSE:
        return False
    return True

def process_video(args):
    vpath, subset, idp = args
    try:
        cap = cv2.VideoCapture(vpath)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if n <= 0:
            cap.release(); return []
        lo, hi = int(0.05 * n), (int(0.95 * n) if n > 10 else n - 1); hi = max(hi, lo)
        k = min(CAND_PER_VIDEO, max(1, hi - lo + 1))
        idxs = sorted(set(int(lo + (hi - lo) * j / max(k - 1, 1)) for j in range(k)))
        kept = []
        for j, fi in enumerate(idxs):
            if len(kept) >= KEEP_CAP_PER_VIDEO:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
            ok, fr = cap.read()
            if not ok or fr is None:
                continue
            img = _downscale(fr)
            if _passes(img):
                op = f"{FR}/{subset}__{idp}_f{j}.jpg"
                cv2.imwrite(op, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                kept.append({"id": f"{idp}_f{j}", "image": op, "cls_label": 0, "subset": subset})
        cap.release()
        return kept
    except Exception:
        return []

def vids(root):
    out = []
    for ext in ("*.mp4", "*.MOV", "*.mov", "*.MP4"):
        out += glob.glob(f"{root}/**/{ext}", recursive=True)
    return sorted(set(out))

def main():
    tune_videos = json.load(open(f"{TUNE}/tune_videos.json"))
    # rebuild held-out real video paths from recorded vidkeys
    jobs = []
    for sub, vks in tune_videos.items():
        if not sub.startswith("real_"):
            continue
        idn = sub[len("real_"):]                       # e.g. id_R_10
        vlist = vids(f"{A}/real/Reals_6ids_177video/{idn}")
        for vk in vks:
            vi = int(vk[len(idn) + 1:])                # {idn}_{vi}
            if 0 <= vi < len(vlist):
                jobs.append((vlist[vi], sub, vk))
    print(f"filtering {len(jobs)} held-out tune real videos (POSE<={POSE}, score>={SCORE}, hfrac[{HMIN},{HMAX}])", flush=True)

    # clear old unfiltered tune real frames + buckets
    if os.path.isdir(FR):
        shutil.rmtree(FR)
    os.makedirs(FR, exist_ok=True)
    for f in glob.glob(f"{TUNE}/buckets/real_*.json"):
        os.remove(f)

    buckets = collections.defaultdict(list)
    nproc = min(16, (os.cpu_count() or 8))
    with Pool(nproc, initializer=_init) as p:
        done = 0
        for recs in p.imap_unordered(process_video, jobs, chunksize=2):
            for r in recs:
                buckets[r["subset"]].append(r)
            done += 1
            if done % 10 == 0:
                print(f"  videos {done}/{len(jobs)} -> {sum(len(v) for v in buckets.values())} kept", flush=True)

    for sub, items in buckets.items():
        json.dump(items, open(f"{TUNE}/buckets/{sub}.json", "w"))
    tot = sum(len(v) for v in buckets.values())
    print("\n=== FILTERED TUNE REAL ===")
    for sub in sorted(buckets):
        print(f"  {sub:16s} {len(buckets[sub]):6d}")
    print(f"REAL tune (filtered) = {tot}")
    open(f"{TUNE}/DONE_FILTER_REAL", "w").close()

if __name__ == "__main__":
    main()
