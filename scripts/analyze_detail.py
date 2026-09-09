#!/usr/bin/env python
"""Detailed EVAL frontier for ONE fusion config across a sweep of TUNE real-recall floors.

Threshold chosen on TUNE at each floor; EVAL (video-masked heldout) reported. Also prints the
per-pad-type EVAL recall at the lowest floor so we can see which attacks bound PAD-recall.
"""
from __future__ import annotations
import argparse, json
import numpy as np
from analyze_frontier import HELD, TUNE, load_dump, recall_at, tau_for_realfloor, auc
from analyze_fusion import aligned_matrix, calibrate_cols, fuse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--variant", default="m1_meanmarg")
    ap.add_argument("--method", default="mean")
    ap.add_argument("--floors", nargs="+", type=float, default=[0.80, 0.82, 0.84, 0.85, 0.86, 0.88, 0.90])
    args = ap.parse_args()
    tune_man = json.load(open(f"{TUNE}/manifest.json"))
    eval_man = json.load(open(f"{HELD}/manifest.json"))
    tune_mask = json.load(open(f"{TUNE}/tune_videos.json"))

    md_t, md_e = [], []
    for spec in args.models:
        name, tdir, edir = spec.split(":")
        md_t.append((name, load_dump(tdir, tune_man)))
        md_e.append((name, load_dump(edir, eval_man, mask_tune=tune_mask)))
    Xt, gt, subt, names = aligned_matrix(md_t, args.variant)
    Xe, ge, sube, _ = aligned_matrix(md_e, args.variant)
    yt = (gt != "REAL").astype(int)
    Ct, Ce = calibrate_cols(Xt, yt, Xe)
    st, se = fuse(args.method, Ct, Ce, yt, gt)

    print(f"\n=== {'+'.join(names)}  [{args.method}/{args.variant}] ===")
    print(f"TUNE auc(fake-vs-real)={auc(st,gt)*100:.2f} | EVAL auc={auc(se,ge)*100:.2f}")
    print(f"{'tune_floor':>10s} {'tau':>7s} | {'EVAL pad':>8s} {'real':>6s} {'df':>6s} {'wpad':>6s}")
    for fl in args.floors:
        tau = tau_for_realfloor(st, gt, fl)
        err, epr, edr, ewp, ewn = recall_at(se, ge, sube, tau)
        print(f"{fl:10.2f} {tau:7.3f} | {epr*100:8.1f} {err*100:6.1f} {edr*100:6.1f} {ewp*100:6.1f}  ({ewn})")

    # per-pad-type EVAL recall at the lowest floor
    fl = min(args.floors); tau = tau_for_realfloor(st, gt, fl)
    print(f"\n  per-PAD-type EVAL recall at tune_floor={fl} (tau={tau:.3f}):")
    for s in sorted(np.unique(sube[ge == "PAD"])):
        m = (ge == "PAD") & (sube == s)
        print(f"    {s:32s} n={int(m.sum()):5d}  rec={ (se[m]>=tau).mean()*100:5.1f}")
    # real subsets too (where do FPs concentrate)
    print(f"  per-REAL-subset EVAL recall at tune_floor={fl}:")
    for s in sorted(np.unique(sube[ge == "REAL"])):
        m = (ge == "REAL") & (sube == s)
        print(f"    {s:32s} n={int(m.sum()):5d}  rec={ (se[m]<tau).mean()*100:5.1f}")


if __name__ == "__main__":
    main()
