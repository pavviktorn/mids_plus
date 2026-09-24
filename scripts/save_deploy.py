#!/usr/bin/env python
"""Fit + persist the recommended MLLM-free fusion operating point, and report final EVAL metrics.

Recipe: per-model Platt calibration of the m1 (mean-marginal P(fake)) score on the TUNE set,
average the calibrated probs, threshold at the TUNE real-recall floor. Everything fit on TUNE;
EVAL = video-masked heldout (never used for fitting). Saves a deployable JSON.
"""
from __future__ import annotations
import argparse, json
import numpy as np
from analyze_frontier import HELD, TUNE, load_dump, recall_at, tau_for_realfloor, auc
from analyze_fusion import aligned_matrix, platt, sigmoid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--variant", default="m1_meanmarg")
    ap.add_argument("--out", default="runs/real/frontier/deploy_config.json")
    args = ap.parse_args()
    tm = json.load(open(f"{TUNE}/manifest.json")); em = json.load(open(f"{HELD}/manifest.json"))
    mask = json.load(open(f"{TUNE}/tune_videos.json"))
    md_t, md_e = [], []
    for spec in args.models:
        n, td, ed = spec.split(":")
        md_t.append((n, load_dump(td, tm))); md_e.append((n, load_dump(ed, em, mask_tune=mask)))
    Xt, gt, subt, names = aligned_matrix(md_t, args.variant)
    Xe, ge, sube, _ = aligned_matrix(md_e, args.variant)
    yt = (gt != "REAL").astype(int)
    platts = {}
    Ct, Ce = np.zeros_like(Xt), np.zeros_like(Xe)
    for j, nm in enumerate(names):
        a, b = platt(Xt[:, j], yt); platts[nm] = [a, b]
        Ct[:, j] = sigmoid(a * Xt[:, j] + b); Ce[:, j] = sigmoid(a * Xe[:, j] + b)
    st, se = Ct.mean(1), Ce.mean(1)
    out = {"models": names, "variant": args.variant, "method": "mean_of_platt",
           "platt": platts, "eval_auc": auc(se, ge), "operating_points": {}}
    print(f"=== DEPLOY: {'+'.join(names)} (mean of Platt-calibrated {args.variant}) ===")
    print(f"EVAL fake-vs-real AUC = {auc(se,ge)*100:.2f}")
    print(f"{'floor':>6s} {'tau':>7s} | EVAL pad / real / deepfake / worst-pad")
    for fl in [0.84, 0.85, 0.86, 0.88, 0.90]:
        tau = tau_for_realfloor(st, gt, fl)
        err, epr, edr, ewp, ewn = recall_at(se, ge, sube, tau)
        out["operating_points"][f"{fl:.2f}"] = dict(tau=float(tau), pad=epr, real=err, deepfake=edr,
                                                     worst_pad=ewp, worst_name=ewn)
        print(f"{fl:6.2f} {tau:7.4f} | {epr*100:5.1f} / {err*100:5.1f} / {edr*100:5.1f} / {ewp*100:5.1f} ({ewn})")
    # per-pad-type @ floor 0.85
    tau = tau_for_realfloor(st, gt, 0.85)
    out["pad_types_at_0.85"] = {}
    print("\nper-PAD-type EVAL recall @ floor 0.85:")
    for s in sorted(np.unique(sube[ge == "PAD"])):
        m = (ge == "PAD") & (sube == s); r = float((se[m] >= tau).mean())
        out["pad_types_at_0.85"][s] = r; print(f"  {s:32s} {r*100:5.1f}")
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
