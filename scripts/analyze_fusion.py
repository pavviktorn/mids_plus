#!/usr/bin/env python
"""Multi-model FUSION + STACKING for binary fake-vs-real, frame-based.

Same discipline as analyze_frontier: everything (calibration, fusion weights, stacker, threshold)
is fit on the TUNE dumps; EVAL = video-masked heldout, used only for final reporting.

Fusion methods (each then thresholded at the TUNE real-recall floor):
  mean   : average of per-model Platt-calibrated P(fake)
  lr     : logistic-regression stacker over per-model scores  (learns who to trust)
  maxor  : max score  (OR-ish: catches any fake; hurts real)  -- reference only
  minand : min score  (AND-ish: conservative; protects real)  -- reference only

Picks the best model-subset+method by TUNE PAD-recall at real>=floor, reports the matching EVAL.
"""
from __future__ import annotations
import argparse, itertools, json
import numpy as np
from analyze_frontier import (HELD, TUNE, load_dump, fakescores, recall_at,
                              tau_for_realfloor, auc, auc_padonly)


def aligned_matrix(models_data, variant):
    """models_data: list of (name, dumpdict). Align per-subset by common ids.
    Returns X (N,M), group (N,), sub (N,), names."""
    names = [n for n, _ in models_data]
    subs = set.intersection(*[set(d.keys()) for _, d in models_data])
    cols, G, SUB = [], [], []
    rows_per_model = {n: [] for n in names}
    for sub in sorted(subs):
        # common ids across models for this subset
        idsets = [{str(i): k for k, i in enumerate(d[sub]["ids"])} for _, d in models_data]
        common = set.intersection(*[set(s.keys()) for s in idsets])
        common = sorted(common)
        if not common:
            continue
        grp = models_data[0][1][sub]["group"]
        G += [grp] * len(common); SUB += [sub] * len(common)
        for (n, d), idmap in zip(models_data, idsets):
            fs = fakescores(d[sub]["probs"])[variant]
            rows_per_model[n].append(fs[[idmap[c] for c in common]])
    X = np.stack([np.concatenate(rows_per_model[n]) for n in names], axis=1)
    return X, np.array(G), np.array(SUB), names


def platt(x, y):
    """Fit 1-D logistic x->P(y=1); return (a,b) for sigmoid(a*x+b)."""
    try:
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(C=1e3, max_iter=1000)
        lr.fit(x.reshape(-1, 1), y)
        return float(lr.coef_[0, 0]), float(lr.intercept_[0])
    except Exception:
        return 1.0, 0.0


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def calibrate_cols(Xt, yt, Xe):
    Ct, Ce = np.zeros_like(Xt), np.zeros_like(Xe)
    for j in range(Xt.shape[1]):
        a, b = platt(Xt[:, j], yt)
        Ct[:, j] = sigmoid(a * Xt[:, j] + b)
        Ce[:, j] = sigmoid(a * Xe[:, j] + b)
    return Ct, Ce


def fuse(method, Ct, Ce, yt, gt):
    if method == "mean":
        return Ct.mean(1), Ce.mean(1)
    if method == "maxor":
        return Ct.max(1), Ce.max(1)
    if method == "minand":
        return Ct.min(1), Ce.min(1)
    if method == "lr":
        try:
            from sklearn.linear_model import LogisticRegression
            # upweight PAD a little so the stacker prioritizes the binding class
            sw = np.where(gt == "PAD", 2.0, 1.0)
            lr = LogisticRegression(C=1.0, max_iter=2000)
            lr.fit(Ct, yt, sample_weight=sw)
            return lr.predict_proba(Ct)[:, 1], lr.predict_proba(Ce)[:, 1]
        except Exception:
            return Ct.mean(1), Ce.mean(1)
    raise ValueError(method)


def evaluate(st, gt, subt, se, ge, sube, floor):
    tau = tau_for_realfloor(st, gt, floor)
    trr, tpr, tdr, twp, twn = recall_at(st, gt, subt, tau)
    err, epr, edr, ewp, ewn = recall_at(se, ge, sube, tau)
    return dict(tau=tau,
                tune=dict(pad=tpr, real=trr, deepfake=tdr, worst_pad=twp, worst_name=twn),
                eval=dict(pad=epr, real=err, deepfake=edr, worst_pad=ewp, worst_name=ewn))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="name:tune_dir:eval_dir")
    ap.add_argument("--variant", default="m1_meanmarg")
    ap.add_argument("--floor", type=float, default=0.90)
    ap.add_argument("--methods", nargs="+", default=["mean", "lr", "maxor", "minand"])
    ap.add_argument("--max-subset", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tune_man = json.load(open(f"{TUNE}/manifest.json"))
    eval_man = json.load(open(f"{HELD}/manifest.json"))
    tune_mask = json.load(open(f"{TUNE}/tune_videos.json"))

    loaded = []
    for spec in args.models:
        name, tdir, edir = spec.split(":")
        td = load_dump(tdir, tune_man)
        ed = load_dump(edir, eval_man, mask_tune=tune_mask)
        loaded.append((name, td, ed))

    results = []
    allnames = [n for n, _, _ in loaded]
    for r in range(1, min(args.max_subset, len(loaded)) + 1):
        for combo in itertools.combinations(range(len(loaded)), r):
            md_t = [(loaded[i][0], loaded[i][1]) for i in combo]
            md_e = [(loaded[i][0], loaded[i][2]) for i in combo]
            Xt, gt, subt, names = aligned_matrix(md_t, args.variant)
            Xe, ge, sube, _ = aligned_matrix(md_e, args.variant)
            yt = (gt != "REAL").astype(int)
            Ct, Ce = calibrate_cols(Xt, yt, Xe)
            methods = args.methods if r > 1 else ["mean"]  # single model: just calibrated score
            for method in methods:
                st, se = fuse(method, Ct, Ce, yt, gt)
                for floor in (0.85, 0.90):
                    e = evaluate(st, gt, subt, se, ge, sube, floor)
                    results.append(dict(models=list(names), method=method, floor=floor, **e))

    # rank by TUNE pad-recall at the requested floor (discipline: select on tune)
    sel = [x for x in results if abs(x["floor"] - args.floor) < 1e-6]
    sel.sort(key=lambda x: (x["tune"]["pad"], x["tune"]["worst_pad"]), reverse=True)
    print(f"\n===== FUSION SEARCH (variant={args.variant}, select by TUNE pad@real{int(args.floor*100)}) =====")
    print(f"{'models':32s} {'meth':6s} | TUNE pad/real/df/wpad | EVAL pad/real/df/wpad  tau")
    for x in sel[:18]:
        t, e = x["tune"], x["eval"]
        print(f"{'+'.join(x['models']):32s} {x['method']:6s} | "
              f"{t['pad']*100:5.1f}/{t['real']*100:5.1f}/{t['deepfake']*100:5.1f}/{t['worst_pad']*100:5.1f} | "
              f"{e['pad']*100:5.1f}/{e['real']*100:5.1f}/{e['deepfake']*100:5.1f}/{e['worst_pad']*100:5.1f}  {x['tau']:.3f}")
    if args.out:
        json.dump(results, open(args.out, "w"), indent=2)
        print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
