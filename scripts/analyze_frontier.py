#!/usr/bin/env python
"""Threshold/frontier analysis for binary fake-vs-real, frame-based.

Pipeline discipline:
  * Thresholds are chosen ONLY on the TUNE dumps (runs/real/tune, video-disjoint from heldout).
  * EVAL = heldout3 dumps, MASKED to videos NOT held out for tuning (tune_videos.json) => the
    reported eval is video-disjoint from whatever picked the threshold. Nothing in eval tunes anything.

Fake-score variants derived from the (N,S,C) softmax (S=3 answers, claim order [deepfake,pad,real]):
  m1_meanmarg : mean over answers of P(true is fake)              [clean marginal, default]
  m2_realclaim: 1 - P(true=real) under the real-claim answer
  m3_selector : forgery_score from make_decision / make_decision9 (the shipped decision rule)

Target: maximize PAD-recall subject to REAL-recall >= floor. The operating threshold is the
real-recall quantile on TUNE; we then report PAD/REAL/DEEPFAKE recall on EVAL at that fixed threshold.
"""
from __future__ import annotations
import argparse, glob, json, os, re
import numpy as np

HELD = "runs/real/heldout3"
TUNE = "runs/real/tune"
CLAIM = [2, 1, 0]   # answer i -> claim type (deepfake, pad, real)


# ----------------------------- fake-score variants -----------------------------
def fakescores(probs):
    """probs: (N,S,C) float. Returns dict name->(N,) fake score in [0,1]."""
    p = probs.astype(np.float32)
    N, S, C = p.shape
    out = {}
    if C == 9:
        # marginal P(true=t) per answer = sum over claim columns {3t+0,3t+1,3t+2}
        def marg(a, t):
            return p[:, a, 3 * t] + p[:, a, 3 * t + 1] + p[:, a, 3 * t + 2]
        # m1: mean over answers of P(fake)=P(pad)+P(deepfake)=1-P(real)
        m1 = np.mean([1.0 - marg(a, 0) for a in range(S)], axis=0)
        # m2: real-claim answer (claim==0) -> index where CLAIM==0
        ar = CLAIM.index(0)
        m2 = 1.0 - marg(ar, 0)
        # m3: selector forgery
        col = np.stack([p[:, i, CLAIM[i]] + p[:, i, 3 + CLAIM[i]] + p[:, i, 6 + CLAIM[i]] for i in range(S)], 1)
        agree = np.stack([p[:, i, 3 * CLAIM[i] + CLAIM[i]] for i in range(S)], 1)
        match = np.where(col > 0, agree / np.maximum(col, 1e-9), 0.0)
        best = np.argmax(match, axis=1)
        claim_best = np.array(CLAIM)[best]
        mbest = match[np.arange(N), best]
        pred_fake = (claim_best != 0)
        m3 = np.where(pred_fake, mbest, 1.0 - mbest)
    else:  # 4-class: 0 r/r,1 r/f,2 f/r,3 f/f ; answers claim [fake,fake,real]
        is_real = np.array([0, 0, 1])  # answer claim real?
        m1 = np.mean([p[:, a, 2] + p[:, a, 3] for a in range(S)], axis=0)  # P(image fake)
        # real-claim answer = last (index 2)
        m2 = 1.0 - (p[:, 2, 0] / np.maximum(p[:, 2, 0] + p[:, 2, 2], 1e-9))
        match, preds = [], []
        for i in range(S):
            if is_real[i]:
                mi = p[:, i, 0] / np.maximum(p[:, i, 0] + p[:, i, 2], 1e-9); pi = 0
            else:
                mi = p[:, i, 3] / np.maximum(p[:, i, 3] + p[:, i, 1], 1e-9); pi = 1
            match.append(mi); preds.append(np.full(N, pi))
        match = np.stack(match, 1); preds = np.stack(preds, 1)
        best = np.argmax(match, axis=1)
        predb = preds[np.arange(N), best]; mbest = match[np.arange(N), best]
        m3 = np.where(predb == 1, mbest, 1.0 - mbest)
    out["m1_meanmarg"] = m1
    out["m2_realclaim"] = m2
    out["m3_selector"] = m3
    return out


# ----------------------------- loading -----------------------------
def vidkey(i):
    return re.sub(r"_f\d+$", "", i)


def load_dump(dump_dir, manifest, mask_tune=None):
    """Return dict: subset -> {probs,cls,group,ids}; mask_tune[subset]=set(vidkeys) to EXCLUDE."""
    data = {}
    for f in sorted(glob.glob(f"{dump_dir}/*.npz")):
        sub = os.path.splitext(os.path.basename(f))[0]
        if sub not in manifest:
            continue
        z = np.load(f, allow_pickle=True)
        probs, cls, ids = z["probs"], z["cls"], z["ids"]
        if mask_tune and sub in mask_tune and len(mask_tune[sub]):
            ex = set(mask_tune[sub])
            keep = np.array([vidkey(str(x)) not in ex for x in ids])
            probs, cls, ids = probs[keep], cls[keep], ids[keep]
        data[sub] = dict(probs=probs, cls=cls, group=manifest[sub]["group"], ids=ids)
    return data


def stacked(data, variant):
    """Concatenate all subsets -> (score, group, subset) arrays for one fake-score variant."""
    S, G, SUB = [], [], []
    for sub, d in data.items():
        if len(d["cls"]) == 0:
            continue
        fs = fakescores(d["probs"])[variant]
        S.append(fs); G += [d["group"]] * len(fs); SUB += [sub] * len(fs)
    return np.concatenate(S), np.array(G), np.array(SUB)


# ----------------------------- metrics -----------------------------
def recall_at(score, group, sub, tau):
    real = score[group == "REAL"]; pad = score[group == "PAD"]; df = score[group == "DEEPFAKE"]
    rr = float((real < tau).mean()) if len(real) else float("nan")
    pr = float((pad >= tau).mean()) if len(pad) else float("nan")
    dr = float((df >= tau).mean()) if len(df) else float("nan")
    # worst per-pad-type recall
    wp, wname = 1.0, None
    for s in np.unique(sub[group == "PAD"]):
        m = (group == "PAD") & (sub == s)
        r = float((score[m] >= tau).mean())
        if r < wp:
            wp, wname = r, s
    return rr, pr, dr, wp, wname


def tau_for_realfloor(score, group, floor):
    real = np.sort(score[group == "REAL"])
    if len(real) == 0:
        return 0.5
    # smallest tau s.t. fraction(real < tau) >= floor  ->  the floor-quantile of real scores
    idx = min(len(real) - 1, int(np.ceil(floor * len(real))) - 1)
    return float(real[max(idx, 0)])


def auc(score, group):
    pos = score[group != "REAL"]; neg = score[group == "REAL"]
    if not len(pos) or not len(neg):
        return float("nan")
    try:
        from sklearn.metrics import roc_auc_score
        y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        return float(roc_auc_score(y, np.concatenate([pos, neg])))
    except Exception:
        alls = np.concatenate([pos, neg]); order = np.argsort(alls, kind="mergesort")
        ranks = np.empty(len(alls)); ranks[order] = np.arange(1, len(alls) + 1)
        return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def auc_padonly(score, group):
    pos = score[group == "PAD"]; neg = score[group == "REAL"]
    if not len(pos) or not len(neg):
        return float("nan")
    try:
        from sklearn.metrics import roc_auc_score
        y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        return float(roc_auc_score(y, np.concatenate([pos, neg])))
    except Exception:
        return float("nan")


def analyze_model(name, tune_dir, eval_dir, tune_man, eval_man, tune_mask, floors=(0.85, 0.90)):
    res = {"model": name, "variants": {}}
    td = load_dump(tune_dir, tune_man)
    ed = load_dump(eval_dir, eval_man, mask_tune=tune_mask)
    for variant in ("m1_meanmarg", "m2_realclaim", "m3_selector"):
        ts, tg, tsub = stacked(td, variant)
        es, eg, esub = stacked(ed, variant)
        v = {"tune_auc": auc(ts, tg), "tune_auc_padonly": auc_padonly(ts, tg),
             "eval_auc": auc(es, eg), "eval_auc_padonly": auc_padonly(es, eg),
             "ops": {}}
        for fl in floors:
            tau = tau_for_realfloor(ts, tg, fl)
            trr, tpr, tdr, twp, twn = recall_at(ts, tg, tsub, tau)
            err, epr, edr, ewp, ewn = recall_at(es, eg, esub, tau)
            v["ops"][f"real{int(fl*100)}"] = dict(
                tau=tau,
                tune=dict(real=trr, pad=tpr, deepfake=tdr, worst_pad=twp, worst_name=twn),
                eval=dict(real=err, pad=epr, deepfake=edr, worst_pad=ewp, worst_name=ewn))
        res["variants"][variant] = v
    return res


def fmt(x):
    return f"{x*100:5.1f}" if x == x else "  -  "


def print_model(r):
    print(f"\n### {r['model']}")
    for variant, v in r["variants"].items():
        print(f"  [{variant}]  tuneAUC={v['tune_auc']*100:.2f} (PADvsREAL {v['tune_auc_padonly']*100:.2f}) | "
              f"evalAUC={v['eval_auc']*100:.2f} (PADvsREAL {v['eval_auc_padonly']*100:.2f})")
        for op, d in v["ops"].items():
            tu, ev = d["tune"], d["eval"]
            print(f"     tau@{op}={d['tau']:.4f}  "
                  f"TUNE pad={fmt(tu['pad'])} real={fmt(tu['real'])} df={fmt(tu['deepfake'])} wpad={fmt(tu['worst_pad'])}({tu['worst_name']})  ||  "
                  f"EVAL pad={fmt(ev['pad'])} real={fmt(ev['real'])} df={fmt(ev['deepfake'])} wpad={fmt(ev['worst_pad'])}({ev['worst_name']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True,
                    help="name:tune_dump_dir:eval_dump_dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    tune_man = json.load(open(f"{TUNE}/manifest.json"))
    eval_man = json.load(open(f"{HELD}/manifest.json"))
    tune_mask = json.load(open(f"{TUNE}/tune_videos.json"))
    allres = []
    for spec in args.models:
        name, tdir, edir = spec.split(":")
        r = analyze_model(name, tdir, edir, tune_man, eval_man, tune_mask)
        allres.append(r); print_model(r)
    if args.out:
        json.dump(allres, open(args.out, "w"), indent=2)
        print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
