#!/usr/bin/env python
"""Assemble the comprehensive 3-mode x 5-model comparison into markdown tables.

Mode 1 MLLM-referred (4-class, MLLM answers)  -- only on in-distribution mids_eval buckets.
Mode 2 MLLM-free 4-class (fixed answers)      -- in-distribution + heldout3.
Mode 3 MLLM-free 9-class (fixed answers)      -- in-distribution (A0-A3) + heldout3 (A0-A3).
"""
import json, os, math

def load(p):
    return json.load(open(p))["per_set"] if os.path.exists(p) else {}
def acc(ps, k):
    return ps[k]["acc"]*100 if k in ps and ps[k].get("acc") is not None else float('nan')
def cell(ps, k):
    v = acc(ps, k); return f"{v:.1f}" if v==v else "-"

MODELS = ["A0","A1","A2","A3","upstream"]

# ---------------- IN-DISTRIBUTION (leaked mids_eval) ----------------
ID_SUBSETS = ["eval2000","DeepLiveCam","WFFD","RealWebcam","ReplayMobile","CelebDF",
              "CeFA","DFMNIST","DeepFakeFace","DeeperForensics","ReplayAttack"]
EXP="runs/real/exp/results"; ABL="runs/real/abl/results"
FA="runs/real/exp_fixedans_all/results"; FA0="runs/real/exp_fixedans/results"
NID="runs/real/9class_resume/results9_indist"
# mode1 MLLM-referred (4-class)
m1 = {
 "A0": load(f"{EXP}/mids_original_full_all.json"),
 "A1": load(f"{ABL}/svd_only_full.json"),
 "A2": load(f"{ABL}/svd_gend_full.json"),
 "A3": load(f"{EXP}/mids_pp_full_all.json"),
 "upstream": {**load(f"{EXP}/upstream_eval2000.json"), **load(f"{EXP}/upstream_buckets.json")},
}
# mode2 MLLM-free 4-class
m2 = {
 "A0": load(f"{FA}/A0_mids_original_full_fixed.json"),
 "A1": load(f"{FA}/A1_svd_only_fixed.json"),
 "A2": load(f"{FA}/A2_svd_gend_fixed.json"),
 "A3": load(f"{FA}/A3_mids_pp_full_fixed.json"),
 "upstream": load(f"{FA0}/upstream_fixed.json"),
}
# mode3 MLLM-free 9-class
m3 = {
 "A0": load(f"{NID}/A0_baseline.json"),
 "A1": load(f"{NID}/A1_svd.json"),
 "A2": load(f"{NID}/A2_svdgend.json"),
 "A3": load(f"{NID}/A3_midspp.json"),
 "upstream": {},
}

def table(modemap, subsets, title):
    print(f"\n### {title}\n")
    print("| subset | " + " | ".join(MODELS) + " |")
    print("|" + " --- |"*(len(MODELS)+1))
    for s in subsets:
        print(f"| {s} | " + " | ".join(cell(modemap[m], s) for m in MODELS) + " |")

print("## A. IN-DISTRIBUTION (mids_eval buckets — 100% in training; per-subset ACC)\n")
print("_Sanity/мode-comparison only; these images were in training. eval2000 = hard subset._")
table(m1, ID_SUBSETS, "A1 · MLLM-referred (4-class, MLLM answers)")
table(m2, ID_SUBSETS, "A2 · MLLM-free (4-class, fixed answers)")
table(m3, ID_SUBSETS, "A3 · MLLM-free (9-class, fixed answers)  [upstream has no 9-class]")

# ---------------- GENERALIZATION (heldout3, full-image) ----------------
H="runs/real/heldout3"; NINE="runs/real/9class_resume/results9"
man=json.load(open(f"{H}/manifest.json"))
def merged4(tag):
    real=load(f"{H}/results_realf/{tag}.json"); full=load(f"{H}/results/{tag}.json")
    m=dict(real)
    for k,v in full.items():
        if man.get(k,{}).get("group") in ("PAD","DEEPFAKE"): m[k]=v
    return m
TAG={"A0":"A0_baseline","A1":"A1_svd","A2":"A2_svdgend","A3":"A3_midspp","upstream":"upstream"}
g4={m:merged4(TAG[m]) for m in MODELS}
g9={m:load(f"{NINE}/{TAG[m]}.json") for m in MODELS}  # upstream -> {} (no 9-class)
import re as _re
def n(ps,k): return ps[k]["n"] if k in ps else 0
def grec(ps,grp):
    ks=[s for s in man if man[s]["group"]==grp]; tot=sum(n(ps,k) for k in ks)
    cor=sum(ps[k]["acc"]*ps[k]["n"] for k in ks if k in ps); return 100*cor/tot if tot else float('nan')
def gcell(ps,grp):
    v=grec(ps,grp); return f"{v:.1f}" if v==v else "-"
def padlabel(s): return _re.sub(r'(_\d+[a-z]*)+$', '', s.replace('pad_',''))
padtypes=sorted([s for s in man if man[s]["group"]=="PAD"], key=lambda s:-n(g4["A3"],s))

def gen_table(gmap, title):
    print(f"\n### {title}\n")
    print("| subset (recall) | " + " | ".join(MODELS) + " |")
    print("|" + " --- |"*(len(MODELS)+1))
    print(f"| REAL (real-recall) | " + " | ".join(gcell(gmap[m],"REAL") for m in MODELS) + " |")
    print(f"| PAD (anti-spoof recall) | " + " | ".join(gcell(gmap[m],"PAD") for m in MODELS) + " |")
    print(f"| DEEPFAKE (recall) | " + " | ".join(gcell(gmap[m],"DEEPFAKE") for m in MODELS) + " |")
    for s in padtypes:
        print(f"|   · PAD/{padlabel(s)} | " + " | ".join(cell(gmap[m], s) for m in MODELS) + " |")

print("\n\n## B. TRUE GENERALIZATION (heldout3 30K, full-image, MLLM-free; per-group recall)\n")
print("_MLLM-referred N/A here (the unseen axonlabs/gasstation data has no MLLM answers). "
      "upstream has no 9-class variant._")
gen_table(g4, "B1 · MLLM-free 4-class")
gen_table(g9, "B2 · MLLM-free 9-class")
