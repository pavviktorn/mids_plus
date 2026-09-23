#!/usr/bin/env bash
# Build the 30K / 46-subset FULL-IMAGE unseen test set, then fixed-answer (--full-image) eval of all
# 5 models on every subset (each image scored once). Aggregates REAL/PAD/DEEPFAKE recall + per-subset
# sACC in the report. Models distributed over 3 GPU queues.
cd /datasets/work/vLLM/temp/mids_plus || exit 1
. .venv/bin/activate
export PYTHONPATH=src TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
H=runs/real/heldout3
mkdir -p "$H/results"
LOG="$H/run.log"
say(){ echo "$(date '+%F %T') | $*" | tee -a "$LOG"; }

if [ ! -f "$H/heldout3_ALL.json" ]; then
  say "=== building 30K/46-subset test set (parallel frame extraction) ==="
  python -u scripts/build_heldout3.py >> "$H/build.log" 2>&1
  say "build exit=$? ; $(grep -E 'GROUP TOTALS|^ALL =' "$H/build.log" | tail -2 | tr '\n' ' ')"
else
  say "=== test set already built, skipping ==="
fi

# specs = all 46 buckets
SPECS=""
for f in "$H"/buckets/*.json; do n=$(basename "$f" .json); SPECS="$SPECS ${n}:$f"; done
NSUB=$(ls "$H"/buckets/*.json | wc -l)
say "=== eval over $NSUB subsets ==="

mapfile -t G < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -3 | cut -d, -f1 | tr -d ' ')
say "GPUs: ${G[*]}"
MODELS=(
 "A0_baseline|mids_plus|runs/real/mids_original_full/best.pt"
 "A1_svd|mids_plus|runs/real/abl/svd_only/best.pt"
 "A2_svdgend|mids_plus|runs/real/abl/svd_gend/best.pt"
 "A3_midspp|mids_plus|runs/real/mids_pp_full/best.pt"
 "upstream|upstream|checkpoints/mids_upstream.pth"
)
run_one(){
  local gpu=$1 tag=$2 type=$3 arg=$4
  local common="--full-image --data $SPECS --batch-size 64 --num-workers 8 --tag $tag --out $H/results/${tag}.json"
  if [ "$type" = upstream ]; then
    CUDA_VISIBLE_DEVICES=$gpu python -u scripts/eval_fixed_answers.py --model-type upstream --upstream-mids "$arg" $common >> "$H/eval_${tag}.log" 2>&1
  else
    CUDA_VISIBLE_DEVICES=$gpu python -u scripts/eval_fixed_answers.py --model-type mids_plus --checkpoint "$arg" $common >> "$H/eval_${tag}.log" 2>&1
  fi
  say "  done $tag (gpu $gpu) exit=$?"
}
declare -a Q0 Q1 Q2
for i in "${!MODELS[@]}"; do case $((i%3)) in 0) Q0+=("${MODELS[$i]}");; 1) Q1+=("${MODELS[$i]}");; 2) Q2+=("${MODELS[$i]}");; esac; done
run_queue(){ local gpu=$1; shift; for m in "$@"; do IFS='|' read -r tag type arg <<< "$m"; say "start $tag on gpu $gpu"; run_one "$gpu" "$tag" "$type" "$arg"; done; }
run_queue "${G[0]}" "${Q0[@]}" & P0=$!
run_queue "${G[1]:-${G[0]}}" "${Q1[@]}" & P1=$!
run_queue "${G[2]:-${G[0]}}" "${Q2[@]}" & P2=$!
wait $P0 $P1 $P2
say "=== all evals done; building report ==="

python -u - <<'PY' > "$H/GENERALIZATION3_REPORT.md" 2>>"$H/build.err"
import json, os, math
H="runs/real/heldout3"; RES=f"{H}/results"
man=json.load(open(f"{H}/manifest.json"))
MODELS=[("A0 baseline","A0_baseline"),("A1 +SVD","A1_svd"),("A2 +SVD+GenD","A2_svdgend"),
        ("A3 MIDS++","A3_midspp"),("upstream mids.pth","upstream")]
def load(t): p=f"{RES}/{t}.json"; return json.load(open(p))["per_set"] if os.path.exists(p) else {}
R={n:load(t) for n,t in MODELS}
subs=sorted(man); NAN=float('nan')
def a(ps,k): return ps[k]["acc"]*100 if k in ps else NAN
def n(ps,k): return ps[k]["n"] if k in ps else 0
def grec(ps,g):
    ks=[s for s in man if man[s]["group"]==g]; tot=sum(n(ps,k) for k in ks)
    cor=sum(ps[k]["acc"]*ps[k]["n"] for k in ks if k in ps); return 100*cor/tot if tot else NAN
def frec(ps):
    ks=[s for s in man if man[s]["group"] in ("PAD","DEEPFAKE")]; tot=sum(n(ps,k) for k in ks)
    cor=sum(ps[k]["acc"]*ps[k]["n"] for k in ks if k in ps); return 100*cor/tot if tot else NAN
def sacc(ps,ks):
    v=[ps[k]["acc"] for k in ks if k in ps]
    if not v: return NAN
    m=sum(v)/len(v); return math.sqrt(sum((x-m)**2 for x in v)/len(v))*100
GRPS=["REAL","PAD","DEEPFAKE"]
gsubs={g:[s for s in subs if man[s]["group"]==g] for g in GRPS}

print("# MIDS++_fixed — 30K full-image unseen test (46 subsets), MLLM-free\n")
tot={g:sum(n(R["A3 MIDS++"],s) for s in gsubs[g]) for g in GRPS}
print(f"REAL={tot['REAL']}  PAD={tot['PAD']}  DEEPFAKE={tot['DEEPFAKE']}  (subsets: "
      f"REAL {len(gsubs['REAL'])}, PAD {len(gsubs['PAD'])}, DEEPFAKE {len(gsubs['DEEPFAKE'])}). "
      "Full image, no face/center crop. Per-subset single-label => ACC = that class's recall.\n")
print("| Model | REAL-rec | PAD-rec | DEEPFAKE-rec | FAKE-rec | Balanced ACC | sACC(46) |")
print("|---|---|---|---|---|---|---|")
for nm,_ in MODELS:
    ps=R[nm]; rr,pr,dr,fr=grec(ps,"REAL"),grec(ps,"PAD"),grec(ps,"DEEPFAKE"),frec(ps)
    bal=(rr+fr)/2
    print(f"| {nm} | {rr:.2f} | {pr:.2f} | {dr:.2f} | {fr:.2f} | {bal:.2f} | {sacc(ps,subs):.2f} |")

print("\n## sACC within each group (robustness across subsets; lower=better)\n")
print("| Model | REAL sACC | PAD sACC | DEEPFAKE sACC |")
print("|---|---|---|---|")
for nm,_ in MODELS:
    ps=R[nm]; print(f"| {nm} | {sacc(ps,gsubs['REAL']):.2f} | {sacc(ps,gsubs['PAD']):.2f} | {sacc(ps,gsubs['DEEPFAKE']):.2f} |")

for g in GRPS:
    print(f"\n## {g} subsets — per-subset recall\n")
    print("| subset | n | " + " | ".join(nm for nm,_ in MODELS) + " |")
    print("|" + " --- |"*(len(MODELS)+2))
    for s in sorted(gsubs[g], key=lambda s:-n(R["A3 MIDS++"],s)):
        print(f"| {s} | {n(R['A3 MIDS++'],s)} | " + " | ".join(f"{a(R[nm],s):.2f}" if s in R[nm] else "-" for nm,_ in MODELS) + " |")
PY
cat "$H/GENERALIZATION3_REPORT.md" >> "$LOG"
say "=== HELDOUT3 DONE ==="
touch "$H/DONE_HELDOUT3"
