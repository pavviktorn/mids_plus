#!/usr/bin/env bash
# Rebuild the REAL subsets with the face-quality filter (delete old real first), then re-eval the
# 5 models on the NEW real subsets only (--full-image). PAD/DEEPFAKE results are reused unchanged.
# Builds a fresh combined report (new real + existing pad/deepfake).
cd /datasets/work/vLLM/temp/mids_plus || exit 1
. .venv/bin/activate
export PYTHONPATH=src TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
H=runs/real/heldout3
mkdir -p "$H/results_realf"
LOG="$H/run_realf.log"
say(){ echo "$(date '+%F %T') | $*" | tee -a "$LOG"; }

say "=== rebuilding REAL subsets with face-quality filter (deletes old real) ==="
python -u scripts/build_real_filtered.py >> "$H/build_realf.log" 2>&1
say "build exit=$? ; $(grep -E 'REAL total' "$H/build_realf.log" | tail -1)"

# new real bucket specs
SPECS=""
for f in "$H"/buckets/real_*.json; do n=$(basename "$f" .json); SPECS="$SPECS ${n}:$f"; done
say "real subsets: $(ls "$H"/buckets/real_*.json | wc -l)"

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
  local common="--full-image --data $SPECS --batch-size 64 --num-workers 8 --tag $tag --out $H/results_realf/${tag}.json"
  if [ "$type" = upstream ]; then
    CUDA_VISIBLE_DEVICES=$gpu python -u scripts/eval_fixed_answers.py --model-type upstream --upstream-mids "$arg" $common >> "$H/eval_realf_${tag}.log" 2>&1
  else
    CUDA_VISIBLE_DEVICES=$gpu python -u scripts/eval_fixed_answers.py --model-type mids_plus --checkpoint "$arg" $common >> "$H/eval_realf_${tag}.log" 2>&1
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
say "=== evals done; building combined report ==="

python -u - <<'PY' > "$H/GENERALIZATION3_FILTERED_REPORT.md" 2>>"$H/build_realf.err"
import json, os, math
H="runs/real/heldout3"
man=json.load(open(f"{H}/manifest.json"))
MODELS=[("A0 baseline","A0_baseline"),("A1 +SVD","A1_svd"),("A2 +SVD+GenD","A2_svdgend"),
        ("A3 MIDS++","A3_midspp"),("upstream mids.pth","upstream")]
def loadj(p): return json.load(open(p))["per_set"] if os.path.exists(p) else {}
# real from NEW filtered eval; pad/deepfake from existing full eval
R={}
for nm,t in MODELS:
    realps=loadj(f"{H}/results_realf/{t}.json")
    fullps=loadj(f"{H}/results/{t}.json")
    merged=dict(realps)  # real_* keys
    for k,v in fullps.items():
        if man.get(k,{}).get("group") in ("PAD","DEEPFAKE"): merged[k]=v
    R[nm]=merged
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
GRPS=["REAL","PAD","DEEPFAKE"]; gsubs={g:[s for s in subs if man[s]["group"]==g] for g in GRPS}
tot={g:sum(n(R["A3 MIDS++"],s) for s in gsubs[g]) for g in GRPS}
print("# MIDS++_fixed — 30K full-image unseen test, REAL set FILTERED to frontal/proper-size faces\n")
print(f"REAL={tot['REAL']} (FILTERED: frontal, proper face size, no heavy pose/occlusion; insightface "
      f"SCRFD+pose) | PAD={tot['PAD']} | DEEPFAKE={tot['DEEPFAKE']}. Full image, no crop. "
      "Per-subset single-label => ACC = recall.\n")
print("| Model | REAL-rec | PAD-rec | DEEPFAKE-rec | FAKE-rec | Balanced ACC | sACC(46) |")
print("|---|---|---|---|---|---|---|")
for nm,_ in MODELS:
    ps=R[nm]; rr,pr,dr,fr=grec(ps,"REAL"),grec(ps,"PAD"),grec(ps,"DEEPFAKE"),frec(ps); bal=(rr+fr)/2
    print(f"| {nm} | {rr:.2f} | {pr:.2f} | {dr:.2f} | {fr:.2f} | {bal:.2f} | {sacc(ps,subs):.2f} |")
print("\n## REAL sACC (across the filtered real subsets)\n")
print("| Model | REAL sACC | per-id range |")
print("|---|---|---|")
for nm,_ in MODELS:
    ps=R[nm]; ids=[s for s in gsubs['REAL'] if s!='real_photo']
    vals=[a(ps,s) for s in ids if s in ps]
    print(f"| {nm} | {sacc(ps,gsubs['REAL']):.2f} | {min(vals):.1f}–{max(vals):.1f} |")
print("\n## REAL subsets — per-subset recall (FILTERED)\n")
print("| subset | n | " + " | ".join(nm for nm,_ in MODELS) + " |")
print("|" + " --- |"*(len(MODELS)+2))
for s in sorted(gsubs['REAL'], key=lambda s:-n(R['A3 MIDS++'],s)):
    print(f"| {s} | {n(R['A3 MIDS++'],s)} | " + " | ".join(f"{a(R[nm],s):.2f}" if s in R[nm] else "-" for nm,_ in MODELS) + " |")
print("\n(PAD/DEEPFAKE recall unchanged from the unfiltered run — only REAL was rebuilt.)")
PY
cat "$H/GENERALIZATION3_FILTERED_REPORT.md" >> "$LOG"
say "=== REALF DONE ==="
touch "$H/DONE_REALF"
