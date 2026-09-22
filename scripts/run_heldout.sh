#!/usr/bin/env bash
# Definitive fixed-answer (MLLM-free) GENERALIZATION eval on the genuinely-unseen held-out set
# (eFFAA_ext_eval, 0% in training). All 5 full models: A0 baseline, A1 +SVD, A2 +SVD+GenD,
# A3 MIDS++, upstream mids.pth. Per-family + ALL/NOVEL/KNOWN aggregates. Distributed over 3 GPU queues.
cd /datasets/work/vLLM/temp/mids_plus || exit 1
. .venv/bin/activate
export PYTHONPATH=src TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

H=runs/real/heldout
mkdir -p "$H/results"
LOG="$H/run.log"
say(){ echo "$(date '+%F %T') | $*" | tee -a "$LOG"; }

# specs: aggregates first, then per-family buckets
SPECS="ALL:$H/heldout_ALL.json NOVEL:$H/heldout_NOVEL.json KNOWN:$H/heldout_KNOWN.json"
for f in "$H"/buckets/*.json; do n=$(basename "$f" .json); SPECS="$SPECS ${n}:$f"; done

# 3 freest GPUs
mapfile -t G < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -3 | cut -d, -f1 | tr -d ' ')
say "=== held-out generalization eval; GPUs: ${G[*]} ==="

# model table: tag|type|arg
MODELS=(
 "A0_baseline|mids_plus|runs/real/mids_original_full/best.pt"
 "A1_svd|mids_plus|runs/real/abl/svd_only/best.pt"
 "A2_svdgend|mids_plus|runs/real/abl/svd_gend/best.pt"
 "A3_midspp|mids_plus|runs/real/mids_pp_full/best.pt"
 "upstream|upstream|checkpoints/mids_upstream.pth"
)

run_one(){
  local gpu=$1 tag=$2 type=$3 arg=$4
  if [ "$type" = upstream ]; then
    CUDA_VISIBLE_DEVICES=$gpu python -u scripts/eval_fixed_answers.py --model-type upstream \
      --upstream-mids "$arg" --data $SPECS --batch-size 64 --num-workers 8 \
      --tag "$tag" --out "$H/results/${tag}.json" >> "$H/eval_${tag}.log" 2>&1
  else
    CUDA_VISIBLE_DEVICES=$gpu python -u scripts/eval_fixed_answers.py --model-type mids_plus \
      --checkpoint "$arg" --data $SPECS --batch-size 64 --num-workers 8 \
      --tag "$tag" --out "$H/results/${tag}.json" >> "$H/eval_${tag}.log" 2>&1
  fi
  say "  done $tag (gpu $gpu) exit=$?"
}

# 3 GPU queues, round-robin models -> queue
declare -a Q0 Q1 Q2
for i in "${!MODELS[@]}"; do
  case $((i % 3)) in 0) Q0+=("${MODELS[$i]}");; 1) Q1+=("${MODELS[$i]}");; 2) Q2+=("${MODELS[$i]}");; esac
done
run_queue(){ local gpu=$1; shift; for m in "$@"; do IFS='|' read -r tag type arg <<< "$m"; say "start $tag on gpu $gpu"; run_one "$gpu" "$tag" "$type" "$arg"; done; }
run_queue "${G[0]}" "${Q0[@]}" &
P0=$!
run_queue "${G[1]:-${G[0]}}" "${Q1[@]}" &
P1=$!
run_queue "${G[2]:-${G[0]}}" "${Q2[@]}" &
P2=$!
wait $P0 $P1 $P2
say "=== all evals done; building report ==="

python -u - <<'PY' > "$H/GENERALIZATION_REPORT.md" 2>>"$H/build.err"
import json, os, math
H="runs/real/heldout/results"
def load(t):
    p=os.path.join(H,f"{t}.json"); return json.load(open(p)) if os.path.exists(p) else None
MODELS=[("A0 baseline","A0_baseline"),("A1 +SVD","A1_svd"),("A2 +SVD+GenD","A2_svdgend"),
        ("A3 MIDS++","A3_midspp"),("upstream mids.pth","upstream")]
R={name:load(tag) for name,tag in MODELS}
def per(d): return d["per_set"] if d else {}
def acc(d,k): ps=per(d); return ps[k]["acc"]*100 if k in ps else float('nan')
def auc(d,k): ps=per(d); return ps[k]["auc"]*100 if k in ps else float('nan')
FAMS=[k for k in per(R["A3 MIDS++"]) if k not in ("ALL","NOVEL","KNOWN")]
def sacc(d,keys):
    a=[per(d)[k]["acc"] for k in keys if k in per(d)]
    if not a: return float('nan')
    m=sum(a)/len(a); return math.sqrt(sum((x-m)**2 for x in a)/len(a))*100

print("# MIDS++_fixed — TRUE generalization (genuinely-unseen held-out set, 0% in training)\n")
print("Set: eFFAA_ext_eval (26,773 imgs, none in mids.json). Fixed-answer (MLLM-free) decision. "
      "NOVEL = forgery sources absent from training (DFDC, 3D-Attacks, SiW-Mv2, FF++, DiscoGAN, "
      "webcam, ...); KNOWN = training families but new files. sACC over per-family ACC (lower=better).\n")
print("| Model | ALL ACC | NOVEL ACC | KNOWN ACC | ALL AUC | family sACC |")
print("|---|---|---|---|---|---|")
for name,_ in MODELS:
    d=R[name]
    if not d: print(f"| {name} | (missing) |"); continue
    print(f"| {name} | {acc(d,'ALL'):.2f} | {acc(d,'NOVEL'):.2f} | {acc(d,'KNOWN'):.2f} | {auc(d,'ALL'):.2f} | {sacc(d,FAMS):.2f} |")

print("\n## Per-family ACC (fixed answers, unseen files)\n")
order=sorted(FAMS, key=lambda k:-per(R["A3 MIDS++"]).get(k,{}).get("n",0))
print("| family | n | " + " | ".join(n for n,_ in MODELS) + " |")
print("|" + " --- |"*(len(MODELS)+2))
for k in order:
    n=per(R["A3 MIDS++"]).get(k,{}).get("n",0)
    print(f"| {k} | {n} | " + " | ".join(f"{acc(R[nm],k):.2f}" if k in per(R[nm]) else "-" for nm,_ in MODELS) + " |")
print("\n(In-distribution reference: on the *training-overlap* mids_eval buckets these models scored "
      "~99.5–99.9 ACC / sACC 0.18–0.98. The gap to the NOVEL column above is the real generalization cost.)")
PY
cat "$H/GENERALIZATION_REPORT.md" >> "$LOG"
say "=== HELDOUT DONE ==="
touch "$H/DONE_HELDOUT"
