#!/usr/bin/env bash
# Build the balanced FULL-IMAGE unseen test set (axonlabs real+PAD videos, photo-reals, gasstation
# deepfakes), then fixed-answer (MLLM-free) eval of all 5 models with --full-image (no face/center
# crop). Each source is single-label so per-source ACC = that class's recall.
cd /datasets/work/vLLM/temp/mids_plus || exit 1
. .venv/bin/activate
export PYTHONPATH=src TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

H=runs/real/heldout2
mkdir -p "$H/results"
LOG="$H/run.log"
say(){ echo "$(date '+%F %T') | $*" | tee -a "$LOG"; }

if [ ! -f "$H/heldout2_ALL.json" ]; then
  say "=== building test set (frame extraction) ==="
  python -u scripts/build_heldout2.py >> "$H/build.log" 2>&1
  say "build exit=$? ; $(grep -E '^(ALL|REAL|FAKE) ' "$H/build.log" | tail -3 | tr '\n' ' ')"
else
  say "=== test set already built, skipping extraction ==="
fi

SPECS="ALL:$H/heldout2_ALL.json REAL:$H/heldout2_REAL.json FAKE:$H/heldout2_FAKE.json"
for f in "$H"/buckets/*.json; do n=$(basename "$f" .json); SPECS="$SPECS ${n}:$f"; done

GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')
say "=== eval on GPU $GPU (full-image) ==="
MODELS=(
 "A0_baseline|mids_plus|runs/real/mids_original_full/best.pt"
 "A1_svd|mids_plus|runs/real/abl/svd_only/best.pt"
 "A2_svdgend|mids_plus|runs/real/abl/svd_gend/best.pt"
 "A3_midspp|mids_plus|runs/real/mids_pp_full/best.pt"
 "upstream|upstream|checkpoints/mids_upstream.pth"
)
for m in "${MODELS[@]}"; do
  IFS='|' read -r tag type arg <<< "$m"
  say "eval $tag"
  if [ "$type" = upstream ]; then
    CUDA_VISIBLE_DEVICES=$GPU python -u scripts/eval_fixed_answers.py --model-type upstream \
      --upstream-mids "$arg" --full-image --data $SPECS --batch-size 64 --num-workers 8 \
      --tag "$tag" --out "$H/results/${tag}.json" >> "$H/eval_${tag}.log" 2>&1
  else
    CUDA_VISIBLE_DEVICES=$GPU python -u scripts/eval_fixed_answers.py --model-type mids_plus \
      --checkpoint "$arg" --full-image --data $SPECS --batch-size 64 --num-workers 8 \
      --tag "$tag" --out "$H/results/${tag}.json" >> "$H/eval_${tag}.log" 2>&1
  fi
  say "  $tag exit=$?"
done

say "=== building report ==="
python -u - <<'PY' > "$H/GENERALIZATION2_REPORT.md" 2>>"$H/build.err"
import json, os
H="runs/real/heldout2/results"
def load(t): p=os.path.join(H,f"{t}.json"); return json.load(open(p)) if os.path.exists(p) else None
MODELS=[("A0 baseline","A0_baseline"),("A1 +SVD","A1_svd"),("A2 +SVD+GenD","A2_svdgend"),
        ("A3 MIDS++","A3_midspp"),("upstream mids.pth","upstream")]
R={n:load(t) for n,t in MODELS}
def per(d): return d["per_set"] if d else {}
def acc(d,k): ps=per(d); return ps[k]["acc"]*100 if k in ps else float('nan')
def auc(d,k): ps=per(d); return ps[k]["auc"]*100 if k in ps else float('nan')
def n(d,k): ps=per(d); return ps[k]["n"] if k in ps else 0
ref=R["A3 MIDS++"]
print("# MIDS++_fixed — FULL-IMAGE generalization on unseen user data (no face crop)\n")
print("Balanced test set, 0% in training: REAL = axonlabs real videos + photo-reals; "
      "FAKE = axonlabs PAD videos + gasstation deepfakes. Images fed FULL (letterbox, no face/center "
      "crop). Each source is single-label, so per-source ACC = that class's recall.\n")
print("| Model | ALL ACC | ALL AUC | REAL-recall | FAKE-recall |")
print("|---|---|---|---|---|")
for nm,_ in MODELS:
    d=R[nm]
    if not d: print(f"| {nm} | missing |"); continue
    print(f"| {nm} | {acc(d,'ALL'):.2f} | {auc(d,'ALL'):.2f} | {acc(d,'REAL'):.2f} | {acc(d,'FAKE'):.2f} |")
SRCS=[("real_video","REAL"),("real_photo","REAL"),("pad_video","FAKE"),("deepfake_img","FAKE")]
print("\n## Per-source recall (ACC; single-label sources)\n")
print("| source | label | n | " + " | ".join(nm for nm,_ in MODELS) + " |")
print("|" + " --- |"*(len(MODELS)+3))
for s,lab in SRCS:
    print(f"| {s} | {lab} | {n(ref,s)} | " + " | ".join(f"{acc(R[nm],s):.2f}" if s in per(R[nm]) else "-" for nm,_ in MODELS) + " |")
PY
cat "$H/GENERALIZATION2_REPORT.md" >> "$LOG"
say "=== HELDOUT2 DONE ==="
touch "$H/DONE_HELDOUT2"
