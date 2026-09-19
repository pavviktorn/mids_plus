#!/usr/bin/env bash
# Dump per-sample softmax for each candidate model on BOTH the tune set and the heldout eval set.
# Round-robins models over the given GPU list; models on the same GPU run sequentially.
# Resumable: dump_scores.py skips any bucket .npz that already exists; per-model DONE markers too.
#   usage: run_dumps.sh "<gpu csv>" [tta]      e.g.  run_dumps.sh "0,1,2,3" none
cd /datasets/work/vLLM/temp/mids_plus || exit 1
. .venv/bin/activate
export PYTHONPATH=src TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GPUS_CSV="${1:-0,1,2,3}"; TTA="${2:-none}"
IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
SUF=""; [ "$TTA" != "none" ] && SUF="_$TTA"
TDUMP=runs/real/tune_dumps; EDUMP=runs/real/eval_dumps
mkdir -p "$TDUMP" "$EDUMP" runs/real/dumplogs
LOG=runs/real/dumplogs/dumps$SUF.log
say(){ echo "$(date '+%F %T') | $*" | tee -a "$LOG"; }

# model tag = checkpoint
declare -a TAGS=(A1_9c A3_9c A2_9c A0_9c A3_4c A1_4c)
declare -A CKPT=(
  [A1_9c]=runs/real/9class_resume/A1_svd/best.pt
  [A3_9c]=runs/real/9class_resume/A3_midspp/best.pt
  [A2_9c]=runs/real/9class_resume/A2_svdgend/best.pt
  [A0_9c]=runs/real/9class_resume/A0_baseline/best.pt
  [A3_4c]=runs/real/mids_pp_full/best.pt
  [A1_4c]=runs/real/abl/svd_only/best.pt
)

TUNE_SPECS=""; for f in runs/real/tune/buckets/*.json;     do TUNE_SPECS="$TUNE_SPECS $(basename "$f" .json):$f"; done
EVAL_SPECS="";  for f in runs/real/heldout3/buckets/*.json; do EVAL_SPECS="$EVAL_SPECS $(basename "$f" .json):$f"; done

dump_model(){
  local tag="$1" gpu="$2" ck="${CKPT[$1]}"
  local td="$TDUMP/${tag}${SUF}" ed="$EDUMP/${tag}${SUF}"
  if [ -f "$ed/DONE" ]; then say "$tag already DONE (gpu$gpu), skip"; return; fi
  say "START $tag on gpu$gpu  ($ck)  tta=$TTA"
  CUDA_VISIBLE_DEVICES=$gpu python -u scripts/dump_scores.py --model-type mids_plus --checkpoint "$ck" \
    --full-image --tta "$TTA" --batch-size 64 --num-workers 8 --out "$td" --data $TUNE_SPECS \
    >> runs/real/dumplogs/${tag}${SUF}.log 2>&1
  say "  $tag tune dump exit=$?"
  CUDA_VISIBLE_DEVICES=$gpu python -u scripts/dump_scores.py --model-type mids_plus --checkpoint "$ck" \
    --full-image --tta "$TTA" --batch-size 64 --num-workers 8 --out "$ed" --data $EVAL_SPECS \
    >> runs/real/dumplogs/${tag}${SUF}.log 2>&1
  say "  $tag eval dump exit=$?"
  touch "$ed/DONE"
}

# launch one queue per GPU (round-robin assignment)
declare -A QUEUE
for idx in "${!TAGS[@]}"; do
  g=${GPUS[$(( idx % ${#GPUS[@]} ))]}
  QUEUE[$g]="${QUEUE[$g]} ${TAGS[$idx]}"
done
PIDS=()
for g in "${!QUEUE[@]}"; do
  ( for tag in ${QUEUE[$g]}; do dump_model "$tag" "$g"; done ) &
  PIDS+=($!)
  say "queue gpu$g: ${QUEUE[$g]}"
done
for p in "${PIDS[@]}"; do wait "$p"; done
say "=== ALL DUMPS DONE (tta=$TTA) ==="
touch "$EDUMP/ALL_DONE$SUF"
