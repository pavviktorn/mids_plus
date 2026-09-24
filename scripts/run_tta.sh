#!/usr/bin/env bash
# Crop-free TTA (flip) dumps for a chosen model list, round-robined over a GPU list. tune+eval.
# usage: run_tta.sh "<tags csv>" "<gpus csv>" <tta>   e.g. run_tta.sh "A1_9c,A2_9c,A3_9c" "2,3" flip
cd /datasets/work/vLLM/temp/mids_plus || exit 1
. .venv/bin/activate
export PYTHONPATH=src TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
IFS=',' read -r -a TAGS <<< "${1:-A1_9c,A2_9c,A3_9c}"
IFS=',' read -r -a GPUS <<< "${2:-2,3}"
TTA="${3:-flip}"
mkdir -p runs/real/dumplogs
LOG=runs/real/dumplogs/tta_${TTA}.log
say(){ echo "$(date '+%F %T') | $*" | tee -a "$LOG"; }
declare -A CKPT=(
  [A1_9c]=runs/real/9class_resume/A1_svd/best.pt
  [A2_9c]=runs/real/9class_resume/A2_svdgend/best.pt
  [A3_9c]=runs/real/9class_resume/A3_midspp/best.pt
  [A0_9c]=runs/real/9class_resume/A0_baseline/best.pt
  [A3_4c]=runs/real/mids_pp_full/best.pt
  [A1_4c]=runs/real/abl/svd_only/best.pt
)
TUNE_SPECS=""; for f in runs/real/tune/buckets/*.json;     do TUNE_SPECS="$TUNE_SPECS $(basename "$f" .json):$f"; done
EVAL_SPECS="";  for f in runs/real/heldout3/buckets/*.json; do EVAL_SPECS="$EVAL_SPECS $(basename "$f" .json):$f"; done
dump(){
  local tag="$1" gpu="$2"; local td="runs/real/tune_dumps/${tag}_${TTA}" ed="runs/real/eval_dumps/${tag}_${TTA}"
  [ -f "$ed/DONE" ] && { say "$tag _$TTA already DONE, skip"; return; }
  say "START $tag _$TTA on gpu$gpu"
  CUDA_VISIBLE_DEVICES=$gpu python -u scripts/dump_scores.py --model-type mids_plus --checkpoint "${CKPT[$tag]}" \
    --full-image --tta "$TTA" --batch-size 64 --num-workers 8 --out "$td" --data $TUNE_SPECS >> "runs/real/dumplogs/${tag}_${TTA}.log" 2>&1
  say "  $tag _$TTA tune exit=$?"
  CUDA_VISIBLE_DEVICES=$gpu python -u scripts/dump_scores.py --model-type mids_plus --checkpoint "${CKPT[$tag]}" \
    --full-image --tta "$TTA" --batch-size 64 --num-workers 8 --out "$ed" --data $EVAL_SPECS >> "runs/real/dumplogs/${tag}_${TTA}.log" 2>&1
  say "  $tag _$TTA eval exit=$?"; touch "$ed/DONE"
}
declare -A Q
for idx in "${!TAGS[@]}"; do g=${GPUS[$(( idx % ${#GPUS[@]} ))]}; Q[$g]="${Q[$g]} ${TAGS[$idx]}"; done
PIDS=()
for g in "${!Q[@]}"; do ( for t in ${Q[$g]}; do dump "$t" "$g"; done ) & PIDS+=($!); say "queue gpu$g:${Q[$g]}"; done
for p in "${PIDS[@]}"; do wait "$p"; done
say "=== TTA($TTA) DUMPS DONE ==="; touch "runs/real/eval_dumps/ALL_DONE_${TTA}"
