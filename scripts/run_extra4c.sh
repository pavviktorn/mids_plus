#!/usr/bin/env bash
# Add the two missing 4-class members (A0_4c, A2_4c) so the fusion search covers all four 4-class
# models, matching the four 9-class. tune+eval, full-image, no TTA. Resumable (skip existing npz).
cd /datasets/work/vLLM/temp/mids_plus || exit 1
. .venv/bin/activate
export PYTHONPATH=src TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p runs/real/dumplogs
LOG=runs/real/dumplogs/extra4c.log
say(){ echo "$(date '+%F %T') | $*" | tee -a "$LOG"; }
declare -A CKPT=([A0_4c]=runs/real/mids_original_full/best.pt [A2_4c]=runs/real/abl/svd_gend/best.pt)
declare -A GP=([A0_4c]=2 [A2_4c]=3)
TUNE_SPECS=""; for f in runs/real/tune/buckets/*.json;     do TUNE_SPECS="$TUNE_SPECS $(basename "$f" .json):$f"; done
EVAL_SPECS="";  for f in runs/real/heldout3/buckets/*.json; do EVAL_SPECS="$EVAL_SPECS $(basename "$f" .json):$f"; done
dump(){
  local tag="$1" gpu="${GP[$1]}" ck="${CKPT[$1]}"
  [ -f "runs/real/eval_dumps/$tag/DONE" ] && { say "$tag DONE, skip"; return; }
  say "START $tag on gpu$gpu ($ck)"
  CUDA_VISIBLE_DEVICES=$gpu python -u scripts/dump_scores.py --model-type mids_plus --checkpoint "$ck" \
    --full-image --tta none --batch-size 64 --num-workers 8 --out "runs/real/tune_dumps/$tag" --data $TUNE_SPECS >> "runs/real/dumplogs/$tag.log" 2>&1
  say "  $tag tune exit=$?"
  CUDA_VISIBLE_DEVICES=$gpu python -u scripts/dump_scores.py --model-type mids_plus --checkpoint "$ck" \
    --full-image --tta none --batch-size 64 --num-workers 8 --out "runs/real/eval_dumps/$tag" --data $EVAL_SPECS >> "runs/real/dumplogs/$tag.log" 2>&1
  say "  $tag eval exit=$?"; touch "runs/real/eval_dumps/$tag/DONE"
}
dump A0_4c & P0=$!
dump A2_4c & P2=$!
wait $P0; wait $P2
say "=== EXTRA 4c DUMPS DONE ==="; touch runs/real/eval_dumps/ALL_DONE_extra4c
