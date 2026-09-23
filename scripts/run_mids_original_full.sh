#!/usr/bin/env bash
# Controlled comparison: train mids_original on the FULL ~740k with the SAME recipe as MIDS++ full,
# then eval on eval2000 + the existing dataset buckets, and regenerate the report.
# Isolates architecture (mids_original vs MIDS++) from training recipe. Detached + logged.

cd /datasets/work/vLLM/temp/mids_plus || exit 1
. .venv/bin/activate
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXP=runs/real/exp
LOG="$EXP/run_origfull.log"
EVAL2000=runs/real/eval2000.json
VAL1K=runs/real/val1000.json
FULL=/datasets/newout/vqa_info_2+13+4+3_fmt/mids.json

say(){ echo "$(date '+%F %T') | $*" | tee -a "$LOG"; }
ckpt(){ [ -f "$1/best.pt" ] && echo "$1/best.pt" || echo "$1/last.pt"; }

# rebuild the bucket SPECS from the buckets produced earlier
SPECS=""
for f in "$EXP"/buckets/*.json; do
  [ -e "$f" ] || continue
  n=$(basename "$f" .json); SPECS="$SPECS $n:$f"
done
say "=== mids_original FULL training started (GPU=$CUDA_VISIBLE_DEVICES) ==="
say "buckets:$SPECS"

# identical recipe to MIDS++ full (configs/mids_pp.yaml stage 3): 1 epoch, batch 12, bf16, full data
python -u -m mids_plus.train --config configs/mids_original.yaml \
  --set train_data_path="$FULL" val_data_path="$VAL1K" skip_missing_images=true \
        epochs=1 batch_size=12 val_batch_size=16 amp_dtype=bf16 num_workers=10 \
        save_every_steps=2000 output_dir=runs/real/mids_original_full \
  > "$EXP/train_mids_original_full.log" 2>&1
say "train exit=$?"

if [ -f runs/real/mids_original_full/best.pt -o -f runs/real/mids_original_full/last.pt ]; then
  say "eval mids_original_full on eval2000 + buckets"
  python -u scripts/eval_models.py --model-type mids_plus --checkpoint "$(ckpt runs/real/mids_original_full)" \
    --data eval2000:"$EVAL2000" $SPECS --tag mids_original_full \
    --out "$EXP/results/mids_original_full_all.json" >> "$EXP/eval_origfull.log" 2>&1
  say "eval exit=$?"
else
  say "ERROR: no mids_original_full checkpoint produced"
fi

python -u scripts/build_report.py --exp-dir "$EXP" > "$EXP/build_report.log" 2>&1
say "=== mids_original FULL DONE ==="
touch "$EXP/DONE_ORIGFULL"
