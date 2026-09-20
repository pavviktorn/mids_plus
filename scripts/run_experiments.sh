#!/usr/bin/env bash
# Orchestrates experiments #1 (controlled A/B), #2 (cross-dataset sACC), #3 (full training),
# sequentially. Designed to run DETACHED (setsid) so it survives agent turns / usage pauses.
# All logs under runs/real/exp/. Writes EXPERIMENT_REPORT.md and a DONE sentinel at the end.
# No `set -e`: a failing stage is logged but does not abort the rest.

cd /datasets/work/vLLM/temp/mids_plus || exit 1
. .venv/bin/activate
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXP=runs/real/exp
mkdir -p "$EXP/results" "$EXP/buckets"
LOG="$EXP/run.log"

EVAL2000=runs/real/eval2000.json
TRAIN40K=runs/real/train40k.json
VAL1K=runs/real/val1000.json
FULL=/datasets/newout/vqa_info_2+13+4+3_fmt/mids.json
MIDS_EVAL=/datasets/newout/vqa_info_2+13+4+3_fmt/mids_eval.json
UPSTREAM=checkpoints/mids_upstream.pth

say(){ echo "$(date '+%F %T') | $*" | tee -a "$LOG"; }
say "=== experiments started (GPU=$CUDA_VISIBLE_DEVICES) ==="

ckpt(){ [ -f "$1/best.pt" ] && echo "$1/best.pt" || echo "$1/last.pt"; }

# ===================== STAGE 1: controlled A/B (mids_original on same 40k) =====================
say "### STAGE 1: train mids_original on 40k (controlled A/B vs mids_pp/40k) ###"
python -u -m mids_plus.train --config configs/mids_original.yaml \
  --set train_data_path="$TRAIN40K" val_data_path="$VAL1K" skip_missing_images=true \
        epochs=1 batch_size=12 val_batch_size=16 amp_dtype=bf16 num_workers=8 \
        save_every_steps=1000 output_dir=runs/real/mids_original \
  > "$EXP/train_mids_original_40k.log" 2>&1
say "stage1 train exit=$?"

say "STAGE 1 eval: upstream / mids_original_40k / mids_pp_40k on eval2000"
python -u scripts/eval_models.py --model-type upstream --upstream-mids "$UPSTREAM" \
  --data eval2000:"$EVAL2000" --tag upstream --out "$EXP/results/upstream_eval2000.json" >> "$EXP/eval_stage1.log" 2>&1
[ -f runs/real/mids_original/best.pt -o -f runs/real/mids_original/last.pt ] && \
python -u scripts/eval_models.py --model-type mids_plus --checkpoint "$(ckpt runs/real/mids_original)" \
  --data eval2000:"$EVAL2000" --tag mids_original_40k --out "$EXP/results/mids_original_40k_eval2000.json" >> "$EXP/eval_stage1.log" 2>&1
[ -f runs/real/mids_pp/best.pt -o -f runs/real/mids_pp/last.pt ] && \
python -u scripts/eval_models.py --model-type mids_plus --checkpoint "$(ckpt runs/real/mids_pp)" \
  --data eval2000:"$EVAL2000" --tag mids_pp_40k --out "$EXP/results/mids_pp_40k_eval2000.json" >> "$EXP/eval_stage1.log" 2>&1
say "stage1 done"

# ===================== STAGE 2: cross-dataset sACC =====================
say "### STAGE 2: partition mids_eval into dataset buckets + sACC ###"
python -u scripts/partition_eval_by_dataset.py --data "$MIDS_EVAL" \
  --out-dir "$EXP/buckets" --min-items 150 --cap 1500 > "$EXP/partition.log" 2>&1
SPECS=$(grep '^SPECS=' "$EXP/partition.log" | sed 's/^SPECS=//')
say "buckets: $SPECS"

if [ -n "$SPECS" ]; then
  python -u scripts/eval_models.py --model-type upstream --upstream-mids "$UPSTREAM" \
    --data $SPECS --tag upstream --out "$EXP/results/upstream_buckets.json" >> "$EXP/eval_stage2.log" 2>&1
  [ -f runs/real/mids_original/best.pt -o -f runs/real/mids_original/last.pt ] && \
  python -u scripts/eval_models.py --model-type mids_plus --checkpoint "$(ckpt runs/real/mids_original)" \
    --data $SPECS --tag mids_original_40k --out "$EXP/results/mids_original_40k_buckets.json" >> "$EXP/eval_stage2.log" 2>&1
  python -u scripts/eval_models.py --model-type mids_plus --checkpoint "$(ckpt runs/real/mids_pp)" \
    --data $SPECS --tag mids_pp_40k --out "$EXP/results/mids_pp_40k_buckets.json" >> "$EXP/eval_stage2.log" 2>&1
else
  say "WARNING: no buckets produced; skipping stage 2 evals"
fi
say "stage2 done"

# ===================== STAGE 3: full training mids_pp on ~740k =====================
say "### STAGE 3: train MIDS++ on FULL train set ($FULL) ###"
python -u -m mids_plus.train --config configs/mids_pp.yaml \
  --set train_data_path="$FULL" val_data_path="$VAL1K" skip_missing_images=true \
        epochs=1 batch_size=12 val_batch_size=16 amp_dtype=bf16 num_workers=10 \
        save_every_steps=2000 output_dir=runs/real/mids_pp_full \
  > "$EXP/train_mids_pp_full.log" 2>&1
say "stage3 train exit=$?"

if [ -f runs/real/mids_pp_full/best.pt -o -f runs/real/mids_pp_full/last.pt ]; then
  say "STAGE 3 eval: mids_pp_full on eval2000 + buckets"
  python -u scripts/eval_models.py --model-type mids_plus --checkpoint "$(ckpt runs/real/mids_pp_full)" \
    --data eval2000:"$EVAL2000" $SPECS --tag mids_pp_full --out "$EXP/results/mids_pp_full_all.json" >> "$EXP/eval_stage3.log" 2>&1
fi
say "stage3 done"

# ===================== assemble report =====================
say "### building report ###"
python -u scripts/build_report.py --exp-dir "$EXP" > "$EXP/build_report.log" 2>&1
say "=== experiments DONE ==="
touch "$EXP/DONE"
