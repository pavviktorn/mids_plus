#!/usr/bin/env bash
# Eval 9-class A0-A3 on the in-distribution mids_eval buckets (eval2000 + 10 datasets), standard
# face-crop transform (NOT --full-image), to complete the 3-mode in-distribution comparison.
cd /datasets/work/vLLM/temp/mids_plus || exit 1
. .venv/bin/activate
export PYTHONPATH=src TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NINE=runs/real/9class_resume; OUT=$NINE/results9_indist; mkdir -p "$OUT"
EXP=runs/real/exp
SPECS="eval2000:runs/real/eval2000.json"
for f in "$EXP"/buckets/*.json; do n=$(basename "$f" .json); SPECS="$SPECS ${n}:$f"; done
ck(){ [ -f "$1/best.pt" ] && echo "$1/best.pt" || echo "$1/last.pt"; }
for tag in A0_baseline A1_svd A2_svdgend A3_midspp; do
  CUDA_VISIBLE_DEVICES=1 python -u scripts/eval_fixed_answers.py --model-type mids_plus \
    --checkpoint "$(ck $NINE/$tag)" --data $SPECS --batch-size 64 --num-workers 8 \
    --tag "${tag}_9cID" --out "$OUT/${tag}.json" >> "$OUT/eval.log" 2>&1
  echo "$(date '+%F %T') done $tag exit=$?" >> "$OUT/eval.log"
done
touch "$OUT/DONE_9C_INDIST"
