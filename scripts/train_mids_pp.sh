#!/usr/bin/env bash
# Launch MIDS++ training. Single- or multi-GPU via torchrun.
#
#   TRAIN_JSON=/path/train.json VAL_JSON=/path/val.json NPROC=4 \
#     bash scripts/train_mids_pp.sh configs/mids_pp.yaml
#
# Point ./models at your FFAA CLIP/T5 weights first, e.g.:
#   ln -s /path/to/FFAA/models models
set -euo pipefail

CONFIG="${1:-configs/mids_pp.yaml}"
NPROC="${NPROC:-1}"
TRAIN_JSON="${TRAIN_JSON:?set TRAIN_JSON=/path/to/train.json}"
VAL_JSON="${VAL_JSON:-}"

OVERRIDES=("train_data_path=${TRAIN_JSON}")
if [[ -n "${VAL_JSON}" ]]; then
  OVERRIDES+=("val_data_path=${VAL_JSON}")
fi

if [[ "${NPROC}" -gt 1 ]]; then
  torchrun --nproc_per_node="${NPROC}" -m mids_plus.train --config "${CONFIG}" --set "${OVERRIDES[@]}"
else
  python -m mids_plus.train --config "${CONFIG}" --set "${OVERRIDES[@]}"
fi
