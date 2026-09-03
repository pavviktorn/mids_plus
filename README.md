# MIDS++

A generalization-hardened, **drop-in replacement for FFAA's MIDS** (Multi-answer Intelligent
Decision System). It keeps MIDS's exact contract — image + masked candidate answers in, per-answer
4-class logits out, scored by the unchanged match-score selector — but replaces the internals that
limit cross-dataset generalisation.

Built to improve FFAA's headline robustness metric **`sACC`** (std of accuracy across unseen
forgery sets; lower is better), using selected ideas from three papers:

- **Effort** — SVD residual adaptation of CLIP instead of unfreezing whole layers (preserves the
  pretrained open-world subspace). *The main lever.*
- **GenD** — hyperspherical alignment/uniformity + L2 discipline on the image embedding.
- **ForensicsAdapter** — a weakly-supervised local artifact head giving the decision a
  text-independent visual cue for the "hard" cases MIDS exists to resolve.

VLAForge was intentionally left out (its cross-modal anomaly idea overlaps what MIDS's fusion
already does). See [docs/improvements_vs_original.md](docs/improvements_vs_original.md) for the
full rationale and the precise diff against the MIDS you currently run.

## Layout

```
mids_plus/
  configs/      mids_pp (full) · mids_svd_only · mids_original (baseline) · smoke
  docs/         design · data_format · integration_dropin · improvements_vs_original
  scripts/      train launcher · synthetic smoke-data generator
  src/mids_plus/
    svd.py        Effort SVD residual linears + CLIP layer replacement + regularisers
    gend.py       GenD alignment / uniformity
    artifact.py   ForensicsAdapter-style local artifact head + weak MIL
    fusion.py     image<->answer cross-modal fusion (unchanged from FFAA)
    model.py      MIDSPlus (drop-in forward signature)
    losses.py     combined objective
    selector.py   match-score decision (identical to FFAA)
    data.py       dataset/collate for the FFAA MIDS JSON format
    train.py      torchrun/DDP + AMP trainer
    evaluate.py   per-set ACC/AUC/AP + sACC
    infer.py      MidsPlusScorer drop-in helper
  tests/        label-convention (no torch) · components · end-to-end smoke
```

## Install

```bash
cd mids_plus
pip install -e ".[dev]"
ln -s /path/to/FFAA/models models   # reuse your existing clip-vit-large-patch14-336 + t5-base
```

## Train

Data is the **same JSON format MIDS already uses** ([docs/data_format.md](docs/data_format.md)) —
a pre-generated FFAA-format file needs no conversion.

```bash
# single GPU
TRAIN_JSON=/path/train.json VAL_JSON=/path/val.json bash scripts/train_mids_pp.sh configs/mids_pp.yaml
# 4 GPUs
TRAIN_JSON=/path/train.json VAL_JSON=/path/val.json NPROC=4 bash scripts/train_mids_pp.sh configs/mids_pp.yaml
```

## Evaluate (OW-FFA-Bench style)

```bash
mids-pp-eval --checkpoint runs/mids_pp/best.pt \
  --data DPF:/data/dpf.json DFD:/data/dfd.json DFDC:/data/dfdc.json
# prints per-set ACC/AUC/AP and the MEAN + sACC summary
```

## Drop into FFAA inference

```python
from mids_plus.infer import MidsPlusScorer
scorer = MidsPlusScorer("runs/mids_pp/best.pt", device="cuda")
best_idx, pred, match_score, forgery_score = scorer.score(image_rgb, masked_texts, results)
```

Full integration notes: [docs/integration_dropin.md](docs/integration_dropin.md).

## Smoke test (no model downloads)

Uses tiny random stand-in encoders so the whole pipeline — SVD replacement, artifact head, every
loss, backward, selector, checkpoint round-trip — runs on CPU in seconds:

```bash
pytest -q                                   # all tests
python -m mids_plus.train --config configs/smoke.yaml \
  --set train_data_path=/tmp/s/train.json val_data_path=/tmp/s/val.json   # after make_synthetic_smoke_data.py
```

## Status / scope
This is a **ready-to-train** project + a passing smoke test, not a finished training run. Real
training on the 1M-image set runs on your cluster (the 4×96GB box this targets handles
`configs/mids_pp.yaml` comfortably). Compare against `configs/mids_original.yaml` to measure the
generalisation gain.
