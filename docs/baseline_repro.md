# Upstream MIDS baseline — reproduced

Validation that MIDS++ is a faithful drop-in: loading the user's trained FFAA `mids.pth` into
`mids_original` mode and evaluating on their real eval set reproduces a strong, sensible result.

## Result (2026-06-13)

Data: `/datasets/newout/vqa_info_2+13+4+3_fmt/mids_eval.json` (37,043 items; 1,341 dropped for
missing images; **first 2,000 of the remainder** evaluated on CPU).

```
loaded 112/112 checkpoint tensors; unexpected=0
upstream MIDS    : ACC=96.20  AUC=99.68  AP=99.66
neutral baseline : ACC=91.60   (trust the neutral answer's verdict)
```

Checkpoint: `checkpoints/mids_upstream.pth` (= their
`FFAA-master-newfmt-faster1_transformers4372/checkpoints_4+5fmt/effaa-llava-mistral-7b-lora_0/mids.pth`).
Encoders: copied `clip-vit-large-patch14-336` + `t5-base` under `models/`.

## Reproduce

```bash
. .venv/bin/activate
PYTHONPATH=src python scripts/import_and_eval_upstream.py \
  --mids-pth checkpoints/mids_upstream.pth \
  --data /datasets/newout/vqa_info_2+13+4+3_fmt/mids_eval.json \
  --limit 2000 --batch-size 32 --num-workers 8
```

(`--limit 0` evaluates all items; on CPU the full set is ~2.5h, so use a GPU env for that.)

## Notes on interpreting this number

- **This set is in-distribution for the trained MIDS** (same `..._fmt` family it was trained on),
  so 96.2% reflects in-distribution accuracy. The MIDS++ changes (Effort SVD, GenD) primarily
  target **cross-dataset generalisation** (`sACC`) — their benefit is largest on forgery types
  **not** seen in training, and may be modest on this particular set.
- A transformers **4.37 ↔ 5.x** CLIP key-naming change (`vision_model.` prefix) is handled by
  `import_upstream_mids_checkpoint`; `unexpected=0` confirms a complete load.

## MIDS++ trained result (short real run, 2026-06-13)

Trained `configs/mids_pp.yaml` (Effort SVD + GenD + ForAda artifact) on **40,000 items / 1 epoch**
from `mids.json` (full train set is ~740k), real CLIP/T5, one Blackwell GPU, bf16, batch 12
(~10 min). Evaluated on the **same** `eval2000.json` as the baseline.

| Model | Training | ACC | AUC | AP |
| --- | --- | --- | --- | --- |
| Upstream MIDS (`mids_upstream.pth`) | full ~740k, converged | 96.20 | 99.68 | 99.66 |
| **MIDS++** (svd+gend+artifact) | **40k / 1 epoch** | **96.90** | 99.35 | 99.48 |

MIDS++ is **+0.70 ACC** and on par on AUC/AP despite ~18x less training. Checkpoint:
`runs/real/mids_pp/best.pt`. Training loss: `ce 1.65 -> 0.30`, `artifact_mil 0.30 -> 0.15`.

**Caveats (read before over-claiming):**
- This is **one in-distribution set**; the SVD/GenD design targets cross-dataset `sACC`, not
  measured here. The real test is multiple held-out forgery sets.
- Not yet a *controlled* A/B: upstream trained on ~740k, MIDS++ on 40k. The clean comparison is
  `mids_original` vs `mids_pp` on the **same** 40k (next step below).
- +0.70 ACC on 2000 samples is ~1.6 SE — suggestive, not yet significant on this set alone; the
  stronger signal is matching/beating a converged model with far less training.

## Recommended comparison protocol for MIDS++

To show the generalisation gain rather than just in-distribution parity:
1. Train `configs/mids_original.yaml` (baseline) and `configs/mids_pp.yaml` (full) on the **same**
   train split.
2. Evaluate both on **multiple held-out** forgery sets (one JSON each) with `mids-pp-eval`.
3. Compare **mean ACC** and especially **`sACC`** (lower = better). Expected ordering on `sACC`:
   `mids_pp ≤ mids_svd_only < mids_original`.
4. As a sanity check, also confirm `mids_pp` is at least on par with the 96.2% in-distribution
   number above on this set.
