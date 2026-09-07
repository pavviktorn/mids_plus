# Dropping MIDS++ into your FFAA pipeline

MIDS++ keeps MIDS's input/output contract, so it replaces the MIDS stage only — the fine-tuned
MLLM, the hypothetical-prompt generation, masking, and `make_decision` all stay as they are.

## 1. Point MIDS++ at your existing encoder weights

MIDS++ uses the **same** `clip-vit-large-patch14-336` and `t5-base` your FFAA install already has.
From the project root:

```bash
ln -s /path/to/FFAA/models models      # so configs' models/clip-... and models/t5-base resolve
```

(or override per-run with `--set image_model_path=... text_model_path=...`).

## 2. Train

```bash
# single GPU
TRAIN_JSON=/path/train.json VAL_JSON=/path/val.json bash scripts/train_mids_pp.sh configs/mids_pp.yaml
# multi-GPU (e.g. 4x)
TRAIN_JSON=/path/train.json VAL_JSON=/path/val.json NPROC=4 bash scripts/train_mids_pp.sh configs/mids_pp.yaml
```

Produces `runs/mids_pp/best.pt` — a small checkpoint holding only the learned tensors.

## 3. Swap into inference

The contract is identical to FFAA's MIDS, so you have two options.

### Option A — use the helper (least code)
Replace your `load_mids(...)` + scoring block with:

```python
from mids_plus.infer import MidsPlusScorer

scorer = MidsPlusScorer("runs/mids_pp/best.pt", device="cuda")
# masked_texts / results: one entry per candidate answer, exactly as you build them today
best_idx, pred, match_score, forgery_score = scorer.score(image_rgb, masked_texts, results)
```

`image_rgb` is an HxWx3 uint8 RGB array (the face crop you already produce).

### Option B — call the module directly (matches upstream call sites)
The model's `forward(texts, image, labels, batch_size, N, M)` and the 4-class logits are the same
shape as upstream MIDS, and `mids_plus.selector.make_decision` is identical to FFAA's. So existing
code that did:

```python
logits = mids(input_ids, images, None, batch_size, 1, 1)["logits"]
scores = F.softmax(logits, dim=2)
... make_decision_batch(answers_result, scores) ...
```

works against MIDS++ unchanged, except you build the model via
`mids_plus.checkpoint.load_checkpoint(path)` instead of FFAA's `load_mids`. The only shape change
is internal (7 fusion tokens instead of 6); inputs and outputs are unchanged.

## 4. Differences you should know about
- The checkpoint stores **only learned tensors**. At load time MIDS++ rebuilds the frozen
  T5/CLIP and **re-derives the SVD principal component** from the original CLIP weights, then loads
  the learned tensors on top. So the same CLIP/T5 weights must be reachable at load time.
- `svd_residual_rank`, `svd_target_last_layers`, `artifact_enabled`, `samples_per_image`, and
  `num_classes` must match between training and loading (they define tensor shapes). They are stored
  in the checkpoint and restored automatically by `load_checkpoint`.
- To A/B against your current model, train `configs/mids_original.yaml` (it reproduces upstream
  MIDS within this codebase) and compare with `mids-pp-eval`.
