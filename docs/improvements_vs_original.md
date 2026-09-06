# What changed vs. upstream FFAA MIDS

This is the precise diff between FFAA's `mids/mids_arch.py` (the version you run, which adds an
optional CE class-weight vector) and MIDS++.

## Summary

| Aspect | Upstream MIDS | MIDS++ | Source idea |
| --- | --- | --- | --- |
| CLIP adaptation | unfreeze **last 2 transformer layers** (~12M params trained directly) | **Effort SVD residual adapters** on the last K layers' `self_attn.out_proj`: freeze the principal SVD component, train only the minor residual factors | Effort |
| Feature-space discipline | none | **GenD alignment + uniformity** on the L2-normalised image embedding, by authenticity; optional CLIP LayerNorm tuning | GenD |
| Visual forgery evidence | only global CLS + image↔answer cross-attention | **ForensicsAdapter-style artifact head**: learnable forgery queries over CLIP patch tokens, weakly supervised by image labels (MIL), injected as a 7th fusion token | ForensicsAdapter |
| Fusion tokens → classifier | 6 tokens → `Linear(768*6, 4)` | 6 + 1 artifact = 7 → `Linear(768*7, 4)` (config-toggleable back to 6) | — |
| Regularisers | none | SVD orthogonality + Frobenius preservation; weak-MIL real-suppression + fake-sparsity; query orthogonality | Effort / ForensicsAdapter |
| **Unchanged** | T5-base text encoder (frozen); MultiModalFusionBlock; 3 self-attention layers; **4-class scheme**; `make_decision` match-score selector; heavy albumentations aug; data JSON format | identical | — |

VLAForge was deliberately **not** adopted: its core trick (image↔text cross-modal anomaly) is
already what MIDS's `MultiModalFusionBlock` does, so it adds overlap and parameters without a
clear new signal. (The codebase leaves room to add a patch-vs-prompt anomaly map later if wanted.)

## Why these three, specifically

MIDS's headline metric is **`sACC`** — the standard deviation of accuracy across unseen forgery
test sets (lower = better generalisation). The paper's own ablation shows MIDS helps almost
entirely on the **"hard"** images (where the 3 hypothetical answers disagree). So the useful
levers are (1) don't destroy CLIP's open-world prior while adapting, and (2) give the decision an
independent visual cue for the hard cases.

### 1. Effort SVD instead of layer unfreezing (the main lever)
Unfreezing two whole CLIP layers trains ~12M parameters directly on a comparatively small MIDS
set. That overwrites pretrained directions and is the classic cause of *seen-forgery overfitting*
→ high `sACC`. Effort factorises each target weight `W = U S Vh`, **freezes the top-`main_rank`
("principal") component** that carries pretrained knowledge, and trains only a small residual
built from the minor singular directions. Two regularisers keep the adapted weight faithful:
- **orthogonality** — the residual basis stays orthonormal w.r.t. the frozen principal basis, so it
  adds new forgery-specific directions instead of corrupting pretrained ones;
- **Frobenius preservation** — total weight energy is preserved, preventing drift.

This is the change most directly targeted at `sACC`.

### 2. GenD hyperspherical discipline
SVD keeps the *weights* faithful; GenD keeps the *features* generalisable. On the L2-normalised
image embedding MIDS already computes (the projected CLS token), we add alignment (pull
same-authenticity embeddings together) and uniformity (spread embeddings over the unit sphere).
GenD shows this geometry is what transfers across generators. Cheap (two losses, no new params)
and complementary to SVD. CLIP LayerNorm tuning is also allowed as a gentle, low-risk adaptation.

### 3. ForensicsAdapter artifact head (for the hard cases)
Upstream MIDS reasons over a global token + the answer text. When the candidate answers disagree,
a **text-independent** visual signal breaks ties. ForensicsAdapter shows a few learnable forgery
query tokens attending to patch tokens localise manipulation artifacts well. Your data has no
masks, so the artifact map is trained by weak image-level MIL (top-k fake patches must score high;
real images suppressed; fake maps kept sparse). A single pooled artifact-evidence token enters the
fusion as a 7th token, so the final 4-class logits can use visual artifact evidence directly.

## Parameter / cost impact
- Trainable params **drop** vs. unfreezing 2 layers: SVD residuals (rank 16 on a handful of
  `out_proj` layers) + the small artifact head + the existing fusion/classifier, instead of two
  full 1024-d transformer blocks. The artifact head adds a few cross-attention blocks (~a few M).
- Inference adds one cross-attention pass (artifact head) + 1 extra fusion token. Negligible vs.
  the MLLM that runs upstream of MIDS.

## How to verify the gain
Train `configs/mids_original.yaml` (baseline) and `configs/mids_pp.yaml` (full) on the same data,
then evaluate both on your held-out forgery sets with `mids-pp-eval` and compare **mean ACC** and
**`sACC`**. Use `configs/mids_svd_only.yaml` to attribute the gain to the SVD change alone. The
expected ordering on cross-dataset `sACC` is `mids_pp ≤ mids_svd_only < mids_original`.
