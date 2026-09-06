# MIDS++ architecture & dataflow

MIDS++ is a compact CLIP-vision + T5-text cross-attention classifier that scores the MLLM's
candidate answers — same role as FFAA's MIDS — hardened for cross-dataset generalisation.

## Forward dataflow

```
image (B,3,336,336)                         answers: B x S masked texts (S = 1+N+M, default 3)
        |                                            |
   CLIP ViT-L/14 @336                          T5-base encoder (frozen)
   (SVD-adapted, last K out_proj;                     |
    LayerNorms optionally tuned)                last_hidden_state -> (B, S, L, 768)
        |
   hidden[-1], hidden[-2]  -> proj 1024->768
        |            \
        |             \---> CLS tokens (layer1, layer2) ----------------------------+
        |                                                                           |
        |  patch tokens (B, P, 768)                                                 |
        |        |                                                                  |
        |   LocalArtifactHead (ForensicsAdapter-style)                              |
        |        |-- artifact_token (B,768) ---------------------------------------+|
        |        |-- artifact_map (B,P)      --> weak MIL loss (image labels)       ||
        |        `-- per-query logits        --> query-orthogonality loss           ||
        |                                                                           ||
        |   CLS(layer1) -> L2 normalise -> image_embed --> GenD align/uniformity    ||
        |                                                                           ||
        `------> MultiModalFusionBlock(layer1, layer2, texts) -- text<->image x-attn||
                          |  tii1, itt1, tii2, itt2  (per (image,answer))           ||
                          v                                                         vv
        concat [ cls1, tii1, itt1, cls2, tii2, itt2, (artifact_token) ]  -> (B*S, 7, 768)
                          |
                  3 x self-attention
                          |
                 flatten -> Linear(768*7, 4)  -> logits (B, S, 4)
                          |
                 softmax + make_decision (unchanged selector) -> verdict + forgery score
```

`N`/`M` only set `S = 1+N+M`; the trainer passes `N=S-1, M=0`. With the default `S=3` this matches
FFAA's "1 neutral + 2 hypothetical".

## Where each adopted idea lives

| Idea | Module | Loss terms |
| --- | --- | --- |
| Effort SVD residual adaptation | `svd.py` (`EffortSVDLinear`, `replace_linears_with_svd`) | `svd_orth`, `svd_keep` |
| GenD hyperspherical discipline | `gend.py` + `image_embed` in `model.py` | `alignment`, `uniformity` |
| ForensicsAdapter artifact head | `artifact.py` (`LocalArtifactHead`) | `artifact_mil`, `query_orth` |
| (unchanged) cross-modal fusion | `fusion.py` (`MultiModalFusionBlock`) | `ce` |

## Training objective
`L = ce + alignment + uniformity + artifact_mil + query_orth + svd_orth + svd_keep`, each gated by
a config weight (`losses.py`). CE is the FFAA 4-class loss with optional class weights + smoothing.
Auxiliary losses use the **per-image** authenticity label (`cls_label`); CE uses the **per-answer**
4-class label. The staged ablations (`mids_original` → `mids_svd_only` → `mids_pp`) are pure config.

## Design choices worth noting
- **Drop-in first.** The text encoder, fusion block, self-attention depth, 4-class head, selector,
  augmentation, and data format are all kept identical to upstream so this is a true MIDS swap.
- **Checkpoints are tiny.** Only learned tensors are saved; frozen T5/CLIP and the SVD principal
  component (deterministically re-derived from CLIP at load) are not. See `checkpoint.py`.
- **Stub encoders for testing.** `encoders.py` can build HF-CLIP-shaped random stand-ins so the
  whole pipeline (including SVD replacement) runs on CPU in the smoke test without any downloads.
