# MIDS++ training-data format

MIDS++ consumes **exactly** the FFAA MIDS JSON schema, so a pre-generated FFAA-format file needs
no conversion. One JSON file = a list of items, one item per image.

```json
[
  {
    "image": "/abs/path/to/face.jpg",
    "cls_label": 0,
    "answers": [
      {
        "content": "Image description: ...\nForgery reasoning: ...",
        "result": "real",
        "label": 0
      },
      {
        "content": "Image description: ...\nForgery reasoning: ...",
        "result": "fake",
        "label": 1
      },
      {
        "content": "Image description: ...\nForgery reasoning: ...",
        "result": "real",
        "label": 0
      }
    ]
  }
]
```

## Fields

- **`image`** — path to the (already face-cropped, per FFAA) image.
- **`cls_label`** — the image's true authenticity: `0 = real`, `1 = fake` (PAD or deepfake both map
  to `1` in your label scheme).
- **`answers`** — the MLLM's candidate answers for this image. FFAA uses **3**: one neutral + two
  hypothetical (`samples_per_image: 3`). Each answer has:
  - **`content`** — the **verdict-masked** answer text: only `Image description:` and
    `Forgery reasoning:` (the `Analysis result` / `Probability` / `Forgery type` lines are stripped,
    exactly as FFAA's `mask_result` does). This is what MIDS sees.
  - **`result`** — the answer's claimed verdict (`"real"` / `"fake"`), kept aside for the selector.
  - **`label`** — the 4-class target: **`2 * cls_label + (1 if result == "fake" else 0)`**.

## The 4-class scheme (unchanged from FFAA)

| label | image | claim | meaning |
| --- | --- | --- | --- |
| 0 | real | real | real image, real claim (correct) |
| 1 | real | fake | real image, fake claim (wrong) |
| 2 | fake | real | fake image, real claim (wrong) |
| 3 | fake | fake | fake image, fake claim (correct) |

At inference the selector turns the softmaxed 4 logits into a per-answer **match score**
(`P(class0)/(P(class0)+P(class2))` for a real claim; `P(class3)/(P(class3)+P(class1))` for a fake
claim), picks the highest, and reports its verdict + a forgery score. This is byte-for-byte the
upstream rule (`mids_plus/selector.py`).

## Notes
- `samples_per_image` is configurable but defaults to `3` to match FFAA. The selector chunks by
  this value.
- If an item has fewer answers than `samples_per_image`, the loader pads with an ignored entry
  (`label = -1`, which CE ignores).
- Validation/test files use the same schema. For OW-FFA-Bench-style reporting, keep one JSON per
  test set and pass them all to `mids-pp-eval` to get per-set ACC + `sACC`.
- A tiny synthetic example generator lives at `scripts/make_synthetic_smoke_data.py`.
