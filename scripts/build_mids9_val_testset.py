#!/usr/bin/env python
"""Build the 9-class validation set from the self-contained testset.

Same schema as build_mids9.py / build_mids9_val_heldout3.py (3 fixed claim templates,
9-class label = 3*true + claim), but items come from /datasets/work/vLLM/temp/testset:

  testset/real            -> true 0 (real)
  testset/fake/pad        -> true 1 (pad)
  testset/fake/deepfake   -> true 2 (deepfake)

true type is the path component (authoritative). subset = id minus the trailing frame/index
suffix (real_id_R_10, pad_Advanced_11ids_159video, df_v2_2025W01), matching the heldout3 build.
Output: runs/real/mids9/mids9_val.json
"""
import json, os, glob, re, collections

BASE = "/datasets/work/vLLM/temp/mids_plus"
TESTSET = "/datasets/work/vLLM/temp/testset"
OUT = os.path.join(BASE, "runs/real/mids9/mids9_val.json")
IMG_EXT = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")

# identical templates/order to build_mids9.py ; claim: 0 real, 1 pad, 2 deepfake
TEMPLATES = [
    (2, "deepfake", "misaligned features, smooth skin, inconsistent lighting, blending artifacts, "
                    "unnatural integration between the face and body/the face and the background"),
    (1, "pad", "flat depth, uniform lighting, screen smoothing, moire patterns, screen borders, "
               "UI, reflections, glossy/plastic skin, rigid edges, hand-holding signs, recapture blur"),
    (0, "real", "consistent facial features, natural skin texture, consistent lighting, "
                "normal facial 3D depth, the absence of clear manipulation or visible artifacts"),
]

def item(id_, image, true, subset):
    return {"id": id_, "image": image,
            "cls_label": 0 if true == 0 else 1,
            "answers": [{"content": txt, "result": name, "label": 3 * true + claim}
                        for (claim, name, txt) in TEMPLATES],
            "subset": subset}

def files(subdir):
    out = []
    for pat in IMG_EXT:
        out += glob.glob(os.path.join(TESTSET, subdir, pat))
    return sorted(out)

def subset_of(bn, true):
    # real_/pad_ frames carry the subset before "__"; df ids end with "_<index>".
    if "__" in bn:
        return bn.split("__")[0]
    return re.sub(r"_\d+$", "", bn)          # df_v2_2025W01_0 -> df_v2_2025W01

out, cnt = [], collections.Counter()
for subdir, true, key in (("real", 0, "real"),
                          ("fake/pad", 1, "pad"),
                          ("fake/deepfake", 2, "deepfake")):
    for fp in files(subdir):
        bn = os.path.splitext(os.path.basename(fp))[0]
        out.append(item(bn, fp, true, subset_of(bn, true)))
        cnt[key] += 1

os.makedirs(os.path.dirname(OUT), exist_ok=True)
json.dump(out, open(OUT, "w"))
print(f"built {len(out)} val items -> {OUT}")
print("true-type counts:", dict(cnt))
hist = collections.Counter(a["label"] for x in out for a in x["answers"])
print("9-class label histogram (0..8):", {k: hist.get(k, 0) for k in range(9)})
miss = sum(not os.path.exists(x["image"]) for x in out)
print("items with a missing image file:", miss)
print("distinct subsets:", len({x["subset"] for x in out}))
