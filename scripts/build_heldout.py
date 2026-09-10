#!/usr/bin/env python
"""Build a genuinely-unseen, labeled held-out eval set for fixed-answer MIDS++.

Source: eFFAA_ext_eval.json (eFFAA conversation format, 0% overlap with mids.json training).
We keep image + cls_label (from the 'Analysis result:' line) + a source-family tag, resolve image
paths under /datasets/newout, and partition by family. Families absent from the training family list
are flagged NOVEL (true zero-shot forgery sources); known families here are still new FILES.
Writes MIDS-format per-family JSONs [{id,image,cls_label}] under runs/real/heldout/buckets/.
"""
import json, os, re, collections

SRC = "/datasets/newout/vqa_info_2+13+4+3_fmt/eFFAA_ext_eval.json"
BASE = "/datasets/newout"
OUT = "runs/real/heldout"
os.makedirs(f"{OUT}/buckets", exist_ok=True)

# (path-token, family-name, is_novel)  -- first match wins; order matters (specific first)
TABLE = [
    ("siw-mv2","SiW-Mv2",True), ("3d_attacks","3D-Attacks",True),
    ("dfdc","DFDC",True), ("disco_gan","DiscoGAN",True),
    ("ff++","FF++",True), ("faceforensics","FF++",True),
    ("deepfake_challenge","DFChallenge",True),
    ("add_webcam","webcam",True), ("crop_mywebcam","webcam",True),
    ("axonlabs","axonlabs",True), ("border_test","border_test",True),
    ("how_fmc","how_fmc",True), ("add_free","add_free",True), ("nizar_tests","nizar_tests",True),
    # known families (in training) -- new files here, still held-out
    ("cefa","CeFA",False), ("replay-attack","ReplayAttack",False), ("replay-mobile","ReplayMobile",False),
    ("celebdf","CelebDF",False), ("dfmnist","DFMNIST",False), ("deepfakeface","DeepFakeFace",False),
    ("deep_live_cam","DeepLiveCam",False), ("deeperforensics","DeeperForensics",False),
    ("wffd","WFFD",False), ("alive-images","alive(real)",False), ("alive_images","alive(real)",False),
]

def parse_label(it):
    g = " \n".join(c.get("value","") for c in it["conversations"] if c.get("from")=="gpt")
    m = re.search(r"Analysis result:\s*(fake|real)", g, re.I)
    return None if not m else (1 if m.group(1).lower()=="fake" else 0)

def family(p):
    low = p.lower()
    for tok,name,novel in TABLE:
        if tok in low: return name, novel
    return "misc_novel", True   # unmatched collections: conservatively novel

data = json.load(open(SRC))
by_fam = collections.defaultdict(list)
novel_of = {}
kept = dropped = 0
for i,it in enumerate(data):
    lab = parse_label(it)
    if lab is None: dropped += 1; continue
    ap = os.path.join(BASE, it["image"])
    fam, novel = family(it["image"])
    by_fam[fam].append({"id": it.get("id", i), "image": ap, "cls_label": lab})
    novel_of[fam] = novel
    kept += 1

# write per-family buckets with >=50 items; lump smaller novel ones into misc_novel
MIN = 50
misc = []
written = {}
for fam, items in by_fam.items():
    if fam == "misc_novel" or len(items) < MIN:
        if novel_of[fam]: misc.extend(items)
        else: written[fam] = items  # keep small known families separate too
    else:
        written[fam] = items
if misc: written["misc_novel"] = misc; novel_of["misc_novel"] = True

# also a combined ALL file
allitems = [x for items in by_fam.values() for x in items]
json.dump(allitems, open(f"{OUT}/heldout_ALL.json","w"))
for fam, items in written.items():
    json.dump(items, open(f"{OUT}/buckets/{fam}.json","w"))

# report
def stats(items):
    r=sum(1 for x in items if x["cls_label"]==0); f=len(items)-r; return r,f
print(f"kept={kept} dropped(no label)={dropped}  families written={len(written)}")
print(f"{'family':18s} {'N':>6s} {'real':>6s} {'fake':>6s}  novel?")
for fam in sorted(written, key=lambda k:-len(written[k])):
    r,f = stats(written[fam])
    print(f"{fam:18s} {len(written[fam]):6d} {r:6d} {f:6d}  {'NOVEL' if novel_of[fam] else 'known(new-files)'}")
nnov=sum(len(written[k]) for k in written if novel_of[k])
print(f"\nNOVEL-source items: {nnov}   KNOWN-family(new-files) items: {kept-nnov - dropped if False else sum(len(written[k]) for k in written if not novel_of[k])}")
print(f"combined ALL: {len(allitems)}")
