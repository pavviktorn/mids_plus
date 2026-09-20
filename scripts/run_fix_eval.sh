#!/usr/bin/env bash
# Test MIDS++ full vs mids_original full on the temp_fix test JSONs (random 5k subsets each).
# Detached + logged. Writes result JSONs + a COMPARISON.md + DONE_FIX sentinel.
cd /datasets/work/vLLM/temp/mids_plus || exit 1
. .venv/bin/activate
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SRC=/datasets/newout/vqa_info_2+13+4+3_fmt/temp_fix
FIX=runs/real/exp_fix
mkdir -p "$FIX/subsets" "$FIX/results"
LOG="$FIX/run.log"
say(){ echo "$(date '+%F %T') | $*" | tee -a "$LOG"; }

say "=== building random 5k subsets from $SRC (GPU=$CUDA_VISIBLE_DEVICES) ==="
python -u scripts/make_test_subsets.py --src-dir "$SRC" --out-dir "$FIX/subsets" --per 5000 --seed 0 > "$FIX/subset.log" 2>&1
SPECS=$(grep '^SPECS=' "$FIX/subset.log" | sed 's/^SPECS=//')
say "subsets: $SPECS"

if [ -z "$SPECS" ]; then say "ERROR: no subsets"; touch "$FIX/DONE_FIX"; exit 1; fi

say "=== eval MIDS++ full ==="
python -u scripts/eval_models.py --model-type mids_plus --checkpoint runs/real/mids_pp_full/best.pt \
  --data $SPECS --batch-size 32 --tag mids_pp_full --out "$FIX/results/mids_pp_full.json" >> "$FIX/eval.log" 2>&1
say "=== eval mids_original full ==="
python -u scripts/eval_models.py --model-type mids_plus --checkpoint runs/real/mids_original_full/best.pt \
  --data $SPECS --batch-size 32 --tag mids_original_full --out "$FIX/results/mids_original_full.json" >> "$FIX/eval.log" 2>&1

say "=== building comparison ==="
python -u - <<'PY' > "$FIX/COMPARISON.md" 2>>"$FIX/eval.log"
import json, os, math
R="runs/real/exp_fix/results"
res={}
for tag in ["mids_original_full","mids_pp_full"]:
    p=os.path.join(R,tag+".json")
    if os.path.exists(p): res[tag]=json.load(open(p))
sets=[]
for t in res.values():
    for k in t["per_set"]:
        if k not in sets: sets.append(k)
sets.sort()
def sacc(d):
    a=[v for v in d.values() if v==v]
    if not a: return float('nan')
    m=sum(a)/len(a); return math.sqrt(sum((x-m)**2 for x in a)/len(a))*100
print("# temp_fix test: MIDS++ full vs mids_original full\n")
print("Random 5,000-item subset per file (seed 0), identical subsets for both models. "
      "ACC/AUC/AP binary; AUC `-` where a file is single-class. sACC = std of per-file ACC (x100).\n")
hdr="| Model | " + " | ".join(sets) + " | mean ACC | sACC |"
print(hdr); print("|"+ " --- |"*(len(sets)+3))
for tag in ["mids_original_full","mids_pp_full"]:
    if tag not in res: continue
    ps=res[tag]["per_set"]; accs={s:ps[s]["acc"] for s in sets if s in ps}
    cells=[f"{ps[s]['acc']*100:.2f}" if s in ps else "-" for s in sets]
    mean=sum(accs.values())/len(accs)*100
    print(f"| {tag} | " + " | ".join(cells) + f" | {mean:.2f} | {sacc(accs):.2f} |")
print("\nPer-file detail (n, real/fake, AUC, AP, neutral-baseline) from MIDS++ full:")
t=res.get("mids_pp_full") or res.get("mids_original_full")
for s in sets:
    m=t["per_set"][s]
    auc = f"{m['auc']*100:.2f}" if m['auc']==m['auc'] else "n/a"
    print(f"- {s}: n={m['n']} (real {m['n_real']}/fake {m['n_fake']}), AUC={auc}, AP={m['ap']*100:.2f}, neutral={m['neutral_acc']*100:.2f}")
PY
cat "$FIX/COMPARISON.md" >> "$LOG"
say "=== DONE_FIX ==="
touch "$FIX/DONE_FIX"
