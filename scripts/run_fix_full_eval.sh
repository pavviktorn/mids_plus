#!/usr/bin/env bash
# FULL evaluation (no subsampling) of MIDS++ full vs mids_original full on the temp_fix test files.
# ~945k items x 2 models. Detached + logged. batch 128 / 16 workers. Writes COMPARISON_FULL.md + sentinel.
cd /datasets/work/vLLM/temp/mids_plus || exit 1
. .venv/bin/activate
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SRC=/datasets/newout/vqa_info_2+13+4+3_fmt/temp_fix
FIX=runs/real/exp_fix
mkdir -p "$FIX/results"
LOG="$FIX/run_full.log"
say(){ echo "$(date '+%F %T') | $*" | tee -a "$LOG"; }

SPECS=""
for f in "$SRC"/*.json; do n=$(basename "$f" .json); SPECS="$SPECS $n:$f"; done
say "=== FULL eval started (GPU=$CUDA_VISIBLE_DEVICES) ==="
say "data:$SPECS"

say "=== eval mids_original full (FULL sets) ==="
python -u scripts/eval_models.py --model-type mids_plus --checkpoint runs/real/mids_original_full/best.pt \
  --data $SPECS --batch-size 128 --num-workers 16 --tag mids_original_full \
  --out "$FIX/results/mids_original_full_FULL.json" >> "$FIX/eval_full.log" 2>&1
say "mids_original full eval exit=$?"

say "=== eval MIDS++ full (FULL sets) ==="
python -u scripts/eval_models.py --model-type mids_plus --checkpoint runs/real/mids_pp_full/best.pt \
  --data $SPECS --batch-size 128 --num-workers 16 --tag mids_pp_full \
  --out "$FIX/results/mids_pp_full_FULL.json" >> "$FIX/eval_full.log" 2>&1
say "mids_pp full eval exit=$?"

say "=== building comparison ==="
python -u - <<'PY' > "$FIX/COMPARISON_FULL.md" 2>>"$FIX/eval_full.log"
import json, os, math
R="runs/real/exp_fix/results"
res={}
for tag in ["mids_original_full","mids_pp_full"]:
    p=os.path.join(R,tag+"_FULL.json")
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
print("# temp_fix FULL test (no subsampling): MIDS++ full vs mids_original full\n")
print("Every item in each file (missing images skipped), identical data for both models. "
      "ACC/AUC/AP binary; AUC `-` where single-class. sACC = std of per-file ACC (x100).\n")
print("| Model | " + " | ".join(f"{s} ACC" for s in sets) + " | mean ACC | sACC |")
print("|"+ " --- |"*(len(sets)+3))
for tag in ["mids_original_full","mids_pp_full"]:
    if tag not in res: continue
    ps=res[tag]["per_set"]; accs={s:ps[s]["acc"] for s in sets if s in ps}
    cells=[f"{ps[s]['acc']*100:.3f}" if s in ps else "-" for s in sets]
    mean=sum(accs.values())/len(accs)*100
    print(f"| {tag} | " + " | ".join(cells) + f" | {mean:.3f} | {sacc(accs):.3f} |")
print("\n## AUC per file")
print("| Model | " + " | ".join(sets) + " |"); print("|"+" --- |"*(len(sets)+1))
for tag in ["mids_original_full","mids_pp_full"]:
    if tag not in res: continue
    ps=res[tag]["per_set"]
    cells=[(f"{ps[s]['auc']*100:.3f}" if (s in ps and ps[s]['auc']==ps[s]['auc']) else "n/a") for s in sets]
    print(f"| {tag} | " + " | ".join(cells) + " |")
print("\n## Per-file size / balance (from MIDS++ full)")
t=res.get("mids_pp_full") or res.get("mids_original_full")
for s in sets:
    m=t["per_set"][s]
    print(f"- {s}: n={m['n']} (real {m['n_real']}/fake {m['n_fake']}), neutral-baseline ACC={m['neutral_acc']*100:.2f}")
PY
cat "$FIX/COMPARISON_FULL.md" >> "$LOG"
say "=== DONE_FIX_FULL ==="
touch "$FIX/DONE_FIX_FULL"
