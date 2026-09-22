#!/usr/bin/env bash
# Fixed-answer vs MLLM-answer A/B for upstream MIDS and MIDS++ full, on eval2000 + mids_eval buckets.
# Fixed-answer numbers computed here; MLLM-answer numbers reused from runs/real/exp/results/.
cd /datasets/work/vLLM/temp/mids_plus || exit 1
. .venv/bin/activate
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

EXP=runs/real/exp
FA=runs/real/exp_fixedans
mkdir -p "$FA/results"
LOG="$FA/run.log"
say(){ echo "$(date '+%F %T') | $*" | tee -a "$LOG"; }

SPECS="eval2000:$EXP/../eval2000.json"
for f in "$EXP"/buckets/*.json; do n=$(basename "$f" .json); SPECS="$SPECS $n:$f"; done
say "=== fixed-answer eval started (GPU=$CUDA_VISIBLE_DEVICES) ==="
say "data:$SPECS"

say "=== upstream MIDS, FIXED answers ==="
python -u scripts/eval_fixed_answers.py --model-type upstream --upstream-mids checkpoints/mids_upstream.pth \
  --data $SPECS --batch-size 64 --num-workers 12 --tag upstream_fixed \
  --out "$FA/results/upstream_fixed.json" >> "$FA/eval.log" 2>&1
say "upstream fixed exit=$?"

say "=== MIDS++ full, FIXED answers ==="
python -u scripts/eval_fixed_answers.py --model-type mids_plus --checkpoint runs/real/mids_pp_full/best.pt \
  --data $SPECS --batch-size 64 --num-workers 12 --tag mids_pp_full_fixed \
  --out "$FA/results/mids_pp_full_fixed.json" >> "$FA/eval.log" 2>&1
say "mids_pp_full fixed exit=$?"

say "=== building comparison (fixed vs MLLM answers) ==="
python -u - <<'PY' > "$FA/COMPARISON_FIXED.md" 2>>"$FA/eval.log"
import json, os, math
EXP="runs/real/exp/results"; FA="runs/real/exp_fixedans/results"
def load(p): return json.load(open(p)) if os.path.exists(p) else None
# MLLM-answer (stored)
up_mllm_e=load(f"{EXP}/upstream_eval2000.json"); up_mllm_b=load(f"{EXP}/upstream_buckets.json")
pp_mllm=load(f"{EXP}/mids_pp_full_all.json")
# fixed-answer (new)
up_fix=load(f"{FA}/upstream_fixed.json"); pp_fix=load(f"{FA}/mids_pp_full_fixed.json")

def per(d): return d["per_set"] if d else {}
def acc(ps,k):
    return ps[k]["acc"]*100 if k in ps else float('nan')
def sacc(ps, keys):
    a=[ps[k]["acc"] for k in keys if k in ps];
    if not a: return float('nan')
    m=sum(a)/len(a); return math.sqrt(sum((x-m)**2 for x in a)/len(a))*100
def meanacc(ps, keys):
    a=[ps[k]["acc"] for k in keys if k in ps]
    return sum(a)/len(a)*100 if a else float('nan')

# bucket key list (from any source)
buckets=[k for k in per(pp_mllm) if k!="eval2000"]
print("# Fixed answers vs MLLM answers (same trained models, same images)\n")
print("MIDS run with the 3 fixed templates from inference_mids.py (no MLLM) vs the real MLLM "
      "answers. eval2000 = hard 2000-item subset; sACC over the 10 mids_eval forgery buckets.\n")
print("| Model | answers | eval2000 ACC | buckets mean ACC | buckets sACC |")
print("|---|---|---|---|---|")
rows=[("Upstream MIDS","MLLM", per(up_mllm_e), per(up_mllm_b)),
      ("Upstream MIDS","FIXED", per(up_fix), per(up_fix)),
      ("MIDS++ full","MLLM", per(pp_mllm), per(pp_mllm)),
      ("MIDS++ full","FIXED", per(pp_fix), per(pp_fix))]
for name,kind,pe,pb in rows:
    print(f"| {name} | {kind} | {acc(pe,'eval2000'):.2f} | {meanacc(pb,buckets):.2f} | {sacc(pb,buckets):.2f} |")
print("\n## Per-bucket ACC (FIXED answers)")
print("| Model | " + " | ".join(buckets) + " |"); print("|"+" --- |"*(len(buckets)+1))
for name,d in [("Upstream FIXED",per(up_fix)),("MIDS++ FIXED",per(pp_fix))]:
    print(f"| {name} | " + " | ".join(f"{acc(d,b):.2f}" if b in d else "-" for b in buckets) + " |")
print("\n## Per-bucket ACC (MLLM answers, for reference)")
print("| Model | " + " | ".join(buckets) + " |"); print("|"+" --- |"*(len(buckets)+1))
for name,d in [("Upstream MLLM",per(up_mllm_b)),("MIDS++ MLLM",per(pp_mllm))]:
    print(f"| {name} | " + " | ".join(f"{acc(d,b):.2f}" if b in d else "-" for b in buckets) + " |")
PY
cat "$FA/COMPARISON_FIXED.md" >> "$LOG"
say "=== DONE_FIXED ==="
touch "$FA/DONE_FIXED"
