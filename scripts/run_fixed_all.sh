#!/usr/bin/env bash
# MLLM-free (fixed-answer) eval for ALL four full-trained models A0..A3, on eval2000 + 10 mids_eval
# buckets. Fixed-answer numbers for A0/A1/A2 are computed here; A3 (mids_pp_full) fixed is reused.
# MLLM-answer numbers for all four are reused from stored result JSONs. Builds a 4-way MLLM-vs-fixed
# comparison so we can attribute MLLM-free robustness to a component (SVD / GenD / Artifact).
cd /datasets/work/vLLM/temp/mids_plus || exit 1
. .venv/bin/activate
export PYTHONPATH=src TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

OUT=runs/real/exp_fixedans_all
EXP=runs/real/exp
ABL=runs/real/abl
mkdir -p "$OUT/results"
LOG="$OUT/run.log"
EVAL2000=runs/real/eval2000.json
say(){ echo "$(date '+%F %T') | $*" | tee -a "$LOG"; }
ckpt(){ [ -f "$1/best.pt" ] && echo "$1/best.pt" || echo "$1/last.pt"; }

# freest GPU
GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -1 | cut -d, -f1 | tr -d ' ')
say "=== MLLM-free eval for A0..A3 started on GPU $GPU ==="

SPECS="eval2000:$EVAL2000"
for f in "$EXP"/buckets/*.json; do n=$(basename "$f" .json); SPECS="$SPECS $n:$f"; done

# A0/A1/A2 checkpoints (A3 fixed already exists, reused below). All load via mids_plus load_checkpoint.
declare -A CK=( [A0_mids_original_full]="runs/real/mids_original_full" \
                [A1_svd_only]="$ABL/svd_only" \
                [A2_svd_gend]="$ABL/svd_gend" )

for tag in A0_mids_original_full A1_svd_only A2_svd_gend; do
  say "=== $tag, FIXED answers ==="
  CUDA_VISIBLE_DEVICES=$GPU python -u scripts/eval_fixed_answers.py --model-type mids_plus \
    --checkpoint "$(ckpt ${CK[$tag]})" --data $SPECS --batch-size 64 --num-workers 12 \
    --tag ${tag}_fixed --out "$OUT/results/${tag}_fixed.json" >> "$OUT/eval.log" 2>&1
  say "$tag fixed exit=$?"
done

# Reuse A3 fixed (already computed) into this folder's results for a single report source.
cp -f runs/real/exp_fixedans/results/mids_pp_full_fixed.json "$OUT/results/A3_mids_pp_full_fixed.json" 2>/dev/null

say "=== building 4-way comparison (MLLM vs FIXED) ==="
python -u - <<'PY' > "$OUT/COMPARISON_FIXED_ALL.md" 2>>"$OUT/eval.log"
import json, os, math
def load(p): return json.load(open(p)) if os.path.exists(p) else None
# MLLM-answer (stored) and FIXED-answer (new) result files, per model
MLLM = {
 "A0 baseline (unfreeze)": "runs/real/exp/results/mids_original_full_all.json",
 "A1 +SVD":                "runs/real/abl/results/svd_only_full.json",
 "A2 +SVD+GenD":           "runs/real/abl/results/svd_gend_full.json",
 "A3 +Artifact (=MIDS++)": "runs/real/exp/results/mids_pp_full_all.json",
}
FIX = {
 "A0 baseline (unfreeze)": "runs/real/exp_fixedans_all/results/A0_mids_original_full_fixed.json",
 "A1 +SVD":                "runs/real/exp_fixedans_all/results/A1_svd_only_fixed.json",
 "A2 +SVD+GenD":           "runs/real/exp_fixedans_all/results/A2_svd_gend_fixed.json",
 "A3 +Artifact (=MIDS++)": "runs/real/exp_fixedans_all/results/A3_mids_pp_full_fixed.json",
}
ORDER = list(MLLM.keys())
def per(d): return d["per_set"] if d else {}
def e2(ps): return ps.get("eval2000",{}).get("acc",float('nan'))*100
def bkeys(ps): return [k for k in ps if k!="eval2000"]
def sacc(ps):
    a=[ps[k]["acc"] for k in ps if k!="eval2000"]
    if not a: return float('nan')
    m=sum(a)/len(a); return math.sqrt(sum((x-m)**2 for x in a)/len(a))*100
def bmean(ps):
    a=[ps[k]["acc"] for k in ps if k!="eval2000"]
    return sum(a)/len(a)*100 if a else float('nan')

M={k:per(load(v)) for k,v in MLLM.items()}
F={k:per(load(v)) for k,v in FIX.items()}

print("# MLLM-free (fixed-answer) inference across all 4 full-trained models\n")
print("Each model decided two ways on the SAME images: with real MLLM-generated candidate answers "
      "(MLLM) and with the 3 fixed templates from inference_mids.py (FIXED, no MLLM; makeup=PAD). "
      "eval2000 = hard in-distribution subset; buckets = 10 mids_eval forgery partitions "
      "(mean ACC and sACC; lower sACC = more robust across forgeries).\n")
print("| Model | eval2000 MLLM | eval2000 FIXED | Δe2 | buckets-mean MLLM | buckets-mean FIXED | Δmean | sACC MLLM | sACC FIXED | ΔsACC |")
print("|---|---|---|---|---|---|---|---|---|---|")
for k in ORDER:
    m,f=M[k],F[k]
    if not m or not f: print(f"| {k} | (missing) |"); continue
    print(f"| {k} | {e2(m):.2f} | {e2(f):.2f} | {e2(f)-e2(m):+.2f} | {bmean(m):.2f} | {bmean(f):.2f} | "
          f"{bmean(f)-bmean(m):+.2f} | {sacc(m):.3f} | {sacc(f):.3f} | {sacc(f)-sacc(m):+.3f} |")

# per-bucket FIXED table
allb=[]
for k in ORDER:
    for b in bkeys(F[k]):
        if b not in allb: allb.append(b)
print("\n## Per-bucket ACC — FIXED answers (MLLM-free)")
print("| Model | " + " | ".join(allb) + " |")
print("|" + " --- |"*(len(allb)+1))
for k in ORDER:
    f=F[k]
    print(f"| {k} | " + " | ".join(f"{f[b]['acc']*100:.2f}" if b in f else "-" for b in allb) + " |")

print("\n## Per-bucket ACC — MLLM answers (reference)")
print("| Model | " + " | ".join(allb) + " |")
print("|" + " --- |"*(len(allb)+1))
for k in ORDER:
    m=M[k]
    print(f"| {k} | " + " | ".join(f"{m[b]['acc']*100:.2f}" if b in m else "-" for b in allb) + " |")

print("\n## MLLM-free robustness = FIXED − MLLM (negative = degrades when MLLM removed)")
print("| Model | Δ eval2000 | Δ buckets-mean | Δ sACC |")
print("|---|---|---|---|")
for k in ORDER:
    m,f=M[k],F[k]
    print(f"| {k} | {e2(f)-e2(m):+.2f} | {bmean(f)-bmean(m):+.2f} | {sacc(f)-sacc(m):+.3f} |")
PY
cat "$OUT/COMPARISON_FIXED_ALL.md" >> "$LOG"
say "=== ALL FIXED DONE ==="
touch "$OUT/DONE_FIXED_ALL"
