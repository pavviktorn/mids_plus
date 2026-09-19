#!/usr/bin/env bash
# Per-component ablation of MIDS++: train A1 (+SVD) and A2 (+SVD+GenD) on the FULL ~740k set with
# the same recipe as mids_pp_full, in PARALLEL on two GPUs. A0 (mids_original_full) and A3
# (mids_pp_full) already exist. Then eval A1/A2 on eval2000 + buckets and build an ablation report.
cd /datasets/work/vLLM/temp/mids_plus || exit 1
. .venv/bin/activate
export PYTHONPATH=src TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ABL=runs/real/abl
EXP=runs/real/exp
mkdir -p "$ABL/results"
LOG="$ABL/run.log"
FULL=/datasets/newout/vqa_info_2+13+4+3_fmt/mids.json
VAL1K=runs/real/val1000.json
EVAL2000=runs/real/eval2000.json
say(){ echo "$(date '+%F %T') | $*" | tee -a "$LOG"; }
ckpt(){ [ -f "$1/best.pt" ] && echo "$1/best.pt" || echo "$1/last.pt"; }

# two freest GPUs
mapfile -t GPUS < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2 -nr | head -2 | cut -d, -f1 | tr -d ' ')
GA=${GPUS[0]}; GB=${GPUS[1]:-${GPUS[0]}}
say "=== ablation started; A1 on GPU $GA, A2 on GPU $GB ==="

COMMON="train_data_path=$FULL val_data_path=$VAL1K skip_missing_images=true epochs=1 batch_size=12 val_batch_size=16 amp_dtype=bf16 num_workers=8 save_every_steps=2000"

# A1 = +SVD only (no GenD losses, no LayerNorm tune, no artifact)
CUDA_VISIBLE_DEVICES=$GA python -u -m mids_plus.train --config configs/mids_pp.yaml \
  --set $COMMON artifact_enabled=false alignment_weight=0 uniformity_weight=0 query_orth_weight=0 \
        tune_layer_norm=false output_dir=$ABL/svd_only > "$ABL/train_svd_only.log" 2>&1 &
PID1=$!
# A2 = +SVD +GenD (alignment/uniformity + LayerNorm tune on; no artifact)
CUDA_VISIBLE_DEVICES=$GB python -u -m mids_plus.train --config configs/mids_pp.yaml \
  --set $COMMON artifact_enabled=false query_orth_weight=0 \
        output_dir=$ABL/svd_gend > "$ABL/train_svd_gend.log" 2>&1 &
PID2=$!
say "A1 pid=$PID1  A2 pid=$PID2 ; waiting..."
wait $PID1; say "A1 (+SVD) train exit=$?"
wait $PID2; say "A2 (+SVD+GenD) train exit=$?"

# evals (sequential, on GPU $GA)
SPECS="eval2000:$EVAL2000"
for f in "$EXP"/buckets/*.json; do n=$(basename "$f" .json); SPECS="$SPECS $n:$f"; done
for tag in svd_only svd_gend; do
  if [ -f "$ABL/$tag/best.pt" -o -f "$ABL/$tag/last.pt" ]; then
    say "eval $tag"
    CUDA_VISIBLE_DEVICES=$GA python -u scripts/eval_models.py --model-type mids_plus \
      --checkpoint "$(ckpt $ABL/$tag)" --data $SPECS --batch-size 64 --num-workers 12 \
      --tag ${tag}_full --out "$ABL/results/${tag}_full.json" >> "$ABL/eval.log" 2>&1
    say "eval $tag exit=$?"
  fi
done

# ablation report (incremental contributions)
say "=== building ablation report ==="
python -u - <<'PY' > "$ABL/ABLATION_REPORT.md" 2>>"$ABL/eval.log"
import json, os, math
def load(p): return json.load(open(p)) if os.path.exists(p) else None
A0=load("runs/real/exp/results/mids_original_full_all.json")   # baseline
A1=load("runs/real/abl/results/svd_only_full.json")            # +SVD
A2=load("runs/real/abl/results/svd_gend_full.json")            # +SVD+GenD
A3=load("runs/real/exp/results/mids_pp_full_all.json")         # +SVD+GenD+Artifact = full
def per(d): return d["per_set"] if d else {}
def e2(d): return per(d).get("eval2000",{}).get("acc",float('nan'))*100
def buckets(d):
    return [k for k in per(d) if k!="eval2000"]
def sacc(d):
    ps=per(d); a=[ps[k]["acc"] for k in ps if k!="eval2000"]
    if not a: return float('nan')
    m=sum(a)/len(a); return math.sqrt(sum((x-m)**2 for x in a)/len(a))*100
def bmean(d):
    ps=per(d); a=[ps[k]["acc"] for k in ps if k!="eval2000"]
    return sum(a)/len(a)*100 if a else float('nan')
rows=[("A0 baseline (mids_original, unfreeze)",A0),
      ("A1  +Effort SVD",A1),
      ("A2  +SVD +GenD",A2),
      ("A3  +SVD +GenD +Artifact (=MIDS++)",A3)]
print("# MIDS++ per-component ablation (full ~740k, 1 epoch, same recipe)\n")
print("Incremental: each row adds one component. eval2000 = hard subset ACC; buckets = 10 mids_eval "
      "forgery partitions (mean ACC and sACC, lower sACC = better robustness).\n")
print("| Variant | eval2000 ACC | buckets mean ACC | buckets sACC |")
print("|---|---|---|---|")
for name,d in rows:
    if d is None: print(f"| {name} | (missing) | | |"); continue
    print(f"| {name} | {e2(d):.2f} | {bmean(d):.2f} | {sacc(d):.3f} |")
def delta(b,a,f):
    if a is None or b is None: return float('nan')
    return f(b)-f(a)
print("\n## Contribution of each part (Δ vs previous row)\n")
print("| Component | Δ eval2000 ACC | Δ buckets mean ACC | Δ buckets sACC |")
print("|---|---|---|---|")
for name,b,a in [("Effort SVD (A1−A0)",A1,A0),("GenD (A2−A1)",A2,A1),("Artifact (A3−A2)",A3,A2)]:
    print(f"| {name} | {delta(b,a,e2):+.2f} | {delta(b,a,bmean):+.2f} | {delta(b,a,sacc):+.3f} |")
print("\n(Δ sACC negative = more robust. Total A3−A0 is the full architecture contribution.)")
PY
cat "$ABL/ABLATION_REPORT.md" >> "$LOG"
say "=== ABLATION DONE ==="
touch "$ABL/DONE_ABL"
