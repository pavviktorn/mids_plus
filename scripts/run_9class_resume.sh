#!/usr/bin/env bash
# 9-class A0-A3, each WARM-STARTED (--resume) from its converged 4-class full model (backbone loads;
# 4->9 classifier head re-initialized). 3 epochs. ONLY GPUs 1 and 3 (two queues of two, sequential).
# Then eval all four on the 30K/46-subset held-out set and build the 9-vs-4 comparison.
cd /datasets/work/vLLM/temp/mids_plus || exit 1
. .venv/bin/activate
export PYTHONPATH=src TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NINE=runs/real/9class_resume
H=runs/real/heldout3
mkdir -p "$NINE/results9"
LOG="$NINE/run.log"
say(){ echo "$(date '+%F %T') | $*" | tee -a "$LOG"; }
ckpt(){ [ -f "$1/best.pt" ] && echo "$1/best.pt" || echo "$1/last.pt"; }

TR=runs/real/mids9/mids9.json; VA=runs/real/mids9/mids9_val.json
COMMON="train_data_path=$TR val_data_path=$VA num_classes=9 samples_per_image=3 skip_missing_images=true epochs=3 batch_size=12 val_batch_size=16 amp_dtype=bf16 num_workers=8 save_every_steps=2000"

# 4-class warm-start checkpoints
INIT_A0=runs/real/mids_original_full/best.pt
INIT_A1=runs/real/abl/svd_only/best.pt
INIT_A2=runs/real/abl/svd_gend/best.pt
INIT_A3=runs/real/mids_pp_full/best.pt

train_A0(){ CUDA_VISIBLE_DEVICES=$1 python -u -m mids_plus.train --config configs/mids_original.yaml \
  --set $COMMON output_dir=$NINE/A0_baseline --resume $INIT_A0 > "$NINE/train_A0.log" 2>&1; say "A0 train exit=$?"; }
train_A1(){ CUDA_VISIBLE_DEVICES=$1 python -u -m mids_plus.train --config configs/mids_pp.yaml \
  --set $COMMON artifact_enabled=false alignment_weight=0 uniformity_weight=0 query_orth_weight=0 tune_layer_norm=false \
  output_dir=$NINE/A1_svd --resume $INIT_A1 > "$NINE/train_A1.log" 2>&1; say "A1 train exit=$?"; }
train_A2(){ CUDA_VISIBLE_DEVICES=$1 python -u -m mids_plus.train --config configs/mids_pp.yaml \
  --set $COMMON artifact_enabled=false query_orth_weight=0 \
  output_dir=$NINE/A2_svdgend --resume $INIT_A2 > "$NINE/train_A2.log" 2>&1; say "A2 train exit=$?"; }
train_A3(){ CUDA_VISIBLE_DEVICES=$1 python -u -m mids_plus.train --config configs/mids_pp.yaml \
  --set $COMMON output_dir=$NINE/A3_midspp --resume $INIT_A3 > "$NINE/train_A3.log" 2>&1; say "A3 train exit=$?"; }

say "=== 9-class warm-start (3 epochs), ALL 4 SIMULTANEOUS: gpu1=A0+A2, gpu3=A1+A3 ==="
train_A0 1 & P0=$!
train_A2 1 & P2=$!
train_A1 3 & P1=$!
train_A3 3 & P3=$!
wait $P0; say "A0 done"
wait $P1; say "A1 done"
wait $P2; say "A2 done"
wait $P3; say "A3 done"

# eval all four (full-image, fixed-answer; 9-class auto) on GPU 1
SPECS=""
for f in "$H"/buckets/*.json; do n=$(basename "$f" .json); SPECS="$SPECS ${n}:$f"; done
for tag in A0_baseline A1_svd A2_svdgend A3_midspp; do
  if [ -f "$NINE/$tag/best.pt" -o -f "$NINE/$tag/last.pt" ]; then
    say "eval $tag (9-class)"
    CUDA_VISIBLE_DEVICES=1 python -u scripts/eval_fixed_answers.py --model-type mids_plus \
      --checkpoint "$(ckpt $NINE/$tag)" --full-image --data $SPECS --batch-size 64 --num-workers 8 \
      --tag "${tag}_9cR" --out "$NINE/results9/${tag}.json" >> "$NINE/eval.log" 2>&1
    say "  eval $tag exit=$?"
  fi
done

say "=== building 9-class(resume) vs 4-class comparison ==="
python -u - <<'PY' > "$NINE/NINE_VS_FOUR_REPORT.md" 2>>"$NINE/build.err"
import json, os, math
H="runs/real/heldout3"; NINE="runs/real/9class_resume/results9"
man=json.load(open(f"{H}/manifest.json"))
MODELS=[("A0 baseline","A0_baseline"),("A1 +SVD","A1_svd"),("A2 +SVD+GenD","A2_svdgend"),("A3 MIDS++","A3_midspp")]
def loadj(p): return json.load(open(p))["per_set"] if os.path.exists(p) else {}
def merged4(tag):
    real=loadj(f"{H}/results_realf/{tag}.json"); full=loadj(f"{H}/results/{tag}.json")
    m=dict(real)
    for k,v in full.items():
        if man.get(k,{}).get("group") in ("PAD","DEEPFAKE"): m[k]=v
    return m
def nine(tag): return loadj(f"{NINE}/{tag}.json")
def n(ps,k): return ps[k]["n"] if k in ps else 0
def grec(ps,g):
    ks=[s for s in man if man[s]["group"]==g]; tot=sum(n(ps,k) for k in ks)
    cor=sum(ps[k]["acc"]*ps[k]["n"] for k in ks if k in ps); return 100*cor/tot if tot else float('nan')
def frec(ps):
    ks=[s for s in man if man[s]["group"] in ("PAD","DEEPFAKE")]; tot=sum(n(ps,k) for k in ks)
    cor=sum(ps[k]["acc"]*ps[k]["n"] for k in ks if k in ps); return 100*cor/tot if tot else float('nan')
print("# 9-class (warm-started from 4-class, 3 epochs) vs 4-class — same 30K full-image held-out set\n")
print("9-class: 3 true x 3 claim, fixed-template claims, true via get_label_all (makeup->pad), each "
      "warm-started from its converged 4-class model and trained 3 epochs. Both eval full-image, "
      "MLLM-free, same buckets. PAD = anti-spoof recall (priority).\n")
print("| Model | scheme | PAD-rec | DEEPFAKE-rec | FAKE-rec | REAL-rec | Balanced ACC |")
print("|---|---|---|---|---|---|---|")
for nm,tag in MODELS:
    for scheme,ps in [("4-class",merged4(tag)),("9-class",nine(tag))]:
        if not ps: print(f"| {nm} | {scheme} | (missing) |"); continue
        pr,dr,fr,rr=grec(ps,"PAD"),grec(ps,"DEEPFAKE"),frec(ps),grec(ps,"REAL")
        print(f"| {nm} | {scheme} | {pr:.2f} | {dr:.2f} | {fr:.2f} | {rr:.2f} | {(rr+fr)/2:.2f} |")
print("\n## PAD per-attack-type recall (9-class)\n")
padsubs=sorted([s for s in man if man[s]["group"]=="PAD"], key=lambda s:-n(nine("A3_midspp"),s))
print("| attack | n | " + " | ".join(nm for nm,_ in MODELS) + " |")
print("|"+" --- |"*(len(MODELS)+2))
for s in padsubs:
    print(f"| {s} | {n(nine('A3_midspp'),s)} | " + " | ".join(f"{nine(t).get(s,{}).get('acc',float('nan'))*100:.2f}" if s in nine(t) else "-" for _,t in MODELS) + " |")
PY
cat "$NINE/NINE_VS_FOUR_REPORT.md" >> "$LOG"
say "=== 9CLASS_RESUME DONE ==="
touch "$NINE/DONE_9CLASS_RESUME"
