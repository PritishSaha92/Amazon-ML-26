#!/bin/bash
# v12 CE arms, run on the node AFTER the clean chain's 'average' step (needs data/oof_scores_v12avg.parquet).
# NEW 2-fold CE over ALL 2,126,821 train-split refs (fold = md5(s1_id) int parity, even -> A, odd -> B: the v7 rule, which
# reproduces data/cefold_v7/folds.parquet exactly), trained FROM PRETRAINED weights (never warm-started from ce_v7f),
# ce_v7f hyper-parameters (lr 3e-5 = ce_v7fA/train_meta.json, batch 96, max-len 96, pos-dup 1, drop syn, 1 epoch, bf16, seed 0),
# ce_v4_hard pool recipe with 5x the DISTINCT hard negatives (band all; shift/sib/hard/medium x5; random fills n-neg x5).
# The training sets (data/v12/cefold2/ce_train_hard_v12f2{A,B}.parquet) are built ONCE and shared by every arm.
# Routing of an arm's scores (independent of the old 400k A/B routing): train ref in new fold A -> the arm's model B only,
# fold B -> A only; val/audit/test band rows -> mean(A, B). Same stage-1 scores, band (0.02-0.98 of v12avg p), band cache
# and decoder as v12ce. Stage 2 of every variant runs with STAGE2_WITHHOLD_AUDIT=1 and is evaluated with
# `v12_eval.py --tag <variant> --ref-tag v12ce --splits val` only (selection protocol; audit only for the locked winner).
#   ARM=v bash cloud/ce2_arm.sh   vanilla e5-base init           GPUs 0,1 -> ce_v12f2{A,B}   data/ce2_oof_v12.parquet    variants v12ce2 (new only), v12ce3 (old+new)
#   ARM=d bash cloud/ce2_arm.sh   DAPT init (models/e5_dapt_v12) GPUs 2,3 -> ce_v12f2d{A,B} data/ce2d_oof_v12.parquet   v12ce3d only (frozen shortlist)
#   ARM=f bash cloud/ce2_arm.sh   field template (cloud/*_field.py) GPUs 6,7 -> ce_v12f2f{A,B} data/ce2f_oof_v12.parquet v12ce5 (field only), v12ce4 (old+field)
#   options: [--limit N (cap training pairs per fold; the SAME N for every arm)] [--only STEP] [--from STEP] [--score-gpus "4 5 6 7"]
#   env: MAXLEN (default 96), S2TH (stage-2 threads, default 40)
# Steps: folds build (shared)  train score assemble stage2 tar (per arm; markers logs/v12ce2/<ARM>/.done_<step>; rerun = resume)
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
P=$HOME/v12env/bin/python; export HF_HOME=$HOME/hf TOKENIZERS_PARALLELISM=true
ARM=${ARM:-v}; D=data/v12/cefold2; LS=logs/v12ce2; L=$LS/$ARM; mkdir -p $L $D output models
E5=$HOME/hf/hub/models--intfloat--multilingual-e5-base/snapshots/d128750597153bb5987e10b1c3493a34e5a4502a
[ -f $E5/model.safetensors ] || { echo "missing $E5: run cloud/env_setup.sh (pinned model download)"; exit 1; }  # packaged copy
LIMIT=0; ONLY=""; FROM=""; SCORE_GPUS="4 5 6 7"
while [ $# -gt 0 ]; do case "$1" in --limit) LIMIT=$2; shift;; --only) ONLY=$2; shift;; --from) FROM=$2; shift;; --score-gpus) SCORE_GPUS=$2; shift;; *) echo "bad arg $1"; exit 2;; esac; shift; done
TRAINER=src/train_ce_text.py; SCORER=src/score_cross_encoder.py; INIT=$E5
case $ARM in
  v) SFX="";  GA=0; GB=1; S2NEW=v12ce2;  S2OLD=v12ce3 ;;
  d) SFX=d;   GA=2; GB=3; S2NEW="";      S2OLD=v12ce3d; INIT=models/e5_dapt_v12 ;;
  f) SFX=f;   GA=6; GB=7; S2NEW=v12ce5;  S2OLD=v12ce4; TRAINER=cloud/train_ce_field.py; SCORER=cloud/score_ce_field.py ;;
  *) echo "ARM must be v, d or f"; exit 2 ;;
esac
MA=models/ce_v12f2${SFX}A; MB=models/ce_v12f2${SFX}B; PF=data/ce2${SFX}_oof_v12.parquet; PT=output/ce2${SFX}_test_v12avg.parquet; PP=$D/pce2${SFX}
STEPS="folds build train score assemble stage2 tar"
say() { echo "[$(date '+%F %H:%M:%S')] [arm $ARM] $*"; }
fail() { say "FAILED: $*"; echo "$(date '+%F %T') $*" > $L/FAILED; exit 1; }
run() { local log=$1; shift; say "START $* > $log"; /usr/bin/time -v "$@" > "$log" 2>&1 || fail "$* (exit $?; see $log)"; grep -E "Elapsed \(wall|Maximum resident" "$log" | sed 's/^\s*/    /'; }
mark() { case $1 in folds|build) echo $LS/.done_$1;; *) echo $L/.done_$1;; esac; }
want() { local s=$1; if [ -n "$ONLY" ]; then [ "$s" = "$ONLY" ]; return; fi
  if [ -n "$FROM" ]; then [[ " ${STEPS#*$FROM} " == *" $s "* ]] || [ "$s" = "$FROM" ] || return 1; fi
  [ -f $(mark $s) ] && { say "skip $s"; return 1; }; return 0; }
done_step() { date '+%F %T' > $(mark $1); }
[ -f data/oof_scores_v12avg.parquet ] && [ -f logs/v12/.done_average ] || fail "clean chain has not finished 'average' yet"
NEGMIX="band=1.0,shift=2250000,sib=1500000,hard=1500000,medium=400000,random=-1"; NNEG=6500000   # v7f: 450k/300k/300k/80k, n-neg 1.3M

if want folds; then say "==== folds (shared)"
  run $LS/folds.log $P - <<'PY'
import hashlib, polars as pl
D = "data/v12/cefold2"
sp = pl.read_parquet("data/split.parquet").filter(pl.col("split") == "train").select("s1_id", "country")
sp = sp.with_columns(fold=pl.Series([("A" if int(hashlib.md5(s.encode()).hexdigest(), 16) % 2 == 0 else "B") for s in sp["s1_id"]]))
import os; old = pl.read_parquet("data/cefold_v7/folds.parquet") if os.path.exists("data/cefold_v7/folds.parquet") else sp.select("s1_id", "fold")  # packaged copy: optional cross-check
chk = old.join(sp, on="s1_id", how="left", suffix="_new"); assert chk["fold_new"].null_count() == 0 and (chk["fold"] == chk["fold_new"]).all(), "md5 rule != v7 folds"
sp.select("s1_id", "fold").write_parquet(f"{D}/folds.parquet")
for f in ("A", "B"): sp.filter(pl.col("fold") == f).select("s1_id").write_parquet(f"{D}/s1_{f}.parquet")
print("train refs", len(sp)); print(sp.group_by("fold", "country").len().sort("fold", "country"))
print("v7 400k folds reproduced exactly by the md5 rule:", len(old))
PY
  cat $LS/folds.log | grep -vE "^\s" | tail -12; done_step folds; fi

if want build; then say "==== build CE training sets (shared, 2 in parallel) + manifests"
  [ -f $(mark folds) ] || fail "folds not done"
  for F in A B; do
    ( run $LS/build_$F.log $P src/build_ce_train_hard.py --pairs data/pairs_train_v12_feat.parquet --oof data/oof_scores_v12avg.parquet \
        --records data/train_records.parquet --s1-ids $D/s1_$F.parquet --out $D/ce_train_hard_v12f2$F.parquet --n-neg $NNEG --n-syn 0 --neg-mix "$NEGMIX" ) & eval "pb_$F=\$!"
  done
  wait $pb_A || fail "build A"; wait $pb_B || fail "build B"
  run $LS/manifest.log $P - <<'PY'
import polars as pl, hashlib, json
D = "data/v12/cefold2"; man = {}
for F in "AB":
    p = f"{D}/ce_train_hard_v12f2{F}.parquet"; d = pl.read_parquet(p, columns=["s1_id", "t_id", "label", "pool"])
    d.write_parquet(f"{D}/manifest_train_{F}.parquet")
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    man[F] = dict(file=p, sha256=h, pairs=len(d), unique_owners=d["s1_id"].n_unique(), pos=int(d["label"].sum()),
                  pools={f"{k}|{l}": n for k, l, n in d.group_by("pool", "label").len().sort("pool").iter_rows()},
                  exposure_after_pos_dup1=len(d) + int((d["pool"] == "pos_hard").sum()))
    print(F, json.dumps(man[F]))
json.dump(man, open(f"{D}/manifest.json", "w"), indent=1)
PY
  cat $LS/manifest.log | grep -E '^[AB] '; done_step build; fi

if want train; then say "==== train the $ARM pair in parallel (GPU $GA = A, GPU $GB = B); init $INIT; limit=$LIMIT"
  [ -f $(mark build) ] || fail "training sets not built"
  [ -e "$INIT" ] || fail "init $INIT missing"
  LIM=""; [ "$LIMIT" -gt 0 ] && LIM="--limit $LIMIT"
  XTRA=""; [ $ARM = f ] && XTRA="--records data/train_records.parquet"
  # wait until both GPUs are free (another agent's screens may hold them): < 2 GB used, checked every 60 s
  gpu_used() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $1 | tr -dc 0-9; }
  while [ "$(gpu_used $GA)" -gt 2000 ] || [ "$(gpu_used $GB)" -gt 2000 ]; do sleep 60; done; say "GPUs $GA/$GB free"
  for F in A B; do G=$([ $F = A ] && echo $GA || echo $GB)
    ( CUDA_VISIBLE_DEVICES=$G run $L/train_$F.log $P $TRAINER --data $D/ce_train_hard_v12f2$F.parquet --drop-pools syn --pos-dup 1 \
        --tag v12f2$SFX$F --batch 96 --max-len ${MAXLEN:-96} --lr 3e-5 --epochs 1 --seed 0 --model $INIT --threads 12 $LIM $XTRA ) & eval "pt_$F=\$!"
  done
  wait $pt_A || fail "train A"; wait $pt_B || fail "train B"
  grep -h "TRAIN DONE" $L/train_A.log $L/train_B.log; done_step train; fi

# band = 0.02 <= p <= 0.98 of the v12avg stage-1 scores (identical to the v12ce band)
if want score; then say "==== score the band with the $ARM pair (GPUs $SCORE_GPUS)"
  # test band is REQUIRED: wait for the clean chain's test step (up to 3 h), else fail loudly (never skip test scoring silently)
  for i in $(seq 1 180); do [ -f logs/v12/.done_test ] && [ -f output/test_scores_v12avg.parquet ] && break; sleep 60; done
  [ -f logs/v12/.done_test ] && [ -f output/test_scores_v12avg.parquet ] || fail "output/test_scores_v12avg.parquet missing after 3 h wait (clean chain test step)"
  $P - <<'PY' > $L/score_plan.log 2>&1 || fail "score plan"
import polars as pl, os
D = "data/v12/cefold2"; K = ["s1_id", "t_id"]
b = pl.read_parquet("data/oof_scores_v12avg.parquet", columns=K + ["split", "p"]).filter(pl.col("p").is_between(0.02, 0.98))
f = pl.read_parquet(f"{D}/folds.parquet"); b = b.join(f, on="s1_id", how="left")
assert b.filter((pl.col("split") == "train") & pl.col("fold").is_null()).is_empty(), "train band ref without a fold"
assert b.filter((pl.col("split") != "train") & pl.col("fold").is_not_null()).is_empty(), "val/audit ref in a fold"
b.filter(pl.col("fold") == "B").select(K + ["split", "p"]).write_parquet(f"{D}/band_train_B.parquet")   # scored by model A
b.filter(pl.col("fold") == "A").select(K + ["split", "p"]).write_parquet(f"{D}/band_train_A.parquet")   # scored by model B
b.filter(pl.col("split") != "train").select(K + ["split", "p"]).write_parquet(f"{D}/band_valaudit.parquet")
print(b.with_columns(grp=pl.when(pl.col("split") != "train").then(pl.col("split")).otherwise(pl.lit("train_") + pl.col("fold"))).group_by("grp").len().sort("grp"))
assert os.path.exists("output/test_scores_v12avg.parquet"), "test scores missing"
if True:
    t = pl.read_parquet("output/test_scores_v12avg.parquet", columns=K + ["p"]).filter(pl.col("p").is_between(0.02, 0.98))
    t.write_parquet(f"{D}/band_test.parquet"); print("test band", len(t))
PY
  cat $L/score_plan.log | grep -vE "^\s" | tail -12
  set -- $SCORE_GPUS; G1=$1; G2=${2:-$1}; G3=${3:-$1}; G4=${4:-$2}
  S="$SCORER --threads 12 --max-len ${MAXLEN:-96}"
  ( CUDA_VISIBLE_DEVICES=$G1 run $L/score_trainB_byA.log $P $S --model $MA --pairs $D/band_train_B.parquet --records data/train_records.parquet --out ${PP}_train_B.parquet ) & p1=$!
  ( CUDA_VISIBLE_DEVICES=$G2 run $L/score_trainA_byB.log $P $S --model $MB --pairs $D/band_train_A.parquet --records data/train_records.parquet --out ${PP}_train_A.parquet ) & p2=$!
  ( CUDA_VISIBLE_DEVICES=$G3 run $L/score_va_A.log $P $S --model $MA --pairs $D/band_valaudit.parquet --records data/train_records.parquet --out ${PP}_va_A.parquet
    [ -f $D/band_test.parquet ] || fail "band_test missing"; if true; then CUDA_VISIBLE_DEVICES=$G3 run $L/score_test_A.log $P $S --model $MA --pairs $D/band_test.parquet --records data/test_records.parquet --out ${PP}_test_A.parquet; fi ) & p3=$!
  ( CUDA_VISIBLE_DEVICES=$G4 run $L/score_va_B.log $P $S --model $MB --pairs $D/band_valaudit.parquet --records data/train_records.parquet --out ${PP}_va_B.parquet
    [ -f $D/band_test.parquet ] || fail "band_test missing"; if true; then CUDA_VISIBLE_DEVICES=$G4 run $L/score_test_B.log $P $S --model $MB --pairs $D/band_test.parquet --records data/test_records.parquet --out ${PP}_test_B.parquet; fi ) & p4=$!
  for p in $p1 $p2 $p3 $p4; do wait $p || fail "scoring"; done
  done_step score; fi

if want assemble; then say "==== assemble + coverage -> $PF"
  export PP PF PT
  run $L/assemble.log $P - <<'PY'
import polars as pl, os, json
D = "data/v12/cefold2"; K = ["s1_id", "t_id"]; PP, PF, PT = os.environ["PP"], os.environ["PF"], os.environ["PT"]
r = lambda f: pl.read_parquet(f"{PP}_{f}.parquet", columns=K + ["p_ce"])
va = r("va_A").rename({"p_ce": "a"}).join(r("va_B").rename({"p_ce": "b"}), on=K, how="full", coalesce=True)
assert va["a"].null_count() == 0 and va["b"].null_count() == 0
va = va.with_columns(p_ce=((pl.col("a") + pl.col("b")) / 2).cast(pl.Float32)).select(K + ["p_ce"])
x = pl.concat([r("train_A"), r("train_B"), va]).with_columns(pl.col("p_ce").cast(pl.Float32))
assert x.select(K).is_duplicated().sum() == 0
b = pl.read_parquet("data/oof_scores_v12avg.parquet", columns=K + ["split", "p"]).filter(pl.col("p").is_between(0.02, 0.98))
f = pl.read_parquet(f"{D}/folds.parquet"); j = b.join(f, on="s1_id", how="left").join(x, on=K, how="left")
miss = j["p_ce"].null_count(); extra = len(x) - (len(j) - miss)
grp = j.with_columns(grp=pl.when(pl.col("split") != "train").then(pl.col("split")).otherwise(pl.lit("train_fold") + pl.col("fold")),
                     route=pl.when(pl.col("split") != "train").then(pl.lit("mean(A,B)")).when(pl.col("fold") == "A").then(pl.lit("B only")).otherwise(pl.lit("A only")))
tab = grp.group_by("grp", "route").agg(pl.len().alias("rows"), pl.col("s1_id").n_unique().alias("refs"), pl.col("p_ce").null_count().alias("missing"), pl.col("p_ce").mean().alias("mean_p_ce")).sort("grp")
print(tab); assert miss == 0 and extra == 0, (miss, extra)
x.write_parquet(PF); print("wrote", PF, len(x))
rep = dict(band_rows=len(j), assembled=len(x), missing=miss, extra=extra, groups=tab.to_dicts())
assert os.path.exists(f"{PP}_test_A.parquet") and os.path.exists(f"{PP}_test_B.parquet"), "test CE scores missing"
if True:
    t = r("test_A").rename({"p_ce": "a"}).join(r("test_B").rename({"p_ce": "b"}), on=K, how="full", coalesce=True)
    bt = pl.read_parquet(f"{D}/band_test.parquet", columns=K); assert len(t) == len(bt) and t["a"].null_count() == 0 and t["b"].null_count() == 0
    t = t.with_columns(p_ce=((pl.col("a") + pl.col("b")) / 2).cast(pl.Float32)).select(K + ["p_ce"]); t.write_parquet(PT)
    print("wrote", PT, len(t)); rep["test_rows"] = len(t)
json.dump(rep, open(f"{PP}_coverage.json", "w"), indent=1, default=str)
PY
  cat $L/assemble.log | grep -vE "^\s+[A-Z]" | tail -20; done_step assemble; fi

if want stage2; then say "==== stage 2 $S2NEW (new CE only) || $S2OLD (old CE + new CE); withheld audit; val-only eval vs v12ce"
  until [ -f logs/v12/.done_stage2 ]; do sleep 60; done        # the clean v12ce fit comes first
  S2="src/stage2.py --pairs data/pairs_train_v12_feat.parquet --scores data/oof_scores_v12avg.parquet --model models/lgbm_v12avg --band-cache data/s2_band_v12avg.parquet"
  [ -f data/s2_band_v12avg.parquet ] || fail "band cache of the clean chain missing"
  # 3 seeds (0 = tag itself, 1, 2) per variant, in parallel; p_s2 averaged -> <tag>_s3avg (protocol)
  for T in $S2NEW $S2OLD; do
    if [ $T = "$S2NEW" ]; then CE="--ce $PF"; else CE="--ce data/ce_oof_v12.parquet --ce2 $PF"; fi
    pids=""
    for SD in 0 1 2; do TT=$T; [ $SD -gt 0 ] && TT=${T}_seed$SD
      ( STAGE2_WITHHOLD_AUDIT=1 run $L/stage2_$TT.log $P $S2 --tag $TT $CE --seed $SD --threads ${S2TH:-40} ) & pids="$pids $!"
    done
    for p in $pids; do wait $p || fail "stage2 $T"; done
    $P cloud/s2avg.py --tag $T > $L/s2avg_$T.log 2>&1 || fail "s2avg $T"
  done
  until [ -f output/val_scores_s2_v12ce_s3avg.parquet ]; do sleep 60; done   # 3-seed clean control (cloud/clean_s3.sh)
  for T in $S2NEW $S2OLD; do
    for X in $T ${T}_s3avg; do $P cloud/v12_eval.py --tag $X --ref-tag $([ $X = $T ] && echo v12ce || echo v12ce_s3avg) --splits val > $L/eval_val_$X.log 2>&1; echo "== $X"; tail -n 2 $L/eval_val_$X.log; done
    md5sum models/stage2_$T.txt models/stage2_${T}_seed1.txt models/stage2_${T}_seed2.txt
  done
  done_step stage2; fi

if want tar; then say "==== return bundle cloud_v12${ARM}_return.tar"
  RET="$MA $MB $PF ${PP}_train_A.parquet ${PP}_train_B.parquet ${PP}_va_A.parquet ${PP}_va_B.parquet ${PP}_coverage.json
       $D/folds.parquet $D/s1_A.parquet $D/s1_B.parquet $D/manifest.json $D/manifest_train_A.parquet $D/manifest_train_B.parquet $L $LS/folds.log $LS/build_A.log $LS/build_B.log $LS/manifest.log"
  for T in $S2NEW $S2OLD; do for X in $T ${T}_seed1 ${T}_seed2; do RET="$RET models/stage2_$X.txt models/stage2_$X.json output/val_scores_s2_$X.parquet output/audit_scores_s2_$X.parquet"; done
    RET="$RET output/val_scores_s2_${T}_s3avg.parquet output/audit_scores_s2_${T}_s3avg.parquet"; done
  [ -f $PT ] || fail "test CE file $PT missing"; RET="$RET $PT ${PP}_test_A.parquet ${PP}_test_B.parquet"
  [ $ARM = d ] && RET="$RET models/e5_dapt_v12"
  for f in $RET; do [ -e "$f" ] || fail "missing $f"; done
  tar -cf cloud_v12${ARM}_return.tar $RET || fail tar
  md5sum cloud_v12${ARM}_return.tar > cloud_v12${ARM}_return.md5; ls -la cloud_v12${ARM}_return.tar; cat cloud_v12${ARM}_return.md5
  done_step tar; say "ARM $ARM DONE (the steward pulls the bundle; ALL_DONE is written by the steward)"; fi
exit 0
