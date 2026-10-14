#!/bin/bash
# v12 = champion recipe (v7 stage 1 A/B -> 5-fold OOF avg -> fold-CE with ownership routing -> stage 2) on ALL ~2.13M labelled
# train-split references (v7 file: 400k train refs + 80k val/audit; + the 1,726,821 complement refs built here). Clean: no test
# labels, val/audit never trained on, same hyper-parameters / fold scheme / CE band / decoder as v7ce and v11.
# Target: Lambda gpu_8x_a100 (124 vCPU, 1.8 TB RAM, 6 TB disk). Run from the project root on the node:
#   bash cloud/chain_v12_cloud.sh [--dry-run] [--with-test] [--from STEP] [--only STEP] [--ncpu N] [--cache DIR]
#                                 [--fold-par N (concurrent OOF folds per model, default 5)] [--seq-models (A then B; small-RAM nodes)]
# Steps (resumable: a step whose marker logs/v12/.done_<step> exists is skipped):
#   preflight complement build_pairs shard features concat prep stage1 average ce stage2 slim test tar
# Every heavy command runs under /usr/bin/time -v with its own log in logs/v12/. On any failure: logs/v12/FAILED is written and
# the logs are tarred (cloud_v12_logs.tar) so they can be pulled before the node is TERMINATED (see cloud/lambda_launch.md).
TEST_PAIRS=${TEST_PAIRS:-data/pairs_test_v10w_feat.parquet}   # test features with the India/US train-table word-LR columns (v10w, public 0.988684)
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1
ROOT=$(pwd)
DRY=0; WITH_TEST=0; FROM=""; ONLY=""; NCPU=$(nproc); CACHE=""; FOLD_PAR=5; SEQ=0
usage() { sed -n 2,12p "$0"; exit "${1:-0}"; }
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY=1 ;; --with-test) WITH_TEST=1 ;;
    --from) FROM=${2:?}; shift ;; --only) ONLY=${2:?}; shift ;;
    --ncpu) NCPU=${2:?}; shift ;; --cache) CACHE=${2:?}; shift ;;
    --fold-par) FOLD_PAR=${2:?}; shift ;; --seq-models) SEQ=1 ;;
    -h|--help) usage 0 ;; *) echo "unknown argument: $1"; usage 2 ;;
  esac; shift
done
STEPS="preflight complement build_pairs shard features concat prep stage1 average ce stage2 slim test tar"
for s in $FROM $ONLY; do [[ " $STEPS " == *" $s "* ]] || { echo "unknown step '$s' (steps: $STEPS)"; exit 2; }; done
[[ "$NCPU" =~ ^[0-9]+$ ]] && [ "$NCPU" -ge 8 ] || { echo "--ncpu must be an integer >= 8"; exit 2; }
[[ "$FOLD_PAR" =~ ^[1-5]$ ]] || { echo "--fold-par must be 1..5"; exit 2; }

VENV=${VENV:-$HOME/v12env}; P=$VENV/bin/python; [ $DRY = 1 ] && [ ! -x "$P" ] && P=python
CACHE=${CACHE:-$ROOT/cache_v12}
L=logs/v12; mkdir -p $L data/v12 output models
export HF_HOME=$HOME/hf TOKENIZERS_PARALLELISM=true  # packaged copy: node model cache written by cloud/env_setup.sh
# thread/RAM budget. Default (124 vCPU, 1.8 TB): A and B side by side (62 threads each for the full fits), then 5+5 OOF folds
# side by side (12 threads each). --seq-models runs A then B; --fold-par caps concurrent folds per model.
NM=$((2 - SEQ)); TH_FULL=$((NCPU / NM)); TH_FOLD=$((NCPU / (NM * FOLD_PAR))); [ $TH_FOLD -ge 1 ] || TH_FOLD=1
NSHARD=8; TH_SHARD=$((NCPU / NSHARD)); [ $TH_SHARD -ge 2 ] || TH_SHARD=2
MIN_RAM=$((150 + 45 * NM * FOLD_PAR))   # GB: page cache for the 66 GB memmaps + ~45 GB per concurrent fold process
RULE="(pl.col('n_views') >= 2) | (pl.col('fwd_rank') <= 8) | (pl.col('rdense_rank') <= 4) | (pl.col('rfull_rank') <= 2)"
HPA="--rounds 1500"                                                                  # v7 A = LightGBM defaults of stage1_lean
HPB="--leaves 255 --lr 0.03 --ff 0.7 --bf 0.8 --min-leaf 50 --l2 1.0 --seed 7"        # v7 B (rounds 3000)
CANDS=data/cands_sparsev_train.parquet,data/cands_dense_train.parquet,data/cands_rdense_train.parquet,data/cands_rsparse_train.parquet

say() { echo "[$(date '+%F %H:%M:%S')] $*"; }
fail() { say "FAILED: $*"; echo "$(date '+%F %T') $*" > $L/FAILED; tar -cf cloud_v12_logs.tar logs/v12 2>/dev/null; kill 0 2>/dev/null; exit 1; }
# run LOG cmd... : GNU time -v into the step log; abort chain on failure
run() { local log=$1; shift
  if [ $DRY = 1 ]; then echo "  + $* > $log"; return 0; fi
  say "START $* > $log"; /usr/bin/time -v "$@" > "$log" 2>&1 || fail "$* (exit $?; see $log)"
  grep -E "Elapsed \(wall|Maximum resident" "$log" | sed 's/^\s*/    /'; }
want() { local s=$1  # should step s run?
  if [ -n "$ONLY" ]; then [ "$s" = "$ONLY" ]; return; fi
  if [ -n "$FROM" ]; then [[ " ${STEPS#*$FROM} " == *" $s "* ]] || [ "$s" = "$FROM" ] || return 1; fi
  if [ $DRY = 0 ] && [ -f $L/.done_$s ]; then say "skip $s (done $(cat $L/.done_$s))"; return 1; fi; return 0; }
done_step() { [ $DRY = 1 ] || date '+%F %T' > $L/.done_$1; }
hdr() { echo; say "==== step $1: $2"; }

if [ $DRY = 1 ]; then cat <<EST
DRY RUN on $(hostname): NCPU=$NCPU models-concurrent=$NM TH_FULL=$TH_FULL fold-par=$FOLD_PAR TH_FOLD=$TH_FOLD shards=$NSHARD x $TH_SHARD workers; min RAM ${MIN_RAM} GB; cache=$CACHE; with-test=$WITH_TEST
Estimated sizes (from v7/v11 row rates: 54.1 pairs/ref, v7 feat 2.99 GB per 25.9M rows, 138 features + country):
  complement refs            1,726,821 (India ~691k / US ~1,036k)
  data/v12/pairs_train_v12new.parquet   ~93.4M rows  ~6.4 GB     (build_pairs, ~5-8 min, ~120 GB RSS)
  data/v12/shards/*_feat.parquet        ~93.4M rows  ~10.8 GB    (8 parallel pair_features, ~6-10 min, ~8x16 GB RSS)
  data/pairs_train_v12_feat.parquet     ~119.3M rows ~13.8 GB    (train ~115.0M + val/audit 4.31M)
  $CACHE/X_train.npy                    115.0M x 139 f32 ~64 GB; X_va.npy ~2.4 GB (prep ~25-35 min, single stream)
  stage 1 full A || full B              ~15-25 / 25-45 min at $TH_FULL threads each; binned Dataset ~20-30 GB RSS each
  OOF 10 folds in parallel              ~25-60 min at $TH_FOLD threads each; ~35-45 GB RSS each (~450 GB total)
  data/oof_scores_v12{,b,avg}.parquet   ~1.2 GB each
  CE band (0.02-0.98 of v12avg OOF p)   ~1.1-1.3M rows; ~0.9M need A and/or B scores (2 GPUs, ~3-5 min)
  data/s2_band_v12avg.parquet           ~8.7M rows ~0.6 GB; stage 2 ~10-20 min
  return bundle cloud_v12_return.tar    ~6-7 GB (+~2 GB with --with-test)
EST
fi

# ---------------------------------------------------------------------------------------------------------------------------
if want preflight; then hdr preflight "files, versions, resources"
  if [ $DRY = 1 ]; then echo "  + check upload files, python versions, nproc/RAM/disk/GPUs"; else
    miss=0; while read -r f; do [ -e "$f" ] || { echo "MISSING $f"; miss=1; }; done < cloud/upload_core.txt
    [ $WITH_TEST = 1 ] && while read -r f; do [ -e "$f" ] || { echo "MISSING $f"; miss=1; }; done < cloud/upload_test_optional.txt
    [ $miss = 0 ] || fail "missing input files"
    [ -x /usr/bin/time ] || fail "/usr/bin/time missing (run cloud/env_setup.sh)"
    $P -c "import torch,polars,lightgbm,rapidfuzz,transformers;assert torch.cuda.device_count()>=2;print('py ok',torch.__version__,polars.__version__,lightgbm.__version__,torch.cuda.device_count(),'GPUs')" || fail "python env"
    mem=$(awk '/MemTotal/{print int($2/1e6)}' /proc/meminfo); dsk=$(df -BG --output=avail . | tail -1 | tr -dc 0-9)
    say "nproc $(nproc) (using $NCPU), RAM ${mem} GB, disk free ${dsk} GB"
    [ "$mem" -ge $MIN_RAM ] || fail "need >= $MIN_RAM GB RAM for fold-par $FOLD_PAR x $NM models (have $mem; use --seq-models / --fold-par)"; [ "$dsk" -ge 300 ] || fail "need >= 300 GB free disk"
    (while true; do echo "$(date '+%T') load $(cut -d' ' -f1 /proc/loadavg) memfree $(free -g | awk '/Mem/{print $7}')G $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | tr '\n' ' ')"; sleep 300; done) >> $L/monitor.log 2>&1 &
    echo $! > $L/monitor.pid
  fi; done_step preflight; fi
[ $DRY = 0 ] && [ -f $L/monitor.pid ] && trap 'kill $(cat $L/monitor.pid) 2>/dev/null' EXIT

if want complement; then hdr complement "train-split refs not in the v7 file"
  run $L/complement.log $P cloud/v12_tools.py complement --out data/v12/s1_new.parquet
  [ $DRY = 1 ] || cat $L/complement.log | grep -v "^\s" | head -3; done_step complement; fi

if want build_pairs; then hdr build_pairs "pairs of the complement refs (context features on the FULL union, like v7/v11)"
  POLARS_MAX_THREADS=$NCPU run $L/build_pairs.log $P src/build_pairs.py --cands $CANDS --split train \
      --keep-s1-file data/v12/s1_new.parquet --keep-rule "$RULE" --out data/v12/pairs_train_v12new.parquet
  [ $DRY = 1 ] || grep -E "keep-s1-file|recall|rows written|label balance" $L/build_pairs.log; done_step build_pairs; fi

if want shard; then hdr shard "row-split into $NSHARD shards"
  run $L/shard.log $P cloud/v12_tools.py shard --pairs data/v12/pairs_train_v12new.parquet --n $NSHARD --outdir data/v12/shards
  done_step shard; fi

if want features; then hdr features "$NSHARD x pair_features in parallel ($TH_SHARD workers each)"
  pids=""
  for i in $(seq -f "%02g" 0 $((NSHARD - 1))); do
    if [ $DRY = 1 ]; then echo "  + [bg] POLARS_MAX_THREADS=$TH_SHARD $P src/pair_features.py --pairs data/v12/shards/shard_$i.parquet --split train --out data/v12/shards/shard_${i}_feat.parquet --workers $TH_SHARD --wordlr-table-out data/v12/wordlr_shard_$i.json > $L/features_$i.log"; continue; fi
    [ -f data/v12/shards/shard_${i}_feat.parquet ] && { say "shard $i done already"; continue; }
    ( POLARS_MAX_THREADS=$TH_SHARD run $L/features_$i.log $P src/pair_features.py --pairs data/v12/shards/shard_$i.parquet --split train \
        --out data/v12/shards/shard_${i}_feat.parquet --workers $TH_SHARD --wordlr-table-out data/v12/wordlr_shard_$i.json ) & pids="$pids $!"
  done
  for p in $pids; do wait $p || fail "a pair_features shard failed"; done
  # word-LR tables are mined split-wide (all train-split refs) -> every shard must equal the v7 table exactly
  run $L/wordlr_check.log $P -c "
import json,glob; ref=json.load(open('data/tables/wordlr_train.json')); fs=sorted(glob.glob('data/v12/wordlr_shard_*.json'))
assert len(fs)==$NSHARD, fs
for f in fs: assert json.load(open(f))==ref, f+' differs from data/tables/wordlr_train.json'
print('word-LR tables identical to wordlr_train.json:', len(fs))"
  done_step features; fi

if want concat; then hdr concat "v7 file (400k train + val/audit) ++ complement shards"
  SH=$(for i in $(seq -f "%02g" 0 $((NSHARD - 1))); do printf ",data/v12/shards/shard_%s_feat.parquet" $i; done)
  run $L/concat.log $P cloud/v12_tools.py concat --inputs data/pairs_train_v7_feat.parquet$SH --out data/pairs_train_v12_feat.parquet
  run $L/concat_check.log $P -c "
import polars as pl; lf=pl.scan_parquet('data/pairs_train_v12_feat.parquet')
g=lf.group_by('split').agg(pl.len().alias('rows'), pl.col('s1_id').n_unique().alias('refs'), pl.col('label').mean().alias('pos')).collect().sort('split'); print(g)
d=lf.select(pl.struct('s1_id','t_id').is_duplicated().sum()).collect().item(); assert d==0, f'{d} duplicate pairs'
sp=pl.read_parquet('data/split.parquet'); tr=lf.filter(pl.col('split')=='train').select('s1_id').unique().collect()
nt=int((sp['split']=='train').sum()); print('train refs with pairs', len(tr), 'of', nt, '(refs with zero candidates have no rows)')
assert len(tr) > 0.995*nt"
  [ $DRY = 1 ] || cat $L/concat_check.log | head -12
  [ $DRY = 1 ] || rm -rf data/v12/shards/shard_??.parquet   # raw shards no longer needed (features shards kept until tar)
  done_step concat; fi

if want prep; then hdr prep "stage-1 memmaps ($CACHE)"
  run $L/prep.log $P src/stage1_lean.py prep --pairs data/pairs_train_v12_feat.parquet --cache $CACHE
  [ $DRY = 1 ] || $P -c "import json;m=json.load(open('$CACHE/meta.json'));f=json.load(open('models/lgbm_v7.json'))['features'];assert m['features']==f,'feature list differs from lgbm_v7';print('features == lgbm_v7:',len(f),'train',m['n_train'],'va',m['n_va'])" || fail "feature list"
  done_step prep; fi

# stage 1: pipeline per recipe = full fit (early stopping on val AUC) -> 5 folds in parallel (rounds = best_iteration) -> merge
pipeline() { local tag=$1; shift; local hp="$*"
  run $L/full_$tag.log $P src/stage1_lean.py full --cache $CACHE --tag $tag $hp --threads $TH_FULL
  local fh; fh=$(echo "$hp" | sed -E 's/--rounds [0-9]+//')   # fold/merge take the same params minus --rounds
  local pids=""
  for f in 0 1 2 3 4; do
    if [ $DRY = 1 ]; then echo "  + [bg] $P cloud/v12_tools.py fold --cache $CACHE --tag $tag --fold $f $fh --threads $TH_FOLD > $L/oof_${tag}_f$f.log"; continue; fi
    [ -f $CACHE/oof_${tag}_f$f.npy ] && continue
    ( run $L/oof_${tag}_f$f.log $P cloud/v12_tools.py fold --cache $CACHE --tag $tag --fold $f $fh --threads $TH_FOLD ) & pids="$pids $!"
    if [ $(echo $pids | wc -w) -ge $FOLD_PAR ]; then for p in $pids; do wait $p || fail "OOF fold of $tag failed"; done; pids=""; fi
  done
  for p in $pids; do wait $p || fail "OOF fold of $tag failed"; done
  run $L/oof_${tag}_merge.log $P cloud/v12_tools.py merge --cache $CACHE --tag $tag --threads $TH_FULL; }
if want stage1; then hdr stage1 "A (v12) || B (v12b), each full fit -> 5 parallel folds -> merge"
  if [ $DRY = 1 ]; then pipeline v12 $HPA; pipeline v12b $HPB --rounds 3000; else
    if [ $SEQ = 1 ]; then pipeline v12 $HPA; pipeline v12b $HPB --rounds 3000; else
      ( pipeline v12 $HPA ) & pa=$!; ( pipeline v12b $HPB --rounds 3000 ) & pb=$!
      wait $pa || fail "stage-1 pipeline A"; wait $pb || fail "stage-1 pipeline B"
    fi
    grep -hE "trained [0-9]+ rounds|best val macro|fixed own" $L/full_v12.log $L/full_v12b.log; grep -h "AUC" $L/oof_*_f*.log | tail -10
  fi; done_step stage1; fi

if want average; then hdr average "OOF avg + val/audit avg + model alias (stage2.py reads only 'countries' from lgbm_<tag>.json)"
  run $L/average_oof.log $P src/average_scores.py --a data/oof_scores_v12.parquet --b data/oof_scores_v12b.parquet --out data/oof_scores_v12avg.parquet
  for S in val audit; do
    run $L/average_$S.log $P src/average_scores.py --a output/${S}_scores_v12.parquet --b output/${S}_scores_v12b.parquet --out output/${S}_scores_v12avg.parquet
  done
  [ $DRY = 1 ] || { cp models/lgbm_v12.json models/lgbm_v12avg.json && ln -sf lgbm_v12.txt models/lgbm_v12avg.txt; } || fail "alias"
  done_step average; fi

if want ce; then hdr ce "fold-CE for the v12 band with ownership routing (old fold-A owners -> ce_v7fB only, fold-B -> ce_v7fA only, new refs + val/audit -> mean)"
  D=data/v12/ce_train
  run $L/ce_plan.log $P src/v11_ce.py plan --scores data/oof_scores_v12avg.parquet --kind train --dir $D
  [ $DRY = 1 ] || cat $L/ce_plan.log | grep -vE "Elapsed|Maximum|^\s+[A-Z]" | head -20
  if [ $DRY = 1 ]; then for F in A B; do echo "  + [bg GPU] $P src/score_cross_encoder.py --model models/ce_v7f$F --pairs $D/need_$F.parquet --records data/train_records.parquet --out $D/new_$F.parquet --threads 16 > $L/ce_score_$F.log"; done
  else
    ( CUDA_VISIBLE_DEVICES=0 run $L/ce_score_A.log $P src/score_cross_encoder.py --model models/ce_v7fA --pairs $D/need_A.parquet --records data/train_records.parquet --out $D/new_A.parquet --threads 16 ) & ga=$!
    ( CUDA_VISIBLE_DEVICES=1 run $L/ce_score_B.log $P src/score_cross_encoder.py --model models/ce_v7fB --pairs $D/need_B.parquet --records data/train_records.parquet --out $D/new_B.parquet --threads 16 ) & gb=$!
    wait $ga || fail "CE A scoring"; wait $gb || fail "CE B scoring"
  fi
  run $L/ce_assemble.log $P src/v11_ce.py assemble --scores data/oof_scores_v12avg.parquet --kind train --dir $D --out data/ce_oof_v12.parquet
  run $L/ce_coverage.log $P cloud/v12_tools.py coverage --scores data/oof_scores_v12avg.parquet --ce-file data/ce_oof_v12.parquet --out data/v12/ce_coverage_v12.json
  [ $DRY = 1 ] || tail -25 $L/ce_coverage.log | grep -vE "^\s+[A-Z]"
  done_step ce; fi

if want stage2; then hdr stage2 "stage 2 with CE (same recipe as v7ce); gate line = 't=0.50' (v7ce: val 0.99031 / audit 0.99046)"
  run $L/stage2_v12ce.log $P src/stage2.py --tag v12ce --pairs data/pairs_train_v12_feat.parquet --scores data/oof_scores_v12avg.parquet \
      --model models/lgbm_v12avg --band-cache data/s2_band_v12avg.parquet --ce data/ce_oof_v12.parquet --threads $TH_FULL
  [ $DRY = 1 ] || grep -E "CE features|with_p_ce|STAGE-1 baseline|STAGE-2|SUMMARY|t=0.50" $L/stage2_v12ce.log
  if [ $DRY = 1 ]; then echo "  + $P cloud/v12_eval.py --tag v12ce > $L/gate_v12ce.log"; else say "START gate"; $P cloud/v12_eval.py --tag v12ce > $L/gate_v12ce.log 2>&1 || say "GATE FAILED (below +0.0001 on val/audit); artifacts kept, continuing to pack/test for analysis"; fi   # fixed decoder: ownership, alpha 0.7, t 0.50
  [ $DRY = 1 ] || cat $L/gate_v12ce.log | grep -E "macro|GATE"
  done_step stage2; fi

if want slim; then hdr slim "s1_id,t_id + stage-2 pass-through columns (lets the band be rebuilt locally without the 14 GB file)"
  run $L/slim.log $P cloud/v12_tools.py slim --pairs data/pairs_train_v12_feat.parquet --out data/v12/pairs_train_v12_s1f.parquet
  done_step slim; fi

if want test; then hdr test "OPTIONAL (--with-test): stage-1 test scores + test CE band with the v12 models"
  if [ $WITH_TEST = 0 ]; then say "skipped (no --with-test): test path runs locally, see cloud/RETURN.md"; else
    for T in v12 v12b; do
      if [ $DRY = 1 ]; then echo "  + [bg] OMP_NUM_THREADS=$TH_FULL $P src/predict.py --pairs ${TEST_PAIRS:-data/pairs_test_v10w_feat.parquet} --model models/lgbm_$T --tag $T > $L/predict_$T.log"; continue; fi
      ( OMP_NUM_THREADS=$TH_FULL run $L/predict_$T.log $P src/predict.py --pairs ${TEST_PAIRS:-data/pairs_test_v10w_feat.parquet} --model models/lgbm_$T --tag $T ) & eval "p_$T=\$!"
    done
    [ $DRY = 1 ] || { wait $p_v12 || fail "predict v12"; wait $p_v12b || fail "predict v12b"; }
    run $L/average_test.log $P src/average_scores.py --a output/test_scores_v12.parquet --b output/test_scores_v12b.parquet --out output/test_scores_v12avg.parquet
    D=data/v12/ce_test
    run $L/ce_plan_test.log $P src/v11_ce.py plan --scores output/test_scores_v12avg.parquet --kind test --dir $D
    for F in A B; do run $L/ce_score_test_$F.log env CUDA_VISIBLE_DEVICES=0 $P src/score_cross_encoder.py --model models/ce_v7f$F --pairs $D/need_$F.parquet --records data/test_records.parquet --out $D/new_$F.parquet --threads 16; done
    run $L/ce_assemble_test.log $P src/v11_ce.py assemble --scores output/test_scores_v12avg.parquet --kind test --dir $D --out output/cef_test_v12avg.parquet
  fi; done_step test; fi

if want tar; then hdr tar "return bundle"
  RET="models/lgbm_v12.txt models/lgbm_v12.json models/lgbm_v12b.txt models/lgbm_v12b.json models/lgbm_v12avg.json models/lgbm_v12avg.txt
       models/stage2_v12ce.txt models/stage2_v12ce.json
       data/oof_scores_v12.parquet data/oof_scores_v12b.parquet data/oof_scores_v12avg.parquet
       data/ce_oof_v12.parquet data/s2_band_v12avg.parquet data/v12/ce_coverage_v12.json data/v12/ce_train data/v12/s1_new.parquet
       data/v12/pairs_train_v12_s1f.parquet data/v12/wordlr_shard_00.json
       output/val_scores_v12.parquet output/audit_scores_v12.parquet output/val_scores_v12b.parquet output/audit_scores_v12b.parquet
       output/val_scores_v12avg.parquet output/audit_scores_v12avg.parquet output/val_scores_s2_v12ce.parquet output/audit_scores_s2_v12ce.parquet
       logs/v12"
  [ $WITH_TEST = 1 ] && RET="$RET output/test_scores_v12.parquet output/test_scores_v12b.parquet output/test_scores_v12avg.parquet output/cef_test_v12avg.parquet data/v12/ce_test"
  if [ $DRY = 1 ]; then echo "  + tar -cf cloud_v12_return.tar <return list> ; cp $CACHE/oof_*_f?.txt data/v12/fold_models/ ; md5sum > cloud_v12_return.md5"; else
    mkdir -p data/v12/fold_models && cp $CACHE/oof_*_f?.txt data/v12/fold_models/ && RET="$RET data/v12/fold_models"
    for f in $RET; do [ -e "$f" ] || fail "return file missing: $f"; done
    tar -cf cloud_v12_return.tar $RET || fail "tar"
    md5sum cloud_v12_return.tar > cloud_v12_return.md5; ls -la cloud_v12_return.tar; cat cloud_v12_return.md5
    echo "$(date '+%F %T')" > $L/DONE; say "DONE v12 cloud chain -> cloud_v12_return.tar (pull it, verify md5, then TERMINATE the instance)"
  fi; done_step tar; fi
[ $DRY = 1 ] && echo && say "dry run complete (nothing executed)"
exit 0
