#!/bin/bash
# fr12d poller (light; heavy work inside eval_arm.sh goes through tools/heavy.sh). Every 5 min: fr12h, fr12, frst12 artifacts ->
# eval_arm.sh, one arm at a time. Markers ${BER_SCRATCH:?}/fr12d/<NAME>.done. Checkpoints must be complete (size + tokenizer + 2 min stable).
cd ${BER_ROOT:?}; W=${BER_SCRATCH:?}/fr12d; M=${BER_SCRATCH:?}/models; P=${BER_PY:?}
ok() { [ -f $1/model.safetensors ] && [ -f $1/tokenizer.json ] && [ "$(stat -c %s $1/model.safetensors)" = 1112201908 ] \
  && [ $(( $(date +%s) - $(stat -c %Y $1/model.safetensors) )) -gt 120 ]; }
run() { local N=$1 A=$2 B=$3
  [ -f $W/$N.done ] && return; ok $A && ok $B || return
  echo "[$(date '+%F %T')] $N present -> eval"; bash research/scripts_62/eval_arm.sh $N $A $B > logs/fr12d/eval_$N.log 2>&1
  touch $W/$N.done; echo "[$(date '+%F %T')] $N eval finished: $(grep -E 'val deltas|best supported|EVAL_DONE' logs/fr12d/eval_$N.log | tr '\n' ' ')"
  echo "- fr12d $(date +%H:%M) arm $N evaluated (research/scripts_62/eval_arm.sh; logs/fr12d/eval_$N.log): $(grep -E 'val deltas|best supported|FINAL|EVAL_DONE' logs/fr12d/eval_$N.log | tr '\n' ' ')" >> LOG.md; }
for i in $(seq 1 60); do
  echo "[$(date '+%F %T')] poll $i"
  for R in $M models; do
    run FR12H $R/fr12h_ceA $R/fr12h_ceB
    run FR12 $R/fr12_ceA $R/fr12_ceB
    run ST12 $R/frst12_ceA $R/frst12_ceB
  done
  # frst12 delivered as node score files only (no models): adapt to eval_arm inputs (OOF step will be skipped)
  if [ ! -f $W/ST12.done ] && [ ! -f $W/ST12_fr_mean.parquet ]; then
    for c in output/frst12_frst_cef_test_france.parquet; do
      V=output/frst12_frst_ce_va.parquet
      if [ -f $c ] && [ -f $V ] && [ $(( $(date +%s) - $(stat -c %Y $V) )) -gt 120 ]; then
        $P -c "
import polars as pl
pl.read_parquet('$c').select('s1_id','t_id','p_ce').write_parquet('$W/ST12_fr_mean.parquet')
pl.read_parquet('$V').filter(pl.col('split')=='val').select('s1_id','t_id','p_ce').write_parquet('$W/ST12_val_mean.parquet')"
        echo "ST12 node scores adapted"; bash research/scripts_62/eval_arm.sh ST12 none none > logs/fr12d/eval_ST12.log 2>&1; touch $W/ST12.done
        echo "- fr12d $(date +%H:%M) arm ST12 evaluated from node scores: $(grep -E 'val deltas|best supported|FINAL|EVAL_DONE' logs/fr12d/eval_ST12.log | tr '\n' ' ')" >> LOG.md
      fi
    done
  fi
  [ -f $W/FR12H.done ] && [ -f $W/FR12.done ] && [ -f $W/ST12.done ] && { echo ALL_DONE; exit 0; }
  sleep 300
done
