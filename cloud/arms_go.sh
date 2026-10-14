#!/bin/bash
# Orchestrates the CE arms on the node: waits for the clean chain's 'average', builds folds + shared training sets (ARM=v),
# then runs the three pairs in parallel, each scoring on its own GPUs: v (GPUs 0/1), d (2/3, DAPT init), f (6/7, field template).
# Training caps: v and d identical (no cap unless the time guard binds); f capped to the vanilla wall-clock (x 1265/1640, measured).
# Time guard: all training must end by TRAIN_END_UTC (default 21:30 UTC = 03:00 IST).
cd "$(dirname "$0")/.." || exit 1; P=$HOME/v12env/bin/python; L=logs/v12ce2; mkdir -p $L
say() { echo "[$(date -u '+%F %H:%M:%S') UTC] $*"; }
until [ -f logs/v12/.done_average ]; do sleep 60; done; say "average done -> folds/build"
ARM=v bash cloud/ce2_arm.sh --only folds || { say "folds failed"; exit 1; }
ARM=v bash cloud/ce2_arm.sh --only build || { say "build failed"; exit 1; }
read NMAX < <($P -c "import json;m=json.load(open('data/v12/cefold2/manifest.json'));print(max(m['A']['exposure_after_pos_dup1'],m['B']['exposure_after_pos_dup1']))")
END=$(date -u -d "${TRAIN_END_UTC:-21:30}" +%s); NOW=$(date -u +%s); AVAIL=$(( END - NOW - 1200 ))     # ~20 min tokenize/load per arm
[[ "$NMAX" =~ ^[0-9]+$ ]] && [ "$NMAX" -gt 0 ] || { say "FATAL: bad exposure '$NMAX' from manifest.json"; exit 1; }
[ "$AVAIL" -gt 600 ] || { say "FATAL: time guard: only ${AVAIL}s left before ${TRAIN_END_UTC:-21:30} UTC (need > 600) -> NOT training (no uncapped fallback)"; exit 1; }
CAPV=$(( AVAIL * 1640 * 95 / 100 )); CAPF=$(( AVAIL * 1265 * 95 / 100 ))
LV=0; [ $CAPV -lt $NMAX ] && LV=$CAPV
LF=$(( NMAX * 1265 / 1640 )); [ $CAPF -lt $LF ] && LF=$CAPF
for X in "$LV" "$LF"; do [[ "$X" =~ ^[0-9]+$ ]] || { say "FATAL: non-numeric/negative cap '$X'"; exit 1; }; done
[ "$LF" -gt 0 ] || { say "FATAL: field cap 0"; exit 1; }
say "exposure max $NMAX; time left for training ${AVAIL}s; vanilla/DAPT limit ${LV} (0 = full); field limit ${LF} (max-len 192, 2-order scoring)"
( ARM=v S2TH=20 bash cloud/ce2_arm.sh --limit $LV --score-gpus "0 1 0 1" > $L/arm_v.out 2>&1 ) & a=$!
# screen markers (screens steward): CANCEL_D -> no DAPT pair; field pair only with PROCEED_F (wait <= 20 min for PROCEED_F/CANCEL_F; missing = cancel)
if [ -f logs/screens/CANCEL_D ]; then say "logs/screens/CANCEL_D present -> DAPT pair SKIPPED"; ( exit 0 ) & b=$!
else ( [ -e models/e5_dapt_v12/model.safetensors ] && ARM=d S2TH=20 bash cloud/ce2_arm.sh --limit $LV --score-gpus "2 3 2 3" > $L/arm_d.out 2>&1 ) & b=$!; fi
( for i in $(seq 1 20); do [ -f logs/screens/PROCEED_F ] || [ -f logs/screens/CANCEL_F ] && break; sleep 60; done
  if [ -f logs/screens/PROCEED_F ] && [ ! -f logs/screens/CANCEL_F ]; then say "PROCEED_F present -> field pair runs"
    ARM=f S2TH=20 MAXLEN=192 bash cloud/ce2_arm.sh --limit $LF --score-gpus "6 7 6 7" > $L/arm_f.out 2>&1
  else say "no PROCEED_F (CANCEL_F or 20-min timeout) -> field pair SKIPPED"; fi ) & c=$!
wait $a; say "arm v exit $?"; wait $b; say "arm d exit $?"; wait $c; say "arm f exit $?"
