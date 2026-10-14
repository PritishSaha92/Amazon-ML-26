#!/bin/bash
# Memory-safe launcher for heavy local jobs (added 27 Sep after two OOM crashes that took VSCode and every job with it).
# Usage: tools/heavy.sh [-m 28G] <command...>
#  - waits for one of 7 slots (flock), so at most 7 heavy jobs run at once  (packaged copy: comment matches the loop)
#  - runs the command in its OWN systemd user scope (survives a VSCode crash) with a hard memory cap (default 28G, max 40G)
#  - the job is killed alone if it exceeds its cap, instead of taking the whole machine down
M=28G; [ "$1" = "-m" ] && { M=$2; shift 2; }
case "$M" in *G) n=${M%G}; [ "$n" -gt 40 ] && M=40G;; esac
D=${BER_ROOT:?}/logs/heavy_slots; mkdir -p $D
while true; do
  for i in 1 2 3 4 5 6 7; do
    exec {fd}>$D/slot$i
    if flock -n $fd; then
      echo "[heavy $(date '+%T')] slot $i, MemoryMax $M: $*" >> $D/heavy.log
      if systemd-run --user --scope --quiet true 2>/dev/null; then  # packaged copy: plain run where systemd user scopes are unavailable
        systemd-run --user --scope --quiet --slice=heavy -p MemoryMax=$M -p MemorySwapMax=0 -- nice -n 10 "$@"; rc=$?
      else nice -n 10 "$@"; rc=$?; fi
      echo "[heavy $(date '+%T')] slot $i done rc=$rc: $1 $2" >> $D/heavy.log
      exit $rc
    fi
    exec {fd}>&-
  done
  sleep 15
done
