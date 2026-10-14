cd ${BER_ROOT:?}; W=${BER_SCRATCH:?}/fr12d
until grep -qE "DONE fr12d_fr12x2r|refusing|Traceback|FAIL" logs/fr12d/fr12d_fr12x2r.out; do sleep 20; done
echo "[$(date +%T)] FR12x2R: $(tail -1 logs/fr12d/fr12d_fr12x2r.out)"
bash research/scripts_62/build_entry.sh ENS $W/ENS_fr_mean.parquet > logs/fr12d/build_ENS.log 2>&1; echo "[$(date +%T)] ENS: $(grep BUILD_ logs/fr12d/build_ENS.log)"
bash research/scripts_62/build_entry.sh FR12x3 $W/FR12X3_fr_mean.parquet > logs/fr12d/build_FR12x3.log 2>&1; echo "[$(date +%T)] FR12x3: $(grep BUILD_ logs/fr12d/build_FR12x3.log)"
cp research/scripts_62/run_arm.sh $W/run_arm_frozen_FR12R.sh
S2MODEL=models/stage2_fr12d_s2FR12 tools/heavy.sh -m 40G bash $W/run_arm_frozen_FR12R.sh fr12d_fr12r - $W/FR12_fr_mean.parquet FR12R > logs/fr12d/fr12d_fr12r.out 2>&1; echo "[$(date +%T)] FR12R: $(tail -1 logs/fr12d/fr12d_fr12r.out)"
echo QUEUE_ENS_DONE
