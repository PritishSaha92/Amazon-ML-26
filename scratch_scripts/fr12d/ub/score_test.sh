cd ${BER_ROOT:?}; export HF_HOME=${BER_SCRATCH:?}/hf POLARS_MAX_THREADS=3 OMP_NUM_THREADS=3; P=${BER_PY:?}; U=${BER_SCRATCH:?}/fr12d/ub
until grep -q DENSE_SCORED logs/fr12d/ub_score_dense.log; do sleep 15; done
nice -n 10 $P src/score_cross_encoder.py --model ${BER_SCRATCH:?}/models/fr12_ceA --pairs $U/test_keys.parquet --records data/test_records.parquet --out $U/test_m1.parquet --threads 3 --max-len 96 --batch 1024 || exit 1
$P research/scripts_62/ub10_test.py cascade || exit 1
for m in fr12_ceB fr12s1_ceA fr12s1_ceB; do nice -n 10 $P src/score_cross_encoder.py --model ${BER_SCRATCH:?}/models/$m --pairs $U/test_stage2.parquet --records data/test_records.parquet --out $U/test_$m.parquet --threads 3 --max-len 96 --batch 1024 || exit 1; done
echo TEST_SCORED
