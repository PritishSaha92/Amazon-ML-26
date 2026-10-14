#!/bin/bash
# wait for the train dense job to finish, then run dense retrieval on the TEST split
cd ${BER_ROOT:?}
while pgrep -f "retrieve_dense.py --split train" >/dev/null; do sleep 30; done
export HF_HOME=${BER_SCRATCH:?}/hf
exec nice -n 10 ${BER_PY:?} src/retrieve_dense.py --split test --k 30 --qbatch 512 --gblock 1000000 --out data/cands_dense_test.parquet
