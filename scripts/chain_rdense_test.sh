#!/bin/bash
cd ${BER_ROOT:?}; export HF_HOME=${BER_SCRATCH:?}/hf
while pgrep -f "chain_dense_test.sh|retrieve_dense.py --split test" >/dev/null; do sleep 30; done   # wait for forward dense test (embeds test)
exec nice -n 10 ${BER_PY:?} src/retrieve_dense.py --split test --reverse --k 10 --qbatch 512 --gblock 1000000 --out data/cands_rdense_test.parquet
