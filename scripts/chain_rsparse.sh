#!/bin/bash
# order: (forward sparse train, already running) -> reverse sparse train -> reverse sparse test -> forward sparse test
cd ${BER_ROOT:?}; export PYTORCH_ALLOC_CONF=expandable_segments:True
while pgrep -f "retrieve_sparse.py --split train --views full,addr,name" >/dev/null; do sleep 60; done
P=${BER_PY:?}
nice -n 10 $P src/retrieve_sparse.py --split train --reverse --views-file data/views_train.parquet --views full --k 5 --out data/cands_rsparse_train.parquet > logs/rsparse_train.log 2>&1
nice -n 10 $P src/retrieve_sparse.py --split test --reverse --views-file data/views_test.parquet --views full --k 5 --out data/cands_rsparse_test.parquet > logs/rsparse_test.log 2>&1
exec nice -n 10 $P src/retrieve_sparse.py --split test --views full,addr,name --k 30 --out data/cands_sparse_test.parquet > logs/sparse_test.log 2>&1
