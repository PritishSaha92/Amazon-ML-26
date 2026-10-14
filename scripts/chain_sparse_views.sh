#!/bin/bash
# License-safe forward sparse retrieval (text_views.py views, no Unidecode) for train and test. Waits for GPU to be free (< 6 GB used).
cd ${BER_ROOT:?}; export PYTORCH_ALLOC_CONF=expandable_segments:True; P=${BER_PY:?}
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)" -gt 6000 ]; do sleep 120; done
nice -n 10 $P src/retrieve_sparse.py --split train --views full,addr,name --k 30 --views-file data/views_train.parquet --out data/cands_sparsev_train.parquet > logs/sparsev_train.log 2>&1
exec nice -n 10 $P src/retrieve_sparse.py --split test --views full,addr,name --k 30 --views-file data/views_test.parquet --out data/cands_sparsev_test.parquet > logs/sparsev_test.log 2>&1
