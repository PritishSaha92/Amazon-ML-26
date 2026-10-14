#!/bin/bash
# One-time workspace setup for the packaged pipeline (see README "Setup").
#   export BER_ROOT=/path/to/repository   # this directory's parent: code + small outputs, cwd of every command
#   export BER_SCRATCH=<large disk>                                 # data/, cross-encoder checkpoints, work dirs, Hugging Face cache
#   export BER_PY=$(command -v python)                              # interpreter with requirements.txt installed
#   bash $BER_ROOT/pipeline/setup_workspace.sh
set -euo pipefail
: "${BER_ROOT:?set BER_ROOT}" "${BER_SCRATCH:?set BER_SCRATCH}" "${BER_PY:?set BER_PY}"
cd "$BER_ROOT"
mkdir -p "$BER_SCRATCH"/{data,models,hf,fr12d/ub,cntall/fr12all,frnlp/train,sfloor/test/r8,sfloor/test/r8s,r43}
mkdir -p "$BER_SCRATCH"/data/{tables,v12/cefold2,cefold_v7fr}
# scripts address the same files as data/X and $BER_SCRATCH/data/X
if [ -L data ]; then [ "$(readlink data)" = "$BER_SCRATCH/data" ] || { echo "data -> $(readlink data), expected $BER_SCRATCH/data"; exit 1; }
elif [ -e data ]; then echo "$BER_ROOT/data exists and is not a link"; exit 1
else ln -s "$BER_SCRATCH/data" data; fi
mkdir -p output models submissions logs/heavy_slots
[ -f models/lgbm_v7avg.json ] || { echo "missing models/lgbm_v7avg.json (decoder policy, ships with the package)"; exit 1; }
# organisers' files: student_resource/dataset/{train,test}/*.tsv and student_resource/utils/validate_submission.py
for f in student_resource/utils/validate_submission.py student_resource/dataset/test/test_source1.tsv; do
  [ -e "$f" ] || echo "NOTE: $BER_ROOT/$f not found: copy or link the organisers' student_resource/ directory to $BER_ROOT/student_resource"
done
"$BER_PY" -c "import torch, transformers, lightgbm, polars; print('python ok, torch', torch.__version__, 'cuda', torch.cuda.is_available())"
"$BER_PY" -c "import duckdb" 2>/dev/null || echo "NOTE: duckdb not importable with $BER_PY (needed by the local floor steps; not on the node)"
echo "workspace ready: BER_ROOT=$BER_ROOT BER_SCRATCH=$BER_SCRATCH"
