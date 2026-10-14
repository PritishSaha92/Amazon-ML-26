#!/bin/bash
# v12 cloud environment: an isolated Python 3.12.12 venv with the SAME package versions as the local `torch` env.
# Target: Lambda Stack 22.04/24.04 image (driver + CUDA preinstalled). We do NOT use Lambda Stack's system torch: our
# scripts are pinned to torch 2.10.0+cu128 / transformers 5.2.0 / polars 1.44.2 / lightgbm 4.7.0, so we build our own venv
# (uv downloads a standalone CPython 3.12.12; no apt python needed). A100 = sm_80 is covered by the cu128 wheel.
# Usage (on the node, from the project root ~/amazon):  bash cloud/env_setup.sh      (~5-10 min; ~6 GB of wheels)
set -euo pipefail
cd "$(dirname "$0")/.."
VENV=${VENV:-$HOME/v12env}
sudo apt-get update -qq || true
command -v /usr/bin/time >/dev/null || sudo apt-get install -y time          # GNU time for `time -v`
command -v zstd >/dev/null || sudo apt-get install -y zstd
if ! command -v uv >/dev/null; then curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; fi
uv venv --python 3.12.12 "$VENV"
. "$VENV/bin/activate"
uv pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0
uv pip install -r cloud/requirements_v12.txt
# packaged copy: mandatory pinned download of e5-base d128750 and e5-small 614241f into $HOME/hf, the cache every cloud/ launcher uses
export HF_HOME=$HOME/hf  # packaged copy
python pipeline/fetch_models.py  # packaged copy
python - <<'PY'
import sys, torch, transformers, polars, lightgbm, numpy, pandas, pyarrow, rapidfuzz, sklearn, sentence_transformers
v = dict(python=sys.version.split()[0], torch=torch.__version__, cuda=torch.version.cuda, transformers=transformers.__version__,
         polars=polars.__version__, lightgbm=lightgbm.__version__, numpy=numpy.__version__, pandas=pandas.__version__,
         pyarrow=pyarrow.__version__, rapidfuzz=rapidfuzz.__version__, sklearn=sklearn.__version__, st=sentence_transformers.__version__)
print(v)
exp = dict(python="3.12.12", torch="2.10.0+cu128", transformers="5.2.0", polars="1.44.2", lightgbm="4.7.0", numpy="2.3.5",
           pandas="3.0.1", pyarrow="23.0.1", rapidfuzz="3.14.6", sklearn="1.8.0", st="5.2.3")
bad = {k: (v[k], e) for k, e in exp.items() if v[k] != e}
assert not bad, f"version mismatch: {bad}"
assert torch.cuda.is_available(), "CUDA not available"
print("GPUs:", torch.cuda.device_count(), torch.cuda.get_device_name(0))
x = torch.randn(1024, 1024, device="cuda", dtype=torch.bfloat16); print("bf16 matmul ok", float((x @ x).float().abs().mean()))
PY
echo "ENV OK: . $VENV/bin/activate"
