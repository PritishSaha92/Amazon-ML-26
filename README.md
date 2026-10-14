# Business Entity Resolution — Amazon ML Challenge 2026

Solution by **Asi baat hai kya**: Ankit Meda, Pritish Saha, Srinjoy Ganguly, and Aadhar Batra.

Match Source 1 businesses to records in Sources 2 and 3 across the US, India, and France using multilingual retrieval, LightGBM ranking, cross-encoders, and ownership decoding.

| Result | Score |
| --- | ---: |
| Public leaderboard | **0.990689** |
| Validation (60,000 US/India references) | **0.991508** |

Scores are reported from the original competition submission. See the [solution report](docs/SOLUTION.md) for evaluation details and the [leaderboard screenshot](docs/figures/public-leaderboard.jpeg).

## Documentation

- [Solution report](docs/SOLUTION.md): approach, experiments, results, and limitations.
- [Reproduction guide](docs/REPRODUCTION.md): ordered commands from raw data to final exports.
- [GitHub upload guide](docs/GITHUB.md): what to commit and how to publish.
- [Output manifest](output/README.md): submission filenames and recorded checksums.

## Repository layout

```text
.
├── src/                 # Core Python retrieval, features, training, scoring, decoding
├── pipeline/            # Workspace setup, model downloads, pipeline helpers
├── scripts/             # Shell launchers and stage orchestration
├── cloud/               # Multi-GPU node launchers and environment setup
├── research/            # Experiment scripts used by the final solution
├── scratch_scripts/     # Auxiliary experiment and verification launchers
├── tools/               # Resource management helpers
├── models/              # Small feature-list and decoder-policy JSON files
├── docs/                # Solution report, reproduction guide, and figures
├── output/              # Local submission exports (TSVs ignored by Git)
└── requirements.txt     # Original pinned Python dependencies
```

Generated data, checkpoints, logs, and submissions stay local. The experiment directory names are preserved because the reproduction pipeline refers to them directly.

## Setup

Use Linux or WSL2 with NVIDIA CUDA support. The original environment used Python 3.12.12 and CUDA 12.8 wheels. Run commands from the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt

export BER_ROOT="$PWD"
export BER_SCRATCH=/path/to/large/disk
export BER_PY="$(command -v python)"
export HF_HOME="$BER_SCRATCH/hf"

ln -s /path/to/student_resource student_resource
bash pipeline/setup_workspace.sh
"$BER_PY" pipeline/fetch_models.py
```

Replace the example paths with your local paths. The organisers' dataset and validation kit must be supplied separately. The two JSON files in `models/` are metadata, not trained checkpoints. Download the pinned pretrained models with the command above, then follow the [reproduction guide](docs/REPRODUCTION.md) to train and run the complete pipeline.

Full reproduction is compute intensive: approximately 400 GB of scratch space, a local GPU workstation, and large-memory multi-GPU nodes for the full training stages. The original stage-1 chain required at least 600 GB RAM; see the guide for stage-specific requirements.

## Pipeline overview

1. Normalize text and build six candidate retrieval routes.
2. Compute pair features and average two stage-1 LightGBM models.
3. Score uncertain pairs with cross-encoders and three stage-2 models.
4. Apply France-specific scoring, ownership decoding, and the US/India drop rule.
5. Recover candidates through per-source retrieval and export the final submissions.

![Pipeline overview](docs/figures/fig1_pipeline.png)

## Data and model attribution

The solution uses the organisers' data and the pretrained `intfloat/multilingual-e5-small` and `intfloat/multilingual-e5-base` models. Model revisions and third-party licences are recorded in the [reproduction guide](docs/REPRODUCTION.md#6-pretrained-models-libraries-data) and [solution report](docs/SOLUTION.md). This repository does not assign a new licence to the team's code.
