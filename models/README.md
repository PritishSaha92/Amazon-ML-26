# Model metadata

- `lgbm_v7.json`: feature-list reference used by the stage-1 preparation checks.
- `lgbm_v7avg.json`: decoder policy and feature/country metadata used with precomputed scores.

These small files are required by the pipeline and tracked in Git. Trained LightGBM boosters and cross-encoder checkpoints are not included. See the [reproduction guide](../docs/REPRODUCTION.md) for training and checkpoint placement. Other generated files in this directory are ignored by Git.
