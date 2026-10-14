# Business Entity Resolution: code and reproduction guide

Team "Asi baat hai kya", Amazon ML Challenge 2026. For the method and results, see [the solution report](SOLUTION.md). This guide explains setup and the pipeline commands.

## 1. What the pipeline does

The pipeline retrieves candidate records, scores possible matches, and selects a consistent set of owners. Six retrieval routes and a
keep rule produce 105,226,424 base test pairs (60.7 per reference). Two LightGBM models score these pairs using 138 features. A text
cross-encoder scores uncertain pairs (p1 in [0.02, 0.98]), and three stage-2 LightGBM models rescore pairs with p1 > 0.01.
France-specific rules adjust scores before the ownership decoder applies thresholds of 0.50 and alpha 0.70. We then remove confident
US/India matches rejected by the cross-encoder and add eligible matches recovered by per-source retrieval. Sections 4.13-4.14 describe
this extension and the final candidate export.

| Tag in file/script names | Meaning |
|---|---|
| `v7` | candidate set: six-route union after the keep rule |
| `v10w` | v7 test features, India/US word log-likelihood columns from the training table |
| `v12`, `v12b`, `v12avg` | the two stage-1 models (all training references); their average |
| `ce_v12f2A/B`, `ce2` | production cross-encoder (two fold models) and its score files |
| `E2` | four rival features of stage 2 (best rival stage-1 score, margin, rivals >= 0.5, rank) |
| `v12ce2_E2` | the three stage-2 models (seeds 0, 1, 2) |
| `fr12`, `fr12s1` | French-form cross-encoder (labelled pairs + French-form rewrites), seeds 0/1, folds A/B |
| `X2` | mean of the two fr12 seed means: cross-encoder column for US/India, the drop rule and the floor |
| `frnlp`, `frnlp_cnt` | name-count-aware cross-encoder (input prefixed with a same-name count bucket) |
| `CNTX3`, `CNTX3G` | 3-seed mean of the count-aware model on the French band; the same after the guard |
| `ENS` | France cross-encoder column = 0.5 * X2 + 0.5 * CNTX3G |
| `_fr`, `_s` (suffixes) | entry after France rules and decoding; entry assembled by splicing countries |
| `UB10`, `UB10ui` | drop rule (stage-1 p > 0.98 and 4-model X2 mean < 0.10); applied to US and India only |
| `R1` | entry after the drop rule, `submissions/v12fr12x2_E2_UB10ui_s` |
| `SF8`, `SF8a` | per-source retrieval floor with full US/India re-decode; its adds-only export (the final matches) |
| `candfull` | the exported candidate file: base candidates plus all floor proposals |

## 2. Hardware and time

Local workstation: 1x RTX 5090 (32 GB), 32 cores, 123 GB RAM (preparation, retrieval, test features, test-side stage 2, X2 scoring,
rules, decoding, drop rule, floor). Rented nodes: 8x A100-40GB, 124 vCPU, ~1.8 TB RAM (stage-1 fits/OOF/test predictions, production
cross-encoder, E2 rescore, stage-2 fits, French-form and count-aware cross-encoder training/scoring; the stage-1 chain needs >= 600 GB
RAM). Longest steps: stage-1 chain ~6 h; French-form cross-encoder ~3 h per model (4); count-aware ~100 min per model (6).
Recommended local machine: 64 GB RAM or more (single local steps peak at ~34 GB; the organisers' validator needs ~18 GB).

## 3. Setup

```bash
# Python 3.12.12
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt          # pinned; includes duckdb and sentencepiece
export BER_ROOT=/path/to/repository   # project root; cwd of every command
export BER_SCRATCH=/path/to/large/disk                          # ~400 GB for a run from the raw files
export BER_PY=$(which python)  HF_HOME=$BER_SCRATCH/hf          # HF_HOME: local model cache (local scripts use $BER_SCRATCH/hf)
cd $BER_ROOT
ln -s /path/to/student_resource student_resource                # organisers' kit
bash pipeline/setup_workspace.sh
$BER_PY pipeline/fetch_models.py    # mandatory: e5-base d128750..., e5-small 614241f... (pinned) -> $HF_HOME/hub
S="$BER_SCRATCH"                    # shorthand used by the commands below
```

Offline use: once both models are in `$HF_HOME`, set `HF_HUB_OFFLINE=1` (for `fetch_models.py` and every later step).

The repository root is the working project root, `BER_ROOT`. The `src/` directory contains the core Python modules;
run commands such as `python src/prep_cache.py` from the repository root.

`setup_workspace.sh` creates the work dirs under `$BER_SCRATCH`, the link `$BER_ROOT/data -> $BER_SCRATCH/data` (scripts use both
spellings) and empty `output/ models/ submissions/ logs/`. The organisers' `student_resource/` (`dataset/{train,test}/*.tsv`, `utils/validate_submission.py`, called by `src/predict.py`
and the entry builders) must be at `$BER_ROOT/student_resource`. Python scripts read `BER_ROOT`/`BER_SCRATCH`; shell scripts call
`${BER_PY}`. LightGBM boosters live in `$BER_ROOT/models/`; cross-encoder checkpoints in `$BER_SCRATCH/models/<name>` (copy them there
after node training).

**Nodes.** Copy the repository to the node and set up once (the inputs to upload are listed in 4.4, 4.9):

```bash
export BER_ROOT=/path/to/repository BER_SCRATCH=/path/to/node/disk BER_PY=$HOME/v12env/bin/python HF_HOME=$HOME/hf
cd $BER_ROOT && bash cloud/env_setup.sh   # pinned venv $HOME/v12env, version/CUDA checks, pinned e5-base + e5-small -> $HOME/hf (mandatory)
ln -s /path/to/student_resource student_resource   # organisers' kit, uploaded too (src/predict.py runs the validator)
bash pipeline/setup_workspace.sh
S="$BER_SCRATCH"
```

Launchers use `$HOME/v12env/bin/python` and the same cache `HF_HOME=$HOME/hf` (they stop if the e5-base snapshot is missing), GPU ids 0-7.
Node-side paths are the same relative paths under `$BER_ROOT` / `$BER_SCRATCH` as locally.

**Checks.** `tools/heavy.sh -m <cap> <cmd>` runs a command in a memory-capped systemd user scope when `systemd-run --user` works,
otherwise directly. Reference values of the cached run (md5s of earlier intermediate files; row counts 1,060,103 band keys, 261,127
French band keys, the 4,956,558-key floor splice mask) only warn unless `BER_STRICT_MD5=1` (set it only when replaying the exact cached
artefacts); a retrained band differs slightly. Key equality, key uniqueness, finite scores and coverage are always asserted.

## 4. Stages

Commands run from `$BER_ROOT` in the shell of Setup (`S="$BER_SCRATCH"`), locally unless marked (node). Waiting drivers (`run_exact_chain.sh`,
`queue_ens.sh`, `arms_go.sh`, `poll.sh`) are replaced by the commands they wrap.

### 4.1 Preparation (CPU, minutes)

```bash
$BER_PY src/prep_cache.py      # -> data/{train,test}_records, train_pairs, train_gt (.parquet)
$BER_PY src/split.py           # -> data/split.parquet: val 60k / audit 20k / train (seed 20260925)
$BER_PY src/mine_tables.py     # -> data/tables/*.json (~20 s)
for SPLIT in train test; do $BER_PY src/text_views.py --bench $SPLIT --write; done   # -> data/views_{train,test}.parquet (~61 s each)
```

### 4.2 Retrieval, six routes (GPU)

Run forward dense first: it caches the e5-small embeddings `data/emb_<split>_multilingual-e5-small_full{,_ids}.npy`, which the
reverse dense route and stage 2 reuse.

```bash
for SPLIT in train test; do
  $BER_PY src/retrieve_sparse.py --split $SPLIT --views full,addr,name --k 30 --views-file data/views_$SPLIT.parquet --out data/cands_sparsev_$SPLIT.parquet
  $BER_PY src/retrieve_dense.py  --split $SPLIT --k 30 --qbatch 512 --gblock 1000000 --out data/cands_dense_$SPLIT.parquet
  $BER_PY src/retrieve_dense.py  --split $SPLIT --reverse --k 10 --qbatch 512 --gblock 1000000 --out data/cands_rdense_$SPLIT.parquet
  $BER_PY src/retrieve_sparse.py --split $SPLIT --reverse --views-file data/views_$SPLIT.parquet --views full --k 5 --out data/cands_rsparse_$SPLIT.parquet
done
```

Train/test: sparse 4,422/2,521 s; dense ~600/683 s embed + 355/176 s; rdense 498/246 s; rsparse 1,865/986 s.

### 4.3 Candidates (keep rule) and pair features

```bash
# keep: >= 2 views, or among the first nine of a forward view, first five of reverse dense, first three of reverse sparse (0-based)
RULE="(pl.col('n_views') >= 2) | (pl.col('fwd_rank') <= 8) | (pl.col('rdense_rank') <= 4) | (pl.col('rfull_rank') <= 2)"
C() { echo data/cands_sparsev_$1.parquet,data/cands_dense_$1.parquet,data/cands_rdense_$1.parquet,data/cands_rsparse_$1.parquet; }
$BER_PY src/build_pairs.py --cands $(C train) --split train --sample-s1 400000 --keep-rule "$RULE" --out data/pairs_train_v7.parquet
$BER_PY src/pair_features.py --pairs data/pairs_train_v7.parquet --split train --out data/pairs_train_v7_feat.parquet --workers 10
$BER_PY src/build_pairs.py --cands $(C test) --split test --keep-rule "$RULE" --out data/pairs_test_v7.parquet            # 241 s
$BER_PY src/pair_features.py --pairs data/pairs_test_v7.parquet --split test --out data/pairs_test_v7_feat.parquet --workers 10   # 1,361 s
$BER_PY src/pair_features.py --split test --wordlr-only --wordlr-countries India,US --wordlr-table-in data/tables/wordlr_train.json \
    --pairs data/pairs_test_v7_feat.parquet --out data/pairs_test_v10w_feat.parquet                                        # v10w, 540 s
```

Train: 400,000 sampled references (seed 20260925) + val/audit; the node chain adds the rest. `pair_features.py` also writes
`data/tables/wordlr_{train,test}.json`. On a subset of the data, `--wordlr-countries` must list only countries present in it.

### 4.4 Stage 1 (node, ~6 h)

Upload to the node (after the first command below, which runs locally): `data/{train,test}_records.parquet`, `data/views_train.parquet`,
`data/split.parquet`, `data/train_pairs.parquet`, `data/train_gt.parquet`, `data/tables/*.json`,
`data/cands_{sparsev,dense,rdense,rsparse}_train.parquet`, `data/pairs_train_v7_feat.parquet`, `data/v12/s1_v7train.parquet`,
`data/emb_train_multilingual-e5-small_full{,_ids}.npy`, `data/pairs_test_v10w_feat.parquet`.

```bash
$BER_PY pipeline/s1_v7train.py      # local, before upload: -> data/v12/s1_v7train.parquet (the 400,000 sampled train references)
# node, after the upload:
( set -e; for s in complement build_pairs shard features concat prep stage1 average; do bash cloud/chain_v12_cloud.sh --only $s; done )   # stops at the first failed stage
```

Steps: feature the other 1,726,821 training references (8 shards) -> `data/pairs_train_v12_feat.parquet` (115.1M rows) -> memmaps
`cache_v12/` -> `lgbm_v12` (defaults, 1,500 rounds; 59 min) in parallel with `lgbm_v12b` (`--leaves 255 --lr 0.03 --ff 0.7 --bf 0.8
--min-leaf 50 --l2 1.0 --seed 7 --rounds 3000`; 70 min) -> 10 concurrent reference-grouped OOF folds (2-3.6 h) -> average
(`data/oof_scores_v12avg.parquet`; `cp models/lgbm_v12.json models/lgbm_v12avg.json`). The `prep` step checks the feature list
against `models/lgbm_v7.json` (shipped). The chain's `ce`, `stage2`, `test` steps are
not used; instead:

```bash
$BER_PY src/predict.py --pairs data/pairs_test_v10w_feat.parquet --model models/lgbm_v12  --tag v12     # 16 min
$BER_PY src/predict.py --pairs data/pairs_test_v10w_feat.parquet --model models/lgbm_v12b --tag v12b    # 18 min
$BER_PY src/average_scores.py --a output/test_scores_v12.parquet --b output/test_scores_v12b.parquet --out output/test_scores_v12avg.parquet
touch logs/v12/.done_test     # marker the cross-encoder scoring step waits for
# stage-2 band cache (p1 > 0.01, no cross-encoder columns): written by the first stage2.py call that names it; this model is unused
$BER_PY src/stage2.py --tag v12band --pairs data/pairs_train_v12_feat.parquet --scores data/oof_scores_v12avg.parquet \
    --model models/lgbm_v12avg --band-cache data/s2_band_v12avg.parquet --threads 60
```

`output/test_scores_v12avg.parquet` (`p` = p1) feeds stage 2, UB10 and the floor. 4.5-4.7 need the node-only training table.

### 4.5 Production cross-encoder `ce_v12f2A/B` (node)

Folds = md5(s1_id) parity; each fold model trains from e5-base d128750 (~6,030 s on one A100). Fold-A rows are scored by B, fold-B
rows by A, val/audit/test by the mean.

```bash
( set -e                                   # subshell: stops at the first failed stage and returns its status
  for st in folds build train; do ARM=v bash cloud/ce2_arm.sh --only $st; done
  ARM=v bash cloud/ce2_arm.sh --only score --score-gpus "0 1 0 1"
  ARM=v bash cloud/ce2_arm.sh --only assemble
)
```

`build` = `src/build_ce_train_hard.py ... --n-neg 6500000 --n-syn 0 --neg-mix band=1.0,shift=2250000,sib=1500000,hard=1500000,medium=400000,random=-1`;
`train` = `src/train_ce_text.py --drop-pools syn --pos-dup 1 --batch 96 --max-len 96 --lr 3e-5 --epochs 1 --seed 0`; `score` =
`src/score_cross_encoder.py --max-len 96` (default `--batch 1024`). The outputs are `models/ce_v12f2{A,B}`, `data/ce2_oof_v12.parquet`, `output/ce2_test_v12avg.parquet` (1,060,103 test keys) and
`data/v12/cefold2/ce_train_hard_v12f2{A,B}.parquet` with its manifests.

### 4.6 E2 rival features

```bash
# (node, ~21 min; needs cache_v12 and the fold boosters)
$BER_PY research/scripts_43/node_producer_rescore.py --cache cache_v12 --scores data/oof_scores_v12avg.parquet \
    --band data/s2_band_v12avg.parquet --tags v12,v12b --pmin 0 --out data/v12/r43_rescore_full.parquet --threads 24
POLARS_MAX_THREADS=2 $BER_PY research/scripts_43/e2_feats_exact.py --rescore data/v12/r43_rescore_full.parquet \
    --tmp $S/r43/tmp_exact --out $S/data/e2_feats_v12_exact.parquet                   # 845 s
cp $S/data/e2_feats_v12_exact.parquet data/e2_feats_v12.parquet
$BER_PY research/scripts_43/e2_test_feats.py      # -> data/e2_feats_test_v12.parquet (6,947,505 rows)
```

### 4.7 Stage 2, three seeds (node, 48-51 min each, in parallel)

```bash
for SD in 0 1 2; do T=v12ce2_E2; [ $SD = 0 ] || T=v12ce2_E2_seed$SD
  STAGE2_WITHHOLD_AUDIT=1 $BER_PY src/stage2.py --tag $T --seed $SD --pairs data/pairs_train_v12_feat.parquet \
    --scores data/oof_scores_v12avg.parquet --model models/lgbm_v12avg --band-cache data/s2_band_v12avg.parquet \
    --ce data/ce2_oof_v12.parquet --extra-feats data/e2_feats_v12.parquet --threads 10 &
done; wait
```

-> `models/stage2_v12ce2_E2{,_seed1,_seed2}`: 52 features, p1 > 0.01. Fitted with the production CE OOF; at test the four CE inputs take
X2 (US/India) or ENS (France), no refit.

Copy back to the local workspace (same paths): `models/lgbm_v12{,b,avg}.*`, `models/stage2_v12ce2_E2{,_seed1,_seed2}.{txt,json}`,
`data/oof_scores_v12avg.parquet`, `data/s2_band_v12avg.parquet`, `data/ce2_oof_v12.parquet`, `data/e2_feats_v12.parquet`,
`data/e2_feats_test_v12.parquet` (read by 4.8), `data/v12/cefold2/`, `output/test_scores_v12{,b,avg}.parquet`,
`output/ce2_test_v12avg.parquet`; the checkpoints `models/ce_v12f2{A,B}` go to local `$S/models/`.

### 4.8 Base test stack, France rules, decoding (local, ~10 min)

```bash
bash cloud/s2_test.sh v12ce2_E2    # 3x src/stage2_apply.py --ce output/ce2_test_v12avg.parquet --extra-feats data/e2_feats_test_v12.parquet
                                   # + cloud/s2avg.py --test -> output/test_scores_v12ce2_E2_s3avg_s2.parquet
bash scripts/chain_rules_on.sh v12ce2_E2 output/test_scores_v12ce2_E2_s3avg_s2.parquet   # -> submissions/v12ce2_E2_fr/
```

`chain_rules_on.sh`: `src/apply_acronym_rule.py` (French initials matches -> >= 0.95), `src/apply_catswap_drop.py` (one-category-word
swaps -> <= 0.05), then `src/predict.py --pairs data/pairs_test_v7_feat.parquet --model models/lgbm_v7avg --scores-from <rules output>
--t 0.50 --alpha 0.7` (decode + validator). **The rules change stage-2 probabilities before decoding.** With `--scores-from`, predict.py
reads only the policy `models/lgbm_v7avg.json` (ownership on; feature/country lists), no booster. The first run also creates
`submissions/v7_s2ce/candidate_pairs.tsv` (export of the 105,226,424 base pairs; same file as `submissions/v10w_fr/`), which later
builders hard-link. Memory: the France rules (`apply_acronym_rule.py`) peak at ~27 GB RAM, the validator at ~18 GB.

### 4.9 French-form cross-encoder, two seeds (X2)

```bash
$BER_PY research/scripts_49/fr_roles.py              # -> data/cefold_v7fr/role_maps.json (word roles from test/train records)
$BER_PY research/scripts_55/rebuild_sets.py          # only if the cefold2 sets were not copied back (from manifests)
$BER_PY research/scripts_55/transplant_v12.py A      # French-form rewrite, seed 49 -> ce_train_hard_v12f2A_frmix.parquet (~10 min)
# upload to the node: data/cefold_v7fr/role_maps.json and data/v12/cefold2/ (incl. the new ce_train_hard_v12f2A_frmix.parquet
# and fr12_transplant_checks_A.json); then on the node, one GPU per model (~3 h); "B 1 0" first rewrites fold B
mkdir -p logs/fr12 logs/n3
bash cloud/n3_fr12.sh A 0 0 & bash cloud/n3_fr12.sh B 1 0 & bash cloud/n3_fr12.sh A 2 1 & bash cloud/n3_fr12.sh B 3 1 & wait
```

Trainer: `research/scripts_44/train_ce_text_swap.py --data <fold>_frmix.parquet --drop-pools syn --pos-dup 1 --batch 96 --threads 8
--lr 3e-5 --max-len 96 --epochs 1 --seed {0,1}` from e5-base d128750 (bf16, warmup 0.06, wd 0.01; ~20.7M pairs per fold) ->
`models/fr12_ce{A,B}` (seed 0), `models/fr12s1_ce{A,B}` (seed 1). Copy these four node directories to local `$S/models/`
(`$S/models/fr12_ceA` etc.). Local scoring, always `--max-len 96 --batch 1024` (bf16):

```bash
$BER_PY pipeline/ce_band_keys.py    # -> data/v12/fr12_frtest_pairs.parquet (French band keys), $S/cntall/fr12all/ui_pairs.parquet (US/India)
bash research/scripts_62/score_arm.sh FR12   $S/models/fr12_ceA   $S/models/fr12_ceB     # -> $S/fr12d/FR12_fr_{A,B,mean}.parquet
bash research/scripts_62/score_arm.sh FR12S1 $S/models/fr12s1_ceA $S/models/fr12s1_ceB   # -> $S/fr12d/FR12S1_fr_*.parquet
$BER_PY pipeline/fr12x2_fr_mean.py  # -> $S/fr12d/FR12X2_fr_mean.parquet = float32 mean of the two seed means
for M in fr12_ceA fr12_ceB fr12s1_ceA fr12s1_ceB; do          # US/India band, ~8 min in total
  $BER_PY src/score_cross_encoder.py --model $S/models/$M --pairs $S/cntall/fr12all/ui_pairs.parquet \
      --records data/test_records.parquet --out $S/cntall/fr12all/ui_$M.parquet --threads 3 --max-len 96 --batch 1024
done
$BER_PY research/scripts_cntall/mk_fr12all_ce.py   # -> $S/cntall/fr12all/fr12x2_test_all.parquet (all 1,060,103 keys)
bash research/scripts_cntall/build_all.sh v12fr12x2_E2 $S/cntall/fr12all/fr12x2_test_all.parquet   # ~18 min
```

`build_all.sh`: X2 in key order (`$S/cntall/v12fr12x2_E2_ce2_test.parquet`) -> 3x stage 2 -> `s2avg_lean.py`
(`output/test_scores_v12fr12x2_E2_s3avg_s2.parquet`) -> rules + decoder -> `submissions/v12fr12x2_E2_fr/`.

### 4.10 Count-aware cross-encoder, three seeds (node)

```bash
bash cloud/n3_frnlp_train.sh  A 4 & bash cloud/n3_frnlp_train.sh  B 5 &       # seed 0 -> models/frnlp_cnt_ce{A,B}
bash cloud/n3_frnlp_train1.sh A 6 & bash cloud/n3_frnlp_train1.sh B 7 &       # seed 1 -> models/frnlp_cnt1_ce{A,B}
bash cloud/n3_frnlp_trainS.sh A 0 2 & bash cloud/n3_frnlp_trainS.sh B 1 2 &   # seed 2 -> models/frnlp_cnt2_ce{A,B} (GPUs free after 4.9)
wait
```

The fold-A seed-0 call first runs `research/scripts_70/frnlp_namecore.py` and `frnlp_prep_train.py` (count tables
`$S/frnlp/cnt_{train,test}.parquet`, prefixed sets `frnlp_train_{A,B}.parquet` from the `_frmix` sets); the others wait. Trainer: `research/scripts_55/train_ce_text_swap_ptok.py --drop-pools syn --pos-dup 1
--batch 96 --lr 3e-5 --max-len 112 --epochs 1 --limit 10350000 --tok-workers 24 --tok-check 50000 --seed {0,1,2}` from e5-base d128750
(5,933-6,513 s each). Scoring (all 1,060,103 band keys, fold mean; `research/scripts_70/frnlp_score.py --max-len 112`, default `--batch 1024`):

```bash
bash cloud/n3_frnlp_testall.sh models/frnlp_cnt_ceA  models/frnlp_cnt_ceB  frnlp_test_all  0 1
bash cloud/n3_frnlp_testall.sh models/frnlp_cnt1_ceA models/frnlp_cnt1_ceB frnlp1_test_all 2 3
bash cloud/n3_frnlp_testall.sh models/frnlp_cnt2_ceA models/frnlp_cnt2_ceB frnlp2_test_all 4 5
$BER_PY research/scripts_70/frnlp_seedmean.py 0 1 2     # -> $S/frnlp/frnlpm_test_all.parquet (+ FRNLPm.seeds.json)
```

Copy back to local `$S/frnlp/`: `frnlpm_test_all.parquet` and `FRNLPm.seeds.json` (read by 4.11), with the per-seed score files
`frnlp{,1,2}_test_all.parquet` and the count tables `cnt_{train,test}.parquet`.

### 4.11 France blend, France entry, splice (local)

```bash
$BER_PY pipeline/cntx3_fr_rows.py           # -> $S/fr12d/CNTX3_fr_mean.parquet (French rows of frnlpm_test_all)
$BER_PY research/scripts_62/guard.py CNTX3  # -> $S/fr12d/CNTX3G_fr_mean.parquet
$BER_PY research/scripts_62/ens.py          # -> $S/fr12d/ENS_fr_mean.parquet = float32(0.5*FR12X2 + 0.5*CNTX3G), 261,127 keys
NO_LEDGER=1 bash research/scripts_62/build_entry.sh ENS $S/fr12d/ENS_fr_mean.parquet      # -> submissions/v12ce2_E2_frENS/ (~45 min)
bash research/scripts_62/splice_on.sh submissions/v12fr12x2_E2_fr submissions/v12ce2_E2_frENS submissions/v12fr12x2_E2_frENS_s
```

**Guard:** where the raw target `business_address` is empty after strip, p := min(count-aware mean, production CE) (80,203 pairs,
17,167 capped).

`build_entry.sh` runs `scripts/build_france_entry.sh` (CE file with France = ENS, 3x stage 2, rules + decoder, France splice onto
`submissions/v12ce2_E2_fr`, validator; if averaging runs out of memory: `s2avg_lean.py` + `builder_resume.sh ENS <file>`).
`splice_on.sh` takes US/India from the X2 entry and France from the ENS entry.

### 4.12 UB10 drop rule, US/India (local, ~20 min)

Drop an accepted pair if p1 > 0.98 and the 4-model X2 mean < 0.10; members 2-4 are scored only where member 1 < 0.40 (required for
a mean < 0.10).

```bash
U=$S/fr12d/ub; mkdir -p $U
$BER_PY research/scripts_62/ub10_test.py keys     # accepted frENS_s pairs with p1 > 0.98 -> $U/test_keys.parquet
$BER_PY src/score_cross_encoder.py --model $S/models/fr12_ceA --pairs $U/test_keys.parquet --records data/test_records.parquet \
    --out $U/test_m1.parquet --threads 3 --max-len 96 --batch 1024                     # ~16 min
$BER_PY research/scripts_62/ub10_test.py cascade  # -> $U/test_stage2.parquet
for m in fr12_ceB fr12s1_ceA fr12s1_ceB; do
  $BER_PY src/score_cross_encoder.py --model $S/models/$m --pairs $U/test_stage2.parquet --records data/test_records.parquet \
      --out $U/test_$m.parquet --threads 3 --max-len 96 --batch 1024
done
POLARS_MAX_THREADS=3 $BER_PY research/scripts_62/ub10_test.py apply   # -> $U/test_scored.parquet, test_dropped.parquet, 3 entries
bash scratch_scripts/fr12d/ub/verify.sh           # candidates hard-linked, validator, md5
```

`apply` writes `submissions/v12fr12x2_E2_{UB10all,frUB10,UB10ui}_s`; the pipeline continues from `UB10ui` (R1).

### 4.13 Per-source retrieval floor: SF8 (steps 1-7) and SF8a (step 8) (local, ~70 min)

The floor covers US/India only. It proposes every pair whose target is among the first nine records **of its own source** in any
forward view but was cut by the pooled ranking (stored ranks are zero-based: per-source ordinal <= 8, pooled rank > 8).

```bash
R=research/scripts_92; H="tools/heavy.sh -m"; T="$S/sfloor/test/r8s"
$H 20G $BER_PY $R/sf_test_load.py                     # 1 -> $S/sfloor/test_adds_full.parquet (25,925,609 pairs; 40 s)
POLARS_MAX_THREADS=6 $H 30G $BER_PY $R/sft_r8_base.py  # 2 -> $S/sfloor/test/r8/new_base_all.parquet
bash $R/sft_r8_run.sh 0 & bash $R/sft_r8_run.sh 1 & wait                 #   16 chunks, ~27 min
$BER_PY $R/sf_test_prep.py oldsf                                         # 3
$BER_PY $R/sf_test_prep.py new                    # -> $T/new.parquet (85,332 with p1 > 0.01), $T/ce_need.parquet
bash $R/sf_ce.sh $T/ce_need.parquet data/test_records.parquet $T/newce.parquet   # 4 X2 models, --batch 1024 --max-len 96
$BER_PY $R/sf_test_prep.py build                  # -> $T/{scores,s1f,e2,ce}.parquet
bash $R/sf_test_s2.sh                             # 4 3x stage2_apply in parallel (~72 GB), float32 mean -> $T/s2_s3avg.parquet
POLARS_MAX_THREADS=1 $H 40G $BER_PY $R/sf_test_splice.py $T $T/s2_final.parquet   # 5 (peak ~34 GB with 1 thread; more threads need > 40 GB)
SF_P1=$T/scores.parquet $H 38G $BER_PY $R/sf_test_decode.py $T/s2_final.parquet $T/dec.parquet $T/newce.parquet   # 6
$H 20G $BER_PY $R/sf_build_entry.py $T/dec.parquet $T/new.parquet        # 7 -> submissions/v12fr12x2_E2_UB10ui_SF8_s
POLARS_MAX_THREADS=3 $H 36G $BER_PY $R/sf_build_entry_a.py              # 8 (peak ~28 GB) -> submissions/v12fr12x2_E2_UB10ui_SF8a_s
E=submissions/v12fr12x2_E2_UB10ui_SF8a_s
$BER_PY student_resource/utils/validate_submission.py --matching $E/matching_results.tsv --candidate $E/candidate_pairs.tsv \
    --test-dir student_resource/dataset/test --check-ids                # ~18 GB RAM
```

`sf_test_s2.sh` starts the three stage-2 seeds in parallel (~24 GB each, ~72 GB in total). With less memory, run the three
`stage2_apply.py` calls one after another and then the same float32 mean instead; this gives the same files:

```bash
SA="--scores $T/scores.parquet --pairs $T/s1f.parquet --split test --ce $T/ce.parquet --extra-feats $T/e2.parquet --threads 3"
POLARS_MAX_THREADS=3 OMP_NUM_THREADS=3 $BER_PY src/stage2_apply.py $SA --model models/stage2_v12ce2_E2       --out $T/s2_seed0.parquet
POLARS_MAX_THREADS=3 OMP_NUM_THREADS=3 $BER_PY src/stage2_apply.py $SA --model models/stage2_v12ce2_E2_seed1 --out $T/s2_seed1.parquet
POLARS_MAX_THREADS=3 OMP_NUM_THREADS=3 $BER_PY src/stage2_apply.py $SA --model models/stage2_v12ce2_E2_seed2 --out $T/s2_seed2.parquet
$BER_PY -c "import polars as pl; T='$T'; K=['s1_id','t_id']; d=[pl.read_parquet(f'{T}/s2_seed{s}.parquet') for s in range(3)]
assert all(x.select(K).equals(d[0].select(K)) for x in d[1:])
d[0].with_columns(((pl.col('p') + d[1]['p'] + d[2]['p']) / 3).cast(d[0].schema['p']).alias('p')).write_parquet(f'{T}/s2_s3avg.parquet')"
```

Step 2: 16 chunks by hash(s1_id, seed 92); context = all proposed pairs on the chunk's targets; per chunk `sft_r8_build.py`,
`sft_newfeat.sh` (pair features + word-LR), `sft_r8_s1.py` (frozen stage 1). Existing pairs keep production features and p1. Step 5:
rows touching an admitted pair take the new stage-2 p, the rest keep `output/test_scores_v12fr12x2_E2_s3avg_s2.parquet`. Step 6:
US/India decode (alpha 0.7, t 0.50) + UB10. Step 8 keeps every R1 match and adds a pair only if SF8 accepts it and its target has no
owner in R1 (1,541 additions, 0 removals). Its own candidate file (R1's + the 85,332 admitted pairs) is replaced at export (4.14).

**Stage-1 admission boundary.** All 25,925,609 floor proposals are scored by stage 1; only the 85,332 above 0.01 enter the later
floor route. They receive stage-2 scores, with cross-encoder scoring where required. Base candidates remain in the cascade, including
first-stage-score bypasses and the France rules. The exported candidate file contains every pair that enters stage 1: all base pairs
and all floor proposals.

### 4.14 Export

The candidate file is the base candidate export (R1's, 105,226,424 pairs) merged with all 25,925,609 US/India floor proposals of
`$S/sfloor/test_adds_full.parquet`: touched rows become the sorted union, all other rows are copied byte for byte. France has no floor
proposals in the pipeline (the French rows of that file are never scored).

```bash
$BER_PY research/pkg/cand_full/build_candfull.py    # -> submissions/v12fr12x2_E2_UB10ui_SF8a_candfull/ (candidates + a copy of SF8a's matches)
$BER_PY research/pkg/cand_full/verify_candfull.py   # re-parses it: union rule, sort order, every match inside, per-country list lengths
E=submissions/v12fr12x2_E2_UB10ui_SF8a_candfull
$BER_PY student_resource/utils/validate_submission.py --matching $E/matching_results.tsv --candidate $E/candidate_pairs.tsv \
    --test-dir student_resource/dataset/test --check-ids
cp $E/matching_results.tsv $E/candidate_pairs.tsv output/ && md5sum output/*.tsv
```

| File | MD5 | Rows | Pairs |
|---|---|---|---|
| `output/matching_results.tsv` | `d5d44ade281e7213af54f65cb968be2d` | 1,732,544 | 5,853,969 matches |
| `output/candidate_pairs.tsv` | `db8f7384277d1a862fd3b15224f00790` | 1,732,544 | 131,152,033 candidates (75.70 per reference) |

## 5. Reproducibility

Training and inference can vary slightly with hardware, threading and bf16 arithmetic. Use the listed model revisions, batch size
(`--batch 1024`) and averaging scripts consistently. Small score changes near a decision threshold can change individual matches.

## 6. Pretrained models, libraries, data

| Model | Parameters | Licence | Use |
|---|---:|---|---|
| intfloat/multilingual-e5-base, revision d128750597153bb5987e10b1c3493a34e5a4502a | 278M | MIT | start of every cross-encoder (2 production, 4 French-form, 6 count-aware) |
| intfloat/multilingual-e5-small, revision 614241f622f53c4eeff9890bdc4f31cfecc418b3 | 118M | MIT | frozen: dense retrieval, stage-2 similarity features |
| LightGBM 4.7.0 | n/a | MIT | stage 1 (2 models), stage 2 (3 seeds) |

Other libraries: PyTorch (BSD-3); transformers, tokenizers, safetensors, sentence-transformers, pyarrow, sentencepiece (Apache-2.0); polars, rapidfuzz, DuckDB (MIT); numpy, scipy, scikit-learn (BSD). No GPL library is imported. No external data is used; all dictionaries and word tables are learned from the organisers' files.

## 7. Inputs

Each stage reads the outputs of the stages before it (paths as in the commands). Not cheaply regenerated: the 115M-row training
feature table (node-only), the cross-encoder checkpoints (GPU training) and the E2 rescore (node memmaps and fold boosters).

## 8. Repository layout

See the [repository overview](../README.md) for the directory map. All executable paths in this guide are relative to the repository root (`BER_ROOT`). Launcher scripts formerly stored in `logs/` now live in `scripts/`; `logs/` is reserved for runtime output.

The `research/` and `scratch_scripts/` paths retain their original experiment names because the final pipeline references them directly. Some historical launchers and cloud upload manifests refer to intermediate files or earlier scripts not shipped in this submission. Follow the explicit stage commands in this guide for the final pipeline.
