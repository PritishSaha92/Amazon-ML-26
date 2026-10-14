# SF8 -> SF8a build sequence (sfloor, 27 Sep 2026)
Entry: submissions/v12fr12x2_E2_UB10ui_SF8a_s  matching_results md5 d5d44ade281e7213af54f65cb968be2d, candidate_pairs md5 29a07b6f8923caa277793a9543a130ed.
Base: submissions/v12fr12x2_E2_UB10ui_s (matching md5 a7edfe5da3b11d1c1182d98ce1cb0e62). Python: $BER_PY; heavy steps via tools/heavy.sh. sha256 prefixes of the code as run:

1. Test proposal (label-free source floor, 4 forward views top-30, within-source ordinal <= 8, pooled rank > 8, anti-join current support):
   sf_test_load.py (e91843caa95c60ee) -> $BER_SCRATCH/sfloor/test_adds_full.parquet (US/India o<=8: 25,925,609 pairs).
2. Base columns for all new pairs + chunked context/features/stage-1 (chunk = s1_id.hash(seed=92) % 16; target context over ALL new pairs on the chunk's targets):
   sft_r8_base.py (fcf6df85aa4a9c6f), sft_r8_build.py (646ca9b0bf02cd97; full 646ca9b0bf02cd974645f3363cabc50c43674a9983cf7a7ad55020df9fe57556), sft_r8_s1.py (2c707f02886812c1), sft_r8_chunk.sh (25d70fb6f5fa9356), sft_r8_run.sh (72a6c60e74b0d94e)
   -> $BER_SCRATCH/sfloor/test/r8/{s1_c,adm_c}{0..15}.parquet, progress.txt.
3. Admission + manifest proof + stage-2 inputs (E2 recompute on touched targets with production parity, CE = production X2 test file + X2 of new band rows):
   sf_test_prep.py (bf30f884a85547c7) 'oldsf', 'new', 'build'; sf_ce.sh (ec7889adfd073cb9) on test/r8s/ce_need.parquet with data/test_records.parquet, --batch 1024 --max-len 96.
4. Stage 2: sf_test_s2.sh (6c207c719a7f351f) = src/stage2_apply.py x3 seeds (models/stage2_v12ce2_E2{,_seed1,_seed2}), 3 threads, float32 mean -> test/r8s/s2_s3avg.parquet.
5. Closure splice: sf_test_splice.py (7acef65e5ca3e3aa) -> test/r8s/s2_final.parquet (mask 4,956,558 keys, hash 11427877698333841077; replay verified equal to the file used).
6. Decode + UB10: SF_P1=$BER_SCRATCH/sfloor/test/r8s/scores.parquet sf_test_decode.py (664191d0977dbf7c) test/r8s/s2_final.parquet test/r8s/dec.parquet test/r8s/newce.parquet.
7. SF8 entry: sf_build_entry.py (2adc58d4a0835632) -> submissions/v12fr12x2_E2_UB10ui_SF8_s (34fccc1f...).
8. SF8a (adds only): sf_build_entry_a.py (7a0659372aaf09d2) -> submissions/v12fr12x2_E2_UB10ui_SF8a_s; then official validator (student_resource/utils/validate_submission.py --check-ids).
The sha256 prefixes above identify the files before the packaging path rewrite (machine paths -> BER_ROOT / BER_SCRATCH); the package README gives the commands for the packaged copies.
