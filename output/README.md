# Submission outputs

The original submission TSV files are retained locally in this directory and excluded from Git. Regenerate them using [reproduction stage 4.14](../docs/REPRODUCTION.md#414-export).

| File | Bytes in original package | Recorded MD5 |
| --- | ---: | --- |
| `candidate_pairs.tsv` | 1,712,730,255 | `db8f7384277d1a862fd3b15224f00790` |
| `matching_results.tsv` | 97,882,416 | `d5d44ade281e7213af54f65cb968be2d` |

Both files have 1,732,544 data rows. The final export contains 131,152,033 candidate pairs and 5,853,969 matched pairs. These counts and checksums come from the original reproduction guide; retraining can produce different results.

Distribute large outputs separately from the source repository if needed and permitted by the competition's data-sharing terms.
