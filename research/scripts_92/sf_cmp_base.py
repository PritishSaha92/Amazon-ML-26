"""compare a decode parquet (s1_id, m) with submissions/v12fr12x2_E2_UB10ui_s US/India rows."""
import sys, polars as pl
cty = pl.read_parquet('data/test_records.parquet', columns=['entity_id', 'country']).rename({'entity_id': 's1_id'})
b = pl.read_csv('submissions/v12fr12x2_E2_UB10ui_s/matching_results.tsv', separator='\t', quote_char=None, schema_overrides={'matched_entity_ids': pl.String}).fill_null('')
b = b.rename({'source1_entity_id': 's1_id', 'matched_entity_ids': 'mb'}).join(cty, on='s1_id').filter(pl.col('country').is_in(['US', 'India']))
d = b.join(pl.read_parquet(sys.argv[1]), on='s1_id', how='left').with_columns(pl.col('m').fill_null(''))
diff = d.filter(pl.col('m') != pl.col('mb')); print('US/India refs', d.height, 'differ', diff.height, diff.group_by('country').len().rows()); print(diff.head(5))
if len(sys.argv) > 2: diff.write_parquet(sys.argv[2])
