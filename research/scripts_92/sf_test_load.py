"""sfloor: label-free TEST load of the astra2 source floor (same rule as research/astra2/source_floor_val_load.py), per country
and per max source ordinal. Current support = keys of output/test_scores_v12avg.parquet (production test candidates)."""
import os as _ber_os; _BER_R = _ber_os.environ["BER_ROOT"]; _BER_S = _ber_os.environ["BER_SCRATCH"]  # packaged copy: machine paths -> environment (README "Setup")
import os, duckdb, time
os.chdir(_BER_R); t0 = time.time()
c = duckdb.connect(); c.execute("SET threads=6; SET memory_limit='16GB'; SET preserve_insertion_order=false")
c.execute("CREATE TEMP TABLE fwd AS SELECT s1_id,t_id,view,rank FROM read_parquet('data/cands_sparsev_test.parquet') UNION ALL SELECT s1_id,t_id,view,rank FROM read_parquet('data/cands_dense_test.parquet')")
print(c.execute("SELECT view,count(*),max(rank) FROM fwd GROUP BY 1").fetchall())
c.execute("CREATE TEMP TABLE ranked AS SELECT *,row_number() OVER(PARTITION BY s1_id,view,substr(t_id,2,1) ORDER BY rank)-1 o FROM fwd WHERE rank<30")
c.execute("CREATE TEMP TABLE prop AS SELECT s1_id,t_id,min(o) o FROM ranked WHERE o<=8 AND rank>8 GROUP BY ALL")
c.execute("CREATE TEMP TABLE cur AS SELECT s1_id,t_id FROM read_parquet('output/test_scores_v12avg.parquet')")
c.execute("CREATE TEMP TABLE adds AS SELECT p.* FROM prop p ANTI JOIN cur USING(s1_id,t_id)")
c.execute("CREATE TEMP TABLE cty AS SELECT entity_id s1_id,country FROM read_parquet('data/test_records.parquet') WHERE substr(entity_id,2,1)='1'")
print(c.execute("SELECT country,count(*) refs FROM cty GROUP BY 1 ORDER BY 1").fetchall())
print('adds by country x min ordinal (cumulative):')
for k in (0, 1, 2, 3, 5, 8):
    print(k, c.execute(f"SELECT country,count(*) FROM adds JOIN cty USING(s1_id) WHERE o<={k} GROUP BY 1 ORDER BY 1").fetchall())
c.execute(("COPY (SELECT a.*,country FROM adds a JOIN cty USING(s1_id) ORDER BY s1_id,t_id) TO '" + _BER_S + "/sfloor/test_adds_full.parquet' (FORMAT PARQUET)"))
print('secs', time.time() - t0)
