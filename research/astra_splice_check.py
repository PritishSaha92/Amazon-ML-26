"""Label-free, disk-backed verifier of France-only matching-result splices."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import tempfile
import pyarrow as pa
import pyarrow.parquet as pq

S1 = re.compile(r'S1-[0-9]+\Z')
TARGET = re.compile(r'S[23]-[0-9]+\Z')
COUNTRIES = {'France', 'US', 'India'}


def digest(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def canonical(raw):
    if raw == '':
        return ''
    ids = raw.split(',')
    if any(not TARGET.fullmatch(x) for x in ids):
        raise ValueError('Malformed target set (expected comma-separated S2/S3 numeric IDs)')
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate target within a target set')
    return ','.join(sorted(ids))


def verify(finalist, splice, countries, control=None, scratch=None):
    pa.set_cpu_count(2); pa.set_io_thread_count(2)
    paths = {'finalist': str(finalist), 'splice': str(splice)}
    if control is not None:
        paths['control'] = str(control)
    report = {'status': 'FAIL', 'files': paths, 'md5': {k: digest(v) for k, v in paths.items()},
              'country_map_md5': digest(countries)}
    with tempfile.TemporaryDirectory(prefix='astra-splice-', dir=scratch) as td:
        db = sqlite3.connect(str(Path(td) / 'check.sqlite'))
        try:
            db.execute('PRAGMA foreign_keys=ON')
            db.execute('PRAGMA cache_size=-32768')
            db.execute('PRAGMA temp_store=FILE')
            db.execute('CREATE TABLE refs (id TEXT PRIMARY KEY, country TEXT NOT NULL) WITHOUT ROWID')
            pf = pq.ParquetFile(countries)
            if not {'entity_id', 'country'} <= set(pf.schema_arrow.names):
                raise ValueError('Country parquet needs entity_id,country columns')
            for rb in pf.iter_batches(batch_size=10000, columns=['entity_id', 'country'], use_threads=False):
                rows = []
                for rid, country in zip(rb.column(0).to_pylist(), rb.column(1).to_pylist()):
                    if not isinstance(rid, str):
                        raise ValueError('Null/nonstring record ID in country map')
                    if not rid.startswith('S1-'):
                        continue
                    if not S1.fullmatch(rid) or country not in COUNTRIES:
                        raise ValueError(f'Invalid S1 ID or unknown country: {rid!r}, {country!r}')
                    rows.append((rid, country))
                db.executemany('INSERT INTO refs VALUES (?,?)', rows)
            expected = db.execute('SELECT COUNT(*) FROM refs').fetchone()[0]
            if not expected:
                raise ValueError('Country map contains no S1 references')
            report['expected_s1_rows'] = expected
            for name, path in paths.items():
                db.execute(f'CREATE TABLE {name} (id TEXT PRIMARY KEY REFERENCES refs(id), canon TEXT NOT NULL, raw TEXT NOT NULL) WITHOUT ROWID')
                with open(path, newline='', encoding='utf-8') as f:
                    reader = csv.reader(f, delimiter='\t', strict=True)
                    if next(reader, None) != ['source1_entity_id', 'matched_entity_ids']:
                        raise ValueError(f'{name}: invalid header')
                    batch = []
                    for line, row in enumerate(reader, 2):
                        if len(row) != 2 or not S1.fullmatch(row[0]):
                            raise ValueError(f'{name}: malformed row at line {line}')
                        batch.append((row[0], canonical(row[1]), row[1]))
                        if len(batch) == 10000:
                            db.executemany(f'INSERT INTO {name} VALUES (?,?,?)', batch); batch.clear()
                    db.executemany(f'INSERT INTO {name} VALUES (?,?,?)', batch)
                count = db.execute(f'SELECT COUNT(*) FROM {name}').fetchone()[0]
                if count != expected:
                    raise ValueError(f'{name}: missing S1 rows ({count} != {expected})')
            db.commit()
            query = 'SELECT r.country, f.canon, s.canon, f.raw, s.raw'
            if control:
                query += ', c.canon, c.raw'
            query += ' FROM refs r JOIN finalist f ON r.id=f.id JOIN splice s ON r.id=s.id'
            if control:
                query += ' JOIN control c ON r.id=c.id'
            changed = Counter(); serialization = Counter(); outside = 0; mismatch = 0; control_serialization = 0
            for row in db.execute(query):
                country, fc, sc, fr, sr = row[:5]
                if fc != sc:
                    changed[country] += 1
                    outside += country != 'France'
                elif fr != sr:
                    serialization[country] += 1
                if control and country == 'France':
                    mismatch += sc != row[5]
                    control_serialization += sc == row[5] and sr != row[6]
            report.update(changed_s1_by_country={c: changed[c] for c in sorted(COUNTRIES)},
                          ordering_only_diffs_by_country={c: serialization[c] for c in sorted(COUNTRIES)},
                          non_france_semantic_changes=outside,
                          france_control_semantic_mismatches=mismatch if control else None,
                          france_control_ordering_only_diffs=control_serialization if control else None,
                          target_record_existence_checked=False,
                          country_map=str(countries))
            if outside or mismatch:
                raise ValueError(f'Splice containment failure: {outside} non-France changes, {mismatch} France/control mismatches')
            report['status'] = 'PASS'
        except (ValueError, sqlite3.IntegrityError, csv.Error) as exc:
            report['error'] = str(exc)
        finally:
            db.close()
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--finalist', required=True)
    ap.add_argument('--splice', required=True)
    ap.add_argument('--countries', default='data/test_records.parquet')
    ap.add_argument('--control')
    ap.add_argument('--scratch', help='Directory for bounded-cache SQLite scratch database')
    ap.add_argument('--report', required=True)
    a = ap.parse_args()
    try:
        result = verify(a.finalist, a.splice, a.countries, a.control, a.scratch)
    except Exception as exc:
        result = {'status': 'FAIL', 'error': f'{type(exc).__name__}: {exc}'}
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
