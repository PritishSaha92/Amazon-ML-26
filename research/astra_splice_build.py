"""Build a France-control splice, preserving finalist row order and selected raw rows."""
import argparse
import csv
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import pyarrow.parquet as pq
from astra_splice_check import verify, digest


def raw_rows(path):
    """Physical-line format only: reject multiline quoted fields rather than rewrite."""
    with open(path, 'rb') as f:
        header = f.readline()
        if header.rstrip(b'\r\n') != b'source1_entity_id\tmatched_entity_ids':
            raise ValueError('Expected literal TSV header')
        yield None, header
        for line in f:
            text = line.decode('utf-8')
            parsed = list(csv.reader([text], delimiter='\t', strict=True))
            if len(parsed) != 1 or len(parsed[0]) != 2:
                raise ValueError('Only one physical line per TSV record is supported')
            yield parsed[0][0], line


def build(finalist, control, countries, out, scratch=None):
    finalist, control, countries, out = map(Path, (finalist, control, countries, out))
    if out.exists() or out.is_symlink():
        raise FileExistsError('Refusing existing output')
    if out.resolve() in {p.resolve() for p in (finalist, control, countries)}:
        raise ValueError('Output aliases an input')
    # Checking each file against itself verifies complete S1 coverage and syntax.
    initial = {}
    for name, path in [('finalist', finalist), ('control', control)]:
        r = verify(path, path, countries, scratch=scratch)
        if r['status'] != 'PASS':
            raise ValueError(f'{name} failed input verification: {r.get("error")}')
        initial[name] = r['md5']['finalist']
    country_md5 = digest(countries)
    out.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.TemporaryDirectory(prefix='astra-splice-build-', dir=scratch) as td:
            db = sqlite3.connect(str(Path(td) / 'rows.sqlite'))
            try:
                db.execute('PRAGMA cache_size=-32768'); db.execute('PRAGMA temp_store=FILE')
                db.execute('CREATE TABLE france(id TEXT PRIMARY KEY, raw BLOB) WITHOUT ROWID')
                pf = pq.ParquetFile(countries)
                for rb in pf.iter_batches(batch_size=10000, columns=['entity_id', 'country'], use_threads=False):
                    rows = [(i,) for i, c in zip(rb.column(0).to_pylist(), rb.column(1).to_pylist()) if i.startswith('S1-') and c == 'France']
                    db.executemany('INSERT INTO france(id) VALUES (?)', rows)
                rows = []
                for rid, line in raw_rows(control):
                    if rid is not None:
                        rows.append((line, rid))
                    if len(rows) == 10000:
                        db.executemany('UPDATE france SET raw=? WHERE id=?', rows); rows.clear()
                db.executemany('UPDATE france SET raw=? WHERE id=?', rows)
                assert db.execute('SELECT COUNT(*) FROM france WHERE raw IS NULL').fetchone()[0] == 0
                db.commit()
                fd, tmpname = tempfile.mkstemp(prefix=out.name + '.', suffix='.partial', dir=out.parent)
                temp = Path(tmpname); selected_france = 0; appended_newlines = 0
                with os.fdopen(fd, 'wb') as stream:
                    iterator = iter(raw_rows(finalist))
                    _, header = next(iterator)
                    stream.write(header if header.endswith(b'\n') else header + b'\n')
                    # Lookahead permits a selected EOF line without newline only at output EOF.
                    previous = None
                    for rid, line in iterator:
                        if previous is not None:
                            if not previous.endswith(b'\n'):
                                previous += b'\n'; appended_newlines += 1
                            stream.write(previous)
                        found = db.execute('SELECT raw FROM france WHERE id=?', (rid,)).fetchone()
                        previous = found[0] if found else line
                        selected_france += found is not None
                    if previous is not None:
                        stream.write(previous)
                    stream.flush(); os.fsync(stream.fileno())
            finally:
                db.close()
        check = verify(finalist, temp, countries, control, scratch)
        if check['status'] != 'PASS':
            raise ValueError(f'Built splice failed verification: {check.get("error")}')
        # Guard against inputs being replaced during a long build.
        if any(digest(p) != initial[k] for k, p in [('finalist', finalist), ('control', control)]) or digest(countries) != country_md5:
            raise ValueError('Input changed during build')
        # Atomic, no-clobber publication on the same filesystem (unlike rename).
        os.chmod(temp, 0o644)  # packaged copy: published file readable (mkstemp creates 0600)
        os.link(temp, out)
        temp.unlink(); temp = None
        check['verified_before_publication'] = True
        check['files']['splice'] = str(out)
        return {'status': 'PASS', 'finalist': str(finalist), 'control': str(control), 'out': str(out),
                'md5': {**initial, 'out': check['md5']['splice'], 'countries': country_md5},
                'france_rows_selected': selected_france, 'missing_newlines_appended': appended_newlines,
                'verification': check}
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for arg in ('finalist', 'control', 'out', 'report'):
        ap.add_argument('--' + arg, required=True)
    ap.add_argument('--countries', default='data/test_records.parquet')
    ap.add_argument('--scratch')
    a = ap.parse_args()
    # A report must not overwrite an input or the freshly built submission.
    protected = (a.finalist, a.control, a.countries, a.out)
    if (Path(a.report).resolve() in {Path(p).resolve() for p in protected}
            or (Path(a.report).exists() and any(Path(p).exists() and os.path.samefile(a.report, p) for p in protected))):
        ap.error('Report aliases an input/output')
    try:
        r = build(a.finalist, a.control, a.countries, a.out, a.scratch)
    except Exception as exc:
        r = {'status': 'FAIL', 'error': f'{type(exc).__name__}: {exc}'}
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps(r, indent=2) + '\n')
    print(json.dumps(r, indent=2))
    raise SystemExit(0 if r['status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
