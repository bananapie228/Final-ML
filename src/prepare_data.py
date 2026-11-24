import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from src.config import (LABELS, MANIFEST, MIN_DESCRIPTION_CHARS, PROCESSED, RAW_COLUMNS, RAW_FILE,
                        ROOT, SAMPLE_FILE, SAMPLE_SIZE, SEED, TEST_IDS, TEST_SIZE, TRAIN_IDS)
from src.data import processed_hashes, sha256

CHUNK_ROWS = 50_000


def check_columns(path):
    header = list(pd.read_csv(path, nrows=0).columns)
    missing = [c for c in RAW_COLUMNS if c not in header]
    if missing:
        raise SystemExit(f'Columns {missing} not found in {path.name}. Available columns: {header}')


def candidate_index(path):
    # Pass 1: keep only id, label and a hash of the description, so the full ~500 MB file never
    # has to fit in memory (this machine has 8 GB RAM).
    parts = []
    reader = pd.read_csv(path, usecols=['id', 'experience_id', 'description', 'employer_id'],
                         dtype={'id': str, 'employer_id': str}, chunksize=CHUNK_ROWS)
    for chunk in reader:
        chunk = chunk[chunk['experience_id'].isin(LABELS) & chunk['id'].notna()]
        chunk = chunk[chunk['description'].fillna('').str.len() >= MIN_DESCRIPTION_CHARS]
        chunk = chunk.assign(desc_hash=chunk['description'].map(
            lambda t: hashlib.md5(t.encode('utf-8')).hexdigest()),
            employer_id=chunk['employer_id'].fillna('no-employer-' + chunk['id']))
        parts.append(chunk[['id', 'experience_id', 'employer_id', 'desc_hash']])
    index = pd.concat(parts, ignore_index=True)
    stats = {'rows_after_filters': len(index)}
    index = index.drop_duplicates('id').drop_duplicates('desc_hash')
    stats['rows_after_dedup'] = len(index)
    return index, stats


def select_sample(index, size):
    # Proportional stratified sample: keeps the real class balance of the 2020 IT vacancies.
    if size >= len(index):
        return index
    sample, _ = train_test_split(index, train_size=size, stratify=index['experience_id'], random_state=SEED)
    return sample


def grouped_split(sample):
    # One fold of a stratified, employer-grouped 5-fold split is the test set (~20%): no employer
    # appears in both train and test, and class proportions stay close to the sample.
    splitter = StratifiedGroupKFold(n_splits=round(1 / TEST_SIZE), shuffle=True, random_state=SEED)
    train_idx, test_idx = next(splitter.split(sample, sample['experience_id'], sample['employer_id']))
    return sample.iloc[train_idx], sample.iloc[test_idx]


def read_rows(path, ids):
    # Pass 2: read all needed columns, but keep only the sampled IDs.
    wanted, parts = set(ids), []
    for chunk in pd.read_csv(path, usecols=RAW_COLUMNS, dtype={'id': str}, chunksize=CHUNK_ROWS):
        parts.append(chunk[chunk['id'].isin(wanted)])
    rows = pd.concat(parts, ignore_index=True).drop_duplicates('id')
    return rows.set_index('id', drop=False).loc[list(ids)].reset_index(drop=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Build the fixed sample and saved train/test split.')
    parser.add_argument('--raw', default=str(RAW_FILE), help='Path to vacancies_<year>.csv.gz from IEEE DataPort')
    parser.add_argument('--size', type=int, default=SAMPLE_SIZE)
    parser.add_argument('--force', action='store_true', help='Overwrite an existing saved split')
    args = parser.parse_args(argv)

    raw = Path(args.raw)
    raw = raw if raw.is_absolute() else ROOT / raw
    if not raw.is_file():
        raise SystemExit(f'Raw file not found: {raw}\nDownload vacancies_2020.csv.gz from '
                         'https://ieee-dataport.org/open-access/it-vacancies-hhru-2006-2020 '
                         f'and put it in {RAW_FILE.parent}')
    if TRAIN_IDS.exists() and not args.force:
        raise SystemExit('A saved split already exists. It defines the experiment; '
                         'rerun with --force only if you really want a new one.')

    check_columns(raw)
    print('Pass 1: scanning labels and descriptions...', flush=True)
    index, stats = candidate_index(raw)
    print(stats, flush=True)
    print(index['experience_id'].value_counts().to_string(), flush=True)
    sample = select_sample(index, args.size)
    train, test = grouped_split(sample)
    if set(train['employer_id']) & set(test['employer_id']):
        raise RuntimeError('An employer appears in both train and test')
    print('Pass 2: reading sampled rows...', flush=True)
    rows = read_rows(raw, sample['id'].tolist())

    PROCESSED.mkdir(parents=True, exist_ok=True)
    rows.to_csv(SAMPLE_FILE, index=False, compression={'method': 'gzip', 'mtime': 0})
    train[['id']].to_csv(TRAIN_IDS, index=False)
    test[['id']].to_csv(TEST_IDS, index=False)
    manifest = {
        'source': 'IEEE DataPort, IT vacancies from hh.ru 2006-2020, doi:10.21227/6naz-wb22, CC BY 4.0',
        'source_file': raw.name,
        'source_sha256': sha256(raw),
        'seed': SEED,
        'sampling': 'proportional stratified by experience_id',
        'filters': {'experience_id_in': LABELS, 'min_description_chars': MIN_DESCRIPTION_CHARS,
                    'dedup': ['id', 'exact description']},
        **stats,
        'sample_rows': len(rows),
        'train_rows': len(train),
        'test_rows': len(test),
        'split': 'StratifiedGroupKFold(5) first fold as test, grouped by employer_id',
        'train_employers': int(train['employer_id'].nunique()),
        'test_employers': int(test['employer_id'].nunique()),
        'train_classes': train['experience_id'].value_counts().to_dict(),
        'test_classes': test['experience_id'].value_counts().to_dict(),
        'population_classes': index['experience_id'].value_counts().to_dict(),
        'sample_classes': rows['experience_id'].value_counts().to_dict(),
        'files': processed_hashes(),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f'Saved {len(rows)} rows ({len(train)} train / {len(test)} test) to {PROCESSED}')


if __name__ == '__main__':
    main(sys.argv[1:])
