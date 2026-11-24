import hashlib
import json

import pandas as pd

from src.config import MANIFEST, ROOT, SAMPLE_FILE, TEST_IDS, TRAIN_IDS
from src.features import build_frame, validate_split

PROCESSED_FILES = [SAMPLE_FILE, TRAIN_IDS, TEST_IDS]


def sha256(path, block=1 << 20):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def processed_hashes():
    return {str(p.relative_to(ROOT)): sha256(p) for p in PROCESSED_FILES}


def verify_processed():
    # The manifest is written once by prepare_data and never rewritten by training code,
    # so any later change to the sample or the saved split is detected here.
    missing = [str(p.relative_to(ROOT)) for p in PROCESSED_FILES + [MANIFEST] if not p.is_file()]
    if missing:
        raise FileNotFoundError(f'Missing processed files {missing}. Run: python -m src.prepare_data')
    expected = json.loads(MANIFEST.read_text())['files']
    actual = processed_hashes()
    if actual != expected:
        changed = sorted(k for k in expected if expected.get(k) != actual.get(k))
        raise RuntimeError(f'Processed data or saved split changed since prepare_data: {changed}')


def load_dataset():
    verify_processed()
    raw = pd.read_csv(SAMPLE_FILE, dtype={'id': str, 'employer_id': str})
    if raw['id'].isna().any() or not raw['id'].is_unique:
        raise ValueError('Missing or duplicate vacancy IDs in the sample')
    raw['employer_id'] = raw['employer_id'].fillna('no-employer-' + raw['id'])
    frame = build_frame(raw).set_index('id', drop=False)
    train_ids = pd.read_csv(TRAIN_IDS, dtype=str)['id']
    test_ids = pd.read_csv(TEST_IDS, dtype=str)['id']
    train_ids, test_ids = validate_split(frame.index, train_ids, test_ids)
    return frame, frame.loc[train_ids].copy(), frame.loc[test_ids].copy()
