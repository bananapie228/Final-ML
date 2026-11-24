import hashlib
import json
import re

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from src.config import CACHE, CV_FOLDS, MODELS, REGISTRY, SEED, TABLES, TEXT_ENCODER
from src.features import MAX_TOKENS, embed_texts, preprocessor

TARGET = 'experience_id'


def cv_splitter():
    # Grouped by employer like the test split, so CV scores are not inflated by shared boilerplate.
    return StratifiedGroupKFold(CV_FOLDS, shuffle=True, random_state=SEED)


def slug(name):
    return re.sub(r'[^0-9a-z]+', '_', name.lower()).strip('_')


def load_registry():
    return json.loads(REGISTRY.read_text()) if REGISTRY.is_file() else {}


def save_registry(registry):
    MODELS.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text(json.dumps(registry, indent=2, default=str))


def fit_search(name, kind, estimator, grid, train, n_jobs=-1, **prep_kwargs):
    # Model choice uses training-fold macro F1 only; the test set is never touched here.
    pipe = Pipeline([('prep', preprocessor(kind, **prep_kwargs)), ('model', estimator)])
    search = GridSearchCV(pipe, grid or [{}], scoring='f1_macro', cv=cv_splitter(), n_jobs=n_jobs,
                          error_score='raise', return_train_score=True)
    search.fit(train, train[TARGET], groups=train['employer_id'])
    TABLES.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(search.cv_results_).to_csv(TABLES / f'search_{slug(name)}.csv', index=False)
    joblib.dump(search.best_estimator_, MODELS / f'{slug(name)}.joblib', compress=3)
    best = search.best_index_
    entry = {
        'file': f'{slug(name)}.joblib',
        'kind': kind,
        'needs_embeddings': 'embed' in kind.split('+'),
        'text_encoder': TEXT_ENCODER if 'embed' in kind.split('+') else None,
        'cv_mean': float(search.best_score_),
        'cv_std': float(search.cv_results_['std_test_score'][best]),
        'params': {k: v for k, v in search.best_params_.items()},
    }
    registry = load_registry()
    registry.setdefault('models', {})[name] = entry
    save_registry(registry)
    print(f"{name}: CV macro F1 {entry['cv_mean']:.4f} ± {entry['cv_std']:.4f} {entry['params']}", flush=True)
    return search.best_estimator_, entry


def embeddings_for(frame):
    # The encoder is a fixed pretrained transform (it learns nothing from our labels), so the
    # vectors are computed once for all rows and cached; classifiers on top are fit on train only.
    cache = CACHE / 'embeddings.npz'
    ids = frame['id'].astype(str).tolist()
    # The key covers the encoder, the token limit and the exact masked texts, so a change to
    # mask_text can never silently reuse stale vectors.
    key = hashlib.sha256('\x00'.join([TEXT_ENCODER, str(MAX_TOKENS), *frame['text']]).encode()).hexdigest()
    if cache.is_file():
        data = np.load(cache, allow_pickle=False)
        if 'key' in data and str(data['key']) == key and data['ids'].tolist() == ids:
            return data['vectors']
    vectors = embed_texts(frame['text'].tolist())
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, ids=np.array(ids), vectors=vectors, key=np.array(key))
    return vectors
