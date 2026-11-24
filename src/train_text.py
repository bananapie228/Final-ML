import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from src.config import SEED, TABLES
from src.data import load_dataset
from src.features import LEAK_PATTERNS, add_embeddings
from src.modeling import embeddings_for, fit_search

LR_GRID = {'model__C': [0.3, 1, 3, 10], 'model__class_weight': [None, 'balanced']}


def logistic():
    return LogisticRegression(max_iter=3000, random_state=SEED)


def text_checks(train):
    # Evidence that masking works: how often each leak pattern fired, and before/after examples.
    TABLES.mkdir(parents=True, exist_ok=True)
    source = train['description_clean']
    audit = [{'pattern': p.pattern[:80], 'train_rows_matched': int(source.map(lambda t: bool(p.search(t))).sum()),
              'share': float(source.map(lambda t: bool(p.search(t))).mean())} for p in LEAK_PATTERNS]
    pd.DataFrame(audit).to_csv(TABLES / 'text_leak_audit.csv', index=False)
    examples = train.sample(min(10, len(train)), random_state=SEED)
    pd.DataFrame({'id': examples['id'], 'title': examples['name'], 'label': examples['experience_id'],
                  'original': examples['description_clean'].str[:500],
                  'masked': examples['text'].str[:500]}).to_csv(TABLES / 'text_examples.csv', index=False)


def top_tfidf_terms(pipeline, k=25):
    # Strongest words per class; used to check that no experience wording survived masking.
    vectorizer = pipeline.named_steps['prep'].named_transformers_['text']
    model = pipeline.named_steps['model']
    terms = np.array(vectorizer.get_feature_names_out())
    rows = []
    for label, coefs in zip(model.classes_, model.coef_):
        for rank, idx in enumerate(np.argsort(coefs)[::-1][:k], 1):
            rows.append({'class': label, 'rank': rank, 'term': terms[idx], 'coef': coefs[idx]})
    pd.DataFrame(rows).to_csv(TABLES / 'tfidf_top_terms.csv', index=False)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-tfidf', action='store_true', help='Only (re)train the embedding models')
    args = parser.parse_args(argv)
    frame, train, test = load_dataset()
    print(f'Train {len(train)} rows, test {len(test)} rows (test is not used here)', flush=True)
    text_checks(train)

    if not args.skip_tfidf:
        tfidf_model, _ = fit_search('TF-IDF + LR', 'tfidf', logistic(), LR_GRID, train)
        top_tfidf_terms(tfidf_model)
        fit_search('Structured + TF-IDF + LR', 'structured+tfidf', logistic(), LR_GRID, train)

    print('Encoding descriptions with the pretrained model (cached after the first run)...', flush=True)
    train_e = add_embeddings(frame, embeddings_for(frame)).loc[train.index]
    fit_search('rubert-tiny2 + LR', 'embed', logistic(),
               {'model__C': [0.1, 1, 10], 'model__class_weight': [None, 'balanced']}, train_e)
    fit_search('Structured + rubert-tiny2 + Boosting', 'structured+embed',
               HistGradientBoostingClassifier(random_state=SEED, early_stopping=False),
               {'model__learning_rate': [0.05, 0.1], 'model__max_leaf_nodes': [15, 31],
                'model__class_weight': [None, 'balanced']}, train_e)


if __name__ == '__main__':
    main()
