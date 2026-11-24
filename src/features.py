import ast
import html
import json
import re
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import LABELS, TEXT_ENCODER

TAG = re.compile(r'<[^>]+>')
SPACE = re.compile(r'\s+')

NUMERIC = ['salary_log', 'salary_missing', 'salary_other_currency', 'n_skills', 'desc_words']
CATEGORICAL = ['schedule_id', 'employment_id', 'salary_gross']
EMBEDDING_PATTERN = r'^emb_\d+$'

# Phrases that state the required experience or seniority directly. They are removed from the
# description so text models cannot read the answer (the employer's experience_id) off the page.
# The quantifier ("не менее", "от", ...) is removed together with the number, otherwise it stays
# behind as a cue. Numbers may carry Russian suffixes: 3х, 3-х, 5-ти, 1-го.
_QUANT = (r'(?:\b(?:не\s+)?(?:менее|более|больше|меньше|от|свыше|минимум|до|at\s+least|over|from|min\.?)'
          r'\s*)?')
_NUM = r'\d+(?:[.,]\d+)?(?:\s*-?\s*(?:х|x|ти|ми|ух|ех|го|ого|и|ый|ой)(?![а-яёa-z]))?'
_WORDNUM = r'(?:одного|двух|тр[её]х|четыр[её]х|пяти|шести|семи|восьми|десяти|полугода)'
LEAK_PATTERNS = [re.compile(p, re.IGNORECASE) for p in [
    rf'{_QUANT}{_NUM}(?:\s*(?:\+|-|–|—|до|или)\s*{_NUM})?\s*\+?\s*(?:\([а-яё\s-]+\)\s*)?'
    r'(?:год(?:а|ов)?|лет|years?|yrs?)\b',
    rf'{_QUANT}{_WORDNUM}(?:\s*\(\s*\d+\s*\))?\s*(?:лет|года)?\b',
    r'\b(?:не\s+)?(?:менее|более|больше|от|свыше|минимум)\s+(?:года|год)\b',
    r'\b(?:без опыта(?: работы)?|нет опыта|no experience|without experience|entry[- ]level)\b',
    r'\bопыт\w*(?:\s+[\w/-]+){0,5}?\s+(?:не\s+)?(?:обязател\w*|требуется|нужен|важен|приветствуется|'
    r'желател\w*|будет плюсом|is a plus|not required)',
    r'\b(?:но\s+)?не\s+(?:обязател\w*|требуется)\b',
    r'\b(?:junior|middle|senior|team\s*lead|lead|intern|internship|trainee|head of|'
    r'джун\w*|мидл\w*|сеньор\w*|синьор\w*|стаж[её]р\w*|стажировк\w*|младш\w*|старш\w*|'
    r'ведущ\w*|тимлид\w*|лид)\b',
]]


def clean_html(text):
    if not isinstance(text, str):
        return ''
    text = html.unescape(TAG.sub(' ', text))
    return SPACE.sub(' ', text).strip()


def parse_skills(value):
    if isinstance(value, (list, tuple)):
        items = list(value)
    elif not isinstance(value, str) or not value.strip():
        return []
    else:
        text = value.strip()
        items = None
        if text[0] in '[(':
            for loader in (json.loads, ast.literal_eval):
                try:
                    parsed = loader(text)
                except (ValueError, SyntaxError):
                    continue
                if isinstance(parsed, (list, tuple)):
                    items = list(parsed)
                    break
        if items is None:
            items = re.split(r'[,;|\n]', text.strip('[]()'))
    skills, seen = [], set()
    for item in items:
        if isinstance(item, dict):
            item = item.get('name', '')
        name = SPACE.sub(' ', str(item)).strip().strip('\'"').strip()
        if name and name.lower() not in seen:
            seen.add(name.lower())
            skills.append(name)
    return skills


def mask_text(text, title=None):
    text = clean_html(text)
    if isinstance(title, str) and title.strip():
        text = re.sub(re.escape(title.strip()), ' ', text, flags=re.IGNORECASE)
    for pattern in LEAK_PATTERNS:
        text = pattern.sub(' ', text)
    return SPACE.sub(' ', text).strip()


def build_frame(raw):
    # Deterministic per-row transformations only: nothing here learns from other rows,
    # so it is safe to apply before the train/test split.
    df = raw.copy()
    for col in ['name', 'description', 'key_skills', 'salary_currency', 'salary_gross',
                'schedule_id', 'employment_id', 'area_name', 'salary_from', 'salary_to']:
        if col not in df:
            df[col] = np.nan
    df['description_clean'] = df['description'].map(clean_html)
    df['text'] = [mask_text(d, t) for d, t in zip(df['description_clean'], df['name'])]
    skills = df['key_skills'].map(parse_skills)
    df['skills_joined'] = skills.map(lambda s: '|'.join(x.lower() for x in s))
    df['n_skills'] = skills.map(len)
    df['desc_words'] = df['text'].map(lambda t: len(t.split()))

    salary_from = pd.to_numeric(df['salary_from'], errors='coerce')
    salary_to = pd.to_numeric(df['salary_to'], errors='coerce')
    salary_mid = pd.concat([salary_from, salary_to], axis=1).mean(axis=1)
    rur = df['salary_currency'].astype('string').str.upper().eq('RUR').fillna(False).astype(bool)
    df['salary_missing'] = salary_mid.isna().astype(int)
    df['salary_other_currency'] = (salary_mid.notna() & ~rur).astype(int)
    df['salary_log'] = np.log1p(salary_mid.where(rur & salary_mid.gt(0)))

    df['salary_gross'] = df['salary_gross'].astype('string').str.lower().fillna('unknown').astype(str)
    for col in ['schedule_id', 'employment_id', 'area_name']:
        df[col] = df[col].astype('string').fillna('unknown').astype(str)
    return df


def skill_tokens(joined):
    return [token for token in joined.split('|') if token]


def _structured_parts(max_skills):
    numeric = Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
    return [
        ('numeric', numeric, NUMERIC),
        ('categorical', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL),
        ('area', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=30,
                               sparse_output=False), ['area_name']),
        ('skills', CountVectorizer(tokenizer=skill_tokens, token_pattern=None, lowercase=False,
                                   binary=True, max_features=max_skills), 'skills_joined'),
    ]


def preprocessor(kind, max_skills=60, tfidf_features=20000, min_df=2):
    # kind: 'structured', 'tfidf', 'embed', 'structured+tfidf' or 'structured+embed'
    blocks = set(kind.split('+'))
    unknown = blocks - {'structured', 'tfidf', 'embed'}
    if unknown or not blocks:
        raise ValueError(f'Unknown feature kind: {kind}')
    parts = []
    if 'structured' in blocks:
        parts += _structured_parts(max_skills)
    if 'tfidf' in blocks:
        parts.append(('text', TfidfVectorizer(max_features=tfidf_features, ngram_range=(1, 2),
                                              min_df=min_df, sublinear_tf=True), 'text'))
    if 'embed' in blocks:
        parts.append(('embedding', StandardScaler(), make_column_selector(pattern=EMBEDDING_PATTERN)))
    return ColumnTransformer(parts, remainder='drop', sparse_threshold=0.3 if 'tfidf' in blocks else 0)


MAX_TOKENS = 512


@lru_cache(maxsize=2)
def load_encoder(model_name=TEXT_ENCODER):
    from sentence_transformers import SentenceTransformer
    # CPU on purpose: on an 8 GB Apple M2 the GPU (MPS) shares memory with the system and ran out
    # of memory. rubert-tiny2 accepts 2048 tokens; the first 512 (the usual BERT limit) are used.
    model = SentenceTransformer(model_name, device='cpu')
    model.max_seq_length = MAX_TOKENS
    return model


def embed_texts(texts, model_name=TEXT_ENCODER, batch_size=16):
    texts = [t if isinstance(t, str) and t.strip() else '' for t in texts]
    return load_encoder(model_name).encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                           convert_to_numpy=True, show_progress_bar=len(texts) > 200)


def add_embeddings(frame, vectors):
    vectors = np.asarray(vectors)
    if vectors.shape[0] != len(frame):
        raise ValueError('Embedding rows do not match the frame')
    cols = [f'emb_{i}' for i in range(vectors.shape[1])]
    return pd.concat([frame, pd.DataFrame(vectors, index=frame.index, columns=cols)], axis=1)


def validate_split(all_ids, train_ids, test_ids):
    train, test = [str(i) for i in train_ids], [str(i) for i in test_ids]
    if len(set(train)) != len(train) or len(set(test)) != len(test):
        raise ValueError('Duplicate split IDs')
    if set(train) & set(test):
        raise ValueError('Overlapping split')
    if set(train) | set(test) != {str(i) for i in all_ids}:
        raise ValueError('Split does not cover exactly the dataset')
    return train, test


def metrics(y_true, y_pred):
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'macro_f1': f1_score(y_true, y_pred, labels=LABELS, average='macro', zero_division=0),
        'macro_precision': precision_score(y_true, y_pred, labels=LABELS, average='macro', zero_division=0),
        'macro_recall': recall_score(y_true, y_pred, labels=LABELS, average='macro', zero_division=0),
    }
