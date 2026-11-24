import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from src.config import LABELS, SEED
from src.features import (NUMERIC, build_frame, clean_html, mask_text, metrics, parse_skills,
                          preprocessor, validate_split)


def toy_raw(n=24):
    rng = np.random.default_rng(SEED)
    skills = ['["Python", "Django"]', "['SQL', 'Git']", 'Java, Spring', '[{"name": "Docker"}]', '']
    rows = []
    for i in range(n):
        has_salary = i % 3 != 0
        rows.append({
            'id': str(1000 + i),
            'name': ['Junior Python developer', 'Java разработчик', 'Senior DevOps'][i % 3],
            'description': f'<p>Мы ищем разработчика.</p> Опыт работы от {i % 5 + 1} лет. '
                           f'Работа с Python и SQL, задача номер {i}.',
            'key_skills': skills[i % len(skills)],
            'experience_id': LABELS[i % len(LABELS)],
            'schedule_id': ['fullDay', 'remote'][i % 2],
            'employment_id': 'full',
            'salary_from': float(rng.integers(50, 300) * 1000) if has_salary else np.nan,
            'salary_to': np.nan,
            'salary_currency': ['RUR', 'USD'][i % 2] if has_salary else np.nan,
            'salary_gross': [True, False][i % 2] if has_salary else np.nan,
            'area_name': ['Москва', 'Санкт-Петербург', 'Алматы'][i % 3],
        })
    return pd.DataFrame(rows)


def test_clean_html_strips_tags_and_entities():
    assert clean_html('<p>Python&nbsp;и <b>SQL</b></p>') == 'Python и SQL'
    assert clean_html(None) == ''


@pytest.mark.parametrize('value,expected', [
    ('["Python", "Django"]', ['Python', 'Django']),
    ("['SQL', 'Git']", ['SQL', 'Git']),
    ('Java, Spring; Kafka', ['Java', 'Spring', 'Kafka']),
    ('[{"name": "Docker"}, {"name": "docker"}]', ['Docker']),
    ('', []),
    (np.nan, []),
])
def test_parse_skills_formats(value, expected):
    assert parse_skills(value) == expected


@pytest.mark.parametrize('text', [
    'Требуется опыт работы от 3 лет с Python',
    'Опыт 1-3 года в разработке',
    'Опыт от трех лет',
    'Опыт не менее года',
    'At least 5+ years of experience',
    'Можно без опыта',
    'Ищем Senior разработчика, ведущий инженер',
    'Стажёр в команду',
    'Опыт работы от 3-х лет в разработке',
    'Опыт работы от 2х лет',
    'Опыт не менее 5-ти лет',
    'Опытом работы более 3 лет',
    'Опыт от 3 (трех) лет',
    'Опыт работы приветствуется, но не обязателен',
    'Опыт работы с Python не требуется',
    'Без опыта работы, научим',
])
def test_mask_text_removes_experience_wording(text):
    masked = mask_text(text).lower()
    for leak in ['лет', 'года', 'менее', 'более', 'от 3', 'от 2', '3х', '2х', '5+ years', 'без опыта',
                 'senior', 'ведущий', 'стажёр', 'приветствуется', 'обязател', 'не требуется', 'трех']:
        assert leak not in masked, (text, masked)


def test_mask_text_keeps_unrelated_years_and_skills():
    masked = mask_text('Подарки на новый год. Опыт работы с Kafka и PostgreSQL. Зарплата от 150000 руб.')
    assert 'новый год' in masked and 'Kafka' in masked and '150000' in masked


def test_mask_text_removes_title_and_keeps_content():
    masked = mask_text('<p>Вакансия Python developer в финтех. Пишем на Django.</p>', 'Python developer')
    assert 'python developer' not in masked.lower()
    assert 'Django' in masked


def test_build_frame_salary_features():
    frame = build_frame(toy_raw())
    rur = frame[(frame.salary_missing == 0) & (frame.salary_other_currency == 0)]
    assert len(rur) and rur.salary_log.notna().all()
    assert frame.loc[frame.salary_other_currency == 1, 'salary_log'].isna().all()
    assert frame.loc[frame.salary_missing == 1, 'salary_log'].isna().all()
    assert (frame.n_skills >= 0).all() and frame.text.str.contains('лет').sum() == 0


@pytest.mark.parametrize('defect', ['duplicate', 'overlap', 'missing', 'unknown'])
def test_bad_splits_rejected(defect):
    ids = [str(i) for i in range(10)]
    train, test = ids[:8], ids[8:]
    if defect == 'duplicate':
        train[1] = train[0]
    if defect == 'overlap':
        test[0] = train[0]
    if defect == 'missing':
        train.pop()
    if defect == 'unknown':
        train[0] = 'nonexistent'
    with pytest.raises(ValueError):
        validate_split(ids, train, test)


def test_valid_split_accepted():
    ids = [str(i) for i in range(10)]
    assert validate_split(ids, ids[:8], ids[8:]) == (ids[:8], ids[8:])


@pytest.mark.parametrize('kind', ['structured', 'tfidf', 'structured+tfidf'])
def test_preprocessors_handle_unseen_values(kind):
    frame = build_frame(toy_raw())
    prep = preprocessor(kind, min_df=1).fit(frame)
    changed = frame.copy()
    changed['schedule_id'] = 'unseen'
    changed['area_name'] = 'Unknown city'
    changed['skills_joined'] = 'never-seen-skill'
    out = prep.transform(changed)
    out = out.toarray() if hasattr(out, 'toarray') else out
    assert np.isfinite(out).all()


def test_structured_excludes_title_and_target():
    frame = build_frame(toy_raw())
    prep = preprocessor('structured').fit(frame)
    changed = frame.copy()
    changed['name'] = 'Senior Lead Architect'
    changed['experience_id'] = 'moreThan6'
    np.testing.assert_array_equal(prep.transform(frame), prep.transform(changed))


def test_imputer_learns_training_fold_only():
    frame = build_frame(toy_raw())
    for tr, va in StratifiedKFold(3, shuffle=True, random_state=SEED).split(frame, frame.experience_id):
        prep = preprocessor('structured').fit(frame.iloc[tr])
        imputer = prep.named_transformers_['numeric'].named_steps['impute']
        np.testing.assert_allclose(imputer.statistics_, frame.iloc[tr][NUMERIC].median().to_numpy())


def test_unknown_kind_rejected():
    with pytest.raises(ValueError):
        preprocessor('images')


def test_metrics_macro_f1():
    result = metrics(['noExperience', 'between1And3'], ['noExperience', 'noExperience'])
    assert result['accuracy'] == pytest.approx(0.5)
    assert 0 <= result['macro_f1'] <= 1


def test_grouped_split_keeps_employers_apart():
    from src.prepare_data import grouped_split
    rng = np.random.default_rng(SEED)
    sample = pd.DataFrame({'id': [str(i) for i in range(400)],
                           'experience_id': rng.choice(LABELS, 400, p=[0.2, 0.5, 0.25, 0.05]),
                           'employer_id': rng.integers(0, 80, 400).astype(str)})
    train, test = grouped_split(sample)
    assert not set(train.employer_id) & set(test.employer_id)
    assert len(train) + len(test) == 400 and 0.1 < len(test) / 400 < 0.3
