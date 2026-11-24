import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
import streamlit as st

from src.config import LABEL_NAMES, MODELS
from src.features import NUMERIC, add_embeddings, build_frame, embed_texts, parse_skills
from src.modeling import load_registry

SCHEDULES = {'fullDay': 'Full day', 'remote': 'Remote', 'flexible': 'Flexible', 'shift': 'Shift',
             'flyInFlyOut': 'Fly-in fly-out'}
EMPLOYMENT = {'full': 'Full time', 'part': 'Part time', 'project': 'Project', 'probation': 'Internship',
              'volunteer': 'Volunteer'}
GROSS = {'unknown': 'Not stated', 'false': 'After tax (net)', 'true': 'Before tax (gross)'}

st.set_page_config(page_title='Required experience predictor')
st.title('Required experience predictor')
st.caption('Predicts the experience an employer would require for an IT vacancy (hh.ru categories). '
           'Trained on 2020 hh.ru IT vacancies from IEEE DataPort (CC BY 4.0).')


@st.cache_resource
def load_model(file_name):
    return joblib.load(MODELS / file_name)


registry = load_registry()
models = {n: m for n, m in registry.get('models', {}).items() if n != 'Dummy'}
if not models:
    st.warning('No trained model yet. Run `python -m src.prepare_data`, `python -m src.train_structured`, '
               '`python -m src.train_text` and `python -m src.evaluate` first.')
    st.stop()

names = sorted(models, key=lambda n: -models[n]['cv_mean'])
default = registry.get('app_model')
choice = st.selectbox('Model', names, index=names.index(default) if default in names else 0)
info = models[choice]
st.caption(f"Training cross-validation macro F1: {info['cv_mean']:.3f} ± {info['cv_std']:.3f}")

with st.form('vacancy'):
    title = st.text_input('Vacancy title', placeholder='Python developer')
    c1, c2, c3 = st.columns(3)
    salary_from = c1.number_input('Salary from', min_value=0, value=None, step=10000)
    salary_to = c2.number_input('Salary to', min_value=0, value=None, step=10000)
    currency = c3.selectbox('Currency', ['RUR', 'KZT', 'USD', 'EUR', 'other'])
    c4, c5, c6 = st.columns(3)
    schedule = c4.selectbox('Schedule', list(SCHEDULES), format_func=SCHEDULES.get)
    employment = c5.selectbox('Employment', list(EMPLOYMENT), format_func=EMPLOYMENT.get)
    gross = c6.selectbox('Salary is', list(GROSS), format_func=GROSS.get)
    area = st.text_input('City', value='Москва')
    skills = st.text_input('Key skills, comma separated', placeholder='Python, Django, PostgreSQL')
    description = st.text_area('Vacancy description', height=220,
                               placeholder='Paste the vacancy text here')
    submitted = st.form_submit_button('Predict')

if submitted:
    if salary_from is not None and salary_to is not None and salary_from > salary_to:
        st.error('"Salary from" is larger than "Salary to". Please fix the salary range.')
        st.stop()
    if not description.strip() and not parse_skills(skills):
        st.error('Enter at least a description or some key skills, otherwise there is nothing to predict from.')
        st.stop()
    if info['kind'] != 'structured' and not description.strip():
        st.info('The description is empty, so the text part of this model has nothing to read. '
                'The prediction relies on the other fields.')

    raw = pd.DataFrame([{
        'id': 'app-input', 'name': title, 'description': description, 'key_skills': skills,
        'salary_from': salary_from, 'salary_to': salary_to,
        'salary_currency': currency if salary_from is not None or salary_to is not None else None,
        'salary_gross': None if gross == 'unknown' else gross, 'schedule_id': schedule,
        'employment_id': employment, 'area_name': area or None,
    }])
    try:
        frame = build_frame(raw)
        if info['needs_embeddings']:
            with st.spinner('Encoding the description with the pretrained text model...'):
                frame = add_embeddings(frame, embed_texts(frame['text'].tolist(), info['text_encoder']))
        model = load_model(info['file'])
        proba = model.predict_proba(frame)[0]
    except Exception as error:
        st.error(f'Prediction failed: {error}')
        st.stop()

    order = proba.argsort()[::-1]
    best = model.classes_[order[0]]
    st.success(f'Predicted required experience: **{LABEL_NAMES.get(best, best)}** '
               f'(probability {proba[order[0]]:.0%})')
    chart = pd.DataFrame({'probability': proba},
                         index=[LABEL_NAMES.get(c, c) for c in model.classes_])
    st.bar_chart(chart)
    with st.expander('What the model received'):
        st.write('Description after cleaning and masking of title and experience phrases:')
        st.code(frame['text'].iloc[0][:1500] or '(empty)')
        st.write('Parsed skills:', parse_skills(skills) or '(none)')
        st.dataframe(frame[NUMERIC + ['schedule_id', 'employment_id', 'salary_gross', 'area_name']])
