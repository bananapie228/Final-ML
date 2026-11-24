from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = ROOT / 'data/raw/vacancies_2020.csv.gz'
PROCESSED = ROOT / 'data/processed'
SAMPLE_FILE = PROCESSED / 'vacancies_sample.csv.gz'
TRAIN_IDS = PROCESSED / 'train_ids.csv'
TEST_IDS = PROCESSED / 'test_ids.csv'
MANIFEST = PROCESSED / 'manifest.json'

RESULTS = ROOT / 'results'
TABLES = RESULTS / 'tables'
FIGURES = RESULTS / 'figures'
CACHE = RESULTS / 'cache'
MODELS = ROOT / 'models'
REGISTRY = MODELS / 'registry.json'

SEED = 42
# moreThan6 is only ~2% of 2020 IT vacancies, so the sample is large enough to keep a few hundred of them.
SAMPLE_SIZE = 20000
TEST_SIZE = 0.2
CV_FOLDS = 5
MIN_DESCRIPTION_CHARS = 200

# Target: required experience chosen by the employer on hh.ru (experience_id).
LABELS = ['noExperience', 'between1And3', 'between3And6', 'moreThan6']
LABEL_NAMES = {
    'noExperience': 'No experience',
    'between1And3': '1-3 years',
    'between3And6': '3-6 years',
    'moreThan6': 'More than 6 years',
}

# Pretrained sentence encoder for Russian text (small enough for CPU / Apple MPS).
TEXT_ENCODER = 'cointegrated/rubert-tiny2'

# Raw columns used from the IEEE DataPort file (names as listed on the dataset page).
RAW_COLUMNS = [
    'id', 'name', 'description', 'key_skills', 'experience_id', 'schedule_id',
    'employment_id', 'salary_from', 'salary_to', 'salary_currency', 'salary_gross',
    'area_name', 'published_at', 'employer_id',
]
# employer_id is not a model feature: it only groups the split and the CV folds, so vacancies from one
# employer (which often share boilerplate text) never sit on both sides.
