from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.config import SEED
from src.data import load_dataset
from src.modeling import fit_search

SPECS = [
    ('Dummy', DummyClassifier(strategy='most_frequent'), None),
    ('Logistic Regression', LogisticRegression(max_iter=3000, random_state=SEED),
     {'model__C': [0.1, 1, 10], 'model__class_weight': [None, 'balanced']}),
    ('Random Forest', RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=1),
     {'model__max_depth': [None, 20], 'model__min_samples_leaf': [1, 5],
      'model__class_weight': [None, 'balanced_subsample']}),
    ('Boosting', HistGradientBoostingClassifier(random_state=SEED, early_stopping=False),
     {'model__learning_rate': [0.05, 0.1], 'model__max_leaf_nodes': [15, 31],
      'model__l2_regularization': [0, 1], 'model__class_weight': [None, 'balanced']}),
]


def main():
    frame, train, test = load_dataset()
    print(f'Train {len(train)} rows, test {len(test)} rows (test is not used here)', flush=True)
    for name, estimator, grid in SPECS:
        fit_search(name, 'structured', estimator, grid, train)


if __name__ == '__main__':
    main()
