# Required Experience from IT Vacancies (Final)

Machine Learning Algorithms, Astana IT University
Zholdubayev Diyar and Galymzhan Turemuratov

**Question:** can we predict the experience an employer requires (no experience, 1-3, 3-6 or 6+ years)
from a real hh.ru IT vacancy? Does the vacancy text, read by a deep-learning text model, add value over
structured fields such as salary, skills, schedule and city?

**Data:** real 2020 IT vacancies from hh.ru, published on IEEE DataPort under CC BY 4.0. See [data/README.md](data/README.md).
Midterm and Endterm used a synthetic dataset. The Final moves to real data, and we document this change.

## Setup
Python 3.12 is the tested version (3.12.16 on macOS, Apple M2).
`requirements.txt` pins the direct dependencies, and `requirements-lock.txt` records the full tested environment.
The saved `.joblib` models need the same scikit-learn version to load.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run order
```sh
python -m src.prepare_data       # once: sample + saved split from data/raw/vacancies_2020.csv.gz
python -m src.train_structured   # Dummy, Logistic Regression, Random Forest, Boosting on structured fields
python -m src.train_text         # TF-IDF + LR (own model), rubert-tiny2 embeddings + LR (pretrained), combined models
python -m src.evaluate           # one comparison on the same test IDs, errors, subgroups, app model choice
python -m pytest -q              # tests
streamlit run app/app.py         # local prediction app
```

The executed notebook [notebooks/Final.ipynb](notebooks/Final.ipynb) walks through the data, masking checks, results, the Endterm hypothesis, real errors and a new prediction. It uses the saved models and recomputes every test metric (about a minute). Set `RUN_TRAINING = True` in it to retrain from scratch. Open it in Jupyter, VS Code or PyCharm with the `.venv` kernel, or run it headless:

```sh
cd notebooks && python -c "import nbformat; from nbclient import NotebookClient; nb=nbformat.read('Final.ipynb',4); NotebookClient(nb, timeout=900, kernel_name='python3').execute(); nbformat.write(nb,'Final.ipynb')"
```

## Method in short
- **Split:** a fixed stratified 80/20 split, saved as ID files, with seed 42. All model choices use 5-fold cross-validation macro F1 on the training set only. The test set is used once, in `evaluate`.
- **Leakage control:** the vacancy title and any experience or seniority wording (for example «опыт от 3 лет», «без опыта», senior, junior, стажёр) are removed from the description before any text model sees it. The structured models do not use the title.
- **Text:** TF-IDF + logistic regression is the simple model trained by us. Sentence embeddings from the pretrained `cointegrated/rubert-tiny2` + logistic regression are the deep-learning component. Combined models add the text to the structured fields.
- **Metric:** macro F1, which treats all four experience classes equally.

## Results (test set: 3,999 vacancies; employers in the test set never appear in training)
Models are ranked by training cross-validation macro F1. The test set was not used to choose any model.

| Model | Input | CV macro F1 | Test macro F1 | Test accuracy |
| --- | --- | --- | --- | --- |
| Structured + TF-IDF + LR | fields + text | 0.435 ± 0.013 | **0.452** | 0.532 |
| TF-IDF + LR (own model) | text | 0.426 ± 0.019 | 0.426 | 0.514 |
| Structured + rubert-tiny2 + Boosting | fields + text | 0.406 ± 0.018 | 0.413 | 0.490 |
| Random Forest (best structured) | fields | 0.376 ± 0.023 | 0.397 | 0.488 |
| Boosting | fields | 0.358 ± 0.020 | 0.364 | 0.426 |
| rubert-tiny2 + LR (pretrained) | text | 0.348 ± 0.016 | 0.369 | 0.559 |
| Logistic Regression | fields | 0.326 ± 0.009 | 0.336 | 0.551 |
| Dummy (majority) | none | 0.172 | 0.172 | 0.523 |

- **Text helps.** On the same test vacancies, the best text model is right where Random Forest is wrong 808 times, and the reverse happens 632 times. The exact McNemar test gives p ≈ 4e-6.
- **Where text helps:** when salary is missing, macro F1 is 0.309 for structured only and 0.424 with text. When there are no key skills, it is 0.375 and 0.456. When salary is present, the two are almost equal (0.447 and 0.447). So text fills the gap left by missing structured fields.
- **The pretrained encoder lost to our own TF-IDF model** (0.348 and 0.426 CV). rubert-tiny2 is a small general-purpose model that reads only the first 512 tokens, while TF-IDF can weight specific skill and duty words.
- **"More than 6 years" stays hard:** F1 is 0.19 even with text. It is about 3% of vacancies (112 in the test set).

Tables are in `results/tables/` and the confusion matrices in `results/figures/`.

## Notes on process (for the report)
- **The first masking version was incomplete.** It missed forms like «от 3-х лет», «5-ти лет», «не менее …» and «опыт … не обязателен». We found this from the strongest TF-IDF terms of the training-set model, fixed the patterns, added tests and retrained everything from scratch. An earlier evaluation with the leaky text had already been run, but nothing about model choice or parameters was changed because of test scores.
- **Some cues remain after masking:** «опыт», «стаж работы», «научим», «начинающего». These are ordinary vacancy content, not explicit statements of the label, but they are close to it.
- **The encoder runs on the CPU with a 512-token limit.** On an 8 GB Apple M2, the GPU (MPS) ran out of memory.

The full written report is in [REPORT.md](REPORT.md).

## Layout
- `src/`: data preparation, features, training, evaluation
- `app/`: Streamlit app
- `notebooks/Final.ipynb`: executed walkthrough with outputs
- `tests/`: tests
- `data/`: sample, saved split and manifest
- `models/`: saved pipelines + `registry.json`
- `results/`: tables and figures
