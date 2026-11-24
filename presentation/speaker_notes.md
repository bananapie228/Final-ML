# Speaker notes: Required Experience from IT Vacancies — Final Defense

Proposed speakers are in brackets; adjust them. Target is about 10 minutes plus about 5 for questions.

## 1. Does the vacancy text tell us how much experience an employer wants?

[Speaker: Diyar] About 30 seconds. Introduce both of us and the one-sentence question: can we predict the experience level an employer requires on hh.ru, and does reading the description text help beyond salary, skills and other fields? Say up front that the Final moved to real data.

## 2. Employers pick one of four experience levels. Can we predict which?

[Speaker: Diyar] About 40 seconds. On hh.ru every vacancy has an experience requirement chosen by the employer from these four options. This is our target, and it is a real label, not one we made up. The classes are imbalanced: half the vacancies ask for 1–3 years and fewer than 3% ask for more than 6. Because of that we use macro F1, which weights all four classes equally. Always guessing "1–3 years" already gives 52% accuracy, so accuracy alone is misleading.

## 3. What changed since the Endterm

[Speaker: Diyar] About 50 seconds. Be open about this: the earlier stages used a synthetic dataset, and the labels came from our own rule. For the Final we switched to real data with a real label. Because the data changed, the objects and the split are new, but the split is saved by ID and fixed with seed 42, as before. The Endterm also named a weakness: vacancies without salary had about three times more errors. The Endterm plan was to add the description text, and today we test whether that fixes the weakness.

## 4. Real 2020 hh.ru IT vacancies, split so no employer is on both sides

[Speaker: Diyar] About 50 seconds. The data is a public dataset of real hh.ru vacancies under a CC BY licence, so we may use it and share our sample with attribution. We did not scrape anything. We took the 2020 file, filtered and de-duplicated it to 391 thousand vacancies, and drew a fixed 20 thousand. The split is grouped by employer: big companies post many vacancies with the same boilerplate text, and a random split would let the text model memorise companies. Half of the vacancies have no salary, which matters later.

## 5. 44% of descriptions state the answer, so we mask it first

[Speaker: Galymzhan] About 60 seconds. This is the most important method slide. In 44% of training descriptions the employer writes the requirement as a number, for example "от 3-х лет". If we leave it in, the text model just reads the answer. So before any text model sees a description we remove the title, every experience phrase and seniority words. The example on the left is an illustration of the kind of text we mask, not a real vacancy. Be honest about the first version: it missed forms like "от 3-х лет" and "не менее". We found this by looking at the strongest TF-IDF words of the model trained on the training set, fixed the patterns, added tests for those forms and retrained every model from scratch. All numbers in this talk come from the corrected run.

## 6. Three kinds of input, compared on the same 3,999 test vacancies

[Speaker: Galymzhan] About 50 seconds. Three kinds of model. First, fields only, our baseline: the best is Random Forest. Second, a simple text model we trained ourselves: TF-IDF with logistic regression. Third, the deep-learning component: rubert-tiny2, a small pretrained Russian BERT. We use it frozen as a sentence encoder and train only a logistic regression on top, because it runs on a laptop CPU. Then combined models put fields and text together. Everything is tuned by grouped cross-validation on training data only, so test scores never influenced a choice.

## 7. Adding the text beats the best fields-only model

[Speaker: Galymzhan] About 50 seconds. Light bars are training cross-validation, dark bars are the held-out test, and the dashed line is always guessing the majority class. The best fields-only model, Random Forest, gets 0.397 macro F1. Text alone already beats it with 0.426, and fields plus text gets 0.452. Cross-validation and test agree, within about 0.02. The difference is significant: on the same 3,999 vacancies, the text model fixes 808 Random Forest errors and makes 632 new ones, and an exact McNemar test gives p about 4 times 10 to the minus 6.

## 8. The text helps exactly where the fields are empty

[Speaker: Galymzhan] About 50 seconds. This answers the Endterm question. We split the test set by whether the salary is given. Without salary, Random Forest falls to 0.31, and adding the text brings it back to 0.42, close to the 0.45 it gets when salary is present. When salary is present, text and fields score the same. The same holds for vacancies with no key skills. So the text matters most when the structured fields are missing, which is the weakness we named in the Endterm. The bar chart with all subgroups is in the report and in results/figures/subgroups.png.

## 9. The pretrained encoder lost to our own TF-IDF model

[Speaker: Galymzhan] About 50 seconds. The rubric asks us to compare the pretrained component with a simple model we trained ourselves. Our own TF-IDF model won clearly. The pretrained model even has the best accuracy, but only because it mostly predicts the biggest class: it finds almost none of the 6+ year vacancies. Our explanation: the encoder is frozen and general, it summarises what the job is about, not how senior it is, and it reads only the first 512 tokens. A paper from ACL 2026 found the same pattern for job grades. Fine-tuning the encoder could change this, but it does not fit on our 8 GB laptop.

## 10. Most mistakes are one level off

[Speaker: Galymzhan] About 50 seconds. Left is Random Forest, right is fields plus text. Most errors are next to the diagonal: 86% are one level off, mostly 1–3 versus 3–6 years. These boundaries are fuzzy in the data itself, because similar jobs get neighbouring labels. Two real test vacancies: a computer-club administrator with no salary. Fields alone guessed 1–3 years, while the text describes a front-desk job and the model correctly says no experience. A head of a BI department needs 6+ years, but the strongest clue is the title, which we hide on purpose, so both models miss it. More examples with IDs are in the report.

## 11. A local app turns a new vacancy into a prediction

[Speaker: Diyar] About 90 seconds including the demo. Switch to the app, which is already running. Demo 1: paste a junior-style vacancy, for example "ищем начинающего аналитика, научим работать с отчётностью", with skills SQL and Excel. The app predicts no experience with high probability. Demo 2: paste a senior vacancy with architecture and team-lead duties and a high salary; it predicts more than 6 years. Demo 3: leave salary empty, or set salary from above salary to, and show the error message. Open "What the model received" to show that the title and experience phrases were removed. Tip: check the app before the defense, because the first run takes a few seconds to load.

## 12. The text helps, mostly when salary or skills are missing

[Speaker: Galymzhan] About 40 seconds. One-sentence conclusion: reading the description significantly improves prediction of the required experience, and most of the gain comes where salary is missing. Our own simple TF-IDF model beat the pretrained encoder. Limits: neighbouring levels are fuzzy in the labels themselves, the 6+ class is rare, the data is 2020 hh.ru only, and masking may miss unusual phrasing. Next we would fine-tune the encoder, add the title, and model the four levels as ordered.

## 13. Contributions

[Both] About 20 seconds. Same split as in the Endterm: Diyar opened and runs the demo, Galymzhan covers methods and results. Each of us should be ready to explain any part of the project, including the partner’s.
