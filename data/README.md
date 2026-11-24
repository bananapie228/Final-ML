# Data

## Source
Aleksandr Bersenev, Andrey Sozykin, Denis Shadrin, Anton Koshelev, Evgeniy Kuklin, Alexander Aksenov,
"IT vacancies from hh.ru, 2006-2020", IEEE DataPort, March 4, 2021, doi:10.21227/6naz-wb22.
https://ieee-dataport.org/open-access/it-vacancies-hhru-2006-2020

License: Creative Commons Attribution 4.0 (CC BY 4.0). The authors collected the vacancies through the public hh.ru API.
Files can be downloaded with a free IEEE account. IEEE membership is not required.

## How to reproduce
1. Download `vacancies_2020.csv.gz` (about 530 MB) from the page above.
2. Put it in `data/raw/`. It is not committed because of its size.
3. Run `python -m src.prepare_data`.

## What `prepare_data` creates (in `data/processed/`, committed)
- `vacancies_sample.csv.gz`: a fixed stratified sample of real vacancies. It is redistributed under CC BY 4.0 with the attribution above.
- `train_ids.csv` and `test_ids.csv`: the saved 80/20 stratified split. This split defines the experiment and is never regenerated.
- `manifest.json`: filters, class counts, the source file hash and SHA-256 hashes of the three files. Training and evaluation refuse to run if these files change.

## Target
`experience_id` is the required experience that the employer chose on hh.ru: noExperience, between1And3, between3And6 or moreThan6.
It is a real label set by the employer, not a heuristic.
