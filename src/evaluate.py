import json

import joblib
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, f1_score

from src.config import FIGURES, LABEL_NAMES, LABELS, MODELS, RESULTS, TABLES
from src.data import load_dataset
from src.features import add_embeddings, metrics
from src.modeling import TARGET, embeddings_for, load_registry, save_registry


def best_by_cv(models, names):
    return max(names, key=lambda n: models[n]['cv_mean']) if names else None


def paired_test(y, a, b):
    # Exact McNemar test on the test rows where exactly one of the two models is right.
    a_only = int(((a == y) & (b != y)).sum())
    b_only = int(((b == y) & (a != y)).sum())
    n = a_only + b_only
    p = binomtest(min(a_only, b_only), n, 0.5).pvalue if n else 1.0
    return {'only_first_correct': a_only, 'only_second_correct': b_only, 'exact_mcnemar_p': float(p)}


def subgroup_table(test, preds, names, train):
    cuts = train['desc_words'].quantile([1 / 3, 2 / 3]).tolist()
    groups = {
        'salary missing': test['salary_missing'] == 1,
        'salary present': test['salary_missing'] == 0,
        'no key skills': test['n_skills'] == 0,
        'short description': test['desc_words'] <= cuts[0],
        'long description': test['desc_words'] > cuts[1],
    }
    rows = []
    for group, mask in groups.items():
        for name in names:
            sub = preds[mask]
            rows.append({'group': group, 'model': name, 'n': int(mask.sum()),
                         'macro_f1': f1_score(sub[TARGET], sub[name], labels=LABELS, average='macro',
                                              zero_division=0) if len(sub) else float('nan')})
    return pd.DataFrame(rows)


def comparison_figure(comparison):
    data = comparison[comparison['model'] != 'Dummy'].iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    y = range(len(data))
    ax.barh([i + 0.2 for i in y], data['cv_macro_f1'], 0.4, xerr=data['cv_std'], label='Training CV (mean ± SD)',
            color='#9bb7d4')
    ax.barh([i - 0.2 for i in y], data['test_macro_f1'], 0.4, label='Held-out test', color='#1f4e79')
    ax.set_yticks(list(y), data['model'])
    dummy = comparison.loc[comparison['model'] == 'Dummy', 'test_macro_f1']
    if len(dummy):
        ax.axvline(float(dummy.iloc[0]), color='grey', ls='--', lw=1, label='Majority-class baseline')
    ax.set_xlabel('Macro F1')
    ax.set_xlim(0, 0.6)
    ax.legend(loc='lower right', fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / 'model_comparison.png', dpi=160)
    plt.close(fig)


def subgroup_figure(table):
    pivot = table.pivot(index='group', columns='model', values='macro_f1')
    order = ['salary missing', 'salary present', 'no key skills', 'short description', 'long description']
    pivot = pivot.loc[[g for g in order if g in pivot.index]]
    counts = table.drop_duplicates('group').set_index('group')['n']
    ax = pivot.plot.bar(figsize=(9, 4.5), rot=0, color=['#9bb7d4', '#1f4e79'])
    ax.set_xticklabels([f'{g}\n(n={counts[g]})' for g in pivot.index], fontsize=8)
    ax.set_ylabel('Test macro F1')
    ax.set_xlabel('')
    ax.set_ylim(0, 0.6)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURES / 'subgroups.png', dpi=160)
    plt.close()


def main():
    frame, train, test = load_dataset()
    registry = load_registry()
    models = registry.get('models', {})
    if not models:
        raise SystemExit('No trained models. Run src.train_structured and src.train_text first.')
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    test_e = None
    if any(m['needs_embeddings'] for m in models.values()):
        test_e = add_embeddings(frame, embeddings_for(frame)).loc[test.index]

    preds = test[['id', 'name', TARGET, 'salary_missing', 'n_skills', 'desc_words']].copy()
    rows, reports = [], {}
    for name, info in models.items():
        model = joblib.load(MODELS / info['file'])
        preds[name] = model.predict(test_e if info['needs_embeddings'] else test)
        reports[name] = classification_report(test[TARGET], preds[name], labels=LABELS,
                                              output_dict=True, zero_division=0)
        rows.append({'model': name, 'kind': info['kind'], 'cv_macro_f1': info['cv_mean'],
                     'cv_std': info['cv_std'],
                     **{f'test_{k}': v for k, v in metrics(test[TARGET], preds[name]).items()}})
    comparison = pd.DataFrame(rows).sort_values('cv_macro_f1', ascending=False)
    comparison.to_csv(TABLES / 'final_comparison.csv', index=False)
    comparison_figure(comparison)
    preds.to_csv(TABLES / 'test_predictions.csv', index=False)
    (TABLES / 'classification_reports.json').write_text(json.dumps(reports, indent=2))

    structured = [n for n, m in models.items() if m['kind'] == 'structured' and n != 'Dummy']
    with_text = [n for n, m in models.items() if m['kind'] != 'structured']
    best_structured, best_text = best_by_cv(models, structured), best_by_cv(models, with_text)
    summary = {'best_structured_by_cv': best_structured, 'best_with_text_by_cv': best_text}
    if best_structured and best_text:
        summary['paired_test'] = paired_test(preds[TARGET], preds[best_text], preds[best_structured])
        pairs = [best_structured, best_text]
        pd.DataFrame([{'class': LABEL_NAMES[c], **{n: reports[n][c]['f1-score'] for n in pairs}}
                      for c in LABELS]).to_csv(TABLES / 'per_class_f1.csv', index=False)
        subgroups = subgroup_table(test, preds, pairs, train)
        subgroups.to_csv(TABLES / 'subgroups.csv', index=False)
        subgroup_figure(subgroups)
        wrong = preds[preds[best_structured] != preds[TARGET]]
        errors = wrong.assign(title=test.loc[wrong.index, 'name'],
                              masked_text=test.loc[wrong.index, 'text'].str[:400])
        errors[['id', 'title', TARGET, best_structured, best_text, 'salary_missing', 'n_skills',
                'masked_text']].to_csv(TABLES / 'errors_best_structured.csv', index=False)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, name in zip(axes, pairs):
            ConfusionMatrixDisplay.from_predictions(
                preds[TARGET], preds[name], labels=LABELS,
                display_labels=[LABEL_NAMES[c] for c in LABELS], ax=ax, colorbar=False, cmap='Blues')
            ax.set_title(name)
            ax.tick_params(axis='x', labelrotation=20)
        fig.tight_layout()
        fig.savefig(FIGURES / 'confusion_matrices.png', dpi=160)
        plt.close(fig)

    candidates = [n for n in models if n != 'Dummy']
    registry['app_model'] = best_by_cv(models, candidates)
    save_registry(registry)
    (RESULTS / 'summary.json').write_text(json.dumps({**summary, 'app_model': registry['app_model']}, indent=2))
    print(comparison.to_string(index=False))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
