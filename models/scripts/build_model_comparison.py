import json
import os
import re
from typing import Dict, List

import numpy as np
import pandas as pd


MODELS = ['mlp', 'xgboost', 'svm', 'logistic_regression']

DEFAULT_TRAIN_SPEED_PRIOR = {
    'logistic_regression': 1.0,
    'mlp': 0.7,
    'xgboost': 0.5,
    'svm': 0.3,
}

DEFAULT_INFER_SPEED_PRIOR = {
    'logistic_regression': 1.0,
    'xgboost': 0.8,
    'mlp': 0.6,
    'svm': 0.2,
}


def load_summary(root: str, model: str) -> Dict:
    path = os.path.join(root, 'results', 'metrics', model, f'{model}_summary.json')
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_confusion_matrix(root: str, model: str):
    path = os.path.join(root, 'results', 'metrics', model, f'{model}_confusion_matrix.csv')
    if not os.path.exists(path):
        return None
    return pd.read_csv(path).values


def load_report_dict(root: str, model: str) -> Dict:
    path = os.path.join(root, 'results', 'metrics', model, f'{model}_classification_report.json')
    if not os.path.exists(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_report_macro_from_txt(root: str, model: str) -> Dict[str, float]:
    path = os.path.join(root, 'results', 'metrics', model, f'{model}_classification_report.txt')
    if not os.path.exists(path):
        return {}

    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    match = re.search(r'^\s*macro avg\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)', text, re.MULTILINE)
    if not match:
        return {}

    return {
        'precision_macro': float(match.group(1)),
        'recall_macro': float(match.group(2)),
        'f1_macro': float(match.group(3)),
    }


def fpr_per_class(conf_matrix: np.ndarray) -> Dict[str, float]:
    total = conf_matrix.sum()
    fprs: Dict[str, float] = {}
    for idx in range(conf_matrix.shape[0]):
        tp = conf_matrix[idx, idx]
        fp = conf_matrix[:, idx].sum() - tp
        fn = conf_matrix[idx, :].sum() - tp
        tn = total - tp - fp - fn
        denom = fp + tn
        fprs[str(idx)] = float(fp / denom) if denom > 0 else 0.0
    return fprs


def attack_metrics_from_confusion(conf_matrix: np.ndarray, benign_idx: int = 0) -> Dict[str, float]:
    if conf_matrix.shape[0] == 0:
        return {
            'attack_precision': 0.0,
            'attack_recall': 0.0,
            'attack_f1': 0.0,
            'attack_fpr': 1.0,
            'attack_fnr': 1.0,
        }

    idx = list(range(conf_matrix.shape[0]))
    attack_idx = [i for i in idx if i != benign_idx]

    tp = int(conf_matrix[np.ix_(attack_idx, attack_idx)].sum())
    fp = int(conf_matrix[benign_idx, attack_idx].sum())
    fn = int(conf_matrix[np.ix_(attack_idx, [benign_idx])].sum())
    tn = int(conf_matrix[benign_idx, benign_idx])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        'attack_precision': float(precision),
        'attack_recall': float(recall),
        'attack_f1': float(f1),
        'attack_fpr': float(fpr),
        'attack_fnr': float(fnr),
    }


def enrich_summary(root: str, model: str, summary: Dict) -> Dict:
    report = load_report_dict(root, model)
    report_macro_txt = load_report_macro_from_txt(root, model)
    conf_matrix = load_confusion_matrix(root, model)

    if 'precision_macro' not in summary and report:
        summary['precision_macro'] = float(report.get('macro avg', {}).get('precision', 0.0))
    if 'recall_macro' not in summary and report:
        summary['recall_macro'] = float(report.get('macro avg', {}).get('recall', 0.0))
    if 'f1_macro' not in summary and report:
        summary['f1_macro'] = float(report.get('macro avg', {}).get('f1-score', 0.0))

    if 'precision_macro' not in summary and 'precision_macro' in report_macro_txt:
        summary['precision_macro'] = report_macro_txt['precision_macro']
    if 'recall_macro' not in summary and 'recall_macro' in report_macro_txt:
        summary['recall_macro'] = report_macro_txt['recall_macro']
    if 'f1_macro' not in summary and 'f1_macro' in report_macro_txt:
        summary['f1_macro'] = report_macro_txt['f1_macro']

    if conf_matrix is not None:
        if 'fpr_per_class_ovr' not in summary:
            summary['fpr_per_class_ovr'] = fpr_per_class(conf_matrix)
        if 'fpr_macro_ovr' not in summary:
            values = list(summary['fpr_per_class_ovr'].values())
            summary['fpr_macro_ovr'] = float(np.mean(values)) if values else 0.0
        if 'attack_detection' not in summary:
            classes = summary.get('classes', [])
            benign_idx = 0
            for idx, cls in enumerate(classes):
                if 'benign' in str(cls).lower():
                    benign_idx = idx
                    break
            summary['attack_detection'] = attack_metrics_from_confusion(conf_matrix, benign_idx)

    if 'efficiency' not in summary:
        summary['efficiency'] = {
            'training_time_seconds': None,
            'inference_time_ms_per_sample': None,
        }

    return summary


def rank_labels(values: Dict[str, float], reverse: bool = False) -> Dict[str, str]:
    available = {k: v for k, v in values.items() if v is not None}
    if not available:
        return {}

    sorted_items = sorted(available.items(), key=lambda kv: kv[1], reverse=reverse)
    labels_pool = ['very_fast', 'fast', 'medium', 'slow'] if not reverse else ['best', 'strong', 'moderate', 'weak']

    labels: Dict[str, str] = {}
    for idx, (name, _) in enumerate(sorted_items):
        labels[name] = labels_pool[min(idx, len(labels_pool) - 1)]
    return labels


def normalize_scores(values: Dict[str, float], higher_is_better: bool) -> Dict[str, float]:
    filtered = {k: v for k, v in values.items() if v is not None}
    if not filtered:
        return {}

    vmin = min(filtered.values())
    vmax = max(filtered.values())
    if vmax == vmin:
        return {k: 1.0 for k in filtered}

    out: Dict[str, float] = {}
    for key, value in filtered.items():
        base = (value - vmin) / (vmax - vmin)
        out[key] = base if higher_is_better else (1.0 - base)
    return out


def composite_score(row: Dict, train_eff: Dict[str, float], infer_eff: Dict[str, float]) -> float:
    model = row['model']
    attack = row.get('attack_detection', {})

    recall_attack = attack.get('attack_recall', row.get('recall_macro', 0.0)) or 0.0
    precision_attack = attack.get('attack_precision', row.get('precision_macro', 0.0)) or 0.0
    attack_fpr = attack.get('attack_fpr', row.get('fpr_macro_ovr', 1.0))
    attack_fpr = 1.0 if attack_fpr is None else attack_fpr
    f1_macro = row.get('f1_macro', 0.0) or 0.0

    score = (
        0.35 * recall_attack
        + 0.30 * (1.0 - attack_fpr)
        + 0.15 * f1_macro
        + 0.10 * precision_attack
        + 0.06 * infer_eff.get(model, 0.0)
        + 0.04 * train_eff.get(model, 0.0)
    )
    return float(score)


def write_markdown_report(root: str, rows: List[Dict]):
    out_path = os.path.join(root, 'results', 'metrics', 'model_comparison_report.md')
    lines = [
        '# Comparativo de Modelos (IDS)',
        '',
        'Criterio de ranking: Recall de ataque e FPR de ataque com maior peso, depois F1, Precision e eficiencia relativa.',
        '',
        '| Rank | Modelo | Recall ataque | Precision ataque | FPR ataque | F1 macro | Velocidade treino | Velocidade inferencia | Score |',
        '|---|---|---:|---:|---:|---:|---|---|---:|',
    ]

    for row in rows:
        attack = row.get('attack_detection', {})
        lines.append(
            '| {rank} | {model} | {recall:.6f} | {precision:.6f} | {fpr:.6f} | {f1:.6f} | {train_label} | {infer_label} | {score:.6f} |'.format(
                rank=row['rank'],
                model=row['model'],
                recall=(attack.get('attack_recall') or 0.0),
                precision=(attack.get('attack_precision') or 0.0),
                fpr=(attack.get('attack_fpr') or 0.0),
                f1=(row.get('f1_macro') or 0.0),
                train_label=row.get('relative_training_speed', 'n/a'),
                infer_label=row.get('relative_inference_speed', 'n/a'),
                score=row.get('ids_composite_score', 0.0),
            )
        )

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))


if __name__ == '__main__':
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    rows = []
    for model in MODELS:
        summary = load_summary(root, model)
        if not summary:
            continue
        summary = enrich_summary(root, model, summary)
        summary['model'] = model
        rows.append(summary)

    if not rows:
        raise SystemExit('Nenhum summary encontrado em results/metrics/*/*_summary.json')

    training_times = {}
    inference_times = {}
    for row in rows:
        model = row['model']
        train_v = row.get('efficiency', {}).get('training_time_seconds')
        infer_v = row.get('efficiency', {}).get('inference_time_ms_per_sample')

        if train_v is None:
            train_v = 1.0 / DEFAULT_TRAIN_SPEED_PRIOR.get(model, 0.5)
        if infer_v is None:
            infer_v = 1.0 / DEFAULT_INFER_SPEED_PRIOR.get(model, 0.5)

        training_times[model] = float(train_v)
        inference_times[model] = float(infer_v)

    train_labels = rank_labels(training_times, reverse=False)
    infer_labels = rank_labels(inference_times, reverse=False)

    train_eff = normalize_scores(training_times, higher_is_better=False)
    infer_eff = normalize_scores(inference_times, higher_is_better=False)

    for row in rows:
        model = row['model']
        row['relative_training_speed'] = train_labels.get(model, 'n/a')
        row['relative_inference_speed'] = infer_labels.get(model, 'n/a')
        row['ids_composite_score'] = composite_score(row, train_eff, infer_eff)

    rows.sort(key=lambda r: r['ids_composite_score'], reverse=True)
    for idx, row in enumerate(rows, start=1):
        row['rank'] = idx

    output = {
        'criteria': {
            'attack_recall_weight': 0.35,
            'attack_fpr_weight': 0.30,
            'f1_macro_weight': 0.15,
            'attack_precision_weight': 0.10,
            'inference_efficiency_weight': 0.06,
            'training_efficiency_weight': 0.04,
        },
        'notes': [
            'FPR definido em visao binaria de ataque (ataque vs benign).',
            'Velocidade classificada de forma relativa entre os modelos disponiveis.',
            'Quando tempos medidos nao existem no summary, usa prior relativa por tipo de algoritmo.',
        ],
        'ranking': rows,
    }

    out_json = os.path.join(root, 'results', 'metrics', 'model_comparison_summary.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

    write_markdown_report(root, rows)

    print(f'Comparativo salvo em: {out_json}')
    print('Relatorio markdown salvo em: results/metrics/model_comparison_report.md')
