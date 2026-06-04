import json
import os
import re
from typing import Dict, List

import numpy as np
import pandas as pd


MODELS = ['mlp', 'xgboost', 'svm', 'logistic_regression']


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


def best_model(rows: List[Dict], key: str, higher_is_better: bool = True) -> Dict[str, float]:
    available = [(row['model'], row.get(key)) for row in rows if row.get(key) is not None]
    if not available:
        return {}
    best = max(available, key=lambda x: x[1]) if higher_is_better else min(available, key=lambda x: x[1])
    return {'model': best[0], 'value': float(best[1])}


def flatten_row(summary: Dict) -> Dict:
    attack = summary.get('attack_detection', {}) or {}
    efficiency = summary.get('efficiency', {}) or {}
    return {
        'model': summary['model'],
        'test_accuracy': summary.get('test_accuracy'),
        'f1_macro': summary.get('f1_macro'),
        'precision_macro': summary.get('precision_macro'),
        'recall_macro': summary.get('recall_macro'),
        'attack_recall': attack.get('attack_recall'),
        'attack_precision': attack.get('attack_precision'),
        'attack_f1': attack.get('attack_f1'),
        'attack_fpr': attack.get('attack_fpr'),
        'attack_fnr': attack.get('attack_fnr'),
        'training_time_seconds': efficiency.get('training_time_seconds'),
        'inference_time_ms_per_sample': efficiency.get('inference_time_ms_per_sample'),
        'inference_throughput_samples_per_second': efficiency.get('inference_throughput_samples_per_second'),
        'model_size_mb': efficiency.get('model_size_mb'),
    }


def write_markdown_report(root: str, rows: List[Dict], highlights: Dict[str, Dict[str, float]]):
    out_dir = os.path.join(root, 'results', 'tcc_package', 'reports')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'model_comparison_report.md')

    def _fmt(value):
        if value is None:
            return 'n/a'
        return f'{float(value):.6f}'

    lines = [
        '# Comparativo de Modelos IDS (Sem Score Composto)',
        '',
        'Este relatorio apresenta comparacao direta por metrica, sem score composto e sem AUC/ROC.',
        '',
        '| Modelo | Accuracy teste | Recall ataque | Precision ataque | FPR ataque | F1 macro | Tempo treino (s) | Inferencia (ms/amostra) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]

    for row in rows:
        lines.append(
            '| {model} | {accuracy} | {recall} | {precision} | {fpr} | {f1} | {train_time} | {infer_ms} |'.format(
                model=row['model'],
                accuracy=_fmt(row.get('test_accuracy')),
                recall=_fmt(row.get('attack_recall')),
                precision=_fmt(row.get('attack_precision')),
                fpr=_fmt(row.get('attack_fpr')),
                f1=_fmt(row.get('f1_macro')),
                train_time=_fmt(row.get('training_time_seconds')),
                infer_ms=_fmt(row.get('inference_time_ms_per_sample')),
            )
        )

    lines.extend([
        '',
        '## Destaques por metrica',
        '',
        f"- Maior recall de ataque: {highlights.get('best_attack_recall', {}).get('model', 'n/a')} ({_fmt(highlights.get('best_attack_recall', {}).get('value'))})",
        f"- Menor FPR de ataque: {highlights.get('lowest_attack_fpr', {}).get('model', 'n/a')} ({_fmt(highlights.get('lowest_attack_fpr', {}).get('value'))})",
        f"- Maior F1 macro: {highlights.get('best_f1_macro', {}).get('model', 'n/a')} ({_fmt(highlights.get('best_f1_macro', {}).get('value'))})",
        f"- Menor tempo de inferencia: {highlights.get('lowest_inference_time_ms', {}).get('model', 'n/a')} ({_fmt(highlights.get('lowest_inference_time_ms', {}).get('value'))} ms/amostra)",
        f"- Menor tempo de treino: {highlights.get('lowest_training_time_s', {}).get('model', 'n/a')} ({_fmt(highlights.get('lowest_training_time_s', {}).get('value'))} s)",
    ])

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
        out_reports = os.path.join(root, 'results', 'tcc_package', 'reports')
        os.makedirs(out_reports, exist_ok=True)
        placeholder = os.path.join(out_reports, 'model_comparison_report.md')
        with open(placeholder, 'w', encoding='utf-8') as f:
            f.write(
                '# Comparativo de Modelos IDS (Sem Score Composto)\n\n'
                'Nenhum summary foi encontrado em results/metrics/*/*_summary.json.\n\n'
                'Execute primeiro os scripts de treino dos quatro modelos e rode novamente este script.\n'
            )
        print('Nenhum summary encontrado. Relatorio placeholder gerado em results/tcc_package/reports/model_comparison_report.md')
        raise SystemExit(0)

    rows = [flatten_row(row) for row in rows]
    rows.sort(key=lambda r: r['model'])

    highlights = {
        'best_attack_recall': best_model(rows, 'attack_recall', higher_is_better=True),
        'lowest_attack_fpr': best_model(rows, 'attack_fpr', higher_is_better=False),
        'best_f1_macro': best_model(rows, 'f1_macro', higher_is_better=True),
        'lowest_inference_time_ms': best_model(rows, 'inference_time_ms_per_sample', higher_is_better=False),
        'lowest_training_time_s': best_model(rows, 'training_time_seconds', higher_is_better=False),
    }

    output = {
        'notes': [
            'Comparacao sem score composto para manter interpretacao direta por metrica.',
            'Avaliacao sem AUC/ROC; foco em recall, precision, F1, FPR e eficiencia.',
            'FPR definido em visao binaria de ataque (ataque vs benign).',
        ],
        'comparison': rows,
        'highlights': highlights,
    }

    out_dir = os.path.join(root, 'results', 'tcc_package', 'tables')
    os.makedirs(out_dir, exist_ok=True)
    out_json = os.path.join(out_dir, 'model_comparison_no_score.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

    write_markdown_report(root, rows, highlights)

    print(f'Comparativo salvo em: {out_json}')
    print('Relatorio markdown salvo em: results/tcc_package/reports/model_comparison_report.md')
