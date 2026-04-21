import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def detect_benign_index(classes: np.ndarray) -> Optional[int]:
    for idx, label in enumerate(classes):
        if 'benign' in str(label).strip().lower():
            return idx
    return None


def per_class_fpr(conf_matrix: np.ndarray, class_labels: List[str]) -> Dict[str, float]:
    total = conf_matrix.sum()
    fpr_map: Dict[str, float] = {}

    for idx, label in enumerate(class_labels):
        tp = conf_matrix[idx, idx]
        fp = conf_matrix[:, idx].sum() - tp
        fn = conf_matrix[idx, :].sum() - tp
        tn = total - tp - fp - fn
        denom = fp + tn
        fpr_map[label] = float(fp / denom) if denom > 0 else 0.0

    return fpr_map


def binary_attack_view(
    y_true: pd.Series,
    y_pred: np.ndarray,
    benign_idx: Optional[int],
) -> Dict[str, Optional[float]]:
    if benign_idx is None:
        return {
            'attack_precision': None,
            'attack_recall': None,
            'attack_f1': None,
            'attack_fpr': None,
            'attack_fnr': None,
        }

    y_true_attack = (y_true != benign_idx).astype(int)
    y_pred_attack = (y_pred != benign_idx).astype(int)

    tp = int(((y_true_attack == 1) & (y_pred_attack == 1)).sum())
    tn = int(((y_true_attack == 0) & (y_pred_attack == 0)).sum())
    fp = int(((y_true_attack == 0) & (y_pred_attack == 1)).sum())
    fn = int(((y_true_attack == 1) & (y_pred_attack == 0)).sum())

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


def efficiency_summary(
    training_time_seconds: float,
    inference_time_seconds: float,
    n_test_samples: int,
    model_path: str,
) -> Dict[str, float]:
    per_sample_ms = (inference_time_seconds / n_test_samples) * 1000.0 if n_test_samples > 0 else 0.0
    throughput = n_test_samples / inference_time_seconds if inference_time_seconds > 0 else 0.0
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024) if os.path.exists(model_path) else 0.0

    return {
        'training_time_seconds': float(training_time_seconds),
        'inference_time_seconds': float(inference_time_seconds),
        'inference_time_ms_per_sample': float(per_sample_ms),
        'inference_throughput_samples_per_second': float(throughput),
        'model_size_mb': float(model_size_mb),
    }


def write_metrics_bundle(
    model_name: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    best_params: Dict,
    best_cv_accuracy: float,
    test_accuracy: float,
    classes: np.ndarray,
    results_dir: str,
    model_path: str,
    training_time_seconds: float,
    inference_time_seconds: float,
) -> Dict:
    metrics_dir = os.path.join(results_dir, 'metrics', model_name)
    os.makedirs(metrics_dir, exist_ok=True)

    report_text = classification_report(y_true, y_pred, zero_division=0)
    report_dict = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    with open(os.path.join(metrics_dir, f'{model_name}_classification_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report_text)

    with open(os.path.join(metrics_dir, f'{model_name}_classification_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report_dict, f, indent=2)

    conf_matrix = confusion_matrix(y_true, y_pred)
    pd.DataFrame(conf_matrix).to_csv(
        os.path.join(metrics_dir, f'{model_name}_confusion_matrix.csv'),
        index=False,
    )

    pd.DataFrame({'y_true': y_true.values, 'y_pred': y_pred}).to_csv(
        os.path.join(metrics_dir, f'{model_name}_test_predictions.csv'),
        index=False,
    )

    class_labels = [str(c) for c in classes]
    multiclass_fpr = per_class_fpr(conf_matrix, class_labels)
    macro_fpr = float(np.mean(list(multiclass_fpr.values()))) if multiclass_fpr else 0.0

    benign_idx = detect_benign_index(classes)
    attack_metrics = binary_attack_view(y_true, y_pred, benign_idx)

    macro_metrics = report_dict.get('macro avg', {})
    weighted_metrics = report_dict.get('weighted avg', {})

    summary = {
        'best_params': best_params,
        'best_cv_accuracy': float(best_cv_accuracy),
        'test_accuracy': float(test_accuracy),
        'n_test_samples': int(y_true.shape[0]),
        'classes': class_labels,
        'precision_macro': float(macro_metrics.get('precision', 0.0)),
        'recall_macro': float(macro_metrics.get('recall', 0.0)),
        'f1_macro': float(macro_metrics.get('f1-score', 0.0)),
        'precision_weighted': float(weighted_metrics.get('precision', 0.0)),
        'recall_weighted': float(weighted_metrics.get('recall', 0.0)),
        'f1_weighted': float(weighted_metrics.get('f1-score', 0.0)),
        'fpr_macro_ovr': macro_fpr,
        'fpr_per_class_ovr': multiclass_fpr,
        'attack_detection': attack_metrics,
    }

    summary['efficiency'] = efficiency_summary(
        training_time_seconds=training_time_seconds,
        inference_time_seconds=inference_time_seconds,
        n_test_samples=int(y_true.shape[0]),
        model_path=model_path,
    )

    with open(os.path.join(metrics_dir, f'{model_name}_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    return summary
