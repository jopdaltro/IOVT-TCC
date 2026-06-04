import json
import os
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, learning_curve, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


def load_data() -> Tuple[pd.DataFrame, pd.Series]:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    path = os.path.join(root, 'data', 'processed', 'all_datasets_aligned_balanced.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f'Dataset nao encontrado: {path}')

    df = pd.read_csv(path, low_memory=False)
    df = df.dropna()
    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    for col in ['val3', 'val4', 'val6', 'val7']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.fillna(df.mean(numeric_only=True))

    le = LabelEncoder()
    df['flag'] = le.fit_transform(df['flag'])

    x_data = df.drop(columns=['flag'])
    y_data = df['flag']

    if x_data.shape[0] > 250_000:
        x_data, _, y_data, _ = train_test_split(
            x_data,
            y_data,
            train_size=250_000,
            stratify=y_data,
            random_state=42,
        )

    return x_data, y_data


def build_estimators() -> Dict[str, object]:
    return {
        'mlp': Pipeline([
            ('scaler', StandardScaler()),
            ('model', MLPClassifier(hidden_layer_sizes=(100,), alpha=0.0001, max_iter=300, random_state=42)),
        ]),
        'xgboost': XGBClassifier(
            random_state=42,
            objective='multi:softprob',
            eval_metric='mlogloss',
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=1.0,
            colsample_bytree=1.0,
        ),
        'svm': Pipeline([
            ('scaler', StandardScaler()),
            ('model', SVC(kernel='rbf', C=10, gamma='scale', random_state=42)),
        ]),
        'logistic_regression': Pipeline([
            ('scaler', StandardScaler()),
            ('model', LogisticRegression(C=1.0, solver='lbfgs', max_iter=1200, random_state=42)),
        ]),
    }


def classify_fit_behavior(train_score: float, validation_score: float) -> str:
    gap = train_score - validation_score

    if train_score >= 0.90 and gap >= 0.05:
        return 'indicativo_de_overfitting'
    if train_score < 0.85 and validation_score < 0.85 and gap < 0.03:
        return 'indicativo_de_underfitting'
    return 'ajuste_equilibrado_ou_aceitavel'


def save_plot(
    model_name: str,
    train_sizes: np.ndarray,
    train_mean: np.ndarray,
    train_std: np.ndarray,
    val_mean: np.ndarray,
    val_std: np.ndarray,
    out_dir: str,
):
    plt.figure(figsize=(9, 6))
    plt.plot(train_sizes, train_mean, marker='o', linewidth=2, label='Treino (F1 macro)')
    plt.plot(train_sizes, val_mean, marker='s', linewidth=2, label='Validacao (F1 macro)')

    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.2)
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.2)

    plt.title(f'Learning Curve - {model_name}')
    plt.xlabel('Tamanho do conjunto de treino')
    plt.ylabel('F1 macro')
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(out_dir, f'{model_name}_learning_curve_f1_macro.png')
    plt.savefig(out_path, dpi=160)
    plt.close()


def main():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    out_figures = os.path.join(root, 'results', 'tcc_package', 'figures', 'learning_curves')
    out_tables = os.path.join(root, 'results', 'tcc_package', 'tables')
    out_reports = os.path.join(root, 'results', 'tcc_package', 'reports')

    os.makedirs(out_figures, exist_ok=True)
    os.makedirs(out_tables, exist_ok=True)
    os.makedirs(out_reports, exist_ok=True)

    x_data, y_data = load_data()
    estimators = build_estimators()

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    train_sizes = np.linspace(0.1, 1.0, 5)

    summary = {}
    report_lines = [
        '# Analise de Overfitting com Learning Curves',
        '',
        'Metrica usada: F1 macro (sem AUC/ROC).',
        '',
    ]

    for model_name, estimator in estimators.items():
        sizes, train_scores, val_scores = learning_curve(
            estimator=estimator,
            X=x_data,
            y=y_data,
            train_sizes=train_sizes,
            cv=cv,
            scoring='f1_macro',
            n_jobs=-1,
            shuffle=True,
            random_state=42,
        )

        train_mean = train_scores.mean(axis=1)
        train_std = train_scores.std(axis=1)
        val_mean = val_scores.mean(axis=1)
        val_std = val_scores.std(axis=1)

        behavior = classify_fit_behavior(float(train_mean[-1]), float(val_mean[-1]))

        df_curve = pd.DataFrame({
            'train_size': sizes,
            'train_f1_macro_mean': train_mean,
            'train_f1_macro_std': train_std,
            'validation_f1_macro_mean': val_mean,
            'validation_f1_macro_std': val_std,
        })
        df_curve.to_csv(os.path.join(out_tables, f'{model_name}_learning_curve_f1_macro.csv'), index=False)

        save_plot(model_name, sizes, train_mean, train_std, val_mean, val_std, out_figures)

        summary[model_name] = {
            'final_train_f1_macro': float(train_mean[-1]),
            'final_validation_f1_macro': float(val_mean[-1]),
            'final_gap_train_minus_validation': float(train_mean[-1] - val_mean[-1]),
            'fit_behavior': behavior,
        }

        report_lines.extend([
            f'## {model_name}',
            '',
            f"- F1 macro final treino: {train_mean[-1]:.6f}",
            f"- F1 macro final validacao: {val_mean[-1]:.6f}",
            f"- Gap treino-validacao: {train_mean[-1] - val_mean[-1]:.6f}",
            f'- Interpretacao: {behavior}',
            '',
        ])

    with open(os.path.join(out_tables, 'learning_curve_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(out_reports, 'learning_curve_analysis.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    print('Learning curves salvas em: results/tcc_package/figures/learning_curves')
    print('Resumo salvo em: results/tcc_package/tables/learning_curve_summary.json')


if __name__ == '__main__':
    main()
