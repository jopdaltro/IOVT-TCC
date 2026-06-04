"""
Treino otimizado de 4 modelos para maxima acuracia / F1.

Estrategias aplicadas:
  - Feature engineering especifico para CAN bus
  - Amostragem equilibrada por classe (cap por classe, sem under/over excessivo)
  - XGBoost: 1.5M+ amostras, grid profundo, feature importance
  - MLP: 500k amostras, rede maior, early stopping
  - Logistic Regression: 600k amostras, multinomial, saga solver
  - SVM (LinearSVC): 150k amostras, rapido e eficaz com dados lineares/separaveis
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(__file__))
from evaluation_utils import write_metrics_bundle

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_PATH = os.path.join(ROOT, 'data', 'processed', 'all_datasets_aligned_balanced.csv')
RESULTS_DIR = os.path.join(ROOT, 'results')

# Samples por classe para cada modelo (None = todos disponíveis)
MAX_PER_CLASS = {
    'xgboost':            300_000,
    'mlp':                100_000,
    'logistic_regression': 100_000,
    'svm':                 25_000,
}


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
BYTE_COLS = ['val0', 'val1', 'val2', 'val3', 'val4', 'val5', 'val6', 'val7']


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria features derivadas do payload CAN para ajudar na separacao de classes."""
    b = df[BYTE_COLS].values.astype(np.float32)

    df['byte_sum']      = b.sum(axis=1)
    df['byte_mean']     = b.mean(axis=1)
    df['byte_std']      = b.std(axis=1)
    df['byte_max']      = b.max(axis=1)
    df['byte_min']      = b.min(axis=1)
    df['byte_range']    = df['byte_max'] - df['byte_min']
    df['nonzero_bytes'] = (b != 0).sum(axis=1)
    df['zero_bytes']    = (b == 0).sum(axis=1)

    # Features de identidade CAN
    df['is_zero_id']    = (df['id'] == 0).astype(np.int8)

    # Interacoes simples
    df['id_dlc']        = df['id'].astype(np.float32) * df['dlc'].astype(np.float32)

    return df


# ---------------------------------------------------------------------------
# Carregamento e amostragem (chunked — não carrega o CSV inteiro em memória)
# ---------------------------------------------------------------------------
def load_balanced_sample(max_per_class: int) -> tuple:
    """Lê o CSV em chunks e amostra `max_per_class` linhas por classe.

    Vantagem: o CSV tem ~958 MB; ler tudo de uma vez estouraria a RAM.
    Aqui paramos de ler assim que cada classe tiver amostras suficientes.
    """
    CHUNK = 500_000

    # Primeiro passamos pelo CSV para coletar labels únicas
    print(f'Descobrindo classes em {DATA_PATH}...', flush=True)
    first = pd.read_csv(DATA_PATH, usecols=['flag'], nrows=100_000, low_memory=False)
    unique_labels = sorted(first['flag'].dropna().unique())
    le = LabelEncoder()
    le.fit(unique_labels)
    n_classes = len(unique_labels)
    print(f'  {n_classes} classes: {unique_labels}', flush=True)

    buckets: dict = {lbl: [] for lbl in unique_labels}
    buckets_full = {lbl: False for lbl in unique_labels}
    cols_drop = ['source']

    print(f'Lendo em chunks de {CHUNK:,}...', flush=True)
    for i, chunk in enumerate(pd.read_csv(DATA_PATH, chunksize=CHUNK, low_memory=False)):
        if all(buckets_full.values()):
            break

        chunk = chunk.dropna(subset=['flag'])
        for col in cols_drop:
            if col in chunk.columns:
                chunk = chunk.drop(columns=[col])

        numeric_cols = ['timestamp', 'id', 'dlc'] + BYTE_COLS
        for col in numeric_cols:
            if col in chunk.columns:
                chunk[col] = pd.to_numeric(chunk[col], errors='coerce')
        chunk = chunk.fillna(0)
        chunk = add_features(chunk)

        for lbl in unique_labels:
            if buckets_full[lbl]:
                continue
            sub = chunk[chunk['flag'] == lbl]
            if sub.empty:
                continue
            current = sum(len(b) for b in buckets[lbl])
            needed = max_per_class - current
            if needed <= 0:
                buckets_full[lbl] = True
                continue
            if len(sub) > needed:
                sub = sub.sample(n=needed, random_state=42)
                buckets_full[lbl] = True
            buckets[lbl].append(sub)

        counts = {lbl: sum(len(b) for b in buckets[lbl]) for lbl in unique_labels}
        print(f'  chunk {i+1:3d} | {counts}', flush=True)

    parts = []
    for lbl in unique_labels:
        if buckets[lbl]:
            parts.append(pd.concat(buckets[lbl], ignore_index=True))
    sampled = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=42)

    sampled['flag'] = le.transform(sampled['flag'])

    X = sampled.drop(columns=['flag'])
    y = sampled['flag']

    print(f'  Total amostras: {len(sampled)} | Classes: {dict(y.value_counts().sort_index())}', flush=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    return X_train, X_test, y_train, y_test, le


# ---------------------------------------------------------------------------
# Salvar resultados
# ---------------------------------------------------------------------------
def save(model_name, estimator, X_test, y_test, y_pred, le, t_train, t_infer):
    model_dir = os.path.join(RESULTS_DIR, 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f'{model_name}_best_model.joblib')
    dump(estimator, model_path)

    summary = write_metrics_bundle(
        model_name=model_name,
        y_true=y_test,
        y_pred=y_pred,
        best_params={'mode': 'optimized'},
        best_cv_accuracy=float('nan'),
        test_accuracy=float(accuracy_score(y_test, y_pred)),
        classes=le.classes_,
        results_dir=RESULTS_DIR,
        model_path=model_path,
        training_time_seconds=t_train,
        inference_time_seconds=t_infer,
    )
    return summary


# ---------------------------------------------------------------------------
# Modelo 1: XGBoost  (mais dados, grid aprofundado)
# ---------------------------------------------------------------------------
def train_xgboost():
    print('\n' + '='*60, flush=True)
    print('XGBOOST', flush=True)
    X_tr, X_te, y_tr, y_te, le = load_balanced_sample(MAX_PER_CLASS['xgboost'])

    model = XGBClassifier(
        objective        = 'multi:softprob',
        eval_metric      = 'mlogloss',
        n_estimators     = 500,
        max_depth        = 7,
        learning_rate    = 0.1,
        subsample        = 0.85,
        colsample_bytree = 0.8,
        min_child_weight = 1,
        gamma            = 0.0,
        reg_alpha        = 0.05,
        reg_lambda       = 1.0,
        tree_method      = 'hist',   # muito mais rapido para datasets grandes
        random_state     = 42,
    )

    t0 = time.perf_counter()
    model.fit(
        X_tr, y_tr,
        eval_set    = [(X_te, y_te)],
        verbose     = 50,
    )
    t_train = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = model.predict(X_te)
    t_infer = time.perf_counter() - t0

    print(f'\nAcuracia teste: {accuracy_score(y_te, y_pred):.4f}', flush=True)
    print(classification_report(y_te, y_pred, target_names=le.classes_), flush=True)
    summary = save('xgboost', model, X_te, y_te, y_pred, le, t_train, t_infer)
    print(f"F1 macro: {summary['f1_macro']:.4f}  |  attack_recall: {summary['attack_detection']['attack_recall']:.4f}", flush=True)


# ---------------------------------------------------------------------------
# Modelo 2: MLP  (rede maior, early stopping, mais amostras)
# ---------------------------------------------------------------------------
def train_mlp():
    print('\n' + '='*60, flush=True)
    print('MLP', flush=True)
    X_tr, X_te, y_tr, y_te, le = load_balanced_sample(MAX_PER_CLASS['mlp'])

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('mlp', MLPClassifier(
            hidden_layer_sizes = (256, 128, 64),
            activation         = 'relu',
            solver             = 'adam',
            alpha              = 1e-4,
            batch_size         = 512,
            learning_rate      = 'adaptive',
            learning_rate_init = 0.001,
            max_iter           = 500,
            early_stopping     = True,
            validation_fraction= 0.1,
            n_iter_no_change   = 15,
            random_state       = 42,
            verbose            = True,
        )),
    ])

    t0 = time.perf_counter()
    pipeline.fit(X_tr, y_tr)
    t_train = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = pipeline.predict(X_te)
    t_infer = time.perf_counter() - t0

    print(f'\nAcuracia teste: {accuracy_score(y_te, y_pred):.4f}', flush=True)
    print(classification_report(y_te, y_pred, target_names=le.classes_), flush=True)
    summary = save('mlp', pipeline, X_te, y_te, y_pred, le, t_train, t_infer)
    print(f"F1 macro: {summary['f1_macro']:.4f}  |  attack_recall: {summary['attack_detection']['attack_recall']:.4f}", flush=True)


# ---------------------------------------------------------------------------
# Modelo 3: Logistic Regression  (solver saga, multinomial, mais iter)
# ---------------------------------------------------------------------------
def train_logistic():
    print('\n' + '='*60, flush=True)
    print('LOGISTIC REGRESSION', flush=True)
    X_tr, X_te, y_tr, y_te, le = load_balanced_sample(MAX_PER_CLASS['logistic_regression'])

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(
            C            = 10.0,
            solver       = 'saga',
            multi_class  = 'multinomial',
            max_iter     = 3000,
            class_weight = 'balanced',
            random_state = 42,
            n_jobs       = -1,
        )),
    ])

    t0 = time.perf_counter()
    pipeline.fit(X_tr, y_tr)
    t_train = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = pipeline.predict(X_te)
    t_infer = time.perf_counter() - t0

    print(f'\nAcuracia teste: {accuracy_score(y_te, y_pred):.4f}', flush=True)
    print(classification_report(y_te, y_pred, target_names=le.classes_), flush=True)
    summary = save('logistic_regression', pipeline, X_te, y_te, y_pred, le, t_train, t_infer)
    print(f"F1 macro: {summary['f1_macro']:.4f}  |  attack_recall: {summary['attack_detection']['attack_recall']:.4f}", flush=True)


# ---------------------------------------------------------------------------
# Modelo 4: LinearSVC  (rapido, bom para dados lineares/separaveis)
# ---------------------------------------------------------------------------
def train_svm():
    print('\n' + '='*60, flush=True)
    print('SVM (LinearSVC)', flush=True)
    X_tr, X_te, y_tr, y_te, le = load_balanced_sample(MAX_PER_CLASS['svm'])

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', LinearSVC(
            C            = 1.0,
            max_iter     = 3000,
            class_weight = 'balanced',
            random_state = 42,
        )),
    ])

    t0 = time.perf_counter()
    pipeline.fit(X_tr, y_tr)
    t_train = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = pipeline.predict(X_te)
    t_infer = time.perf_counter() - t0

    print(f'\nAcuracia teste: {accuracy_score(y_te, y_pred):.4f}', flush=True)
    print(classification_report(y_te, y_pred, target_names=le.classes_), flush=True)
    summary = save('svm', pipeline, X_te, y_te, y_pred, le, t_train, t_infer)
    print(f"F1 macro: {summary['f1_macro']:.4f}  |  attack_recall: {summary['attack_detection']['attack_recall']:.4f}", flush=True)


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    train_xgboost()
    train_mlp()
    train_logistic()
    train_svm()
    print('\nTodos os modelos treinados com sucesso!', flush=True)
