import os
import time
import json
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier

# evaluation_utils from repo
import sys
sys.path.insert(0, os.path.dirname(__file__))
from evaluation_utils import write_metrics_bundle

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_PATH = os.path.join(ROOT, 'data', 'processed', 'bytes_only_aligned_balanced.csv')
RESULTS_DIR = os.path.join(ROOT, 'results')
BYTE_COLS = ['val0','val1','val2','val3','val4','val5','val6','val7']


def _entropy(arr: np.ndarray) -> np.ndarray:
    s = arr.sum(axis=1, keepdims=True).clip(min=1)
    p = arr / s
    p = np.clip(p, 1e-9, 1.0)
    return -(p * np.log2(p)).sum(axis=1)


def add_byte_features(df: pd.DataFrame) -> pd.DataFrame:
    b = df[BYTE_COLS].values.astype(np.float64)
    df['byte_sum'] = b.sum(axis=1)
    df['byte_mean'] = b.mean(axis=1)
    df['byte_std'] = b.std(axis=1)
    df['byte_max'] = b.max(axis=1)
    df['byte_min'] = b.min(axis=1)
    df['byte_range'] = df['byte_max'] - df['byte_min']
    df['nonzero_bytes'] = (b != 0).sum(axis=1)
    df['zero_bytes'] = (b == 0).sum(axis=1)
    df['byte_entropy'] = _entropy(b)
    b_uint8 = np.clip(df[BYTE_COLS].values, 0, 255).astype(np.uint8)
    df['bit_count'] = np.unpackbits(b_uint8, axis=1).sum(axis=1).astype(np.int32)
    even = b[:, [0,2,4,6]].var(axis=1)
    odd  = b[:, [1,3,5,7]].var(axis=1)
    df['byte_var_even'] = even
    df['byte_var_odd'] = odd
    df['byte_var_diff'] = even - odd
    for i in range(7):
        df[f'b{i}_b{i+1}'] = df[BYTE_COLS[i]].astype(float) * df[BYTE_COLS[i+1]].astype(float)
    for col in BYTE_COLS:
        df[f'{col}_norm'] = df[col].astype(float) / 255.0
    return df


def main():
    print('Loading CSV:', DATA_PATH)
    df = pd.read_csv(DATA_PATH, low_memory=False)

    # If columns are val1..val8, rename to val0..val7
    cols = df.columns.tolist()
    if all(f'val{i+1}' in cols for i in range(8)) and not all(f'val{i}' in cols for i in range(8)):
        rename_map = {f'val{i+1}': f'val{i}' for i in range(8)}
        df = df.rename(columns=rename_map)
        print('Renamed val1..val8 -> val0..val7')

    # Keep only bytes and flag
    keep = [c for c in BYTE_COLS if c in df.columns] + ['flag']
    df = df[keep].dropna(subset=['flag']).copy()

    for col in BYTE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    df = add_byte_features(df)

    le = LabelEncoder()
    y = le.fit_transform(df['flag'].astype(str))
    X = df.drop(columns=['flag'])

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    print(f'Train: {len(X_tr)}  Test: {len(X_te)}')

    sw = compute_sample_weight('balanced', y_tr)

    model = XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        n_estimators=800,
        max_depth=10,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method='hist',
        random_state=42,
        n_jobs=-1,
    )

    t0 = time.perf_counter()
    model.fit(X_tr, y_tr, sample_weight=sw, verbose=False)
    t_train = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = model.predict(X_te)
    t_infer = time.perf_counter() - t0

    acc = accuracy_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred, average='macro')
    print(f'Acc: {acc:.4f}  F1 macro: {f1:.4f}  train_time: {t_train:.1f}s')
    print(classification_report(y_te, y_pred, target_names=le.classes_, zero_division=0))

    # save model
    model_dir = os.path.join(RESULTS_DIR, 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'xgboost_article_best_model.joblib')
    dump(model, model_path)

    summary = write_metrics_bundle(
        model_name='xgboost_article',
        y_true=pd.Series(y_te),
        y_pred=y_pred,
        best_params={'mode': 'article_bytes_only'},
        best_cv_accuracy=float('nan'),
        test_accuracy=float(acc),
        classes=le.classes_,
        results_dir=RESULTS_DIR,
        model_path=model_path,
        training_time_seconds=float(t_train),
        inference_time_seconds=float(t_infer),
    )

    print('Saved model and metrics. Summary test_accuracy=', summary['test_accuracy'])


if __name__ == '__main__':
    main()
