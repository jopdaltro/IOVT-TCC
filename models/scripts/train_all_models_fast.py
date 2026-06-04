import os
import time

import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from evaluation_utils import write_metrics_bundle


def load_data(max_rows=30_000):
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    path = os.path.join(root, 'data', 'processed', 'all_datasets_aligned_balanced.csv')
    read_rows = max_rows * 3
    print(f'Reading up to {read_rows} rows from {path}...', flush=True)
    df = pd.read_csv(path, low_memory=False, nrows=read_rows)
    df = df.dropna()

    if 'id' in df.columns:
        df['id'] = pd.to_numeric(df['id'], errors='coerce').fillna(0)

    for col in ['val0', 'val1', 'val2', 'val3', 'val4', 'val5', 'val6', 'val7', 'timestamp', 'dlc']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = df.fillna(df.mean(numeric_only=True))

    if 'source' in df.columns:
        df = df.drop(columns=['source'])

    le = LabelEncoder()
    df['flag'] = le.fit_transform(df['flag'])

    x_data = df.drop(columns=['flag'])
    y_data = df['flag']

    if x_data.shape[0] > max_rows:
        x_data, _, y_data, _ = train_test_split(
            x_data,
            y_data,
            train_size=max_rows,
            stratify=y_data,
            random_state=42,
        )

    x_train, x_test, y_train, y_test = train_test_split(
        x_data,
        y_data,
        test_size=0.2,
        stratify=y_data,
        random_state=42,
    )

    return root, x_train, x_test, y_train, y_test, le


def fit_and_save(model_name, estimator, x_train, x_test, y_train, y_test, label_encoder, root):
    results_dir = os.path.join(root, 'results')
    model_dir = os.path.join(results_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)

    print(f'Training {model_name}...', flush=True)
    start_fit = time.perf_counter()
    estimator.fit(x_train, y_train)
    training_time = time.perf_counter() - start_fit

    start_pred = time.perf_counter()
    y_pred = estimator.predict(x_test)
    inference_time = time.perf_counter() - start_pred

    model_path = os.path.join(model_dir, f'{model_name}_best_model.joblib')
    dump(estimator, model_path)

    summary = write_metrics_bundle(
        model_name=model_name,
        y_true=y_test,
        y_pred=y_pred,
        best_params={'mode': 'fast_train_fixed_params'},
        best_cv_accuracy=float('nan'),
        test_accuracy=float(accuracy_score(y_test, y_pred)),
        classes=label_encoder.classes_,
        results_dir=results_dir,
        model_path=model_path,
        training_time_seconds=training_time,
        inference_time_seconds=inference_time,
    )

    print(f"[{model_name}] test_accuracy={summary['test_accuracy']:.6f} f1_macro={summary['f1_macro']:.6f}")


def main():
    root, x_train, x_test, y_train, y_test, le = load_data(max_rows=30_000)
    print(f'Dataset prepared: train={x_train.shape}, test={x_test.shape}', flush=True)

    models = {
        'logistic_regression': Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(random_state=42, max_iter=1200, C=1.0, solver='lbfgs')),
        ]),
        'mlp': Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPClassifier(random_state=42, max_iter=120, hidden_layer_sizes=(80,), alpha=0.0001, learning_rate='adaptive')),
        ]),
        'xgboost': XGBClassifier(
            random_state=42,
            objective='multi:softprob',
            eval_metric='mlogloss',
            n_estimators=80,
            max_depth=4,
            learning_rate=0.1,
            subsample=1.0,
            colsample_bytree=1.0,
        ),
        'svm': Pipeline([
            ('scaler', StandardScaler()),
            ('svc', SVC(kernel='linear', C=1, gamma='scale', random_state=42)),
        ]),
    }

    for name, estimator in models.items():
        fit_and_save(name, estimator, x_train, x_test, y_train, y_test, le, root)

    print('Treinamento rapido dos 4 modelos concluido.')


if __name__ == '__main__':
    main()
