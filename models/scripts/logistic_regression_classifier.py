import os
import time

import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from evaluation_utils import write_metrics_bundle


def load_data():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    path = os.path.join(root, 'data', 'processed', 'all_datasets_aligned_balanced.csv')
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

    X = df.drop(columns=['flag'])
    y = df['flag']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    return X_train, X_test, y_train, y_test, le


def save_results(model_name, y_test, y_pred, grid_search, results_dir, label_encoder, training_time_seconds, inference_time_seconds):
    model_dir = os.path.join(results_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, f'{model_name}_best_model.joblib')
    dump(grid_search.best_estimator_, model_path)

    summary = write_metrics_bundle(
        model_name=model_name,
        y_true=y_test,
        y_pred=y_pred,
        best_params=grid_search.best_params_,
        best_cv_accuracy=float(grid_search.best_score_),
        test_accuracy=float(accuracy_score(y_test, y_pred)),
        classes=label_encoder.classes_,
        results_dir=results_dir,
        model_path=model_path,
        training_time_seconds=training_time_seconds,
        inference_time_seconds=inference_time_seconds,
    )

    print(f"Resumo salvo em: {os.path.join(results_dir, 'metrics', model_name, f'{model_name}_summary.json')}")
    print(f'Modelo salvo em: {model_path}')
    print(f"Recall macro: {summary['recall_macro']:.6f} | FPR macro OvR: {summary['fpr_macro_ovr']:.6f}")


def main():
    X_train, X_test, y_train, y_test, label_encoder = load_data()

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(random_state=42, max_iter=1200, multi_class='auto')),
    ])

    param_grid = {
        'lr__C': [0.1, 1.0, 10.0],
        'lr__solver': ['lbfgs', 'newton-cg'],
        'lr__class_weight': [None, 'balanced'],
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='accuracy',
        cv=cv,
        n_jobs=-1,
        verbose=2,
    )

    fit_start = time.perf_counter()
    grid_search.fit(X_train, y_train)
    training_time_seconds = time.perf_counter() - fit_start

    pred_start = time.perf_counter()
    y_pred = grid_search.predict(X_test)
    inference_time_seconds = time.perf_counter() - pred_start

    print('Logistic Regression melhor score CV:', grid_search.best_score_)
    print(classification_report(y_test, y_pred))

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    results_dir = os.path.join(root, 'results')

    save_results(
        'logistic_regression',
        y_test,
        y_pred,
        grid_search,
        results_dir,
        label_encoder,
        training_time_seconds,
        inference_time_seconds,
    )


if __name__ == '__main__':
    main()
