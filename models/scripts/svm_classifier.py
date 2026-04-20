import json
import os

import pandas as pd
from joblib import dump
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC


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
    if X_train.shape[0] > 200_000:
        X_train, _, y_train, _ = train_test_split(
            X_train, y_train, train_size=200_000, stratify=y_train, random_state=42
        )
    return X_train, X_test, y_train, y_test, le


def save_results(model_name, y_test, y_pred, grid_search, results_dir):
    metrics_dir = os.path.join(results_dir, 'metrics', model_name)
    model_dir = os.path.join(results_dir, 'models')
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    report_text = classification_report(y_test, y_pred)
    with open(os.path.join(metrics_dir, f'{model_name}_classification_report.txt'), 'w', encoding='utf-8') as f:
        f.write(report_text)

    cm = confusion_matrix(y_test, y_pred)
    pd.DataFrame(cm).to_csv(os.path.join(metrics_dir, f'{model_name}_confusion_matrix.csv'), index=False)

    pd.DataFrame({'y_true': y_test.values, 'y_pred': y_pred}).to_csv(
        os.path.join(metrics_dir, f'{model_name}_test_predictions.csv'), index=False
    )

    summary = {
        'best_params': grid_search.best_params_,
        'best_cv_accuracy': float(grid_search.best_score_),
        'test_accuracy': float(accuracy_score(y_test, y_pred)),
        'n_test_samples': int(y_test.shape[0])
    }
    with open(os.path.join(metrics_dir, f'{model_name}_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)

    dump(grid_search.best_estimator_, os.path.join(model_dir, f'{model_name}_best_model.joblib'))

    print(f'Resultados salvos em: {metrics_dir}')
    print(f'Modelo salvo em: {os.path.join(model_dir, f"{model_name}_best_model.joblib")}')


def main():
    X_train, X_test, y_train, y_test, _ = load_data()

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svc', SVC(random_state=42))
    ])

    param_grid = {
        'svc__kernel': ['rbf', 'linear'],
        'svc__C': [1, 10],
        'svc__gamma': ['scale', 'auto']
    }

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='accuracy',
        cv=cv,
        n_jobs=-1,
        verbose=2
    )
    grid_search.fit(X_train, y_train)

    y_pred = grid_search.predict(X_test)
    print('SVM melhor score CV:', grid_search.best_score_)
    print(classification_report(y_test, y_pred))

    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    results_dir = os.path.join(root, 'results')
    save_results('svm', y_test, y_pred, grid_search, results_dir)


if __name__ == '__main__':
    main()
