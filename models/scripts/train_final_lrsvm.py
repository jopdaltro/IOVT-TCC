"""
train_final_lrsvm.py – Completa o treino de LR e SVM (continuação de train_final.py)
Re-usa os dados já carregados de forma idêntica e salva métricas.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC

sys.path.insert(0, os.path.dirname(__file__))
from evaluation_utils import write_metrics_bundle

ROOT      = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH = os.path.join(ROOT, "data", "processed", "all_datasets_aligned_balanced.csv")
RESULTS_DIR = os.path.join(ROOT, "results")

BYTE_COLS    = ["val0","val1","val2","val3","val4","val5","val6","val7"]
NUMERIC_COLS = ["timestamp","id","dlc"] + BYTE_COLS


def _entropy(arr):
    s = arr.sum(axis=1, keepdims=True).clip(min=1)
    p = arr / s
    p = np.clip(p, 1e-9, 1.0)
    return -(p * np.log2(p)).sum(axis=1)


def add_features(df):
    b = df[BYTE_COLS].values.astype(np.float64)
    df["byte_sum"]       = b.sum(axis=1)
    df["byte_mean"]      = b.mean(axis=1)
    df["byte_std"]       = b.std(axis=1)
    df["byte_max"]       = b.max(axis=1)
    df["byte_min"]       = b.min(axis=1)
    df["byte_range"]     = df["byte_max"] - df["byte_min"]
    df["nonzero_bytes"]  = (b != 0).sum(axis=1)
    df["zero_bytes"]     = (b == 0).sum(axis=1)
    df["byte_entropy"]   = _entropy(b)
    b_uint8              = np.clip(df[BYTE_COLS].values, 0, 255).astype(np.uint8)
    df["bit_count"]      = np.unpackbits(b_uint8, axis=1).sum(axis=1).astype(np.int32)
    df["id_mod_16"]      = (df["id"] % 16).astype(np.int16)
    df["id_mod_256"]     = (df["id"] % 256).astype(np.int16)
    df["id_log1p"]       = np.log1p(df["id"].astype(np.float64))
    df["id_dlc"]         = df["id"].astype(np.float64) * df["dlc"].astype(np.float64)
    df["val0_x_id"]      = df["val0"].astype(np.float64) * df["id"].astype(np.float64)
    df["val1_x_id"]      = df["val1"].astype(np.float64) * df["id"].astype(np.float64)
    even                 = b[:, [0,2,4,6]].var(axis=1)
    odd                  = b[:, [1,3,5,7]].var(axis=1)
    df["byte_var_even"]  = even
    df["byte_var_odd"]   = odd
    df["byte_var_diff"]  = even - odd
    return df


def load_data(max_per_class_majority=250_000, chunk_size=500_000):
    print(f"\nLendo CSV em chunks...", flush=True)
    first = pd.read_csv(DATA_PATH, usecols=["flag"], nrows=200_000, low_memory=False)
    unique_labels = sorted(first["flag"].dropna().astype(str).unique())

    buckets = {lbl: [] for lbl in unique_labels}
    filled  = {lbl: False for lbl in unique_labels}

    for i, chunk in enumerate(pd.read_csv(DATA_PATH, chunksize=chunk_size, low_memory=False)):
        majority_done = all(filled[lbl] for lbl in unique_labels if lbl != "SPEED")
        speed_done    = filled.get("SPEED", False)
        if majority_done and speed_done:
            break

        chunk = chunk.dropna(subset=["flag"]).copy()
        if "source" in chunk.columns:
            chunk.drop(columns=["source"], inplace=True)
        for col in NUMERIC_COLS:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
        chunk.fillna(0, inplace=True)
        chunk = add_features(chunk)

        for lbl in unique_labels:
            if filled[lbl]:
                continue
            sub = chunk[chunk["flag"].astype(str) == lbl]
            if sub.empty:
                continue
            current = sum(len(p) for p in buckets[lbl])
            if lbl == "SPEED":
                buckets[lbl].append(sub)
            else:
                need = max_per_class_majority - current
                if need <= 0:
                    filled[lbl] = True
                    continue
                if len(sub) >= need:
                    sub = sub.sample(n=need, random_state=42)
                    filled[lbl] = True
                buckets[lbl].append(sub)

        counts = {lbl: sum(len(p) for p in buckets[lbl]) for lbl in unique_labels}
        print(f"  chunk {i+1:3d} | {counts}", flush=True)

    frames = [pd.concat(buckets[lbl], ignore_index=True) for lbl in unique_labels if buckets[lbl]]
    df = pd.concat(frames, ignore_index=True).sample(frac=1.0, random_state=42)
    df.reset_index(drop=True, inplace=True)

    le = LabelEncoder()
    y  = le.fit_transform(df["flag"].astype(str))
    X  = df.drop(columns=["flag"])
    print(f"\n  Total: {len(df):,} | classes: {dict(zip(*np.unique(y, return_counts=True)))}", flush=True)
    return X, y, le


def save_model(model_name, estimator, X_te, y_te, y_pred, le, t_train, t_infer):
    model_dir = os.path.join(RESULTS_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{model_name}_best_model.joblib")
    dump(estimator, model_path)
    return write_metrics_bundle(
        model_name=model_name,
        y_true=pd.Series(y_te),
        y_pred=y_pred,
        best_params={"mode": "final_v2"},
        best_cv_accuracy=float("nan"),
        test_accuracy=float(accuracy_score(y_te, y_pred)),
        classes=le.classes_,
        results_dir=RESULTS_DIR,
        model_path=model_path,
        training_time_seconds=t_train,
        inference_time_seconds=t_infer,
    )


def train_logistic(X_tr, X_te, y_tr, y_te, le):
    print("\n" + "="*60, flush=True)
    print("Logistic Regression (50k amostras)", flush=True)
    MAX_LR = 50_000
    if len(X_tr) > MAX_LR:
        idx = np.random.RandomState(42).choice(len(X_tr), MAX_LR, replace=False)
        X_sub, y_sub = X_tr.iloc[idx], y_tr[idx]
    else:
        X_sub, y_sub = X_tr, y_tr

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C=5.0, solver="saga", max_iter=2000,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )),
    ])
    t0 = time.perf_counter()
    pipeline.fit(X_sub, y_sub)
    t_train = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = pipeline.predict(X_te)
    t_infer = time.perf_counter() - t0

    print(f"  acc={accuracy_score(y_te,y_pred):.4f}  F1={f1_score(y_te,y_pred,average='macro'):.4f}", flush=True)
    print(classification_report(y_te, y_pred, target_names=le.classes_, zero_division=0), flush=True)
    return save_model("logistic_regression", pipeline, X_te, y_te, y_pred, le, t_train, t_infer)


def train_svm(X_tr, X_te, y_tr, y_te, le):
    print("\n" + "="*60, flush=True)
    print("SVM LinearSVC (30k amostras)", flush=True)
    MAX_SVM = 30_000
    if len(X_tr) > MAX_SVM:
        idx = np.random.RandomState(42).choice(len(X_tr), MAX_SVM, replace=False)
        X_sub, y_sub = X_tr.iloc[idx], y_tr[idx]
    else:
        X_sub, y_sub = X_tr, y_tr

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", LinearSVC(
            C=1.0, max_iter=5000, class_weight="balanced", random_state=42,
        )),
    ])
    t0 = time.perf_counter()
    pipeline.fit(X_sub, y_sub)
    t_train = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = pipeline.predict(X_te)
    t_infer = time.perf_counter() - t0

    print(f"  acc={accuracy_score(y_te,y_pred):.4f}  F1={f1_score(y_te,y_pred,average='macro'):.4f}", flush=True)
    print(classification_report(y_te, y_pred, target_names=le.classes_, zero_division=0), flush=True)
    return save_model("svm", pipeline, X_te, y_te, y_pred, le, t_train, t_infer)


def write_final_report(summaries):
    out_dir = os.path.join(RESULTS_DIR, "tcc_package", "reports")
    os.makedirs(out_dir, exist_ok=True)

    # Lê os summaries já existentes (xgboost, two_stage, mlp)
    for name in ["xgboost", "two_stage", "mlp"]:
        path = os.path.join(RESULTS_DIR, "metrics", name, f"{name}_summary.json")
        if os.path.exists(path) and name not in summaries:
            with open(path) as f:
                summaries[name] = json.load(f)

    lines = [
        "# Relatório Final – IDS IoVT (train_final.py v2)",
        "",
        "| Modelo | Accuracy | F1 macro | Attack Recall | Attack FPR | Treino (s) | Inf. (ms/smp) |",
        "|--------|---:|---:|---:|---:|---:|---:|",
    ]
    for name, s in summaries.items():
        if not s:
            continue
        eff = s.get("efficiency", {})
        att = s.get("attack_detection", {})
        lines.append(
            f"| {name} "
            f"| {s.get('test_accuracy',0):.4f} "
            f"| {s.get('f1_macro',0):.4f} "
            f"| {att.get('attack_recall',0):.4f} "
            f"| {att.get('attack_fpr',0):.4f} "
            f"| {eff.get('training_time_seconds',0):.1f} "
            f"| {eff.get('inference_time_ms_per_sample',0):.4f} |"
        )

    report_path = os.path.join(out_dir, "final_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    json_path = os.path.join(out_dir, "final_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({k: v for k, v in summaries.items() if v}, f, indent=2, default=str)

    print(f"\nRelatório salvo: {report_path}", flush=True)


if __name__ == "__main__":
    X, y, le = load_data()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    print(f"Treino: {len(X_tr):,}  | Teste: {len(X_te):,}", flush=True)

    summaries = {}
    summaries["logistic_regression"] = train_logistic(X_tr, X_te, y_tr, y_te, le)
    summaries["svm"]                 = train_svm(X_tr, X_te, y_tr, y_te, le)

    write_final_report(summaries)
    print("\nConcluído!", flush=True)
