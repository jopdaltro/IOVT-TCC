"""
train_bytes_only.py – Teste comparativo: apenas bytes como features (sem id, timestamp, dlc)

Hipótese: dropar o CAN Arbitration ID força o modelo a aprender pelos padrões
de payload, potencialmente melhorando generalização e acurácia.

Features usadas: val0..val7 + derivados dos bytes (entropia, bit_count, estatísticas)
Features removidas: id, timestamp, dlc e todos os derivados de id

Modelos: XGBoost multiclasse + Two-Stage (melhor do train_final.py)
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(__file__))
from evaluation_utils import write_metrics_bundle

ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH  = os.path.join(ROOT, "data", "processed", "bytes_only_aligned_balanced.csv")
RESULTS_DIR = os.path.join(ROOT, "results")

BYTE_COLS = ["val0", "val1", "val2", "val3", "val4", "val5", "val6", "val7"]
# Removemos id, timestamp, dlc — apenas bytes brutos
RAW_COLS  = BYTE_COLS


# ---------------------------------------------------------------------------
# Feature engineering — apenas derivados dos bytes
# ---------------------------------------------------------------------------
def _entropy(arr: np.ndarray) -> np.ndarray:
    s = arr.sum(axis=1, keepdims=True).clip(min=1)
    p = arr / s
    p = np.clip(p, 1e-9, 1.0)
    return -(p * np.log2(p)).sum(axis=1)


def add_byte_features(df: pd.DataFrame) -> pd.DataFrame:
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

    even                 = b[:, [0, 2, 4, 6]].var(axis=1)
    odd                  = b[:, [1, 3, 5, 7]].var(axis=1)
    df["byte_var_even"]  = even
    df["byte_var_odd"]   = odd
    df["byte_var_diff"]  = even - odd

    # Interações entre bytes adjacentes (padrão de sequência)
    for i in range(7):
        df[f"b{i}_b{i+1}"] = (
            df[BYTE_COLS[i]].astype(np.float64) * df[BYTE_COLS[i+1]].astype(np.float64)
        )

    # Bytes individuais normalizados (0-255 → 0-1)
    for col in BYTE_COLS:
        df[f"{col}_norm"] = df[col].astype(np.float64) / 255.0

    return df


# ---------------------------------------------------------------------------
# Carregamento balanceado (idêntico ao train_final.py)
# ---------------------------------------------------------------------------
def load_data(max_per_class_majority: int = 250_000, chunk_size: int = 500_000):
    print(f"\nLendo CSV em chunks ({chunk_size:,} linhas)...", flush=True)

    first = pd.read_csv(DATA_PATH, usecols=["flag"], nrows=200_000, low_memory=False)
    unique_labels = sorted(first["flag"].dropna().astype(str).unique())
    print(f"  Classes: {unique_labels}", flush=True)

    buckets = {lbl: [] for lbl in unique_labels}
    filled  = {lbl: False for lbl in unique_labels}

    for i, chunk in enumerate(pd.read_csv(DATA_PATH, chunksize=chunk_size, low_memory=False)):
        majority_done = all(filled[lbl] for lbl in unique_labels if lbl != "SPEED")
        speed_done    = filled.get("SPEED", False)
        if majority_done and speed_done:
            break

        chunk = chunk.dropna(subset=["flag"]).copy()

        # --- DIFERENÇA PRINCIPAL: manter apenas bytes + flag ---
        drop_cols = [c for c in ["id", "timestamp", "dlc", "source"] if c in chunk.columns]
        chunk.drop(columns=drop_cols, inplace=True)

        for col in BYTE_COLS:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
        chunk.fillna(0, inplace=True)
        chunk = add_byte_features(chunk)

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

    counts_final = dict(zip(*np.unique(y, return_counts=True)))
    print(f"\n  Total: {len(df):,} | features: {X.shape[1]} | distribuição: {counts_final}", flush=True)
    print(f"  Features: {list(X.columns)}", flush=True)
    return X, y, le


# ---------------------------------------------------------------------------
# Salvar resultado
# ---------------------------------------------------------------------------
def save_model(model_name, estimator, X_te, y_te, y_pred, le, t_train, t_infer):
    model_dir = os.path.join(RESULTS_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{model_name}_best_model.joblib")
    dump(estimator, model_path)
    return write_metrics_bundle(
        model_name=model_name,
        y_true=pd.Series(y_te),
        y_pred=y_pred,
        best_params={"mode": "bytes_only"},
        best_cv_accuracy=float("nan"),
        test_accuracy=float(accuracy_score(y_te, y_pred)),
        classes=le.classes_,
        results_dir=RESULTS_DIR,
        model_path=model_path,
        training_time_seconds=t_train,
        inference_time_seconds=t_infer,
    )


# ---------------------------------------------------------------------------
# XGBoost multiclasse — bytes only
# ---------------------------------------------------------------------------
def train_xgboost_bytes(X_tr, X_te, y_tr, y_te, le):
    print("\n" + "=" * 60, flush=True)
    print("XGBoost — BYTES ONLY (sem id/timestamp/dlc)", flush=True)

    sw = compute_sample_weight("balanced", y_tr)

    model = XGBClassifier(
        objective        = "multi:softprob",
        eval_metric      = "mlogloss",
        n_estimators     = 600,
        max_depth        = 8,
        learning_rate    = 0.07,
        subsample        = 0.85,
        colsample_bytree = 0.85,
        min_child_weight = 3,
        gamma            = 0.1,
        reg_alpha        = 0.1,
        reg_lambda       = 1.5,
        tree_method      = "hist",
        random_state     = 42,
    )

    t0 = time.perf_counter()
    model.fit(X_tr, y_tr, sample_weight=sw, verbose=False)
    t_train = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = model.predict(X_te)
    t_infer = time.perf_counter() - t0

    acc = accuracy_score(y_te, y_pred)
    f1  = f1_score(y_te, y_pred, average="macro")
    print(f"  Acuracia: {acc:.4f}  |  F1 macro: {f1:.4f}  |  treino: {t_train:.1f}s", flush=True)
    print(classification_report(y_te, y_pred, target_names=le.classes_, zero_division=0), flush=True)

    return save_model("xgboost_bytes_only", model, X_te, y_te, y_pred, le, t_train, t_infer), model


# ---------------------------------------------------------------------------
# Two-Stage — bytes only
# ---------------------------------------------------------------------------
def train_two_stage_bytes(X_tr, X_te, y_tr, y_te, le):
    print("\n" + "=" * 60, flush=True)
    print("Two-Stage — BYTES ONLY (sem id/timestamp/dlc)", flush=True)

    benign_idx = int(np.where(le.classes_ == "BENIGN")[0][0])

    y_bin_tr = (y_tr != benign_idx).astype(int)
    y_bin_te = (y_te != benign_idx).astype(int)

    stage_a = XGBClassifier(
        objective        = "binary:logistic",
        n_estimators     = 500,
        max_depth        = 8,
        learning_rate    = 0.06,
        subsample        = 0.9,
        colsample_bytree = 0.9,
        tree_method      = "hist",
        random_state     = 42,
    )
    t0 = time.perf_counter()
    stage_a.fit(X_tr, y_bin_tr, verbose=False)
    t_a = time.perf_counter() - t0
    pred_bin = stage_a.predict(X_te)
    print(f"  Stage A: acc={accuracy_score(y_bin_te, pred_bin):.4f}  f1={f1_score(y_bin_te, pred_bin):.4f}  treino={t_a:.1f}s", flush=True)

    attack_classes = [c for c in le.classes_ if c != "BENIGN"]
    le_attack = LabelEncoder()
    le_attack.fit(attack_classes)

    mask_tr  = y_tr != benign_idx
    X_att_tr = X_tr[mask_tr]
    y_att_tr = le_attack.transform(le.inverse_transform(y_tr[mask_tr]))
    sw_b     = compute_sample_weight("balanced", y_att_tr)

    stage_b = XGBClassifier(
        objective        = "multi:softprob",
        eval_metric      = "mlogloss",
        n_estimators     = 500,
        max_depth        = 8,
        learning_rate    = 0.07,
        subsample        = 0.85,
        colsample_bytree = 0.85,
        min_child_weight = 3,
        tree_method      = "hist",
        random_state     = 42,
    )
    t0 = time.perf_counter()
    stage_b.fit(X_att_tr, y_att_tr, sample_weight=sw_b, verbose=False)
    t_b = time.perf_counter() - t0

    pred_full = np.empty(len(y_te), dtype=y_te.dtype)
    mask_att  = pred_bin == 1
    pred_full[~mask_att] = benign_idx
    if mask_att.sum() > 0:
        pred_type = stage_b.predict(X_te[mask_att])
        pred_full[mask_att] = le.transform(le_attack.inverse_transform(pred_type))

    acc = accuracy_score(y_te, pred_full)
    f1  = f1_score(y_te, pred_full, average="macro")
    print(f"  Stage B treino: {t_b:.1f}s", flush=True)
    print(f"  Combined: acc={acc:.4f}  f1_macro={f1:.4f}", flush=True)
    print(classification_report(y_te, pred_full, target_names=le.classes_, zero_division=0), flush=True)

    model_dir = os.path.join(RESULTS_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "two_stage_bytes_only_best_model.joblib")
    dump({"stage_a": stage_a, "stage_b": stage_b, "le": le,
          "le_attack": le_attack, "benign_idx": benign_idx}, model_path)

    return write_metrics_bundle(
        model_name="two_stage_bytes_only",
        y_true=pd.Series(y_te),
        y_pred=pred_full,
        best_params={"mode": "bytes_only_two_stage"},
        best_cv_accuracy=float("nan"),
        test_accuracy=float(acc),
        classes=le.classes_,
        results_dir=RESULTS_DIR,
        model_path=model_path,
        training_time_seconds=float(t_a + t_b),
        inference_time_seconds=float(t_a),
    )


# ---------------------------------------------------------------------------
# Relatório comparativo
# ---------------------------------------------------------------------------
def write_comparison(summaries: dict):
    # Carrega resultados anteriores (com id) para comparar
    prev = {}
    for name in ["xgboost", "two_stage"]:
        path = os.path.join(RESULTS_DIR, "metrics", name, f"{name}_summary.json")
        if os.path.exists(path):
            with open(path) as f:
                prev[name] = json.load(f)

    out_dir = os.path.join(RESULTS_DIR, "tcc_package", "reports")
    os.makedirs(out_dir, exist_ok=True)

    lines = [
        "# Comparativo: Com ID vs Sem ID (bytes only)",
        "",
        "| Modelo | Features | Accuracy | F1 macro | Attack Recall | Attack FPR |",
        "|--------|----------|---:|---:|---:|---:|",
    ]

    if "xgboost" in prev:
        s = prev["xgboost"]
        att = s.get("attack_detection", {})
        lines.append(f"| XGBoost | com id+ts+dlc | {s['test_accuracy']:.4f} | {s['f1_macro']:.4f} | {att.get('attack_recall',0):.4f} | {att.get('attack_fpr',0):.4f} |")

    if "xgboost_bytes_only" in summaries:
        s = summaries["xgboost_bytes_only"]
        att = s.get("attack_detection", {})
        lines.append(f"| XGBoost | **bytes only** | {s['test_accuracy']:.4f} | {s['f1_macro']:.4f} | {att.get('attack_recall',0):.4f} | {att.get('attack_fpr',0):.4f} |")

    if "two_stage" in prev:
        s = prev["two_stage"]
        att = s.get("attack_detection", {})
        lines.append(f"| Two-Stage | com id+ts+dlc | {s['test_accuracy']:.4f} | {s['f1_macro']:.4f} | {att.get('attack_recall',0):.4f} | {att.get('attack_fpr',0):.4f} |")

    if "two_stage_bytes_only" in summaries:
        s = summaries["two_stage_bytes_only"]
        att = s.get("attack_detection", {})
        lines.append(f"| Two-Stage | **bytes only** | {s['test_accuracy']:.4f} | {s['f1_macro']:.4f} | {att.get('attack_recall',0):.4f} | {att.get('attack_fpr',0):.4f} |")

    report_path = os.path.join(out_dir, "bytes_only_comparison.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nComparativo salvo: {report_path}", flush=True)

    json_path = os.path.join(out_dir, "bytes_only_comparison.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, default=str)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    X, y, le = load_data()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    print(f"\nTreino: {len(X_tr):,}  |  Teste: {len(X_te):,}", flush=True)

    summaries = {}
    s_xgb, _  = train_xgboost_bytes(X_tr, X_te, y_tr, y_te, le)
    summaries["xgboost_bytes_only"]     = s_xgb
    summaries["two_stage_bytes_only"]   = train_two_stage_bytes(X_tr, X_te, y_tr, y_te, le)

    write_comparison(summaries)
    print("\nConcluído!", flush=True)
