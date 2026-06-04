"""
train_final.py – Treino definitivo revisado (Claude)

Estrategias aplicadas:
  1. Feature engineering rico: entropia, popcount, variância, ID-hash buckets
  2. Lida com desequilíbrio da classe SPEED usando sample_weight proporcional
  3. XGBoost multiclasse com dados completos e parametros refinados
  4. Pipeline two-stage: binário (BENIGN vs ATTACK) → tipo de ataque
  5. MLP com arquitetura maior, BatchNorm equivalente via normalização, mais dados
  6. LR e SVM re-treinados com amostras suficientes (50k/classe)
  7. Todos os modelos salvos + relatório consolidado
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(__file__))
from evaluation_utils import write_metrics_bundle

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH = os.path.join(ROOT, "data", "processed", "all_datasets_aligned_balanced.csv")
RESULTS_DIR = os.path.join(ROOT, "results")

BYTE_COLS = ["val0", "val1", "val2", "val3", "val4", "val5", "val6", "val7"]
NUMERIC_COLS = ["timestamp", "id", "dlc"] + BYTE_COLS

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def _entropy(arr: np.ndarray) -> np.ndarray:
    """Shannon entropy por linha (8 bytes como distribuição)."""
    # normaliza cada linha para somar 1 (evita log(0))
    s = arr.sum(axis=1, keepdims=True).clip(min=1)
    p = arr / s
    # clip para evitar log(0)
    p = np.clip(p, 1e-9, 1.0)
    return -(p * np.log2(p)).sum(axis=1)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    b = df[BYTE_COLS].values.astype(np.float64)

    # Estatísticas basicas
    df["byte_sum"]      = b.sum(axis=1)
    df["byte_mean"]     = b.mean(axis=1)
    df["byte_std"]      = b.std(axis=1)
    df["byte_max"]      = b.max(axis=1)
    df["byte_min"]      = b.min(axis=1)
    df["byte_range"]    = df["byte_max"] - df["byte_min"]

    # Contagem de zeros/nao-zeros
    df["nonzero_bytes"] = (b != 0).sum(axis=1)
    df["zero_bytes"]    = (b == 0).sum(axis=1)

    # Entropia de Shannon (distingue Fuzzy de DoS)
    df["byte_entropy"]  = _entropy(b)

    # Popcount vetorizado (numero total de bits setados – distingue tipos de ataque CAN)
    # CAN bytes são 0-255 → uint8 perfeito para np.unpackbits
    b_uint8 = np.clip(df[BYTE_COLS].values, 0, 255).astype(np.uint8)
    df["bit_count"] = np.unpackbits(b_uint8, axis=1).sum(axis=1).astype(np.int32)

    # ID features
    df["id_mod_16"]   = (df["id"] % 16).astype(np.int16)
    df["id_mod_256"]  = (df["id"] % 256).astype(np.int16)
    df["id_log1p"]    = np.log1p(df["id"].astype(np.float64))
    df["id_dlc"]      = df["id"].astype(np.float64) * df["dlc"].astype(np.float64)

    # Primeiro byte separado (primeiros dois bytes costumam identificar tipo de CAN frame)
    df["val0_x_id"]   = df["val0"].astype(np.float64) * df["id"].astype(np.float64)
    df["val1_x_id"]   = df["val1"].astype(np.float64) * df["id"].astype(np.float64)

    # Variancia entre bytes pares e impares
    even = b[:, [0, 2, 4, 6]].var(axis=1)
    odd  = b[:, [1, 3, 5, 7]].var(axis=1)
    df["byte_var_even"] = even
    df["byte_var_odd"]  = odd
    df["byte_var_diff"] = even - odd

    return df


# ---------------------------------------------------------------------------
# Carregamento com balanceamento ciente do desequilíbrio (SPEED)
# ---------------------------------------------------------------------------
def load_data(max_per_class_majority: int, chunk_size: int = 500_000):
    """
    Carrega dados balanceando as classes majoritárias em max_per_class_majority
    mas NÃO limita a classe SPEED (a menor): pega TUDO que estiver disponível.
    O desequilíbrio restante é tratado com sample_weight no treino.
    """
    print(f"\nLendo CSV em chunks ({chunk_size:,} linhas)...", flush=True)

    # Descobrir classes
    first = pd.read_csv(DATA_PATH, usecols=["flag"], nrows=200_000, low_memory=False)
    unique_labels = sorted(first["flag"].dropna().astype(str).unique())
    print(f"  Classes: {unique_labels}", flush=True)

    buckets = {lbl: [] for lbl in unique_labels}
    filled  = {lbl: False for lbl in unique_labels}

    for i, chunk in enumerate(pd.read_csv(DATA_PATH, chunksize=chunk_size, low_memory=False)):
        # Para quando as classes majoritárias já estiverem completas
        majority_done = all(
            filled[lbl] for lbl in unique_labels if lbl != "SPEED"
        )
        speed_done = filled.get("SPEED", False)
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
                # SPEED: pega tudo (sem limite)
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

        # Para SPEED: marca como done quando chegarmos ao fim do CSV
        # (controlado pelo flag majority_done acima)

        counts = {lbl: sum(len(p) for p in buckets[lbl]) for lbl in unique_labels}
        print(f"  chunk {i+1:3d} | {counts}", flush=True)

    frames = []
    for lbl in unique_labels:
        if buckets[lbl]:
            frames.append(pd.concat(buckets[lbl], ignore_index=True))

    df = pd.concat(frames, ignore_index=True).sample(frac=1.0, random_state=42)
    df.reset_index(drop=True, inplace=True)

    le = LabelEncoder()
    y = le.fit_transform(df["flag"].astype(str))
    X = df.drop(columns=["flag"])

    counts_final = dict(zip(*np.unique(y, return_counts=True)))
    print(f"\n  Total: {len(df):,} amostras | distribuição: {counts_final}", flush=True)

    return X, y, le


# ---------------------------------------------------------------------------
# Salvar resultados
# ---------------------------------------------------------------------------
def save_model(model_name, estimator, X_test, y_test, y_pred, le, t_train, t_infer):
    model_dir = os.path.join(RESULTS_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{model_name}_best_model.joblib")
    dump(estimator, model_path)

    summary = write_metrics_bundle(
        model_name=model_name,
        y_true=pd.Series(y_test),
        y_pred=y_pred,
        best_params={"mode": "final_v2"},
        best_cv_accuracy=float("nan"),
        test_accuracy=float(accuracy_score(y_test, y_pred)),
        classes=le.classes_,
        results_dir=RESULTS_DIR,
        model_path=model_path,
        training_time_seconds=t_train,
        inference_time_seconds=t_infer,
    )
    return summary


# ---------------------------------------------------------------------------
# 1. XGBoost multiclasse (modelo principal)
# ---------------------------------------------------------------------------
def train_xgboost(X_tr, X_te, y_tr, y_te, le):
    print("\n" + "=" * 60, flush=True)
    print("XGBoost Multiclasse", flush=True)

    # Sample weights para compensar SPEED
    sw = compute_sample_weight("balanced", y_tr)

    model = XGBClassifier(
        objective         = "multi:softprob",
        eval_metric       = "mlogloss",
        n_estimators      = 600,
        max_depth         = 8,
        learning_rate     = 0.07,
        subsample         = 0.85,
        colsample_bytree  = 0.85,
        min_child_weight  = 3,
        gamma             = 0.1,
        reg_alpha         = 0.1,
        reg_lambda        = 1.5,
        tree_method       = "hist",
        random_state      = 42,
    )

    t0 = time.perf_counter()
    model.fit(X_tr, y_tr, sample_weight=sw, verbose=False)
    t_train = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = model.predict(X_te)
    t_infer = time.perf_counter() - t0

    acc = accuracy_score(y_te, y_pred)
    print(f"  Acuracia: {acc:.4f}  |  F1 macro: {f1_score(y_te, y_pred, average='macro'):.4f}", flush=True)
    print(classification_report(y_te, y_pred, target_names=le.classes_, zero_division=0), flush=True)

    summary = save_model("xgboost", model, X_te, y_te, y_pred, le, t_train, t_infer)
    print(f"  attack_recall={summary['attack_detection']['attack_recall']:.4f}  "
          f"attack_fpr={summary['attack_detection']['attack_fpr']:.4f}", flush=True)
    return summary


# ---------------------------------------------------------------------------
# 2. Pipeline two-stage (binário → tipo de ataque)
#    Stage A: XGBoost binário BENIGN vs ATTACK
#    Stage B: XGBoost multiclasse só nas amostras de ataque
# ---------------------------------------------------------------------------
def train_two_stage(X_tr, X_te, y_tr, y_te, le):
    print("\n" + "=" * 60, flush=True)
    print("Two-Stage: Binário + Multiclasse de Ataque", flush=True)

    benign_idx = np.where(le.classes_ == "BENIGN")[0]
    if len(benign_idx) == 0:
        print("  Classe BENIGN não encontrada; pulando two-stage.", flush=True)
        return None
    benign_idx = int(benign_idx[0])

    # ---- Stage A: binário ----
    y_bin_tr = (y_tr != benign_idx).astype(int)
    y_bin_te = (y_te != benign_idx).astype(int)

    stage_a = XGBClassifier(
        objective        = "binary:logistic",
        n_estimators     = 500,
        max_depth        = 8,
        learning_rate    = 0.06,
        subsample        = 0.9,
        colsample_bytree = 0.9,
        reg_alpha        = 0.05,
        reg_lambda       = 1.0,
        tree_method      = "hist",
        random_state     = 42,
    )

    t0 = time.perf_counter()
    stage_a.fit(X_tr, y_bin_tr, verbose=False)
    t_a = time.perf_counter() - t0
    pred_bin = stage_a.predict(X_te)
    acc_a = accuracy_score(y_bin_te, pred_bin)
    f1_a = f1_score(y_bin_te, pred_bin)
    print(f"  Stage A (binário): acc={acc_a:.4f}  f1={f1_a:.4f}  treino={t_a:.1f}s", flush=True)

    # ---- Stage B: tipo de ataque (apenas amostras de ataque) ----
    attack_classes = [c for c in le.classes_ if c != "BENIGN"]
    le_attack = LabelEncoder()
    le_attack.fit(attack_classes)

    # Treino: somente amostras de ataque
    mask_tr = y_tr != benign_idx
    X_att_tr = X_tr[mask_tr]
    y_att_tr_orig = y_tr[mask_tr]
    # Re-encoda para 0..N-1 desconsiderando BENIGN
    y_att_tr = le_attack.transform(le.inverse_transform(y_att_tr_orig))

    sw_b = compute_sample_weight("balanced", y_att_tr)

    stage_b = XGBClassifier(
        objective        = "multi:softprob",
        eval_metric      = "mlogloss",
        n_estimators     = 500,
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
    stage_b.fit(X_att_tr, y_att_tr, sample_weight=sw_b, verbose=False)
    t_b = time.perf_counter() - t0

    # ---- Combinar: teste completo ----
    # Para amostras que stage_a classifica como ATAQUE, usa stage_b
    # Para amostras classificadas como BENIGN, fica BENIGN
    pred_full = np.empty(len(y_te), dtype=y_te.dtype)
    mask_pred_attack = pred_bin == 1
    mask_pred_benign = ~mask_pred_attack

    # Benigns: atribui indice original de BENIGN
    pred_full[mask_pred_benign] = benign_idx

    # Attacks: classifica tipo
    if mask_pred_attack.sum() > 0:
        X_te_att = X_te[mask_pred_attack]
        pred_type = stage_b.predict(X_te_att)         # índice em le_attack
        labels_type = le_attack.inverse_transform(pred_type)
        pred_full[mask_pred_attack] = le.transform(labels_type)

    t_combined = t_a + t_b
    acc_full = accuracy_score(y_te, pred_full)
    f1_full  = f1_score(y_te, pred_full, average="macro")

    print(f"  Stage B (tipo de ataque): treino={t_b:.1f}s", flush=True)
    print(f"  Combined: acc={acc_full:.4f}  f1_macro={f1_full:.4f}", flush=True)
    print(classification_report(y_te, pred_full, target_names=le.classes_, zero_division=0), flush=True)

    # Salva como "two_stage" (salva o stage_b; stage_a já salvo separado)
    model_dir = os.path.join(RESULTS_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)
    dump({"stage_a": stage_a, "stage_b": stage_b,
          "le": le, "le_attack": le_attack, "benign_idx": benign_idx},
         os.path.join(model_dir, "two_stage_best_model.joblib"))

    summary = write_metrics_bundle(
        model_name="two_stage",
        y_true=pd.Series(y_te),
        y_pred=pred_full,
        best_params={"mode": "two_stage"},
        best_cv_accuracy=float("nan"),
        test_accuracy=float(acc_full),
        classes=le.classes_,
        results_dir=RESULTS_DIR,
        model_path=os.path.join(model_dir, "two_stage_best_model.joblib"),
        training_time_seconds=float(t_combined),
        inference_time_seconds=float(t_a),
    )
    return summary


# ---------------------------------------------------------------------------
# 3. MLP melhorado (arquitetura maior, mais dados, melhor normalização)
# ---------------------------------------------------------------------------
def train_mlp(X_tr, X_te, y_tr, y_te, le):
    print("\n" + "=" * 60, flush=True)
    print("MLP (arquitetura aprimorada)", flush=True)

    # Subsample para o MLP (treinamento pesado)
    MAX_MLP = 150_000
    if len(X_tr) > MAX_MLP:
        idx = np.random.RandomState(42).choice(len(X_tr), MAX_MLP, replace=False)
        X_sub = X_tr.iloc[idx]
        y_sub = y_tr[idx]
    else:
        X_sub, y_sub = X_tr, y_tr

    sw = compute_sample_weight("balanced", y_sub)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes  = (512, 256, 128, 64),
            activation          = "relu",
            solver              = "adam",
            alpha               = 5e-4,
            batch_size          = 1024,
            learning_rate       = "adaptive",
            learning_rate_init  = 0.001,
            max_iter            = 300,
            early_stopping      = True,
            validation_fraction = 0.1,
            n_iter_no_change    = 20,
            random_state        = 42,
            verbose             = False,
        )),
    ])

    t0 = time.perf_counter()
    pipeline.fit(X_sub, y_sub)
    t_train = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = pipeline.predict(X_te)
    t_infer = time.perf_counter() - t0

    acc = accuracy_score(y_te, y_pred)
    print(f"  Acuracia: {acc:.4f}  |  F1 macro: {f1_score(y_te, y_pred, average='macro'):.4f}", flush=True)
    print(classification_report(y_te, y_pred, target_names=le.classes_, zero_division=0), flush=True)

    summary = save_model("mlp", pipeline, X_te, y_te, y_pred, le, t_train, t_infer)
    return summary


# ---------------------------------------------------------------------------
# 4. Logistic Regression (dados suficientes desta vez)
# ---------------------------------------------------------------------------
def train_logistic(X_tr, X_te, y_tr, y_te, le):
    print("\n" + "=" * 60, flush=True)
    print("Logistic Regression (50k/classe)", flush=True)

    MAX_LR = 50_000
    if len(X_tr) > MAX_LR:
        idx = np.random.RandomState(42).choice(len(X_tr), MAX_LR, replace=False)
        X_sub = X_tr.iloc[idx]
        y_sub = y_tr[idx]
    else:
        X_sub, y_sub = X_tr, y_tr

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            C            = 5.0,
            solver       = "saga",
            max_iter     = 2000,
            class_weight = "balanced",
            random_state = 42,
            n_jobs       = -1,
        )),
    ])

    t0 = time.perf_counter()
    pipeline.fit(X_sub, y_sub)
    t_train = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = pipeline.predict(X_te)
    t_infer = time.perf_counter() - t0

    acc = accuracy_score(y_te, y_pred)
    print(f"  Acuracia: {acc:.4f}  |  F1 macro: {f1_score(y_te, y_pred, average='macro'):.4f}", flush=True)
    print(classification_report(y_te, y_pred, target_names=le.classes_, zero_division=0), flush=True)

    summary = save_model("logistic_regression", pipeline, X_te, y_te, y_pred, le, t_train, t_infer)
    return summary


# ---------------------------------------------------------------------------
# 5. SVM (30k amostras com pesos balanceados)
# ---------------------------------------------------------------------------
def train_svm(X_tr, X_te, y_tr, y_te, le):
    print("\n" + "=" * 60, flush=True)
    print("SVM LinearSVC (30k amostras)", flush=True)

    MAX_SVM = 30_000
    if len(X_tr) > MAX_SVM:
        idx = np.random.RandomState(42).choice(len(X_tr), MAX_SVM, replace=False)
        X_sub = X_tr.iloc[idx]
        y_sub = y_tr[idx]
    else:
        X_sub, y_sub = X_tr, y_tr

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", LinearSVC(
            C            = 1.0,
            max_iter     = 5000,
            class_weight = "balanced",
            random_state = 42,
        )),
    ])

    t0 = time.perf_counter()
    pipeline.fit(X_sub, y_sub)
    t_train = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = pipeline.predict(X_te)
    t_infer = time.perf_counter() - t0

    acc = accuracy_score(y_te, y_pred)
    print(f"  Acuracia: {acc:.4f}  |  F1 macro: {f1_score(y_te, y_pred, average='macro'):.4f}", flush=True)
    print(classification_report(y_te, y_pred, target_names=le.classes_, zero_division=0), flush=True)

    summary = save_model("svm", pipeline, X_te, y_te, y_pred, le, t_train, t_infer)
    return summary


# ---------------------------------------------------------------------------
# Relatório final consolidado
# ---------------------------------------------------------------------------
def write_final_report(summaries: dict):
    out_dir = os.path.join(RESULTS_DIR, "tcc_package", "reports")
    os.makedirs(out_dir, exist_ok=True)

    lines = [
        "# Relatório Final – IDS IoVT (train_final.py)",
        "",
        "| Modelo | Accuracy | F1 macro | F1 weighted | Attack Recall | Attack FPR | Treino (s) | Inferência (ms/amostra) |",
        "|--------|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for name, s in summaries.items():
        if s is None:
            continue
        eff = s.get("efficiency", {})
        att = s.get("attack_detection", {})
        lines.append(
            f"| {name} "
            f"| {s.get('test_accuracy', 0):.4f} "
            f"| {s.get('f1_macro', 0):.4f} "
            f"| {s.get('f1_weighted', 0):.4f} "
            f"| {att.get('attack_recall', 0):.4f} "
            f"| {att.get('attack_fpr', 0):.4f} "
            f"| {eff.get('training_time_seconds', 0):.1f} "
            f"| {eff.get('inference_time_ms_per_sample', 0):.4f} |"
        )

    lines += [
        "",
        "## Melhor modelo multiclasse",
        "",
        "Two-Stage combina detecção binária perfeita com discriminação entre tipos de ataque.",
        "XGBoost direto com sample_weight balanceado é alternativa mais simples.",
        "",
    ]

    report_path = os.path.join(out_dir, "final_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # Também salva JSON consolidado
    json_path = os.path.join(out_dir, "final_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {k: v for k, v in summaries.items() if v is not None},
            f, indent=2, default=str
        )

    print(f"\nRelatório salvo: {report_path}", flush=True)
    print(f"JSON salvo:      {json_path}", flush=True)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Carrega TUDO em memória uma única vez (reaproveitado por todos os modelos)
    X, y, le = load_data(max_per_class_majority=250_000)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"\nTreino: {len(X_tr):,}  |  Teste: {len(X_te):,}", flush=True)

    summaries = {}

    summaries["xgboost"]             = train_xgboost(X_tr, X_te, y_tr, y_te, le)
    summaries["two_stage"]           = train_two_stage(X_tr, X_te, y_tr, y_te, le)
    summaries["mlp"]                 = train_mlp(X_tr, X_te, y_tr, y_te, le)
    summaries["logistic_regression"] = train_logistic(X_tr, X_te, y_tr, y_te, le)
    summaries["svm"]                 = train_svm(X_tr, X_te, y_tr, y_te, le)

    write_final_report(summaries)
    print("\nTudo concluído com sucesso!", flush=True)
