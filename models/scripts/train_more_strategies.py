import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_PATH = os.path.join(ROOT, "data", "processed", "all_datasets_aligned_balanced.csv")
OUT_DIR = os.path.join(ROOT, "results", "metrics", "strategy_tests")

BYTE_COLS = ["val0", "val1", "val2", "val3", "val4", "val5", "val6", "val7"]
NUMERIC_COLS = ["timestamp", "id", "dlc"] + BYTE_COLS
DATA_CACHE: Dict[str, Tuple[pd.DataFrame, pd.Series, List[str]]] = {}


@dataclass
class ExperimentResult:
    experiment: str
    mode: str
    feature_variant: str
    model: str
    n_train: int
    n_test: int
    accuracy: float
    f1_macro: float
    f1_weighted: float
    train_seconds: float
    infer_seconds: float


def ensure_out_dir() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)


def add_can_features(df: pd.DataFrame) -> pd.DataFrame:
    b = df[BYTE_COLS].values.astype(np.float32)
    df["byte_sum"] = b.sum(axis=1)
    df["byte_mean"] = b.mean(axis=1)
    df["byte_std"] = b.std(axis=1)
    df["byte_max"] = b.max(axis=1)
    df["byte_min"] = b.min(axis=1)
    df["byte_range"] = df["byte_max"] - df["byte_min"]
    df["nonzero_bytes"] = (b != 0).sum(axis=1)
    df["id_mod_16"] = (df["id"] % 16).astype(np.int16)
    df["id_mod_256"] = (df["id"] % 256).astype(np.int16)
    df["id_dlc"] = df["id"].astype(np.float32) * df["dlc"].astype(np.float32)
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["dt"] = df["timestamp"].diff().fillna(0).clip(lower=-1e6, upper=1e6)
    df["did"] = df["id"].diff().fillna(0).clip(lower=-1e5, upper=1e5)
    df["ddl"] = df["dlc"].diff().fillna(0).clip(lower=-16, upper=16)
    return df


def to_binary_flag(series: pd.Series) -> pd.Series:
    return (series.astype(str).str.upper() != "BENIGN").astype(np.int8)


def load_balanced(
    mode: str,
    max_per_class: int,
    feature_variant: str,
    max_chunks: int,
    chunk_size: int = 500_000,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    if mode not in {"multiclass", "binary"}:
        raise ValueError("mode must be multiclass or binary")

    cache_key = f"{mode}|{feature_variant}|{max_per_class}|{max_chunks}|{chunk_size}"
    if cache_key in DATA_CACHE:
        Xc, yc, names = DATA_CACHE[cache_key]
        return Xc.copy(), yc.copy(), names[:]

    print(
        f"Loading mode={mode} feature_variant={feature_variant} max_chunks={max_chunks} ...",
        flush=True,
    )

    first = pd.read_csv(DATA_PATH, usecols=["flag"], nrows=200_000, low_memory=False)
    labels = sorted(first["flag"].dropna().astype(str).unique())

    if mode == "multiclass":
        target_names = labels
        buckets: Dict[str, List[pd.DataFrame]] = {k: [] for k in target_names}
        filled = {k: False for k in target_names}
    else:
        target_names = ["BENIGN", "ATTACK"]
        buckets = {k: [] for k in target_names}
        filled = {k: False for k in target_names}

    for i, chunk in enumerate(pd.read_csv(DATA_PATH, chunksize=chunk_size, low_memory=False)):
        if i >= max_chunks:
            break
        if all(filled.values()):
            break

        chunk = chunk.dropna(subset=["flag"]).copy()
        if "source" in chunk.columns:
            chunk = chunk.drop(columns=["source"])

        for col in NUMERIC_COLS:
            chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
        chunk = chunk.fillna(0)

        chunk = add_can_features(chunk)
        if feature_variant == "temporal":
            chunk = add_temporal_features(chunk)

        if mode == "binary":
            chunk["target"] = np.where(chunk["flag"].astype(str).str.upper() == "BENIGN", "BENIGN", "ATTACK")
        else:
            chunk["target"] = chunk["flag"].astype(str)

        for t in target_names:
            if filled[t]:
                continue
            sub = chunk[chunk["target"] == t]
            if sub.empty:
                continue

            current = sum(len(p) for p in buckets[t])
            need = max_per_class - current
            if need <= 0:
                filled[t] = True
                continue

            if len(sub) > need:
                sub = sub.sample(n=need, random_state=42)
                filled[t] = True

            buckets[t].append(sub)

        counts = {k: sum(len(p) for p in buckets[k]) for k in target_names}
        print(f"  chunk {i+1:3d} counts={counts}", flush=True)

    per_target_frames: Dict[str, pd.DataFrame] = {}
    for t, parts in buckets.items():
        if parts:
            per_target_frames[t] = pd.concat(parts, ignore_index=True)

    available = {k: len(v) for k, v in per_target_frames.items() if len(v) > 0}
    if not available:
        raise RuntimeError("No samples collected for this strategy.")

    min_count = min(available.values())
    if min_count < 500:
        raise RuntimeError(f"Too few samples collected (min class count={min_count}).")

    balanced_parts = []
    for t in target_names:
        if t not in per_target_frames:
            continue
        part = per_target_frames[t]
        if len(part) > min_count:
            part = part.sample(n=min_count, random_state=42)
        balanced_parts.append(part)

    df = pd.concat(balanced_parts, ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)

    y = df.pop("target")
    _ = df.pop("flag")
    X = df

    DATA_CACHE[cache_key] = (X.copy(), y.copy(), target_names[:])
    return X, y, target_names


def run_multiclass_xgb(feature_variant: str, max_per_class: int, max_chunks: int) -> Tuple[ExperimentResult, str]:
    X, y_raw, target_names = load_balanced(
        "multiclass",
        max_per_class=max_per_class,
        feature_variant=feature_variant,
        max_chunks=max_chunks,
    )
    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = XGBClassifier(
        objective="multi:softprob",
        eval_metric="mlogloss",
        n_estimators=450,
        max_depth=8,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        min_child_weight=1,
        reg_alpha=0.05,
        reg_lambda=1.0,
        tree_method="hist",
        random_state=42,
    )

    t0 = time.perf_counter()
    model.fit(X_train, y_train, verbose=False)
    train_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    infer_s = time.perf_counter() - t0

    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    f1w = f1_score(y_test, y_pred, average="weighted")

    report = classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0)
    result = ExperimentResult(
        experiment=f"multiclass_xgb_{feature_variant}",
        mode="multiclass",
        feature_variant=feature_variant,
        model="xgboost",
        n_train=int(X_train.shape[0]),
        n_test=int(X_test.shape[0]),
        accuracy=float(acc),
        f1_macro=float(f1m),
        f1_weighted=float(f1w),
        train_seconds=float(train_s),
        infer_seconds=float(infer_s),
    )
    return result, report


def run_binary_xgb(feature_variant: str, max_per_class: int, max_chunks: int) -> Tuple[ExperimentResult, str]:
    X, y_raw, _ = load_balanced(
        "binary",
        max_per_class=max_per_class,
        feature_variant=feature_variant,
        max_chunks=max_chunks,
    )
    y = to_binary_flag(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=500,
        max_depth=8,
        learning_rate=0.06,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.05,
        reg_lambda=1.0,
        tree_method="hist",
        random_state=42,
    )

    t0 = time.perf_counter()
    model.fit(X_train, y_train, verbose=False)
    train_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    infer_s = time.perf_counter() - t0

    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    f1w = f1_score(y_test, y_pred, average="weighted")

    report = classification_report(y_test, y_pred, target_names=["BENIGN", "ATTACK"], zero_division=0)
    result = ExperimentResult(
        experiment=f"binary_xgb_{feature_variant}",
        mode="binary",
        feature_variant=feature_variant,
        model="xgboost",
        n_train=int(X_train.shape[0]),
        n_test=int(X_test.shape[0]),
        accuracy=float(acc),
        f1_macro=float(f1m),
        f1_weighted=float(f1w),
        train_seconds=float(train_s),
        infer_seconds=float(infer_s),
    )
    return result, report


def run_binary_rf(feature_variant: str, max_per_class: int, max_chunks: int) -> Tuple[ExperimentResult, str]:
    X, y_raw, _ = load_balanced(
        "binary",
        max_per_class=max_per_class,
        feature_variant=feature_variant,
        max_chunks=max_chunks,
    )
    y = to_binary_flag(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    infer_s = time.perf_counter() - t0

    acc = accuracy_score(y_test, y_pred)
    f1m = f1_score(y_test, y_pred, average="macro")
    f1w = f1_score(y_test, y_pred, average="weighted")

    report = classification_report(y_test, y_pred, target_names=["BENIGN", "ATTACK"], zero_division=0)
    result = ExperimentResult(
        experiment=f"binary_rf_{feature_variant}",
        mode="binary",
        feature_variant=feature_variant,
        model="random_forest",
        n_train=int(X_train.shape[0]),
        n_test=int(X_test.shape[0]),
        accuracy=float(acc),
        f1_macro=float(f1m),
        f1_weighted=float(f1w),
        train_seconds=float(train_s),
        infer_seconds=float(infer_s),
    )
    return result, report


def save_outputs(results: List[ExperimentResult], reports: Dict[str, str]) -> None:
    ensure_out_dir()

    summary_json = os.path.join(OUT_DIR, "strategy_experiments_summary.json")
    summary_csv = os.path.join(OUT_DIR, "strategy_experiments_summary.csv")
    reports_txt = os.path.join(OUT_DIR, "strategy_experiments_reports.txt")

    rows = [asdict(r) for r in results]
    rows = sorted(rows, key=lambda x: (x["accuracy"], x["f1_macro"]), reverse=True)

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    pd.DataFrame(rows).to_csv(summary_csv, index=False)

    with open(reports_txt, "w", encoding="utf-8") as f:
        for name, rep in reports.items():
            f.write(f"\n{'='*80}\n{name}\n{'='*80}\n")
            f.write(rep)
            f.write("\n")

    print(f"Saved: {summary_json}")
    print(f"Saved: {summary_csv}")
    print(f"Saved: {reports_txt}")


def main() -> None:
    tests = [
        ("multiclass_xgb_engineered", lambda: run_multiclass_xgb("engineered", max_per_class=80_000, max_chunks=12)),
        ("multiclass_xgb_temporal", lambda: run_multiclass_xgb("temporal", max_per_class=80_000, max_chunks=12)),
        ("binary_xgb_temporal", lambda: run_binary_xgb("temporal", max_per_class=250_000, max_chunks=8)),
        ("binary_rf_engineered", lambda: run_binary_rf("engineered", max_per_class=250_000, max_chunks=8)),
    ]

    all_results: List[ExperimentResult] = []
    all_reports: Dict[str, str] = {}

    for name, fn in tests:
        print("\n" + "=" * 80)
        print(f"Running {name}")
        print("=" * 80)
        result, report = fn()
        all_results.append(result)
        all_reports[name] = report
        print(
            f"{name}: acc={result.accuracy:.6f} f1_macro={result.f1_macro:.6f} "
            f"train_s={result.train_seconds:.2f}",
            flush=True,
        )

    save_outputs(all_results, all_reports)


if __name__ == "__main__":
    main()
