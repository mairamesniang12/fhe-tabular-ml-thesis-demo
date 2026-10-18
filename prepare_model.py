"""
One-time setup: for each of DT / RF / XGB on WDBC --
  1. Train the plaintext model, measure its plaintext accuracy/F1.
  2. Fit and compile the FHE-equivalent model, with the same
     hyperparameters, seed and calibration draw as the thesis notebook.
  3. Evaluate it with Concrete-ML's "simulate" mode on the FULL local
     test set to get a genuine FHE-simulate accuracy.
  4. Run a SMALL number of REAL encrypted executions (stratified
     subsample) to get a measured mean FHE latency. This number uses a
     deliberately reduced deployment budget and is labeled as such
     wherever it is displayed.
  5. Package the model with FHEModelDev for the client/server deployment.

Everything computed here is written to real_metrics.json, which the
Streamlit app reads verbatim.

Protocol (aligned with the thesis notebook, Chapter 5):
  - 80/20 stratified split, seed 42
  - FHE models fitted with random_state=42, D=4, p=5 (T=15 for RF/XGB)
  - calibration set: 50 training samples drawn with RandomState(42)
  - real-FHE subsample: stratified, drawn with the same RandomState(42)
    (after the calibration draw), then shuffled
  - one warm-up call excluded from timing
  - sample standard deviation (ddof=1)
"""
import json
import os
import shutil
import time

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from concrete.ml.deployment import FHEModelDev
from concrete.ml.sklearn import (
    DecisionTreeClassifier as FHE_DT,
    RandomForestClassifier as FHE_RF,
    XGBClassifier as FHE_XGB,
)

RANDOM_STATE = 42
N_CALIBRATION = 50
PROTOCOL_VERSION = "v2-seeded"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
METRICS_FILE = os.path.join(BASE_DIR, "real_metrics.json")

# Reduced deployment budget (Streamlit Community Cloud free-tier memory
# and time limits). The thesis protocol uses 30 observations and 5 runs.
# N_REAL_FHE_SAMPLES_AT_DEPLOY should be even (split equally per class).
N_REAL_FHE_SAMPLES_AT_DEPLOY = 16
N_LATENCY_REPEATS_AT_DEPLOY = 2

MODEL_CONFIGS = {
    "Decision Tree (DT)": {
        "key": "dt",
        "plain_cls": DecisionTreeClassifier,
        "fhe_cls": FHE_DT,
        "kwargs": {"max_depth": 4},
        "fhe_kwargs": {"max_depth": 4, "n_bits": 5},
    },
    "Random Forest (RF)": {
        "key": "rf",
        "plain_cls": RandomForestClassifier,
        "fhe_cls": FHE_RF,
        "kwargs": {"n_estimators": 15, "max_depth": 4},
        "fhe_kwargs": {"n_estimators": 15, "max_depth": 4, "n_bits": 5},
    },
    "XGBoost (XGB)": {
        "key": "xgb",
        "plain_cls": XGBClassifier,
        "fhe_cls": FHE_XGB,
        "kwargs": {
            "n_estimators": 15, "max_depth": 4,
            "eval_metric": "logloss", "verbosity": 0,
        },
        "fhe_kwargs": {"n_estimators": 15, "max_depth": 4, "n_bits": 5},
    },
}


def deploy_dir_for(model_key: str) -> str:
    return os.path.join(BASE_DIR, f"fhe_deployment_wdbc_{model_key}")


def _already_prepared():
    """True only if metrics exist for the CURRENT protocol version and
    every model has been packaged."""
    if not os.path.exists(METRICS_FILE):
        return None
    with open(METRICS_FILE) as f:
        existing = json.load(f)
    ok = all(
        name in existing
        and existing[name].get("protocol_version") == PROTOCOL_VERSION
        and os.path.exists(os.path.join(deploy_dir_for(cfg["key"]), ".ready"))
        for name, cfg in MODEL_CONFIGS.items()
    )
    return existing if ok else None


def prepare_all_models():
    existing = _already_prepared()
    if existing is not None:
        print("[prepare_model] All models already prepared -- skipping.")
        return existing

    wdbc = load_breast_cancer()
    X_train, X_test, y_train, y_test = train_test_split(
        wdbc.data, wdbc.target, test_size=0.2, random_state=RANDOM_STATE,
        stratify=wdbc.target,
    )

    all_metrics = {}

    for display_name, cfg in MODEL_CONFIGS.items():
        print(f"[prepare_model] Preparing {display_name}...")
        deploy_dir = deploy_dir_for(cfg["key"])
        if os.path.exists(deploy_dir):
            shutil.rmtree(deploy_dir)
        os.makedirs(deploy_dir)

        # ---- 1. Plaintext model ----
        plain_model = cfg["plain_cls"](random_state=RANDOM_STATE, **cfg["kwargs"])
        t0 = time.time()
        plain_model.fit(X_train, y_train)
        train_time = time.time() - t0
        y_pred_plain = plain_model.predict(X_test)
        acc_plain = accuracy_score(y_test, y_pred_plain)
        f1_plain = f1_score(y_test, y_pred_plain, average="weighted")

        # ---- 2. FHE model: fit (seeded) + compile (random calibration) ----
        fhe_model = cfg["fhe_cls"](random_state=RANDOM_STATE, **cfg["fhe_kwargs"])
        fhe_model.fit(X_train, y_train)

        rng = np.random.RandomState(RANDOM_STATE)
        n_calib = min(N_CALIBRATION, len(X_train))
        calib_idx = rng.choice(len(X_train), size=n_calib, replace=False)
        X_calib = X_train[calib_idx]

        t0 = time.time()
        fhe_model.compile(X_calib)
        compile_time = time.time() - t0

        # ---- 3. FHE-simulate accuracy on the FULL local test set ----
        y_pred_sim = fhe_model.predict(X_test, fhe="simulate")
        acc_fhe_sim = accuracy_score(y_test, y_pred_sim)
        f1_fhe_sim = f1_score(y_test, y_pred_sim, average="weighted")

        # ---- 4. A SMALL number of REAL encrypted executions ----
        classes = np.unique(y_test)
        n_per_class = max(1, N_REAL_FHE_SAMPLES_AT_DEPLOY // len(classes))
        sample_idx = []
        for c in classes:
            c_idx = np.where(y_test == c)[0]
            chosen = rng.choice(c_idx, size=min(n_per_class, len(c_idx)), replace=False)
            sample_idx.extend(chosen.tolist())
        sample_idx = np.array(sample_idx)
        rng.shuffle(sample_idx)
        X_real = X_test[sample_idx]

        _ = fhe_model.predict(X_real[:1], fhe="execute")  # warm-up, not timed
        lat_runs = []
        for _ in range(N_LATENCY_REPEATS_AT_DEPLOY):
            t0 = time.time()
            fhe_model.predict(X_real, fhe="execute")
            lat_runs.append((time.time() - t0) / len(X_real))
        latency_mean_s = float(np.mean(lat_runs))
        latency_std_s = (
            float(np.std(lat_runs, ddof=1)) if len(lat_runs) > 1 else 0.0
        )

        # ---- 5. Package for real client/server deployment ----
        dev = FHEModelDev(path_dir=deploy_dir, model=fhe_model)
        dev.save()
        with open(os.path.join(deploy_dir, ".ready"), "w") as f:
            f.write("ready\n")

        all_metrics[display_name] = {
            "model_key": cfg["key"],
            "deploy_dir": deploy_dir,
            "protocol_version": PROTOCOL_VERSION,
            "random_seed": RANDOM_STATE,
            "n_calibration": int(n_calib),
            "train_time_s": round(train_time, 4),
            "compile_time_s": round(compile_time, 4),
            "acc_plain": round(float(acc_plain), 4),
            "f1_plain": round(float(f1_plain), 4),
            "acc_fhe_simulate_full": round(float(acc_fhe_sim), 4),
            "f1_fhe_simulate_full": round(float(f1_fhe_sim), 4),
            "n_test_samples_full": int(len(y_test)),
            "n_real_fhe_samples": int(len(X_real)),
            "n_latency_repeats": N_LATENCY_REPEATS_AT_DEPLOY,
            "latency_fhe_mean_s": round(latency_mean_s, 4),
            "latency_fhe_std_s": round(latency_std_s, 4),
        }
        print(
            f"[prepare_model] {display_name}: "
            f"acc_plain={acc_plain:.4f} acc_fhe_sim={acc_fhe_sim:.4f} "
            f"latency={latency_mean_s:.3f}s (n={len(X_real)} real samples)"
        )

    with open(METRICS_FILE, "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"[prepare_model] Done. Metrics written to {METRICS_FILE}")
    return all_metrics


if __name__ == "__main__":
    prepare_all_models()
