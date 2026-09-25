"""
Train the RTO risk model.

- Time-based split: train (first 70%) / val (next 10%) / test (last 20%)
- LightGBM, early stopping on val AUC, calibrated probabilities (no loss reweighting)
- Decision threshold chosen on val by maximising net INR savings (ml/economics.py)
- Everything logged to MLflow; deployable artifacts saved to ml/artifacts/

Run:  python -m ml.train
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ml.economics import ECON
from ml.features import CATEGORICAL, FEATURES, add_pincode_history, build_features

DATA = Path("data/raw/orders.parquet")
ARTIFACTS = Path("ml/artifacts")

PARAMS = {
    "objective": "binary",
    "metric": ["auc", "binary_logloss"],
    "first_metric_only": True,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 100,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": 42,
}


def load() -> pd.DataFrame:
    df = pd.read_parquet(DATA).sort_values("order_ts", kind="stable").reset_index(drop=True)
    return df


def time_split(n: int) -> tuple[slice, slice, slice]:
    a, b = int(n * 0.70), int(n * 0.80)
    return slice(0, a), slice(a, b), slice(b, n)


def best_threshold(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    grid = np.round(np.arange(0.05, 0.95, 0.01), 2)
    values = [ECON.net_value(y, (p >= t).astype(int)).sum() for t in grid]
    i = int(np.argmax(values))
    return float(grid[i]), float(values[i])


def evaluate(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    flagged = (p >= threshold).astype(int)
    net = ECON.net_value(y, flagged)
    baseline_loss = y.sum() * ECON.rto_cost
    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "precision": precision_score(y, flagged, zero_division=0),
        "recall": recall_score(y, flagged),
        "flag_rate": flagged.mean(),
        "net_savings_inr": float(net.sum()),
        "net_savings_per_1k_orders": float(net.sum() / len(y) * 1000),
        "rto_loss_reduction_pct": float(net.sum() / baseline_loss * 100),
    }


def main() -> None:
    df = load()
    global_rate = float(df["is_rto"].iloc[: int(len(df) * 0.7)].mean())  # train-only
    df = add_pincode_history(df, global_rate)
    X = build_features(df)
    y = df["is_rto"].astype(int).to_numpy()

    tr, va, te = time_split(len(df))
    # No scale_pos_weight: we need calibrated probabilities for cost-based decisions.
    # Imbalance is handled by the threshold, not by distorting the loss.
    params = dict(PARAMS)

    mlflow.set_experiment("rto-guard")
    with mlflow.start_run(run_name="lgbm"):
        mlflow.log_params(params)
        mlflow.log_params({k: v for k, v in ECON.__dict__.items()})

        dtrain = lgb.Dataset(X.iloc[tr], y[tr], categorical_feature=CATEGORICAL)
        dval = lgb.Dataset(X.iloc[va], y[va], categorical_feature=CATEGORICAL, reference=dtrain)
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(0)],
        )

        p_val = model.predict(X.iloc[va], num_iteration=model.best_iteration)
        threshold, _ = best_threshold(y[va], p_val)

        p_test = model.predict(X.iloc[te], num_iteration=model.best_iteration)
        metrics = evaluate(y[te], p_test, threshold)
        metrics["threshold"] = threshold
        metrics["best_iteration"] = model.best_iteration

        # naive baseline: flag every COD order
        cod_flag = (X.iloc[te]["is_cod"].to_numpy()).astype(int)
        metrics["baseline_all_cod_savings_inr"] = float(ECON.net_value(y[te], cod_flag).sum())

        mlflow.log_metrics(metrics)

        # --- deployable artifacts (used by predictor + Streamlit, no MLflow needed) ---
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        model.save_model(ARTIFACTS / "model.txt", num_iteration=model.best_iteration)

        pin_stats = (
            df.groupby("pincode")["is_rto"].agg(orders="count", rtos="sum").reset_index()
        )
        pin_stats.to_parquet(ARTIFACTS / "pincode_stats.parquet", index=False)

        cat_levels = {c: X[c].cat.categories.tolist() for c in CATEGORICAL}
        meta = {
            "features": FEATURES,
            "categorical_levels": cat_levels,
            "threshold": threshold,
            "global_rto_rate": global_rate,
            "metrics": metrics,
        }
        (ARTIFACTS / "meta.json").write_text(json.dumps(meta, indent=2))
        mlflow.log_artifacts(str(ARTIFACTS), artifact_path="deploy")

    print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()},
                     indent=2))


if __name__ == "__main__":
    main()
