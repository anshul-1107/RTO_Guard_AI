"""
Live inference for a single order.

Loads the artifacts saved by ml/train.py (no MLflow needed), so the same code
runs locally, inside the LangGraph agent, and on Streamlit Cloud.

Top reasons come from LightGBM's built-in TreeSHAP (pred_contrib=True).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from ml.features import CATEGORICAL, REASON_TEXT, build_features, smoothed_rate

ARTIFACTS = Path("ml/artifacts")


@dataclass
class Reason:
    feature: str
    text: str
    impact: float  # SHAP value in log-odds; >0 pushes towards RTO


@dataclass
class Prediction:
    order_id: str
    rto_probability: float
    risk_band: str  # LOW | MEDIUM | HIGH
    threshold: float
    reasons: list[Reason] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "order_id": self.order_id,
            "rto_probability": round(self.rto_probability, 4),
            "risk_band": self.risk_band,
            "threshold": self.threshold,
            "reasons": [r.__dict__ for r in self.reasons],
        }


class RTOPredictor:
    def __init__(self, artifacts: Path = ARTIFACTS):
        self.model = lgb.Booster(model_file=str(artifacts / "model.txt"))
        self.meta = json.loads((artifacts / "meta.json").read_text())
        self.threshold: float = self.meta["threshold"]
        self.high_cutoff: float = min(0.95, self.threshold * 2)
        self.global_rate: float = self.meta["global_rto_rate"]
        pin = pd.read_parquet(artifacts / "pincode_stats.parquet")
        self.pin_stats = pin.set_index("pincode")[["orders", "rtos"]].to_dict("index")

    def _prepare(self, orders: pd.DataFrame) -> pd.DataFrame:
        df = orders.copy()
        stats = df["pincode"].map(lambda p: self.pin_stats.get(p, {"orders": 0, "rtos": 0}))
        n = stats.map(lambda s: s["orders"]).astype(int)
        r = stats.map(lambda s: s["rtos"]).astype(int)
        df["pincode_hist_orders"] = n
        df["pincode_hist_rto_rate"] = smoothed_rate(r, n, self.global_rate)
        X = build_features(df)
        # lock categories to training levels (unseen values -> NaN, handled by LightGBM)
        for c in CATEGORICAL:
            X[c] = pd.Categorical(X[c].astype(str), categories=self.meta["categorical_levels"][c])
        return X

    def _band(self, p: float) -> str:
        if p >= self.high_cutoff:
            return "HIGH"
        if p >= self.threshold:
            return "MEDIUM"
        return "LOW"

    def predict_batch(self, orders: pd.DataFrame, top_k: int = 3) -> list[Prediction]:
        X = self._prepare(orders)
        probs = self.model.predict(X)
        contrib = self.model.predict(X, pred_contrib=True)[:, :-1]  # drop bias column
        feats = X.columns.to_list()

        out = []
        for i, oid in enumerate(orders["order_id"].to_list()):
            row = contrib[i]
            idx = np.argsort(-row)[:top_k]
            reasons = [
                Reason(feats[j], REASON_TEXT.get(feats[j], feats[j]), round(float(row[j]), 3))
                for j in idx
                if row[j] > 0
            ]
            p = float(probs[i])
            out.append(Prediction(oid, p, self._band(p), self.threshold, reasons))
        return out

    def predict(self, order: dict, top_k: int = 3) -> Prediction:
        return self.predict_batch(pd.DataFrame([order]), top_k)[0]


@lru_cache(maxsize=1)
def get_predictor() -> RTOPredictor:
    return RTOPredictor()


if __name__ == "__main__":
    sample = {
        "order_id": "DEMO001",
        "order_hour": 2,
        "pincode": "899999",
        "state": "Bihar",
        "city_tier": 3,
        "courier": "Shadowfax",
        "category": "fashion",
        "payment_mode": "COD",
        "order_value": 2499.0,
        "num_items": 1,
        "discount_pct": 45.0,
        "is_first_order": True,
        "customer_prior_orders": 0,
        "customer_prior_rto": 0,
        "address_text": "Station Road, 899999",
        "address_len": 20,
        "address_has_landmark": False,
    }
    print(json.dumps(get_predictor().predict(sample).to_dict(), indent=2))
