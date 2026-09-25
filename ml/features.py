"""
Feature engineering for the RTO model.

Every feature is computable at order-placement time using only past data,
so the same code works for training and live inference (no leakage).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORICAL = ["category", "courier", "state", "payment_mode"]
NUMERIC = [
    "city_tier",
    "order_hour",
    "log_order_value",
    "num_items",
    "discount_pct",
    "is_cod",
    "is_night_order",
    "is_high_discount",
    "is_first_order",
    "customer_prior_orders",
    "customer_prior_rto",
    "customer_prior_rto_rate",
    "address_len",
    "address_has_landmark",
    "address_has_house_no",
    "pincode_hist_orders",
    "pincode_hist_rto_rate",
    "cod_x_log_value",
]
FEATURES = NUMERIC + CATEGORICAL

# Smoothing for rates computed on small histories (Bayesian average)
PRIOR_STRENGTH = 20


def smoothed_rate(rtos: pd.Series | float, n: pd.Series | float, global_rate: float) -> pd.Series:
    return (rtos + PRIOR_STRENGTH * global_rate) / (n + PRIOR_STRENGTH)


def add_pincode_history(df: pd.DataFrame, global_rate: float) -> pd.DataFrame:
    """Expanding, past-only RTO stats per pincode. Requires df sorted by order_ts."""
    g = df.groupby("pincode")["is_rto"]
    prior_n = g.cumcount()
    prior_rto = g.cumsum() - df["is_rto"].astype(int)
    df["pincode_hist_orders"] = prior_n
    df["pincode_hist_rto_rate"] = smoothed_rate(prior_rto, prior_n, global_rate)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Row-level features. Pincode history columns must already exist."""
    out = pd.DataFrame(index=df.index)
    out["city_tier"] = df["city_tier"].astype(int)
    out["order_hour"] = df["order_hour"].astype(int)
    out["log_order_value"] = np.log1p(df["order_value"].astype(float))
    out["num_items"] = df["num_items"].astype(int)
    out["discount_pct"] = df["discount_pct"].astype(float)
    out["is_cod"] = (df["payment_mode"] == "COD").astype(int)
    out["is_night_order"] = df["order_hour"].between(0, 4).astype(int)
    out["is_high_discount"] = (df["discount_pct"] > 30).astype(int)
    out["is_first_order"] = df["is_first_order"].astype(int)
    out["customer_prior_orders"] = df["customer_prior_orders"].astype(int)
    out["customer_prior_rto"] = df["customer_prior_rto"].astype(int)
    out["customer_prior_rto_rate"] = (
        df["customer_prior_rto"] / df["customer_prior_orders"].replace(0, np.nan)
    ).fillna(-1)  # -1 = no history (tree models handle this cleanly)
    out["address_len"] = df["address_len"].astype(int)
    out["address_has_landmark"] = df["address_has_landmark"].astype(int)
    out["address_has_house_no"] = df["address_text"].str.contains("H.No", regex=False).astype(int)
    out["pincode_hist_orders"] = df["pincode_hist_orders"].astype(int)
    out["pincode_hist_rto_rate"] = df["pincode_hist_rto_rate"].astype(float)
    out["cod_x_log_value"] = out["is_cod"] * out["log_order_value"]
    for c in CATEGORICAL:
        out[c] = df[c].astype("category")
    return out[FEATURES]


# Human-readable explanations used by the agent and dashboard
REASON_TEXT = {
    "is_cod": "Cash-on-delivery order",
    "payment_mode": "Cash-on-delivery order",
    "cod_x_log_value": "High-value COD order",
    "log_order_value": "Order value",
    "city_tier": "Tier-2/3 delivery location",
    "state": "Delivery state",
    "pincode_hist_rto_rate": "Pincode has high past RTO rate",
    "pincode_hist_orders": "Little order history for this pincode",
    "is_first_order": "First-time customer",
    "customer_prior_orders": "Customer order history",
    "customer_prior_rto": "Customer has past RTOs",
    "customer_prior_rto_rate": "Customer's past RTO rate",
    "address_len": "Short / incomplete address",
    "address_has_landmark": "No landmark in address",
    "address_has_house_no": "No house number in address",
    "is_night_order": "Late-night order (12–5 AM)",
    "order_hour": "Order time",
    "is_high_discount": "Heavy discount (>30%)",
    "discount_pct": "Discount level",
    "category": "Product category",
    "courier": "Courier partner",
    "num_items": "Number of items",
}
