"""
Semi-synthetic Indian D2C order generator for RTO Guard.

RTO labels come from a hidden logistic model (true drivers + noise), so the
ML model in Step 3 has real but imperfect signal to learn. Hidden drivers such
as customer propensity and pincode risk are NOT exported as columns; the model
has to learn them through observable proxies (prior RTO count, tier, etc.).

Run:  python -m data.generate
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from rto_guard.config import settings

OUT_DIR = Path("data/raw")

STATES = {
    # state: (tier weights t1/t2/t3, pincode prefix)
    "Maharashtra": ((0.45, 0.30, 0.25), "4"),
    "Karnataka": ((0.50, 0.25, 0.25), "5"),
    "Delhi": ((0.90, 0.10, 0.00), "1"),
    "Tamil Nadu": ((0.35, 0.35, 0.30), "6"),
    "Uttar Pradesh": ((0.15, 0.35, 0.50), "2"),
    "Bihar": ((0.05, 0.30, 0.65), "8"),
    "West Bengal": ((0.30, 0.30, 0.40), "7"),
    "Gujarat": ((0.35, 0.35, 0.30), "3"),
    "Rajasthan": ((0.15, 0.40, 0.45), "3"),
    "Kerala": ((0.20, 0.45, 0.35), "6"),
}
STATE_WEIGHTS = np.array([0.14, 0.12, 0.10, 0.09, 0.14, 0.07, 0.08, 0.08, 0.08, 0.10])

CATEGORIES = ["fashion", "beauty", "electronics", "home", "footwear"]
CATEGORY_WEIGHTS = [0.35, 0.20, 0.15, 0.15, 0.15]
CATEGORY_PRICE = {  # lognormal (mean of log, sigma)
    "fashion": (6.6, 0.5),
    "beauty": (6.2, 0.5),
    "electronics": (7.6, 0.7),
    "home": (6.9, 0.6),
    "footwear": (7.0, 0.5),
}
CATEGORY_EFFECT = {
    "fashion": 0.40, "footwear": 0.30, "electronics": 0.15, "beauty": 0.0, "home": -0.10,
}

COURIERS = ["Delhivery", "Bluedart", "Ecom Express", "Xpressbees", "Shadowfax"]
COURIER_WEIGHTS = [0.35, 0.15, 0.20, 0.20, 0.10]
COURIER_EFFECT = {
    "Delhivery": 0.0, "Bluedart": -0.25, "Ecom Express": 0.10,
    "Xpressbees": 0.05, "Shadowfax": 0.15,
}

TIER_EFFECT = {1: 0.0, 2: 0.35, 3: 0.70}
COD_SHARE_BY_TIER = {1: 0.45, 2: 0.65, 3: 0.80}

FIRST_NAMES = [
    "Aarav", "Priya", "Rahul", "Ananya", "Vikram", "Sneha", "Arjun", "Pooja", "Rohan", "Kavya",
    "Amit", "Neha", "Suresh", "Divya", "Karan", "Meera", "Imran", "Fatima", "Joseph", "Lakshmi",
]

STREETS = ["MG Road", "Station Road", "Main Bazaar", "Gandhi Nagar", "Nehru Colony", "Ring Road"]
LANDMARKS = [
    "near SBI ATM", "opp. Govt School", "behind Hanuman Mandir",
    "near Bus Stand", "next to Petrol Pump",
]


@dataclass
class Pincodes:
    code: np.ndarray
    state: np.ndarray
    tier: np.ndarray
    risk: np.ndarray  # hidden


def make_pincodes(rng: np.random.Generator, n: int = 2000) -> Pincodes:
    state_names = list(STATES)
    states = rng.choice(state_names, size=n, p=STATE_WEIGHTS / STATE_WEIGHTS.sum())
    tiers = np.array([rng.choice([1, 2, 3], p=STATES[s][0]) for s in states])
    codes = np.array(
        [f"{STATES[s][1]}{rng.integers(10000, 99999)}" for s in states]
    )
    risk = rng.normal(0, 0.4, size=n)
    return Pincodes(code=codes, state=states, tier=tiers, risk=risk)


def make_address(rng: np.random.Generator, pincode: str, quality: float) -> tuple[str, bool]:
    """Low-quality addresses are short, missing house no / landmark."""
    parts = []
    if quality > 0.35:
        parts.append(f"H.No {rng.integers(1, 999)}")
    parts.append(str(rng.choice(STREETS)))
    has_landmark = quality > 0.55 and rng.random() < 0.8
    if has_landmark:
        parts.append(str(rng.choice(LANDMARKS)))
    parts.append(pincode)
    return ", ".join(parts), bool(has_landmark)


def generate(n_customers: int, n_orders: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pins = make_pincodes(rng)

    # --- customers (hidden propensity + home pincode + address habit) ---
    cust_ids = np.array([f"C{i:06d}" for i in range(n_customers)])
    cust_pin_idx = rng.integers(0, len(pins.code), size=n_customers)
    cust_propensity = rng.normal(0, 0.8, size=n_customers)  # hidden
    cust_addr_quality = np.clip(rng.beta(4, 2, size=n_customers), 0, 1)
    cust_activity = rng.pareto(1.5, size=n_customers) + 1  # few heavy buyers
    cust_name = rng.choice(FIRST_NAMES, size=n_customers)
    cust_phone = np.array([f"9{rng.integers(100000000, 999999999)}" for _ in range(n_customers)])

    # --- orders ---
    order_cust = rng.choice(n_customers, size=n_orders, p=cust_activity / cust_activity.sum())
    start = pd.Timestamp("2025-09-01")
    offsets = rng.integers(0, 365 * 24 * 3600, size=n_orders)
    order_ts = start + pd.to_timedelta(offsets, unit="s")

    pin_idx = cust_pin_idx[order_cust]
    tier = pins.tier[pin_idx]
    category = rng.choice(CATEGORIES, size=n_orders, p=CATEGORY_WEIGHTS)
    courier = rng.choice(COURIERS, size=n_orders, p=COURIER_WEIGHTS)

    mu = np.array([CATEGORY_PRICE[c][0] for c in category])
    sd = np.array([CATEGORY_PRICE[c][1] for c in category])
    order_value = np.round(np.exp(rng.normal(mu, sd)), 2)
    num_items = rng.choice([1, 1, 1, 2, 2, 3, 4], size=n_orders)
    discount_pct = np.round(np.clip(rng.gamma(2, 8, size=n_orders), 0, 70), 2)

    cod_p = np.array([COD_SHARE_BY_TIER[t] for t in tier])
    is_cod = rng.random(n_orders) < cod_p

    # per-order address quality jitters around the customer's habit
    addr_q = np.clip(cust_addr_quality[order_cust] + rng.normal(0, 0.1, n_orders), 0, 1)

    hour = pd.DatetimeIndex(order_ts).hour.to_numpy()

    df = pd.DataFrame(
        {
            "customer_idx": order_cust,
            "order_ts": order_ts,
            "order_hour": hour,
            "pincode": pins.code[pin_idx],
            "state": pins.state[pin_idx],
            "city_tier": tier,
            "courier": courier,
            "category": category,
            "payment_mode": np.where(is_cod, "COD", "PREPAID"),
            "order_value": order_value,
            "num_items": num_items,
            "discount_pct": discount_pct,
            "_addr_q": addr_q,
            "_pin_risk": pins.risk[pin_idx],
        }
    ).sort_values("order_ts", kind="stable").reset_index(drop=True)

    df["customer_id"] = cust_ids[df["customer_idx"]]
    df["customer_phone"] = cust_phone[df["customer_idx"]]
    df["customer_name"] = cust_name[df["customer_idx"]]
    df["customer_prior_orders"] = df.groupby("customer_idx").cumcount()
    df["is_first_order"] = df["customer_prior_orders"] == 0

    # --- hidden RTO logit ---
    cod = (df["payment_mode"] == "COD").to_numpy()
    logit = (
        -3.9
        + 2.0 * cod
        + df["city_tier"].map(TIER_EFFECT).to_numpy()
        + df["_pin_risk"].to_numpy()
        + cust_propensity[df["customer_idx"]]
        + 0.40 * df["is_first_order"].to_numpy()
        - 1.4 * (df["_addr_q"].to_numpy() - 0.5)
        + cod * 0.35 * np.log(df["order_value"].to_numpy() / 800)
        + 0.45 * df["order_hour"].between(0, 4).to_numpy()
        + 0.30 * (df["discount_pct"].to_numpy() > 30)
        + df["category"].map(CATEGORY_EFFECT).to_numpy()
        + df["courier"].map(COURIER_EFFECT).to_numpy()
    )
    p = 1 / (1 + np.exp(-logit))
    df["is_rto"] = rng.random(len(df)) < p

    # prior RTO count uses only PAST orders (no leakage)
    df["customer_prior_rto"] = (
        df.groupby("customer_idx")["is_rto"].cumsum() - df["is_rto"]
    ).astype(int)

    addrs = [make_address(rng, pc, q) for pc, q in zip(df["pincode"], df["_addr_q"], strict=True)]
    df["address_text"] = [a for a, _ in addrs]
    df["address_has_landmark"] = [lm for _, lm in addrs]
    df["address_len"] = df["address_text"].str.len()

    df["order_id"] = [f"ORD{i:07d}" for i in range(len(df))]

    cols = [
        "order_id", "customer_id", "order_ts", "order_hour", "pincode", "state",
        "city_tier", "courier", "category", "payment_mode", "order_value", "num_items",
        "discount_pct", "is_first_order", "customer_prior_orders", "customer_prior_rto",
        "address_text", "address_len", "address_has_landmark", "customer_name", "customer_phone",
        "is_rto",
    ]
    return df[cols]


def summarize(df: pd.DataFrame) -> None:
    print(f"orders: {len(df):,}  customers: {df['customer_id'].nunique():,}")
    print(f"overall RTO rate: {df['is_rto'].mean():.1%}")
    print(df.groupby("payment_mode")["is_rto"].mean().map("{:.1%}".format).to_string())
    print(df.groupby("city_tier")["is_rto"].mean().map("{:.1%}".format).to_string())


def main() -> None:
    df = generate(settings.n_customers, settings.n_orders, settings.data_seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "orders.parquet"
    df.to_parquet(out, index=False)
    summarize(df)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
