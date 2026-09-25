import pandas as pd

from data.generate import generate


def _df() -> pd.DataFrame:
    return generate(n_customers=2000, n_orders=10000, seed=7)


def test_shape_and_ids_unique():
    df = _df()
    assert len(df) == 10000
    assert df["order_id"].is_unique


def test_rto_rates_realistic():
    df = _df()
    cod = df.loc[df.payment_mode == "COD", "is_rto"].mean()
    prepaid = df.loc[df.payment_mode == "PREPAID", "is_rto"].mean()
    assert 0.15 < cod < 0.40
    assert prepaid < 0.10
    assert cod > prepaid * 3


def test_tier3_riskier_than_tier1():
    rates = _df().groupby("city_tier")["is_rto"].mean()
    assert rates[3] > rates[1]


def test_no_leakage_in_prior_rto():
    """prior_rto must never exceed prior_orders, and first orders have zero history."""
    df = _df()
    assert (df.customer_prior_rto <= df.customer_prior_orders).all()
    assert (df.loc[df.is_first_order, "customer_prior_rto"] == 0).all()


def test_hidden_columns_not_exported():
    cols = set(_df().columns)
    assert not {"_addr_q", "_pin_risk", "customer_idx"} & cols


def test_deterministic_with_seed():
    a = generate(500, 2000, seed=1)
    b = generate(500, 2000, seed=1)
    pd.testing.assert_frame_equal(a, b)
