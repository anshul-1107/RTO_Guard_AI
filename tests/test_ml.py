from pathlib import Path

import pandas as pd
import pytest

from ml.economics import ECON
from ml.features import add_pincode_history

ARTIFACTS = Path("ml/artifacts/model.txt")
needs_model = pytest.mark.skipif(not ARTIFACTS.exists(), reason="run `make train` first")

BASE = {
    "order_id": "T1",
    "order_hour": 14,
    "pincode": "560001",
    "state": "Karnataka",
    "city_tier": 1,
    "courier": "Bluedart",
    "category": "beauty",
    "payment_mode": "PREPAID",
    "order_value": 499.0,
    "num_items": 1,
    "discount_pct": 5.0,
    "is_first_order": False,
    "customer_prior_orders": 8,
    "customer_prior_rto": 0,
    "address_text": "H.No 12, MG Road, near SBI ATM, 560001",
    "address_len": 38,
    "address_has_landmark": True,
}
RISKY = {
    **BASE,
    "order_id": "T2",
    "order_hour": 2,
    "state": "Bihar",
    "city_tier": 3,
    "category": "fashion",
    "payment_mode": "COD",
    "order_value": 2999.0,
    "discount_pct": 50.0,
    "is_first_order": True,
    "customer_prior_orders": 0,
    "address_text": "Station Road, 800001",
    "address_len": 20,
    "address_has_landmark": False,
}


def test_pincode_history_is_past_only():
    df = pd.DataFrame(
        {"pincode": ["A", "A", "A", "B"], "is_rto": [True, False, True, True]}
    )
    out = add_pincode_history(df, global_rate=0.2)
    assert out["pincode_hist_orders"].tolist() == [0, 1, 2, 0]
    # first order of each pincode sees only the prior (global rate)
    assert out.loc[0, "pincode_hist_rto_rate"] == pytest.approx(0.2)


def test_economics_sign():
    assert ECON.net_value(1, 1) > 0      # flagging a real RTO pays off
    assert ECON.net_value(0, 1) < 0      # flagging a good order costs money
    assert ECON.net_value(1, 0) == 0     # not flagging = no change vs baseline


@needs_model
def test_prediction_contract():
    from ml.predict import get_predictor

    pred = get_predictor().predict(BASE).to_dict()
    assert 0 <= pred["rto_probability"] <= 1
    assert pred["risk_band"] in {"LOW", "MEDIUM", "HIGH"}
    assert len(pred["reasons"]) <= 3


@needs_model
def test_risky_order_scores_higher():
    from ml.predict import get_predictor

    p = get_predictor()
    safe, risky = p.predict(BASE), p.predict(RISKY)
    assert risky.rto_probability > safe.rto_probability * 3
    assert safe.risk_band == "LOW"
    assert risky.risk_band in {"MEDIUM", "HIGH"}


@needs_model
def test_unseen_category_does_not_crash():
    from ml.predict import get_predictor

    pred = get_predictor().predict({**BASE, "courier": "NewCourierCo", "pincode": "000000"})
    assert 0 <= pred.rto_probability <= 1
