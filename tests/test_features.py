"""Unit tests for data cleaning, features, labels and evaluation helpers."""
import numpy as np
import pandas as pd
import pytest

from src import config as C
from src.features import (FEATURES, LABEL, build_dataset, build_product_day_features,
                          calendar_features, clean_trend, features_for_single_product,
                          load_raw, time_split)
from src.modeling import best_threshold, precision_at_k_per_day


def make_history(sales, start="2026-06-01", pid="p1"):
    n = len(sales)
    return pd.DataFrame({
        "dt": pd.date_range(start, periods=n).strftime("%Y-%m-%d"),
        "product_id": pid, "_product_id": pid,
        "spu_avg_price": 5.0, "total_sale_1d_cnt": sales,
        "total_sale_cnt": np.cumsum(sales), "total_sale_gmv_1d_amt": np.array(sales) * 5.0,
        "total_ifl_cnt": np.arange(n) + 100, "total_video_cnt": np.arange(n) * 2 + 300,
        "total_live_cnt": np.arange(n) + 10,
    })


def test_label_is_surge_when_next_week_is_20pct_higher():
    sales = [1000] * 35 + [1500] * 7           # big jump in the last week
    feat = build_product_day_features(clean_trend(make_history(sales)))
    day = feat.iloc[34]                        # last day before the jump
    assert day[LABEL] == 1 and day["future_ratio"] == pytest.approx(1501 / 1001, rel=1e-3)
    flat = build_product_day_features(clean_trend(make_history([1000] * 42)))
    assert flat[LABEL].dropna().eq(0).all()


def test_features_do_not_use_future_data():
    a = make_history([1000] * 42)
    b = a.copy(); b.loc[35:, "total_sale_1d_cnt"] = 9999     # change only the future
    fa = build_product_day_features(clean_trend(a)).iloc[34]
    fb = build_product_day_features(clean_trend(b)).iloc[34]
    past_cols = [c for c in fa.index if c not in ("future_sales_mean_7d", "future_ratio", LABEL)]
    pd.testing.assert_series_equal(fa[past_cols], fb[past_cols])
    assert fa[LABEL] != fb[LABEL]                            # ...but the label does change


def test_suspicious_zero_is_treated_as_missing():
    sales = [1000] * 30; sales[10] = 0
    panel = clean_trend(make_history(sales))
    assert np.isnan(panel.loc[10, "total_sale_1d_cnt"])


def test_missing_days_are_reindexed():
    h = make_history([1000] * 30).drop(index=[5, 6, 7])
    panel = clean_trend(h)
    assert len(panel) == 30 and panel["observed"].sum() == 27
    assert panel["total_ifl_cnt"].is_monotonic_increasing      # interpolated, no gaps


def test_calendar_double_day():
    cal = calendar_features(pd.Series(pd.to_datetime(["2026-09-05", "2026-09-20"])))
    assert cal.loc[0, "double_day_in_horizon"] == 1 and cal.loc[0, "days_to_double_day"] == 4
    assert cal.loc[1, "double_day_in_horizon"] == 0 and cal.loc[1, "days_to_double_day"] == 20


def test_time_split_has_gap_and_no_overlap():
    df = build_dataset()
    tr, va, te = time_split(df)
    assert tr["dt"].max() + pd.Timedelta(days=C.HORIZON_DAYS) <= va["dt"].min()
    assert va["dt"].max() + pd.Timedelta(days=C.HORIZON_DAYS) <= te["dt"].min()
    assert set(FEATURES).issubset(df.columns) and df[LABEL].isin([0, 1]).all()


def test_serving_features_match_training_features():
    trend, products = load_raw()
    full = build_dataset(trend, products, drop_unlabelled=False)
    pid = trend["_product_id"].iloc[0]
    hist = trend[trend["_product_id"] == pid].sort_values("dt").tail(45)
    prod = products[products["product_id"] == pid].iloc[0].to_dict()
    one = features_for_single_product(hist, prod)
    ref = full[full["product_id"] == pid].sort_values("dt").tail(1)
    np.testing.assert_allclose(one[FEATURES[:-1]].astype(float).values,
                               ref[FEATURES[:-1]].astype(float).values, rtol=1e-9)


def test_precision_at_k_and_threshold():
    df = pd.DataFrame({"dt": ["d1"] * 4 + ["d2"] * 4, "surge": [1, 1, 0, 0, 0, 1, 0, 0]})
    p = np.array([.9, .8, .1, .2, .9, .8, .1, .2])
    assert precision_at_k_per_day(df, p, k=2) == pytest.approx((1.0 + 0.5) / 2)
    assert 0.1 <= best_threshold(df["surge"], p) <= 0.9
