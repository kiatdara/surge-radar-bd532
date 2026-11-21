"""Data loading, cleaning, feature engineering and labelling.

One row = one product on one day t. Every feature uses only data observed on or
before day t (plus the shop calendar, which is known in advance). The label looks
at days t+1 ... t+7, so it is never used as an input.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

# Columns of the daily panel after cleaning
CUMULATIVE = ["total_ifl_cnt", "total_video_cnt", "total_live_cnt", "total_sale_cnt"]

NUMERIC_FEATURES = [
    # sales level and momentum (from daily sales)
    "log_sales_mean_7d", "log_sales_mean_28d", "momentum_7_28", "momentum_3_7",
    "sales_cv_7d", "sales_slope_14d", "days_since_28d_peak",
    "week_over_week", "last_day_vs_7d", "mean7_vs_max28",
    # price
    "price", "price_change_7d",
    # creator / content activity (growth of cumulative counters)
    "log_ifl_total", "new_ifl_7d_rate", "new_video_7d_rate", "new_live_7d_rate",
    "videos_per_creator",
    # static product attributes
    "commission_rate", "rating", "log_reviews", "discount_pct", "shop_type", "product_age_days",
    # shop calendar (known in advance)
    "double_day_in_horizon", "days_to_double_day", "payday_in_horizon", "dow",
]
CATEGORICAL_FEATURES = ["category"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
LABEL = "surge"


# --------------------------------------------------------------------------- #
# Loading and cleaning
# --------------------------------------------------------------------------- #
def load_raw(trend_path=C.TREND_FILE, product_path=C.PRODUCT_FILE):
    trend = pd.read_csv(trend_path, dtype={"product_id": str, "_product_id": str})
    products = pd.read_csv(product_path, dtype={"product_id": str, "seller_id": str,
                                                "category_id": str})
    return trend, products


def clean_trend(trend: pd.DataFrame) -> pd.DataFrame:
    """Daily panel per product: one row per calendar day, gaps kept as NaN for daily
    sales and linearly interpolated for cumulative counters (they only go up)."""
    df = trend.copy()
    df["product_id"] = df["_product_id"].fillna(df["product_id"]).astype(str)
    df["dt"] = pd.to_datetime(df["dt"])
    df = (df.sort_values(["product_id", "dt"])
            .drop_duplicates(["product_id", "dt"], keep="last"))
    df.loc[df["total_sale_1d_cnt"] < 0, "total_sale_1d_cnt"] = np.nan
    # A day with 0 sales for a product that normally sells hundreds a day is a crawl gap,
    # not a real zero (seen in EDA as sudden drops to 0 and straight back up).
    med = df.groupby("_product_id" if "_product_id" in df else "product_id")[
        "total_sale_1d_cnt"].transform("median")
    df.loc[(df["total_sale_1d_cnt"] == 0) & (med > 50), "total_sale_1d_cnt"] = np.nan
    out = []
    for pid, g in df.groupby("product_id", sort=False):
        g = g.set_index("dt").asfreq("D")
        g["product_id"] = pid
        g["observed"] = g["total_sale_1d_cnt"].notna().astype(int)
        g[CUMULATIVE] = g[CUMULATIVE].interpolate(limit_area="inside")
        g["spu_avg_price"] = g["spu_avg_price"].ffill()
        out.append(g.reset_index())
    keep = ["dt", "product_id", "observed", "spu_avg_price", "total_sale_1d_cnt",
            "total_sale_gmv_1d_amt"] + CUMULATIVE
    return pd.concat(out, ignore_index=True)[keep]


def clean_products(products: pd.DataFrame) -> pd.DataFrame:
    p = products.drop_duplicates("product_id").copy()
    for col in ["product_name", "seller_id", "discount", "review_count"]:
        if col not in p:
            p[col] = np.nan
    p["discount_pct"] = (p["discount"].astype(str).str.rstrip("%")
                         .replace({"": "0", "nan": "0", "None": "0"}).astype(float))
    p["category"] = np.where(p["category_id"].isin(C.TOP_CATEGORIES), p["category_id"], "other")
    p["first_crawl_dt"] = pd.to_datetime(p["first_crawl_dt"].astype(str), format="%Y%m%d",
                                         errors="coerce")
    p["log_reviews"] = np.log1p(p["review_count"].fillna(0))
    p = p.rename(columns={"product_commission_rate": "commission_rate", "product_rating": "rating"})
    return p[["product_id", "product_name", "category", "category_id", "seller_id",
              "commission_rate", "rating", "log_reviews", "discount_pct", "shop_type",
              "first_crawl_dt"]]


# --------------------------------------------------------------------------- #
# Calendar helpers (Thai TikTok Shop campaigns: double days 9.9, 10.10 ... and payday)
# --------------------------------------------------------------------------- #
def _is_double_day(d: pd.Timestamp) -> bool:
    return d.day == d.month


def _is_payday(d: pd.Timestamp) -> bool:
    return d.day >= 25 or d.day == 1


def calendar_features(dates: pd.Series, horizon: int = C.HORIZON_DAYS) -> pd.DataFrame:
    uniq = pd.Series(pd.to_datetime(dates.unique()))
    rows = []
    for d in uniq:
        ahead = [d + pd.Timedelta(days=k) for k in range(1, horizon + 1)]
        nxt = next(k for k in range(0, 400) if _is_double_day(d + pd.Timedelta(days=k)))
        rows.append({"dt": d,
                     "double_day_in_horizon": int(any(_is_double_day(a) for a in ahead)),
                     "days_to_double_day": nxt,
                     "payday_in_horizon": int(any(_is_payday(a) for a in ahead)),
                     "dow": d.dayofweek})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
def _slope(y: np.ndarray) -> float:
    m = ~np.isnan(y)
    if m.sum() < 5:
        return np.nan
    x = np.arange(len(y))[m]
    return float(np.polyfit(x, y[m], 1)[0])


def build_product_day_features(panel: pd.DataFrame) -> pd.DataFrame:
    """Rolling features per product (only past data) + the forward-looking label."""
    frames = []
    for pid, g in panel.groupby("product_id", sort=False):
        g = g.sort_values("dt").reset_index(drop=True)
        s = g["total_sale_1d_cnt"]
        f = pd.DataFrame({"dt": g["dt"], "product_id": pid})
        m7 = s.rolling(7, min_periods=C.MIN_OBS).mean()
        m28 = s.rolling(28, min_periods=20).mean()
        m3 = s.rolling(3, min_periods=2).mean()
        f["sales_mean_7d"] = m7
        f["log_sales_mean_7d"] = np.log1p(m7)
        f["log_sales_mean_28d"] = np.log1p(m28)
        f["momentum_7_28"] = (m7 + 1) / (m28 + 1)
        f["momentum_3_7"] = (m3 + 1) / (m7 + 1)
        f["sales_cv_7d"] = s.rolling(7, min_periods=C.MIN_OBS).std() / (m7 + 1)
        f["sales_slope_14d"] = np.log1p(s).rolling(14, min_periods=7).apply(_slope, raw=True)
        f["days_since_28d_peak"] = s.rolling(28, min_periods=20).apply(
            lambda a: len(a) - 1 - np.nanargmax(a) if np.isfinite(a).any() else np.nan, raw=True)
        f["week_over_week"] = (m7 + 1) / (m7.shift(7) + 1)
        f["last_day_vs_7d"] = (s.ffill() + 1) / (m7 + 1)
        f["mean7_vs_max28"] = (m7 + 1) / (s.rolling(28, min_periods=10).max() + 1)
        f["price"] = g["spu_avg_price"]
        f["price_change_7d"] = g["spu_avg_price"].pct_change(7, fill_method=None)
        ifl, vid, live = g["total_ifl_cnt"], g["total_video_cnt"], g["total_live_cnt"]
        f["log_ifl_total"] = np.log1p(ifl)
        f["new_ifl_7d_rate"] = ifl.diff(7) / (ifl.shift(7) + 1)
        f["new_video_7d_rate"] = vid.diff(7) / (vid.shift(7) + 1)
        f["new_live_7d_rate"] = live.diff(7) / (live.shift(7) + 1)
        f["videos_per_creator"] = vid / (ifl + 1)
        # ---- label: future 7 days vs last 7 days (future values only used here) ----
        fut = s[::-1].rolling(C.HORIZON_DAYS, min_periods=C.MIN_OBS).mean()[::-1].shift(-1)
        f["future_sales_mean_7d"] = fut
        f["future_ratio"] = (fut + 1) / (m7 + 1)
        frames.append(f)
    feat = pd.concat(frames, ignore_index=True)
    feat[LABEL] = np.where(feat["future_ratio"].isna(), np.nan,
                           (feat["future_ratio"] >= C.SURGE_RATIO).astype(float))
    return feat


def build_dataset(trend: pd.DataFrame | None = None, products: pd.DataFrame | None = None,
                  drop_unlabelled: bool = True) -> pd.DataFrame:
    if trend is None or products is None:
        trend, products = load_raw()
    panel = clean_trend(trend)
    feat = build_product_day_features(panel)
    prod = clean_products(products)
    df = feat.merge(prod, on="product_id", how="left").merge(calendar_features(feat["dt"]),
                                                             on="dt", how="left")
    df["product_age_days"] = (df["dt"] - df["first_crawl_dt"]).dt.days.clip(lower=0)
    df["category"] = df["category"].fillna("other")
    needed = ["log_sales_mean_7d", "log_sales_mean_28d"]
    df = df.dropna(subset=needed)
    if drop_unlabelled:
        df = df.dropna(subset=[LABEL])
        df[LABEL] = df[LABEL].astype(int)
    return df.reset_index(drop=True)


def time_split(df: pd.DataFrame):
    """Train / validation / test by date, with a 7-day gap so labels don't overlap."""
    tr = df[df["dt"] <= C.TRAIN_END]
    va = df[(df["dt"] >= C.VAL_START) & (df["dt"] <= C.VAL_END)]
    te = df[(df["dt"] >= C.TEST_START) & (df["dt"] <= C.TEST_END)]
    return tr, va, te


def features_for_single_product(history: pd.DataFrame, product: dict) -> pd.DataFrame:
    """Serving path: same code as training, for one product sent to the API.

    history: daily rows with dt, total_sale_1d_cnt, spu_avg_price, total_ifl_cnt,
             total_video_cnt, total_live_cnt (total_sale_cnt optional).
    product: static attributes (category_id, commission rate, rating, ...).
    Returns one row (the last day in the history) with all FEATURES.
    """
    pid = str(product.get("product_id") or "api-product")
    h = history.copy()
    h["product_id"] = h["_product_id"] = pid
    for col in ["total_sale_cnt", "total_sale_gmv_1d_amt"]:
        if col not in h:
            h[col] = np.nan
    panel = clean_trend(h)
    feat = build_product_day_features(panel)
    prod = clean_products(pd.DataFrame([{**product, "product_id": pid}]))
    row = feat.tail(1).merge(prod, on="product_id", how="left")
    row = row.merge(calendar_features(row["dt"]), on="dt", how="left")
    row["product_age_days"] = (row["dt"] - row["first_crawl_dt"]).dt.days.clip(lower=0)
    row["category"] = row["category"].fillna("other")
    return row.reset_index(drop=True)
