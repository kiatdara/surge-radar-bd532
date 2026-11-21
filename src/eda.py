"""3.1 Basic input data exploration -> reports/eda_summary.json + reports/figures/eda_*.png

Run:  python -m src.eda
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config as C
from .features import FEATURES, LABEL, build_dataset, clean_trend, load_raw, time_split

BLUE, ORANGE, GRAY = "#2563eb", "#f97316", "#9ca3af"


def main():
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)
    trend, products = load_raw()
    panel = clean_trend(trend)
    df = build_dataset(trend, products)
    tr, va, te = time_split(df)

    summary = {
        "raw_trend_rows": int(len(trend)),
        "raw_trend_columns": {c: str(t) for c, t in trend.dtypes.items()},
        "products_in_trend": int(trend["_product_id"].nunique()),
        "product_list_rows": int(len(products)),
        "date_min": str(pd.to_datetime(trend["dt"]).min().date()),
        "date_max": str(pd.to_datetime(trend["dt"]).max().date()),
        "calendar_days_after_reindex": int(len(panel)),
        "missing_day_share": round(1 - panel["observed"].mean(), 4),
        "zero_sales_day_share": round(float((trend["total_sale_1d_cnt"] == 0).mean()), 4),
        "daily_sales_stats": trend["total_sale_1d_cnt"].describe().round(1).to_dict(),
        "modelling_rows": int(len(df)),
        "surge_rate_overall": round(float(df[LABEL].mean()), 4),
        "split_rows": {"train": len(tr), "val": len(va), "test": len(te)},
        "split_surge_rate": {k: round(float(d[LABEL].mean()), 4)
                             for k, d in [("train", tr), ("val", va), ("test", te)]},
        "category_counts_products": products.loc[products["product_id"].isin(
            trend["_product_id"]), "category_id"].value_counts().head(8).to_dict(),
        "shop_type_counts_products": products.loc[products["product_id"].isin(
            trend["_product_id"]), "shop_type"].value_counts().to_dict(),
        "feature_stats": df[FEATURES[:-1]].describe().T[["mean", "std", "min", "max"]]
                            .round(3).to_dict(orient="index"),
        "missing_feature_share": df[FEATURES].isna().mean().round(4)[lambda s: s > 0].to_dict(),
    }
    print(json.dumps({k: v for k, v in summary.items() if k not in ("feature_stats",
                                                                     "raw_trend_columns")},
                     indent=2, default=str))
    json.dump(summary, open(C.REPORT_DIR / "eda_summary.json", "w"), indent=2, default=str)

    # Figure 1: daily sales of 4 example products
    fig, ax = plt.subplots(figsize=(10, 4))
    top = (panel.groupby("product_id")["total_sale_1d_cnt"].mean()
           .sort_values(ascending=False).index[[0, 10, 30, 60]])
    for pid in top:
        g = panel[panel["product_id"] == pid]
        ax.plot(g["dt"], g["total_sale_1d_cnt"].rolling(3, min_periods=1).mean(), lw=1.4,
                label=f"…{pid[-6:]}")
    for d in pd.date_range("2026-04-01", "2026-09-30"):
        if d.day == d.month:
            ax.axvline(d, color=GRAY, ls=":", lw=1)
    ax.set_yscale("log"); ax.set_ylim(bottom=300); ax.set_ylabel("units sold per day (3-day avg, log)")
    ax.set_title("Daily sales of four example products (dotted = double-day campaigns)")
    ax.legend(frameon=False, fontsize=8); fig.tight_layout()
    fig.savefig(C.FIG_DIR / "eda_sales_examples.png", dpi=130); plt.close(fig)

    # Figure 2: label balance by month
    fig, ax = plt.subplots(figsize=(6, 3.2))
    m = df.groupby(df["dt"].dt.to_period("M"))[LABEL].mean()
    ax.bar(m.index.astype(str), m.values, color=BLUE)
    ax.axhline(df[LABEL].mean(), color=ORANGE, ls="--", lw=1, label="overall")
    ax.set_ylabel("share of surge rows"); ax.set_title("Class balance: surge (1) share by month")
    ax.legend(frameon=False); fig.tight_layout()
    fig.savefig(C.FIG_DIR / "eda_label_by_month.png", dpi=130); plt.close(fig)

    # Figure 3: distribution of future/past ratio
    fig, ax = plt.subplots(figsize=(6, 3.2))
    r = np.log2(df["future_ratio"].clip(1 / 16, 16))
    ax.hist(r, bins=60, color=BLUE)
    ax.axvline(np.log2(C.SURGE_RATIO), color=ORANGE, lw=1.5, label="threshold 1.2×")
    ax.set_xlabel("log2(next-7-day avg / last-7-day avg)"); ax.set_ylabel("rows")
    ax.set_title("How much sales change over the next 7 days"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "eda_ratio_hist.png", dpi=130); plt.close(fig)

    # Figure 4: surge rate by feature deciles for key features
    keys = ["momentum_7_28", "new_video_7d_rate", "days_to_double_day", "commission_rate"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3))
    for ax, k in zip(axes, keys):
        q = pd.qcut(df[k].rank(method="first"), 5, labels=False)
        ax.bar(range(1, 6), df.groupby(q)[LABEL].mean().values, color=BLUE)
        ax.set_title(k, fontsize=9); ax.set_xlabel("quintile"); ax.set_ylim(0, .6)
    axes[0].set_ylabel("surge rate"); fig.suptitle("Surge rate by feature quintile", y=1.02)
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "eda_feature_quintiles.png", dpi=130,
                                    bbox_inches="tight"); plt.close(fig)

    # Figure 5: correlation heat-map of numeric features
    corr = df[FEATURES[:-1] + [LABEL]].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr))); ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr))); ax.set_yticklabels(corr.columns, fontsize=7)
    fig.colorbar(im, shrink=.7); ax.set_title("Feature correlation")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "eda_correlation.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
