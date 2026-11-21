"""Generate SYNTHETIC sample data with the same columns as the EchoTik exports.

The real EchoTik data used in the report comes from a paid API and is not published in
this repository. This script writes a made-up dataset with the same schema and similar
behaviour (campaign spikes on double days, creator growth, crawl gaps, duplicate days)
so that every script, test and the API can run end to end without the real data.

Run:  python -m src.make_sample_data      -> data/sample/product_trend.csv, product_list.csv
Numbers produced from this sample are NOT the results in the report.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

CATEGORIES = ["601450"] * 46 + ["600001"] * 13 + ["700437"] * 9 + ["601152"] * 9 + \
             ["601739"] * 8 + ["700645"] * 5 + ["600024"] * 10
NAMES = {"601450": "Sample serum", "600001": "Sample home item", "700437": "Sample snack",
         "601152": "Sample dress", "601739": "Sample phone case", "700645": "Sample vitamin",
         "600024": "Sample gadget"}


def make(n_products: int = 100, start: str = "2026-03-31", days: int = 170, seed: int = 7):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start, periods=days)
    trend_rows, prod_rows = [], []
    for i in range(n_products):
        pid = str(1_730_000_000_000_000_000 + i * 7_919)
        cat = CATEGORIES[i % len(CATEGORIES)]
        mall = int(rng.random() < 0.43)
        base = rng.lognormal(7.4, 0.9)                        # typical units per day
        price = round(float(rng.lognormal(1.3, 0.7)), 2)
        commission = float(rng.choice([0, .03, .05, .1, .15], p=[.45, .1, .15, .2, .1]))
        level = np.log(base)
        ifl, vid, live, total = rng.integers(20, 300), rng.integers(50, 900), rng.integers(5, 80), 0
        first_seen = dates[0] - pd.Timedelta(days=int(rng.integers(0, 400)))
        for d in dates:
            level += rng.normal(0, 0.06) - 0.02 * (level - np.log(base))   # slow random walk
            boost = 1.0
            if d.day == d.month:                                            # double-day campaign
                boost = rng.uniform(1.8, 3.2) * (1.3 if mall else 1.0)
            elif 0 < (pd.Timestamp(d.year, d.month, d.month) - d).days <= 3:
                boost = rng.uniform(1.0, 1.4)                               # campaign run-up
            if d.day >= 25:
                boost *= 1.1                                                 # payday
            sales = max(0, int(np.exp(level) * boost * rng.lognormal(0, 0.25)))
            new_ifl = rng.poisson(1 + sales / 1500)
            ifl += new_ifl; vid += rng.poisson(2 + new_ifl * 2.5); live += rng.poisson(0.3 + sales / 6000)
            total += sales
            p = round(price * rng.uniform(0.97, 1.03), 2)
            if rng.random() < 0.12:                                          # crawl gap
                continue
            row = {"dt": d.strftime("%Y-%m-%d"), "product_id": pid, "spu_avg_price": p,
                   "total_ifl_cnt": int(ifl), "total_live_cnt": int(live),
                   "total_sale_1d_cnt": sales, "total_sale_cnt": int(total),
                   "total_sale_gmv_1d_amt": round(sales * p, 2),
                   "total_sale_gmv_amt": round(total * price, 2),
                   "total_video_cnt": int(vid), "_product_id": pid}
            trend_rows.append(row)
            if rng.random() < 0.02:                                          # duplicate re-crawl
                trend_rows.append(dict(row))
        prod_rows.append({
            "product_id": pid, "product_name": f"{NAMES[cat]} #{i + 1:03d} (synthetic)",
            "category_id": cat, "category_l2_id": "", "seller_id": str(9_000_000 + i),
            "spu_avg_price": price, "min_price": round(price * .8, 2),
            "max_price": round(price * 1.5, 2), "discount": f"{int(rng.integers(0, 80))}%",
            "free_shipping": int(rng.random() < .3), "shop_type": mall, "is_s_shop": 0,
            "product_commission_rate": commission,
            "product_rating": round(float(np.clip(rng.normal(4.7, .2), 3.5, 5)), 1),
            "review_count": int(rng.lognormal(8, 1.5)),
            "first_crawl_dt": int(first_seen.strftime("%Y%m%d")),
            "last_crawl_dt": int(dates[-1].strftime("%Y%m%d")),
            "total_ifl_cnt": int(ifl), "total_video_cnt": int(vid), "total_live_cnt": int(live),
            "total_sale_cnt": int(total), "total_sale_30d_cnt": "", "total_sale_7d_cnt": "",
            "total_views_30d_cnt": "", "region": "TH"})
    out = C.ROOT / "data" / "sample"
    out.mkdir(parents=True, exist_ok=True)
    trend = pd.DataFrame(trend_rows).sample(frac=1, random_state=seed)   # unsorted, like the API
    trend.to_csv(out / "product_trend.csv", index=False)
    pd.DataFrame(prod_rows).to_csv(out / "product_list.csv", index=False)
    print(f"wrote {len(trend)} trend rows and {len(prod_rows)} products to {out}")


if __name__ == "__main__":
    make()
