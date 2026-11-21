"""3.2 Model fairness analysis.

Who could be treated unfairly? The model ranks products for affiliate creators, so a
systematic bias against one *kind of seller* means those sellers get fewer creators and
less exposure. We check two seller-side groups:
  * shop type: regular shop (0) vs TikTok Shop Mall (1)
  * price tier: low / mid / high (tertiles of product price)
Metrics per group: selection rate (demographic parity), TPR (equal opportunity),
FPR and precision. Because the product shows creators a daily top-10 list, we also measure *exposure*: the
share of each shop type in that list versus its share of real surges. Mitigation demo:
exposure-aware re-ranking with a minimum number of slots per shop type.

Run:  python -m src.fairness   (after python -m src.train)
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from . import config as C
from .features import LABEL


def group_metrics(d: pd.DataFrame, col: str, pred_col="pred") -> pd.DataFrame:
    rows = {}
    for g, x in d.groupby(col):
        y, p = x[LABEL], x[pred_col]
        rows[g] = {
            "n": len(x), "base_rate": y.mean(), "selection_rate": p.mean(),
            "TPR": p[y == 1].mean() if (y == 1).any() else np.nan,
            "FPR": p[y == 0].mean() if (y == 0).any() else np.nan,
            "precision": y[p == 1].mean() if (p == 1).any() else np.nan,
            "pr_auc": average_precision_score(y, x["proba"]) if y.nunique() > 1 else np.nan,
        }
    return pd.DataFrame(rows).T


def gaps(m: pd.DataFrame) -> dict:
    return {"demographic_parity_diff": float(m["selection_rate"].max() - m["selection_rate"].min()),
            "disparate_impact_ratio": float(m["selection_rate"].min() / m["selection_rate"].max()),
            "equal_opportunity_diff": float(m["TPR"].max() - m["TPR"].min()),
            "fpr_diff": float(m["FPR"].max() - m["FPR"].min())}


def main():
    te = pd.read_csv(C.PROCESSED_DIR / "test_predictions.csv", parse_dates=["dt"],
                     dtype={"product_id": str})
    te["shop"] = te["shop_type"].map({0: "regular", 1: "mall"})
    te["price_tier"] = pd.qcut(te["price"], 3, labels=["low", "mid", "high"]).astype(str)
    report = {}
    for col in ["shop", "price_tier"]:
        m = group_metrics(te, col)
        report[col] = {"metrics": m.round(4).to_dict(orient="index"), "gaps": gaps(m)}
        print(f"\n== {col} ==\n{m.round(3)}\n{ {k: round(v, 3) for k, v in gaps(m).items()} }")

    # Exposure in the product's real output: the daily top-10 list shown to creators
    top = te.sort_values("proba", ascending=False).groupby("dt").head(10)
    exposure = {"mall_share_population": float((te["shop"] == "mall").mean()),
                "mall_share_actual_surges": float((te.loc[te[LABEL] == 1, "shop"] == "mall").mean()),
                "mall_share_top10": float((top["shop"] == "mall").mean()),
                "precision_at_10_mall": float(top.loc[top["shop"] == "mall", LABEL].mean()),
                "precision_at_10_regular": float(top.loc[top["shop"] == "regular", LABEL].mean())}
    report["top10_exposure"] = exposure

    # Mitigation: exposure-aware re-ranking. Each day, reserve at least
    # floor(10 x group share of products) slots for each shop type, filled by the
    # highest-scoring products of that type; the rest go to the best remaining scores.
    def fair_top10(day):
        quota = {g: int(np.floor(10 * (te["shop"] == g).mean())) for g in ["mall", "regular"]}
        picked = pd.concat([day[day["shop"] == g].nlargest(q, "proba") for g, q in quota.items()])
        rest = day.drop(picked.index).nlargest(10 - len(picked), "proba")
        return pd.concat([picked, rest])
    fair = pd.concat(fair_top10(d) for _, d in te.groupby("dt"))
    report["mitigation_rerank"] = {
        "mall_share_top10_before": exposure["mall_share_top10"],
        "mall_share_top10_after": float((fair["shop"] == "mall").mean()),
        "precision_at_10_before": float(top.groupby("dt")[LABEL].mean().mean()),
        "precision_at_10_after": float(fair.groupby("dt")[LABEL].mean().mean())}
    print("Re-ranking mitigation:", {k: round(v, 3) for k, v in report["mitigation_rerank"].items()})
    print("\nTop-10 exposure:", {k: round(v, 3) for k, v in exposure.items()})
    json.dump(report, open(C.REPORT_DIR / "fairness.json", "w"), indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for ax, col in zip(axes, ["shop", "price_tier"]):
        m = group_metrics(te, col)[["base_rate", "selection_rate", "TPR", "FPR"]]
        m.plot.bar(ax=ax, color=["#d1d5db", "#2563eb", "#16a34a", "#f97316"], width=.8)
        ax.set_ylim(0, 1); ax.set_title(f"Group metrics by {col}"); ax.set_xlabel("")
        ax.legend(frameon=False, fontsize=8, ncol=2); ax.tick_params(axis="x", rotation=0)
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "fairness.png", dpi=130); plt.close(fig)


if __name__ == "__main__":
    main()
