"""3.4 Model explainability (global) + 3.5 prediction reasoning (local, 3 test cases).

Run:  python -m src.explain     (after python -m src.train)
"""
import json
import warnings

warnings.filterwarnings("ignore")
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance

from . import config as C
from .explain_utils import Reasoner
from .features import FEATURES, LABEL


def main():
    bundle = joblib.load(C.MODEL_DIR / "surge_model.joblib")
    pipe, thr = bundle["pipeline"], bundle["threshold"]
    te = pd.read_csv(C.PROCESSED_DIR / "test_predictions.csv", parse_dates=["dt"],
                     dtype={"product_id": str, "category": str})
    r = Reasoner(pipe, bundle["numeric"])
    Xt, sv, base = r.shap_values(te[FEATURES])

    # ---- Global: SHAP beeswarm + mean |SHAP| bar ----
    shap.summary_plot(sv, Xt, show=False, max_display=15, plot_size=(9, 6))
    plt.title("SHAP summary – what pushes a product toward a sales surge")
    plt.tight_layout(); plt.savefig(C.FIG_DIR / "shap_summary.png", dpi=130); plt.close()
    imp = pd.Series(np.abs(sv).mean(0), index=r.names).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    imp.head(15)[::-1].plot.barh(ax=ax, color="#2563eb")
    ax.set_xlabel("mean |SHAP value| (probability points)")
    ax.set_title("Global feature importance (SHAP)")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "shap_bar.png", dpi=130); plt.close(fig)

    # ---- Permutation importance (model-agnostic check) on the test period ----
    pi = permutation_importance(pipe, te[FEATURES], te[LABEL], scoring="average_precision",
                                n_repeats=5, random_state=C.RANDOM_STATE, n_jobs=-1)
    perm = pd.Series(pi.importances_mean, index=FEATURES).sort_values(ascending=False)

    # ---- Dependence plots for the top-2 numeric features ----
    top2 = [f for f in imp.index if not f.startswith("category_")][:2]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, f in zip(axes, top2):
        j = r.names.index(f)
        ax.scatter(te[f], sv[:, j], s=6, alpha=.4, c=te[LABEL], cmap="coolwarm")
        ax.axhline(0, color="gray", lw=.8); ax.set_xlabel(f); ax.set_ylabel("SHAP value")
        if te[f].max() > 20 * max(te[f].median(), 1e-9):
            ax.set_xscale("symlog")
    fig.suptitle("Decision behaviour of the top-2 features (red = real surge)")
    fig.tight_layout(); fig.savefig(C.FIG_DIR / "shap_dependence.png", dpi=130); plt.close(fig)

    # ---- Local reasoning: one true positive, one true negative, one false positive ----
    te = te.assign(proba=pipe.predict_proba(te[FEATURES])[:, 1])
    te["pred"] = (te["proba"] >= thr).astype(int)
    picks = {
        "A_true_positive": te[(te.pred == 1) & (te[LABEL] == 1)].sort_values("proba").iloc[-1],
        "B_true_negative": te[(te.pred == 0) & (te[LABEL] == 0)].sort_values("proba").iloc[0],
        "C_false_positive": te[(te.pred == 1) & (te[LABEL] == 0)].sort_values("proba").iloc[-1],
    }
    cases = {}
    for tag, row in picks.items():
        X = row[FEATURES].to_frame().T.astype({f: float for f in bundle["numeric"]})
        reasons = r.explain_row(X, top_k=4)
        cases[tag] = {
            "product_id": row["product_id"], "product_name": str(row["product_name"])[:70],
            "date": str(row["dt"].date()), "probability": round(float(row["proba"]), 3),
            "predicted": "surge" if row["pred"] else "no surge",
            "actual": "surge" if row[LABEL] else "no surge",
            "actual_ratio_next7_vs_last7": round(float(row["future_ratio"]), 2),
            "reasons": reasons,
        }
        i = te.index.get_loc(row.name)
        exp = shap.Explanation(values=sv[i], base_values=base, data=Xt.iloc[i].values,
                               feature_names=r.names)
        shap.plots.waterfall(exp, max_display=8, show=False)
        plt.title(f"Case {tag[0]}: p(surge)={row['proba']:.2f}, actual={cases[tag]['actual']}")
        plt.tight_layout(); plt.savefig(C.FIG_DIR / f"case_{tag}.png", dpi=120, bbox_inches="tight")
        plt.close()
        print(f"\n[{tag}] {cases[tag]['product_name']}  ({cases[tag]['date']})")
        print(f"  p(surge)={cases[tag]['probability']}  predicted={cases[tag]['predicted']}  "
              f"actual={cases[tag]['actual']} (next7/last7={cases[tag]['actual_ratio_next7_vs_last7']})")
        for x in reasons:
            print("   -", x["text"]); print("     ", x["text_th"])

    out = {"shap_mean_abs_top15": imp.head(15).round(4).to_dict(),
           "permutation_importance_top10": perm.head(10).round(4).to_dict(),
           "base_value": base, "cases": cases}
    json.dump(out, open(C.REPORT_DIR / "explainability.json", "w"), indent=2, ensure_ascii=False)
    print("\nTop SHAP:", imp.head(8).round(3).to_dict())
    print("Top permutation:", perm.head(8).round(3).to_dict())


if __name__ == "__main__":
    main()
