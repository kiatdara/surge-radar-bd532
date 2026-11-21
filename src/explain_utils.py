"""3.5 Prediction reasoning: turn SHAP values for one product-day into plain sentences.
Used by src/explain.py (report) and by the API (/api/v1/predict returns `reasons`)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import shap

# feature -> (English description, Thai description, how to format the raw value)
DESCRIBE = {
    "log_sales_mean_7d": ("sales level last 7 days", "ยอดขายเฉลี่ย 7 วันล่าสุด",
                          lambda v: f"{np.expm1(v):,.0f} units/day"),
    "log_sales_mean_28d": ("sales level last 28 days", "ยอดขายเฉลี่ย 28 วัน",
                           lambda v: f"{np.expm1(v):,.0f} units/day"),
    "momentum_7_28": ("last 7 days vs last 28 days", "ยอด 7 วันเทียบ 28 วัน", lambda v: f"{v:.2f}×"),
    "momentum_3_7": ("last 3 days vs last 7 days", "ยอด 3 วันเทียบ 7 วัน", lambda v: f"{v:.2f}×"),
    "sales_cv_7d": ("day-to-day volatility", "ความผันผวนรายวัน", lambda v: f"{v:.2f}"),
    "sales_slope_14d": ("14-day sales trend", "แนวโน้มยอดขาย 14 วัน", lambda v: f"{v:+.3f}/day (log)"),
    "days_since_28d_peak": ("days since 28-day peak", "จำนวนวันนับจากยอดสูงสุดใน 28 วัน",
                            lambda v: f"{v:.0f} days"),
    "week_over_week": ("this week vs last week", "สัปดาห์นี้เทียบสัปดาห์ก่อน", lambda v: f"{v:.2f}×"),
    "last_day_vs_7d": ("yesterday vs 7-day average", "ยอดเมื่อวานเทียบค่าเฉลี่ย 7 วัน",
                       lambda v: f"{v:.2f}×"),
    "mean7_vs_max28": ("7-day average vs 28-day best day", "ค่าเฉลี่ย 7 วันเทียบวันที่ขายดีสุด",
                       lambda v: f"{v:.2f}"),
    "price": ("price", "ราคา", lambda v: f"{v:.2f}"),
    "price_change_7d": ("price change over 7 days", "ราคาเปลี่ยนใน 7 วัน", lambda v: f"{v:+.1%}"),
    "log_ifl_total": ("creators promoting it", "จำนวนครีเอเตอร์ที่โปรโมท",
                      lambda v: f"{np.expm1(v):,.0f}"),
    "new_ifl_7d_rate": ("new creators in 7 days", "ครีเอเตอร์ใหม่ใน 7 วัน", lambda v: f"{v:+.1%}"),
    "new_video_7d_rate": ("new videos in 7 days", "คลิปใหม่ใน 7 วัน", lambda v: f"{v:+.1%}"),
    "new_live_7d_rate": ("new lives in 7 days", "ไลฟ์ใหม่ใน 7 วัน", lambda v: f"{v:+.1%}"),
    "videos_per_creator": ("videos per creator", "คลิปต่อครีเอเตอร์", lambda v: f"{v:.1f}"),
    "commission_rate": ("commission rate", "ค่าคอมมิชชั่น", lambda v: f"{v:.0%}"),
    "rating": ("product rating", "เรตติ้งสินค้า", lambda v: f"{v:.1f}"),
    "log_reviews": ("number of reviews", "จำนวนรีวิว", lambda v: f"{np.expm1(v):,.0f}"),
    "discount_pct": ("discount", "ส่วนลด", lambda v: f"{v:.0f}%"),
    "shop_type": ("shop type", "ประเภทร้าน", lambda v: "Mall" if v >= .5 else "regular shop"),
    "product_age_days": ("days on the platform", "อายุสินค้าบนแพลตฟอร์ม", lambda v: f"{v:.0f} days"),
    "double_day_in_horizon": ("double-day campaign in next 7 days", "มีแคมเปญวันเลขเบิ้ลใน 7 วันข้างหน้า",
                              lambda v: "yes" if v >= .5 else "no"),
    "days_to_double_day": ("days to next double-day", "จำนวนวันถึงแคมเปญวันเลขเบิ้ล",
                           lambda v: f"{v:.0f} days"),
    "payday_in_horizon": ("payday in next 7 days", "มีวันเงินเดือนออกใน 7 วันข้างหน้า",
                          lambda v: "yes" if v >= .5 else "no"),
    "dow": ("day of week", "วันในสัปดาห์",
            lambda v: ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][int(v) % 7]),
}


class Reasoner:
    """Wraps a fitted pipeline (prep + tree model) with a SHAP TreeExplainer."""

    def __init__(self, pipeline, numeric: list[str]):
        self.prep = pipeline.named_steps["prep"]
        self.clf = pipeline.named_steps["clf"]
        self.names = list(self.prep.get_feature_names_out())
        self.numeric = numeric
        self.explainer = shap.TreeExplainer(self.clf)

    def shap_values(self, X: pd.DataFrame):
        Xt = pd.DataFrame(self.prep.transform(X), columns=self.names, index=X.index)
        sv = self.explainer.shap_values(Xt, check_additivity=False)
        sv = np.asarray(sv)
        if sv.ndim == 3:                       # (rows, features, classes) -> class 1
            sv = sv[..., 1]
        base = np.ravel(self.explainer.expected_value)[-1]
        return Xt, sv, float(base)

    def explain_row(self, X: pd.DataFrame, top_k: int = 3) -> list[dict]:
        """Top-k reasons for a single row. Category one-hot columns are merged."""
        Xt, sv, _ = self.shap_values(X.iloc[[0]])
        contrib = {}
        for n, v in zip(self.names, sv[0]):
            key = "category" if n.startswith("category_") else n
            contrib[key] = contrib.get(key, 0.0) + float(v)
        out = []
        for feat, val in sorted(contrib.items(), key=lambda t: -abs(t[1]))[:top_k]:
            raw = X.iloc[0][feat]
            if feat == "category":
                en, th, shown = "product category", "หมวดสินค้า", str(raw)
            else:
                en, th, fmt = DESCRIBE.get(feat, (feat, feat, lambda v: f"{v}"))
                shown = fmt(float(raw)) if pd.notna(raw) else "missing"
            direction = "raises" if val > 0 else "lowers"
            out.append({
                "feature": feat, "value": shown, "shap": round(val, 4),
                "direction": "up" if val > 0 else "down",
                "text": f"{en.capitalize()} = {shown} {direction} the surge chance "
                        f"by {abs(val) * 100:.1f} points",
                "text_th": f"{th} = {shown} ทำให้โอกาสยอดพุ่ง{'เพิ่มขึ้น' if val > 0 else 'ลดลง'} "
                           f"{abs(val) * 100:.1f} จุด",
            })
        return out
