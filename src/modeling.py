"""Model definitions, preprocessing and evaluation helpers shared by training and tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from . import config as C
from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

CALENDAR = ["double_day_in_horizon", "days_to_double_day", "payday_in_horizon", "dow"]


def make_preprocessor(numeric=NUMERIC_FEATURES, scale=True):
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), list(numeric)),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
         CATEGORICAL_FEATURES),
    ], verbose_feature_names_out=False)


def candidate_models(pos_weight: float):
    """name -> (estimator, numeric feature list, params to log, description)."""
    no_cal = [f for f in NUMERIC_FEATURES if f not in CALENDAR]
    return {
        "v1_logreg": (
            LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000),
            NUMERIC_FEATURES, {"C": 0.5, "class_weight": "balanced"},
            "Linear baseline, standardised features"),
        "v2_random_forest": (
            RandomForestClassifier(n_estimators=400, max_depth=8, min_samples_leaf=50,
                                   class_weight="balanced", n_jobs=-1,
                                   random_state=C.RANDOM_STATE),
            NUMERIC_FEATURES, {"n_estimators": 400, "max_depth": 8, "min_samples_leaf": 50},
            "Bagged trees"),
        "v3_xgboost": (
            XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.03, subsample=0.8,
                          colsample_bytree=0.8, min_child_weight=5, reg_lambda=2.0,
                          scale_pos_weight=pos_weight, eval_metric="aucpr",
                          random_state=C.RANDOM_STATE, n_jobs=4),
            NUMERIC_FEATURES, {"n_estimators": 400, "max_depth": 4, "learning_rate": 0.03,
                               "subsample": 0.8, "colsample_bytree": 0.8},
            "Gradient-boosted trees, all features"),
        "v4_lightgbm": (
            LGBMClassifier(n_estimators=500, num_leaves=15, learning_rate=0.03,
                           min_child_samples=30, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, reg_lambda=2.0, is_unbalance=True,
                           random_state=C.RANDOM_STATE, verbose=-1),
            NUMERIC_FEATURES, {"n_estimators": 500, "num_leaves": 15, "learning_rate": 0.03},
            "Leaf-wise gradient boosting"),
        "v5_xgboost_no_calendar": (
            XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.03, subsample=0.8,
                          colsample_bytree=0.8, min_child_weight=5, reg_lambda=2.0,
                          scale_pos_weight=pos_weight, eval_metric="aucpr",
                          random_state=C.RANDOM_STATE, n_jobs=4),
            no_cal, {"n_estimators": 400, "max_depth": 4, "learning_rate": 0.03,
                     "feature_set": "no_calendar"},
            "Ablation: same as v3 without shop-calendar features"),
    }


def build_pipeline(estimator, numeric):
    scale = isinstance(estimator, LogisticRegression)
    return Pipeline([("prep", make_preprocessor(numeric, scale=scale)), ("clf", estimator)])


def precision_at_k_per_day(df: pd.DataFrame, proba: np.ndarray, k: int = 10) -> float:
    """Business metric: each day an affiliate looks at the top-k products the model ranks
    highest. What share of those really surged? Averaged over days."""
    d = df[["dt", "surge"]].assign(p=proba)
    return float(d.sort_values("p", ascending=False).groupby("dt").head(k)
                  .groupby("dt")["surge"].mean().mean())


def best_threshold(y, proba) -> float:
    grid = np.linspace(0.1, 0.9, 81)
    scores = [f1_score(y, proba >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def evaluate(df, y, proba, threshold) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
        "f1": f1_score(y, pred, zero_division=0),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "precision_at_10": precision_at_k_per_day(df, proba, 10),
        "base_rate": float(np.mean(y)),
    }
