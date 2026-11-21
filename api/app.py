"""Surge Radar REST API (Sections 3.6 and 4).

Start:  uvicorn api.app:app --host 0.0.0.0 --port 8000
Docs:   http://localhost:8000/docs   (OpenAPI / Swagger, generated from api/schemas.py)
UI:     http://localhost:8000/
"""
from __future__ import annotations

import json
import os
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from src import config as C
from src.explain_utils import Reasoner
from src.features import FEATURES, features_for_single_product, load_raw

from .schemas import (ErrorResponse, FeedbackRequest, Prediction, PredictRequest,
                      PredictResponse, TopResponse)

MODEL_PATH = Path(os.getenv("MODEL_PATH", C.MODEL_DIR / "surge_model.joblib"))
FEEDBACK_PATH = Path(os.getenv("FEEDBACK_PATH", C.ROOT / "data" / "feedback" / "feedback.jsonl"))
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "120"))
MAX_BODY_BYTES = 1_000_000
MIN_HISTORY, MAX_HISTORY = 28, 120
HOT, WATCH = 0.55, None            # WATCH threshold = model's F1-optimal threshold

STATE: dict = {"ready": False}


class APIError(Exception):
    def __init__(self, status: int, code: str, message: str, field: str | None = None,
                 request_id: str | None = None, headers: dict | None = None):
        self.status, self.code, self.message, self.field = status, code, message, field
        self.request_id, self.headers = request_id, headers or {}


def load_state():
    bundle = joblib.load(MODEL_PATH)
    clf = bundle["pipeline"].named_steps["clf"]
    if hasattr(clf, "n_jobs"):
        clf.n_jobs = 1                      # one request = few rows: threads only add overhead
    STATE.update(bundle=bundle, pipe=bundle["pipeline"], threshold=bundle["threshold"],
                 reasoner=Reasoner(bundle["pipeline"], bundle["numeric"]))
    latest = pd.read_csv(C.PROCESSED_DIR / "latest_features.csv",
                         dtype={"product_id": str, "category": str}, parse_dates=["dt"])
    trend, products = load_raw()
    STATE.update(latest=latest.set_index("product_id"), trend=trend,
                 products=products.set_index("product_id"), ready=True)


@asynccontextmanager
async def lifespan(_app):
    try:
        load_state()
    except Exception as e:                      # API still starts; /health reports not ready
        STATE.update(ready=False, load_error=str(e))
    yield


app = FastAPI(title="Surge Radar API – TikTok Shop sales-surge prediction",
              version="1.0.0", lifespan=lifespan,
              description="Predicts whether a TikTok Shop (TH) product's average daily sales "
                          "in the next 7 days will be at least 1.2× the last 7 days, and "
                          "explains why. Data: EchoTik API.",
              responses={s: {"model": ErrorResponse} for s in (400, 404, 413, 415, 422, 429,
                                                               500, 503)})


# ----------------------------- error handling ----------------------------- #
def error_json(status, code, message, field=None, request_id=None, headers=None):
    body = {"request_id": request_id, "status": "error",
            "error": {"code": code, "message": message, "field": field}}
    return JSONResponse(status_code=status, content=body, headers=headers)


@app.exception_handler(APIError)
async def _api_error(_req, e: APIError):
    return error_json(e.status, e.code, e.message, e.field, e.request_id, e.headers)


@app.exception_handler(RequestValidationError)
async def _validation(req: Request, e: RequestValidationError):
    err = e.errors()[0]
    if err.get("type") == "json_invalid":
        return error_json(400, "INVALID_JSON", "Body is not valid JSON")
    loc = ".".join(str(x) for x in err.get("loc", []) if x != "body")
    code = "MISSING_FIELD" if err.get("type") == "missing" else "INVALID_VALUE"
    status = 400 if code == "MISSING_FIELD" else 422
    rid = None
    try:
        rid = (await req.json()).get("request_id")
    except Exception:
        pass
    return error_json(status, code, f"{loc}: {err.get('msg')}", loc or None, rid)


@app.exception_handler(StarletteHTTPException)
async def _http(_req, e: StarletteHTTPException):
    code = {404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED"}.get(e.status_code, "HTTP_ERROR")
    return error_json(e.status_code, code, str(e.detail))


_hits: dict[str, deque] = defaultdict(deque)


@app.middleware("http")
async def guard(request: Request, call_next):
    """415 for non-JSON bodies, 413 for oversized bodies, 429 per-client rate limit."""
    if request.url.path.startswith("/api/"):
        now, q = time.time(), _hits[request.client.host if request.client else "?"]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= RATE_LIMIT_PER_MIN:
            return error_json(429, "RATE_LIMITED",
                              f"More than {RATE_LIMIT_PER_MIN} requests per minute",
                              headers={"Retry-After": str(int(60 - (now - q[0])) + 1)})
        q.append(now)
        if request.method == "POST":
            ctype = request.headers.get("content-type", "")
            if not ctype.startswith("application/json"):
                return error_json(415, "UNSUPPORTED_MEDIA_TYPE",
                                  "Content-Type must be application/json")
            if int(request.headers.get("content-length") or 0) > MAX_BODY_BYTES:
                return error_json(413, "PAYLOAD_TOO_LARGE", "Body larger than 1 MB")
    return await call_next(request)


def require_ready(request_id=None):
    if not STATE.get("ready"):
        raise APIError(503, "MODEL_NOT_READY", "Model is loading or failed to load; "
                       "retry /api/v1/health in 5 s", request_id=request_id,
                       headers={"Retry-After": "5"})


# ------------------------------- helpers ---------------------------------- #
def label_for(p: float) -> tuple[str, str]:
    if p >= HOT:
        return "hot", "Promote now: post a video or book a live within 2–3 days"
    if p >= STATE["threshold"]:
        return "watch", "Add to watch-list and check again tomorrow"
    return "skip", "No action needed"


def predict_rows(X: pd.DataFrame, explain=True, k=3, lang="en"):
    t0 = time.perf_counter()
    proba = STATE["pipe"].predict_proba(X[FEATURES])[:, 1]
    t1 = time.perf_counter()
    reasons = []
    for i in range(len(X)):
        if not explain:
            reasons.append([]); continue
        rs = STATE["reasoner"].explain_row(X.iloc[[i]][FEATURES], top_k=k)
        if lang == "th":
            for r in rs:
                r["text"] = r["text_th"].replace("units/day", "ชิ้น/วัน").replace(" days", " วัน")
                r["value"] = r["value"].replace("units/day", "ชิ้น/วัน").replace(" days", " วัน")
        reasons.append([{kk: v for kk, v in r.items() if kk != "text_th"} for r in rs])
    t2 = time.perf_counter()
    return proba, reasons, {"inference": round((t1 - t0) * 1e3, 2),
                            "explain": round((t2 - t1) * 1e3, 2)}


def to_prediction(row, p, reasons) -> Prediction:
    label, action = label_for(float(p))
    return Prediction(product_id=row.get("product_id"),
                      product_name=(str(row.get("product_name"))[:120]
                                    if pd.notna(row.get("product_name")) else None),
                      as_of=pd.Timestamp(row["dt"]).date(), horizon_days=C.HORIZON_DAYS,
                      surge_probability=round(float(p), 4), label=label,
                      recommended_action=action, reasons=reasons,
                      current_sales_per_day=(round(float(row["sales_mean_7d"]), 1)
                                             if pd.notna(row.get("sales_mean_7d")) else None))


# ------------------------------- endpoints -------------------------------- #
@app.get("/api/v1/health", tags=["system"])
def health():
    b = STATE.get("bundle", {})
    return {"status": "ok" if STATE.get("ready") else "not_ready",
            "model_version": b.get("model_version"), "model_name": b.get("model_name"),
            "trained_until": b.get("train_end"), "threshold": STATE.get("threshold"),
            "products_available": len(STATE["latest"]) if STATE.get("ready") else 0,
            "error": STATE.get("load_error")}


@app.post("/api/v1/predict", response_model=PredictResponse, tags=["prediction"])
def predict(req: PredictRequest):
    """Score one product from its own daily history (28–120 days). The server builds the
    features with exactly the same code that built the training data."""
    t0 = time.perf_counter()
    require_ready(req.request_id)
    n = len(req.history)
    if n < MIN_HISTORY:
        raise APIError(422, "INSUFFICIENT_HISTORY", f"history has {n} days; need at least "
                       f"{MIN_HISTORY} to compute 28-day features", "history", req.request_id)
    if n > MAX_HISTORY:
        raise APIError(413, "HISTORY_TOO_LONG", f"history has {n} days; send at most "
                       f"{MAX_HISTORY}", "history", req.request_id)
    hist = pd.DataFrame([r.model_dump() for r in req.history])
    hist["dt"] = hist["dt"].astype(str)
    observed_last7 = hist.tail(7)["total_sale_1d_cnt"].gt(0).sum()
    if observed_last7 < C.MIN_OBS:
        raise APIError(422, "INSUFFICIENT_HISTORY", "need sales on at least 5 of the last "
                       "7 days", "history", req.request_id)
    try:
        X = features_for_single_product(hist, req.product.model_dump())
        t1 = time.perf_counter()
        if X[["log_sales_mean_7d", "log_sales_mean_28d"]].isna().any(axis=None):
            raise APIError(422, "INSUFFICIENT_HISTORY", "too many missing days in the last "
                           "28 days", "history", req.request_id)
        proba, reasons, lat = predict_rows(X, req.options.explain, req.options.top_k_reasons,
                                           req.options.language)
    except APIError:
        raise
    except Exception as e:  # pragma: no cover - defensive
        raise APIError(500, "INFERENCE_ERROR", f"model failed: {type(e).__name__}",
                       request_id=req.request_id)
    row = X.iloc[0].to_dict() | {"product_name": req.product.product_name}
    return PredictResponse(request_id=req.request_id,
                           model_version=STATE["bundle"]["model_version"],
                           prediction=to_prediction(row, proba[0], reasons[0]),
                           latency_ms={"features": round((t1 - t0) * 1e3, 2), **lat,
                                       "total": round((time.perf_counter() - t0) * 1e3, 2)})


@app.get("/api/v1/products/top", response_model=TopResponse, tags=["prediction"])
def top_products(k: int = Query(10, ge=1, le=50),
                 category: str | None = Query(None, max_length=16),
                 shop_type: int | None = Query(None, ge=0, le=1),
                 fair: bool = Query(False, description="exposure-aware re-ranking: reserve "
                                                       "slots for each shop type"),
                 lang: str = Query("en", pattern="^(en|th)$")):
    """Daily ranked list for affiliate creators (the main UI screen)."""
    require_ready()
    df = STATE["latest"].reset_index()
    df = df[df["dt"] == df["dt"].max()]
    if category:
        df = df[df["category"] == category]
    if shop_type is not None:
        df = df[df["shop_type"] == shop_type]
    if df.empty:
        raise APIError(404, "NO_PRODUCTS", "no products match the filters")
    df = df.assign(p=STATE["pipe"].predict_proba(df[FEATURES])[:, 1])
    if fair and shop_type is None:
        quota = {g: int(np.floor(k * (df["shop_type"] == g).mean())) for g in (0, 1)}
        picked = pd.concat([df[df["shop_type"] == g].nlargest(q, "p") for g, q in quota.items()])
        df = pd.concat([picked, df.drop(picked.index).nlargest(k - len(picked), "p")])
    df = df.nlargest(k, "p").reset_index(drop=True)
    _, reasons, _ = predict_rows(df, True, 3, lang)
    items = [to_prediction(df.iloc[i].to_dict(), df["p"].iloc[i], reasons[i])
             for i in range(len(df))]
    return TopResponse(model_version=STATE["bundle"]["model_version"],
                       as_of=pd.Timestamp(df["dt"].max()).date(), k=k, fair_rerank=fair,
                       items=items)


@app.get("/api/v1/products/{product_id}/prediction", response_model=Prediction,
         tags=["prediction"])
def product_prediction(product_id: str, lang: str = Query("en", pattern="^(en|th)$")):
    require_ready()
    if product_id not in STATE["latest"].index:
        raise APIError(404, "PRODUCT_NOT_FOUND", f"unknown product_id {product_id}",
                       "product_id")
    row = STATE["latest"].loc[[product_id]].reset_index()
    proba, reasons, _ = predict_rows(row, True, 4, lang)
    return to_prediction(row.iloc[0].to_dict(), proba[0], reasons[0])


@app.get("/api/v1/samples/{product_id}", tags=["helpers"])
def sample_request(product_id: str, days: int = Query(45, ge=MIN_HISTORY, le=MAX_HISTORY)):
    """Builds a ready-to-send /predict body from stored EchoTik data (for Postman / UI)."""
    require_ready()
    t, p = STATE["trend"], STATE["products"]
    if product_id not in p.index:
        raise APIError(404, "PRODUCT_NOT_FOUND", f"unknown product_id {product_id}",
                       "product_id")
    # raw EchoTik trend can contain the same day twice (re-crawls): keep the last copy
    h = (t[t["_product_id"] == product_id].sort_values("dt")
         .drop_duplicates("dt", keep="last").tail(days))
    cols = ["dt", "total_sale_1d_cnt", "spu_avg_price", "total_ifl_cnt", "total_video_cnt",
            "total_live_cnt"]
    info = p.loc[product_id]
    return {"request_id": str(uuid.uuid4()),
            "product": {"product_id": product_id, "product_name": str(info["product_name"])[:120],
                        "category_id": str(info["category_id"]),
                        "product_commission_rate": float(info["product_commission_rate"]),
                        "product_rating": float(info["product_rating"]),
                        "review_count": int(info["review_count"]),
                        "discount": str(info["discount"]) if pd.notna(info["discount"]) else "0%",
                        "shop_type": int(info["shop_type"]),
                        "first_crawl_dt": int(info["first_crawl_dt"])},
            "history": h[cols].to_dict(orient="records"),
            "options": {"explain": True, "top_k_reasons": 3, "language": "en"}}


@app.post("/api/v1/feedback", status_code=201, tags=["feedback"])
def feedback(fb: FeedbackRequest):
    """Stores the user's verdict; used as extra labels and to monitor usefulness."""
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    fid = f"fb-{uuid.uuid4().hex[:8]}"
    rec = fb.model_dump(mode="json") | {"feedback_id": fid,
                                        "received_at": datetime.now(timezone.utc).isoformat()}
    with open(FEEDBACK_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return {"feedback_id": fid, "status": "recorded"}


# ------------------------------- web UI ----------------------------------- #
STATIC = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(STATIC / "index.html")
