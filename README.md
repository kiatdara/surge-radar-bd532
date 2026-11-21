# Surge Radar — TikTok Shop (Thailand) sales-surge prediction

BD532 term project. An AI component that tells affiliate creators and sellers **which TikTok Shop
products will sell at least 20% more next week**, explains why, and serves the answer through a
REST API and a small web dashboard. The data comes from the EchoTik API (via RapidAPI), the same
pipeline that feeds my Independent Study data warehouse.

| | |
|---|---|
| **Prediction task** | For product *p* on day *t*: will average daily sales over days *t+1…t+7* be ≥ 1.2× the average over *t−6…t*? (binary, "surge") |
| **Data** | 100 products × 170 days (31 Mar – 16 Sep 2026) of daily EchoTik product trends + static product attributes (paid data, not published; a synthetic sample is included, see [Data](#data)) |
| **Final model** | Random forest (v2), selected from 5 versions tracked in MLflow |
| **Test result (24 Aug – 9 Sep 2026)** | ROC-AUC 0.70, PR-AUC 0.45 (base rate 0.21), precision of the daily top-10 = 0.58 vs 0.32 for a momentum rule |
| **API latency** | p95 93 ms server-side for one product with SHAP reasons |

## Repository layout

```
data/sample/                   SYNTHETIC sample with the EchoTik columns, created by run_pipeline.sh
data/raw/                      put the real EchoTik exports here (not published, see "Data")
data/processed/                generated: latest feature snapshot, test predictions
src/make_sample_data.py        generates the synthetic sample
src/config.py                  label definition, split dates, paths
src/features.py                cleaning, feature engineering, label, time split, serving features
src/modeling.py                model versions, preprocessing, metrics (incl. precision@10/day)
src/eda.py                     3.1 data exploration  -> reports/eda_summary.json, figures
src/train.py                   3.1/3.3 training loop, time-series CV, MLflow tracking, selection
src/explain.py                 3.4/3.5 SHAP global + local explanations, permutation importance
src/explain_utils.py           SHAP -> plain-language reasons (English / Thai), used by the API
src/fairness.py                3.2 group fairness + top-10 exposure + re-ranking mitigation
src/benchmark.py               API latency benchmark
api/app.py, api/schemas.py     3.6/4 FastAPI service, payloads, error handling
api/static/index.html          web UI (dashboard + request tester)
tests/                         22 pytest tests (features, leakage, API scenarios, error codes)
postman/                       Postman collection (13 requests, 27 assertions) + newman results
models/                        generated: trained model bundle (pipeline, threshold, metadata)
reports/                       metrics, fairness, explainability JSON FROM THE REAL DATA
                               (figures are in the report PDF; rerunning recreates them in reports/figures/)
docs/sequence_diagram.mmd      source of the sequence diagram in the report
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
bash run_pipeline.sh        # EDA -> train (MLflow) -> SHAP -> fairness -> tests  (~1 min)
```

`run_pipeline.sh` builds the model from `data/raw/` when the real EchoTik files are there,
otherwise from the synthetic sample in `data/sample/`. Run it once after cloning: the API needs
`models/surge_model.joblib`, which is not committed.

## Run the service

```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

* Web UI: <http://localhost:8000/>
* Interactive API docs (OpenAPI/Swagger): <http://localhost:8000/docs>
* MLflow Tracking UI: `mlflow ui --backend-store-uri sqlite:///mlflow.db` → <http://127.0.0.1:5000>

### Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/v1/health` | Model version, readiness (UI polls every 5 s) |
| GET | `/api/v1/products/top?k=10&category=&shop_type=&fair=&lang=` | Today's ranked list with reasons |
| GET | `/api/v1/products/{product_id}/prediction` | Prediction for one stored product |
| POST | `/api/v1/predict` | Score any product from its own 28–120 days of history |
| GET | `/api/v1/samples/{product_id}` | Builds a valid `/predict` body from stored data |
| POST | `/api/v1/feedback` | Store the user's verdict (`useful`, `not_useful`, `wrong_surge`, `missed_surge`) |

Example:

```bash
PID=$(curl -s "localhost:8000/api/v1/products/top?k=1" | python -c "import sys,json;print(json.load(sys.stdin)['items'][0]['product_id'])")
curl -s localhost:8000/api/v1/samples/$PID > body.json
curl -s -X POST localhost:8000/api/v1/predict -H "Content-Type: application/json" -d @body.json
```

Errors always use one body shape `{"request_id", "status": "error", "error": {"code", "message", "field"}}`
with status codes 400 (`INVALID_JSON`, `MISSING_FIELD`), 404 (`PRODUCT_NOT_FOUND`), 413
(`PAYLOAD_TOO_LARGE`, `HISTORY_TOO_LONG`), 415 (`UNSUPPORTED_MEDIA_TYPE`), 422 (`INVALID_VALUE`,
`INSUFFICIENT_HISTORY`), 429 (`RATE_LIMITED`, with `Retry-After`), 500 (`INFERENCE_ERROR`) and
503 (`MODEL_NOT_READY`).

## Tests

```bash
python -m pytest -q tests                                     # unit + API tests
newman run postman/surge_radar.postman_collection.json        # with the server running
```

Or import `postman/surge_radar.postman_collection.json` into Postman and press **Run collection**.

## Data

The results in the report and in `reports/` come from **real EchoTik data** (100 products,
31 Mar – 16 Sep 2026). That data is bought from a paid API, so it is **not in this repository**,
and neither is anything derived from it row by row (feature snapshots, predictions, model file).

- **Out of the box** the code runs on `data/sample/`, a **synthetic** dataset that
  `bash run_pipeline.sh` creates on first run (fixed random seed, so everyone gets the same file), with the same
  columns and similar behaviour (double-day campaign spikes, creator growth, crawl gaps,
  duplicate days), made by `python -m src.make_sample_data`. Every script, test and the API work
  on it, but its numbers are not the report's numbers, and running the pipeline overwrites
  `reports/` with sample results.
- **To use real data**, get an EchoTik key on RapidAPI (`tiktok-ultra-api1.p.rapidapi.com`, region
  `TH`) and export endpoint *product trend* (daily rows: `dt, product_id, spu_avg_price,
  total_ifl_cnt, total_live_cnt, total_sale_1d_cnt, total_sale_cnt, total_sale_gmv_1d_amt,
  total_sale_gmv_amt, total_video_cnt, _product_id`) and endpoint *product list* (columns as in
  `data/sample/product_list.csv`) to `data/raw/product_trend.csv` and `data/raw/product_list.csv`,
  then run `bash run_pipeline.sh`. The real files are picked up automatically. Never commit the API
  key or the raw files (`.gitignore` already excludes them).

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `MODEL_PATH` | `models/surge_model.joblib` | Model bundle to serve |
| `RATE_LIMIT_PER_MIN` | `120` | Requests per minute per client before 429 |
| `FEEDBACK_PATH` | `data/feedback/feedback.jsonl` | Where feedback is appended |
