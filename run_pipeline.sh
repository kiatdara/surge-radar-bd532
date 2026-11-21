#!/usr/bin/env bash
# Rebuild everything from the raw EchoTik CSVs: EDA -> training (MLflow) -> SHAP -> fairness -> tests
set -euo pipefail
# No real EchoTik data in data/raw/? Use (or create) the synthetic sample.
[ -f data/raw/product_trend.csv ] || [ -f data/sample/product_trend.csv ] || python -m src.make_sample_data
python -m src.eda
python -m src.train
python -m src.explain
python -m src.fairness
python -m pytest -q tests
