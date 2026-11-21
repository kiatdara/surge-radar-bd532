"""Project-wide settings shared by training, API and tests."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"

SAMPLE_DIR = ROOT / "data" / "sample"

# Real EchoTik exports go in data/raw/ (not published: paid API data). Without them the
# project runs on the synthetic sample in data/sample/ (python -m src.make_sample_data).
# DATA_DIR=<folder> overrides both.
import os as _os
DATA_DIR = Path(_os.getenv("DATA_DIR") or (RAW_DIR if (RAW_DIR / "product_trend.csv").exists()
                                           else SAMPLE_DIR))
TREND_FILE = DATA_DIR / "product_trend.csv"     # EchoTik endpoint 13 (product trend, daily)
PRODUCT_FILE = DATA_DIR / "product_list.csv"    # EchoTik endpoint 01 (product list, static attributes)

# ---- Label definition -------------------------------------------------------
HORIZON_DAYS = 7          # look this many days ahead
LOOKBACK_DAYS = 7         # compare with this many days just before the prediction date
SURGE_RATIO = 1.20        # "surge" = next-7-day average daily sales >= 1.2 x last-7-day average
MIN_OBS = 5               # need at least 5 observed days in each 7-day window

# ---- Time-based split (gaps of HORIZON_DAYS stop label leakage across splits) --
TRAIN_END = "2026-07-20"
VAL_START, VAL_END = "2026-07-28", "2026-08-16"
TEST_START, TEST_END = "2026-08-24", "2026-09-09"

TOP_CATEGORIES = ["601450", "600001", "700437", "601152", "601739", "700645"]
RANDOM_STATE = 42
