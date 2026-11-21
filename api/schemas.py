"""Request / response payloads of the Surge Radar API (Section 4.1)."""
from __future__ import annotations

from datetime import date
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ------------------------------- requests --------------------------------- #
class DailyRecord(BaseModel):
    dt: date = Field(description="Calendar day (YYYY-MM-DD)")
    total_sale_1d_cnt: float = Field(ge=0, description="Units sold that day")
    spu_avg_price: float = Field(ge=0, description="Average selling price")
    total_ifl_cnt: float = Field(ge=0, description="Cumulative number of creators")
    total_video_cnt: float = Field(ge=0, description="Cumulative number of videos")
    total_live_cnt: float = Field(ge=0, description="Cumulative number of live streams")


class ProductInfo(BaseModel):
    product_id: Optional[str] = Field(None, max_length=32)
    product_name: Optional[str] = Field(None, max_length=300)
    category_id: str = Field(min_length=1, max_length=16)
    product_commission_rate: float = Field(ge=0, le=1)
    product_rating: float = Field(ge=0, le=5)
    review_count: int = Field(ge=0)
    discount: str = Field("0%", pattern=r"^\d{1,2}%?$|^100%?$")
    shop_type: Literal[0, 1] = Field(0, description="0 = regular shop, 1 = TikTok Shop Mall")
    first_crawl_dt: int = Field(ge=20180101, le=20991231, description="YYYYMMDD first seen")


class PredictOptions(BaseModel):
    explain: bool = True
    top_k_reasons: int = Field(3, ge=1, le=8)
    language: Literal["en", "th"] = "en"


class PredictRequest(BaseModel):
    request_id: str = Field(min_length=1, max_length=64)
    product: ProductInfo
    history: list[DailyRecord] = Field(description="28–120 consecutive-ish daily records, "
                                                   "oldest first; the last day is 'today'")
    options: PredictOptions = PredictOptions()

    @field_validator("history")
    @classmethod
    def sorted_unique(cls, v):
        days = [r.dt for r in v]
        if days != sorted(days) or len(set(days)) != len(days):
            raise ValueError("history must be sorted by dt with no duplicate days")
        return v


class FeedbackRequest(BaseModel):
    request_id: Optional[str] = Field(None, max_length=64)
    product_id: str = Field(min_length=1, max_length=32)
    prediction_date: date
    label: Literal["useful", "not_useful", "wrong_surge", "missed_surge"]
    source: Literal["dashboard", "api", "creator_app"] = "dashboard"
    comment: Optional[str] = Field(None, max_length=500)


# ------------------------------- responses -------------------------------- #
class Reason(BaseModel):
    feature: str
    value: str
    shap: float
    direction: Literal["up", "down"]
    text: str


class Prediction(BaseModel):
    product_id: Optional[str]
    product_name: Optional[str] = None
    as_of: date
    horizon_days: int
    surge_probability: float
    label: Literal["hot", "watch", "skip"]
    recommended_action: str
    reasons: list[Reason] = []
    current_sales_per_day: Optional[float] = None


class PredictResponse(BaseModel):
    request_id: str
    status: Literal["ok"] = "ok"
    model_version: str
    prediction: Prediction
    latency_ms: dict


class TopResponse(BaseModel):
    status: Literal["ok"] = "ok"
    model_version: str
    as_of: date
    k: int
    fair_rerank: bool
    items: list[Prediction]


class ErrorBody(BaseModel):
    code: str
    message: str
    field: Optional[str] = None


class ErrorResponse(BaseModel):
    request_id: Optional[str] = None
    status: Literal["error"] = "error"
    error: ErrorBody
