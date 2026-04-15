from typing import Literal

from pydantic import BaseModel, ConfigDict


KitchenQualValue = Literal["Po", "Fa", "TA", "Gd", "Ex"]


class PropertyFeaturesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    overall_qual: int
    gr_liv_area: float
    neighborhood: str
    kitchen_qual: KitchenQualValue
    garage_cars: float
    total_bsmt_sf: float
    year_built: int
    year_remod_add: int
    full_bath: float
    house_style: str


class PredictionResponse(BaseModel):
    predicted_price: float
    currency: str
    model_name: str
    target_transform: str
    used_features: list[str]


class QueryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str


class ExtractedPropertyFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid")

    overall_qual: int | None = None
    gr_liv_area: float | None = None
    neighborhood: str | None = None
    kitchen_qual: KitchenQualValue | None = None
    garage_cars: int | None = None
    total_bsmt_sf: float | None = None
    year_built: int | None = None
    year_remod_add: int | None = None
    full_bath: int | None = None
    house_style: str | None = None


class ExtractionResponse(BaseModel):
    query: str
    features: ExtractedPropertyFeatures
    extracted_fields: list[str]
    missing_fields: list[str]
    is_complete: bool
    prompt_version: str
    validation_passed: bool
    notes: str | None = None


class ErrorResponse(BaseModel):
    detail: str
