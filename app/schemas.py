from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator


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

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        cleaned_value = value.strip()
        if not cleaned_value:
            raise ValueError("query must not be empty")
        return cleaned_value


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


class FeatureOverridesInput(BaseModel):
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


class ChainedQueryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    overrides: FeatureOverridesInput | None = None

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        cleaned_value = value.strip()
        if not cleaned_value:
            raise ValueError("query must not be empty")
        return cleaned_value


class InterpretationOutput(BaseModel):
    summary: str
    price_position: str
    key_drivers: list[str]
    caveats: list[str]


class ChainedQueryResponse(BaseModel):
    query: str
    extraction: ExtractionResponse
    final_features: ExtractedPropertyFeatures
    missing_fields_after_overrides: list[str]
    is_ready_for_prediction: bool
    prediction_ran: bool
    predicted_price: float | None
    currency: str | None
    model_name: str | None
    target_transform: str | None
    interpretation: InterpretationOutput | None
    notes: str | None


class ErrorResponse(BaseModel):
    detail: str
