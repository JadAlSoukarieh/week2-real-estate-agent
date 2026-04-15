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


class ErrorResponse(BaseModel):
    detail: str
