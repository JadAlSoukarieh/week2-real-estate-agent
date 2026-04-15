from __future__ import annotations

from typing import Any

import joblib
import numpy as np
import pandas as pd

from app.config import (
    API_TO_AMES_FEATURE_MAP,
    MODEL_ARTIFACT_PATH,
    SELECTED_FEATURES,
)
from app.schemas import PropertyFeaturesInput

_model: Any | None = None


def load_model() -> Any:
    global _model

    if _model is None:
        if not MODEL_ARTIFACT_PATH.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {MODEL_ARTIFACT_PATH}. "
                "Run the training script first."
            )
        _model = joblib.load(MODEL_ARTIFACT_PATH)

    return _model


def predict_from_features(features: PropertyFeaturesInput) -> float:
    model = load_model()

    input_frame = pd.DataFrame([features.model_dump()])
    ames_frame = input_frame.rename(columns=API_TO_AMES_FEATURE_MAP)
    ames_frame = ames_frame[SELECTED_FEATURES]

    log_prediction = model.predict(ames_frame)
    final_prediction = float(np.expm1(log_prediction[0]))

    return final_prediction
