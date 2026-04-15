from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import (
    CANONICAL_DATASET_PATH,
    FALLBACK_DATASET_PATH,
    MODEL_ARTIFACT_PATH,
    SELECTED_FEATURES,
    TARGET_COLUMN,
)


def resolve_dataset_path() -> Path:
    if CANONICAL_DATASET_PATH.exists():
        return CANONICAL_DATASET_PATH
    if FALLBACK_DATASET_PATH.exists():
        return FALLBACK_DATASET_PATH

    raise FileNotFoundError(
        f"Dataset not found. Expected {CANONICAL_DATASET_PATH} "
        f"or fallback {FALLBACK_DATASET_PATH}."
    )


def main() -> None:
    if not MODEL_ARTIFACT_PATH.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {MODEL_ARTIFACT_PATH}. "
            "Run the training script first."
        )

    dataset_path = resolve_dataset_path()
    model = joblib.load(MODEL_ARTIFACT_PATH)

    dataframe = pd.read_csv(dataset_path, usecols=SELECTED_FEATURES + [TARGET_COLUMN])
    samples = dataframe.head(5).copy()

    log_predictions = model.predict(samples[SELECTED_FEATURES])
    predictions = np.expm1(log_predictions)

    print("Sample predictions in original dollar scale:")
    for index, (_, row) in enumerate(samples.iterrows(), start=1):
        actual_price = float(row[TARGET_COLUMN])
        predicted_price = float(predictions[index - 1])
        error = predicted_price - actual_price
        print(
            f"{index}. actual=${actual_price:,.2f} | "
            f"predicted=${predicted_price:,.2f} | "
            f"error=${error:,.2f}"
        )


if __name__ == "__main__":
    main()
