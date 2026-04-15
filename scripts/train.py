from __future__ import annotations

import json
import sys
from math import sqrt
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import (
    API_TO_AMES_FEATURE_MAP,
    CANONICAL_DATASET_PATH,
    FALLBACK_DATASET_PATH,
    FEATURE_CONFIG_PATH,
    KITCHEN_QUAL_ORDER,
    MODEL_ARTIFACT_PATH,
    MODEL_NAME,
    NOMINAL_FEATURES,
    NUMERIC_FEATURES,
    ORDINAL_FEATURES,
    RANDOM_STATE,
    SELECTED_FEATURES,
    TARGET_COLUMN,
    TARGET_TRANSFORM,
    TRAINING_SUMMARY_PATH,
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


def build_model_pipeline() -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    ordinal_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OrdinalEncoder(
                    categories=[KITCHEN_QUAL_ORDER],
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                ),
            ),
        ]
    )

    nominal_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("ordinal", ordinal_pipeline, ORDINAL_FEATURES),
            ("nominal", nominal_pipeline, NOMINAL_FEATURES),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", Ridge(alpha=1.0)),
        ]
    )


def evaluate_split(model: Pipeline, features: pd.DataFrame, target: pd.Series) -> dict[str, float]:
    log_predictions = model.predict(features)
    predictions = np.expm1(log_predictions)

    mae = mean_absolute_error(target, predictions)
    rmse = sqrt(mean_squared_error(target, predictions))
    r2 = r2_score(target, predictions)

    return {
        "mae": round(float(mae), 2),
        "rmse": round(float(rmse), 2),
        "r2": round(float(r2), 4),
    }


def print_metrics(split_name: str, metrics: dict[str, float]) -> None:
    print(f"{split_name} metrics:")
    print(f"  MAE:  {metrics['mae']:.2f}")
    print(f"  RMSE: {metrics['rmse']:.2f}")
    print(f"  R2:   {metrics['r2']:.4f}")


def main() -> None:
    dataset_path = resolve_dataset_path()

    dataframe = pd.read_csv(dataset_path, usecols=SELECTED_FEATURES + [TARGET_COLUMN])
    features = dataframe[SELECTED_FEATURES]
    target = dataframe[TARGET_COLUMN]

    x_train_temp, x_test, y_train_temp, y_test = train_test_split(
        features,
        target,
        test_size=0.15,
        random_state=RANDOM_STATE,
    )

    validation_size = 0.15 / 0.85
    x_train, x_validation, y_train, y_validation = train_test_split(
        x_train_temp,
        y_train_temp,
        test_size=validation_size,
        random_state=RANDOM_STATE,
    )

    model = build_model_pipeline()
    model.fit(x_train, np.log1p(y_train))

    validation_metrics = evaluate_split(model, x_validation, y_validation)
    test_metrics = evaluate_split(model, x_test, y_test)

    print(f"Training completed using dataset: {dataset_path}")
    print(f"Training rows:   {len(x_train)}")
    print(f"Validation rows: {len(x_validation)}")
    print(f"Test rows:       {len(x_test)}")
    print()
    print_metrics("Validation", validation_metrics)
    print()
    print_metrics("Test", test_metrics)

    MODEL_ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_ARTIFACT_PATH)

    quantiles = y_train.quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    training_summary = {
        "model_name": MODEL_NAME,
        "target_name": TARGET_COLUMN,
        "target_transform": TARGET_TRANSFORM,
        "selected_features": SELECTED_FEATURES,
        "random_state": RANDOM_STATE,
        "split_sizes": {
            "train": int(len(x_train)),
            "validation": int(len(x_validation)),
            "test": int(len(x_test)),
        },
        "split_ratios": {
            "train": 0.70,
            "validation": 0.15,
            "test": 0.15,
        },
        "train_row_count": int(len(x_train)),
        "validation_row_count": int(len(x_validation)),
        "test_row_count": int(len(x_test)),
        "train_sale_price_median": round(float(y_train.median()), 2),
        "train_sale_price_mean": round(float(y_train.mean()), 2),
        "train_price_quantiles": {
            "0.1": round(float(quantiles.loc[0.1]), 2),
            "0.25": round(float(quantiles.loc[0.25]), 2),
            "0.5": round(float(quantiles.loc[0.5]), 2),
            "0.75": round(float(quantiles.loc[0.75]), 2),
            "0.9": round(float(quantiles.loc[0.9]), 2),
        },
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }

    feature_config = {
        "target_column": TARGET_COLUMN,
        "selected_features": SELECTED_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "ordinal_features": ORDINAL_FEATURES,
        "nominal_features": NOMINAL_FEATURES,
        "kitchen_qual_order": KITCHEN_QUAL_ORDER,
        "api_to_ames_feature_map": API_TO_AMES_FEATURE_MAP,
        "model_name": MODEL_NAME,
        "target_transform": TARGET_TRANSFORM,
        "random_state": RANDOM_STATE,
        "canonical_dataset_path": str(CANONICAL_DATASET_PATH),
    }

    TRAINING_SUMMARY_PATH.write_text(
        json.dumps(training_summary, indent=2),
        encoding="utf-8",
    )
    FEATURE_CONFIG_PATH.write_text(
        json.dumps(feature_config, indent=2),
        encoding="utf-8",
    )

    print()
    print(f"Saved model artifact to {MODEL_ARTIFACT_PATH}")
    print(f"Saved training summary to {TRAINING_SUMMARY_PATH}")
    print(f"Saved feature config to {FEATURE_CONFIG_PATH}")


if __name__ == "__main__":
    main()
