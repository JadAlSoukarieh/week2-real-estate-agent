from __future__ import annotations

import logging
import time

from pydantic import ValidationError

from app.config import (
    API_FEATURES,
    DEFAULT_EXTRACTION_PROMPT_VERSION,
    DEFAULT_INTERPRETATION_PROMPT_VERSION,
    MODEL_NAME,
    TARGET_TRANSFORM,
)
from app.schemas import (
    ChainedQueryResponse,
    ChainedQueryInput,
    ExtractedPropertyFeatures,
    FeatureOverridesInput,
    PropertyFeaturesInput,
)
from app.services.extraction_service import (
    ExtractionServiceUnavailableError,
    extract_features_from_query,
    normalize_feature_payload,
)
from app.services.interpretation_service import interpret_prediction
from app.services.prediction_service import predict_from_features

logger = logging.getLogger(__name__)


class ChainServiceError(RuntimeError):
    """Raised when the chain cannot build a valid prediction request."""


def analyze_query(payload: ChainedQueryInput) -> ChainedQueryResponse:
    total_start = time.perf_counter()
    extraction = extract_features_from_query(
        query=payload.query,
        prompt_version=DEFAULT_EXTRACTION_PROMPT_VERSION,
    )

    merged_features = merge_features(
        extracted_features=extraction.features,
        overrides=payload.overrides,
    )
    missing_fields = get_missing_fields(merged_features)

    if missing_fields:
        logger.info(
            "[chain] total_duration=%.2fs outcome=incomplete missing_fields=%s",
            time.perf_counter() - total_start,
            ",".join(missing_fields),
        )
        return ChainedQueryResponse(
            query=payload.query,
            extraction=extraction,
            final_features=merged_features,
            missing_fields_after_overrides=missing_fields,
            is_ready_for_prediction=False,
            prediction_ran=False,
            predicted_price=None,
            currency=None,
            model_name=None,
            target_transform=None,
            interpretation=None,
            notes="Prediction was skipped because required features are still missing.",
        )

    strict_features = validate_prediction_features(merged_features)
    prediction_start = time.perf_counter()
    predicted_price = predict_from_features(strict_features)
    logger.info(
        "[predict] duration=%.2fs outcome=success",
        time.perf_counter() - prediction_start,
    )
    interpretation = interpret_prediction(
        query=payload.query,
        final_features=merged_features.model_dump(),
        predicted_price=predicted_price,
        prompt_version=DEFAULT_INTERPRETATION_PROMPT_VERSION,
    )

    logger.info(
        "[chain] total_duration=%.2fs outcome=complete",
        time.perf_counter() - total_start,
    )

    return ChainedQueryResponse(
        query=payload.query,
        extraction=extraction,
        final_features=merged_features,
        missing_fields_after_overrides=[],
        is_ready_for_prediction=True,
        prediction_ran=True,
        predicted_price=predicted_price,
        currency="USD",
        model_name=MODEL_NAME,
        target_transform=TARGET_TRANSFORM,
        interpretation=interpretation,
        notes=None,
    )


def merge_features(
    extracted_features: ExtractedPropertyFeatures,
    overrides: FeatureOverridesInput | None,
) -> ExtractedPropertyFeatures:
    merged_payload = extracted_features.model_dump()

    if overrides is not None:
        override_payload = overrides.model_dump(exclude_none=True)
        merged_payload.update(override_payload)

    normalized_payload = normalize_feature_payload(merged_payload)
    return ExtractedPropertyFeatures.model_validate(normalized_payload)


def get_missing_fields(features: ExtractedPropertyFeatures) -> list[str]:
    feature_values = features.model_dump()
    return [
        feature_name
        for feature_name in API_FEATURES
        if feature_values.get(feature_name) is None
    ]


def validate_prediction_features(
    features: ExtractedPropertyFeatures,
) -> PropertyFeaturesInput:
    try:
        return PropertyFeaturesInput.model_validate(features.model_dump())
    except ValidationError as exc:
        raise ChainServiceError(
            "Merged features failed strict prediction validation."
        ) from exc
