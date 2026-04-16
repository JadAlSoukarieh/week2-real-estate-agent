from __future__ import annotations

import json
import logging
import time
from typing import Any
from urllib import error, request

from pydantic import ValidationError

from app.config import (
    DEFAULT_INTERPRETATION_PROMPT_VERSION,
    INTERPRETATION_PROMPT_V1_PATH,
    OLLAMA_API_URL,
    OLLAMA_MODEL_NAME,
    OLLAMA_TIMEOUT_SECONDS,
    TRAINING_SUMMARY_PATH,
)
from app.schemas import InterpretationOutput
from app.services.extraction_service import strip_code_fences

logger = logging.getLogger(__name__)
_training_summary_cache: dict[str, Any] | None = None


def interpret_prediction(
    query: str,
    final_features: dict[str, Any],
    predicted_price: float,
    prompt_version: str = DEFAULT_INTERPRETATION_PROMPT_VERSION,
) -> InterpretationOutput:
    start_time = time.perf_counter()
    training_summary = load_training_summary()
    price_position = get_price_position(
        predicted_price=predicted_price,
        quantiles=training_summary["train_price_quantiles"],
    )
    prompt_text = load_prompt_text(prompt_version)
    prompt = prompt_text.replace("{query}", query)
    prompt = prompt.replace(
        "{final_features_json}",
        json.dumps(final_features, indent=2, sort_keys=True),
    )
    prompt = prompt.replace("{predicted_price}", f"{predicted_price:.2f}")
    prompt = prompt.replace("{price_position}", price_position)
    prompt = prompt.replace(
        "{training_summary_json}",
        json.dumps(training_summary, indent=2, sort_keys=True),
    )

    try:
        raw_model_text = call_ollama(
            prompt=prompt,
            response_schema=get_interpretation_response_schema(),
        )
        parsed_output = parse_interpretation_output(raw_model_text)
        validated_output = InterpretationOutput.model_validate(parsed_output)
        result = harden_interpretation_output(
            interpretation=validated_output,
            final_features=final_features,
            predicted_price=predicted_price,
            training_summary=training_summary,
        )
        logger.info(
            "[interpret] duration=%.2fs outcome=success",
            time.perf_counter() - start_time,
        )
        return result
    except (InterpretationServiceUnavailableError, json.JSONDecodeError, ValidationError, TypeError, ValueError):
        logger.info(
            "[interpret] duration=%.2fs outcome=fallback",
            time.perf_counter() - start_time,
        )
        return build_fallback_interpretation(
            final_features=final_features,
            predicted_price=predicted_price,
            training_summary=training_summary,
        )


def load_training_summary() -> dict[str, Any]:
    global _training_summary_cache

    if _training_summary_cache is None:
        _training_summary_cache = json.loads(
            TRAINING_SUMMARY_PATH.read_text(encoding="utf-8")
        )

    return _training_summary_cache


def load_prompt_text(prompt_version: str) -> str:
    prompt_path = get_prompt_path(prompt_version)
    return prompt_path.read_text(encoding="utf-8")


def get_prompt_path(prompt_version: str):
    if prompt_version == "interpretation_v1":
        return INTERPRETATION_PROMPT_V1_PATH
    raise ValueError(f"Unsupported prompt version: {prompt_version}")


class InterpretationServiceUnavailableError(RuntimeError):
    """Raised when the local Ollama service cannot be reached for interpretation."""


def call_ollama(prompt: str, response_schema: dict) -> str:
    payload = {
        "model": OLLAMA_MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "format": response_schema,
        "options": {"temperature": 0},
    }

    request_body = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        OLLAMA_API_URL,
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(http_request, timeout=OLLAMA_TIMEOUT_SECONDS) as response:
            response_body = response.read().decode("utf-8")
    except (error.URLError, TimeoutError, ConnectionError) as exc:
        raise InterpretationServiceUnavailableError(
            "Could not reach Ollama service."
        ) from exc

    try:
        response_payload = json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise InterpretationServiceUnavailableError(
            "Could not reach Ollama service."
        ) from exc

    generated_text = response_payload.get("response")
    if isinstance(generated_text, str):
        return generated_text

    message_payload = response_payload.get("message")
    if isinstance(message_payload, dict):
        message_content = message_payload.get("content")
        if isinstance(message_content, str):
            return message_content

    raise InterpretationServiceUnavailableError("Could not reach Ollama service.")


def get_interpretation_response_schema() -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "summary": {"type": "string"},
            "price_position": {"type": "string"},
            "key_drivers": {
                "type": "array",
                "items": {"type": "string"},
            },
            "caveats": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["summary", "price_position", "key_drivers", "caveats"],
    }


def parse_interpretation_output(raw_model_text: str) -> dict[str, Any]:
    cleaned_text = strip_code_fences(raw_model_text).strip()
    parsed_output = json.loads(cleaned_text)
    if not isinstance(parsed_output, dict):
        raise ValueError("Interpretation output must be a JSON object.")
    return parsed_output


def build_fallback_interpretation(
    final_features: dict[str, Any],
    predicted_price: float,
    training_summary: dict[str, Any],
) -> InterpretationOutput:
    quantiles = training_summary["train_price_quantiles"]
    price_position = get_price_position(predicted_price, quantiles)

    summary = (
        f"The estimated price is about ${predicted_price:,.0f}, which sits "
        f"{price_position} in the training data."
    )

    key_drivers = []
    overall_qual = final_features.get("overall_qual")
    gr_liv_area = final_features.get("gr_liv_area")
    kitchen_qual = final_features.get("kitchen_qual")
    year_built = final_features.get("year_built")
    year_remod_add = final_features.get("year_remod_add")

    if overall_qual is not None:
        if overall_qual >= 8:
            key_drivers.append("Higher overall quality likely supports the estimate.")
        elif overall_qual <= 4:
            key_drivers.append("Lower overall quality likely pulls the estimate down.")

    if gr_liv_area is not None and gr_liv_area >= 1800:
        key_drivers.append("The above-grade living area is on the larger side.")

    if kitchen_qual in {"Gd", "Ex"}:
        key_drivers.append("Kitchen quality likely adds value relative to a typical home.")

    if year_built is not None and year_built >= 2000:
        key_drivers.append("A newer build year may support a stronger price point.")
    elif year_remod_add is not None and year_remod_add >= 2000:
        key_drivers.append("A more recent remodel may support the estimate.")

    if not key_drivers:
        key_drivers.append("The estimate mainly reflects the provided feature mix.")

    caveats = [
        "This estimate is based only on the provided features.",
        "It does not account for broader market timing or listing-specific details.",
    ]

    return InterpretationOutput(
        summary=summary,
        price_position=price_position,
        key_drivers=key_drivers[:3],
        caveats=caveats,
    )


def harden_interpretation_output(
    interpretation: InterpretationOutput,
    final_features: dict[str, Any],
    predicted_price: float,
    training_summary: dict[str, Any],
) -> InterpretationOutput:
    fallback = build_fallback_interpretation(
        final_features=final_features,
        predicted_price=predicted_price,
        training_summary=training_summary,
    )

    safe_key_drivers = sanitize_text_list(
        items=interpretation.key_drivers,
        final_features=final_features,
    )
    if not safe_key_drivers:
        safe_key_drivers = fallback.key_drivers

    safe_caveats = sanitize_text_list(
        items=interpretation.caveats,
        final_features=final_features,
    )
    if not safe_caveats:
        safe_caveats = fallback.caveats

    # Keep the top-line market framing deterministic and tied only to the
    # training distribution so the UI stays demo-safe.
    return InterpretationOutput(
        summary=fallback.summary,
        price_position=fallback.price_position,
        key_drivers=safe_key_drivers[:3],
        caveats=safe_caveats[:3],
    )


def sanitize_text_list(items: list[str], final_features: dict[str, Any]) -> list[str]:
    cleaned_items: list[str] = []
    for item in items:
        if not isinstance(item, str):
            continue
        stripped = item.strip()
        if not stripped:
            continue
        if contains_unsafe_segment_comparison(stripped, final_features):
            continue
        if contains_unsupported_model_detail(stripped):
            continue
        humanized = humanize_interpretation_text(stripped)
        if looks_like_raw_feature_fragment(humanized):
            continue
        cleaned_items.append(humanized)
    return cleaned_items


def contains_unsafe_segment_comparison(text: str, final_features: dict[str, Any]) -> bool:
    lowered_text = text.lower()
    banned_phrases = [
        "above average for",
        "below average for",
        "average for",
        "compared to similar",
        "compared with similar",
        "compared to other",
        "compared with other",
        "relative to similar",
        "relative to other",
    ]
    if any(phrase in lowered_text for phrase in banned_phrases):
        return True

    neighborhood = final_features.get("neighborhood")
    if isinstance(neighborhood, str) and neighborhood.strip():
        if neighborhood.strip().lower() in lowered_text:
            return True

    house_style = final_features.get("house_style")
    if isinstance(house_style, str) and house_style.strip():
        house_style_terms = get_house_style_terms(house_style)
        if any(term in lowered_text for term in house_style_terms):
            return True

    return False


def contains_unsupported_model_detail(text: str) -> bool:
    lowered_text = text.lower()
    banned_terms = [
        "rmse",
        "mae",
        "r2",
        "test set",
        "validation set",
        "training summary",
        "model performance",
        "pipeline",
    ]
    return any(term in lowered_text for term in banned_terms)


def humanize_interpretation_text(text: str) -> str:
    replacements = {
        "Overall Qual": "Overall quality",
        "Gr Liv Area": "Above-ground living area",
        "Kitchen Qual": "Kitchen quality",
        "Garage Cars": "Garage spaces",
        "Total Bsmt SF": "Total basement size",
        "Year Remod/Add": "Year remodeled",
        "Full Bath": "Full bathrooms",
        "sqft": "sq ft",
    }

    humanized = text
    for source, target in replacements.items():
        humanized = humanized.replace(source, target)

    return " ".join(humanized.split())


def looks_like_raw_feature_fragment(text: str) -> bool:
    lowered_text = text.lower()
    raw_prefixes = [
        "overall quality (",
        "above-ground living area (",
        "kitchen quality (",
        "garage spaces (",
        "total basement size (",
        "year built (",
        "year remodeled (",
        "full bathrooms (",
        "house style (",
    ]
    return any(lowered_text.startswith(prefix) for prefix in raw_prefixes)


def get_house_style_terms(house_style: str) -> list[str]:
    style_map = {
        "1Story": ["1story", "1 story", "1-story", "one story", "one-story"],
        "2Story": ["2story", "2 story", "2-story", "two story", "two-story"],
        "1.5Fin": ["1.5fin", "1.5 fin"],
        "1.5Unf": ["1.5unf", "1.5 unf"],
        "SFoyer": ["sfoyer", "split foyer"],
        "SLvl": ["slvl", "split level", "split-level"],
    }
    return style_map.get(house_style, [house_style.lower()])


def get_price_position(predicted_price: float, quantiles: dict[str, float]) -> str:
    lower_quartile = float(quantiles["0.25"])
    upper_quartile = float(quantiles["0.75"])

    if predicted_price < lower_quartile:
        return "below typical range"
    if predicted_price > upper_quartile:
        return "above typical range"
    return "around typical range"
