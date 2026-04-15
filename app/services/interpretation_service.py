from __future__ import annotations

import json
import time
from typing import Any
from urllib import error, request

from pydantic import ValidationError

from app.config import (
    DEFAULT_INTERPRETATION_PROMPT_VERSION,
    INTERPRETATION_PROMPT_V1_PATH,
    OLLAMA_API_URL,
    OLLAMA_MAX_RETRIES,
    OLLAMA_MODEL_NAME,
    OLLAMA_RETRY_DELAY_SECONDS,
    OLLAMA_TIMEOUT_SECONDS,
    TRAINING_SUMMARY_PATH,
)
from app.schemas import InterpretationOutput
from app.services.extraction_service import strip_code_fences

_training_summary_cache: dict[str, Any] | None = None


def interpret_prediction(
    query: str,
    final_features: dict[str, Any],
    predicted_price: float,
    prompt_version: str = DEFAULT_INTERPRETATION_PROMPT_VERSION,
) -> InterpretationOutput:
    training_summary = load_training_summary()
    prompt_text = load_prompt_text(prompt_version)
    prompt = prompt_text.replace("{query}", query)
    prompt = prompt.replace(
        "{final_features_json}",
        json.dumps(final_features, indent=2, sort_keys=True),
    )
    prompt = prompt.replace("{predicted_price}", f"{predicted_price:.2f}")
    prompt = prompt.replace(
        "{training_summary_json}",
        json.dumps(training_summary, indent=2, sort_keys=True),
    )

    try:
        raw_model_text = call_ollama(prompt)
        parsed_output = parse_interpretation_output(raw_model_text)
        return InterpretationOutput.model_validate(parsed_output)
    except (InterpretationServiceUnavailableError, json.JSONDecodeError, ValidationError, TypeError, ValueError):
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


def call_ollama(prompt: str) -> str:
    payload = {
        "model": OLLAMA_MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
    }

    request_body = json.dumps(payload).encode("utf-8")
    http_request = request.Request(
        OLLAMA_API_URL,
        data=request_body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    for attempt in range(OLLAMA_MAX_RETRIES + 1):
        try:
            with request.urlopen(http_request, timeout=OLLAMA_TIMEOUT_SECONDS) as response:
                response_body = response.read().decode("utf-8")
            break
        except (error.URLError, TimeoutError, ConnectionError) as exc:
            if attempt == OLLAMA_MAX_RETRIES:
                raise InterpretationServiceUnavailableError(
                    "Could not reach Ollama service."
                ) from exc
            time.sleep(OLLAMA_RETRY_DELAY_SECONDS)

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
        f"The estimated price is about ${predicted_price:,.0f}, which looks "
        f"{price_position} compared with the training distribution."
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


def get_price_position(predicted_price: float, quantiles: dict[str, float]) -> str:
    lower_quartile = float(quantiles["0.25"])
    upper_quartile = float(quantiles["0.75"])

    if predicted_price < lower_quartile:
        return "below typical range"
    if predicted_price > upper_quartile:
        return "above typical range"
    return "around typical range"
