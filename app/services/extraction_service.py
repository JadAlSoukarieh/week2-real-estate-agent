from __future__ import annotations

import json
import time
from dataclasses import dataclass
from urllib import error, request

from pydantic import ValidationError

from app.config import (
    API_FEATURES,
    DEFAULT_EXTRACTION_PROMPT_VERSION,
    EXTRACTION_PROMPT_V1_PATH,
    EXTRACTION_PROMPT_V2_PATH,
    HOUSE_STYLE_NORMALIZATION_MAP,
    OLLAMA_API_URL,
    OLLAMA_MAX_RETRIES,
    OLLAMA_MODEL_NAME,
    OLLAMA_RETRY_DELAY_SECONDS,
    OLLAMA_TIMEOUT_SECONDS,
)
from app.schemas import ExtractedPropertyFeatures, ExtractionResponse


class ExtractionServiceUnavailableError(RuntimeError):
    """Raised when the local Ollama service cannot be reached."""


@dataclass
class ExtractionRunResult:
    response: ExtractionResponse
    raw_model_text: str
    used_fallback: bool
    error_type: str | None = None


def extract_features_from_query(
    query: str,
    prompt_version: str = DEFAULT_EXTRACTION_PROMPT_VERSION,
) -> ExtractionResponse:
    return run_extraction(query=query, prompt_version=prompt_version).response


def run_extraction(
    query: str,
    prompt_version: str = DEFAULT_EXTRACTION_PROMPT_VERSION,
) -> ExtractionRunResult:
    prompt_text = load_prompt_text(prompt_version)
    prompt = prompt_text.replace("{query}", query.strip())
    raw_model_text = call_ollama(prompt)

    try:
        parsed_output = parse_model_output(raw_model_text)
    except (json.JSONDecodeError, ValueError):
        return build_fallback_result(
            query=query,
            prompt_version=prompt_version,
            raw_model_text=raw_model_text,
            note="Model returned invalid JSON.",
            error_type="invalid_json",
        )

    try:
        features_model, notes = normalize_model_output(parsed_output)
    except ValidationError:
        return build_fallback_result(
            query=query,
            prompt_version=prompt_version,
            raw_model_text=raw_model_text,
            note="Model response failed schema validation.",
            error_type="schema_validation",
        )
    except (TypeError, ValueError):
        return build_fallback_result(
            query=query,
            prompt_version=prompt_version,
            raw_model_text=raw_model_text,
            note="Model response failed schema validation.",
            error_type="schema_validation",
        )

    response = build_extraction_response(
        query=query,
        features=features_model,
        prompt_version=prompt_version,
        validation_passed=True,
        notes=notes,
    )
    return ExtractionRunResult(
        response=response,
        raw_model_text=raw_model_text,
        used_fallback=False,
        error_type=None,
    )


def load_prompt_text(prompt_version: str) -> str:
    prompt_path = get_prompt_path(prompt_version)
    return prompt_path.read_text(encoding="utf-8")


def get_prompt_path(prompt_version: str):
    if prompt_version == "extraction_v1":
        return EXTRACTION_PROMPT_V1_PATH
    if prompt_version == "extraction_v2":
        return EXTRACTION_PROMPT_V2_PATH
    raise ValueError(f"Unsupported prompt version: {prompt_version}")


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
                raise ExtractionServiceUnavailableError(
                    "Could not reach Ollama service."
                ) from exc
            time.sleep(OLLAMA_RETRY_DELAY_SECONDS)

    try:
        response_payload = json.loads(response_body)
    except json.JSONDecodeError as exc:
        raise ExtractionServiceUnavailableError(
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

    raise ExtractionServiceUnavailableError("Could not reach Ollama service.")


def parse_model_output(raw_model_text: str) -> dict:
    cleaned_text = strip_code_fences(raw_model_text).strip()
    parsed_output = json.loads(cleaned_text)
    if not isinstance(parsed_output, dict):
        raise ValueError("Model output must be a JSON object.")
    return parsed_output


def strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]

    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]

    return cleaned.strip()


def normalize_model_output(parsed_output: dict) -> tuple[ExtractedPropertyFeatures, str | None]:
    notes = normalize_notes(parsed_output.get("notes"))
    if notes is not None and not isinstance(notes, str):
        raise TypeError("notes must be a string when present")

    if "features" in parsed_output:
        feature_payload = parsed_output["features"]
    else:
        feature_payload = parsed_output

    if not isinstance(feature_payload, dict):
        raise TypeError("features payload must be a JSON object")

    normalized_payload = normalize_feature_payload(feature_payload)
    features_model = ExtractedPropertyFeatures.model_validate(normalized_payload)
    return features_model, notes


def normalize_feature_payload(feature_payload: dict) -> dict:
    normalized_payload = dict(feature_payload)
    house_style = normalized_payload.get("house_style")
    if isinstance(house_style, str):
        normalized_payload["house_style"] = normalize_house_style(house_style)
    return normalized_payload


def normalize_house_style(value: str) -> str:
    canonical_values = {
        "1Story",
        "2Story",
        "1.5Fin",
        "1.5Unf",
        "2.5Fin",
        "2.5Unf",
        "SFoyer",
        "SLvl",
    }
    if value in canonical_values:
        return value

    normalized_key = " ".join(value.strip().lower().split())
    normalized_key = normalized_key.replace("  ", " ")
    compact_key = normalized_key.replace(" ", "")

    if normalized_key in HOUSE_STYLE_NORMALIZATION_MAP:
        return HOUSE_STYLE_NORMALIZATION_MAP[normalized_key]
    if compact_key in HOUSE_STYLE_NORMALIZATION_MAP:
        return HOUSE_STYLE_NORMALIZATION_MAP[compact_key]

    return value


def normalize_notes(notes: object) -> str | None:
    if notes is None:
        return None
    if not isinstance(notes, str):
        return notes  # validated by caller

    cleaned_notes = notes.strip()
    return cleaned_notes or None


def build_extraction_response(
    query: str,
    features: ExtractedPropertyFeatures,
    prompt_version: str,
    validation_passed: bool,
    notes: str | None,
) -> ExtractionResponse:
    feature_values = features.model_dump()
    extracted_fields = [
        feature_name
        for feature_name in API_FEATURES
        if feature_values.get(feature_name) is not None
    ]
    missing_fields = [
        feature_name
        for feature_name in API_FEATURES
        if feature_values.get(feature_name) is None
    ]

    return ExtractionResponse(
        query=query,
        features=features,
        extracted_fields=extracted_fields,
        missing_fields=missing_fields,
        is_complete=len(missing_fields) == 0,
        prompt_version=prompt_version,
        validation_passed=validation_passed,
        notes=notes,
    )


def build_fallback_result(
    query: str,
    prompt_version: str,
    raw_model_text: str,
    note: str,
    error_type: str,
) -> ExtractionRunResult:
    response = build_extraction_response(
        query=query,
        features=ExtractedPropertyFeatures(),
        prompt_version=prompt_version,
        validation_passed=False,
        notes=note,
    )
    return ExtractionRunResult(
        response=response,
        raw_model_text=raw_model_text,
        used_fallback=True,
        error_type=error_type,
    )
