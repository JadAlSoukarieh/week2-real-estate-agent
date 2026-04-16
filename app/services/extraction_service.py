from __future__ import annotations

import json
import logging
import re
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

logger = logging.getLogger(__name__)


class ExtractionServiceUnavailableError(RuntimeError):
    """Raised when the local Ollama service cannot be reached."""


class ExtractionOutputError(RuntimeError):
    """Raised when the model returns malformed or schema-invalid extraction output."""

    def __init__(self, message: str, raw_model_text: str, error_type: str) -> None:
        super().__init__(message)
        self.raw_model_text = raw_model_text
        self.error_type = error_type


@dataclass
class ExtractionRunResult:
    response: ExtractionResponse
    raw_model_text: str
    used_fallback: bool
    used_retry: bool = False
    used_recovery_prompt: bool = False
    error_type: str | None = None


PRIMARY_OUTPUT_RETRIES = 1
RECOVERY_PROMPT_VERSION = "extraction_v2"


def extract_features_from_query(
    query: str,
    prompt_version: str = DEFAULT_EXTRACTION_PROMPT_VERSION,
) -> ExtractionResponse:
    return run_extraction(query=query, prompt_version=prompt_version).response


def run_extraction(
    query: str,
    prompt_version: str = DEFAULT_EXTRACTION_PROMPT_VERSION,
) -> ExtractionRunResult:
    used_retry = False
    last_output_error: ExtractionOutputError | None = None

    for attempt_index in range(PRIMARY_OUTPUT_RETRIES + 1):
        try:
            result = run_single_extraction_attempt(
                query=query,
                prompt_version=prompt_version,
                stage_name="primary",
                attempt_label=f"primary-{attempt_index + 1}",
            )
            result.used_retry = used_retry
            return result
        except ExtractionOutputError as exc:
            last_output_error = exc
            if attempt_index < PRIMARY_OUTPUT_RETRIES:
                used_retry = True
                continue

    if prompt_version != RECOVERY_PROMPT_VERSION:
        try:
            recovery_result = run_single_extraction_attempt(
                query=query,
                prompt_version=RECOVERY_PROMPT_VERSION,
                stage_name="recovery",
                attempt_label="recovery-1",
            )
            recovery_result.used_retry = used_retry
            recovery_result.used_recovery_prompt = True
            return recovery_result
        except ExtractionOutputError as exc:
            last_output_error = exc
            return build_fallback_result(
                query=query,
                prompt_version=prompt_version,
                raw_model_text=exc.raw_model_text,
                note=str(exc),
                error_type=exc.error_type,
                used_retry=used_retry,
                used_recovery_prompt=True,
            )

    return build_fallback_result(
        query=query,
        prompt_version=prompt_version,
        raw_model_text=last_output_error.raw_model_text if last_output_error else "",
        note=str(last_output_error) if last_output_error else "Model returned invalid JSON.",
        error_type=last_output_error.error_type if last_output_error else "invalid_json",
        used_retry=used_retry,
        used_recovery_prompt=False,
    )


def run_single_extraction_attempt(
    query: str,
    prompt_version: str,
    stage_name: str,
    attempt_label: str,
) -> ExtractionRunResult:
    start_time = time.perf_counter()
    prompt_text = load_prompt_text(prompt_version)
    prompt = prompt_text.replace("{query}", query.strip())
    raw_model_text = call_ollama(
        prompt=prompt,
        response_schema=get_extraction_response_schema(),
    )

    try:
        parsed_output = parse_model_output(raw_model_text)
    except (json.JSONDecodeError, ValueError):
        duration = time.perf_counter() - start_time
        logger.info(
            "[extract] stage=%s attempt=%s prompt=%s duration=%.2fs outcome=invalid_json",
            stage_name,
            attempt_label,
            prompt_version,
            duration,
        )
        raise ExtractionOutputError(
            "Model returned invalid JSON.",
            raw_model_text=raw_model_text,
            error_type="invalid_json",
        )

    try:
        features_model, notes = normalize_model_output(parsed_output)
    except (ValidationError, TypeError, ValueError):
        duration = time.perf_counter() - start_time
        logger.info(
            "[extract] stage=%s attempt=%s prompt=%s duration=%.2fs outcome=schema_validation",
            stage_name,
            attempt_label,
            prompt_version,
            duration,
        )
        raise ExtractionOutputError(
            "Model response failed schema validation.",
            raw_model_text=raw_model_text,
            error_type="schema_validation",
        )

    features_model = apply_explicit_query_fallbacks(
        features=features_model,
        query=query,
    )

    duration = time.perf_counter() - start_time
    logger.info(
        "[extract] stage=%s attempt=%s prompt=%s duration=%.2fs outcome=success",
        stage_name,
        attempt_label,
        prompt_version,
        duration,
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


def get_extraction_response_schema() -> dict:
    nullable_numeric = {"anyOf": [{"type": "integer"}, {"type": "null"}]}
    nullable_number = {"anyOf": [{"type": "number"}, {"type": "null"}]}
    nullable_string = {"anyOf": [{"type": "string"}, {"type": "null"}]}

    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "features": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "overall_qual": nullable_numeric,
                    "gr_liv_area": nullable_number,
                    "neighborhood": nullable_string,
                    "kitchen_qual": {
                        "anyOf": [
                            {"enum": ["Po", "Fa", "TA", "Gd", "Ex"]},
                            {"type": "null"},
                        ]
                    },
                    "garage_cars": nullable_numeric,
                    "total_bsmt_sf": nullable_number,
                    "year_built": nullable_numeric,
                    "year_remod_add": nullable_numeric,
                    "full_bath": nullable_numeric,
                    "house_style": nullable_string,
                },
                "required": API_FEATURES,
            },
            "notes": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        },
        "required": ["features", "notes"],
    }


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


def apply_explicit_query_fallbacks(
    features: ExtractedPropertyFeatures,
    query: str,
) -> ExtractedPropertyFeatures:
    feature_payload = features.model_dump()

    if feature_payload.get("overall_qual") is None:
        overall_qual = extract_overall_qual_from_query(query)
        if overall_qual is not None:
            feature_payload["overall_qual"] = overall_qual

    if feature_payload.get("house_style") is None:
        house_style = extract_house_style_from_query(query)
        if house_style is not None:
            feature_payload["house_style"] = house_style

    return ExtractedPropertyFeatures.model_validate(feature_payload)


def extract_overall_qual_from_query(query: str) -> int | None:
    normalized_query = " ".join(query.strip().lower().split())
    if not normalized_query:
        return None

    patterns = [
        r"\boverall quality\s*(?:is\s*)?([1-9]|10)\s*(?:out of 10|/10)\b",
        r"\boverall quality\s*(?:score\s*)?([1-9]|10)\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, normalized_query)
        if match is None:
            continue
        value = int(match.group(1))
        if 1 <= value <= 10:
            return value

    return None


def extract_house_style_from_query(query: str) -> str | None:
    normalized_query = " ".join(query.strip().lower().split())
    if not normalized_query:
        return None

    explicit_phrase_order = [
        ("split foyer", "SFoyer"),
        ("split-level", "SLvl"),
        ("split level", "SLvl"),
        ("1-story", "1Story"),
        ("1 story", "1Story"),
        ("one-story", "1Story"),
        ("one story", "1Story"),
        ("2-story", "2Story"),
        ("2 story", "2Story"),
        ("two-story", "2Story"),
        ("two story", "2Story"),
    ]

    for phrase, canonical_value in explicit_phrase_order:
        if phrase in normalized_query:
            return canonical_value

    return None


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
    used_retry: bool = False,
    used_recovery_prompt: bool = False,
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
        used_retry=used_retry,
        used_recovery_prompt=used_recovery_prompt,
        error_type=error_type,
    )
