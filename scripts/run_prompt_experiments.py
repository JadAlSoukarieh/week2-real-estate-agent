from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.config import API_FEATURES, PROMPT_EVAL_RESULTS_PATH
from app.services.extraction_service import (
    ExtractionServiceUnavailableError,
    run_extraction,
)

PROMPT_VERSIONS = ["extraction_v1", "extraction_v2"]
EXPERIMENT_QUERIES = [
    "Estimate the price of a 1,850 square foot 2-story house in NAmes with a good kitchen, 2 full baths, 2 garage spaces, built in 2004 and remodeled in 2008.",
    "What might a 1-story house in NAmes with a good kitchen and 2-car garage cost?",
    "How much is a nice family home with a lot of space?",
    "Estimate a split level home in Gilbert with 1,600 square feet above grade, 2 full baths, and a fair kitchen.",
    "What is a home in OldTown with an excellent kitchen and 1 garage space worth?",
    "Price a house in NAmes.",
]


def main() -> None:
    all_results = []

    for prompt_version in PROMPT_VERSIONS:
        for query in EXPERIMENT_QUERIES:
            result = run_single_experiment(prompt_version=prompt_version, query=query)
            all_results.append(result)
            print_debug_line(result)

    PROMPT_EVAL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROMPT_EVAL_RESULTS_PATH.write_text(
        json.dumps(all_results, indent=2),
        encoding="utf-8",
    )

    print(f"Saved prompt evaluation results to {PROMPT_EVAL_RESULTS_PATH}")
    print()
    print_summary(all_results)


def run_single_experiment(prompt_version: str, query: str) -> dict:
    try:
        run_result = run_extraction(query=query, prompt_version=prompt_version)
        response = run_result.response
        return {
            "prompt_version": prompt_version,
            "query": query,
            "raw_model_text": run_result.raw_model_text,
            "normalized_features": response.features.model_dump(),
            "validation_passed": response.validation_passed,
            "used_fallback": run_result.used_fallback,
            "extracted_fields": response.extracted_fields,
            "missing_fields": response.missing_fields,
            "extracted_field_count": len(response.extracted_fields),
            "is_complete": response.is_complete,
            "notes": response.notes,
            "error_type": run_result.error_type,
        }
    except ExtractionServiceUnavailableError:
        return {
            "prompt_version": prompt_version,
            "query": query,
            "raw_model_text": "",
            "normalized_features": {feature_name: None for feature_name in API_FEATURES},
            "validation_passed": False,
            "used_fallback": True,
            "extracted_fields": [],
            "missing_fields": API_FEATURES,
            "extracted_field_count": 0,
            "is_complete": False,
            "notes": "Could not reach Ollama service.",
            "error_type": "transport",
        }


def print_debug_line(result: dict) -> None:
    print(
        f"[{result['prompt_version']}] "
        f"extracted={result['extracted_field_count']} "
        f"fallback={result['used_fallback']} "
        f"valid={result['validation_passed']} "
        f"error={result['error_type']}"
    )


def print_summary(results: list[dict]) -> None:
    print("Prompt comparison summary:")
    for prompt_version in PROMPT_VERSIONS:
        prompt_results = [
            result for result in results if result["prompt_version"] == prompt_version
        ]
        total_runs = len(prompt_results)
        validation_pass_count = sum(
            1 for result in prompt_results if result["validation_passed"]
        )
        complete_count = sum(1 for result in prompt_results if result["is_complete"])
        fallback_count = sum(1 for result in prompt_results if result["used_fallback"])
        house_style_count = sum(
            1
            for result in prompt_results
            if result["normalized_features"].get("house_style") is not None
        )
        kitchen_qual_count = sum(
            1
            for result in prompt_results
            if result["normalized_features"].get("kitchen_qual") is not None
        )
        average_extracted = (
            sum(result["extracted_field_count"] for result in prompt_results) / total_runs
            if total_runs
            else 0.0
        )

        print(
            f"- {prompt_version}: "
            f"validation_passed={validation_pass_count}/{total_runs}, "
            f"avg_extracted_fields={average_extracted:.2f}, "
            f"complete={complete_count}, "
            f"fallbacks={fallback_count}, "
            f"house_style_extracted={house_style_count}, "
            f"kitchen_qual_extracted={kitchen_qual_count}"
        )


if __name__ == "__main__":
    main()
