from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.schemas import ChainedQueryInput, FeatureOverridesInput
from app.services.chain_service import analyze_query
from app.services.extraction_service import ExtractionServiceUnavailableError


def main() -> None:
    scenarios = [
        (
            "incomplete_query",
            ChainedQueryInput(
                query="What might a 1-story house in NAmes with a good kitchen cost?"
            ),
        ),
        (
            "query_plus_overrides",
            ChainedQueryInput(
                query="What might a 1-story house in NAmes with a good kitchen and 2-car garage cost?",
                overrides=FeatureOverridesInput(
                    overall_qual=7,
                    gr_liv_area=1710,
                    total_bsmt_sf=1080,
                    year_built=2003,
                    year_remod_add=2003,
                    full_bath=2,
                ),
            ),
        ),
        (
            "mostly_complete_query",
            ChainedQueryInput(
                query=(
                    "Estimate the price of a 1,850 square foot 2-story house in NAmes "
                    "with a good kitchen, 2 full baths, 2 garage spaces, built in 2004 "
                    "and remodeled in 2008."
                ),
                overrides=FeatureOverridesInput(
                    overall_qual=8,
                    total_bsmt_sf=1200,
                ),
            ),
        ),
    ]

    for name, payload in scenarios:
        print(f"[{name}]")
        try:
            response = analyze_query(payload)
        except ExtractionServiceUnavailableError as exc:
            print(f"  extraction_failed={exc}")
            continue

        print(f"  extraction_complete={response.extraction.is_complete}")
        print(f"  missing_fields={response.missing_fields_after_overrides}")
        print(f"  prediction_ran={response.prediction_ran}")
        if response.predicted_price is not None:
            print(f"  predicted_price=${response.predicted_price:,.2f}")
        if response.interpretation is not None:
            print(f"  interpretation_summary={response.interpretation.summary}")
        print()


if __name__ == "__main__":
    main()
