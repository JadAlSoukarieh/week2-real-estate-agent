from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st

st.set_page_config(
    page_title="AI Real Estate Agent",
    layout="wide",
)

BACKEND_BASE_URL = os.getenv("BACKEND_BASE_URL", "http://127.0.0.1:8000")
REQUEST_TIMEOUT_SECONDS = 400

FEATURE_ORDER = [
    "overall_qual",
    "gr_liv_area",
    "neighborhood",
    "kitchen_qual",
    "garage_cars",
    "total_bsmt_sf",
    "year_built",
    "year_remod_add",
    "full_bath",
    "house_style",
]

FIELD_LABELS = {
    "overall_qual": "Overall Quality",
    "gr_liv_area": "Above Ground Living Area",
    "neighborhood": "Neighborhood",
    "kitchen_qual": "Kitchen Quality",
    "garage_cars": "Garage Spaces",
    "total_bsmt_sf": "Total Basement Square Feet",
    "year_built": "Year Built",
    "year_remod_add": "Year Remodeled",
    "full_bath": "Full Bathrooms",
    "house_style": "House Style",
}

NUMERIC_FIELDS = {
    "overall_qual": int,
    "gr_liv_area": float,
    "garage_cars": int,
    "total_bsmt_sf": float,
    "year_built": int,
    "year_remod_add": int,
    "full_bath": int,
}

OVERALL_QUAL_OPTIONS = [""] + [str(value) for value in range(1, 11)]
KITCHEN_QUAL_OPTIONS = ["", "Po", "Fa", "TA", "Gd", "Ex"]
HOUSE_STYLE_OPTIONS = ["", "1Story", "2Story", "1.5Fin", "1.5Unf", "SFoyer", "SLvl"]

FIELD_HELP_TEXT = {
    "overall_qual": "1–10 scale. 5 = average, 7 = good, 10 = excellent.",
    "gr_liv_area": "Above-ground living area in square feet, e.g. 1200, 1850, 2400.",
    "garage_cars": "Garage capacity in cars, e.g. 0, 1, 2, 3.",
    "total_bsmt_sf": "Total basement size in square feet, e.g. 800, 1100, 1500.",
    "year_built": "Four-digit year, e.g. 1978 or 2003.",
    "year_remod_add": "Most recent remodel year, e.g. 1998 or 2015.",
    "full_bath": "Whole bathrooms only, e.g. 1, 2, 3.",
}

FIELD_PLACEHOLDERS = {
    "gr_liv_area": "Example: 1850",
    "garage_cars": "Example: 2",
    "total_bsmt_sf": "Example: 1100",
    "year_built": "Example: 2003",
    "year_remod_add": "Example: 2008",
    "full_bath": "Example: 2",
}

DISPLAY_KITCHEN_QUAL = {
    "Po": "Poor (Po)",
    "Fa": "Fair (Fa)",
    "TA": "Typical/Average (TA)",
    "Gd": "Good (Gd)",
    "Ex": "Excellent (Ex)",
}

DISPLAY_HOUSE_STYLE = {
    "1Story": "1 Story",
    "2Story": "2 Story",
    "1.5Fin": "1.5 Story Finished",
    "1.5Unf": "1.5 Story Unfinished",
    "SFoyer": "Split Foyer",
    "SLvl": "Split Level",
}

EXAMPLE_QUERIES = {
    "Partial 1Story": "What might a 1-story house in NAmes with a good kitchen and 2-car garage cost?",
    "Detailed 2Story": (
        "Estimate the price of a 1,850 square foot 2-story house in NAmes "
        "with a good kitchen, 2 full baths, 2 garage spaces, built in 2004 "
        "and remodeled in 2008."
    ),
    "Split Level": "Estimate a split level home in Gilbert with 1,600 square feet above grade, 2 full baths, and a fair kitchen.",
    "Excellent Kitchen": "What is a home in OldTown with an excellent kitchen and 1 garage space worth?",
}


def inject_custom_css() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background:
                    radial-gradient(circle at top right, rgba(41, 127, 185, 0.18), transparent 32%),
                    linear-gradient(180deg, #0b1220 0%, #111827 100%);
                color: #e5eef8;
            }
            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
                max-width: 1200px;
            }
            .hero-card, .panel-card, .result-card, .incomplete-card, .footer-card {
                border-radius: 24px;
                padding: 1.5rem 1.6rem;
                border: 1px solid rgba(148, 163, 184, 0.14);
                box-shadow: 0 18px 40px rgba(2, 6, 23, 0.28);
                background: rgba(15, 23, 42, 0.84);
                backdrop-filter: blur(8px);
            }
            .hero-card {
                padding: 2rem 2rem 1.6rem 2rem;
                margin-bottom: 1rem;
            }
            .hero-title {
                font-size: 2.4rem;
                line-height: 1.1;
                font-weight: 700;
                color: #f8fafc;
                margin-bottom: 0.35rem;
            }
            .hero-subtitle {
                color: #cbd5e1;
                font-size: 1.05rem;
                max-width: 48rem;
                margin-bottom: 1rem;
            }
            .status-chip, .field-chip, .missing-chip, .info-chip {
                display: inline-block;
                padding: 0.4rem 0.75rem;
                border-radius: 999px;
                font-size: 0.82rem;
                font-weight: 600;
                margin-right: 0.5rem;
                margin-bottom: 0.45rem;
                border: 1px solid rgba(148, 163, 184, 0.16);
            }
            .status-healthy {
                color: #7dd3fc;
                background: rgba(8, 47, 73, 0.85);
            }
            .status-unhealthy {
                color: #fecaca;
                background: rgba(69, 10, 10, 0.88);
            }
            .field-chip {
                color: #dbeafe;
                background: rgba(15, 118, 110, 0.18);
            }
            .missing-chip {
                color: #fde68a;
                background: rgba(120, 53, 15, 0.32);
            }
            .info-chip {
                color: #bfdbfe;
                background: rgba(30, 64, 175, 0.28);
            }
            .section-heading {
                font-size: 1.15rem;
                font-weight: 700;
                color: #f8fafc;
                margin-bottom: 0.6rem;
            }
            .section-copy {
                color: #cbd5e1;
                margin-bottom: 0.85rem;
            }
            .feature-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                gap: 0.85rem;
                margin-top: 0.8rem;
            }
            .feature-card {
                background: rgba(30, 41, 59, 0.9);
                border: 1px solid rgba(148, 163, 184, 0.12);
                border-radius: 18px;
                padding: 0.9rem 1rem;
            }
            .feature-label {
                color: #94a3b8;
                font-size: 0.82rem;
                margin-bottom: 0.25rem;
            }
            .feature-value {
                color: #f8fafc;
                font-size: 1rem;
                font-weight: 600;
            }
            .result-card {
                background:
                    linear-gradient(135deg, rgba(14, 116, 144, 0.2), rgba(15, 23, 42, 0.98)),
                    rgba(15, 23, 42, 0.92);
                border: 1px solid rgba(103, 232, 249, 0.2);
            }
            .result-kicker {
                text-transform: uppercase;
                letter-spacing: 0.08em;
                color: #7dd3fc;
                font-size: 0.78rem;
                font-weight: 700;
                margin-bottom: 0.5rem;
            }
            .result-price {
                font-size: 3rem;
                line-height: 1;
                font-weight: 800;
                color: #f8fafc;
                margin-bottom: 0.55rem;
            }
            .result-copy {
                color: #dbeafe;
                font-size: 1rem;
            }
            .incomplete-card {
                background:
                    linear-gradient(135deg, rgba(120, 53, 15, 0.22), rgba(15, 23, 42, 0.98)),
                    rgba(15, 23, 42, 0.9);
                border: 1px solid rgba(251, 191, 36, 0.18);
            }
            .list-block {
                margin-top: 0.65rem;
                color: #dbeafe;
            }
            .list-block li {
                margin-bottom: 0.35rem;
            }
            .footer-note {
                color: #94a3b8;
                font-size: 0.92rem;
            }
            .helper-copy {
                color: #8ca3ba;
                font-size: 0.82rem;
                margin-top: -0.35rem;
                margin-bottom: 0.55rem;
            }
            div[data-testid="stForm"] {
                border: 1px solid rgba(148, 163, 184, 0.14);
                background: rgba(15, 23, 42, 0.82);
                border-radius: 24px;
                padding: 1rem 1rem 0.25rem 1rem;
            }
            div.stButton > button, div.stForm button {
                border-radius: 999px;
                border: 1px solid rgba(56, 189, 248, 0.25);
                background: linear-gradient(135deg, #0369a1, #0f766e);
                color: white;
                font-weight: 600;
                padding: 0.55rem 1rem;
            }
            div.stButton > button:hover, div.stForm button:hover {
                border-color: rgba(125, 211, 252, 0.42);
                color: white;
            }
            div[data-testid="stMarkdownContainer"] p {
                color: #d6e2ee;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_session_state() -> None:
    defaults = {
        "query_text": "",
        "selected_example": None,
        "analysis_response": None,
        "analysis_error": None,
        "override_baseline": {field: None for field in FEATURE_ORDER},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    for field in FEATURE_ORDER:
        key = override_widget_key(field)
        if key not in st.session_state:
            st.session_state[key] = default_widget_value(field, None)


def override_widget_key(field_name: str) -> str:
    return f"override__{field_name}"


@st.cache_data(ttl=15, show_spinner=False)
def fetch_backend_health() -> dict[str, Any]:
    try:
        response = requests.get(
            f"{BACKEND_BASE_URL}/health",
            timeout=5,
        )
        response.raise_for_status()
        payload = response.json()
        return {
            "ok": True,
            "status": payload.get("status", "ok"),
            "detail": "Backend connected",
        }
    except requests.RequestException:
        return {
            "ok": False,
            "status": "offline",
            "detail": "Backend unavailable",
        }


def call_analyze_query(query: str, overrides: dict[str, Any] | None) -> tuple[dict[str, Any] | None, str | None]:
    payload: dict[str, Any] = {"query": query}
    if overrides:
        payload["overrides"] = overrides

    try:
        response = requests.post(
            f"{BACKEND_BASE_URL}/analyze-query",
            json=payload,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except requests.Timeout:
        return None, "The backend took too long to respond. Please try again."
    except requests.RequestException:
        return None, "Could not reach the backend. Make sure FastAPI is running."

    try:
        body = response.json()
    except ValueError:
        body = {}

    if response.status_code >= 400:
        detail = body.get("detail") if isinstance(body, dict) else None
        return None, detail or "The backend returned an error while analyzing the property."

    return body, None


def normalize_analyze_response(raw_response: dict[str, Any] | None) -> dict[str, Any]:
    empty_features = {field: None for field in FEATURE_ORDER}
    if not isinstance(raw_response, dict):
        return {
            "query": "",
            "extraction": {
                "prompt_version": None,
                "features": empty_features,
                "extracted_fields": [],
                "missing_fields": FEATURE_ORDER.copy(),
                "is_complete": False,
            },
            "final_features": empty_features,
            "missing_fields_after_overrides": FEATURE_ORDER.copy(),
            "is_ready_for_prediction": False,
            "prediction_ran": False,
            "predicted_price": None,
            "interpretation": None,
            "notes": None,
        }

    extraction_block = raw_response.get("extraction") or {}
    extraction_features = extraction_block.get("features") or {}
    final_features = raw_response.get("final_features") or {}
    interpretation = raw_response.get("interpretation")

    normalized_extraction_features = {
        field: extraction_features.get(field)
        for field in FEATURE_ORDER
    }
    normalized_final_features = {
        field: final_features.get(field)
        for field in FEATURE_ORDER
    }

    normalized_interpretation = None
    if isinstance(interpretation, dict):
        normalized_interpretation = {
            "summary": interpretation.get("summary"),
            "price_position": interpretation.get("price_position"),
            "key_drivers": interpretation.get("key_drivers") or [],
            "caveats": interpretation.get("caveats") or [],
        }

    extracted_fields = extraction_block.get("extracted_fields")
    if not isinstance(extracted_fields, list):
        extracted_fields = []

    extraction_missing_fields = extraction_block.get("missing_fields")
    if not isinstance(extraction_missing_fields, list):
        extraction_missing_fields = FEATURE_ORDER.copy()

    missing_fields_after_overrides = raw_response.get("missing_fields_after_overrides")
    if not isinstance(missing_fields_after_overrides, list):
        missing_fields_after_overrides = []

    return {
        "query": raw_response.get("query", ""),
        "extraction": {
            "prompt_version": extraction_block.get("prompt_version"),
            "features": normalized_extraction_features,
            "extracted_fields": extracted_fields,
            "missing_fields": extraction_missing_fields,
            "is_complete": bool(extraction_block.get("is_complete", False)),
        },
        "final_features": normalized_final_features,
        "missing_fields_after_overrides": missing_fields_after_overrides,
        "is_ready_for_prediction": bool(raw_response.get("is_ready_for_prediction", False)),
        "prediction_ran": bool(raw_response.get("prediction_ran", False)),
        "predicted_price": raw_response.get("predicted_price"),
        "interpretation": normalized_interpretation,
        "notes": raw_response.get("notes"),
    }


def format_price(value: float | None) -> str:
    if value is None:
        return "—"
    return f"${value:,.0f}"


def format_feature_value(field: str, value: Any) -> str:
    if value is None:
        return "—"

    if field in {"gr_liv_area", "total_bsmt_sf"}:
        numeric = float(value)
        return f"{numeric:,.0f} sq ft"

    if field == "house_style":
        return DISPLAY_HOUSE_STYLE.get(str(value), str(value))

    if field == "kitchen_qual":
        return DISPLAY_KITCHEN_QUAL.get(str(value), str(value))

    if field == "overall_qual":
        normalized = normalize_for_form_value(value)
        return f"{normalized} / 10"

    if field in {"garage_cars", "full_bath"}:
        normalized = normalize_for_form_value(value)
        return str(normalized)

    if field in {"year_built", "year_remod_add"}:
        normalized = normalize_for_form_value(value)
        return str(normalized)

    return str(normalize_for_form_value(value))


def parse_optional_numeric(field_name: str, raw_value: str) -> tuple[int | float | None, str | None]:
    cleaned = raw_value.strip()
    if not cleaned:
        return None, None

    parser = NUMERIC_FIELDS[field_name]
    try:
        parsed = parser(cleaned)
    except ValueError:
        return None, f"{FIELD_LABELS[field_name]} must be a valid number."

    return parsed, None


def normalize_for_form_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def default_widget_value(field_name: str, value: Any) -> str:
    if value is None:
        return ""
    if field_name in NUMERIC_FIELDS:
        normalized = normalize_for_form_value(value)
        return str(normalized)
    return str(value)


def get_prefill_source(normalized_response: dict[str, Any] | None) -> dict[str, Any]:
    if not normalized_response:
        return {field: None for field in FEATURE_ORDER}

    final_features = normalized_response.get("final_features") or {}
    if any(final_features.get(field) is not None for field in FEATURE_ORDER):
        return {field: final_features.get(field) for field in FEATURE_ORDER}

    extraction_features = normalized_response.get("extraction", {}).get("features") or {}
    return {field: extraction_features.get(field) for field in FEATURE_ORDER}


def sync_override_state_from_response(normalized_response: dict[str, Any] | None) -> None:
    baseline = get_prefill_source(normalized_response)
    if baseline == st.session_state.get("override_baseline"):
        return

    st.session_state["override_baseline"] = baseline
    for field in FEATURE_ORDER:
        st.session_state[override_widget_key(field)] = default_widget_value(field, baseline.get(field))


def build_override_payload() -> tuple[dict[str, Any] | None, list[str]]:
    baseline = st.session_state.get("override_baseline", {})
    overrides: dict[str, Any] = {}
    errors: list[str] = []

    for field in FEATURE_ORDER:
        widget_value = st.session_state.get(override_widget_key(field), "")
        baseline_value = normalize_for_form_value(baseline.get(field))

        if field in NUMERIC_FIELDS:
            parsed_value, error_message = parse_optional_numeric(field, widget_value)
            if error_message:
                errors.append(error_message)
                continue
            if parsed_value is None:
                continue
            if normalize_for_form_value(parsed_value) != baseline_value:
                overrides[field] = parsed_value
            continue

        cleaned_value = str(widget_value).strip()
        if not cleaned_value:
            continue
        if cleaned_value != str(baseline_value) if baseline_value is not None else cleaned_value != "":
            overrides[field] = cleaned_value

    return overrides or None, errors


def render_header(health: dict[str, Any]) -> None:
    status_class = "status-healthy" if health["ok"] else "status-unhealthy"
    status_label = "Backend Ready" if health["ok"] else "Backend Offline"
    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-title">AI Real Estate Agent</div>
            <div class="hero-subtitle">
                Describe a property in plain English, review the extracted features,
                fill any missing details, and get a chained estimate with concise AI interpretation.
            </div>
            <span class="status-chip {status_class}">{status_label}</span>
            <span class="info-chip">{health['detail']}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_example_queries() -> None:
    st.markdown('<div class="section-heading">Quick Demo Prompts</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Use one of these to speed up a live walkthrough. They only prefill the text area.</div>',
        unsafe_allow_html=True,
    )
    columns = st.columns(len(EXAMPLE_QUERIES))
    for column, (label, query) in zip(columns, EXAMPLE_QUERIES.items()):
        with column:
            if st.button(label, use_container_width=True):
                st.session_state["query_text"] = query
                st.session_state["selected_example"] = label


def render_query_form() -> None:
    st.markdown('<div class="section-heading">Describe the Property</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Write the property the way a buyer or seller would describe it. The app will pull out the details it can, highlight anything missing, and generate an estimate when enough information is available.</div>',
        unsafe_allow_html=True,
    )
    with st.form("query_form"):
        query_text = st.text_area(
            "Property description",
            key="query_text",
            height=140,
            placeholder="Example: What might a 1-story house in NAmes with a good kitchen and 2-car garage cost?",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Analyze Property", use_container_width=True)

    if submitted:
        cleaned_query = query_text.strip()
        if not cleaned_query:
            st.session_state["analysis_error"] = "Please enter a property description before running the analysis."
            return

        with st.spinner("Analyzing the property through the full chain..."):
            raw_response, error_message = call_analyze_query(cleaned_query, overrides=None)

        if error_message:
            st.session_state["analysis_error"] = error_message
            return

        normalized_response = normalize_analyze_response(raw_response)
        st.session_state["analysis_response"] = normalized_response
        st.session_state["analysis_error"] = None
        sync_override_state_from_response(normalized_response)


def render_extraction_section(normalized_response: dict[str, Any]) -> None:
    extraction = normalized_response["extraction"]
    features = extraction["features"]
    extracted_fields = extraction["extracted_fields"]
    missing_fields = extraction["missing_fields"]
    readiness_class = "status-healthy" if normalized_response["is_ready_for_prediction"] else "status-unhealthy"
    readiness_label = "Ready for prediction" if normalized_response["is_ready_for_prediction"] else "Missing details"

    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Extraction Review</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Here is what the app understood from your description before any edits or additions.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<span class="status-chip {readiness_class}">{readiness_label}</span>',
        unsafe_allow_html=True,
    )

    if extracted_fields:
        render_feature_cards(
            {field: features.get(field) for field in extracted_fields if features.get(field) is not None}
        )
    else:
        st.markdown(
            '<div class="section-copy">No property details were confidently extracted yet. Use the form below to fill in the profile.</div>',
            unsafe_allow_html=True,
        )

    if missing_fields:
        st.markdown(
            '<div class="section-copy" style="margin-top: 1rem;">Still missing</div>',
            unsafe_allow_html=True,
        )
        render_missing_field_chips(missing_fields)
        st.markdown(
            '<div class="helper-copy">Need help? The form below shows examples and expected formats for each missing field.</div>',
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)


def render_override_form(normalized_response: dict[str, Any] | None) -> None:
    st.markdown('<div class="section-heading">Review or Fill Missing Details</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">Only the changes you make here are applied. Leaving a field blank keeps the current value unchanged.</div>',
        unsafe_allow_html=True,
    )

    with st.form("override_form"):
        columns = st.columns(2)
        for index, field in enumerate(FEATURE_ORDER):
            column = columns[index % 2]
            with column:
                render_override_widget(field)

        submitted = st.form_submit_button("Apply Overrides and Reanalyze", use_container_width=True)

    if submitted:
        query = st.session_state.get("query_text", "").strip()
        if not query:
            st.warning("Enter or select a property query before applying overrides.")
            return

        overrides, errors = build_override_payload()
        if errors:
            for error_message in errors:
                st.warning(error_message)
            return

        with st.spinner("Reanalyzing with your explicit overrides..."):
            raw_response, error_message = call_analyze_query(query, overrides=overrides)

        if error_message:
            st.session_state["analysis_error"] = error_message
            return

        normalized = normalize_analyze_response(raw_response)
        st.session_state["analysis_response"] = normalized
        st.session_state["analysis_error"] = None
        st.rerun()


def render_override_widget(field: str) -> None:
    key = override_widget_key(field)
    label = FIELD_LABELS[field]

    if field == "overall_qual":
        current_value = st.session_state.get(key, "")
        if current_value not in OVERALL_QUAL_OPTIONS:
            current_value = ""
            st.session_state[key] = current_value
        st.selectbox(label, OVERALL_QUAL_OPTIONS, key=key)
        st.markdown(
            f'<div class="helper-copy">{FIELD_HELP_TEXT[field]}</div>',
            unsafe_allow_html=True,
        )
    elif field in NUMERIC_FIELDS:
        st.text_input(
            label,
            key=key,
            placeholder=FIELD_PLACEHOLDERS.get(field, "Leave blank to keep the extracted value"),
        )
        helper_text = FIELD_HELP_TEXT.get(field)
        if helper_text:
            st.markdown(
                f'<div class="helper-copy">{helper_text}</div>',
                unsafe_allow_html=True,
            )
    elif field == "kitchen_qual":
        current_value = st.session_state.get(key, "")
        if current_value not in KITCHEN_QUAL_OPTIONS:
            current_value = ""
            st.session_state[key] = current_value
        st.selectbox(label, KITCHEN_QUAL_OPTIONS, key=key)
    elif field == "house_style":
        current_value = st.session_state.get(key, "")
        if current_value not in HOUSE_STYLE_OPTIONS:
            current_value = ""
            st.session_state[key] = current_value
        st.selectbox(label, HOUSE_STYLE_OPTIONS, key=key)
    else:
        st.text_input(label, key=key, placeholder="Leave blank to keep the extracted value")


def render_result_section(normalized_response: dict[str, Any]) -> None:
    if normalized_response["prediction_ran"]:
        render_complete_result(normalized_response)
    else:
        render_incomplete_state(normalized_response)


def render_incomplete_state(normalized_response: dict[str, Any]) -> None:
    missing_fields = normalized_response["missing_fields_after_overrides"]
    st.markdown(
        f"""
        <div class="incomplete-card">
            <div class="section-heading">Not Ready for Prediction Yet</div>
            <div class="section-copy">
                The estimate is not ready yet because a few important property details are still missing.
                Add the remaining information below to continue.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    render_missing_field_chips(missing_fields)
    st.markdown(
        '<div class="helper-copy">Need help? The form above shows examples and expected formats for each missing field.</div>',
        unsafe_allow_html=True,
    )


def render_complete_result(normalized_response: dict[str, Any]) -> None:
    price = format_price(normalized_response["predicted_price"])
    interpretation = normalized_response["interpretation"] or {}
    key_drivers = interpretation.get("key_drivers") or []
    caveats = interpretation.get("caveats") or []

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-kicker">Estimated Price</div>
            <div class="result-price">{price}</div>
            <div class="result-copy">Model-based estimate from the provided feature set.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left, right = st.columns([1.35, 1])
    with left:
        st.markdown('<div class="panel-card">', unsafe_allow_html=True)
        st.markdown('<div class="section-heading">Interpretation</div>', unsafe_allow_html=True)
        if interpretation.get("price_position"):
            st.markdown(
                f'<span class="field-chip">{interpretation["price_position"]}</span>',
                unsafe_allow_html=True,
            )
        if interpretation.get("summary"):
            st.markdown(
                f'<div class="section-copy" style="margin-top: 0.9rem;">{interpretation["summary"]}</div>',
                unsafe_allow_html=True,
            )
        if key_drivers:
            st.markdown('<div class="section-copy" style="margin-top: 1rem;">Key drivers</div>', unsafe_allow_html=True)
            st.markdown(
                "<ul class='list-block'>" + "".join(f"<li>{driver}</li>" for driver in key_drivers) + "</ul>",
                unsafe_allow_html=True,
            )
        if caveats:
            st.markdown('<div class="section-copy" style="margin-top: 1rem;">Caveats</div>', unsafe_allow_html=True)
            st.markdown(
                "<ul class='list-block'>" + "".join(f"<li>{item}</li>" for item in caveats) + "</ul>",
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        render_final_features_panel(normalized_response["final_features"])


def render_final_features_panel(final_features: dict[str, Any]) -> None:
    st.markdown('<div class="panel-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Final Features Used</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">These are the normalized property details used for the final estimate.</div>',
        unsafe_allow_html=True,
    )
    render_feature_cards(
        {field: final_features.get(field) for field in FEATURE_ORDER if final_features.get(field) is not None}
    )
    st.markdown("</div>", unsafe_allow_html=True)


def render_feature_cards(feature_map: dict[str, Any]) -> None:
    items = list(feature_map.items())
    if not items:
        return

    for start in range(0, len(items), 4):
        row_items = items[start : start + 4]
        columns = st.columns(len(row_items))
        for column, (field, value) in zip(columns, row_items):
            with column:
                st.markdown(
                    f"""
                    <div class="feature-card">
                        <div class="feature-label">{FIELD_LABELS[field]}</div>
                        <div class="feature-value">{format_feature_value(field, value)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_missing_field_chips(fields: list[str]) -> None:
    if not fields:
        return

    chip_html = " ".join(
        f'<span class="missing-chip">{FIELD_LABELS[field]}</span>'
        for field in fields
    )
    st.markdown(chip_html, unsafe_allow_html=True)


def render_errors() -> None:
    if st.session_state.get("analysis_error"):
        st.error(st.session_state["analysis_error"])


def render_footer() -> None:
    st.markdown(
        """
        <div class="footer-card">
            <div class="footer-note">
                This estimate is based only on the details provided and the project’s trained pricing model.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def clear_demo_state() -> None:
    st.session_state["query_text"] = ""
    st.session_state["selected_example"] = None
    st.session_state["analysis_response"] = None
    st.session_state["analysis_error"] = None
    st.session_state["override_baseline"] = {field: None for field in FEATURE_ORDER}
    for field in FEATURE_ORDER:
        st.session_state[override_widget_key(field)] = default_widget_value(field, None)


def main() -> None:
    inject_custom_css()
    init_session_state()

    health = fetch_backend_health()
    render_header(health)

    top_controls = st.columns([4, 1])
    with top_controls[1]:
        if st.button("Reset Demo", use_container_width=True):
            clear_demo_state()
            st.rerun()

    render_example_queries()
    st.write("")
    render_query_form()
    render_errors()

    normalized_response = st.session_state.get("analysis_response")
    sync_override_state_from_response(normalized_response)

    if normalized_response:
        st.write("")
        render_extraction_section(normalized_response)
        st.write("")

    render_override_form(normalized_response)

    if normalized_response:
        st.write("")
        render_result_section(normalized_response)

    st.write("")
    render_footer()


if __name__ == "__main__":
    main()
