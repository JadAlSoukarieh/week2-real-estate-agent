from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.config import (
    DEFAULT_EXTRACTION_PROMPT_VERSION,
    MODEL_ARTIFACT_PATH,
    MODEL_NAME,
    TARGET_TRANSFORM,
    USED_FEATURES_FOR_RESPONSE,
)
from app.schemas import (
    ChainedQueryInput,
    ChainedQueryResponse,
    ErrorResponse,
    ExtractionResponse,
    PredictionResponse,
    PropertyFeaturesInput,
    QueryInput,
)
from app.services.chain_service import ChainServiceError, analyze_query
from app.services.extraction_service import (
    ExtractionServiceUnavailableError,
    extract_features_from_query,
)
from app.services.interpretation_service import load_training_summary
from app.services.prediction_service import load_model, predict_from_features


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_model()
    load_training_summary()
    yield


app = FastAPI(
    title="AI Real Estate Agent API",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health_check() -> dict[str, str | bool]:
    return {
        "status": "ok",
        "model_loaded": True,
        "model_artifact_path": str(MODEL_ARTIFACT_PATH),
    }


@app.post(
    "/predict-features",
    response_model=PredictionResponse,
    responses={
        503: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
def predict_features(payload: PropertyFeaturesInput) -> PredictionResponse:
    try:
        predicted_price = predict_from_features(payload)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed due to an internal server error.",
        ) from exc

    return PredictionResponse(
        predicted_price=predicted_price,
        currency="USD",
        model_name=MODEL_NAME,
        target_transform=TARGET_TRANSFORM,
        used_features=USED_FEATURES_FOR_RESPONSE,
    )


@app.post(
    "/extract-features",
    response_model=ExtractionResponse,
    responses={
        503: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
def extract_features(payload: QueryInput) -> ExtractionResponse:
    try:
        return extract_features_from_query(
            query=payload.query,
            prompt_version=DEFAULT_EXTRACTION_PROMPT_VERSION,
        )
    except ExtractionServiceUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Feature extraction failed due to an internal server error.",
        ) from exc


@app.post(
    "/analyze-query",
    response_model=ChainedQueryResponse,
    responses={
        503: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
def analyze_query_endpoint(payload: ChainedQueryInput) -> ChainedQueryResponse:
    try:
        return analyze_query(payload)
    except ExtractionServiceUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ChainServiceError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Query analysis failed due to an internal server error.",
        ) from exc
