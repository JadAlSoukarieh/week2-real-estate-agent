from pathlib import Path

TARGET_COLUMN = "SalePrice"

SELECTED_FEATURES = [
    "Overall Qual",
    "Gr Liv Area",
    "Neighborhood",
    "Kitchen Qual",
    "Garage Cars",
    "Total Bsmt SF",
    "Year Built",
    "Year Remod/Add",
    "Full Bath",
    "House Style",
]

NUMERIC_FEATURES = [
    "Overall Qual",
    "Gr Liv Area",
    "Garage Cars",
    "Total Bsmt SF",
    "Year Built",
    "Year Remod/Add",
    "Full Bath",
]

ORDINAL_FEATURES = [
    "Kitchen Qual",
]

NOMINAL_FEATURES = [
    "Neighborhood",
    "House Style",
]

KITCHEN_QUAL_ORDER = ["Po", "Fa", "TA", "Gd", "Ex"]

MODEL_NAME = "Ridge(alpha=1.0)"
TARGET_TRANSFORM = "log1p(SalePrice) -> expm1(prediction)"
RANDOM_STATE = 42

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_DIR = PROJECT_ROOT / "app" / "prompts"
CANONICAL_DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "AmesHousing.csv"
FALLBACK_DATASET_PATH = PROJECT_ROOT / "data" / "raw" / "ames.csv"
MODEL_ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "best_model.joblib"
TRAINING_SUMMARY_PATH = PROJECT_ROOT / "artifacts" / "training_summary.json"
FEATURE_CONFIG_PATH = PROJECT_ROOT / "artifacts" / "feature_config.json"
PROMPT_EVAL_RESULTS_PATH = PROJECT_ROOT / "artifacts" / "prompt_eval_results.json"

OLLAMA_MODEL_NAME = "llama3.1:latest"
OLLAMA_API_URL = "http://127.0.0.1:11434/api/generate"
OLLAMA_TIMEOUT_SECONDS = 30
OLLAMA_MAX_RETRIES = 2
OLLAMA_RETRY_DELAY_SECONDS = 1
EXTRACTION_PROMPT_V1_PATH = PROMPTS_DIR / "extraction_v1.txt"
EXTRACTION_PROMPT_V2_PATH = PROMPTS_DIR / "extraction_v2.txt"
DEFAULT_EXTRACTION_PROMPT_VERSION = "extraction_v1"
INTERPRETATION_PROMPT_V1_PATH = PROMPTS_DIR / "interpretation_v1.txt"
DEFAULT_INTERPRETATION_PROMPT_VERSION = "interpretation_v1"

API_TO_AMES_FEATURE_MAP = {
    "overall_qual": "Overall Qual",
    "gr_liv_area": "Gr Liv Area",
    "neighborhood": "Neighborhood",
    "kitchen_qual": "Kitchen Qual",
    "garage_cars": "Garage Cars",
    "total_bsmt_sf": "Total Bsmt SF",
    "year_built": "Year Built",
    "year_remod_add": "Year Remod/Add",
    "full_bath": "Full Bath",
    "house_style": "House Style",
}

API_FEATURES = list(API_TO_AMES_FEATURE_MAP.keys())
USED_FEATURES_FOR_RESPONSE = API_FEATURES.copy()

HOUSE_STYLE_NORMALIZATION_MAP = {
    "1story": "1Story",
    "1 story": "1Story",
    "1-story": "1Story",
    "one story": "1Story",
    "one-story": "1Story",
    "2story": "2Story",
    "2 story": "2Story",
    "2-story": "2Story",
    "two story": "2Story",
    "two-story": "2Story",
    "split foyer": "SFoyer",
    "split-level": "SLvl",
    "split level": "SLvl",
}
