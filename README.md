# AI Real Estate Agent - Phase 3 Foundation

## Project purpose
This project builds the ML artifact and prompt-chain foundation for a Week 2 AI Real Estate Agent. The current phase includes the frozen Ames Housing model pipeline, a strict structured-feature prediction API, and Stage 1 LLM extraction from natural-language property queries.

## Current phase scope
- Train a Ridge regression model on `log1p(SalePrice)`
- Save the fitted preprocessing + model artifact
- Save training summary and feature config metadata
- Expose a strict FastAPI endpoint for structured feature prediction
- Extract partial structured features from natural-language queries with Ollama
- Compare extraction prompt versions with a simple experiment script

## Folder structure
```text
app/
  config.py
  main.py
  prompts/
    extraction_v1.txt
    extraction_v2.txt
  schemas.py
  services/
    extraction_service.py
    prediction_service.py
artifacts/
data/
  raw/
    AmesHousing.csv
scripts/
  train.py
  evaluate.py
  run_prompt_experiments.py
```

## How to run training
```bash
python3 -m pip install -r requirements.txt
python3 scripts/train.py
```

## How to run evaluation
```bash
python3 scripts/evaluate.py
```

## How to run the API
```bash
uvicorn app.main:app --reload
```

## Example prediction request payload
```bash
curl -X POST "http://127.0.0.1:8000/predict-features" \
  -H "Content-Type: application/json" \
  -d '{
    "overall_qual": 7,
    "gr_liv_area": 1710,
    "neighborhood": "NAmes",
    "kitchen_qual": "Gd",
    "garage_cars": 2,
    "total_bsmt_sf": 1080,
    "year_built": 2003,
    "year_remod_add": 2003,
    "full_bath": 2,
    "house_style": "2Story"
  }'
```

## Phase 3 Stage 1 extraction
Stage 1 converts a plain-English property query into validated partial structured features and completeness metadata. It does not perform price prediction from natural language yet.

### Example extraction request
```bash
curl -X POST "http://127.0.0.1:8000/extract-features" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How much would a 1-story house in NAmes with a good kitchen and 2-car garage cost?"
  }'
```

### Run prompt experiments
```bash
python3 scripts/run_prompt_experiments.py
```

## What is not built yet
- No Stage 2 interpretation
- No end-to-end orchestration from extraction to prediction
- No UI
- No Docker workflow beyond placeholder file presence
