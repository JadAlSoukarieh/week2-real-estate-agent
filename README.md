# AI Real Estate Agent - Phase 4 Foundation

## Project purpose
This project builds the ML artifact and prompt-chain foundation for a Week 2 AI Real Estate Agent. The current phase includes the frozen Ames Housing model pipeline, Stage 1 extraction from natural-language queries, and a connected Stage 2 interpretation flow when the final feature set is complete.

## Current phase scope
- Train a Ridge regression model on `log1p(SalePrice)`
- Save the fitted preprocessing + model artifact
- Save training summary and feature config metadata
- Expose a strict FastAPI endpoint for structured feature prediction
- Extract partial structured features from natural-language queries with Ollama
- Compare extraction prompt versions with a simple experiment script
- Run a connected extraction -> overrides -> prediction -> interpretation chain

## Folder structure
```text
app/
  config.py
  main.py
  prompts/
    extraction_v1.txt
    extraction_v2.txt
    interpretation_v1.txt
  schemas.py
  services/
    chain_service.py
    extraction_service.py
    interpretation_service.py
    prediction_service.py
artifacts/
data/
  raw/
    AmesHousing.csv
scripts/
  train.py
  evaluate.py
  run_chain_smoke_test.py
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

## Phase 4 chained analysis
The new chained endpoint connects Stage 1 extraction, optional overrides, Phase 2 prediction, and Stage 2 interpretation.

### Override behavior
- Overrides win only when a non-null override value is provided.
- A null override does not erase an extracted value.
- Prediction only runs when all 10 required fields are present after extraction plus overrides.

### Example chained request
```bash
curl -X POST "http://127.0.0.1:8000/analyze-query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What might a 1-story house in NAmes with a good kitchen and 2-car garage cost?",
    "overrides": {
      "overall_qual": 7,
      "gr_liv_area": 1710,
      "total_bsmt_sf": 1080,
      "year_built": 2003,
      "year_remod_add": 2003,
      "full_bath": 2
    }
  }'
```

### Run chain smoke test
```bash
python3 scripts/run_chain_smoke_test.py
```

## What is not built yet
- No UI
- No Docker workflow refinement
- No bonus market insights
