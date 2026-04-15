# AI Real Estate Agent - Phase 2 Foundation

## Project purpose
This project builds the ML artifact and inference-ready backend foundation for a Week 2 AI Real Estate Agent. The current phase focuses on a frozen Ames Housing feature set, a reproducible training pipeline, and a temporary structured-feature prediction API.

## Current phase scope
- Train a Ridge regression model on `log1p(SalePrice)`
- Save the fitted preprocessing + model artifact
- Save training summary and feature config metadata
- Expose a strict FastAPI endpoint for structured feature prediction

## Folder structure
```text
app/
  config.py
  main.py
  schemas.py
  services/
    prediction_service.py
artifacts/
data/
  raw/
    AmesHousing.csv
scripts/
  train.py
  evaluate.py
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

## Example request payload
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

## What is not built yet
- No natural-language extraction
- No Stage 1 or Stage 2 LLM chain
- No UI
- No Docker workflow beyond placeholder file presence
