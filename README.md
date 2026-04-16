# AI Real Estate Agent - Phase 5 Foundation

## Project purpose
This project builds the ML artifact, prompt-chain flow, and a polished Streamlit product demo for a Week 2 AI Real Estate Agent. The current phase includes the frozen Ames Housing model pipeline, Stage 1 extraction, Stage 2 interpretation, and a presentation-ready UI on top of the existing FastAPI backend.

## Current phase scope
- Train a Ridge regression model on `log1p(SalePrice)`
- Save the fitted preprocessing + model artifact
- Save training summary and feature config metadata
- Expose a strict FastAPI endpoint for structured feature prediction
- Extract partial structured features from natural-language queries with a hosted OpenAI model
- Compare extraction prompt versions with a simple experiment script
- Run a connected extraction -> overrides -> prediction -> interpretation chain
- Run a polished Streamlit UI for end-to-end demo flow

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
ui/
  streamlit_app.py
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

## Phase 5 Streamlit UI
The Streamlit frontend sits on top of the existing FastAPI backend and uses `/health` plus `/analyze-query` as the main integration points.

### Main UI flow
- enter a natural-language property query
- review extracted features
- fill or correct overrides
- run the chained analysis again
- view prediction and interpretation in a polished result view

### Backend requirement
Start the FastAPI backend before launching the UI.

### Run the UI
```bash
streamlit run ui/streamlit_app.py
```

## Hosted LLM configuration
Hosted LLM calls now use OpenAI instead of local Ollama for extraction and interpretation.

### Required environment variables
- `OPENAI_API_KEY`
- optional: `OPENAI_MODEL` (defaults to `gpt-4.1-mini`)
- optional: `LLM_PROVIDER` (defaults to `openai`)

Do not commit secrets to source control, Dockerfiles, or tracked config files.

## Phase 6 Docker backend
The FastAPI backend can now run from Docker as a backend-only container. This phase does not Dockerize the Streamlit UI.

### Important
The runtime artifacts must already exist before building the image:
- `artifacts/best_model.joblib`
- `artifacts/training_summary.json`
- `artifacts/feature_config.json`

### Build the image
```bash
docker build -t real-estate-agent-api:latest .
```

### Run the backend container on Linux
Pass the OpenAI environment variables into the container at runtime:

```bash
docker run --rm \
  -p 8000:8000 \
  -e OPENAI_API_KEY=your_api_key_here \
  -e OPENAI_MODEL=gpt-4.1-mini \
  real-estate-agent-api:latest
```

### Test the container
```bash
curl http://127.0.0.1:8000/health
```

```bash
curl -X POST "http://127.0.0.1:8000/extract-features" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Estimate the price of a 2-story house in NAmes with overall quality 8 out of 10, 1,850 square feet above ground living area, a good kitchen, 2 garage spaces, 1,100 square feet of basement, built in 2004, remodeled in 2008, and 2 full bathrooms."
  }'
```

```bash
curl -X POST "http://127.0.0.1:8000/analyze-query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Estimate the price of a 2-story house in NAmes with overall quality 8 out of 10, 1,850 square feet above ground living area, a good kitchen, 2 garage spaces, 1,100 square feet of basement, built in 2004, remodeled in 2008, and 2 full bathrooms."
  }'
```

### Stop/remove the container
The example run command already uses `--rm`, so the container is removed automatically when it stops. Use `Ctrl+C` in the terminal running Docker to stop it.

## What is not built yet
- No Docker workflow refinement
- No bonus market insights
