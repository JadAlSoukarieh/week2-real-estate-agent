# AI Real Estate Agent

A FastAPI + Streamlit project that estimates Ames, Iowa home prices from either structured feature inputs or a plain-English property description.

The app combines:
- a trained Ridge regression pricing model
- LLM-based feature extraction from natural language
- an optional override flow for missing details
- a concise interpretation layer for the final estimate

## What the app does

The backend supports three main workflows:
- `POST /predict-features`
  - predict from a complete structured feature payload
- `POST /extract-features`
  - extract partial structured features from a natural-language property query
- `POST /analyze-query`
  - run the connected flow:
    - extraction
    - optional overrides
    - prediction
    - interpretation

The Streamlit UI sits on top of `/health` and `/analyze-query` for a cleaner end-to-end demo.

## Tech Stack

- FastAPI
- Streamlit
- scikit-learn
- OpenAI API for hosted LLM calls
- Docker for backend containerization

## Project Structure

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
    openai_service.py
    prediction_service.py
artifacts/
  best_model.joblib
  feature_config.json
  training_summary.json
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
Dockerfile
requirements.txt
```

## Required Environment Variables

Hosted LLM calls now use OpenAI.

Required:
- `OPENAI_API_KEY`

Optional:
- `LLM_PROVIDER`
  - defaults to `openai`
- `OPENAI_MODEL`
  - defaults to `gpt-4.1-mini`
- `BACKEND_BASE_URL`
  - only needed when pointing the Streamlit UI at a non-default backend URL

Do not commit secrets to the repository, Dockerfile, or tracked config files.

## Local Setup

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Export the required env vars:

```bash
export LLM_PROVIDER=openai
export OPENAI_API_KEY="your_openai_api_key"
export OPENAI_MODEL="gpt-4.1-mini"
```

## Train the Model

If you need to regenerate artifacts:

```bash
python3 scripts/train.py
```

Evaluate the saved model:

```bash
python3 scripts/evaluate.py
```

## Run the FastAPI Backend

```bash
uvicorn app.main:app --reload
```

Health check:

```bash
curl http://127.0.0.1:8000/health
```

## Run the Streamlit UI

Start the FastAPI backend first, then launch the UI:

```bash
streamlit run ui/streamlit_app.py
```

Main UI flow:
- describe a property in plain English
- review extracted fields
- fill or correct any missing details
- rerun the analysis
- view the estimate and interpretation

## API Examples

### Predict from structured features

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

### Extract structured features from a query

```bash
curl -X POST "http://127.0.0.1:8000/extract-features" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How much would a 1-story house in NAmes with a good kitchen and 2-car garage cost?"
  }'
```

### Run the full chained analysis

```bash
curl -X POST "http://127.0.0.1:8000/analyze-query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Estimate the price of a 2-story house in NAmes with overall quality 8 out of 10, 1,850 square feet above ground living area, a good kitchen, 2 garage spaces, 1,100 square feet of basement, built in 2004, remodeled in 2008, and 2 full bathrooms."
  }'
```

## Prompt and Chain Utilities

Compare extraction prompt versions:

```bash
python3 scripts/run_prompt_experiments.py
```

Run the chain smoke test:

```bash
python3 scripts/run_chain_smoke_test.py
```

## Docker Backend

This project includes a backend-only Docker image for FastAPI.

Important:
- `artifacts/` must already exist before building the image
- the Docker image does not include Streamlit
- the Docker image does not store your API key

Build the image:

```bash
docker build -t real-estate-agent-api:latest .
```

Run the backend container:

```bash
docker run --rm \
  -p 8000:8000 \
  -e LLM_PROVIDER=openai \
  -e OPENAI_API_KEY="your_openai_api_key" \
  -e OPENAI_MODEL="gpt-4.1-mini" \
  real-estate-agent-api:latest
```

Test the running container:

```bash
curl http://127.0.0.1:8000/health
```

```bash
curl -X POST "http://127.0.0.1:8000/analyze-query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Estimate the price of a 2-story house in NAmes with overall quality 8 out of 10, 1,850 square feet above ground living area, a good kitchen, 2 garage spaces, 1,100 square feet of basement, built in 2004, remodeled in 2008, and 2 full bathrooms."
  }'
```

Stop the container with `Ctrl+C`. The example uses `--rm`, so Docker removes it automatically after stop.

## Deployment Notes

For Railway or any similar hosted backend deployment, set these environment variables in the platform:
- `LLM_PROVIDER=openai`
- `OPENAI_API_KEY=...`
- `OPENAI_MODEL=gpt-4.1-mini`

Make sure the deployed service also includes:
- `app/`
- `artifacts/`
- `requirements.txt`

If you deploy only the backend, the Streamlit UI can stay local or be deployed separately later.

## Security Notes

- Never hardcode `OPENAI_API_KEY` in source code
- Never commit `.env` files with real secrets
- Prefer environment variables locally, in Docker, and in Railway
- Rotate any API key that has ever been pasted into chat, screenshots, or shared logs

## Current Limitations

- The pricing model is trained on the Ames Housing dataset, so predictions are scoped to that feature distribution
- The backend container is Dockerized; the Streamlit UI is not containerized yet
- The app does not currently include broader market-insight or live listing data
