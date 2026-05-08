# LLM Comparison API

A FastAPI REST API that compares multiple LLM models in parallel and returns 
structured performance metrics for systematic differential analysis.

## Models Compared
- **Llama-3.1-8b-instant** — fast, lightweight
- **Llama-4-Scout-17b** — newer generation, mid-size  
- **Llama-3.3-70b-versatile** — most capable, largest

## Metrics Measured
- Response latency (ms)
- Token usage (input/output)
- Word count per response
- Token efficiency (output/input ratio)
- Summary: fastest, most detailed, most efficient model

## Run Locally
```bash
pip install -r requirements.txt
uvicorn main:app --reload
```
Open http://127.0.0.1:8000/docs to test interactively.

## Tech Stack
Python · FastAPI · Groq API · Docker · python-dotenv