from fastapi import FastAPI
from pydantic import BaseModel
from groq import Groq
import time
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="LLM Comparison API")

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 200

def call_llama(prompt, max_tokens):
    start = time.time()
    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens
    )
    latency = round((time.time() - start) * 1000)
    return {
        "response": response.choices[0].message.content,
        "latency_ms": latency,
        "model": "llama-3.1-8b-instant",
        "tokens_input": response.usage.prompt_tokens,
        "tokens_output": response.usage.completion_tokens,
        "cost_usd": 0.0
    }

def call_llama4(prompt, max_tokens):
    start = time.time()
    response = groq_client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens
    )
    latency = round((time.time() - start) * 1000)
    return {
        "response": response.choices[0].message.content,
        "latency_ms": latency,
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "tokens_input": response.usage.prompt_tokens,
        "tokens_output": response.usage.completion_tokens,
        "cost_usd": 0.0
    }

def call_llama70b(prompt, max_tokens):
    start = time.time()
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens
    )
    latency = round((time.time() - start) * 1000)
    return {
        "response": response.choices[0].message.content,
        "latency_ms": latency,
        "model": "llama-3.3-70b-versatile",
        "tokens_input": response.usage.prompt_tokens,
        "tokens_output": response.usage.completion_tokens,
        "cost_usd": 0.0
    }

@app.post("/compare")
def compare_llms(request: PromptRequest):
    results = {}

    try:
        results["llama-3.1-8b"] = call_llama(request.prompt, request.max_tokens)
    except Exception as e:
        results["llama-3.1-8b"] = {"error": str(e)}

    try:
        results["llama-4-scout"] = call_llama4(request.prompt, request.max_tokens)
    except Exception as e:
        results["llama-4-scout"] = {"error": str(e)}

    try:
        results["llama-3.3-70b"] = call_llama70b(request.prompt, request.max_tokens)
    except Exception as e:
        results["llama-3.3-70b"] = {"error": str(e)}

    # Add quality metrics to each valid result
    for model_name, result in results.items():
        if "error" not in result:
            result["word_count"] = len(result["response"].split())
            result["token_efficiency"] = round(
                result["tokens_output"] / result["tokens_input"], 2
            ) if result.get("tokens_input", 0) > 0 else 0

    valid = {k: v for k, v in results.items() if "error" not in v}
    fastest = min(valid, key=lambda k: valid[k]["latency_ms"]) if valid else None
    most_detailed = max(valid, key=lambda k: valid[k]["word_count"]) if valid else None
    most_efficient = max(valid, key=lambda k: valid[k]["token_efficiency"]) if valid else None

    return {
        "prompt": request.prompt,
        "results": results,
        "summary": {
            "fastest": fastest,
            "fastest_latency_ms": valid[fastest]["latency_ms"] if fastest else None,
            "most_detailed": most_detailed,
            "most_efficient": most_efficient,
            "models_compared": len(valid)
        }
    }

@app.get("/")
def root():
    return {"message": "LLM Comparison API — Llama3.1 vs Llama4-Scout vs Llama3.3-70b, 100% free via Groq"}