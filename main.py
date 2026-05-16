import asyncio
import os
import time
from dotenv import load_dotenv
from fastapi import FastAPI
from groq import AsyncGroq
from pydantic import BaseModel
import httpx
from sentence_transformers import SentenceTransformer, util

load_dotenv()

app = FastAPI(title="LLM Comparison API — Multi-Provider Enterprise Benchmark")

# ── Clients 
groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))

# Cerebras and Mistral use OpenAI-compatible endpoints
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")
MISTRAL_API_KEY  = os.getenv("MISTRAL_API_KEY")

# Sentence transformer for consistency scoring (runs locally, no API cost)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 200

# ── Token pricing (USD per 1M tokens, as of May 2026) ─────────────────────────
# Groq free tier = $0.00. Mistral and Cerebras publish production prices.
# We calculate what production cost WOULD be — useful for enterprise cost analysis.
PRICING = {
    "llama-3.1-8b-instant":                  {"input": 0.05,  "output": 0.08},
    "meta-llama/llama-4-scout-17b-16e-instruct": {"input": 0.11,  "output": 0.34},
    "llama-3.3-70b-versatile":               {"input": 0.59,  "output": 0.79},
    "gpt-oss-120b":                          {"input": 0.60,  "output": 0.60},  # Cerebras
    "mistral-small-latest":                  {"input": 0.10,  "output": 0.30},
}

def calculate_cost(model: str, tokens_input: int, tokens_output: int) -> float:
    price = PRICING.get(model, {"input": 0.0, "output": 0.0})
    return round(
        (tokens_input  / 1_000_000) * price["input"] +
        (tokens_output / 1_000_000) * price["output"],
        8
    )

#  Consistency scoring 
def score_consistency(responses: list[str]) -> float:
    """
    Runs same prompt N times, measures semantic similarity between responses.
    Returns mean pairwise cosine similarity (0-1). Higher = more consistent.
    """
    if len(responses) < 2:
        return 1.0
    embeddings = embedder.encode(responses, convert_to_tensor=True)
    scores = []
    for i in range(len(responses)):
        for j in range(i + 1, len(responses)):
            scores.append(float(util.cos_sim(embeddings[i], embeddings[j])))
    return round(sum(scores) / len(scores), 4)

# ── Provider call functions ────────────────────────────────────────────────────
async def call_llama(prompt: str, max_tokens: int):
    start = time.perf_counter()
    response = await groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    latency = round((time.perf_counter() - start) * 1000)
    model = "llama-3.1-8b-instant"
    ti, to = response.usage.prompt_tokens, response.usage.completion_tokens
    return {
        "response": response.choices[0].message.content,
        "latency_ms": latency,
        "model": model,
        "provider": "Groq",
        "tokens_input": ti,
        "tokens_output": to,
        "cost_usd": calculate_cost(model, ti, to),
        "cost_note": "free tier — production price shown",
    }


async def call_llama4(prompt: str, max_tokens: int):
    start = time.perf_counter()
    model = "meta-llama/llama-4-scout-17b-16e-instruct"
    response = await groq_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    latency = round((time.perf_counter() - start) * 1000)
    ti, to = response.usage.prompt_tokens, response.usage.completion_tokens
    return {
        "response": response.choices[0].message.content,
        "latency_ms": latency,
        "model": model,
        "provider": "Groq",
        "tokens_input": ti,
        "tokens_output": to,
        "cost_usd": calculate_cost(model, ti, to),
        "cost_note": "free tier — production price shown",
    }


async def call_llama70b(prompt: str, max_tokens: int):
    start = time.perf_counter()
    model = "llama-3.3-70b-versatile"
    response = await groq_client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    latency = round((time.perf_counter() - start) * 1000)
    ti, to = response.usage.prompt_tokens, response.usage.completion_tokens
    return {
        "response": response.choices[0].message.content,
        "latency_ms": latency,
        "model": model,
        "provider": "Groq",
        "tokens_input": ti,
        "tokens_output": to,
        "cost_usd": calculate_cost(model, ti, to),
        "cost_note": "free tier — production price shown",
    }


async def call_cerebras(prompt: str, max_tokens: int):
    """
    Cerebras — official async SDK.
    Same Llama 70B model as Groq — different hardware — latency comparison is meaningful.
    """
    from cerebras.cloud.sdk import AsyncCerebras
    start = time.perf_counter()
    model = "gpt-oss-120b"
    client = AsyncCerebras(api_key=CEREBRAS_API_KEY)
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    latency = round((time.perf_counter() - start) * 1000)
    ti = response.usage.prompt_tokens
    to = response.usage.completion_tokens
    return {
        "response": response.choices[0].message.content,
        "latency_ms": latency,
        "model": model,
        "provider": "Cerebras",
        "tokens_input": ti,
        "tokens_output": to,
        "cost_usd": calculate_cost(model, ti, to),
        "cost_note": "production price",
    }


async def call_mistral(prompt: str, max_tokens: int):
    """
    Mistral AI — European provider, GDPR-native, relevant for Mercedes/German enterprise context.
    """
    start = time.perf_counter()
    model = "mistral-small-latest"
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.mistral.ai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {MISTRAL_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
            },
        )
        resp.raise_for_status()
        data = resp.json()
    latency = round((time.perf_counter() - start) * 1000)
    ti = data["usage"]["prompt_tokens"]
    to = data["usage"]["completion_tokens"]
    return {
        "response": data["choices"][0]["message"]["content"],
        "latency_ms": latency,
        "model": model,
        "provider": "Mistral AI (EU)",
        "tokens_input": ti,
        "tokens_output": to,
        "cost_usd": calculate_cost(model, ti, to),
        "cost_note": "production price",
    }


async def safe_task_wrapper(task_func, prompt, max_tokens):
    try:
        return await task_func(prompt, max_tokens)
    except Exception as e:
        return {"error": str(e)}


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.post("/compare")
async def compare_llms(request: PromptRequest):
    """
    Standard comparison — all 5 models called in parallel.
    Returns latency, cost, token efficiency per model + summary rankings.
    """
    tasks = [
        safe_task_wrapper(call_llama,    request.prompt, request.max_tokens),
        safe_task_wrapper(call_llama4,   request.prompt, request.max_tokens),
        safe_task_wrapper(call_llama70b, request.prompt, request.max_tokens),
        safe_task_wrapper(call_cerebras, request.prompt, request.max_tokens),
        safe_task_wrapper(call_mistral,  request.prompt, request.max_tokens),
    ]

    results_list = await asyncio.gather(*tasks)
    keys = ["llama-3.1-8b", "llama-4-scout", "llama-3.3-70b-groq",
            "gpt-oss-120b-cerebras", "mistral-small"]
    results = dict(zip(keys, results_list))

    for result in results.values():
        if "error" not in result:
            result["word_count"] = len(result["response"].split())
            result["token_efficiency"] = (
                round(result["tokens_output"] / result["tokens_input"], 2)
                if result.get("tokens_input", 0) > 0 else 0
            )

    valid = {k: v for k, v in results.items() if "error" not in v}
    fastest      = min(valid, key=lambda k: valid[k]["latency_ms"]) if valid else None
    cheapest     = min(valid, key=lambda k: valid[k]["cost_usd"])   if valid else None
    most_detailed = max(valid, key=lambda k: valid[k]["word_count"]) if valid else None

    return {
        "prompt": request.prompt,
        "results": results,
        "summary": {
            "fastest":           fastest,
            "fastest_latency_ms": valid[fastest]["latency_ms"] if fastest else None,
            "cheapest":          cheapest,
            "cheapest_cost_usd": valid[cheapest]["cost_usd"]   if cheapest else None,
            "most_detailed":     most_detailed,
            "models_compared":   len(valid),
        },
    }


@app.post("/consistency")
async def consistency_test(request: PromptRequest):
    """
    Consistency dimension — runs the same prompt 3x per model in parallel.
    Measures latency variance and semantic similarity between responses.
    High variance = unreliable model for enterprise use.
    """
    RUNS = 3
    call_fns = {
        "llama-3.1-8b":           call_llama,
        "llama-3.3-70b-groq":     call_llama70b,
        "gpt-oss-120b-cerebras": call_cerebras,
        "mistral-small":          call_mistral,
    }

    async def run_model_consistency(name, fn):
        tasks = [safe_task_wrapper(fn, request.prompt, request.max_tokens)
                 for _ in range(RUNS)]
        runs = await asyncio.gather(*tasks)
        valid_runs = [r for r in runs if "error" not in r]
        if not valid_runs:
            return name, {"error": "all runs failed"}

        latencies = [r["latency_ms"] for r in valid_runs]
        responses = [r["response"]   for r in valid_runs]
        consistency_score = score_consistency(responses)
        latency_variance  = round(max(latencies) - min(latencies), 2)

        return name, {
            "runs": len(valid_runs),
            "latency_ms_per_run":  latencies,
            "latency_variance_ms": latency_variance,
            "latency_mean_ms":     round(sum(latencies) / len(latencies), 2),
            "consistency_score":   consistency_score,   # 0-1, higher = more consistent
            "consistent": consistency_score >= 0.85,
        }

    tasks = [run_model_consistency(name, fn) for name, fn in call_fns.items()]
    results_list = await asyncio.gather(*tasks)
    results = dict(results_list)

    valid = {k: v for k, v in results.items() if "error" not in v}
    most_consistent = max(valid, key=lambda k: valid[k]["consistency_score"]) if valid else None
    most_stable_latency = min(valid, key=lambda k: valid[k]["latency_variance_ms"]) if valid else None

    return {
        "prompt": request.prompt,
        "runs_per_model": RUNS,
        "results": results,
        "summary": {
            "most_consistent":       most_consistent,
            "most_stable_latency":   most_stable_latency,
        },
    }


@app.get("/")
async def root():
    return {
        "message": "Multi-Provider Enterprise LLM Benchmark API",
        "providers": ["Groq (Llama 3.1 8B, Llama 4, Llama 70B)", "Cerebras (GPT-OSS 120B)", "Mistral AI EU (Mistral Small)"],
        "endpoints": {
            "/compare":     "Parallel latency + cost + token analysis across all 5 models",
            "/consistency": "3-run consistency + latency variance analysis per model",
        },
    }
