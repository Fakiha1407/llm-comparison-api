import asyncio
import os
import time
from dotenv import load_dotenv
from fastapi import FastAPI
from groq import AsyncGroq  # Crucial upgrade for true parallelism
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="LLM Comparison API — Enterprise Scaled")

# Initialize the Asynchronous Client
groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))


class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 200


# All target wrapper functions converted to async coroutines
async def call_llama(prompt: str, max_tokens: int):
    start = time.perf_counter()  # perf_counter is more accurate for benchmarking
    response = await groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    latency = round((time.perf_counter() - start) * 1000)
    return {
        "response": response.choices[0].message.content,
        "latency_ms": latency,
        "model": "llama-3.1-8b-instant",
        "tokens_input": response.usage.prompt_tokens,
        "tokens_output": response.usage.completion_tokens,
        "cost_usd": 0.0,
    }


async def call_llama4(prompt: str, max_tokens: int):
    start = time.perf_counter()
    response = await groq_client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    latency = round((time.perf_counter() - start) * 1000)
    return {
        "response": response.choices[0].message.content,
        "latency_ms": latency,
        "model": "meta-llama/llama-4-scout-17b-16e-instruct",
        "tokens_input": response.usage.prompt_tokens,
        "tokens_output": response.usage.completion_tokens,
        "cost_usd": 0.0,
    }


async def call_llama70b(prompt: str, max_tokens: int):
    start = time.perf_counter()
    response = await groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    latency = round((time.perf_counter() - start) * 1000)
    return {
        "response": response.choices[0].message.content,
        "latency_ms": latency,
        "model": "llama-3.3-70b-versatile",
        "tokens_input": response.usage.prompt_tokens,
        "tokens_output": response.usage.completion_tokens,
        "cost_usd": 0.0,
    }


# Wrap individual execution calls with try/except blocks to preserve async pooling
async def safe_task_wrapper(task_func, prompt, max_tokens):
    try:
        return await task_func(prompt, max_tokens)
    except Exception as e:
        return {"error": str(e)}


@app.post("/compare")
async def compare_llms(request: PromptRequest):
    # Launch all tasks concurrently using asyncio.gather.
    # The event loop schedules all 3 HTTP calls simultaneously without blocking.
    tasks = [
        safe_task_wrapper(call_llama, request.prompt, request.max_tokens),
        safe_task_wrapper(call_llama4, request.prompt, request.max_tokens),
        safe_task_wrapper(call_llama70b, request.prompt, request.max_tokens),
    ]

    # The total wall-clock execution time will only equal the SLOWEST single model response
    llama_31, llama_4, llama_33 = await asyncio.gather(*tasks)

    results = {
        "llama-3.1-8b": llama_31,
        "llama-4-scout": llama_4,
        "llama-3.3-70b": llama_33,
    }

    # Extract metrics out of the non-faulty responses
    for model_name, result in results.items():
        if "error" not in result:
            result["word_count"] = len(result["response"].split())
            result["token_efficiency"] = (
                round(result["tokens_output"] / result["tokens_input"], 2)
                if result.get("tokens_input", 0) > 0
                else 0
            )

    valid = {k: v for k, v in results.items() if "error" not in v}
    fastest = min(valid, key=lambda k: valid[k]["latency_ms"]) if valid else None
    most_detailed = (
        max(valid, key=lambda k: valid[k]["word_count"]) if valid else None
    )
    most_efficient = (
        max(valid, key=lambda k: valid[k]["token_efficiency"]) if valid else None
    )

    return {
        "prompt": request.prompt,
        "results": results,
        "summary": {
            "fastest": fastest,
            "fastest_latency_ms": (
                valid[fastest]["latency_ms"] if fastest else None
            ),
            "most_detailed": most_detailed,
            "most_efficient": most_efficient,
            "models_compared": len(valid),
        },
    }


@app.get("/")
async def root():
    return {
        "message": "Asynchronous LLM Comparison Core Framework — Powered by AsyncGroq"
    }
