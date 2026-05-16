# LLM Comparison API — Multi-Provider Enterprise Benchmark

A benchmarking framework that evaluates commercial LLM services across multiple dimensions relevant to enterprise use: **latency, cost, response quality, and consistency**.

---

## What This Does

Most LLM benchmarks ask: *can the model answer correctly?*

This framework asks: *which model should an enterprise engineer actually use — given speed, cost, consistency, and response quality under identical conditions?*

It calls 5 models across 4 providers **simultaneously** (not sequentially), ensuring fair wall-clock latency measurement. It then scores each model across multiple dimensions and produces a ranked summary.

---

## Providers & Models

| Provider | Model | Notes |
|---|---|---|
| Groq | `llama-3.1-8b-instant` | Fastest small model |
| Groq | `llama-4-scout-17b` | Meta's multimodal-capable model |
| Groq | `llama-3.3-70b-versatile` | Largest Llama variant |
| Cerebras | `gpt-oss-120b` | OpenAI OSS 120B on wafer-scale hardware |
| Mistral AI (EU) | `mistral-small-latest` | European provider, GDPR-native |

---

## Architecture — How It Works

```
User sends prompt
       │
       ▼
  FastAPI server (async)
       │
       ├──► call_llama()      ──► Groq API  (Llama 3.1 8B)
       ├──► call_llama4()     ──► Groq API  (Llama 4 Scout)
       ├──► call_llama70b()   ──► Groq API  (Llama 70B)      ← all 5 launched
       ├──► call_cerebras()   ──► Cerebras SDK (GPT-OSS 120B)   simultaneously
       └──► call_mistral()    ──► Mistral REST API              via asyncio.gather()
                │
                ▼
     safe_task_wrapper() catches any provider failure
     without crashing the other 4 calls
                │
                ▼
     Results aggregated — rankings computed
                │
                ▼
          JSON response
```

**Key design decision:** `asyncio.gather()` launches all 5 HTTP calls at the same time. Total wall-clock time equals the slowest single response, not the sum of all responses. This is the only fair way to measure latency for enterprise comparison.

---

## Endpoints

### `POST /compare`

Calls all 5 models in parallel with the same prompt and token budget. Returns per-model metrics and a ranked summary.

**Request:**
```json
{
  "prompt": "Explain transformer architecture",
  "max_tokens": 200
}
```

**Response fields per model:**

| Field | What it means |
|---|---|
| `response` | The model's actual answer |
| `latency_ms` | Wall-clock time from request sent to response received, in milliseconds |
| `model` | Exact model identifier used |
| `provider` | Which company's infrastructure served the request |
| `tokens_input` | Number of tokens in your prompt (affects cost) |
| `tokens_output` | Number of tokens in the response (affects cost) |
| `cost_usd` | Calculated production cost based on published token pricing |
| `cost_note` | Whether this is a free-tier or production price |
| `word_count` | Number of words in the response — proxy for response detail |
| `token_efficiency` | `tokens_output / tokens_input` — how much output per unit of input |

**Summary fields:**

| Field | What it means |
|---|---|
| `fastest` | Model with lowest `latency_ms` |
| `fastest_latency_ms` | That model's latency |
| `cheapest` | Model with lowest `cost_usd` |
| `cheapest_cost_usd` | That model's cost |
| `most_detailed` | Model with highest `word_count` |
| `models_compared` | How many providers responded successfully |

**Example finding:** Cerebras GPT-OSS 120B (741ms) was faster than Groq Llama 3.1 8B (875ms) in a single-call test — a 120B parameter model beating an 8B model. This is the hardware story: Cerebras wafer-scale silicon vs GPU clusters.

---

### `POST /consistency`

Runs the same prompt **3 times per model** in parallel, then measures:
- How similar the responses are to each other (semantic consistency)
- How stable the latency is across runs

This reveals which models are reliable in production vs which give unpredictable results.

**Request:**
```json
{
  "prompt": "What is the capital of Germany?",
  "max_tokens": 200
}
```

**Response fields per model:**

| Field | What it means |
|---|---|
| `runs` | Number of successful runs (max 3) |
| `latency_ms_per_run` | Latency for each individual run, e.g. `[620, 684, 551]` |
| `latency_variance_ms` | Difference between fastest and slowest run — lower is more predictable |
| `latency_mean_ms` | Average latency across 3 runs — more reliable than a single measurement |
| `consistency_score` | Semantic similarity between responses (0–1). Calculated using `sentence-transformers` cosine similarity. **1.0 = identical meaning, 0.0 = completely different** |
| `consistent` | `true` if `consistency_score >= 0.85`, `false` otherwise |

**How consistency scoring works:**

Each response is converted into a vector embedding using `all-MiniLM-L6-v2` (runs locally, no API cost). The cosine similarity between every pair of response vectors is calculated, and the mean is returned as the consistency score.

A score of 1.0 means all 3 responses said the same thing in equivalent ways. A score below 0.85 means the model gave meaningfully different answers to the same question — a reliability problem for enterprise use.

**Real finding from testing:**

| Model | Mean Latency | Latency Variance | Consistency Score | Reliable? |
|---|---|---|---|---|
| Llama 3.1 8B (Groq) | 2423ms | 88ms | 1.0 | ok |
| Llama 70B (Groq) | 2564ms | 275ms | 1.0 | ok |
| GPT-OSS 120B (Cerebras) | 2026ms | **1207ms** | **0.79** | not ok |
| Mistral Small | **618ms** | 133ms | 1.0 | ok |

The single-call `/compare` test ranked Cerebras as fastest. The `/consistency` test revealed 1207ms latency variance and inconsistent answers. **A snapshot benchmark would have recommended the wrong model.**

---

## Cost Calculation

Cost is calculated as:

```
cost = (tokens_input / 1,000,000) × input_price
     + (tokens_output / 1,000,000) × output_price
```

Prices used (USD per 1M tokens, May 2026):

| Model | Input | Output |
|---|---|---|
| Llama 3.1 8B (Groq) | $0.05 | $0.08 |
| Llama 4 Scout (Groq) | $0.11 | $0.34 |
| Llama 70B (Groq) | $0.59 | $0.79 |
| GPT-OSS 120B (Cerebras) | $0.60 | $0.60 |
| Mistral Small | $0.10 | $0.30 |

Groq models are currently free tier — the cost shown is what production pricing would be, enabling cost projection at enterprise scale.

---

## Setup

**1. Clone the repo**
```bash
git clone https://github.com/Fakiha1407/llm-comparison-api
cd llm-comparison-api
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Add API keys**

Copy `.env.example` to `.env` and fill in your keys:
```
GROQ_API_KEY=your_key_here
CEREBRAS_API_KEY=your_key_here
MISTRAL_API_KEY=your_key_here
```

All three providers offer free tiers:
- Groq: console.groq.com
- Cerebras: cloud.cerebras.ai
- Mistral: console.mistral.ai

**4. Run**
```bash
uvicorn main:app --reload
```

**5. Open the interactive API docs**
```
http://localhost:8000/docs
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `fastapi` | API framework |
| `uvicorn` | ASGI server |
| `groq` | Async Groq SDK |
| `httpx` | Async HTTP client for Mistral |
| `cerebras-cloud-sdk` | Official Cerebras async SDK |
| `sentence-transformers` | Local consistency scoring — no API cost |
| `python-dotenv` | Load API keys from `.env` |
| `pydantic` | Request validation |

---

## Author

**Fakiha Balouch** — M.Sc. Data Science, FAU Erlangen-Nürnberg  
github.com/Fakiha1407
