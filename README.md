# LLM Comparison API

A production-grade, fully asynchronous FastAPI REST API built to perform systematic differential analysis of multiple LLM models in parallel. By utilizing native asynchronous pooling, the system evaluates and benchmarks response models concurrently under identical network states to ensure mathematically fair latency tracking.

## 🏗️ Technical Architecture
Unlike traditional benchmarking scripts that execute API calls sequentially—which artificially inflates the latency metrics of trailing models due to socket pooling and serialization bottlenecks—this application enforces a non-blocking asynchronous architecture.

Using `AsyncGroq` and `asyncio.gather()`, multiple HTTP request contexts are dispatched simultaneously. As a result, the total wall-clock turnaround time for the endpoint is strictly bound to the maximum latency of the single slowest provider, rather than the cumulative sum of all providers combined.

## 🚀 Key Features & Metrics Captured
* **True Concurrency:** Non-blocking async event loop architecture.
* **Latency Benchmarking:** High-resolution `time.perf_counter()` tracking to capture exact wall-clock response times in milliseconds.
* **Token Efficiency Analysis:** Calculates output-to-input token distribution ratios to track structural generation density.
* **Granular Payload Extraction:** Captures exact input tokens, output tokens, word counts, and response texts across all evaluated engines.
* **Robust Task Handling:** Features an isolated `safe_task_wrapper` context per model execution block. This prevents an external API rate limit or outage on a single model from causing a cascading failure of the entire evaluation runtime.

---

## 📊 Models Compared (via Groq API)
1. **Llama-3.1-8b-instant** — Ultra-lightweight, high-velocity baseline.
2. **Llama-4-Scout-17b-instruct** — Advanced next-generation, mid-tier efficiency profile.
3. **Llama-3.3-70b-versatile** — Maximum capacity, high-parameter structural reasoning engine.

---

## 🛠️ Installation & Local Setup

### 1. Clone the Repository
```bash
git clone [https://github.com/Fakiha1407/llm-comparison-api.git](https://github.com/Fakiha1407/llm-comparison-api.git)
cd llm-comparison-api
