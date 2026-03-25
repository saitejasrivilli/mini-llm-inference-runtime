# Mini LLM Inference Runtime

A portfolio-quality transformer inference runtime built from scratch to demonstrate **ML systems**, **runtime**, **memory**, and **performance engineering** skills.

This project focuses on the **systems side of LLM inference** rather than model training. Instead of relying on a high-level `generate()` API, it exposes the runtime mechanics that matter in real serving systems:

- manual **KV-cache** handling
- **naive vs optimized** autoregressive decoding
- **static** and **continuous batching**
- **prefix caching**
- **paged KV-cache bookkeeping**
- **latency / throughput / TTFT** profiling
- optional **dynamic quantization** on CPU
- lightweight **FastAPI serving layer**
- correctness checks and benchmarking utilities

The goal is not to replicate production runtimes like vLLM or TensorRT-LLM. The goal is to build a technically honest, inspectable runtime that surfaces the same core tradeoffs those systems manage.

---

## Why this project

Transformer inference is a **runtime and systems problem** as much as a modeling problem.

For autoregressive generation, end-to-end performance depends heavily on:

- avoiding redundant computation during decode
- managing persistent decode state efficiently
- batching requests to improve utilization
- balancing throughput vs latency
- controlling memory growth as sequence length increases
- understanding hardware/backend constraints for optimization paths like quantization

This project was built to make those tradeoffs explicit and measurable.

---

## Features

### Core runtime
- **Naive autoregressive decoding**
  - recomputes the full prefix every generation step
  - serves as a correctness and performance baseline

- **KV-cache decoding**
  - stores and reuses per-layer key/value tensors across decode steps
  - avoids repeated attention computation over the full prompt

- **Static batching**
  - processes a fixed batch of requests together
  - useful for throughput-oriented offline-style workloads

- **Continuous batching**
  - maintains a live set of active requests
  - admits new requests as capacity becomes available
  - more representative of online inference serving

### Runtime/memory features
- **Prefix cache**
  - reuses prefill state for exact prompt-prefix matches

- **Paged KV-cache manager**
  - introduces block-based bookkeeping for KV memory
  - demonstrates allocator-style thinking and page-table-style request state

- **Separated prefill vs decode metrics**
  - prefill latency
  - average decode-step latency
  - peak decode-step latency
  - TTFT
  - end-to-end latency
  - throughput

### Performance features
- **Optional CPU int8 path**
  - uses PyTorch dynamic quantization
  - backend-dependent
  - successfully validated with **QNNPACK** on Apple Silicon/macOS

- **Prompt-length scaling benchmark**
  - compares how naive and KV-cache paths behave as prompt length grows

### Serving and testing
- **FastAPI inference API**
  - `/health`
  - `/generate`

- **Correctness checks**
  - validates greedy output equivalence between naive and KV-cache paths

- **Tests**
  - metrics
  - scheduler
  - prefix cache
  - correctness

---

## Architecture

```text
mini-llm-inference-runtime/
├── main.py
├── requirements.txt
├── src/
│   ├── core/
│   │   ├── types.py
│   │   ├── model_adapter.py
│   │   └── metrics.py
│   ├── runtime/
│   │   ├── kv_cache.py
│   │   ├── paged_kv_cache.py
│   │   ├── prefix_cache.py
│   │   ├── correctness.py
│   │   └── speculative.py
│   ├── scheduler/
│   │   ├── request_queue.py
│   │   ├── policies.py
│   │   └── continuous_batcher.py
│   ├── serving/
│   │   ├── schemas.py
│   │   ├── worker.py
│   │   └── api.py
│   ├── benchmarks/
│   │   ├── benchmark_suite.py
│   │   ├── plot_results.py
│   │   ├── scaling.py
│   │   ├── quantization.py
│   │   └── compile_compare.py
│   └── utils/
│       └── io.py
└── tests/
    ├── test_correctness.py
    ├── test_metrics.py
    ├── test_prefix_cache.py
    └── test_scheduler.py
```

---

## Data flow

For a single request, the runtime follows this flow:

1. **Tokenize input prompt**
2. **Prefill**
   - run the model on the full prompt
   - initialize per-layer KV-cache
3. **Decode loop**
   - feed only the newest token
   - reuse prior KV-cache
   - append new KV state
4. **Scheduler / batching policy**
   - execute request individually or as part of a batch
5. **Token selection**
   - greedy decode in the current implementation
6. **Completion check**
   - stop on EOS or max token budget
7. **Profiling**
   - record TTFT, latency, decode-step cost, throughput, and memory stats

---

## Benchmarks

### Main benchmark setup
Final core comparisons were collected locally using:

- model: `distilgpt2`
- requests: `6`
- max new tokens per request: `20`

### Core runtime results

| Mode | Throughput (tok/s) | Avg Latency (s) | Avg TTFT (s) |
|---|---:|---:|---:|
| naive_single | 30.10 | 0.664 | 0.215 |
| kv_single | 84.51 | 0.237 | 0.018 |
| kv_static_batch | 44.05 | 0.454 | 0.056 |
| kv_dynamic_batch | 47.91 | 0.418 | 0.035 |

### Key observations
- Manual **KV-cache reuse** produced the largest runtime improvement.
- Compared with naive decoding, **KV single-request throughput improved ~2.8x**.
- Average latency dropped by roughly **64%**.
- TTFT improved substantially because decode no longer recomputed the full prefix every step.
- On this local CPU setup with a relatively small model, batching delivered smaller gains than a production GPU-serving scenario because framework and scheduling overheads were a larger fraction of total runtime.

---

## Quantized CPU path

Dynamic int8 quantization was benchmarked on Apple Silicon/macOS after explicitly enabling the **QNNPACK** quantization backend.

| Mode | Throughput (tok/s) | Avg Latency (s) | Avg TTFT (s) |
|---|---:|---:|---:|
| FP32 KV single | 11.68 | 1.712 | 1.450 |
| INT8 dynamic KV single | 71.92 | 0.278 | 0.098 |

### Quantization observations
- Dynamic int8 quantization improved throughput by roughly **6.2x** in this local CPU benchmark.
- Average latency dropped by roughly **84%**.
- Process RSS did **not** decrease in this setup because the memory metric reflects whole-process memory, including backend overhead and weight packing effects.
- Quantization support is **platform/backend dependent**. On Apple Silicon, this required explicitly activating `qnnpack`.

---

## Prompt scaling

The project includes a scaling benchmark to compare naive decoding vs KV-cache decoding as prompt length grows.

This is useful for demonstrating a core systems insight:

> The value of KV-cache increases as prompt length grows because naive decoding repeatedly recomputes attention over the expanding prefix, while cached decoding only processes the newest token.

Outputs are saved automatically for further analysis.

---

## What is real vs simplified

### Real
- real transformer inference using open-source causal language models
- real manual KV-cache use across decode steps
- real latency / TTFT / throughput / memory instrumentation
- real prefix cache
- real request scheduler loop
- real benchmark outputs and plots
- real FastAPI inference API
- real optional quantized CPU path

### Simplified
- paged KV-cache is **bookkeeping / memory management scaffolding**, not a fused paged-attention kernel
- continuous batching is a **scheduler-driven prototype**, not a production-grade packed GPU serving system
- speculative decoding is a **clear runtime scaffold**, not a production verifier implementation
- quantization uses **PyTorch dynamic quantization**, not custom low-bit inference kernels
- generation uses **greedy decoding** for reproducibility and correctness comparison

This is deliberate. The project is designed to be technically honest and interview-defensible.

---

## Installation

### Clone the repo
```bash
git clone https://github.com/saitejasrivilli/mini-llm-inference-runtime.git
cd mini-llm-inference-runtime
```

### Create environment and install dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## Usage

### Run the main benchmark
```bash
python main.py --model distilgpt2 --n_requests 6 --max_new_tokens 20
```

### Run correctness check
```bash
python main.py --model sshleifer/tiny-gpt2 --check
```

### Run prompt scaling benchmark
```bash
python main.py --model distilgpt2 --scaling --max_new_tokens 12
```

### Run quantization benchmark
```bash
python main.py --model distilgpt2 --quant --max_new_tokens 20
```

### Run speculative decoding demo
```bash
python main.py --model distilgpt2 --speculative --max_new_tokens 12
```

### Start API server
```bash
uvicorn src.serving.api:app --reload
```

---

## API

### Health check
```http
GET /health
```

### Generate
```http
POST /generate
Content-Type: application/json
```

Example request:
```json
{
  "request_id": "demo-1",
  "prompt": "Explain KV-cache in one paragraph.",
  "max_new_tokens": 20
}
```

Example response:
```json
{
  "request_id": "demo-1",
  "text": "...",
  "total_tokens": 20,
  "latency_s": 0.23,
  "ttft_s": 0.02,
  "finish_reason": "max_tokens"
}
```

---

## Outputs

The benchmark pipeline saves results under `outputs/`:

- benchmark summary CSV
- benchmark summary JSON
- throughput plot
- latency plot
- TTFT plot
- prefill latency plot
- decode latency plot
- prompt scaling CSV

This makes the project easier to inspect and easier to present in interviews.

---

## Tests

Run the test suite with:

```bash
pytest tests -q
```

The tests cover:
- correctness between naive and KV-cache decode paths
- scheduler execution sanity
- prefix cache behavior
- metrics summarization

---

## Engineering takeaways

This project reinforced several key inference-systems lessons:

- **Runtime design matters.** Once decoding becomes incremental, inference performance depends heavily on cache policy, batching strategy, and scheduler design.
- **The best optimization is workload-dependent.** On small local CPU runs, KV-cache delivered the largest gains, while batching gains were limited by framework and scheduler overhead.
- **Backend details matter.** Quantization required explicit activation of the correct backend (`qnnpack`) on Apple Silicon before the int8 path became usable.
- **Memory is part of the story.** Persistent decode state, prefix reuse, and KV allocation policy all affect scalability.

---

## Resume bullets

- Built a mini transformer inference runtime in Python with manual KV-cache handling, batching, prefix caching, profiling, and a FastAPI serving layer; improved single-request decode throughput from **30.1 to 84.5 tok/s (~2.8x)** over naive autoregressive inference.
- Added an optional CPU int8 quantization path using PyTorch dynamic quantization with **QNNPACK**, improving KV-cache decoding throughput from **11.7 to 71.9 tok/s (~6.2x)** and reducing latency by **~84%** on local Apple Silicon benchmarks.
- Implemented and benchmarked **single-request, static-batch, and continuous-batch** execution paths, analyzing latency, TTFT, throughput, and memory tradeoffs in a simplified LLM serving runtime.

---

## Interview explanation

### What I built
I built a simplified transformer inference runtime to understand inference as a systems problem rather than just calling `generate()`.

### Why it matters
At inference time, runtime choices strongly affect user experience and infrastructure efficiency. Cache reuse changes compute cost, batching changes utilization, and memory policy changes how the system scales under longer contexts and higher concurrency.

### Key tradeoffs
- KV-cache reduces redundant computation but adds persistent memory state
- batching can improve utilization, but not every workload/hardware setup benefits equally
- quantization can produce major speedups, but backend support is platform-specific
- prefix caching helps repeated prompts, but only when cache keys and reuse semantics are correct

---

## Future work

Planned next improvements include:
- fully packed continuous batch decode
- stronger paged KV-cache implementation
- better memory accounting for cache footprint
- `torch.compile` comparison across models
- improved speculative decoding verification
- asynchronous multi-worker serving loop
- scheduler policies beyond FCFS / shortest-prompt-first

---

## Final note

This project is intentionally designed to be **substantive, finishable, and explainable**. It does not pretend to be a production LLM serving stack. Instead, it demonstrates the engineering instincts that matter for building one.
