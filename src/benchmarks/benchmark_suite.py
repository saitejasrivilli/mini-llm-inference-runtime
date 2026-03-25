from typing import Tuple, List

import pandas as pd

from src.core.metrics import summarize_states
from src.core.types import InferenceRequest
from src.scheduler.continuous_batcher import RuntimeEngine
from src.utils.io import save_json, save_csv
from src.benchmarks.plot_results import plot_metric


def benchmark_suite(
    model_name: str = "distilgpt2",
    n_requests: int = 6,
    max_new_tokens: int = 20,
) -> Tuple[pd.DataFrame, List[dict]]:
    engine = RuntimeEngine(model_name=model_name)

    prompts = [
        "Explain transformers in simple terms.",
        "What is KV-cache in LLM inference?",
        "Why does batching improve throughput?",
        "Describe dynamic batching for inference servers.",
        "What is the tradeoff between latency and throughput?",
        "Why does autoregressive decoding benefit from caching?",
    ][:n_requests]

    requests = [
        InferenceRequest(request_id=f"req_{i}", prompt=prompts[i], max_new_tokens=max_new_tokens)
        for i in range(len(prompts))
    ]

    naive_states = [engine.run_naive_single(r) for r in requests]
    kv_states = [engine.run_kv_single(r) for r in requests]
    static_states = engine.run_static_batch(requests)
    continuous_states = engine.run_continuous_batch(requests, max_batch_size=min(4, len(requests)))

    summaries = [
        summarize_states("naive_single", naive_states),
        summarize_states("kv_single", kv_states),
        summarize_states("kv_static_batch", static_states),
        summarize_states("kv_continuous_batch", continuous_states),
    ]

    df = pd.DataFrame(summaries)

    save_csv(df, "outputs/benchmark_summary.csv")
    save_json(summaries, "outputs/benchmark_summary.json")

    plot_metric(df, "throughput_tok_s", "Throughput comparison", "outputs/plots/throughput.png")
    plot_metric(df, "avg_latency_s", "Average latency comparison", "outputs/plots/latency.png")
    plot_metric(df, "avg_ttft_s", "Average TTFT comparison", "outputs/plots/ttft.png")
    plot_metric(df, "avg_prefill_latency_s", "Average prefill latency", "outputs/plots/prefill_latency.png")
    plot_metric(df, "avg_decode_step_latency_s", "Average decode-step latency", "outputs/plots/decode_latency.png")

    return df, summaries