import os
from typing import List, Dict, Any

import numpy as np
import psutil

from src.core.types import RequestState


def current_rss_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)


def peak_decode_latency(states: List[RequestState]) -> float:
    vals = []
    for s in states:
        vals.extend(s.decode_step_latencies_s)
    return float(max(vals)) if vals else 0.0


def avg_decode_latency(states: List[RequestState]) -> float:
    vals = []
    for s in states:
        vals.extend(s.decode_step_latencies_s)
    return float(np.mean(vals)) if vals else 0.0


def avg_prefill_latency(states: List[RequestState]) -> float:
    vals = [s.prefill_latency_s for s in states]
    return float(np.mean(vals)) if vals else 0.0


def summarize_states(mode: str, states: List[RequestState]) -> Dict[str, Any]:
    latencies = [s.end_time - s.start_time for s in states if s.end_time is not None]
    ttfts = [
        s.first_token_time - s.start_time
        for s in states
        if s.first_token_time is not None
    ]
    total_tokens = sum(len(s.generated_ids) for s in states)
    total_time = sum(latencies)

    return {
        "mode": mode,
        "n_requests": len(states),
        "total_generated_tokens": total_tokens,
        "avg_latency_s": float(np.mean(latencies)) if latencies else 0.0,
        "p95_latency_s": float(np.percentile(latencies, 95)) if latencies else 0.0,
        "avg_ttft_s": float(np.mean(ttfts)) if ttfts else 0.0,
        "avg_prefill_latency_s": avg_prefill_latency(states),
        "avg_decode_step_latency_s": avg_decode_latency(states),
        "peak_decode_step_latency_s": peak_decode_latency(states),
        "throughput_tok_s": float(total_tokens / total_time) if total_time > 0 else 0.0,
        "rss_mb": current_rss_mb(),
    }