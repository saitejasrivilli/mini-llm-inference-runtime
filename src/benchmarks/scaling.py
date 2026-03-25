from typing import List
import pandas as pd

from src.core.types import InferenceRequest
from src.scheduler.continuous_batcher import RuntimeEngine
from src.core.metrics import summarize_states
from src.utils.io import save_csv


def make_prompt_of_length(base: str, repeat_n: int) -> str:
    return " ".join([base] * repeat_n)


def prompt_scaling_benchmark(
    model_name: str = "distilgpt2",
    repeat_sizes: List[int] = None,
    max_new_tokens: int = 16,
):
    repeat_sizes = repeat_sizes or [4, 16, 64, 128]
    engine = RuntimeEngine(model_name=model_name)

    rows = []
    for rep in repeat_sizes:
        prompt = make_prompt_of_length("Transformers use attention to model dependencies.", rep)

        req_naive = InferenceRequest(request_id=f"naive_{rep}", prompt=prompt, max_new_tokens=max_new_tokens)
        req_kv = InferenceRequest(request_id=f"kv_{rep}", prompt=prompt, max_new_tokens=max_new_tokens)

        naive_state = engine.run_naive_single(req_naive)
        kv_state = engine.run_kv_single(req_kv)

        s_naive = summarize_states("naive_single", [naive_state])
        s_kv = summarize_states("kv_single", [kv_state])

        rows.append({
            "repeat_size": rep,
            "naive_avg_latency_s": s_naive["avg_latency_s"],
            "naive_ttft_s": s_naive["avg_ttft_s"],
            "naive_throughput_tok_s": s_naive["throughput_tok_s"],
            "kv_avg_latency_s": s_kv["avg_latency_s"],
            "kv_ttft_s": s_kv["avg_ttft_s"],
            "kv_throughput_tok_s": s_kv["throughput_tok_s"],
            "kv_speedup_vs_naive": (
                s_kv["throughput_tok_s"] / s_naive["throughput_tok_s"]
                if s_naive["throughput_tok_s"] > 0 else 0.0
            ),
        })

    df = pd.DataFrame(rows)
    save_csv(df, "outputs/prompt_scaling.csv")
    return df