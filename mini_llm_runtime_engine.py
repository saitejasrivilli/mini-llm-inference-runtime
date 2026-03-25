import argparse
import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def current_rss_mb() -> float:
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 2)


@dataclass
class InferenceRequest:
    request_id: str
    prompt: str
    max_new_tokens: int = 20


@dataclass
class RequestState:
    request: InferenceRequest
    input_ids: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    generated_ids: List[int] = field(default_factory=list)
    kv_cache: Optional[tuple] = None
    start_time: float = 0.0
    first_token_time: Optional[float] = None
    end_time: Optional[float] = None
    finished: bool = False


class ModelAdapter:
    def __init__(self, model_name: str = "distilgpt2", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def tokenize(self, texts: List[str]):
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        return {k: v.to(self.device) for k, v in enc.items()}

    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, use_cache: bool = True):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            return_dict=True,
        )

    @torch.no_grad()
    def decode_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        past_key_values=None,
        use_cache: bool = True,
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,
        )


class KVCacheManager:
    def __init__(self):
        self.cache: Dict[str, tuple] = {}

    def set(self, request_id: str, kv):
        self.cache[request_id] = kv

    def get(self, request_id: str):
        return self.cache.get(request_id)

    def clear(self, request_id: str):
        if request_id in self.cache:
            del self.cache[request_id]


class BufferPool:
    def __init__(self):
        self.pool: Dict[Tuple[Tuple[int, ...], torch.dtype, str], List[torch.Tensor]] = {}

    def acquire(self, shape, dtype, device):
        key = (tuple(shape), dtype, str(device))
        if key in self.pool and self.pool[key]:
            return self.pool[key].pop()
        return torch.empty(shape, dtype=dtype, device=device)

    def release(self, tensor: torch.Tensor):
        key = (tuple(tensor.shape), tensor.dtype, str(tensor.device))
        self.pool.setdefault(key, []).append(tensor)


class RuntimeEngine:
    def __init__(self, model_name: str = "distilgpt2"):
        self.adapter = ModelAdapter(model_name=model_name)
        self.kv_manager = KVCacheManager()
        self.buffer_pool = BufferPool()

    @staticmethod
    def _normalize_cache(cache):
        if cache is None:
            return None
        if hasattr(cache, "to_legacy_cache"):
            return cache.to_legacy_cache()
        return cache

    def _slice_legacy_cache(self, cache, select):
        cache = self._normalize_cache(cache)
        if cache is None:
            return None
        sliced = []
        for layer in cache:
            k, v = layer[0], layer[1]
            sliced.append((k[select, ...].contiguous(), v[select, ...].contiguous()))
        return tuple(sliced)

    def _greedy_next_token(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

    def run_naive_single(self, request: InferenceRequest):
        state = RequestState(request=request)
        state.start_time = time.perf_counter()

        prompt_ids = self.adapter.tokenize([request.prompt])
        generated = prompt_ids["input_ids"]
        attention_mask = prompt_ids["attention_mask"]

        for step in range(request.max_new_tokens):
            out = self.adapter.prefill(generated, attention_mask, use_cache=False)
            next_token = self._greedy_next_token(out.logits)
            token_id = int(next_token[0, 0].item())
            state.generated_ids.append(token_id)

            if state.first_token_time is None:
                state.first_token_time = time.perf_counter()

            generated = torch.cat([generated, next_token], dim=1)
            new_mask = torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([attention_mask, new_mask], dim=1)

            if token_id == self.adapter.tokenizer.eos_token_id:
                break

        state.end_time = time.perf_counter()
        state.finished = True
        return state

    def run_kv_single(self, request: InferenceRequest):
        state = RequestState(request=request)
        state.start_time = time.perf_counter()

        enc = self.adapter.tokenize([request.prompt])
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        out = self.adapter.prefill(input_ids, attention_mask, use_cache=True)
        kv = self._normalize_cache(out.past_key_values)
        self.kv_manager.set(request.request_id, kv)

        next_token = self._greedy_next_token(out.logits)
        state.generated_ids.append(int(next_token[0, 0].item()))
        state.first_token_time = time.perf_counter()

        current_input = next_token
        current_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=attention_mask.device)],
            dim=1,
        )

        for _ in range(request.max_new_tokens - 1):
            out = self.adapter.decode_step(
                input_ids=current_input,
                attention_mask=current_mask,
                past_key_values=kv,
                use_cache=True,
            )
            kv = self._normalize_cache(out.past_key_values)
            self.kv_manager.set(request.request_id, kv)

            next_token = self._greedy_next_token(out.logits)
            token_id = int(next_token[0, 0].item())
            state.generated_ids.append(token_id)

            current_input = next_token
            current_mask = torch.cat(
                [current_mask, torch.ones((1, 1), dtype=current_mask.dtype, device=current_mask.device)],
                dim=1,
            )

            if token_id == self.adapter.tokenizer.eos_token_id:
                break

        state.end_time = time.perf_counter()
        state.finished = True
        return state

    def run_static_batch(self, requests: List[InferenceRequest]):
        states = [RequestState(request=r, start_time=time.perf_counter()) for r in requests]
        enc = self.adapter.tokenize([r.prompt for r in requests])
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        out = self.adapter.prefill(input_ids, attention_mask, use_cache=True)
        batch_kv = self._normalize_cache(out.past_key_values)
        next_tokens = self._greedy_next_token(out.logits)

        current_tokens = next_tokens
        current_mask = torch.cat(
            [attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)],
            dim=1,
        )

        active = list(range(len(requests)))
        for i in range(len(states)):
            states[i].generated_ids.append(int(next_tokens[i, 0].item()))
            states[i].first_token_time = time.perf_counter()
            states[i].kv_cache = self._slice_legacy_cache(batch_kv, [i])

        for _ in range(requests[0].max_new_tokens - 1):
            if not active:
                break

            out = self.adapter.decode_step(
                input_ids=current_tokens,
                attention_mask=current_mask,
                past_key_values=batch_kv,
                use_cache=True,
            )
            batch_kv = self._normalize_cache(out.past_key_values)
            next_tokens = self._greedy_next_token(out.logits)

            new_active = []
            for idx in active:
                token_id = int(next_tokens[idx, 0].item())
                states[idx].generated_ids.append(token_id)
                if token_id != self.adapter.tokenizer.eos_token_id and len(states[idx].generated_ids) < states[idx].request.max_new_tokens:
                    new_active.append(idx)
                else:
                    states[idx].finished = True
                    states[idx].end_time = time.perf_counter()

            current_tokens = next_tokens
            current_mask = torch.cat(
                [current_mask, torch.ones((current_mask.shape[0], 1), dtype=current_mask.dtype, device=current_mask.device)],
                dim=1,
            )
            active = new_active

        for s in states:
            if s.end_time is None:
                s.end_time = time.perf_counter()
                s.finished = True
        return states

    def run_dynamic_batch(self, requests: List[InferenceRequest], batch_size: int = 4):
        # simplified dynamic batching: micro-batches requests in arrival order
        all_states = []
        for i in range(0, len(requests), batch_size):
            chunk = requests[i:i + batch_size]
            all_states.extend(self.run_static_batch(chunk))
        return all_states


# FIX 1: moved out of RuntimeEngine class — was incorrectly indented as a method
# but lacked `self`, instantiated RuntimeEngine itself, and was called as a bare
# function from main().
def benchmark_suite(model_name: str = "distilgpt2", n_requests: int = 6, max_new_tokens: int = 20):
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
    dynamic_states = engine.run_dynamic_batch(requests, batch_size=min(4, len(requests)))

    def summarize_states(mode: str, states: List[RequestState]):
        latencies = [s.end_time - s.start_time for s in states]
        ttfts = [(s.first_token_time - s.start_time) if s.first_token_time else None for s in states]
        total_tokens = sum(len(s.generated_ids) for s in states)
        total_time = sum(latencies)

        return {
            "mode": mode,
            "n_requests": len(states),
            "total_generated_tokens": total_tokens,
            "avg_latency_s": float(np.mean(latencies)),
            "avg_ttft_s": float(np.mean([x for x in ttfts if x is not None])),
            "throughput_tok_s": float(total_tokens / total_time) if total_time > 0 else 0.0,
            "rss_mb": current_rss_mb(),
        }

    summaries = [
        summarize_states("naive_single", naive_states),
        summarize_states("kv_single", kv_states),
        summarize_states("kv_static_batch", static_states),
        summarize_states("kv_dynamic_batch", dynamic_states),
    ]
    summary_df = pd.DataFrame(summaries)
    return summary_df, summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="distilgpt2")
    parser.add_argument("--n_requests", type=int, default=6)
    parser.add_argument("--max_new_tokens", type=int, default=20)
    args = parser.parse_args()

    summary_df, summaries = benchmark_suite(
        model_name=args.model,
        n_requests=args.n_requests,
        max_new_tokens=args.max_new_tokens,
    )

    print("\n=== Benchmark Summary ===")
    print(summary_df.round(4).to_string(index=False))

    with open("benchmark_summary.json", "w") as f:
        json.dump(summaries, f, indent=2)
    print("\nSaved benchmark_summary.json")


if __name__ == "__main__":
    main()