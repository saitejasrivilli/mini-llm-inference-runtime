from typing import Dict, Any

import torch

from src.core.types import InferenceRequest
from src.scheduler.continuous_batcher import RuntimeEngine
from src.core.metrics import summarize_states


def build_compiled_engine(model_name: str = "distilgpt2") -> RuntimeEngine:
    engine = RuntimeEngine(model_name=model_name)

    if not hasattr(torch, "compile"):
        raise RuntimeError("torch.compile is not available in this PyTorch version.")

    try:
        engine.adapter.model = torch.compile(engine.adapter.model)
    except Exception as e:
        raise RuntimeError(f"torch.compile failed: {e}")

    engine.adapter.model.eval()
    return engine


def compile_compare(
    model_name: str = "distilgpt2",
    prompt: str = "Explain the throughput-latency tradeoff in LLM inference systems.",
    max_new_tokens: int = 20,
) -> Dict[str, Any]:
    eager_engine = RuntimeEngine(model_name=model_name)
    compiled_engine = build_compiled_engine(model_name=model_name)

    req_a = InferenceRequest(request_id="eager", prompt=prompt, max_new_tokens=max_new_tokens)
    req_b = InferenceRequest(request_id="compiled", prompt=prompt, max_new_tokens=max_new_tokens)

    eager_state = eager_engine.run_kv_single(req_a)
    compiled_state = compiled_engine.run_kv_single(req_b)

    return {
        "eager": summarize_states("eager_kv_single", [eager_state]),
        "compiled": summarize_states("compiled_kv_single", [compiled_state]),
        "eager_text": eager_engine.adapter.decode_tokens(eager_state.generated_ids),
        "compiled_text": compiled_engine.adapter.decode_tokens(compiled_state.generated_ids),
    }