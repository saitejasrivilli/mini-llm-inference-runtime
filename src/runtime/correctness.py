from src.core.model_adapter import ModelAdapter
from src.core.types import InferenceRequest
from src.scheduler.continuous_batcher import RuntimeEngine


def validate_single_vs_kv(model_name: str = "distilgpt2", prompt: str = "Explain KV cache."):
    engine = RuntimeEngine(model_name=model_name)
    req_a = InferenceRequest(request_id="naive", prompt=prompt, max_new_tokens=12)
    req_b = InferenceRequest(request_id="kv", prompt=prompt, max_new_tokens=12)

    naive = engine.run_naive_single(req_a)
    kv = engine.run_kv_single(req_b)

    return {
        "naive_ids": naive.generated_ids,
        "kv_ids": kv.generated_ids,
        "match": naive.generated_ids == kv.generated_ids,
    }