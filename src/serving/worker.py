from src.core.types import InferenceRequest
from src.scheduler.continuous_batcher import RuntimeEngine


class InferenceWorker:
    def __init__(self, model_name: str = "distilgpt2"):
        self.engine = RuntimeEngine(model_name=model_name)

    def generate(self, request_id: str, prompt: str, max_new_tokens: int = 20):
        req = InferenceRequest(
            request_id=request_id,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
        )
        state = self.engine.run_kv_single(req)
        text = self.engine.adapter.decode_tokens(state.generated_ids)
        latency = (state.end_time - state.start_time) if state.end_time else 0.0
        ttft = (state.first_token_time - state.start_time) if state.first_token_time else 0.0
        return {
            "request_id": request_id,
            "text": text,
            "total_tokens": len(state.generated_ids),
            "latency_s": latency,
            "ttft_s": ttft,
            "finish_reason": state.finish_reason or "unknown",
        }