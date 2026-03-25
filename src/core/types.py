from dataclasses import dataclass, field
from typing import List, Optional, Any


@dataclass
class InferenceRequest:
    request_id: str
    prompt: str
    max_new_tokens: int = 20
    arrival_time: float = 0.0


@dataclass
class RequestState:
    request: InferenceRequest
    generated_ids: List[int] = field(default_factory=list)
    start_time: float = 0.0
    first_token_time: Optional[float] = None
    end_time: Optional[float] = None
    finished: bool = False
    finish_reason: Optional[str] = None

    # runtime state
    prompt_input_ids: Any = None
    prompt_attention_mask: Any = None
    kv_cache: Any = None
    current_input_ids: Any = None
    current_attention_mask: Any = None

    # fine-grained metrics
    prefill_latency_s: float = 0.0
    decode_step_latencies_s: List[float] = field(default_factory=list)

    def record_token(self, token_id: int, now: float, eos_token_id: Optional[int], max_new_tokens: int) -> None:
        self.generated_ids.append(token_id)
        if self.first_token_time is None:
            self.first_token_time = now

        if eos_token_id is not None and token_id == eos_token_id:
            self.finished = True
            self.finish_reason = "eos"
            self.end_time = now
        elif len(self.generated_ids) >= max_new_tokens:
            self.finished = True
            self.finish_reason = "max_tokens"
            self.end_time = now