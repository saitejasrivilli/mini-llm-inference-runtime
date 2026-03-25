from pydantic import BaseModel


class GenerateRequest(BaseModel):
    request_id: str
    prompt: str
    max_new_tokens: int = 20


class GenerateResponse(BaseModel):
    request_id: str
    text: str
    total_tokens: int
    latency_s: float
    ttft_s: float
    finish_reason: str