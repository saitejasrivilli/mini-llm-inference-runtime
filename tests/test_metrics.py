from src.core.metrics import summarize_states
from src.core.types import InferenceRequest, RequestState


def test_summarize_states_basic():
    req = InferenceRequest(request_id="r1", prompt="hi", max_new_tokens=2)
    s = RequestState(request=req)
    s.start_time = 0.0
    s.first_token_time = 0.2
    s.end_time = 1.0
    s.generated_ids = [1, 2]
    out = summarize_states("test", [s])

    assert out["mode"] == "test"
    assert out["n_requests"] == 1
    assert out["total_generated_tokens"] == 2
    assert out["avg_latency_s"] == 1.0