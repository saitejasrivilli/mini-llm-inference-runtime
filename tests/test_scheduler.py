from src.core.types import InferenceRequest
from src.scheduler.continuous_batcher import RuntimeEngine


def test_continuous_batch_runs():
    engine = RuntimeEngine(model_name="sshleifer/tiny-gpt2")
    requests = [
        InferenceRequest(request_id=f"r{i}", prompt=f"Explain topic {i}.", max_new_tokens=4)
        for i in range(3)
    ]
    states = engine.run_continuous_batch(requests, max_batch_size=2)
    assert len(states) == 3
    assert all(len(s.generated_ids) > 0 for s in states)