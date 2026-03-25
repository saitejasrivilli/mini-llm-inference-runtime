from src.runtime.correctness import validate_single_vs_kv


def test_naive_vs_kv_match():
    out = validate_single_vs_kv(model_name="sshleifer/tiny-gpt2", prompt="Explain caching.")
    assert out["match"] is True