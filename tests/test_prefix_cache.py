from src.runtime.prefix_cache import PrefixCache
from src.core.model_adapter import ModelAdapter


def test_prefix_cache_roundtrip():
    adapter = ModelAdapter(model_name="sshleifer/tiny-gpt2")
    enc = adapter.tokenize(["Hello world"])
    input_ids = enc["input_ids"]

    cache = PrefixCache()
    cache.set(input_ids, {"dummy": 1})
    out = cache.get(input_ids)

    assert out == {"dummy": 1}