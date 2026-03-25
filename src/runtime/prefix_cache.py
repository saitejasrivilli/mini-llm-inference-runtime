import hashlib
from typing import Dict, Any, Optional


class PrefixCache:
    def __init__(self):
        self.cache: Dict[str, Any] = {}

    @staticmethod
    def make_key(input_ids) -> str:
        # exact-prefix cache key
        flat = input_ids.detach().cpu().view(-1).tolist()
        raw = ",".join(map(str, flat)).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    def get(self, input_ids) -> Optional[Any]:
        return self.cache.get(self.make_key(input_ids))

    def set(self, input_ids, kv_cache) -> None:
        self.cache[self.make_key(input_ids)] = kv_cache

    def clear(self) -> None:
        self.cache.clear()