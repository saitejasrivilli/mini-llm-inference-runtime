from typing import Dict, Any


class KVCacheManager:
    def __init__(self):
        self.cache: Dict[str, Any] = {}

    def set(self, request_id: str, kv) -> None:
        self.cache[request_id] = kv

    def get(self, request_id: str):
        return self.cache.get(request_id)

    def clear(self, request_id: str) -> None:
        if request_id in self.cache:
            del self.cache[request_id]

    def clear_all(self) -> None:
        self.cache.clear()