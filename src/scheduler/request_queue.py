from collections import deque
from typing import List, Optional

from src.core.types import InferenceRequest


class RequestQueue:
    def __init__(self):
        self._queue = deque()

    def push(self, request: InferenceRequest) -> None:
        self._queue.append(request)

    def pop(self) -> Optional[InferenceRequest]:
        return self._queue.popleft() if self._queue else None

    def pop_many(self, n: int) -> List[InferenceRequest]:
        out = []
        for _ in range(min(n, len(self._queue))):
            out.append(self._queue.popleft())
        return out

    def __len__(self):
        return len(self._queue)

    def empty(self) -> bool:
        return len(self._queue) == 0