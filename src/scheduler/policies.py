from typing import List

from src.core.types import InferenceRequest


def fcfs(requests: List[InferenceRequest]) -> List[InferenceRequest]:
    return list(requests)


def shortest_prompt_first(requests: List[InferenceRequest]) -> List[InferenceRequest]:
    return sorted(requests, key=lambda r: len(r.prompt))


POLICIES = {
    "fcfs": fcfs,
    "shortest_prompt_first": shortest_prompt_first,
}