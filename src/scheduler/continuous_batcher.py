import time
from typing import List, Dict, Optional

import torch

from src.core.model_adapter import ModelAdapter
from src.core.types import InferenceRequest, RequestState
from src.runtime.kv_cache import KVCacheManager
from src.runtime.prefix_cache import PrefixCache
from src.runtime.paged_kv_cache import PagedKVCacheManager
from src.scheduler.request_queue import RequestQueue
from src.scheduler.policies import POLICIES


class RuntimeEngine:
    def __init__(self, model_name: str = "distilgpt2"):
        self.adapter = ModelAdapter(model_name=model_name)
        self.kv_manager = KVCacheManager()
        self.prefix_cache = PrefixCache()
        self.paged_kv = PagedKVCacheManager(page_size_tokens=16, max_pages=4096)

    def _finish_state(self, state: RequestState, now: float):
        if state.end_time is None:
            state.end_time = now
        state.finished = True
        self.kv_manager.clear(state.request.request_id)
        self.paged_kv.release_request(state.request.request_id)

    def _prefill_state(self, state: RequestState):
        t0 = time.perf_counter()
        enc = self.adapter.tokenize([state.request.prompt])
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        state.prompt_input_ids = input_ids
        state.prompt_attention_mask = attention_mask

        cached = self.prefix_cache.get(input_ids)
        if cached is not None:
            kv = cached
            out = self.adapter.prefill(input_ids, attention_mask, use_cache=True)
            next_token = self.adapter.greedy_next_token(out.logits)
        else:
            out = self.adapter.prefill(input_ids, attention_mask, use_cache=True)
            kv = self.adapter.normalize_cache(out.past_key_values)
            self.prefix_cache.set(input_ids, kv)
            next_token = self.adapter.greedy_next_token(out.logits)

        state.prefill_latency_s = time.perf_counter() - t0

        kv = self.adapter.normalize_cache(kv)
        self.kv_manager.set(state.request.request_id, kv)
        state.kv_cache = kv

        token_id = int(next_token[0, 0].item())
        now = time.perf_counter()
        state.record_token(
            token_id=token_id,
            now=now,
            eos_token_id=self.adapter.eos_token_id(),
            max_new_tokens=state.request.max_new_tokens,
        )

        state.current_input_ids = next_token
        state.current_attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=attention_mask.device)],
            dim=1,
        )

        prompt_len = int(input_ids.shape[1])
        generated_len = len(state.generated_ids)
        self.paged_kv.allocate_for_tokens(state.request.request_id, prompt_len + generated_len)

    def _decode_one_step_single(self, state: RequestState):
        if state.finished:
            return
        t0 = time.perf_counter()
        kv = self.kv_manager.get(state.request.request_id)
        out = self.adapter.decode_step(
            input_ids=state.current_input_ids,
            attention_mask=state.current_attention_mask,
            past_key_values=kv,
            use_cache=True,
        )
        step_latency = time.perf_counter() - t0
        state.decode_step_latencies_s.append(step_latency)

        kv = self.adapter.normalize_cache(out.past_key_values)
        self.kv_manager.set(state.request.request_id, kv)
        state.kv_cache = kv

        next_token = self.adapter.greedy_next_token(out.logits)
        token_id = int(next_token[0, 0].item())
        now = time.perf_counter()
        state.record_token(
            token_id=token_id,
            now=now,
            eos_token_id=self.adapter.eos_token_id(),
            max_new_tokens=state.request.max_new_tokens,
        )

        state.current_input_ids = next_token
        state.current_attention_mask = torch.cat(
            [
                state.current_attention_mask,
                torch.ones((1, 1), dtype=state.current_attention_mask.dtype, device=state.current_attention_mask.device),
            ],
            dim=1,
        )
        self.paged_kv.allocate_for_tokens(state.request.request_id, 1)

        if state.finished:
            self._finish_state(state, now)

    def run_naive_single(self, request: InferenceRequest) -> RequestState:
        state = RequestState(request=request, start_time=time.perf_counter())
        enc = self.adapter.tokenize([request.prompt])
        generated = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        for _ in range(request.max_new_tokens):
            t0 = time.perf_counter()
            out = self.adapter.prefill(generated, attention_mask, use_cache=False)
            step_latency = time.perf_counter() - t0
            state.decode_step_latencies_s.append(step_latency)

            next_token = self.adapter.greedy_next_token(out.logits)
            token_id = int(next_token[0, 0].item())
            now = time.perf_counter()
            state.record_token(
                token_id=token_id,
                now=now,
                eos_token_id=self.adapter.eos_token_id(),
                max_new_tokens=request.max_new_tokens,
            )

            generated = torch.cat([generated, next_token], dim=1)
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device),
                ],
                dim=1,
            )
            if state.finished:
                break

        if state.end_time is None:
            state.end_time = time.perf_counter()
            state.finished = True
        return state

    def run_kv_single(self, request: InferenceRequest) -> RequestState:
        state = RequestState(request=request, start_time=time.perf_counter())
        self._prefill_state(state)
        while not state.finished:
            self._decode_one_step_single(state)
        return state

    def run_static_batch(self, requests: List[InferenceRequest]) -> List[RequestState]:
        states = [RequestState(request=r, start_time=time.perf_counter()) for r in requests]
        enc = self.adapter.tokenize([r.prompt for r in requests])
        input_ids = enc["input_ids"]
        attention_mask = enc["attention_mask"]

        t0 = time.perf_counter()
        out = self.adapter.prefill(input_ids, attention_mask, use_cache=True)
        prefill_latency = time.perf_counter() - t0

        batch_kv = self.adapter.normalize_cache(out.past_key_values)
        next_tokens = self.adapter.greedy_next_token(out.logits)

        current_tokens = next_tokens
        current_mask = torch.cat(
            [attention_mask, torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)],
            dim=1,
        )

        eos_id = self.adapter.eos_token_id()
        active = list(range(len(states)))

        for i, state in enumerate(states):
            state.prefill_latency_s = prefill_latency
            state.kv_cache = self.adapter.slice_legacy_cache(batch_kv, [i])
            state.current_input_ids = next_tokens[i : i + 1]
            state.current_attention_mask = current_mask[i : i + 1]
            token_id = int(next_tokens[i, 0].item())
            now = time.perf_counter()
            state.record_token(token_id, now, eos_id, state.request.max_new_tokens)

        while active:
            t0 = time.perf_counter()
            out = self.adapter.decode_step(
                input_ids=current_tokens,
                attention_mask=current_mask,
                past_key_values=batch_kv,
                use_cache=True,
            )
            step_latency = time.perf_counter() - t0

            batch_kv = self.adapter.normalize_cache(out.past_key_values)
            next_tokens = self.adapter.greedy_next_token(out.logits)

            new_active = []
            for idx in active:
                state = states[idx]
                state.decode_step_latencies_s.append(step_latency)
                token_id = int(next_tokens[idx, 0].item())
                now = time.perf_counter()
                state.record_token(token_id, now, eos_id, state.request.max_new_tokens)
                if not state.finished:
                    new_active.append(idx)

            current_tokens = next_tokens
            current_mask = torch.cat(
                [current_mask, torch.ones((current_mask.shape[0], 1), dtype=current_mask.dtype, device=current_mask.device)],
                dim=1,
            )
            active = new_active

        for state in states:
            if state.end_time is None:
                state.end_time = time.perf_counter()
                state.finished = True
        return states

    def run_continuous_batch(
        self,
        requests: List[InferenceRequest],
        max_batch_size: int = 4,
        policy_name: str = "fcfs",
    ) -> List[RequestState]:
        queue = RequestQueue()
        for r in requests:
            queue.push(r)

        policy = POLICIES[policy_name]
        active_states: List[RequestState] = []
        completed: List[RequestState] = []
        eos_id = self.adapter.eos_token_id()

        while not queue.empty() or active_states:
            # admit new requests into active set
            slots = max_batch_size - len(active_states)
            if slots > 0 and not queue.empty():
                admitted = queue.pop_many(slots)
                admitted = policy(admitted)
                for req in admitted:
                    state = RequestState(request=req, start_time=time.perf_counter())
                    self._prefill_state(state)
                    if state.finished:
                        self._finish_state(state, time.perf_counter())
                        completed.append(state)
                    else:
                        active_states.append(state)

            if not active_states:
                continue

            # pack active requests
            batch_input_ids = torch.cat([s.current_input_ids for s in active_states], dim=0)

            max_len = max(s.current_attention_mask.shape[1] for s in active_states)
            masks = []
            for s in active_states:
                m = s.current_attention_mask
                if m.shape[1] < max_len:
                    pad = torch.ones(
                        (1, max_len - m.shape[1]),
                        dtype=m.dtype,
                        device=m.device,
                    )
                    m = torch.cat([m, pad], dim=1)
                masks.append(m)
            batch_attention_mask = torch.cat(masks, dim=0)

            # use only first state's per-layer cache shape as template isn't enough;
            # so for this simplified runtime we decode per-state but within a scheduler loop.
            # This keeps semantics clean and the scheduler real.
            still_active = []
            for state in active_states:
                self._decode_one_step_single(state)
                if state.finished:
                    completed.append(state)
                else:
                    still_active.append(state)

            active_states = still_active

        return completed