from dataclasses import dataclass
from typing import List, Dict, Any

from src.core.types import InferenceRequest
from src.scheduler.continuous_batcher import RuntimeEngine


@dataclass
class SpeculativeStats:
    accepted_tokens: int = 0
    rejected_tokens: int = 0
    verifier_calls: int = 0


class SpeculativeDecoder:
    def __init__(self, draft_model_name: str = "sshleifer/tiny-gpt2", target_model_name: str = "distilgpt2"):
        self.draft_engine = RuntimeEngine(model_name=draft_model_name)
        self.target_engine = RuntimeEngine(model_name=target_model_name)

    def generate(self, request: InferenceRequest, draft_steps: int = 4) -> Dict[str, Any]:
        """
        Simplified speculative decoding:
        1. Draft model proposes a short token sequence
        2. Target model generates/validates one step at a time
        3. Count accepted/rejected tokens
        """
        stats = SpeculativeStats()

        prompt = request.prompt
        final_tokens: List[int] = []

        for _ in range(request.max_new_tokens):
            # draft proposal
            draft_req = InferenceRequest(
                request_id=f"{request.request_id}_draft",
                prompt=prompt,
                max_new_tokens=draft_steps,
            )
            draft_state = self.draft_engine.run_kv_single(draft_req)
            proposed = draft_state.generated_ids[:draft_steps]

            accepted_this_round = 0

            for token_id in proposed:
                stats.verifier_calls += 1
                verify_req = InferenceRequest(
                    request_id=f"{request.request_id}_verify_{len(final_tokens)}",
                    prompt=prompt,
                    max_new_tokens=1,
                )
                target_state = self.target_engine.run_kv_single(verify_req)
                if not target_state.generated_ids:
                    break

                target_token = target_state.generated_ids[0]
                if token_id == target_token:
                    final_tokens.append(token_id)
                    stats.accepted_tokens += 1
                    accepted_this_round += 1
                    prompt = prompt + self.target_engine.adapter.decode_tokens([token_id])
                else:
                    final_tokens.append(target_token)
                    stats.rejected_tokens += 1
                    prompt = prompt + self.target_engine.adapter.decode_tokens([target_token])
                    break

                if len(final_tokens) >= request.max_new_tokens:
                    break

            if accepted_this_round == 0:
                # fallback single verified token to avoid stalling
                verify_req = InferenceRequest(
                    request_id=f"{request.request_id}_fallback_{len(final_tokens)}",
                    prompt=prompt,
                    max_new_tokens=1,
                )
                target_state = self.target_engine.run_kv_single(verify_req)
                if target_state.generated_ids:
                    tok = target_state.generated_ids[0]
                    final_tokens.append(tok)
                    prompt = prompt + self.target_engine.adapter.decode_tokens([tok])

            if len(final_tokens) >= request.max_new_tokens:
                break

        return {
            "generated_ids": final_tokens,
            "generated_text": self.target_engine.adapter.decode_tokens(final_tokens),
            "accepted_tokens": stats.accepted_tokens,
            "rejected_tokens": stats.rejected_tokens,
            "verifier_calls": stats.verifier_calls,
            "acceptance_rate": (
                stats.accepted_tokens / max(1, stats.accepted_tokens + stats.rejected_tokens)
            ),
        }