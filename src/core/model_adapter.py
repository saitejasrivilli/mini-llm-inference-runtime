from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelAdapter:
    def __init__(self, model_name: str = "distilgpt2", device: Optional[str] = None):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def eos_token_id(self) -> Optional[int]:
        return self.tokenizer.eos_token_id

    def tokenize(self, texts: List[str]):
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        return {k: v.to(self.device) for k, v in enc.items()}

    def decode_tokens(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    @staticmethod
    def normalize_cache(cache):
        if cache is None:
            return None
        if hasattr(cache, "to_legacy_cache"):
            return cache.to_legacy_cache()
        return cache

    @staticmethod
    def slice_legacy_cache(cache, select):
        cache = ModelAdapter.normalize_cache(cache)
        if cache is None:
            return None
        sliced = []
        for layer in cache:
            k, v = layer[0], layer[1]
            sliced.append((k[select, ...].contiguous(), v[select, ...].contiguous()))
        return tuple(sliced)

    @staticmethod
    def greedy_next_token(logits: torch.Tensor) -> torch.Tensor:
        return torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

    @torch.no_grad()
    def prefill(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, use_cache: bool = True):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            return_dict=True,
        )

    @torch.no_grad()
    def decode_step(self, input_ids, attention_mask, past_key_values=None, use_cache=True):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,
        )