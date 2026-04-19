from __future__ import annotations
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Optional
from nano_sglang.kv_cache import KVCache
from nano_sglang.sampling import SamplingParams


class Engine:
    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B", device: str = "cuda", dtype: torch.dtype = torch.float16) -> None:
        self.device = device
        self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()
        self.eos_token_id: int = self.tokenizer.eos_token_id

    def prefill(self, input_ids: torch.Tensor, kv_cache: KVCache) -> int:
        """
        Process all prompt tokens at once.
        Stores the raw past_key_values object in kv_cache._store under key 'raw'.
        Returns first generated token (greedy).
        """
        input_ids = input_ids.to(self.device)
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                use_cache=True,
                return_dict=True,
            )
        # Store the raw cache object — works with any transformers version
        kv_cache._store['raw'] = outputs.past_key_values
        return outputs.logits[0, -1, :].argmax().item()

    def decode_step(self, token_id: int, kv_cache: KVCache) -> int:
        """
        One decode step — feeds only the new token.
        Passes the raw past_key_values directly back to the model.
        """
        input_ids = torch.tensor([[token_id]], device=self.device)
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                past_key_values=kv_cache._store['raw'],
                use_cache=True,
                return_dict=True,
            )
        # Update cache with new output
        kv_cache._store['raw'] = outputs.past_key_values
        return outputs.logits[0, -1, :].argmax().item()

    def generate(self, prompt: str, sampling_params: Optional[SamplingParams] = None, max_tokens: int = 128) -> str:
        """Full generation: prefill then decode loop until EOS or max_tokens."""
        if sampling_params is not None:
            max_tokens = sampling_params.max_tokens

        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        kv_cache = KVCache()

        # Phase 1: Prefill — all prompt tokens in one forward pass
        next_token = self.prefill(input_ids, kv_cache)
        generated_ids = [next_token]

        # Phase 2: Decode — one new token per step, K,V read from cache
        for _ in range(max_tokens - 1):
            if next_token == self.eos_token_id:
                break
            next_token = self.decode_step(next_token, kv_cache)
            generated_ids.append(next_token)

        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)
