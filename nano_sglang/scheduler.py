from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import torch
from nano_sglang.kv_cache import KVCache


@dataclass
class Request:
    request_id: int
    prompt_ids: List[int]
    max_tokens: int = 128
    kv_cache: KVCache = field(default_factory=KVCache)
    generated_ids: List[int] = field(default_factory=list)
    next_token_id: Optional[int] = None
    finished: bool = False

    @property
    def is_prefilled(self) -> bool:
        return self.next_token_id is not None

    @property
    def output_len(self) -> int:
        return len(self.generated_ids)


class Scheduler:
    def __init__(self, engine, eos_token_id: int, max_batch_size: int = 8) -> None:
        self.engine = engine
        self.eos_token_id = eos_token_id
        self.max_batch_size = max_batch_size
        self.waiting: List[Request] = []
        self.running: List[Request] = []
        self.finished: List[Request] = []

    def add_request(self, req: Request) -> None:
        self.waiting.append(req)

    def step(self) -> None:
        self._prefill_waiting()
        if self.running:
            self._decode_running()

    def run_to_completion(self) -> List[Request]:
        while self.waiting or self.running:
            self.step()
        return self.finished

    def _prefill_waiting(self) -> None:
        while self.waiting and len(self.running) < self.max_batch_size:
            req = self.waiting.pop(0)
            input_ids = torch.tensor([req.prompt_ids])
            first_token = self.engine.prefill(input_ids, req.kv_cache)
            req.next_token_id = first_token
            req.generated_ids.append(first_token)
            if first_token == self.eos_token_id or req.output_len >= req.max_tokens:
                req.finished = True
                self.finished.append(req)
            else:
                self.running.append(req)

    def _decode_running(self) -> None:
        still_running: List[Request] = []
        for req in self.running:
            next_tok = self.engine.decode_step(req.next_token_id, req.kv_cache)
            req.next_token_id = next_tok
            req.generated_ids.append(next_tok)
            if next_tok == self.eos_token_id or req.output_len >= req.max_tokens:
                req.finished = True
                self.finished.append(req)
            else:
                still_running.append(req)
        self.running = still_running
