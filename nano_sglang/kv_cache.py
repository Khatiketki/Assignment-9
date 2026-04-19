from __future__ import annotations
import torch
from typing import Dict, Tuple


class KVCache:
    def __init__(self) -> None:
        self._store: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    def update(self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        if layer_idx not in self._store:
            self._store[layer_idx] = (k_new, v_new)
        else:
            k_prev, v_prev = self._store[layer_idx]
            self._store[layer_idx] = (
                torch.cat([k_prev, k_new], dim=0),
                torch.cat([v_prev, v_new], dim=0),
            )

    def get(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if layer_idx not in self._store:
            raise KeyError(f"No cache for layer {layer_idx}")
        return self._store[layer_idx]

    def num_layers(self) -> int:
        return len(self._store)

    def seq_len(self) -> int:
        if not self._store:
            return 0
        k, _ = next(iter(self._store.values()))
        return k.shape[0]

    def clear(self) -> None:
        self._store.clear()
