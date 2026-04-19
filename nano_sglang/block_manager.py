from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Block:
    block_id: int
    ref_count: int = 0


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int = 16) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        self._all_blocks: List[Block] = [Block(i) for i in range(num_blocks)]
        self._free_blocks: List[Block] = list(self._all_blocks)
        self._req_table: Dict[int, List[int]] = {}

    def allocate(self, request_id: int, num_tokens: int) -> List[int]:
        blocks_needed = (num_tokens + self.block_size - 1) // self.block_size
        if len(self._free_blocks) < blocks_needed:
            raise MemoryError(f"Not enough blocks: need {blocks_needed}, have {len(self._free_blocks)}")
        allocated = []
        for _ in range(blocks_needed):
            block = self._free_blocks.pop()
            block.ref_count = 1
            allocated.append(block)
        block_ids = [b.block_id for b in allocated]
        self._req_table[request_id] = block_ids
        return block_ids

    def free(self, request_id: int) -> None:
        if request_id not in self._req_table:
            return
        for bid in self._req_table.pop(request_id):
            block = self._all_blocks[bid]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._free_blocks.append(block)

    def num_free_blocks(self) -> int:
        return len(self._free_blocks)

    def num_used_blocks(self) -> int:
        return self.num_blocks - self.num_free_blocks()

    def can_allocate(self, num_tokens: int) -> bool:
        return len(self._free_blocks) >= (num_tokens + self.block_size - 1) // self.block_size

    def get_block_table(self, request_id: int) -> List[int]:
        return self._req_table.get(request_id, [])
