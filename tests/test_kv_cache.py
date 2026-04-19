"""
tests/test_kv_cache.py
nano-sglang  |  Week 10: Inference Systems

Run with:
    pytest tests/test_kv_cache.py -v          # local, no GPU needed
    modal run modal_run.py::test              # all tests on GPU
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
from nano_sglang.kv_cache import KVCache
from nano_sglang.sampling import SamplingParams
from nano_sglang.block_manager import BlockManager


# ------------------------------------------------------------------ #
#  KVCache tests  (Part 1)                                            #
# ------------------------------------------------------------------ #

class TestKVCache:

    def test_update_single_layer(self):
        """update() stores tensors for a new layer."""
        cache = KVCache()
        k = torch.randn(4, 2, 8)   # (seq=4, heads=2, head_dim=8)
        v = torch.randn(4, 2, 8)
        cache.update(0, k, v)

        k_out, v_out = cache.get(0)
        assert torch.allclose(k_out, k)
        assert torch.allclose(v_out, v)

    def test_update_appends_tokens(self):
        """Calling update() twice appends along the sequence dimension."""
        cache = KVCache()
        k1 = torch.randn(3, 2, 8)
        v1 = torch.randn(3, 2, 8)
        k2 = torch.randn(1, 2, 8)
        v2 = torch.randn(1, 2, 8)

        cache.update(0, k1, v1)
        cache.update(0, k2, v2)

        k_out, v_out = cache.get(0)
        assert k_out.shape[0] == 4   # 3 + 1 tokens
        assert v_out.shape[0] == 4
        assert torch.allclose(k_out[:3], k1)
        assert torch.allclose(k_out[3:], k2)

    def test_get_missing_layer_raises(self):
        """get() raises KeyError for a layer that was never updated."""
        cache = KVCache()
        with pytest.raises(KeyError):
            cache.get(99)

    def test_multiple_layers(self):
        """update/get work independently across multiple layers."""
        cache = KVCache()
        for layer in range(4):
            k = torch.ones(2, 1, 4) * layer
            v = torch.ones(2, 1, 4) * layer
            cache.update(layer, k, v)

        assert cache.num_layers() == 4
        for layer in range(4):
            k_out, v_out = cache.get(layer)
            assert torch.allclose(k_out, torch.ones(2, 1, 4) * layer)

    def test_seq_len(self):
        """seq_len() returns the current number of cached tokens."""
        cache = KVCache()
        assert cache.seq_len() == 0

        cache.update(0, torch.randn(5, 2, 8), torch.randn(5, 2, 8))
        assert cache.seq_len() == 5

        cache.update(0, torch.randn(1, 2, 8), torch.randn(1, 2, 8))
        assert cache.seq_len() == 6

    def test_clear(self):
        """clear() resets the cache."""
        cache = KVCache()
        cache.update(0, torch.randn(3, 2, 8), torch.randn(3, 2, 8))
        cache.clear()
        assert cache.seq_len() == 0
        assert cache.num_layers() == 0

    def test_prefill_example_from_lecture(self):
        """
        Verify the exact numbers from lecture Image 5:
        k1 = [1,0,2] @ W^K = [3,2]
        k2 = [0,1,1] @ W^K = [1,2]
        stored as K_cache = [[3,2],[1,2]]
        """
        cache = KVCache()
        # Store lecture example values directly
        k_prefill = torch.tensor([[3., 2.], [1., 2.]])   # (2, 2)  seq=2, head_dim=2
        v_prefill = torch.tensor([[3., 1.], [1., 1.]])
        # Reshape to (seq, heads=1, head_dim=2)
        cache.update(0, k_prefill.unsqueeze(1), v_prefill.unsqueeze(1))

        k_out, v_out = cache.get(0)
        assert k_out.shape == (2, 1, 2)
        assert torch.allclose(k_out[:, 0, :], k_prefill)

        # Now append decode step 1: k3 = [1,1]
        k3 = torch.tensor([[1., 1.]]).unsqueeze(1)   # (1, 1, 2)
        v3 = torch.tensor([[1., 2.]]).unsqueeze(1)
        cache.update(0, k3, v3)

        k_full, _ = cache.get(0)
        assert k_full.shape == (3, 1, 2)   # 2 prompt + 1 decode token


# ------------------------------------------------------------------ #
#  SamplingParams tests  (Part 2 helper)                              #
# ------------------------------------------------------------------ #

class TestSamplingParams:

    def test_defaults(self):
        params = SamplingParams()
        assert params.temperature == 0.0
        assert params.max_tokens == 128
        assert params.top_p == 1.0
        assert params.top_k == -1

    def test_custom(self):
        params = SamplingParams(temperature=0.7, max_tokens=50)
        assert params.temperature == 0.7
        assert params.max_tokens == 50


# ------------------------------------------------------------------ #
#  BlockManager tests  (Part 5 stretch)                               #
# ------------------------------------------------------------------ #

class TestBlockManager:

    def test_initial_state(self):
        bm = BlockManager(num_blocks=16, block_size=4)
        assert bm.num_free_blocks() == 16
        assert bm.num_used_blocks() == 0

    def test_allocate_returns_block_ids(self):
        bm = BlockManager(num_blocks=16, block_size=4)
        ids = bm.allocate(request_id=1, num_tokens=4)
        assert len(ids) == 1   # exactly 1 block for 4 tokens with block_size=4
        assert bm.num_free_blocks() == 15

    def test_allocate_multiple_blocks(self):
        bm = BlockManager(num_blocks=16, block_size=4)
        ids = bm.allocate(request_id=2, num_tokens=9)
        assert len(ids) == 3   # ceil(9/4) = 3 blocks
        assert bm.num_free_blocks() == 13

    def test_free_returns_blocks(self):
        bm = BlockManager(num_blocks=16, block_size=4)
        bm.allocate(request_id=1, num_tokens=8)
        assert bm.num_free_blocks() == 14
        bm.free(request_id=1)
        assert bm.num_free_blocks() == 16

    def test_free_nonexistent_is_safe(self):
        bm = BlockManager(num_blocks=8, block_size=4)
        bm.free(request_id=999)   # should not raise

    def test_out_of_memory_raises(self):
        bm = BlockManager(num_blocks=2, block_size=4)
        with pytest.raises(MemoryError):
            bm.allocate(request_id=1, num_tokens=100)

    def test_can_allocate(self):
        bm = BlockManager(num_blocks=4, block_size=4)
        assert bm.can_allocate(16) is True
        assert bm.can_allocate(17) is False

    def test_get_block_table(self):
        bm = BlockManager(num_blocks=8, block_size=4)
        ids = bm.allocate(request_id=5, num_tokens=8)
        assert bm.get_block_table(5) == ids
        bm.free(5)
        assert bm.get_block_table(5) == []

    def test_reuse_freed_blocks(self):
        """Freed blocks can be reallocated to new requests."""
        bm = BlockManager(num_blocks=4, block_size=4)
        bm.allocate(request_id=1, num_tokens=16)   # uses all 4 blocks
        assert bm.num_free_blocks() == 0
        bm.free(request_id=1)
        assert bm.num_free_blocks() == 4
        # Should be able to allocate again
        bm.allocate(request_id=2, num_tokens=16)
        assert bm.num_free_blocks() == 0
