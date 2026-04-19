"""
benchmark.py  —  Part 4: Throughput Benchmark
nano-sglang   |  Week 10: Inference Systems

Measures tokens/sec vs. number of concurrent requests and compares:
  A) Sequential  — one request at a time (batch_size=1)
  B) Batched     — continuous batching with increasing batch sizes

Expected result: batched scheduler achieves higher throughput because
multiple decode steps share a single GPU forward pass, amortising the
cost of loading model weights and the KV cache from HBM.
"""

from __future__ import annotations

import time
import argparse
from typing import List

from engine import Engine
from scheduler import Scheduler, Request


# ------------------------------------------------------------------ #
#  Helper                                                              #
# ------------------------------------------------------------------ #

PROMPTS = [
    "Explain the KV cache in one sentence.",
    "What is continuous batching in LLM serving?",
    "Describe the prefill phase of transformer inference.",
    "What is inter-token latency and why does it matter?",
    "How does paged attention reduce memory fragmentation?",
    "Compare MQA, GQA, and MHA attention mechanisms.",
    "Why is decoding memory-bound rather than compute-bound?",
    "What is time-to-first-token and how is it measured?",
] * 2   # 16 prompts total — adjust as needed


def run_benchmark(
    engine: Engine,
    prompts: List[str],
    max_tokens: int,
    batch_size: int,
) -> dict:
    """
    Run all prompts through the scheduler with the given batch_size.
    Returns a dict with timing and throughput statistics.
    """
    scheduler = Scheduler(
        engine=engine,
        eos_token_id=engine.eos_token_id,
        max_batch_size=batch_size,
    )

    # Enqueue all requests
    for i, prompt in enumerate(prompts):
        req = Request(
            request_id=i,
            prompt_ids=engine.tokenizer.encode(prompt),
            max_tokens=max_tokens,
        )
        scheduler.add_request(req)

    # Run to completion and measure wall time
    t_start = time.perf_counter()
    completed = scheduler.run_to_completion()
    elapsed = time.perf_counter() - t_start

    total_tokens = sum(r.output_len for r in completed)
    throughput = total_tokens / elapsed

    return {
        "batch_size":    batch_size,
        "num_requests":  len(prompts),
        "total_tokens":  total_tokens,
        "elapsed_s":     elapsed,
        "throughput":    throughput,
    }


def print_row(res: dict) -> None:
    print(
        f"  batch={res['batch_size']:2d} | "
        f"requests={res['num_requests']:3d} | "
        f"tokens={res['total_tokens']:5d} | "
        f"time={res['elapsed_s']:6.2f}s | "
        f"throughput={res['throughput']:7.1f} tok/s"
    )


# ------------------------------------------------------------------ #
#  Main                                                                #
# ------------------------------------------------------------------ #

def main(args) -> None:
    print(f"\nLoading model: {args.model}")
    engine = Engine(model_name=args.model, device=args.device)
    print("Model loaded.\n")

    prompts = PROMPTS[:args.num_prompts]

    print("=" * 65)
    print(f"Benchmark: {len(prompts)} prompts, max_tokens={args.max_tokens}")
    print("=" * 65)

    results = []

    # ── A: Sequential baseline (batch_size=1) ─────────────────────────
    print("\n[A] Sequential  (batch_size = 1)")
    res_seq = run_benchmark(engine, prompts, args.max_tokens, batch_size=1)
    print_row(res_seq)
    results.append(res_seq)

    # ── B: Batched with increasing concurrency ─────────────────────────
    for bs in [2, 4, 8]:
        print(f"\n[B] Batched      (batch_size = {bs})")
        res = run_benchmark(engine, prompts, args.max_tokens, batch_size=bs)
        print_row(res)
        results.append(res)

    # ── Summary ────────────────────────────────────────────────────────
    seq_thr = results[0]["throughput"]
    print("\n" + "=" * 65)
    print("Throughput speedup vs. sequential baseline:")
    for res in results[1:]:
        speedup = res["throughput"] / seq_thr
        print(f"  batch={res['batch_size']:2d}  →  {speedup:.2f}x")
    print("=" * 65)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="nano-sglang throughput benchmark")
    parser.add_argument("--model",       default="Qwen/Qwen3-0.6B")
    parser.add_argument("--device",      default="cuda")
    parser.add_argument("--max-tokens",  type=int, default=50)
    parser.add_argument("--num-prompts", type=int, default=8)
    main(parser.parse_args())
