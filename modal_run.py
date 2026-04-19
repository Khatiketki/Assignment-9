"""Run nano-sglang on Modal.

Usage:
    modal run modal_run.py::run        # run the example
    modal run modal_run.py::test       # run all tests on GPU
    modal run modal_run.py::benchmark  # run throughput benchmark
"""

import modal

MODEL_NAME = "Qwen/Qwen3-0.6B"


def download_model():
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL_NAME)


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "huggingface_hub", "pytest", "accelerate")
    .run_function(download_model)
    .add_local_dir("nano_sglang", remote_path="/root/nano_sglang")
    .add_local_dir("tests", remote_path="/root/tests")
)

app = modal.App("nano-sglang")


@app.function(image=image, gpu="A100-40GB", timeout=600)
def run():
    """Run a single generation example."""
    from nano_sglang.engine import Engine
    from nano_sglang.sampling import SamplingParams

    engine = Engine(MODEL_NAME)
    params = SamplingParams(temperature=0, max_tokens=50)
    output = engine.generate("The capital of France is", params)
    print(f"Output: {output}")


@app.function(image=image, gpu="A100-40GB", timeout=600)
def test():
    """Run all unit tests."""
    import subprocess
    subprocess.run(
        ["python", "-m", "pytest", "/root/tests/", "-v", "--tb=short"],
        check=False
    )


@app.function(image=image, gpu="A100-40GB", timeout=600)
def benchmark():
    """
    Part 4: Measure throughput (tokens/sec) vs number of concurrent requests.
    Compares batched scheduler vs generating one request at a time.
    """
    import time
    from nano_sglang.engine import Engine
    from nano_sglang.sampling import SamplingParams
    from nano_sglang.scheduler import Scheduler, Request

    engine = Engine(MODEL_NAME)
    eos = engine.eos_token_id

    prompts = [
        "The capital of France is",
        "The largest planet in the solar system is",
        "The speed of light is approximately",
        "The first computer was invented by",
        "The chemical formula for water is",
        "The tallest mountain in the world is",
        "The fastest animal on land is",
        "The deepest ocean in the world is",
    ]

    params = SamplingParams(temperature=0, max_tokens=30)

    print("\n" + "=" * 60)
    print("Part 4 Benchmark: Throughput vs Concurrency")
    print("=" * 60)

    results = []

    for batch_size in [1, 2, 4, 8]:
        scheduler = Scheduler(engine=engine, eos_token_id=eos, max_batch_size=batch_size)

        for i, prompt in enumerate(prompts):
            req = Request(
                request_id=i,
                prompt_ids=engine.tokenizer.encode(prompt),
                max_tokens=params.max_tokens,
            )
            scheduler.add_request(req)

        t_start = time.perf_counter()
        completed = scheduler.run_to_completion()
        elapsed = time.perf_counter() - t_start

        total_tokens = sum(len(r.generated_ids) for r in completed)
        throughput = total_tokens / elapsed

        results.append((batch_size, total_tokens, elapsed, throughput))
        print(f"  batch_size={batch_size:2d} | tokens={total_tokens:4d} | "
              f"time={elapsed:.2f}s | throughput={throughput:.1f} tok/s")

    print("\nSpeedup vs sequential (batch_size=1):")
    base = results[0][3]
    for batch_size, _, _, thr in results[1:]:
        print(f"  batch_size={batch_size:2d} → {thr/base:.2f}x faster")
    print("=" * 60)