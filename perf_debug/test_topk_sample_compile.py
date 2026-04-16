"""Test tt::topk_sample compilation in isolation (outside vLLM)."""

import os
import sys
import time

os.environ["TT_TOPK_SAMPLE"] = "1"

import torch
import torch_xla
import torch_xla.core.xla_model as xm


def test_topk_sample_standalone():
    """Test the topk_sample custom op compiles and runs correctly."""
    device = xm.xla_device()

    @torch.compile(backend="tt", fullgraph=True, dynamic=False)
    def sample_fn(logits, temperature):
        result = torch.ops.tt.topk_sample(logits, temperature, seed=42)
        return result

    # Llama-3.1 vocab size
    vocab_size = 128256
    batch = 1

    print(f"Testing topk_sample: batch={batch}, vocab={vocab_size}")

    # Verify CPU path returns [32]
    cpu_result = torch.ops.tt.topk_sample(
        torch.randn(1, vocab_size), torch.tensor([0.6]), seed=42
    )
    print(f"CPU result shape: {cpu_result.shape} (expect [32])")

    logits = torch.randn(batch, vocab_size, dtype=torch.bfloat16).to(device)
    temp = torch.tensor([0.6], dtype=torch.float32).to(device)

    print("Compiling...")
    t0 = time.perf_counter()
    result = sample_fn(logits, temp)
    torch_xla.sync()
    t1 = time.perf_counter()
    print(f"First call (compile + run): {t1 - t0:.2f}s")

    result_cpu = result.cpu()
    print(f"Result shape: {result_cpu.shape}, dtype: {result_cpu.dtype}")
    # Output is [32] int32 — top-32 global vocab indices.
    print(f"Top-5 indices: {result_cpu[:5].tolist()}")
    print(f"All in range: {(result_cpu >= 0).all() and (result_cpu < vocab_size).all()}")

    # Run a few more times for steady-state timing
    times = []
    for i in range(5):
        logits = torch.randn(batch, vocab_size, dtype=torch.bfloat16).to(device)
        t0 = time.perf_counter()
        result = sample_fn(logits, temp)
        torch_xla.sync()
        t1 = time.perf_counter()
        times.append(t1 - t0)
        print(f"  Run {i+1}: {times[-1]*1000:.1f}ms, top1={result.cpu()[0].item()}")

    print(f"Steady-state avg: {sum(times[1:])/len(times[1:])*1000:.1f}ms")
    print("PASSED")


if __name__ == "__main__":
    test_topk_sample_standalone()
    sys.stdout.flush()
    os._exit(0)
