# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Standalone sampler throughput benchmark using saved logits.

Measures wallclock ms per sampling step without running the full model —
no model compilation, no KV cache, no vLLM engine. Total startup < 30s.

Batch=32 note: logits are tiled from the saved single-user fixture. All rows
have identical distributions, which differs from a real vLLM batch where each
user has distinct logits. The sampler is row-independent so kernel dispatch
overhead and throughput are representative of real batch=32.

Usage:
    # Non-greedy, device sampling (ttnn.sampling ON by default):
    python perf_debug/test_sampler_throughput.py

    # Greedy, device sampling:
    python perf_debug/test_sampler_throughput.py --greedy

    # Non-greedy, CPU sampling:
    python perf_debug/test_sampler_throughput.py --cpu-sampling

    # Greedy, CPU sampling:
    python perf_debug/test_sampler_throughput.py --greedy --cpu-sampling

    # Disable ttnn.sampling path (use topk+gather device path instead):
    TT_USE_TTNN_SAMPLING=0 python perf_debug/test_sampler_throughput.py

    # Different batch size:
    python perf_debug/test_sampler_throughput.py --batch 32

    # With Tracy (fewer iterations):
    python -m tracy -p -r --sync-host-device -o tracy_sampler \\
        -m perf_debug.test_sampler_throughput

Fixture:
    Requires tests/integrations/vllm_plugin/sampling/fixtures/llama3_2_3b_decode_step1.pt
    Generate with: python tests/integrations/vllm_plugin/sampling/capture_logits.py
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Heavy imports (torch_xla, vLLM plugin, TTModelRunner) are deferred to
# benchmark() so --cpu-sampling starts in ~1s without XLA/plugin init.


FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__),
    "../tests/integrations/vllm_plugin/sampling/fixtures/llama3_2_3b_decode_step1.pt",
)

# Tracy mode: fewer iterations to keep trace manageable.
TRACY_MODE = os.environ.get("TRACY_PROFILING_ACTIVE", "") == "1"
N_WARMUP = 3
N_ITERS = 5 if TRACY_MODE else 100


def run_sampler(logits, metadata):
    from integrations.vllm_plugin.vllm_tt.sampler import Sampler

    sampler = Sampler()
    return sampler(logits, metadata).sampled_token_ids


def run_greedy(logits, _):
    # Mirrors the production fast-path in sample_from_logits: pure argmax,
    # no compiled sampler. This is what vLLM uses for all-greedy batches.
    return torch.argmax(logits, dim=-1, keepdim=True)


# Compiled lazily in benchmark() after torch_xla/TT backend is imported.
compiled_sampler = None
compiled_greedy = None


def make_metadata(batch: int, temperature: float, greedy: bool, device):
    from integrations.vllm_plugin.vllm_tt.metadata import XLASupportedSamplingMetadata

    return XLASupportedSamplingMetadata(
        temperature=torch.full((batch,), temperature, device=device),
        top_k=None,
        top_p=None,
        min_p=torch.zeros(batch, device=device),
        all_greedy=greedy,
    )


def timed_loop(fn, sync_fn, label: str) -> float:
    """Run fn() N_WARMUP times (untimed), then N_ITERS times (timed).
    sync_fn() is called after each timed call to force device sync.
    Returns mean ms."""
    for _ in range(N_WARMUP):
        out = fn()
        sync_fn(out)

    times_ms = []
    for _ in range(N_ITERS):
        t0 = time.perf_counter()
        out = fn()
        sync_fn(out)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000)

    times_ms.sort()
    mean_ms = sum(times_ms) / len(times_ms)
    median_ms = times_ms[len(times_ms) // 2]
    p10_ms = times_ms[int(len(times_ms) * 0.10)]
    p90_ms = times_ms[int(len(times_ms) * 0.90)]

    print(f"Results [{label}] ({N_ITERS} iters):")
    print(f"  mean   = {mean_ms:.2f} ms")
    print(f"  median = {median_ms:.2f} ms")
    print(f"  p10    = {p10_ms:.2f} ms")
    print(f"  p90    = {p90_ms:.2f} ms")

    return mean_ms


def benchmark(
    logits_base: torch.Tensor,
    batch: int,
    greedy: bool,
    cpu_sampling: bool,
):
    temperature = 0.0 if greedy else 0.6
    use_ttnn = os.environ.get("TT_USE_TTNN_SAMPLING", "1") != "0"
    mode = "greedy" if greedy else "non-greedy"
    sampler_loc = (
        "CPU"
        if cpu_sampling
        else f"device ({'ttnn.sampling' if use_ttnn else 'topk+gather'})"
    )

    print(f"\n{'='*60}")
    print(f"Sampler throughput benchmark")
    print(f"  vocab_size  = {logits_base.shape[-1]}")
    print(f"  batch       = {batch}")
    print(f"  mode        = {mode}  (temperature={temperature})")
    print(f"  sampling    = {sampler_loc}")
    print(f"  iterations  = {N_ITERS} (warmup={N_WARMUP})")
    print(f"{'='*60}")

    if cpu_sampling:
        # Uses the real production path: TTModelRunner.sample_from_logits_cpu.
        # This is the same code vLLM runs when cpu_sampling=True.
        # XLA/vLLM imports are deferred so this path starts fast.
        from integrations.vllm_plugin.vllm_tt.model_runner import TTModelRunner

        def cpu_sample(logits, metadata):
            return TTModelRunner.sample_from_logits_cpu(None, logits, metadata)

        logits = logits_base.repeat(batch, 1)  # stays on CPU
        metadata = make_metadata(batch, temperature, greedy, torch.device("cpu"))

        print("Running on CPU (real sample_from_logits_cpu path)...\n")
        mean_ms = timed_loop(
            fn=lambda: cpu_sample(logits, metadata),
            sync_fn=lambda out: None,
            label=f"cpu/{'greedy' if greedy else 'non-greedy'}",
        )
    else:
        import torch_xla.core.xla_model as xm

        global compiled_sampler, compiled_greedy
        if compiled_sampler is None:
            compiled_sampler = torch.compile(run_sampler, backend="tt", dynamic=False)
            compiled_greedy = torch.compile(run_greedy, backend="tt", dynamic=False)

        device = xm.xla_device()
        logits = logits_base.repeat(batch, 1).to(device)
        metadata = make_metadata(batch, temperature, greedy, device)

        if greedy:
            print("Compiling greedy (argmax)...")
            _ = compiled_greedy(logits, None)
            xm.mark_step()
            xm.wait_device_ops()
            print("Compile done.\n")
            mean_ms = timed_loop(
                fn=lambda: compiled_greedy(logits, None),
                sync_fn=lambda out: out.cpu(),
                label="device/greedy",
            )
        else:
            print("Compiling non-greedy sampler...")
            _ = compiled_sampler(logits, metadata)
            xm.mark_step()
            xm.wait_device_ops()
            print("Compile done.\n")
            mean_ms = timed_loop(
                fn=lambda: compiled_sampler(logits, metadata),
                sync_fn=lambda out: out.cpu(),
                label="device/non-greedy",
            )

    throughput = batch / (mean_ms / 1000)
    print(f"  throughput = {throughput:.1f} tok/s  (batch={batch} / {mean_ms:.2f}ms)")

    return mean_ms


def main():
    parser = argparse.ArgumentParser(description="Sampler throughput benchmark")
    parser.add_argument("--batch", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--greedy", action="store_true", help="Greedy mode (temperature=0)"
    )
    parser.add_argument(
        "--cpu-sampling", action="store_true", help="Run sampler on CPU"
    )
    parser.add_argument(
        "--fixture", type=str, default=FIXTURE_PATH, help="Path to logits .pt file"
    )
    args = parser.parse_args()

    if not os.path.exists(args.fixture):
        print(f"ERROR: fixture not found: {args.fixture}")
        print("Generate it with:")
        print("  python tests/integrations/vllm_plugin/sampling/capture_logits.py")
        sys.exit(1)

    t_start = time.perf_counter()

    fixture = torch.load(args.fixture, weights_only=False)
    logits = fixture["logits"].float()  # [1, vocab_size] float32
    print(
        f"Loaded fixture: vocab_size={logits.shape[-1]}, "
        f"greedy_token={fixture.get('greedy_token')}"
    )

    benchmark(
        logits_base=logits,
        batch=args.batch,
        greedy=args.greedy,
        cpu_sampling=args.cpu_sampling,
    )

    print(f"\nTotal elapsed: {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    main()
