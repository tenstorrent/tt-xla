"""Standalone repro: ttnn.sampling produces clustered outputs across the
32-row batch dim instead of independent per-row samples.

Root cause: the kernel passes a single scalar `seed` compile-time arg to
all 32 cores. Each core's SFPU RNG is initialized identically (or skipped
identically when seed=0), so all rows draw correlated random values.

Repro design:
  - Build 32 batch rows with an identical uniform input distribution.
  - With proper per-row RNG, expect ~28-32 distinct output indices
    (independent draws from 128 candidates — birthday problem says
    ~5-6 collisions in 32 draws from 128 buckets).
  - With shared/correlated RNG, observe heavy clustering (1-6 unique values).

Run on a system with at least one TT chip:

    source /path/to/tt-metal/python_env/bin/activate
    python repro_ttnn_sampling_diversity.py

Reference the kernel code at
  ttnn/cpp/ttnn/operations/reduction/sampling/device/
    sampling_program_factory.cpp     (lines 25-69, 300-319)
    kernels/compute/sampling.cpp      (lines 26-46, 368-372)
"""

import torch
import ttnn

BATCH = 32  # ttnn.sampling requires exactly N*C*H == 32 ("32 users")
CANDIDATES = 128  # last dim, must be multiple of 32


def main() -> None:
    device = ttnn.open_device(device_id=0)
    try:
        # Identical uniform distribution across all 32 rows. With proper
        # per-row RNG, every row independently samples one of `CANDIDATES`
        # tokens, giving ~28-32 distinct outputs in 32 rows.
        values = torch.zeros(1, 1, BATCH, CANDIDATES, dtype=torch.bfloat16)
        indices = (
            torch.arange(CANDIDATES, dtype=torch.int32)
            .unsqueeze(0)
            .expand(BATCH, -1)
            .reshape(1, 1, BATCH, CANDIDATES)
            .contiguous()
        )
        k = torch.full((BATCH,), 32, dtype=torch.int32)
        p = torch.ones(BATCH, dtype=torch.bfloat16)
        temp = torch.ones(BATCH, dtype=torch.bfloat16)

        values_dev = ttnn.from_torch(values, device=device, layout=ttnn.TILE_LAYOUT)
        indices_dev = ttnn.from_torch(
            indices, device=device, layout=ttnn.ROW_MAJOR_LAYOUT
        )
        k_dev = ttnn.from_torch(k, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
        p_dev = ttnn.from_torch(p, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)
        temp_dev = ttnn.from_torch(temp, device=device, layout=ttnn.ROW_MAJOR_LAYOUT)

        # ------------------------------------------------------------------
        # Case 1: seed=0  -> kernel skips rand_tile_init, RNG advances naturally.
        # All 32 cores' SFPU RNG starts in the same state; advances in lockstep.
        # ------------------------------------------------------------------
        print(f"=== Case 1: seed=0, uniform distribution, batch={BATCH} ===")
        out = ttnn.sampling(values_dev, indices_dev, k_dev, p_dev, temp_dev, seed=0)
        samples = ttnn.to_torch(out).flatten().tolist()
        distinct = len(set(samples))
        print(f"Output samples: {samples}")
        print(f"Distinct values across {BATCH} rows: {distinct}")
        print(f"Expected (independent per-row RNG): ~28-32 distinct")
        if distinct < 10:
            print(f"-> BUG REPRODUCED (only {distinct} distinct out of {BATCH} rows)")
        print()

        # Second call with seed=0 — RNG has advanced, so output should differ
        # from the first call (this part may work even with the row-clustering bug).
        out2 = ttnn.sampling(values_dev, indices_dev, k_dev, p_dev, temp_dev, seed=0)
        samples2 = ttnn.to_torch(out2).flatten().tolist()
        print(f"Second run with seed=0 (RNG advanced): {samples2}")
        print(f"Identical to first run: {samples == samples2}")
        print()

        # ------------------------------------------------------------------
        # Case 2: seed=42 -> kernel calls rand_tile_init(42) on every call,
        # every core. All 32 cores produce identical samples by design.
        # Calling twice with the same seed should give the same output, but
        # the per-row diversity is destroyed.
        # ------------------------------------------------------------------
        print(f"=== Case 2: seed=42, uniform distribution, batch={BATCH} ===")
        out3 = ttnn.sampling(values_dev, indices_dev, k_dev, p_dev, temp_dev, seed=42)
        samples3 = ttnn.to_torch(out3).flatten().tolist()
        distinct3 = len(set(samples3))
        print(f"Output samples: {samples3}")
        print(f"Distinct values across {BATCH} rows: {distinct3}")
        print(f"Expected (independent per-row RNG, seeded): ~28-32 distinct")
        if distinct3 == 1:
            print(
                "-> All 32 rows byte-identical (rand_tile_init resets all cores "
                "to the same seed)."
            )
        print()
    finally:
        ttnn.close_device(device)


if __name__ == "__main__":
    main()
