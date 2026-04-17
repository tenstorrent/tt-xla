#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Z-Image-Turbo interactive server.

Loads the full model stack once, warms up with a dummy prompt (paying the
Metal Trace capture + VAE consteval cost up front), then drops into a
REPL where the user keeps feeding prompts. Control only returns to the
prompt once the previous image has finished generating.

Outputs are written to ``outputs/out_<idx>.png`` relative to the working
directory, starting at idx=0 for the warmup run and incrementing from 1
for each user prompt.

    python server.py
    python server.py --steps 9 --seed 42
"""

import argparse
import os
import sys
import time

# Make imports work regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import readline  # noqa: F401  -- enables history/editing inside input()
except ImportError:
    pass

from generate import Models, _run_one

DEFAULT_STEPS = 9
DEFAULT_SEED  = 42
OUTPUT_DIR    = "outputs"
WARMUP_PROMPT = "a cat sitting on a mat"


def _generate(models, prompt, steps, seed, idx):
    """Run one generation and print end-to-end time + absolute path."""
    path = os.path.join(OUTPUT_DIR, f"out_{idx}.png")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    t_start = time.time()
    _run_one(models, prompt, steps, seed, path, 1, 1)
    elapsed = time.time() - t_start

    full_path = os.path.abspath(path)
    print(f"  END-TO-END: {elapsed:.2f} s  |  {full_path}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Z-Image-Turbo interactive prompt server",
    )
    parser.add_argument(
        "--steps", type=int, default=DEFAULT_STEPS,
        help=f"Denoising steps (default: {DEFAULT_STEPS})",
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"Base random seed (default: {DEFAULT_SEED}); "
             "each prompt uses seed + idx so repeats yield new images",
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1) Load models once.
    models = Models()

    # 2) Warm up so Metal Trace + VAE consteval are paid before the user waits.
    bar = "=" * 72
    print(bar)
    print(f"Warming up with dummy prompt {WARMUP_PROMPT!r}")
    print("(first run is slow: ~minutes for Metal Trace capture + VAE consteval)")
    print(bar)
    _generate(models, WARMUP_PROMPT, args.steps, args.seed, idx=0)

    # 3) REPL loop.
    print(bar)
    print("Ready. Type a prompt and press ENTER to generate.")
    print("Ctrl-D or Ctrl-C to exit.")
    print(bar)

    idx = 1
    while True:
        try:
            prompt = input(f"\nprompt [{idx}]> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break

        if not prompt:
            continue

        try:
            _generate(models, prompt, args.steps, args.seed + idx, idx)
            idx += 1
        except KeyboardInterrupt:
            print("\n  Interrupted during generation; shutting down.")
            break
        except Exception as e:
            print(f"\n  ERROR: {e}\n")


if __name__ == "__main__":
    main()
