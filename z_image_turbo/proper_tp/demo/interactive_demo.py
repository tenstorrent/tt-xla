#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Z-Image-Turbo interactive demo — load models once, generate images in a loop.

Hardware: 4x Blackhole P150, tensor-parallel across (1,4) mesh.

Usage:
    python interactive_demo.py
    python interactive_demo.py --output-dir results/ --steps 9 --seed 42

Commands (type at the prompt):
    <text>              Generate an image from the text prompt
    /seed <N>           Change the random seed
    /steps <N>          Change the number of denoising steps
    /settings           Show current seed, steps, and output directory
    /quit or /exit      Exit the demo
"""

import argparse
import os
import sys

# Reuse everything from generate.py
from generate import Models, _run_one, _output_path


def interactive_loop(steps, seed, output_dir):
    print("Loading models (this is the slow part, only happens once) ...\n")
    models = Models()

    print("=" * 60)
    print("  Z-Image-Turbo Interactive Demo")
    print("  Type a prompt to generate an image. Commands:")
    print("    /seed <N>      change seed")
    print("    /steps <N>     change denoising steps")
    print("    /settings      show current settings")
    print("    /quit           exit")
    print("=" * 60)
    print()

    count = 0

    while True:
        try:
            prompt = input("prompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not prompt:
            continue

        if prompt in ("/quit", "/exit", "/q"):
            print("Bye!")
            break

        if prompt == "/settings":
            print(f"  seed={seed}  steps={steps}  output-dir={output_dir}")
            print()
            continue

        if prompt.startswith("/seed"):
            parts = prompt.split()
            if len(parts) == 2 and parts[1].isdigit():
                seed = int(parts[1])
                print(f"  seed set to {seed}\n")
            else:
                print("  usage: /seed <N>\n")
            continue

        if prompt.startswith("/steps"):
            parts = prompt.split()
            if len(parts) == 2 and parts[1].isdigit() and int(parts[1]) > 0:
                steps = int(parts[1])
                print(f"  steps set to {steps}\n")
            else:
                print("  usage: /steps <N>  (must be > 0)\n")
            continue

        if prompt.startswith("/"):
            print(f"  unknown command: {prompt}\n")
            continue

        count += 1
        path = _output_path(prompt, count, output_dir)
        _run_one(models, prompt, steps, seed, path, count, count)


def main():
    parser = argparse.ArgumentParser(
        description="Z-Image-Turbo interactive demo — load once, generate in a loop",
    )
    parser.add_argument(
        "--output-dir", default="generated",
        help="Output directory for images (default: generated/)",
    )
    parser.add_argument(
        "--steps", type=int, default=9,
        help="Denoising steps (default: 9)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Initial random seed (default: 42)",
    )
    args = parser.parse_args()

    interactive_loop(steps=args.steps, seed=args.seed, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
