#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# Instrumented bleed repro that dumps block tables on failure.
#
# Monkey-patches the model runner to log page_table contents
# before each decode step, then runs the standard bleed test.
# On failure, prints the block table that was in use.
#
# Usage:
#   python3 dump_block_table.py

import os
import sys
import time
from datetime import datetime

os.environ["TTXLA_LOGGER_LEVEL"] = "WARNING"

import torch
import vllm

MODEL = os.environ.get("MODEL", "meta-llama/Llama-3.2-1B-Instruct")

PROMPTS = [
    "Explain how penguins survive in Antarctica. Always use the word penguin.",
    "Explain how volcanoes form islands. Always use the word volcano.",
    "Describe how to make an origami crane. Always use the word origami.",
    "Describe how submarine sonar works. Always use the word submarine.",
    "Explain how dinosaurs went extinct. Always use the word dinosaur.",
    "Describe different types of chocolate and their flavors. Always use the word chocolate.",
    "Explain how castles were built in the Middle Ages. Always use the word castle.",
    "Describe different types of galaxies in the universe. Always use the word galaxy.",
    "Explain how dolphins communicate underwater. Always use the word dolphin.",
    "Describe how earthquakes are measured on the Richter scale. Always use the word earthquake.",
    "Explain the basic rules of cricket. Always use the word cricket.",
    "Describe how lightning forms in thunderstorms. Always use the word lightning.",
    "Explain how pyramids were built in ancient Egypt. Always use the word pyramid.",
    "Describe how telescopes work to observe distant stars. Always use the word telescope.",
    "Explain how glaciers shape mountain landscapes. Always use the word glacier.",
    "Describe how coral reefs form in tropical oceans. Always use the word coral.",
]
KEYWORDS = [
    "penguin",
    "volcano",
    "origami",
    "submarine",
    "dinosaur",
    "chocolate",
    "castle",
    "galaxy",
    "dolphin",
    "earthquake",
    "cricket",
    "lightning",
    "pyramid",
    "telescope",
    "glacier",
    "coral",
]

BATCH_SIZE = len(PROMPTS)

# Storage for captured block tables
captured_block_tables = []


def patch_model_runner():
    """Monkey-patch the model runner to capture block tables during decode."""
    from vllm_tt.model_runner import TTModelRunner

    original_execute = TTModelRunner.execute_model

    def instrumented_execute(self, scheduler_output, *args, **kwargs):
        # Capture the block table before execution
        try:
            num_reqs = self.input_batch.num_reqs
            if num_reqs > 1:  # Only capture batched decode
                bt = self.input_batch.block_table[0].get_cpu_tensor()[:num_reqs].clone()
                seq_lens = (
                    self.input_batch.seq_lens_cpu[:num_reqs].clone()
                    if hasattr(self.input_batch, "seq_lens_cpu")
                    else None
                )
                captured_block_tables.append(
                    {
                        "block_table": bt,
                        "num_reqs": num_reqs,
                        "seq_lens": seq_lens,
                    }
                )
        except Exception as e:
            pass  # Don't crash on instrumentation failures

        return original_execute(self, scheduler_output, *args, **kwargs)

    TTModelRunner.execute_model = instrumented_execute
    print("Model runner patched for block table capture.")


def check_bleed(responses):
    bleeds = []
    for i, response in enumerate(responses):
        for j, kw in enumerate(KEYWORDS):
            if i != j and kw.lower() in response.lower():
                bleeds.append((j, i, kw))
    return bleeds


def main():
    num_runs = int(sys.argv[1]) if len(sys.argv) > 1 else 30

    print(f"Model:         {MODEL}")
    print(f"Batch size:    {BATCH_SIZE}")
    print(f"Runs:          {num_runs}")
    print()

    # Patch before creating engine
    patch_model_runner()

    llm = vllm.LLM(
        model=MODEL,
        max_model_len=512,
        max_num_batched_tokens=512 * BATCH_SIZE,
        max_num_seqs=BATCH_SIZE,
        gpu_memory_utilization=0.05,
        disable_log_stats=True,
        additional_config={
            "enable_const_eval": True,
            "min_context_len": 32,
            "experimental_weight_dtype": "bfp8",
            "cpu_sampling": True,
        },
    )

    sampling_params = vllm.SamplingParams(max_tokens=64, temperature=0.0)
    llm.generate(["Hello"], vllm.SamplingParams(max_tokens=8, temperature=0.0))
    print("Ready.\n")

    passes = 0
    fails = 0

    for run in range(1, num_runs + 1):
        # Clear captured tables for this run
        captured_block_tables.clear()

        t0 = time.perf_counter()
        outputs = llm.generate(PROMPTS, sampling_params)
        dt = time.perf_counter() - t0
        responses = [o.outputs[0].text for o in outputs]

        ts = datetime.now().strftime("%H:%M:%S")
        bleeds = check_bleed(responses)

        if bleeds:
            fails += 1
            for src, vic, kw in bleeds:
                print(
                    f"  BLEED: '{kw}' (slot {src}/{KEYWORDS[src]}) found in slot {vic}/{KEYWORDS[vic]}"
                )
                print(f"    Response: {responses[vic][:150]}...")

            # Dump captured block tables
            print(f"\n  Captured {len(captured_block_tables)} block table snapshots:")
            for idx, bt_info in enumerate(captured_block_tables[-3:]):  # Last 3
                print(
                    f"\n  === Block table snapshot {idx} (num_reqs={bt_info['num_reqs']}) ==="
                )
                bt = bt_info["block_table"]
                for row in range(min(bt.shape[0], BATCH_SIZE)):
                    blocks = bt[row].tolist()
                    print(f"    Slot {row:2d}: {blocks}")
                if bt_info["seq_lens"] is not None:
                    print(f"    Seq lens: {bt_info['seq_lens'].tolist()}")

            print(f"\n[{ts}] Run {run}/{num_runs} ({dt:.1f}s) FAIL\n")
        else:
            passes += 1
            print(f"[{ts}] Run {run}/{num_runs} ({dt:.1f}s) PASS  [{passes}P/{fails}F]")

    print()
    print("============================================")
    print(f"Results: {passes} PASS / {fails} FAIL out of {num_runs}")
    if fails > 0:
        print(f"Failure rate: {fails / num_runs * 100:.1f}%")
    print("============================================")


if __name__ == "__main__":
    main()
