# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Serving-telemetry demo for the vLLM TT plugin.

Runs a small batch of prompts through Qwen3-0.6B with telemetry enabled, then verifies the JSON-lines sinks were written and
prints a short summary. Enabling telemetry here goes through ``additional_config``
so the script is self-contained; the equivalent env-var form is in the README.

    source venv/activate
    python examples/vllm/telemetry/run_telemetry_demo.py

Visualize the result with:

    python scripts/telemetry/telemetry_viz.py report --dir tt_telemetry/demo   # static HTML
    python scripts/telemetry/telemetry_viz.py live   --dir tt_telemetry/demo   # live dashboard
"""
import argparse
import json
from pathlib import Path

import vllm

PROMPTS = [
    "Write a haiku about the ocean.",
    "List three prime numbers.",
    "What is the capital of France?",
    "Explain gravity in one sentence.",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="tt_telemetry/demo", help="telemetry output dir")
    ap.add_argument("--max-num-seqs", type=int, default=4, help="batch capacity")
    ap.add_argument("--max-tokens", type=int, default=32, help="tokens per request")
    args = ap.parse_args()

    tele_dir = Path(args.dir).resolve()
    max_model_len = 1024
    # Single-shot prefill: the budget must cover a full prompt per row
    # (max_num_batched_tokens >= max_model_len * max_num_seqs).
    llm = vllm.LLM(
        model="Qwen/Qwen3-0.6B",
        max_model_len=max_model_len,
        max_num_batched_tokens=max_model_len * args.max_num_seqs,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=0.2,
        additional_config={
            "enable_const_eval": False,
            "min_context_len": 256,
            # Turn telemetry on; v2 is the default runner on this branch.
            "telemetry_enabled": True,
            "telemetry_dir": str(tele_dir),
            "telemetry_flush_ms": 200,
        },
    )

    params = vllm.SamplingParams(temperature=0, max_tokens=args.max_tokens)
    # A single batched generate() enqueues all prompts before the engine steps,
    # so they share the batch (up to max_num_seqs run concurrently).
    outs = llm.generate(PROMPTS, params)
    print("\n=== generations ===", flush=True)
    for o in outs:
        print(repr(o.outputs[0].text[:60]), flush=True)

    verify(tele_dir)


def verify(tele_dir):
    sched = tele_dir / "scheduler.jsonl"
    runner = tele_dir / "runner.jsonl"
    snap = tele_dir / "runner_snapshot.json"

    def _jsonl(p):
        if not p.exists():
            return []
        return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

    s_recs, r_recs = _jsonl(sched), _jsonl(runner)
    steps = [r for r in r_recs if r.get("event") == "step"]
    admits = {r["request_id"] for r in r_recs if r.get("event") == "request_admitted"}
    completes = {
        r["request_id"] for r in r_recs if r.get("event") == "request_completed"
    }
    rates = [
        r["decode_rate_toks_per_s"] for r in steps if r.get("decode_rate_toks_per_s")
    ]

    print("\n=== telemetry files ===", flush=True)
    for p in (sched, runner, snap):
        size = p.stat().st_size if p.exists() else 0
        print(f"  {p.name}: {'OK' if p.exists() else 'MISSING'} ({size} bytes)")

    print("\n=== summary ===", flush=True)
    print(f"  scheduler records  : {len(s_recs)}")
    print(f"  runner step records: {len(steps)}")
    print(f"  requests admitted / completed: {len(admits)} / {len(completes)}")
    print(
        f"  max slots occupied : {max((r['slots_occupied'] for r in steps), default=0)}"
    )
    print(
        f"  stalled-by-prefill : {sum(1 for r in s_recs if r.get('decode_gated'))} steps"
    )
    print(f"  sample decode rates (tok/s): {rates[:5]}")

    assert s_recs, "no scheduler telemetry emitted"
    assert steps, "no runner step telemetry emitted"
    assert admits, "no request_admitted records"
    print(f"\nTelemetry written to {tele_dir}", flush=True)
    print(
        "Visualize: python scripts/telemetry/telemetry_viz.py report --dir "
        f"{tele_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
