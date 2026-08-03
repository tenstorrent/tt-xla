# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Oversubscribed serving-telemetry demo for the vLLM TT plugin.

Like run_telemetry_demo.py, but submits MORE requests than the batch has slots
(8 prompts, max_num_seqs=4). A single batched generate() enqueues all of them,
so the scheduler can only admit `max_num_seqs` at a time and the rest wait in
the queue, getting admitted as running requests finish. This exercises the
signals that a full-batch run does not: non-zero queue depth (`num_waiting`) and
re-admission (requests admitted well after step 0).

    source venv/activate
    python examples/vllm/telemetry/run_telemetry_oversubscribed.py

Then visualize -- the KV/batch-utilization chart shows the batch pinned at
capacity while the queue drains:

    python scripts/telemetry/telemetry_viz.py report --dir tt_telemetry/oversubscribed
"""
import argparse
import json
from pathlib import Path

import vllm

# Eight distinct prompts (> the default 4 slots) so the queue stays non-empty.
PROMPTS = [
    "Write a haiku about the ocean.",
    "List three prime numbers.",
    "What is the capital of France?",
    "Explain gravity in one sentence.",
    "Name a primary color.",
    "What is 12 times 8?",
    "Give one fact about the moon.",
    "Translate 'hello' into Spanish.",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dir", default="tt_telemetry/oversubscribed", help="telemetry output dir"
    )
    ap.add_argument(
        "--max-num-seqs", type=int, default=4, help="batch capacity (slots)"
    )
    ap.add_argument(
        "--num-prompts",
        type=int,
        default=len(PROMPTS),
        help="how many prompts to submit (cycled from the pool)",
    )
    ap.add_argument("--max-tokens", type=int, default=48, help="tokens per request")
    args = ap.parse_args()

    if args.num_prompts <= args.max_num_seqs:
        print(
            f"warning: num_prompts ({args.num_prompts}) <= max_num_seqs "
            f"({args.max_num_seqs}); nothing will queue. Raise --num-prompts "
            "or lower --max-num-seqs to oversubscribe."
        )
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(args.num_prompts)]

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
            "telemetry_enabled": True,
            "telemetry_dir": str(tele_dir),
            "telemetry_flush_ms": 200,
        },
    )

    params = vllm.SamplingParams(temperature=0, max_tokens=args.max_tokens)
    outs = llm.generate(prompts, params)
    print(
        f"\n=== {len(outs)} generations (batch capacity {args.max_num_seqs}) ===",
        flush=True,
    )
    for o in outs:
        print(repr(o.outputs[0].text[:50]), flush=True)

    verify(tele_dir, args.max_num_seqs)


def verify(tele_dir, max_num_seqs):
    sched = tele_dir / "scheduler.jsonl"
    runner = tele_dir / "runner.jsonl"

    def _jsonl(p):
        if not p.exists():
            return []
        return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

    s_recs, r_recs = _jsonl(sched), _jsonl(runner)
    steps = [r for r in r_recs if r.get("event") == "step"]
    admits = [r for r in r_recs if r.get("event") == "request_admitted"]
    completes = [r for r in r_recs if r.get("event") == "request_completed"]
    max_wait = max((r.get("num_waiting", 0) for r in s_recs), default=0)
    max_occ = max((r.get("slots_occupied", 0) for r in steps), default=0)
    # Re-admissions: requests admitted after the batch first filled (step > 0),
    # i.e. only after an earlier request freed a slot.
    readmitted = [a for a in admits if a.get("step", 0) > 0]
    preempted = max((r.get("cum_preempted", 0) for r in s_recs), default=0)

    print("\n=== summary ===", flush=True)
    print(f"  requests admitted / completed: {len(admits)} / {len(completes)}")
    print(f"  batch capacity (max slots)   : {max_num_seqs}")
    print(f"  max slots occupied           : {max_occ}")
    print(f"  max queue depth (num_waiting) : {max_wait}")
    print(f"  requests that waited for a slot: {len(readmitted)}")
    print(f"  preemptions                  : {preempted}")

    assert s_recs and steps, "no telemetry emitted"
    if len(admits) > max_num_seqs:
        assert max_wait > 0, (
            "expected a non-empty queue when oversubscribed, but num_waiting "
            "was always 0"
        )
        print(
            "\nOversubscription confirmed: more requests than slots, queue drained "
            "as requests completed.",
            flush=True,
        )
    print(f"Telemetry written to {tele_dir}", flush=True)
    print(
        f"Visualize: python scripts/telemetry/telemetry_viz.py report --dir {tele_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
