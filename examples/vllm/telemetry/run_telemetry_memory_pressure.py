# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Memory-pressure serving-telemetry demo for the vLLM TT plugin.

Submits several requests with a LARGE input sequence length (ISL). Each long
prompt holds many KV-cache blocks, so a handful of them concurrently drive the
KV pool toward the fresh-prefill watermark. That surfaces the pressure signals a
short-prompt run never triggers: high ``kv_util``, ``watermark_rejects`` (the TT
scheduler declining to admit a fresh prefill so in-flight decodes keep their KV
instead of being preempted), and a batch that self-limits below ``max_num_seqs``
because KV -- not the slot count -- is the binding constraint.

    source venv/activate
    python examples/vllm/telemetry/run_telemetry_memory_pressure.py

The prompts are built to an exact token length AND made distinct per request, so
prefix caching cannot share their KV blocks (identical prompts would collapse to
one shared prefix and remove the pressure this demo is about).

    python scripts/telemetry/telemetry_viz.py report --dir tt_telemetry/memory_pressure
"""
import argparse
import json
from pathlib import Path

import vllm
from vllm import TokensPrompt

# Varied sentences; rotated + prefixed per request so no two prompts share a
# leading KV block (prefix caching keys on 32-token blocks).
SENTENCES = [
    "The turbine spun in the coastal wind under a bright and cloudless sky.",
    "A cartographer folded the old map along its worn and softened creases.",
    "Migrating cranes traced a long arc above the flooded rice paddies.",
    "The archivist catalogued each letter by date, sender, and faded postmark.",
    "Basalt columns stepped down to the sea in perfect hexagonal tiles.",
    "A luthier sanded the maple back until the grain caught the lamplight.",
    "The glacier calved with a report that rolled across the still fjord.",
    "Fireflies pulsed in slow waves along the edge of the summer meadow.",
    "The signalman lowered the lamp and the night train sighed to a halt.",
    "Salt crystals bloomed white across the cracked floor of the dry lake.",
    "A beekeeper lifted the frame, heavy and golden, from the humming hive.",
    "The observatory dome rotated, seams grinding, toward the rising planet.",
]


def build_prompts(tokenizer, num_prompts, isl):
    """Return `num_prompts` distinct TokensPrompts, each exactly `isl` tokens."""
    prompts = []
    for i in range(num_prompts):
        rotated = " ".join(
            SENTENCES[(i + j) % len(SENTENCES)] for j in range(len(SENTENCES))
        )
        seed = f"Passage {i}. {rotated} "
        ids = tokenizer(seed, add_special_tokens=False).input_ids or [0]
        reps = -(-isl // len(ids))  # ceil division
        prompts.append(TokensPrompt(prompt_token_ids=(ids * reps)[:isl]))
    return prompts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dir", default="tt_telemetry/memory_pressure", help="telemetry output dir"
    )
    ap.add_argument("--isl", type=int, default=2048, help="prompt length in tokens")
    ap.add_argument("--max-num-seqs", type=int, default=16, help="batch capacity")
    ap.add_argument("--num-prompts", type=int, default=16, help="requests to submit")
    ap.add_argument("--max-tokens", type=int, default=48, help="tokens per request")
    ap.add_argument(
        "--prefill-chunk",
        type=int,
        default=512,
        help="per-sequence prefill chunk (0 = single-shot; see note in main)",
    )
    ap.add_argument(
        "--gpu-mem-util",
        type=float,
        default=0.2,
        help="fraction of device memory (smaller -> smaller KV pool -> more pressure)",
    )
    args = ap.parse_args()

    tele_dir = Path(args.dir).resolve()
    max_model_len = args.isl + args.max_tokens + 64
    if args.prefill_chunk > 0:
        # Chunked prefill requires max_model_len to be a multiple of 256.
        max_model_len = -(-max_model_len // 256) * 256
    # Single-shot prefill at this scale needs a 34560-token budget, which does
    # not compile; the chunk bounds the graph and TTPlatform derives the budget.
    tt_config = {
        "enable_const_eval": False,
        "min_context_len": 256,
        "telemetry_enabled": True,
        "telemetry_dir": str(tele_dir),
        "telemetry_flush_ms": 200,
    }
    if args.prefill_chunk > 0:
        tt_config["prefill_chunk_size"] = args.prefill_chunk
        budget = args.prefill_chunk * args.max_num_seqs
    else:
        budget = max_model_len * args.max_num_seqs
    llm = vllm.LLM(
        model="Qwen/Qwen3-0.6B",
        max_model_len=max_model_len,
        max_num_batched_tokens=budget,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_mem_util,
        additional_config=tt_config,
    )

    prompts = build_prompts(llm.get_tokenizer(), args.num_prompts, args.isl)
    params = vllm.SamplingParams(temperature=0, max_tokens=args.max_tokens)
    print(
        f"\n=== {len(prompts)} requests x {args.isl}-token prompts "
        f"(batch capacity {args.max_num_seqs}) ===",
        flush=True,
    )
    outs = llm.generate(prompts, params)
    print(f"completed {len(outs)} requests", flush=True)

    verify(tele_dir, args.max_num_seqs)


def verify(tele_dir, max_num_seqs):
    def _jsonl(p):
        p = tele_dir / p
        if not p.exists():
            return []
        return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]

    s_recs = _jsonl("scheduler.jsonl")
    r_recs = _jsonl("runner.jsonl")
    steps = [r for r in r_recs if r.get("event") == "step"]
    admits = [r for r in r_recs if r.get("event") == "request_admitted"]
    completes = [r for r in r_recs if r.get("event") == "request_completed"]

    peak_kv = max((r.get("kv_util", 0) or 0 for r in s_recs), default=0)
    wm_rejects = max((r.get("cum_watermark_rejects", 0) for r in s_recs), default=0)
    preempted = max((r.get("cum_preempted", 0) for r in s_recs), default=0)
    max_wait = max((r.get("num_waiting", 0) for r in s_recs), default=0)
    max_occ = max((r.get("slots_occupied", 0) for r in steps), default=0)

    print("\n=== summary ===", flush=True)
    print(f"  requests admitted / completed: {len(admits)} / {len(completes)}")
    print(f"  batch capacity (max slots)   : {max_num_seqs}")
    print(f"  max slots occupied           : {max_occ}")
    print(f"  peak KV utilization          : {peak_kv * 100:.1f}%")
    print(f"  watermark rejects (fresh prefill declined): {wm_rejects}")
    print(f"  preemptions                  : {preempted}")
    print(f"  max queue depth (num_waiting) : {max_wait}")

    assert s_recs and steps, "no telemetry emitted"
    pressure = peak_kv >= 0.5 or wm_rejects > 0 or preempted > 0
    if pressure:
        print(
            "\nMemory pressure observed (high KV utilization / watermark rejects / "
            "preemption).",
            flush=True,
        )
    else:
        print(
            "\nNo memory pressure seen. Increase --isl or --num-prompts, or lower "
            "--gpu-mem-util, to shrink KV headroom.",
            flush=True,
        )
    print(f"Telemetry written to {tele_dir}", flush=True)
    print(
        f"Visualize: python scripts/telemetry/telemetry_viz.py report --dir {tele_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
