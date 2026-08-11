# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Offline batch-invariance repro for tt-xla's vLLM plugin -- no server, no eval.

Greedy decoding is a deterministic function of a prompt, so the text generated
for a prompt must not depend on which other prompts happened to share its batch.
This script issues the same prompts two ways against one engine instance:

  ALONE    one llm.generate() call per prompt, so the batch holds one request
  BATCHED  a single llm.generate() call with all prompts, so they share batches

and diffs the per-prompt text. ALONE is run twice first, which separates
"nondeterministic" from "batch-dependent": if the two ALONE passes disagree the
problem is not about batching at all.

Engine config mirrors the Falcon3-7B-Instruct eval server that showed an 11-point
ifeval drop at concurrency 16 versus sequential (tt-inference-server#4752).

The --set flag overrides any engine or additional_config key, which is how the
serving layer gets bisected -- run the baseline, then re-run disabling one
subsystem at a time:

  --set prefill_batch_threshold=0        force prefills to batch  <-- the trigger
  --set prefill_batch_threshold=4096     force prefills to run serially
  --set enable_prefix_caching=False      shared-prefix block reuse
  --set cpu_sampling=True                on-device sampling op
  --set enable_trace=False               metal trace capture/replay

That knob is what localised this defect to batched (multi-request) prefill: at a
fixed 8 requests, forcing prefills to batch corrupts every row but the first,
and at a fixed 32 requests, forcing them serial makes the batch exactly
invariant. Request count on its own is only a proxy for which path is taken.

Usage:
  # reproduces at 8 requests
  python batch_invariance_repro.py --bisect --counts 0 --set prefill_batch_threshold=0
  # clean at 8 requests (production threshold serialises small bursts)
  python batch_invariance_repro.py --bisect --counts 0
  # sweep request count; repeat a count to test run-to-run determinism
  python batch_invariance_repro.py --bisect --counts 0,8,16,24 --tag sweep
"""

import argparse
import json
import os

PROMPTS = [
    "Write a short paragraph about the ocean.",
    "List exactly 3 bullet points about trees. Use markdown bullets.",
    "Explain photosynthesis in two sentences.",
    "Write a JSON object describing a book. Output only JSON.",
    "Describe the water cycle in exactly four sentences.",
    "Name three programming languages and one strength of each.",
    "Write a haiku about winter mountains.",
    "Summarise why the sky appears blue, in plain language.",
]

BASE_ENGINE = {
    "model": "tiiuae/Falcon3-7B-Instruct",
    "max_num_seqs": 32,
    "max_model_len": 2048,
    "max_num_batched_tokens": 1024,
    "gpu_memory_utilization": 0.15,
    "enable_prefix_caching": True,
}

BASE_ADDITIONAL = {
    "enable_const_eval": True,
    "min_context_len": 128,
    "experimental_weight_dtype": "",
    "experimental_kv_cache_dtype": "none",
    "cpu_sampling": False,
    "optimization_level": 1,
    "enable_trace": True,
    "prefill_chunk_size": 1024,
    "fp32_dest_acc_en": True,
    "math_fidelity": "hifi4",
    "min_num_seqs": 1,
    "prefill_batch_threshold": 16,
}


def coerce(v):
    if v in ("True", "true"):
        return True
    if v in ("False", "false"):
        return False
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def divergence(a, b):
    if a == b:
        return None
    for k in range(min(len(a), len(b))):
        if a[k] != b[k]:
            return k
    return min(len(a), len(b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--n-prompts", type=int, default=len(PROMPTS))
    ap.add_argument("--tag", default="baseline")
    ap.add_argument(
        "--bisect",
        action="store_true",
        help="sweep total request count instead of running the scenario set",
    )
    ap.add_argument(
        "--counts",
        default="0,4,6,8,9,12,16,24",
        help="--bisect: comma-separated filler counts to sweep",
    )
    ap.add_argument(
        "--same-prompt",
        type=int,
        default=0,
        metavar="N",
        help="submit the same prompt N times in one batch and compare rows to "
        "each other; separates a positional bug from graph-shape rounding",
    )
    ap.add_argument(
        "--logprob-probe",
        action="store_true",
        help="compare per-step logprob values to separate a bad prefill KV write "
        "from a bad first decode write",
    )
    ap.add_argument(
        "--token-sweep",
        default="",
        metavar="T1,T2,...",
        help="sweep max_tokens to find the first divergent generated token",
    )
    ap.add_argument(
        "--equal-len",
        action="store_true",
        help="truncate all target prompts to a common token length, so a batched "
        "prefill has no ragged rows",
    )
    ap.add_argument("--outdir", default=".")
    ap.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override an engine arg or additional_config key",
    )
    args = ap.parse_args()

    engine = dict(BASE_ENGINE)
    additional = dict(BASE_ADDITIONAL)
    for kv in args.set:
        k, v = kv.split("=", 1)
        if k in additional:
            additional[k] = coerce(v)
        else:
            engine[k] = coerce(v)

    import vllm

    prompts = PROMPTS[: args.n_prompts]
    if args.equal_len:
        # Trim every prompt to the shortest one's token count, so each row of a
        # batched prefill has identical real length and raggedness is eliminated
        # as a variable.
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(BASE_ENGINE["model"])
        ids = [tok(p, add_special_tokens=False).input_ids for p in prompts]
        L = min(len(i) for i in ids)
        prompts = [tok.decode(i[:L]) for i in ids]
        print(f"equal-len: all {len(prompts)} prompts truncated to {L} tokens")
    sp = vllm.SamplingParams(temperature=0, max_tokens=args.max_tokens)

    print(f"tag={args.tag}")
    print(f"engine overrides : {args.set}")
    print(
        f"max_num_seqs={engine['max_num_seqs']}  n_prompts={len(prompts)}  "
        f"max_tokens={args.max_tokens}"
    )

    llm = vllm.LLM(**engine, additional_config=additional)

    def alone():
        return [llm.generate([p], sp)[0].outputs[0].text for p in prompts]

    def batched():
        outs = llm.generate(prompts, sp)
        # generate() may return out of order; key by prompt.
        by_prompt = {o.prompt: o.outputs[0].text for o in outs}
        return [by_prompt[p] for p in prompts]

    def batched_with(extra, extra_params):
        """Run targets alongside `extra` requests, return only the targets' text."""
        allp = prompts + extra
        params = [sp] * len(prompts) + extra_params
        outs = llm.generate(allp, params)
        by_prompt = {}
        for o in outs:
            by_prompt.setdefault(o.prompt, o.outputs[0].text)
        return [by_prompt[p] for p in prompts]

    print("\n--- pass 1: ALONE (batch of one) ---")
    a1 = alone()
    print("--- pass 2: ALONE again (determinism control) ---")
    a2 = alone()

    # Scenarios, in increasing structural distance from "all requests in lockstep".
    # Submitting every prompt at once makes them prefill together and decode in
    # step, which is the one thing a real server never does. The later scenarios
    # reintroduce what it does do: enough requests to cross
    # prefill_batch_threshold, prefills long enough to be chunked across decode
    # steps of other requests, and requests retiring at different times so batch
    # composition changes mid-generation.
    filler_short = [f"Count from 1 to {n} and then stop." for n in range(5, 45)]
    # ~1500 tokens: longer than prefill_chunk_size (1024) so the prefill is split
    # across steps, but short enough that prompt + max_tokens still fits
    # max_model_len.
    long_body = "The quick brown fox jumps over the lazy dog. " * 150
    filler_long = [f"Notes {i}:\n{long_body}\nReply with one word." for i in range(4)]

    if args.same_prompt:
        # The same prompt N times in one batched prefill. Every row is then
        # mathematically identical, so comparing rows *within one batch* needs no
        # cross-run reference and isolates two very different causes:
        #   rows disagree with each other  -> position-dependent bug (row index
        #                                     enters the arithmetic)
        #   rows agree but differ from the alone run -> the [1,P] and [N,P] prefill
        #                                     graphs merely round differently
        # Prefix caching is forced off so identical prompts cannot share blocks.
        n = args.same_prompt
        p = prompts[0]
        spL = vllm.SamplingParams(temperature=0, max_tokens=1, logprobs=5)
        print(f"--- same prompt x{n} in one batched prefill ---", flush=True)
        print(f"    prompt: {p!r}", flush=True)

        def top(out):
            d = (out.outputs[0].logprobs or [{}])[0]
            if not d:
                return (None, None)
            tid = max(d, key=lambda k: d[k].logprob)
            return (tid, d[tid].logprob)

        alone_top = top(llm.generate([p], spL)[0])
        outs = llm.generate([p] * n, [spL] * n)
        rows = [top(o) for o in outs]
        print(f"    alone            : token={alone_top[0]} logprob={alone_top[1]!r}")
        for i, r in enumerate(rows):
            flag = "" if r == rows[0] else "  <-- differs from row 0"
            same_as_alone = " (== alone)" if r == alone_top else ""
            print(
                f"    row {i:<2} in batch    : token={r[0]} logprob={r[1]!r}"
                f"{same_as_alone}{flag}"
            )
        uniq = len({r[1] for r in rows})
        print()
        print(f"    distinct logprobs across {n} identical rows: {uniq}")
        if uniq > 1:
            print("    => POSITION-DEPENDENT: row index affects the arithmetic")
        elif rows and rows[0] != alone_top:
            print(
                "    => uniform shift only: batched graph rounds differently, "
                "no positional term"
            )
        else:
            print("    => batched prefill matches the alone run exactly here")
        with open(os.path.join(args.outdir, f"batchinv_{args.tag}.json"), "w") as f:
            json.dump(
                {"tag": args.tag, "alone": alone_top, "rows": rows},
                f,
                indent=2,
                default=str,
            )
        return

    if args.logprob_probe:
        # Where does the numeric error first appear, as opposed to where the
        # sampled token first flips? Compare per-step logprob VALUES, not argmax.
        #   step 1 (from the prefill forward, no KV read-back)
        #   step 2 (first decode step: reads prefill-written KV)
        #   step 3+ (reads KV written by earlier decode steps)
        # A value difference at step 2 means prefill wrote bad KV; values identical
        # through step 2 but differing at step 3 implicates the first decode write.
        T = args.max_tokens
        spL = vllm.SamplingParams(temperature=0, max_tokens=T, logprobs=5)
        print(f"--- logprob probe (max_tokens={T}) ---", flush=True)

        def steps(out):
            """Per-step (top token, its logprob) for one completion."""
            lps = out.outputs[0].logprobs or []
            res = []
            for d in lps:
                if not d:
                    res.append((None, None))
                    continue
                tid = max(d, key=lambda k: d[k].logprob)
                res.append((tid, d[tid].logprob))
            return res

        alone_s = [steps(llm.generate([p], spL)[0]) for p in prompts]
        outs = llm.generate(prompts, [spL] * len(prompts))
        by_prompt = {}
        for o in outs:
            by_prompt.setdefault(o.prompt, o)
        batch_s = [steps(by_prompt[p]) for p in prompts]

        print(
            f"  {'row':>3}  {'first step with different logprob VALUE':<40} "
            f"{'first step with different TOKEN':<32}"
        )
        summary = []
        for i in range(len(prompts)):
            a, b = alone_s[i], batch_s[i]
            n = min(len(a), len(b))
            val_step = next((s + 1 for s in range(n) if a[s][1] != b[s][1]), None)
            tok_step = next((s + 1 for s in range(n) if a[s][0] != b[s][0]), None)
            summary.append(
                {
                    "row": i,
                    "value_step": val_step,
                    "token_step": tok_step,
                    "alone": a,
                    "batched": b,
                }
            )
            print(f"  {i:>3}  {str(val_step):<40} {str(tok_step):<32}")
        with open(os.path.join(args.outdir, f"batchinv_{args.tag}.json"), "w") as f:
            json.dump(
                {"tag": args.tag, "logprob_probe": summary}, f, indent=2, default=str
            )
        return

    if args.token_sweep:
        # How many tokens in does the corruption appear? The token returned at
        # max_tokens=1 comes from the final prefill position, computed inside the
        # prefill forward before any KV read-back. So:
        #   T=1 clean, T=2 broken -> the prefill compute is fine and the defect is
        #                            in the KV written by prefill (read at decode)
        #   T=1 already broken    -> the prefill forward itself is wrong
        print("--- token sweep: first divergent generated token ---", flush=True)
        results = {}
        for T in [int(x) for x in args.token_sweep.split(",")]:
            spT = vllm.SamplingParams(temperature=0, max_tokens=T)
            aloneT = [llm.generate([p], spT)[0].outputs[0].text for p in prompts]
            outs = llm.generate(prompts, [spT] * len(prompts))
            by_prompt = {}
            for o in outs:
                by_prompt.setdefault(o.prompt, o.outputs[0].text)
            batchedT = [by_prompt[p] for p in prompts]
            bad = [i for i in range(len(prompts)) if aloneT[i] != batchedT[i]]
            results[f"T{T}"] = {"alone": aloneT, "batched": batchedT, "bad": bad}
            print(
                f"  max_tokens={T:<4} {len(prompts) - len(bad)}/{len(prompts)} identical"
                f"  {'invariant' if not bad else f'divergent rows {bad}'}",
                flush=True,
            )
        with open(os.path.join(args.outdir, f"batchinv_{args.tag}.json"), "w") as f:
            json.dump({"tag": args.tag, "token_sweep": results}, f, indent=2)
        return

    if args.bisect:
        # Total request count is the variable; everything else is held fixed.
        # 8 targets + N short fillers, stepping across prefill_batch_threshold.
        counts = [int(x) for x in args.counts.split(",")]
        scenarios = {
            f"n{len(prompts) + n}#{j}": (filler_short[:n], [sp] * n)
            for j, n in enumerate(counts)
        }
        results = {}
        print(
            f"--- bisect on total request count (threshold suspect: "
            f"{additional['prefill_batch_threshold']}) ---",
            flush=True,
        )
        for name, (extra, eparams) in scenarios.items():
            try:
                got = batched_with(extra, eparams)
            except Exception as e:
                print(f"  {name}: FAILED {type(e).__name__}: {e}", flush=True)
                continue
            results[name] = got
            bd = [
                i for i in range(len(prompts)) if divergence(a1[i], got[i]) is not None
            ]
            print(
                f"  {name:<6} total={len(prompts) + len(extra):<4} "
                f"{len(prompts) - len(bd)}/{len(prompts)} identical  "
                f"{'invariant' if not bd else 'BATCH-DEPENDENT'}",
                flush=True,
            )
        # Repeated arms with the same request count: do they agree with each
        # other? Deterministic-but-wrong points at a systematic indexing bug;
        # disagreement points at a race.
        by_count = {}
        for name, got in results.items():
            by_count.setdefault(name.split("#")[0], []).append(got)
        for cnt, runs in by_count.items():
            if len(runs) > 1:
                same = all(r == runs[0] for r in runs[1:])
                print(
                    f"  {cnt} repeatability across {len(runs)} runs: "
                    f"{'REPEATABLE (deterministic)' if same else 'NONDETERMINISTIC'}",
                    flush=True,
                )
        with open(os.path.join(args.outdir, f"batchinv_{args.tag}.json"), "w") as f:
            json.dump({"tag": args.tag, "alone": a1, "scenarios": results}, f, indent=2)
        return

    scenarios = {
        "batched": ([], []),
        "wide24": (filler_short[:24], [sp] * 24),
        "longprefill": (filler_long, [sp] * len(filler_long)),
        "stagger": (
            filler_short[:24],
            [
                vllm.SamplingParams(temperature=0, max_tokens=4 + 5 * i)
                for i in range(24)
            ],
        ),
        # More requests than max_num_seqs, so the scheduler admits them in waves
        # and a fresh prefill joins a batch whose other rows are mid-decode. The
        # scenarios above all fit in one admission and never produce that.
        "waves": (
            (filler_short * 3)[: 3 * engine["max_num_seqs"]],
            [
                vllm.SamplingParams(temperature=0, max_tokens=8 + 3 * (i % 17))
                for i in range(3 * engine["max_num_seqs"])
            ],
        ),
    }

    # Report and persist per scenario, so a failure in a later arm cannot discard
    # the arms that already ran.
    results = {}
    for name, (extra, eparams) in scenarios.items():
        print(
            f"--- scenario {name}: targets + {len(extra)} co-resident ---", flush=True
        )
        try:
            got = batched_with(extra, eparams)
        except Exception as e:
            print(f"    scenario {name} FAILED: {type(e).__name__}: {e}", flush=True)
            continue
        results[name] = got
        bd = [
            (i, divergence(a1[i], got[i]))
            for i in range(len(prompts))
            if divergence(a1[i], got[i]) is not None
        ]
        print(
            f"    {name}: {len(prompts) - len(bd)}/{len(prompts)} identical to ALONE"
            f"  {'invariant' if not bd else 'BATCH-DEPENDENT'}",
            flush=True,
        )

    nondet = sum(1 for x, y in zip(a1, a2) if x != y)
    print()
    print("=" * 78)
    print(
        f"ALONE vs ALONE   : {len(prompts) - nondet}/{len(prompts)} identical", end=""
    )
    print(
        "   (sequential is deterministic)"
        if nondet == 0
        else f"   *** {nondet} NONDETERMINISTIC -- not a batching question ***"
    )

    diffs = {}
    for name, got in results.items():
        d = [(i, divergence(a1[i], got[i])) for i in range(len(prompts))]
        diffs[name] = [(i, k) for i, k in d if k is not None]

    print()
    print(f"  {'scenario':<14} {'identical':>12}  verdict")
    for name, bd in diffs.items():
        n_same = len(prompts) - len(bd)
        print(
            f"  {name:<14} {f'{n_same}/{len(prompts)}':>12}  "
            f"{'invariant' if not bd else f'BATCH-DEPENDENT ({len(bd)} prompts)'}"
        )

    for name, bd in diffs.items():
        if not bd:
            continue
        print(f"\n  --- {name} divergences ---")
        for i, d in bd:
            print(f"  prompt {i}: diverges at char {d}  | {PROMPTS[i][:48]}")
            print(f"    alone  : ...{a1[i][max(0, d - 30):d + 60]!r}")
            print(f"    {name:<7}: ...{results[name][i][max(0, d - 30):d + 60]!r}")
    if not any(diffs.values()):
        print("\n  BATCH-INVARIANT across all scenarios on this configuration")
    print("=" * 78)

    out = os.path.join(args.outdir, f"batchinv_{args.tag}.json")
    with open(out, "w") as f:
        json.dump(
            {
                "tag": args.tag,
                "overrides": args.set,
                "engine": {k: str(v) for k, v in engine.items()},
                "additional_config": {k: str(v) for k, v in additional.items()},
                "max_tokens": args.max_tokens,
                "nondeterministic": nondet,
                "batch_dependent": {
                    k: [{"prompt": i, "char": d} for i, d in v]
                    for k, v in diffs.items()
                },
                "n_prompts": len(prompts),
                "alone": a1,
                "alone2": a2,
                "scenarios": results,
            },
            f,
            indent=2,
        )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
