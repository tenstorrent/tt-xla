"""DP correctness probe.

usage: dp_probe.py <num_reqs> <chunk> <prefix_cache 0|1> <distinct 0|1> <cpu_sampling 0|1> [reps]

reps controls prompt length: 3 reps ~= 40 tokens (single chunk at chunk=128),
18 reps ~= 240 tokens (two chunks -- exercises continuation prefill).

Coherence check mirrors tests/.../generative/conftest.py::assert_output_coherent
but adds a unique-token floor, which the repo version lacks -- without it,
degenerate output like "BlockBlockBlock..." matches as a single word and
short-circuits to "pass".
"""
import os
import re
import sys

import vllm

BASES = [
    "The history of computing spans many centuries and each generation improved speed. ",
    "Marine biology studies ocean ecosystems from coral reefs to deep sea vents today. ",
    "Bridge engineering balances tension and compression across spans of steel and stone. ",
    "Culinary tradition in mountain villages relies on preserved vegetables and cured meat. ",
    "Astronomers map galaxy clusters using redshift surveys and gravitational lensing data. ",
    "Textile manufacturing moved from handlooms to power looms during rapid industrial change. ",
    "Volcanic soil supports vineyards that produce distinctly mineral flavored white wines. ",
    "Railway signalling evolved from mechanical semaphores into digital interlocking systems. ",
]

STOP = {
    "the", "of", "and", "to", "a", "in", "is", "that", "for", "with", "as",
    "on", "by", "are", "was", "from", "it", "an", "be", "this", "which", "or",
    "at",
}
WORD_RE = re.compile(r"[A-Za-z']+")


def main():
    n, chunk = int(sys.argv[1]), int(sys.argv[2])
    prefix_cache, distinct = bool(int(sys.argv[3])), bool(int(sys.argv[4]))
    cpu_sampling = bool(int(sys.argv[5])) if len(sys.argv) > 5 else False
    reps = int(sys.argv[6]) if len(sys.argv) > 6 else 3
    prompts = (
        [BASES[i % len(BASES)] * reps for i in range(n)]
        if distinct
        else [BASES[0] * reps] * n
    )

    additional = {
        "min_context_len": 128,
        "enable_data_parallel": True,
        "prefill_chunk_size": chunk,
        "cpu_sampling": cpu_sampling,
    }
    if os.environ.get("IR_EXPORT_PATH"):
        additional["export_path"] = os.environ["IR_EXPORT_PATH"]
        additional["export_model_name"] = "dpprobe"

    llm = vllm.LLM(
        model="Qwen/Qwen3-0.6B",
        max_num_seqs=n,
        max_model_len=512,
        gpu_memory_utilization=0.1,
        enable_prefix_caching=prefix_cache,
        additional_config=additional,
    )
    out = llm.generate(
        prompts, vllm.SamplingParams(temperature=0.0, max_tokens=24)
    )
    bad = []
    for i, o in enumerate(out):
        t = o.outputs[0].text
        w = [x.lower() for x in WORD_RE.findall(t)]
        sr = (sum(1 for x in w if x in STOP) / len(w)) if w else 0.0
        ok = len(w) >= 5 and sr >= 0.10 and len(set(w)) > 2
        if not ok:
            bad.append(i)
        print(
            f"row {i}: {len(w):3d}w sr={sr:.2f} uniq={len(set(w)):3d} ok={ok}  {t[:52]!r}",
            flush=True,
        )
    print(
        f"\nRESULT n={n} chunk={chunk} cache={int(prefix_cache)} "
        f"distinct={int(distinct)} cpu_sampling={int(cpu_sampling)} reps={reps}: "
        f"{len(bad)}/{n} bad rows={bad}",
        flush=True,
    )
    llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
