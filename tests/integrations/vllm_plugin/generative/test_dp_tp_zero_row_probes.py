# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Zero-scheduled-row / early-departure probes for the gemma-4-31B DP+TP bug.

Established so far (see test_dp_tp_debug_probes.py): a heterogeneous batch
corrupts the last batch slots; it is deterministic, position-bound (not
content-bound), unchanged by cpu_sampling / enable_const_eval=False, and
DISAPPEARS when ignore_eos=True prevents any sequence from finishing early. So
the trigger is early departure + batch re-packing.

These probes narrow *why* early departure matters, and test the
"zero-scheduled row lands mid-slice (not tail)" / "does it need TP?" hypotheses.
Every probe prints the RAW output per prompt with a tail-aware garbage flag, so
nothing is masked by a lenient coherence heuristic.

Probe                     | tests
--------------------------|------------------------------------------------------
repro                     | baseline reproduction (gemma [8,4] diverse)
all_long                  | T1: no early EOS w/o ignore_eos (all prompts >32 tok)
sync_short                | N4: synchronized early EOS (all finish ~same step)
protect_departers         | T2b: ignore_eos on the SHORT finishers only
protect_survivors         | T2a: ignore_eos on the LONG survivors only
inject_short_first        | T3: one early finisher in slot 0
inject_short_last         | T3: one early finisher in the last slot
chunked_prefill           | N3: force chunked prefill (low max_num_batched_tokens)
small_repro               | N0: does it reproduce on Qwen3-0.6B [8,4]?
small_dp_only             | N2: Qwen3-0.6B DP-only [32,1] (no TP)
small_dp_tp               | N2: Qwen3-0.6B DP+TP [8,4] (same workload)

See run_zero_row_probes.sh for the run order. All are report-only (expect
inspection of the printed output); the controls that must be clean assert.
"""
import re

import pytest
import vllm
from conftest import check_host_memory

GEMMA = "google/gemma-4-31B-it"
QWEN = "Qwen/Qwen3-0.6B"

# Mixed-length set; #2 (France) finishes very early, #6/#7 run longest.
DIVERSE = [
    "Describe Tenstorrent in one sentence.",
    "Explain what a neural network is in one sentence.",
    "What is the capital of France?",
    "Write one sentence about the ocean.",
    "Summarize the theory of relativity in one sentence.",
    "Give me a one-sentence description of photosynthesis.",
    "What is machine learning, in one sentence?",
    "Describe the sun in one sentence.",
]
# Indices that hit natural EOS early (the "departers") vs long survivors.
EARLY_IDX = {1, 2, 3, 5}
SURVIVOR_IDX = {0, 4, 6, 7}

# Open-ended prompts whose greedy completion exceeds 32 tokens -> no early EOS.
ALL_LONG = [
    "Write a long, detailed paragraph explaining what Tenstorrent builds and why it matters.",
    "Explain in several detailed sentences how a neural network learns from data.",
    "Describe the ocean in a long, detailed multi-sentence paragraph.",
    "Give a thorough multi-sentence explanation of the theory of relativity.",
    "Explain photosynthesis in detail across several sentences.",
    "Write a long paragraph about what machine learning is and how it works.",
    "Describe the sun in a detailed, multi-sentence paragraph.",
    "Write several detailed sentences about the history of computing.",
]
# Equal-shape short factual Qs: all reach EOS early at ~the same step (synchronized).
SYNC_SHORT = [
    "What is the capital of France?",
    "What is the capital of Japan?",
    "What is the capital of Italy?",
    "What is the capital of Spain?",
    "What is the capital of Egypt?",
    "What is the capital of Peru?",
    "What is the capital of Cuba?",
    "What is the capital of Chile?",
]
SHORT_ONE = "What is 1 + 1?"  # extremely early finisher for injection tests

_GREEDY = dict(temperature=0.0, top_p=1.0, max_tokens=32)


# ---------------------------------------------------------------------------
# Tail-aware garbage detection (do NOT rely on whole-string stopword ratio).
# ---------------------------------------------------------------------------
_CHAR_RUN = re.compile(
    r"(.)\1{3,}"
)  # 4+ of same char: ////, LBBBB, aaaa (".." ellipsis is only 3, safe)
_NONASCII_RUN = re.compile(r"[^\x00-\x7f]{4,}")  # runs of non-ASCII (額額額, ेंें)


def garbage_reason(text):
    """Return a short reason string if `text` looks like token-soup, else None.

    Tuned to the observed failure signatures: repeated punctuation/letters
    (`//LBBBBBB`, `sphereb////LLLL`), repeated single tokens (`a a a a`,
    `la la la`), and non-ASCII runs. Repetition anywhere is flagged; note that
    the benign ignore_eos artifact (`.of France is Paris.of ...`) will also
    flag as word-repeat — that's expected on ignore_eos probes, read the text.
    """
    s = text.strip()
    if not s:
        return "empty"
    if _CHAR_RUN.search(s):
        return "char-run"
    if _NONASCII_RUN.search(s):
        return "nonascii-run"
    words = s.split()
    if len(words) >= 6 and len(set(w.lower() for w in words[-6:])) <= 2:
        return "word-repeat-tail"
    return None


def _report(label, results):
    print(f"\n===== {label} =====")
    bad = []
    for i, (prompt, text) in enumerate(results):
        reason = garbage_reason(text)
        tag = "BAD" if reason else "OK "
        if reason:
            bad.append(i)
        print(
            f"[{tag}] #{i} {('('+reason+')') if reason else ''} | {prompt!r}\n      -> {text!r}"
        )
    print(
        f"----- {label}: {len(results)-len(bad)}/{len(results)} clean; bad slots={bad} -----"
    )
    return bad


def _to_messages(prompts):
    return [[{"role": "user", "content": p}] for p in prompts]


def _chat(llm, prompts, sampling_params, chat_template_kwargs=None):
    kw = {"chat_template_kwargs": chat_template_kwargs} if chat_template_kwargs else {}
    outs = llm.chat(_to_messages(prompts), sampling_params, **kw)
    assert len(outs) == len(prompts)
    return [(p, o.outputs[0].text) for p, o in zip(prompts, outs)]


def _base_llm_args(
    model, mesh_shape, max_num_seqs, max_num_batched_tokens=8192, gpu_mem=0.3, **ac
):
    additional_config = {
        "enable_const_eval": True,
        "min_context_len": 32,
        "enable_data_parallel": True,
        "enable_tensor_parallel": True,
        "shard_weights_on_batch_axis": True,
        "experimental_weight_dtype": "",
        "mesh_shape": mesh_shape,
        "cpu_sampling": False,
        "flat_model_io": True,
    }
    additional_config.update(ac)
    return {
        "model": model,
        "limit_mm_per_prompt": {"image": 0, "video": 0, "audio": 0},
        "max_num_batched_tokens": max_num_batched_tokens,
        "max_num_seqs": max_num_seqs,
        "max_model_len": 128,
        "gpu_memory_utilization": gpu_mem,
        "additional_config": additional_config,
    }


# ---------------------------------------------------------------------------
# Probe registry. Each entry builds llm_args + prompts + sampling, then reports.
# ---------------------------------------------------------------------------
def _greedy(prompts):
    return vllm.SamplingParams(**_GREEDY)


def _per_req_ignore_eos(protect):
    """SamplingParams list: ignore_eos=True for indices in `protect`."""

    def build(prompts):
        return [
            vllm.SamplingParams(**{**_GREEDY, "ignore_eos": i in protect})
            for i in range(len(prompts))
        ]

    return build


PROBES = {
    # --- baseline ---
    "repro": dict(
        model=GEMMA,
        mesh=[8, 4],
        nseq=8,
        prompts=DIVERSE,
        sampling=_greedy,
        expect="reproduce",
        note="baseline reproduction",
    ),
    # --- T1: no early EOS without ignore_eos (all prompts naturally > 32 tok) ---
    "all_long": dict(
        model=GEMMA,
        mesh=[8, 4],
        nseq=8,
        prompts=ALL_LONG,
        sampling=_greedy,
        expect="clean?",
        note="no early departures via long prompts (no ignore_eos)",
    ),
    # --- N4: synchronized early EOS ---
    "sync_short": dict(
        model=GEMMA,
        mesh=[8, 4],
        nseq=8,
        prompts=SYNC_SHORT,
        sampling=_greedy,
        expect="clean?",
        note="all finish early at ~same step (synchronized departure)",
    ),
    # --- T2b: protect the departers (ignore_eos on the short finishers) ---
    "protect_departers": dict(
        model=GEMMA,
        mesh=[8, 4],
        nseq=8,
        prompts=DIVERSE,
        sampling=_per_req_ignore_eos(EARLY_IDX),
        expect="clean?",
        note="ignore_eos on early finishers only -> no departures",
    ),
    # --- T2a: protect the survivors (ignore_eos on long ones; short ones still depart) ---
    "protect_survivors": dict(
        model=GEMMA,
        mesh=[8, 4],
        nseq=8,
        prompts=DIVERSE,
        sampling=_per_req_ignore_eos(SURVIVOR_IDX),
        expect="reproduce",
        note="ignore_eos on survivors only; departers still leave",
    ),
    # --- T3: single early finisher, slot 0 vs last ---
    "inject_short_first": dict(
        model=GEMMA,
        mesh=[8, 4],
        nseq=8,
        prompts=[SHORT_ONE] + ALL_LONG[:7],
        sampling=_greedy,
        expect="reproduce",
        note="one early finisher in slot 0",
    ),
    "inject_short_last": dict(
        model=GEMMA,
        mesh=[8, 4],
        nseq=8,
        prompts=ALL_LONG[:7] + [SHORT_ONE],
        sampling=_greedy,
        expect="clean?",
        note="one early finisher in the last slot",
    ),
    # --- N3: force chunked prefill ---
    "chunked_prefill": dict(
        model=GEMMA,
        mesh=[8, 4],
        nseq=8,
        prompts=DIVERSE,
        sampling=_greedy,
        max_num_batched_tokens=256,
        expect="reproduce",
        note="low max_num_batched_tokens -> chunked prefill",
    ),
    # --- N0: small-model reproduction ---
    "small_repro": dict(
        model=QWEN,
        mesh=[8, 4],
        nseq=8,
        prompts=DIVERSE,
        sampling=_greedy,
        gpu_mem=0.5,
        expect="reproduce?",
        note="does it reproduce on Qwen3-0.6B?",
    ),
    # --- N2: DP-degree sweep on the small model (same 32-prompt workload) ---
    "small_dp_tp": dict(
        model=QWEN,
        mesh=[8, 4],
        nseq=32,
        prompts=DIVERSE * 4,
        sampling=_greedy,
        gpu_mem=0.5,
        expect="reproduce?",
        note="DP=8/TP=4, same workload",
    ),
    # --- DP-degree sweep (low TP to avoid the TP=8/16 compile crashes). The
    # per-replica-KV + condense hypothesis predicts corruption at all dp>1, and
    # (weakly) severity scaling with dp. dp_size=1 is NOT reachable on 32 chips
    # (it needs TP=32), so the clean reference requires a sub-mesh / fewer
    # visible devices -- see note in the response. Confirmation here comes from
    # the N1b condense-move instrumentation, not a dp=1 baseline. ---
    "small_dp16": dict(
        model=QWEN,
        mesh=[16, 2],
        nseq=32,
        prompts=DIVERSE * 4,
        sampling=_greedy,
        gpu_mem=0.5,
        expect="reproduce?",
        note="DP=16/TP=2",
    ),
    "small_dp32": dict(
        model=QWEN,
        mesh=[32, 1],
        nseq=32,
        prompts=DIVERSE * 4,
        sampling=_greedy,
        gpu_mem=0.5,
        expect="reproduce?",
        note="DP=32/TP=1 (max replicas)",
    ),
}


@pytest.mark.nightly
@pytest.mark.tensor_parallel
@pytest.mark.bh_galaxy
@pytest.mark.parametrize("probe", list(PROBES), ids=lambda n: f"probe_{n}")
def test_zero_row_probe(probe):
    spec = PROBES[probe]
    print(f"\n########## PROBE: {probe} ({spec['expect']}) ##########")
    print(f"# {spec['note']}")
    llm_args = _base_llm_args(
        spec["model"],
        spec["mesh"],
        spec["nseq"],
        max_num_batched_tokens=spec.get("max_num_batched_tokens", 8192),
        gpu_mem=spec.get("gpu_mem", 0.3),
        **spec.get("ac", {}),
    )
    print("llm_args:")
    for k, v in llm_args.items():
        print(f"  {k}: {v}")

    llm = vllm.LLM(**llm_args)
    prompts = spec["prompts"]
    sampling = spec["sampling"](prompts)
    # Qwen3 is a reasoning model: disable thinking so factual prompts give short
    # answers and hit EOS early -> triggers condense (the whole point of the DP
    # probes). Gemma has no thinking mode, so pass nothing.
    ctk = {"enable_thinking": False} if spec["model"] == QWEN else None
    bad = _report(probe, _chat(llm, prompts, sampling, chat_template_kwargs=ctk))

    # Host-mem threshold is informational here; don't let it fail the probe.
    try:
        check_host_memory(spec["model"])
    except AssertionError as e:
        print(f"[MEM] non-fatal in probe context: {e}")

    # Only hard-assert the configs we EXPECT to be clean as controls.
    if spec["expect"] == "clean":
        assert not bad, f"probe {probe!r} expected all-clean, bad slots={bad}"
    # "reproduce" / "clean?" / "reproduce?": report-only — read the printed text.
