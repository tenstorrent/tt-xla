# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""DP+TP generation debug probes for the gemma-4-31B accuracy bug.

Symptom under investigation: with mesh [8,4] (DP=8, TP=4) and
shard_weights_on_batch_axis=True, a batch of *diverse* prompts produces
progressively corrupted output. A homogeneous batch (same prompt repeated) is
clean; corruption grows with decode length and concentrates in long,
non-EOS-terminating sequences. Batch=8 with no repetition already reproduces it,
so continuous-batching churn is ruled out.

Each ``probe`` isolates one hypothesis. Run them in the order in HOW TO RUN
(bottom of this docstring). Every probe prints a full per-prompt report BEFORE
any assertion so you always see the whole pattern, even on failure.

Hypotheses (see probe table for the mapping):
  H1  decode-length accumulation (short/EOS seqs escape); H1a = KV block boundary
  H2  FSDP weight sharding (shard_weights_on_batch_axis) reshard mis-lowering
  H3  DP data/KV path + global->per-device slot mapping
  H4  cross-sequence contamination within a batch (content-visible)
  H5  fused runtime / on-device sampling path
  H6  compilation bucket / padding (num_reqs / token ladder)
  H7  nondeterminism / multi-device race
  H8  const-eval mis-caching under multi-device

Probe -> what it tests -> what a result means:
  main               reproduce the failing reference [8,4]           (expect: reproduce)
  baseline_homogeneous   H4/control: 8x the SAME long prompt          -> must be clean
  baseline_determinism   H7: same batch twice, tokens must match      -> flags a race
  sweep_max_tokens       H1: 8/24/48 tokens                          -> clean-short => H1
  force_full_len         H1 positive test: ignore_eos + min_tokens=32 -> France garbles too => pure length
  reorder                content- vs position-bound                   -> same prompts vs same slots
  cpu_sampling           H5: fused->unfused runtime + host argmax      -> fixes => H5
  const_eval_off         H8                                           -> fixes => H8
  shard_weights_off      H2 (config-only; SEE CAVEAT)                 -> may be incoherent by construction
  mesh_dp2               DP=2, TP=16 (batch axis smaller)             -> direction disambiguates DP vs TP
  mesh_dp4               DP=4, TP=8

HOW TO RUN (galaxy, 32 chips). Wave order = cheapest / most-decisive first:

  # Wave 0 - controls (must pass, or nothing downstream is trustworthy)
  pytest -svv test_dp_tp_debug_probes.py -k "baseline_homogeneous or baseline_determinism"

  # Wave 1 - characterize the corruption (single 31B load each; no config/mesh change)
  pytest -svv test_dp_tp_debug_probes.py -k "main or sweep_max_tokens or force_full_len or reorder"

  # Wave 2 - localize the code path (one reload each; TP stays 4)
  pytest -svv test_dp_tp_debug_probes.py -k "cpu_sampling or const_eval_off or shard_weights_off"

  # Wave 3 - confirm the axis (expensive; TP degree changes too)
  pytest -svv test_dp_tp_debug_probes.py -k "mesh_dp2 or mesh_dp4"

  # single probe:
  pytest -svv test_dp_tp_debug_probes.py -k "probe_sweep_max_tokens"
"""
import pytest
import vllm
from conftest import assert_output_coherent, check_host_memory

MODEL_NAME = "google/gemma-4-31B-it"

# The 8-prompt diverse set that already reproduces the bug at batch=8.
PROMPTS = [
    "Describe Tenstorrent in one sentence.",
    "Explain what a neural network is in one sentence.",
    "What is the capital of France?",
    "Write one sentence about the ocean.",
    "Summarize the theory of relativity in one sentence.",
    "Give me a one-sentence description of photosynthesis.",
    "What is machine learning, in one sentence?",
    "Describe the sun in one sentence.",
]


def _to_messages(prompts):
    return [[{"role": "user", "content": p}] for p in prompts]


def _base_llm_args(mesh_shape, max_num_seqs, **additional_config_overrides):
    """The exact failing config, with additional_config overridable per probe."""
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
    additional_config.update(additional_config_overrides)
    return {
        "model": MODEL_NAME,
        # Text-only path on a multimodal model: zero every modality so the
        # mm-encoder graph doesn't compile the vision tower at all.
        "limit_mm_per_prompt": {"image": 0, "video": 0, "audio": 0},
        "max_num_batched_tokens": 8192,
        "max_num_seqs": max_num_seqs,
        "max_model_len": 128,
        "gpu_memory_utilization": 0.3,
        "additional_config": additional_config,
    }


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def _is_coherent(text):
    try:
        assert_output_coherent(text)
        return True
    except AssertionError:
        return False


def _degrades_midway(text):
    """True if the head reads as language but the tail is token-soup — the
    signature of decode-length corruption (clean start, garbage end)."""
    words = text.split()
    if len(words) < 8:
        return False
    head = " ".join(words[: len(words) // 2])
    tail = " ".join(words[len(words) // 2 :])
    return _is_coherent(head) and not _is_coherent(tail)


def _report(label, results):
    """Print a per-prompt report and return list[bool] coherence flags.

    results: list of (prompt, text, token_ids).
    """
    print(f"\n===== {label} =====")
    flags = []
    for i, (prompt, text, token_ids) in enumerate(results):
        coherent = _is_coherent(text)
        flags.append(coherent)
        tag = "OK " if coherent else "BAD"
        mid = (
            " [degrades-midway]"
            if (coherent is False and _degrades_midway(text))
            else ""
        )
        n_tok = len(token_ids) if token_ids is not None else -1
        print(f"[{tag}] #{i} ntok={n_tok}{mid} | {prompt!r}\n      -> {text!r}")
    n_bad = flags.count(False)
    print(
        f"----- {label}: {flags.count(True)}/{len(flags)} coherent, {n_bad} bad -----"
    )
    return flags


def _chat(llm, prompts, sampling_params):
    outputs = llm.chat(_to_messages(prompts), sampling_params)
    assert len(outputs) == len(prompts)
    return [
        (p, o.outputs[0].text, o.outputs[0].token_ids) for p, o in zip(prompts, outputs)
    ]


_GREEDY32 = dict(temperature=0.0, top_p=1.0, max_tokens=32)


# ---------------------------------------------------------------------------
# Probe runners. Each takes the loaded llm and does its chat call(s),
# returning True if the probe's outputs are all coherent (fix confirmed) so
# the test body can assert per the probe's `expect`.
# ---------------------------------------------------------------------------
def _run_once(llm, prompts=PROMPTS):
    sp = vllm.SamplingParams(**_GREEDY32)
    flags = _report("single greedy batch", _chat(llm, prompts, sp))
    return all(flags)


def _run_determinism(llm):
    """H7: identical batch twice must yield identical tokens."""
    sp = vllm.SamplingParams(**_GREEDY32)
    r1 = _chat(llm, PROMPTS, sp)
    r2 = _chat(llm, PROMPTS, sp)
    _report("determinism run 1", r1)
    _report("determinism run 2", r2)
    mismatched = [p for (p, _, t1), (_, _, t2) in zip(r1, r2) if list(t1) != list(t2)]
    print(f"\n[determinism] {len(mismatched)}/{len(r1)} prompts differ between runs")
    for p in mismatched:
        print(f"  NONDETERMINISTIC: {p!r}")
    # This is the real check for this probe; a race => mismatch.
    assert not mismatched, f"nondeterministic outputs (H7): {mismatched}"
    return all(_is_coherent(t) for _, t, _ in r1)


def _run_max_tokens_sweep(llm):
    """H1: does corruption only appear past a length threshold?"""
    all_clean_short = True
    for mt in (8, 24, 48):
        sp = vllm.SamplingParams(temperature=0.0, top_p=1.0, max_tokens=mt)
        flags = _report(f"max_tokens={mt}", _chat(llm, PROMPTS, sp))
        if mt <= 8 and not all(flags):
            all_clean_short = False
    print(
        "\n[sweep_max_tokens] Interpretation: clean at 8 then garbage at 24/48 "
        "=> H1 (length-gated). Garbage already at 8 => structural (H2/H3/H4/H5/H6)."
    )
    return all_clean_short  # informational; body does not hard-assert coherence


def _run_force_full_len(llm):
    """H1 positive test: force every seq to exactly 32 tokens (no early EOS).

    If short/EOS prompts (e.g. 'capital of France') now ALSO garble at 32
    tokens => pure decode-length (H1). If they stay clean while long ones
    garble => content-specific, not length (points at H4)."""
    sp = vllm.SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=32, min_tokens=32, ignore_eos=True
    )
    _report("force_full_len (ignore_eos, min_tokens=32)", _chat(llm, PROMPTS, sp))
    return True  # informational; inspect which prompts garble


def _run_reorder(llm):
    """Content- vs position/slot-bound. Same PROMPTS garble in both orders
    => content/length (H1/H4). Same POSITIONS garble => slot/device (H3/H7)."""
    sp = vllm.SamplingParams(**_GREEDY32)
    fwd = _chat(llm, PROMPTS, sp)
    rev = _chat(llm, list(reversed(PROMPTS)), sp)
    fwd_bad = {p for p, t, _ in fwd if not _is_coherent(t)}
    rev_bad = {p for p, t, _ in rev if not _is_coherent(t)}
    _report("reorder forward", fwd)
    _report("reorder reversed", rev)
    print(f"\n[reorder] bad prompts forward:  {sorted(fwd_bad)}")
    print(f"[reorder] bad prompts reversed: {sorted(rev_bad)}")
    print(
        "[reorder] Interpretation: same prompt SET bad in both => content-bound; "
        "bad set follows position instead => slot/device-bound."
    )
    return len(fwd_bad) == 0 and len(rev_bad) == 0


# ---------------------------------------------------------------------------
# Probe registry
# ---------------------------------------------------------------------------
# expect:
#   "reproduce" - known-failing reference; report only, never assert coherence
#   "clean"     - must be all-coherent (baseline / hoped-for fix)
#   "custom"    - runner does its own assertions (e.g. determinism)
PROBES = {
    # --- Wave 0: controls ---
    "baseline_homogeneous": dict(
        mesh_shape=[8, 4],
        max_num_seqs=8,
        ac={},
        expect="clean",
        runner=lambda llm: _run_once(llm, prompts=[PROMPTS[0]] * 8),
        note="8x the same long prompt; a homogeneous batch must be clean (control).",
    ),
    "baseline_determinism": dict(
        mesh_shape=[8, 4],
        max_num_seqs=8,
        ac={},
        expect="custom",
        runner=_run_determinism,
        note="H7: identical batch twice must produce identical tokens.",
    ),
    # --- Wave 1: characterize (single load each; no config change) ---
    "main": dict(
        mesh_shape=[8, 4],
        max_num_seqs=8,
        ac={},
        expect="reproduce",
        runner=_run_once,
        note="The failing reference. Report only.",
    ),
    "sweep_max_tokens": dict(
        mesh_shape=[8, 4],
        max_num_seqs=8,
        ac={},
        expect="reproduce",
        runner=_run_max_tokens_sweep,
        note="H1: length-gated?",
    ),
    "force_full_len": dict(
        mesh_shape=[8, 4],
        max_num_seqs=8,
        ac={},
        expect="reproduce",
        runner=_run_force_full_len,
        note="H1 positive test.",
    ),
    "reorder": dict(
        mesh_shape=[8, 4],
        max_num_seqs=8,
        ac={},
        expect="reproduce",
        runner=_run_reorder,
        note="content vs position.",
    ),
    # --- Wave 2: localize the code path (one reload each; TP stays 4) ---
    "cpu_sampling": dict(
        mesh_shape=[8, 4],
        max_num_seqs=8,
        ac={"cpu_sampling": True},
        expect="clean",
        runner=_run_once,
        note="H5: fused->unfused runtime + host argmax. Clean => fused path bug.",
    ),
    "const_eval_off": dict(
        mesh_shape=[8, 4],
        max_num_seqs=8,
        ac={"enable_const_eval": False},
        expect="clean",
        runner=_run_once,
        note="H8: const-eval mis-caching. Clean => const-eval bug.",
    ),
    "shard_weights_off": dict(
        mesh_shape=[8, 4],
        max_num_seqs=8,
        ac={"shard_weights_on_batch_axis": False},
        expect="reproduce",
        runner=_run_once,
        note=(
            "H2 (config-only). CAVEAT: model_runner's select_hidden_states spec "
            "(None,None,'batch') is NOT gated on this flag, so flag=False in DTP "
            "mode is likely INCOHERENT by construction and may also OOM (8x per-chip "
            "weights). A clean H2 test needs the coordinated source edit (flip the "
            "hidden spec to ('batch',None,None)) - see coherent_no_fsdp skip below."
        ),
    ),
    # --- Wave 3: confirm the axis (expensive; TP degree also changes) ---
    "mesh_dp2": dict(
        mesh_shape=[2, 16],
        max_num_seqs=8,
        ac={},
        expect="reproduce",
        runner=_run_once,
        note="DP=2, TP=16. Lowest DP reachable on 32 chips.",
    ),
    "mesh_dp4": dict(
        mesh_shape=[4, 8],
        max_num_seqs=8,
        ac={},
        expect="reproduce",
        runner=_run_once,
        note="DP=4, TP=8.",
    ),
}


@pytest.mark.nightly
@pytest.mark.tensor_parallel
@pytest.mark.bh_galaxy
@pytest.mark.parametrize("probe", list(PROBES), ids=lambda n: f"probe_{n}")
def test_dp_tp_debug_probe(probe: str):
    spec = PROBES[probe]
    print(f"\n########## PROBE: {probe} ##########")
    print(f"# {spec['note']}")

    llm_args = _base_llm_args(spec["mesh_shape"], spec["max_num_seqs"], **spec["ac"])
    print("llm_args:")
    for key, value in llm_args.items():
        print(f"  {key}: {value}")

    llm = vllm.LLM(**llm_args)
    all_coherent = spec["runner"](llm)
    check_host_memory(MODEL_NAME)

    expect = spec["expect"]
    if expect == "clean":
        assert all_coherent, f"probe {probe!r} expected all-coherent output"
    # "reproduce" and "custom": no coherence assertion here (report-only, or the
    # runner already asserted). Inspect the printed report to read the result.


@pytest.mark.skip(
    reason=(
        "Requires a source edit, not just config: set shard_weights_on_batch_axis=False "
        "AND flip model_runner select_hidden_states spec (None,None,'batch') -> "
        "('batch',None,None) so weights and activations stay coherent. This is the only "
        "clean end-to-end H2 A/B; may still OOM (replicated weights). Prefer the "
        "standalone reshard unit test for a memory-proof H2 check."
    )
)
def test_coherent_no_fsdp_placeholder():
    pass
