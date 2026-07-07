# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Standalone bring-up test for the XTTS-v2 autoregressive decode step.

The end-to-end pipeline (``examples/pytorch/xtts_pipeline.py``) drives the GPT2
audio-token loop on TT with a pre-allocated HF ``StaticCache`` so the single
decode graph compiles once and is reused every step (see the pipeline docstring).
This test pins down the two properties that claim depends on, directly against
the shipped ``GptCachedStep``:

    1. Correctness -- each decode step's logits match the same step run on CPU
       (per-step PCC), i.e. the on-device KV-cached step is numerically faithful
       to the eager CPU decode.
    2. Single-compile -- running many decode steps does NOT recompile per step.
       The compile count is flat from the second decode step onward (the first
       decode step compiles the reused graph; every later step is a cache hit).

The component runner (``xtts_v2/pytorch-gpt_decode-single_device-inference``)
PCC-checks one decode step in isolation; this test additionally exercises the
*loop* to prove the no-recompile property, which a single fixed-shape forward
cannot show.

Skipped unless the optional ``coqui-tts`` / ``torchaudio`` deps, the CPML-gated
weights, and a TT device are all available.
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch

pytestmark = [
    pytest.mark.nightly,
    pytest.mark.model_test,
    pytest.mark.single_device,
]

_REPO_ROOT = Path(__file__).resolve().parents[4]

# Number of decode steps to run. Enough to distinguish "single-compile" (flat
# compile count) from "recompiles per step" (count grows with steps).
N_DECODE_STEPS = 8
# Per-step correlation floor between TT and CPU decode logits.
PCC_THRESHOLD = 0.99


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    """Pearson correlation of two tensors, flattened and cast to float32."""
    a = a.detach().to("cpu").to(torch.float32).flatten()
    b = b.detach().to("cpu").to(torch.float32).flatten()
    va, vb = a - a.mean(), b - b.mean()
    denom = va.norm() * vb.norm()
    if denom == 0:
        return float("nan")
    return float((va @ vb) / denom)


def _load_example_module():
    """Import ``examples/pytorch/xtts_pipeline.py`` (not a package) by path."""
    path = _REPO_ROOT / "examples" / "pytorch" / "xtts_pipeline.py"
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    spec = importlib.util.spec_from_file_location("xtts_pipeline_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _run_decode_loop(decode_step, prefix_emb, cfg, start_token, n_steps, device):
    """Prefill ``[prefix, START]`` then run ``n_steps`` greedy decode steps.

    Mirrors ``XTTSPipeline._generate_codes_tt`` exactly (StaticCache built on CPU
    then moved to ``device``; one token per step; growing attention mask over a
    fixed-length cache). Returns ``(per_step_logits, compile_counts)`` where
    ``per_step_logits[i]`` is the last-position logit row for step ``i`` (on CPU)
    and ``compile_counts[i]`` is the cumulative torch_xla compile count observed
    right after step ``i`` (``None`` entries on CPU).
    """
    from transformers import StaticCache

    on_device = device != "cpu"
    if on_device:
        import torch_xla.debug.metrics as met

        def compile_count():
            data = met.metric_data("CompileTime")
            return data[0] if data else 0

    else:

        def compile_count():
            return None

    prefix_len = prefix_emb.shape[1]
    max_cache_len = prefix_len + n_steps + 1
    n_head = cfg.num_attention_heads
    head_dim = cfg.hidden_size // n_head

    cache = StaticCache(config=cfg, max_cache_len=max_cache_len)
    cache.early_initialization(
        batch_size=1,
        num_heads=n_head,
        head_dim=head_dim,
        dtype=torch.float32,
        device="cpu",
    )
    if on_device:
        for layer in cache.layers:
            layer.keys = layer.keys.to(device)
            layer.values = layer.values.to(device)
            layer.cumulative_length = layer.cumulative_length.to(device)
            layer.device = device

    decode_step = decode_step.to(device)

    def mask(valid):
        m = torch.zeros((1, max_cache_len), dtype=torch.long)
        m[:, :valid] = 1
        return m.to(device)

    per_step_logits = []
    compile_counts = []
    with torch.no_grad():
        # --- Prefill: [prefix, START(audio pos 0)] -> first audio token ---
        start_ids = torch.tensor([[start_token]], dtype=torch.long, device=device)
        pos0 = torch.tensor([0], dtype=torch.long, device=device)
        logits = decode_step(
            start_ids, pos0, mask(prefix_len + 1), cache, prefix_emb.to(device)
        )
        row = logits[:, -1, :].to("cpu")
        per_step_logits.append(row)
        compile_counts.append(compile_count())
        next_token = int(row.argmax(dim=-1).item())

        cur = prefix_len + 1
        # --- Decode loop: one token per step, audio position = step ---
        for step in range(1, n_steps):
            tok = torch.tensor([[next_token]], dtype=torch.long, device=device)
            pos = torch.tensor([step], dtype=torch.long, device=device)
            logits = decode_step(tok, pos, mask(cur + 1), cache, None)
            row = logits[:, -1, :].to("cpu")
            per_step_logits.append(row)
            compile_counts.append(compile_count())
            next_token = int(row.argmax(dim=-1).item())
            cur += 1

    return per_step_logits, compile_counts


_LOADER_PATH = str(
    _REPO_ROOT / "third_party" / "tt_forge_models" / "xtts_v2" / "pytorch" / "loader.py"
)


@pytest.fixture(scope="module")
def xtts():
    """Build the full XTTS model once, with the loader's own requirements.

    This test does not run through ``tests/runner/test_models.py`` (which wraps
    every component test in ``RequirementsManager.for_loader``), so we invoke the
    same manager here to install the loader's ``requirements.txt``
    (``coqui-tts`` + ``torchaudio``) for the duration and roll it back on exit --
    the manager stays open for the whole module (it yields inside the ``with``),
    since the loader's input derivation and this test both use torchaudio.
    """
    from tests.runner.requirements import RequirementsManager

    RequirementsManager.capture_golden_state()
    with RequirementsManager.for_loader(_LOADER_PATH, framework="torch"):
        # torchaudio is safe to probe directly; do NOT import TTS here -- its
        # import chain needs the isin_mps_friendly shim that _build_xtts installs
        # first (transformers>=5 dropped it), so let the loader do the TTS import.
        pytest.importorskip("torchaudio", reason="torchaudio not installed")

        os.environ.setdefault("COQUI_TOS_AGREED", "1")
        from third_party.tt_forge_models.xtts_v2.pytorch import ModelLoader

        loader = ModelLoader()
        try:
            loader._build_xtts()  # installs the shim, then imports TTS + weights
        except Exception as exc:  # deps missing / CPML weights ungated -> skip
            pytest.skip(f"Could not build XTTS-v2 (weights/deps): {exc}")
        yield loader


def test_decode_step_matches_cpu_and_single_compiles(xtts):
    """One-per-step PCC vs CPU + no per-step recompile on TT."""
    torch_xla = pytest.importorskip("torch_xla")
    import torch_xla.runtime as xr

    example = _load_example_module()

    loader = xtts
    model = loader._xtts
    gpt = model.gpt
    cfg = gpt.gpt.config

    # Real prefix embedding (cond latents + [START]text[STOP]), as the pipeline
    # and the GPT_PREFILL component build it.
    gpt_cond_latent = loader._gpt_cond_latent()
    text_tokens = loader._text_tokens()
    with torch.no_grad():
        gpt.compute_embeddings(gpt_cond_latent, text_tokens)
    prefix_emb = gpt.gpt_inference.cached_prefix_emb.clone()
    start_token = gpt.start_audio_token

    # --- CPU golden ---
    cpu_logits, _ = _run_decode_loop(
        example.GptCachedStep(model).eval(),
        prefix_emb,
        cfg,
        start_token,
        N_DECODE_STEPS,
        "cpu",
    )

    # --- TT ---
    xr.set_device_type("TT")
    if xr.global_runtime_device_count() < 1:
        pytest.skip("No TT device available")
    torch_xla.set_custom_compile_options({"optimization_level": 0})
    import torch_xla.core.xla_model as xm

    tt_logits, compile_counts = _run_decode_loop(
        example.GptCachedStep(model).eval(),
        prefix_emb,
        cfg,
        start_token,
        N_DECODE_STEPS,
        xm.xla_device(),
    )

    # 1) Per-step correctness vs CPU.
    pccs = [_pcc(tt_logits[i], cpu_logits[i]) for i in range(N_DECODE_STEPS)]
    worst = min(pccs)
    assert worst >= PCC_THRESHOLD, (
        f"decode-step logits diverged from CPU: per-step PCC={pccs}, "
        f"worst={worst:.5f} < {PCC_THRESHOLD}"
    )

    # 2) No per-step recompile: the decode graph compiles once (first decode
    # step) and every later step is a cache hit, so the compile count is flat
    # from step 2 onward regardless of how many steps we run.
    decode_counts = compile_counts[1:]  # index 0 is the prefill graph
    assert all(c is not None for c in decode_counts)
    growth = decode_counts[-1] - decode_counts[0]
    assert growth == 0, (
        "decode loop recompiled per step (expected a single reused graph): "
        f"cumulative compile counts across decode steps = {decode_counts}"
    )
