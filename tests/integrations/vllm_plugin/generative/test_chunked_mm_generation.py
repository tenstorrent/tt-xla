# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""On-device chunked multi-modal prefill equivalence (tt-xla #5824).

Greedy generation must agree whether an image's placeholder span is prefilled in
one chunk or split across several. Unlike
``test_chunked_mm_embed_equivalence.py`` (host-side index arithmetic only), this
runs the real path on device: cached encoder output -> per-chunk slice ->
``index_copy`` scatter -> cached-prefix attention -> logits -> sampled tokens.

The vision tower is replaced by deterministic synthetic embeddings. That is
deliberate and load-bearing, not a convenience: Qwen2-VL's vision encoder
currently emits ~39% ``inf`` on TT (tt-xla #5858), and comparing two token
streams computed from saturated inputs proves nothing -- any difference in graph
shape just relocates the ``inf``/``nan``. Substituting well-scaled values makes
the comparison meaningful and keeps this test independent of #5858. Everything
downstream of the encoder -- the part #5824 actually changed -- is real.

Each configuration runs in its own subprocess: the in-process engine
(``VLLM_ENABLE_V1_MULTIPROCESSING=0``, required so the class-level patch reaches
the model runner) calls ``sys.exit`` on shutdown, so two engines cannot share
one interpreter.
"""

import base64
import io
import json
import os
import subprocess
import sys

import pytest

_MODEL = "Qwen/Qwen2-VL-2B-Instruct"
_MAX_MODEL_LEN = 2048
_MAX_TOKENS = 16
# ~1024 image tokens (roughly max_pixels/784): larger than the split chunk below,
# and small enough that the vision tower still compiles.
_MAX_PIXELS = 802816
_SPLIT_CHUNK = 512  # image spans 3 chunks
_WHOLE_CHUNK = _MAX_MODEL_LEN  # single prefill chunk (the oracle)

# The two paths use different attention kernels (single-shot SDPA vs the chunked
# SDPA op) and different prefill bucket shapes, so low-precision drift can flip a
# late greedy argmax. Compare a leading prefix, as test_chunked_prefill.py does:
# a real slice/scatter bug corrupts the very first generated token.
_MATCH_PREFIX_TOKENS = 8
_WORKER_TIMEOUT = 1800


def _image_url() -> str:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (1024, 1024), (20, 80, 200))
    ImageDraw.Draw(image).rectangle([256, 256, 768, 768], fill=(220, 40, 40))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


def _install_synthetic_encoder() -> None:
    """Replace the vision tower with deterministic, well-scaled embeddings.

    Patches ``_execute_mm_encoder`` -- the single seam between "run the ViT" and
    "publish rows into ``encoder_cache``" -- so the slicing/scatter path under
    test is untouched. Values are bounded (|x| < 0.02) and vary per row and
    column, so a misplaced row changes the result.
    """
    # `vllm` must be imported first: importing vllm_tt directly re-enters the
    # plugin registration and raises "partially initialized module 'vllm_tt'".
    import torch
    import vllm  # noqa: F401
    from vllm_tt.model_runner import TTModelRunner

    def _fake_execute_mm_encoder(self, scheduler_output):
        scheduled = scheduler_output.scheduled_encoder_inputs
        if not scheduled:
            return
        hidden = self.inputs_embeds_size
        for req_id, input_ids in scheduled.items():
            req_state = self.requests[req_id]
            for i in input_ids:
                feature = req_state.mm_features[i]
                mm_hash = feature.identifier
                if mm_hash in self.encoder_cache:
                    continue
                n = feature.mm_position.get_num_embeds()
                rows = torch.arange(n, dtype=torch.float32).unsqueeze(1)
                cols = torch.arange(hidden, dtype=torch.float32).unsqueeze(0)
                # Bounded, non-repeating across both axes.
                vals = torch.sin(rows * 0.017 + cols * 0.011) * 0.02
                self.encoder_cache[mm_hash] = vals.to(torch.bfloat16).to(self.device)

    TTModelRunner._execute_mm_encoder = _fake_execute_mm_encoder


def _worker(chunk: int) -> int:
    """Build one engine, generate greedily, print the token ids as JSON."""
    import vllm

    _install_synthetic_encoder()

    llm = vllm.LLM(
        model=_MODEL,
        max_num_seqs=1,
        max_model_len=_MAX_MODEL_LEN,
        gpu_memory_utilization=0.1,
        enable_prefix_caching=False,
        limit_mm_per_prompt={"image": 1, "video": 0, "audio": 0},
        mm_processor_kwargs={"max_pixels": _MAX_PIXELS},
        additional_config={
            "min_context_len": 512,
            "enable_tensor_parallel": False,
            "cpu_sampling": True,
            "prefill_chunk_size": chunk,
        },
    )
    messages = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _image_url()}},
                    {"type": "text", "text": "Describe this image in one sentence."},
                ],
            }
        ]
    ]
    sp = vllm.SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=_MAX_TOKENS, ignore_eos=True
    )
    out = llm.chat(messages, sp)[0].outputs[0]
    print(f"__RESULT__{json.dumps(list(out.token_ids))}", flush=True)
    return 0


def _run_worker(chunk: int) -> list[int]:
    env = {**os.environ, "VLLM_ENABLE_V1_MULTIPROCESSING": "0"}
    proc = subprocess.run(
        [sys.executable, __file__, str(chunk)],
        env=env,
        capture_output=True,
        text=True,
        timeout=_WORKER_TIMEOUT,
    )
    marker = [ln for ln in proc.stdout.splitlines() if ln.startswith("__RESULT__")]
    if not marker:
        print(proc.stdout[-4000:], flush=True)
        print(proc.stderr[-4000:], file=sys.stderr, flush=True)
        pytest.fail(
            f"worker for chunk={chunk} produced no result (exit={proc.returncode})"
        )
    return json.loads(marker[-1][len("__RESULT__") :])


def _require_qwen2vl_vision_rope() -> None:
    """Skip until the Qwen2-VL vision rotary embedding lands (tt-xla #5859).

    Qwen2VisionTransformer.rot_pos_emb calls ``get_cos_sin`` on the rotary module,
    which the TT replacement (``TTRotaryEmbedding``) does not implement yet, so
    the model cannot be built at all. Nothing about #5824 depends on it -- this is
    purely the vehicle. The check runs here rather than at import so collection
    stays cheap.
    """
    import vllm  # noqa: F401  must precede vllm_tt (plugin registration)
    from vllm_tt.layers.rotary_embedding import TTRotaryEmbedding

    if not hasattr(TTRotaryEmbedding, "get_cos_sin"):
        pytest.skip(
            "TTRotaryEmbedding.get_cos_sin is required to build Qwen2-VL's vision "
            "tower; blocked on tt-xla #5859"
        )


@pytest.mark.nightly
@pytest.mark.single_device
def test_chunked_mm_generation_matches_whole_item():
    """Splitting an image across prefill chunks must not change greedy output."""
    _require_qwen2vl_vision_rope()

    whole = _run_worker(_WHOLE_CHUNK)
    split = _run_worker(_SPLIT_CHUNK)

    print(f"whole-item prefill: {whole}")
    print(f"split-item prefill: {split}")

    n = _MATCH_PREFIX_TOKENS
    assert len(whole) >= n and len(split) >= n, (
        f"need >= {n} generated tokens to compare "
        f"(whole={len(whole)}, split={len(split)})"
    )
    assert whole[:n] == split[:n], (
        f"first {n} greedy tokens differ between whole-item and split-item "
        "prefill -- the per-chunk encoder-output slice or its scatter position is "
        f"wrong.\n  whole={whole}\n  split={split}"
    )


if __name__ == "__main__":
    sys.exit(_worker(int(sys.argv[1])))
