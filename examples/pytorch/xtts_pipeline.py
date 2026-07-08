# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""XTTS-v2 (coqui/XTTS-v2) — end-to-end text-to-speech pipeline on Tenstorrent.

This glues the five per-component nn.Module graphs brought up in
``third_party/tt_forge_models/xtts_v2`` into the full ``Xtts.inference`` path
(see https://huggingface.co/coqui/XTTS-v2), running the heavy learned modules on
TT and keeping only the fixed DSP / orchestration on CPU. It mirrors the
component-then-pipeline pattern used by the Stable-Diffusion examples
(``sd_v1_4_pipeline.py``) and the Janus-Pro e2e pipeline: a model whose
components pass bring-up gets an e2e pipeline that chains them on device.

Stages (matching ``Xtts.inference`` + ``Xtts.get_conditioning_latents``):

    reference wav ─(CPU STFT/mel)─► speaker_encoder  [TT] ─► speaker_embedding
    reference wav ─(CPU mel)──────► conditioning     [TT] ─► gpt_cond_latent
    text ─(CPU tokenizer)─► gpt autoregressive loop  [TT] ─► gpt_codes
    text + gpt_codes ─────────────► gpt_latents      [TT] ─► gpt_latents
    gpt_latents + speaker_embedding► hifigan_decoder  [TT] ─► 24 kHz waveform

Every learned nn.Module runs on TT. CPU is used only for what does not lower to
device or is not a learned graph: the STFT/mel front-ends (complex FFT, problem
#5216), text tokenization, token sampling / loop control, and audio I/O. Token
sampling and conditioning use the reference ``Xtts.inference`` params (sampling:
temperature 0.75 / top_k 50 / top_p 0.85 / repetition_penalty 10.0; conditioning:
gpt_cond_len 6 / chunk 6), so behavior matches the source except for running the
learned modules on TT.

The autoregressive audio-token loop (``gpt_codes``) is the only host-driven
stage and it also runs the GPT2 trunk on TT. On CPU the original XTTS uses HF
``gpt.generate`` with a *dynamic* KV cache that grows one slot per step; those
ever-changing cache shapes would force a fresh compile every step on TT (torch_xla
keys its graph cache on tensor shapes). Instead we drive the loop with a
pre-allocated HF ``StaticCache`` (``GptCachedStep`` below): the K/V buffers are
sized once to ``prefix_len + max_tokens`` and written in place, and every step
feeds a fixed-shape single token, position, and ``[1, max_cache_len]`` attention
mask — only the mask *values* change, not shapes. XTTS's GPT2 (transformers 5.x)
does not pass ``cache_position``, so we rely on HF's ``StaticLayer`` self-managing
its write index (an internal ``cumulative_length`` + ``index_copy_``). Net effect:
the prefill graph and the single-token decode graph each compile exactly once and
every later step is a graph cache hit — no per-step recompiles, O(1) tokens
embedded per step (attention is over the fixed padded cache). Cap generation with
``--max-audio-tokens`` to bound ``max_cache_len``.

Requires the optional ``coqui-tts`` + ``torchaudio`` dependencies and the
CPML-gated coqui/XTTS-v2 weights (``COQUI_TOS_AGREED=1``; set ``HF_TOKEN`` or a
prior ``huggingface-cli login`` if not cached).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from loguru import logger

# Make the repo root importable so `third_party.tt_forge_models` resolves when
# this example is run directly (python examples/pytorch/xtts_pipeline.py).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# The tt_forge_models loader owns the weight download, the isin_mps_friendly
# shim, and the exact same component wrappers the bring-up tests use.
from third_party.tt_forge_models.xtts_v2.pytorch import ModelLoader, ModelVariant
from third_party.tt_forge_models.xtts_v2.pytorch.src.model import (
    ConditioningWrapper,
    GptLatentsWrapper,
    HifiganDecoderWrapper,
    SpeakerEncoderWrapper,
)

# Model-card example text; any real sentence works.
DEFAULT_TEXT = (
    "It took me quite a long time to develop a voice, and now that I have it "
    "I'm not going to be silent."
)
DEFAULT_LANGUAGE = "en"
OUTPUT_SAMPLE_RATE = 24000

# Reference autoregressive sampling params, matching ``Xtts.inference`` defaults
# (coqui/XTTS-v2). The decode loop applies these host-side via HF's own logits
# processors so token selection matches the reference generate() exactly.
REF_TEMPERATURE = 0.75
REF_TOP_K = 50
REF_TOP_P = 0.85
REF_REPETITION_PENALTY = 10.0
# Reference conditioning params, matching ``Xtts.get_conditioning_latents``
# defaults: gpt_cond_len == gpt_cond_chunk_len == 6 -> a single 6 s chunk, with
# multi-chunk mean when they differ (see ``Xtts.get_gpt_cond_latents``).
GPT_COND_LEN = 6
GPT_COND_CHUNK_LEN = 6
MIN_AUDIO_SECONDS = 0.33  # chunks shorter than this are skipped (reference)


class XTTSConfig:
    def __init__(
        self,
        text: str = DEFAULT_TEXT,
        language: str = DEFAULT_LANGUAGE,
        speaker_wav: Optional[str] = None,
        max_audio_tokens: Optional[int] = None,
        seed: int = 0,
    ):
        self.text = text
        self.language = language
        # A reference speaker clip; defaults to the same public LibriSpeech
        # utterance the component loader uses so the pipeline runs out of the box.
        self.speaker_wav = speaker_wav
        # Cap on generated audio tokens (each ~= 1024 output samples / 24 kHz);
        # keeps the single-compile TT decode loop demo-sized. None = model max.
        self.max_audio_tokens = max_audio_tokens
        # Seed for the (stochastic) reference sampling, so runs are reproducible.
        self.seed = seed


class GptCachedStep(torch.nn.Module):
    """GPT2 trunk + audio ``lm_head`` for one KV-cached decode step (Option B).

    Runs the shared GPT2 trunk with ``use_cache=True`` and an HF ``StaticCache``
    pre-allocated to a fixed ``max_cache_len``, so shapes stay constant across
    steps. The audio tokens are embedded here (``mel_embedding`` +
    ``mel_pos_embedding`` at the given absolute audio positions) so all learned
    ops run on TT; on the prefill step the conditioning+text ``prefix_emb`` is
    prepended.

    XTTS's GPT2 has a null ``wpe`` (positions come from ``mel_pos_embedding``),
    and HF's ``StaticLayer`` self-manages its write index via an in-place
    ``cumulative_length`` tensor + ``index_copy_`` (no ``cache_position`` from the
    model, no mutable Python state). So the prefill graph and the single-token
    decode graph each compile exactly once and the decode graph is a cache hit
    every subsequent step — no per-step recompiles, O(1) work per token.
    """

    def __init__(self, xtts):
        super().__init__()
        gpt = xtts.gpt
        self.gpt2 = gpt.gpt  # HF GPT2Model (wpe is null; positions are external)
        self.mel_embedding = gpt.mel_embedding
        self.mel_pos_embedding = gpt.mel_pos_embedding
        self.final_norm = gpt.final_norm
        self.mel_head = gpt.mel_head

    def forward(self, audio_ids, positions, attention_mask, past_key_values, prefix_emb=None):
        emb = self.mel_embedding(audio_ids)
        emb = emb + self.mel_pos_embedding.emb(positions).unsqueeze(0)
        if prefix_emb is not None:  # prefill only
            emb = torch.cat([prefix_emb.to(emb.dtype), emb], dim=1)
        out = self.gpt2(
            inputs_embeds=emb,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        hidden = self.final_norm(out.last_hidden_state)
        return self.mel_head(hidden)


class XTTSPipeline:
    """coqui/XTTS-v2 text-to-speech pipeline (components chained on TT)."""

    def __init__(self, config: XTTSConfig):
        self.config = config
        self._loader = ModelLoader(variant=ModelVariant.GPT_LATENTS)

    # ------------------------------------------------------------------ #
    # setup: build the full model once, wrap each component, compile for TT
    # ------------------------------------------------------------------ #
    def setup(self):
        os.environ.setdefault("COQUI_TOS_AGREED", "1")
        self.xtts = self._loader._build_xtts()  # downloads weights, eval mode
        gpt = self.xtts.gpt

        self.start_audio_token = gpt.start_audio_token
        self.stop_audio_token = gpt.stop_audio_token
        self.code_stride_len = gpt.code_stride_len
        self.model_max_audio_tokens = gpt.max_gen_mel_tokens

        # Component graphs (same wrappers as the bring-up tests). Register the
        # `tt` backend now; the actual .to(device) happens per stage in run().
        self.speaker_encoder = SpeakerEncoderWrapper(self.xtts).eval()
        self.conditioning = ConditioningWrapper(self.xtts).eval()
        self.gpt_latents = GptLatentsWrapper(self.xtts).eval()
        self.hifigan = HifiganDecoderWrapper(self.xtts).eval()

        self.speaker_encoder.compile(backend="tt")
        self.conditioning.compile(backend="tt")
        self.gpt_latents.compile(backend="tt")
        self.hifigan.compile(backend="tt")

        self.decode_step = GptCachedStep(self.xtts).eval()
        self.decode_step.compile(backend="tt")

    # ------------------------------------------------------------------ #
    # CPU preprocessing (mirrors get_conditioning_latents + tokenizer)
    # ------------------------------------------------------------------ #
    def _reference_audio_22k(self) -> torch.Tensor:
        import numpy as np
        import torchaudio

        if self.config.speaker_wav:
            audio, sr = torchaudio.load(self.config.speaker_wav)
            audio = audio.mean(0, keepdim=True)  # mono
        else:
            # Same public LibriSpeech reference the component loader uses.
            from third_party.tt_forge_models.tools.utils import get_file

            sample = torch.load(
                get_file(self._loader.REFERENCE_AUDIO), weights_only=False
            )
            sr = int(sample["audio"].get("sampling_rate", 16000))
            audio = torch.tensor(
                np.asarray(sample["audio"]["array"], dtype="float32")
            ).unsqueeze(0)
        if sr != 22050:
            audio = torchaudio.functional.resample(audio, sr, 22050)
        return audio

    def _speaker_mel(self, audio22: torch.Tensor) -> torch.Tensor:
        """16 kHz reference -> speaker-encoder mel (CPU torch.stft)."""
        import torchaudio

        audio16 = torchaudio.functional.resample(audio22, 22050, 16000)
        # use_torch_spec is disabled inside the wrapper, so feed the mel directly.
        with torch.no_grad():
            return self.xtts.hifigan_decoder.speaker_encoder.torch_spec(audio16)

    def _conditioning_mels(self, audio22: torch.Tensor) -> list:
        """Reference ``get_gpt_cond_latents`` chunking (perceiver path).

        The first ``GPT_COND_LEN`` s of the reference are split into
        ``GPT_COND_CHUNK_LEN``-second chunks; each chunk is turned into a mel on
        CPU. The conditioning encoder then runs per chunk on TT and the results
        are mean-averaged in ``run()`` -- exactly as ``Xtts.get_gpt_cond_latents``
        averages per-chunk ``get_style_emb`` outputs. With the reference defaults
        (6 == 6) this is a single 6 s chunk.
        """
        from TTS.tts.models.xtts import wav_to_mel_cloning

        audio = audio22[:, : 22050 * GPT_COND_LEN]
        step = 22050 * GPT_COND_CHUNK_LEN
        mels = []
        for i in range(0, audio.shape[1], step):
            chunk = audio[:, i : i + step]
            if chunk.size(-1) < 22050 * MIN_AUDIO_SECONDS:
                continue  # skip too-short trailing chunk (reference behavior)
            mels.append(
                wav_to_mel_cloning(
                    chunk,
                    mel_norms=self.xtts.mel_stats.cpu(),
                    n_fft=2048,
                    hop_length=256,
                    win_length=1024,
                    power=2,
                    normalized=False,
                    sample_rate=22050,
                    f_min=0,
                    f_max=8000,
                    n_mels=80,
                )
            )
        if not mels:
            raise RuntimeError(
                f"Reference audio too short (< {MIN_AUDIO_SECONDS:.2f}s) for conditioning."
            )
        return mels

    def _text_tokens(self) -> torch.Tensor:
        toks = self.xtts.tokenizer.encode(
            self.config.text.strip().lower(), lang=self.config.language
        )
        return torch.IntTensor(toks).unsqueeze(0)

    # ------------------------------------------------------------------ #
    # gpt_codes: autoregressive audio-token loop
    # ------------------------------------------------------------------ #
    def _make_static_cache(self, max_cache_len: int, device):
        """StaticCache for XTTS's GPT2, built on CPU then moved to device.

        Mirrors examples/pytorch/llama.py: build + early_initialization on CPU
        (a trace/fusion issue otherwise, tt-xla#1645), then move each layer's
        buffers to device. GPT2's StaticLayer self-manages its write index, so
        no cache_position is needed from the model.
        """
        from transformers import StaticCache

        cfg = self.xtts.gpt.gpt.config
        n_head = cfg.num_attention_heads
        head_dim = cfg.hidden_size // n_head
        cache = StaticCache(
            config=cfg,
            max_batch_size=1,
            max_cache_len=max_cache_len,
            device="cpu",
            dtype=torch.float32,
        )
        cache.early_initialization(
            batch_size=1,
            num_heads=n_head,
            head_dim=head_dim,
            dtype=torch.float32,
            device="cpu",
        )
        if device != "cpu":
            for layer in cache.layers:
                layer.keys = layer.keys.to(device)
                layer.values = layer.values.to(device)
                layer.cumulative_length = layer.cumulative_length.to(device)
                layer.device = device
        return cache

    def _generate_codes_tt(self, gpt_cond_latent, text_tokens) -> torch.Tensor:
        """KV-cached, single-compile decode loop with the GPT2 trunk on TT.

        Prefills the conditioning+text prefix + [START] token into a StaticCache,
        then samples audio tokens one at a time using the same params as the
        reference ``Xtts.inference`` (``do_sample=True``, temperature 0.75, top_k
        50, top_p 0.85, repetition_penalty 10.0). Sampling is host-side via HF's
        own logits processors, so token selection matches ``gpt.generate``; the TT
        decode graph feeds only the new token (O(1) work) with constant shapes, so
        it compiles once and is reused every step (no per-step recompiles).
        """
        from transformers import (
            LogitsProcessorList,
            RepetitionPenaltyLogitsProcessor,
            TemperatureLogitsWarper,
            TopKLogitsWarper,
            TopPLogitsWarper,
        )

        device = xm.xla_device()
        gpt = self.xtts.gpt

        # Prefix embedding (cond latents + [START]text[STOP]); host side.
        with torch.no_grad():
            gpt.compute_embeddings(gpt_cond_latent, text_tokens)
        prefix_emb = gpt.gpt_inference.cached_prefix_emb.clone().to(device)  # [1,P,1024]
        prefix_len = prefix_emb.shape[1]

        max_tokens = self.config.max_audio_tokens or self.model_max_audio_tokens
        max_tokens = int(min(max_tokens, self.model_max_audio_tokens))
        max_cache_len = prefix_len + max_tokens

        cache = self._make_static_cache(max_cache_len, device)
        self.decode_step = self.decode_step.to(device)

        def mask(valid):  # [1, max_cache_len]; 1s for written cache slots
            m = torch.zeros((1, max_cache_len), dtype=torch.long)
            m[:, :valid] = 1
            return m.to(device)

        # Reference sampling stack: repetition penalty (processor) then the
        # temperature/top-k/top-p warpers, in HF's generate() order. Applied on
        # the CPU logits each step; the running audio-token sequence (starting
        # with [START]) drives the repetition penalty, exactly like HF generate.
        processors = LogitsProcessorList(
            [
                RepetitionPenaltyLogitsProcessor(penalty=REF_REPETITION_PENALTY),
                TemperatureLogitsWarper(REF_TEMPERATURE),
                TopKLogitsWarper(REF_TOP_K),
                TopPLogitsWarper(REF_TOP_P),
            ]
        )
        rng = torch.Generator().manual_seed(int(self.config.seed))
        seq = [self.start_audio_token]  # running sequence for repetition penalty

        def sample(logits_row_cpu):  # logits_row_cpu: [1, vocab] on CPU
            input_ids = torch.tensor([seq], dtype=torch.long)
            scores = processors(input_ids, logits_row_cpu.float())
            probs = torch.softmax(scores, dim=-1)
            return int(torch.multinomial(probs, num_samples=1, generator=rng).item())

        generated = []
        with torch.no_grad():
            # --- Prefill: [prefix, START(audio pos 0)] -> first audio token ---
            start_ids = torch.tensor(
                [[self.start_audio_token]], dtype=torch.long, device=device
            )
            pos0 = torch.tensor([0], dtype=torch.long, device=device)
            logits = self.decode_step(
                start_ids, pos0, mask(prefix_len + 1), cache, prefix_emb
            )
            next_token = sample(logits[:, -1, :].to("cpu"))

            cur = prefix_len + 1  # cache positions written so far
            # --- Decode loop: feed 1 token per step, audio position = step ---
            for step in range(1, max_tokens):
                if next_token == self.stop_audio_token:
                    break
                generated.append(next_token)
                seq.append(next_token)  # extend history before sampling the next
                tok = torch.tensor([[next_token]], dtype=torch.long, device=device)
                pos = torch.tensor([step], dtype=torch.long, device=device)
                logits = self.decode_step(tok, pos, mask(cur + 1), cache, None)
                next_token = sample(logits[:, -1, :].to("cpu"))
                cur += 1
                if step % 32 == 0:
                    logger.info(f"[gpt_codes] {step}/{max_tokens} tokens")

        self.decode_step = self.decode_step.to("cpu")
        return torch.tensor(generated, dtype=torch.long).unsqueeze(0)

    # ------------------------------------------------------------------ #
    # run: full text -> waveform
    # ------------------------------------------------------------------ #
    def run(self) -> torch.Tensor:
        device = xm.xla_device()
        tt = lambda x: x.to(device)
        cpu = lambda x: x.to("cpu")

        with torch.no_grad():
            # --- CPU preprocessing ---
            audio22 = self._reference_audio_22k()
            speaker_mel = self._speaker_mel(audio22)
            cond_mels = self._conditioning_mels(audio22)
            text_tokens = self._text_tokens()
            text_len = torch.tensor([text_tokens.shape[-1]])

            # --- speaker_embedding [TT] ---
            logger.info("[STAGE] speaker_encoder (TT)")
            self.speaker_encoder = self.speaker_encoder.to(device)
            speaker_embedding = cpu(self.speaker_encoder(tt(speaker_mel)))
            self.speaker_encoder = self.speaker_encoder.to("cpu")
            # Match get_speaker_embedding output shape [1, 512, 1].
            speaker_embedding = speaker_embedding.unsqueeze(-1)

            # --- gpt_cond_latent [TT] ---
            # Per-chunk conditioning on TT, then mean over chunks (reference
            # get_gpt_cond_latents). Each chunk's style_emb is [1, 1024, 32].
            logger.info("[STAGE] conditioning encoder (TT)")
            self.conditioning = self.conditioning.to(device)
            style_embs = [cpu(self.conditioning(tt(m))) for m in cond_mels]
            self.conditioning = self.conditioning.to("cpu")
            conds = torch.stack(style_embs).mean(dim=0)  # [1, 1024, 32]
            gpt_cond_latent = conds.transpose(1, 2)  # [1, 32, 1024]

            # --- gpt_codes (autoregressive, GPT2 trunk on TT) ---
            logger.info("[STAGE] gpt_codes (TT)")
            gpt_codes = self._generate_codes_tt(gpt_cond_latent, text_tokens)
            logger.info(f"[gpt_codes] produced {gpt_codes.shape[-1]} audio tokens")

            expected_output_len = torch.tensor(
                [gpt_codes.shape[-1] * self.code_stride_len]
            )

            # --- gpt_latents [TT] ---
            logger.info("[STAGE] gpt_latents (TT)")
            self.gpt_latents = self.gpt_latents.to(device)
            gpt_latents = cpu(
                self.gpt_latents(
                    tt(text_tokens),
                    tt(text_len),
                    tt(gpt_codes),
                    tt(expected_output_len),
                    tt(gpt_cond_latent),
                )
            )
            self.gpt_latents = self.gpt_latents.to("cpu")

            # --- waveform [TT] ---
            logger.info("[STAGE] hifigan_decoder (TT)")
            self.hifigan = self.hifigan.to(device)
            wav = cpu(self.hifigan(tt(gpt_latents), tt(speaker_embedding)))
            self.hifigan = self.hifigan.to("cpu")

        return wav  # [1, 1, S] @ 24 kHz


def save_wav(wav: torch.Tensor, filepath: str = "xtts_output.wav"):
    """Save the pipeline waveform as a 24 kHz 16-bit PCM WAV.

    Uses the stdlib ``wave`` module rather than ``torchaudio.save`` so it does
    not depend on FFmpeg / torchcodec (often absent on bring-up hosts).
    """
    import wave

    audio = wav.detach().cpu().float().reshape(-1)  # mono [S]
    audio = torch.clamp(audio, -1.0, 1.0)
    pcm = (audio * 32767.0).round().to(torch.int16).numpy()
    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(OUTPUT_SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())
    return filepath


def run_xtts_pipeline(
    output_path: str = "xtts_output.wav",
    text: str = DEFAULT_TEXT,
    language: str = DEFAULT_LANGUAGE,
    speaker_wav: Optional[str] = None,
    max_audio_tokens: Optional[int] = None,
    seed: int = 0,
):
    """Run the XTTS-v2 pipeline and write a WAV file. Returns the path."""
    # optimization_level 0: the memory-layout optimizer probes ttnn op
    # constraints by allocating buffers on-device at compile time, which can
    # OOM when an earlier stage's weights are still resident. Level 0 skips it.
    torch_xla.set_custom_compile_options({"optimization_level": 0})

    config = XTTSConfig(
        text=text,
        language=language,
        speaker_wav=speaker_wav,
        max_audio_tokens=max_audio_tokens,
        seed=seed,
    )
    pipeline = XTTSPipeline(config)
    pipeline.setup()
    wav = pipeline.run()
    save_wav(wav, output_path)
    logger.info(f"[XTTS] wrote {output_path} ({wav.shape[-1]} samples @ {OUTPUT_SAMPLE_RATE} Hz)")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XTTS-v2 e2e text-to-speech on TT")
    parser.add_argument("--text", default=DEFAULT_TEXT, help="Text to synthesize")
    parser.add_argument("--language", default=DEFAULT_LANGUAGE)
    parser.add_argument(
        "--speaker-wav",
        default=None,
        help="Reference speaker WAV (defaults to a public LibriSpeech clip)",
    )
    parser.add_argument("--output", default="xtts_output.wav")
    parser.add_argument(
        "--max-audio-tokens",
        type=int,
        default=None,
        help="Cap on generated audio tokens (keeps the TT decode demo short)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the reference stochastic sampling (reproducible runs)",
    )
    args = parser.parse_args()

    # torch_xla defaults to CPU; point it at the Tenstorrent device.
    xr.set_device_type("TT")

    run_xtts_pipeline(
        output_path=args.output,
        text=args.text,
        language=args.language,
        speaker_wav=args.speaker_wav,
        max_audio_tokens=args.max_audio_tokens,
        seed=args.seed,
    )
