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
#5216), text tokenization, greedy token sampling / loop control, and audio I/O.

The autoregressive audio-token loop (``gpt_codes``) is the only host-driven
stage and it also runs the GPT2 trunk on TT. XTTS's GPT2 (transformers 5.x)
does not accept ``cache_position`` and so cannot use an HF ``StaticCache`` the
way the Llama loops do; instead we re-run a fixed-length (padded) sequence each
step with a growing attention mask, so the graph compiles exactly once and every
step reuses it (no per-step recompiles). Correct and single-compile, but
O(max_len) per step — cap generation with ``--max-audio-tokens``.

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


class XTTSConfig:
    def __init__(
        self,
        text: str = DEFAULT_TEXT,
        language: str = DEFAULT_LANGUAGE,
        speaker_wav: Optional[str] = None,
        max_audio_tokens: Optional[int] = None,
    ):
        self.text = text
        self.language = language
        # A reference speaker clip; defaults to the same public LibriSpeech
        # utterance the component loader uses so the pipeline runs out of the box.
        self.speaker_wav = speaker_wav
        # Cap on generated audio tokens (each ~= 1024 output samples / 24 kHz);
        # keeps the single-compile TT decode loop demo-sized. None = model max.
        self.max_audio_tokens = max_audio_tokens


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

    def _conditioning_mel(self, audio22: torch.Tensor) -> torch.Tensor:
        """First ``gpt_cond_len`` s of reference -> conditioning mel (CPU)."""
        from TTS.tts.models.xtts import wav_to_mel_cloning

        cond_len = self._loader.GPT_COND_LEN
        chunk = audio22[:, : 22050 * cond_len]
        return wav_to_mel_cloning(
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
        then greedily samples audio tokens one at a time; each step feeds only the
        new token (O(1) work), the StaticCache keeps shapes constant, and the
        decode graph is reused every step (no per-step recompiles).
        """
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
            next_token = int(logits[:, -1, :].argmax(dim=-1).to("cpu").item())

            cur = prefix_len + 1  # cache positions written so far
            # --- Decode loop: feed 1 token per step, audio position = step ---
            for step in range(1, max_tokens):
                if next_token == self.stop_audio_token:
                    break
                generated.append(next_token)
                tok = torch.tensor([[next_token]], dtype=torch.long, device=device)
                pos = torch.tensor([step], dtype=torch.long, device=device)
                logits = self.decode_step(tok, pos, mask(cur + 1), cache, None)
                next_token = int(logits[:, -1, :].argmax(dim=-1).to("cpu").item())
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
            cond_mel = self._conditioning_mel(audio22)
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
            logger.info("[STAGE] conditioning encoder (TT)")
            self.conditioning = self.conditioning.to(device)
            conds = cpu(self.conditioning(tt(cond_mel)))  # [1, 1024, 32]
            self.conditioning = self.conditioning.to("cpu")
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
    args = parser.parse_args()

    # torch_xla defaults to CPU; point it at the Tenstorrent device.
    xr.set_device_type("TT")

    run_xtts_pipeline(
        output_path=args.output,
        text=args.text,
        language=args.language,
        speaker_wav=args.speaker_wav,
        max_audio_tokens=args.max_audio_tokens,
    )
