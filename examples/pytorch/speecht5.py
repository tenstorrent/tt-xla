# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from typing import Optional, Union, Any
import types
import time
import torch
import torch.nn as nn
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr

from datasets import load_dataset
import soundfile as sf

from transformers import SpeechT5Config, SpeechT5PreTrainedModel, SpeechT5Processor, SpeechT5HifiGan, SpeechT5ForTextToSpeech
from transformers.cache_utils import EncoderDecoderCache, StaticCache
from transformers.models.speecht5.modeling_speecht5 import SpeechT5EncoderWithSpeechPrenet


EXPORT_PATH = "speecht5"


class SpeechT5RelativePositionalEncodingFixed(nn.Module):
    """
    Implementation of the relative positional encoding that avoids advanced indexing
    to avoid graph breaks with TT compile.
    """
    def __init__(self, original_module):
        super().__init__()
        self.dim = original_module.dim
        self.max_length = original_module.max_length
        self.pe_k = original_module.pe_k

    def forward(self, hidden_states):
        seq_len = hidden_states.shape[1]
        pos_seq = torch.arange(0, seq_len, device=hidden_states.device, dtype=torch.long)
        pos_seq = pos_seq.unsqueeze(-1) - pos_seq.unsqueeze(0)

        # Replacing advanced indexing with clamp to avoid graph breaks with TT compile.
        # Original code that causes issues:
        #   pos_seq[pos_seq < -self.max_length] = -self.max_length
        #   pos_seq[pos_seq >= self.max_length] = self.max_length - 1
        pos_seq = torch.where(
            pos_seq < -self.max_length, -self.max_length,
            torch.where(pos_seq >= self.max_length, self.max_length - 1, pos_seq)
        )
        pos_seq = pos_seq + self.max_length

        return self.pe_k(pos_seq)


# This is a workaround to avoid an issue in ttnn.fill_cache, where the size of the fill values is too large to split the work over
# the device grid.
def static_cache_layer_update_workaround(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update the key and value caches in-place, and return the necessary keys and value states.

        Args:
            key_states (`torch.Tensor`): The new key states to cache.
            value_states (`torch.Tensor`): The new value states to cache.
            cache_kwargs (`dict[str, Any]`, *optional*): Additional arguments for the cache.

        Returns:
            tuple[`torch.Tensor`, `torch.Tensor`]: The key and value states.
        """
        # Lazy initialization
        if not self.is_initialized:
            self.lazy_initialization(key_states)

        # Some old models give None for `cache_position` or even omit passing `cache_kwargs` when used as cross-attention,
        # in which case we should copy the whole Layer (key_states.shape[-2] == self.max_cache_len)
        cache_position = cache_kwargs.get("cache_position") if cache_kwargs is not None else None

        # If there is no cache position, instead of updating the cache at every position in the states, we copy the entire states into the cache.
        # This works because with the generate implementation, prefill states are always the same size as the cache.
        if cache_position is None:
            self.keys[:, :, :key_states.shape[2], :] = key_states
            self.values[:, :, :value_states.shape[2], :] = value_states
            return self.keys, self.values

        # Update the cache
        try:
            self.keys.index_copy_(2, cache_position, key_states)
            self.values.index_copy_(2, cache_position, value_states)
        except NotImplementedError:
            # Fallback for devices like MPS where index_copy_ might not be supported.
            self.keys[:, :, cache_position] = key_states
            self.values[:, :, cache_position] = value_states
        return self.keys, self.values


def apply_cache_workaround(cache: EncoderDecoderCache):
    for layer in cache.self_attention_cache.layers:
        layer.update = types.MethodType(static_cache_layer_update_workaround, layer)
    for layer in cache.cross_attention_cache.layers:
        layer.update = types.MethodType(static_cache_layer_update_workaround, layer)

def get_model():
    config = SpeechT5Config.from_pretrained("microsoft/speecht5_tts")
    config.max_speech_positions = 8192
    model = SpeechT5ForTextToSpeech.from_pretrained(
        "microsoft/speecht5_tts", config=config
    )
    model.eval()
    model.speecht5.encoder.wrapped_encoder.embed_positions = SpeechT5RelativePositionalEncodingFixed(model.speecht5.encoder.wrapped_encoder.embed_positions)
    return model


def get_processor():
    processor = SpeechT5Processor.from_pretrained(
        "microsoft/speecht5_tts"
    )   
    return processor

def get_vocoder():
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
    return vocoder


def get_input(sentence: str, device: torch.device):
    processor = get_processor()

    inputs = processor(
        text=sentence, 
        return_tensors="pt")

    inputs = {
        key : nn.functional.pad(value, (0, next_power_of_two(value.shape[1]) - value.shape[1])).to(device) 
        for key, value in inputs.items()}
    
    return inputs


def get_speaker_embeddings():
    """
    Load speaker embeddings from CMU Arctic dataset.
    This file looks messy because HF docs are not updated, so we're downloading the file however we can.
    """
    import pandas as pd
    import urllib.request
    import tempfile

    url = "https://huggingface.co/datasets/Matthijs/cmu-arctic-xvectors/resolve/refs%2Fconvert%2Fparquet/default/validation/0000.parquet"

    with tempfile.NamedTemporaryFile(delete=False, suffix=".parquet") as tmp_file:
        urllib.request.urlretrieve(url, tmp_file.name)

        df = pd.read_parquet(tmp_file.name)

        # Speaker embedding at index 7306 according to example https://huggingface.co/microsoft/speecht5_tts
        xvector = df.iloc[7306]["xvector"]
        speaker_embeddings = torch.tensor(xvector, dtype=torch.float32).unsqueeze(0)

        os.unlink(tmp_file.name)
    return speaker_embeddings.to(dtype=torch.float32) # speaker embeddings are only used in decoder prenet, for accuracy purposes we use float32

def dump_tensors():
    xr.set_device_type("TT")

    model = get_model()
    model.compile(backend="tt")

    model_inputs = get_input()

    # Connect the device.
    device = xm.xla_device()

    # Move inputs and model to device if needed
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    model = model.to(device)

    torch_xla.set_custom_compile_options(
        {
            "export_path": EXPORT_PATH,
            "dump_inputs": True,
        }
    )
    output = model(**model_inputs)

    return

def dump_code():
    xr.set_device_type("TT")

    model = get_model()
    model.compile(backend="tt")

    device = xm.xla_device()

    model_inputs = get_input()

    # Move inputs and model to device if needed
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    model = model.to(device)

    torch_xla.set_custom_compile_options(
        {
            "export_path": EXPORT_PATH,
            "backend": "codegen_py",
        }
    )

    output = model(**model_inputs)
    print(output)


def run_on_cpu():
    model = get_model()
    processor = get_processor()

    # Process text input using the processor
    text = "Hello, my dog is cute."
    inputs = processor(text=text, return_tensors="pt", max_length=256, padding="max_length")
    
    # Create decoder input values (zeros for initial state)
    decoder_input_values = torch.zeros((1, 1, model.config.num_mel_bins))

    # Prepare model inputs
    model_inputs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs["attention_mask"],
        "decoder_input_values": decoder_input_values,
    }

    output = model(**model_inputs)
    print(output)


def next_power_of_two(x: int):
    return 1 << (x - 1).bit_length()

def pad_spectrograms_to_nearest_power_of_two(tensor: torch.Tensor):
    dim_size = tensor.shape[1]
    padding = next_power_of_two(dim_size) - dim_size
    return torch.nn.functional.pad(tensor, (0, 0, 0, padding))

@torch.no_grad()
def _generate_speech(
    model: SpeechT5PreTrainedModel,
    input_ids: torch.FloatTensor,
    speaker_embeddings: Optional[torch.FloatTensor] = None,
    attention_mask: Optional[torch.LongTensor] = None,
    threshold: float = 0.5,
    minlenratio: float = 0.0,
    maxlenratio: float = 20.0,
    vocoder: Optional[nn.Module] = None,
    output_cross_attentions: bool = False,
    return_output_lengths: bool = False,
    is_precompile: bool = False,
) -> Union[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor]]:
    if speaker_embeddings is None:
        raise ValueError(
            """`speaker_embeddings` must be specified. For example, you can use a speaker embeddings by following
                    the code snippet provided in this link:
                    https://huggingface.co/datasets/Matthijs/cmu-arctic-xvectors
                    """
        )
    device = input_ids.device
    if attention_mask is None:
        encoder_attention_mask = 1 - (input_ids == model.config.pad_token_id).int()
    else:
        encoder_attention_mask = attention_mask

    bsz = input_ids.size(0)
    encoder_len = input_ids.size(1)
    encoder_out = model.speecht5.encoder(
        input_values=input_ids,
        attention_mask=encoder_attention_mask,
        return_dict=True,
    )

    encoder_last_hidden_state = encoder_out.last_hidden_state

    # downsample encoder attention mask
    if isinstance(model.speecht5.encoder, SpeechT5EncoderWithSpeechPrenet):
        encoder_attention_mask = model.speecht5.encoder.prenet._get_feature_vector_attention_mask(
            encoder_out[0].shape[1], encoder_attention_mask
        )

    maxlen = int(encoder_last_hidden_state.size(1) * maxlenratio / model.config.reduction_factor)
    minlen = int(encoder_last_hidden_state.size(1) * minlenratio / model.config.reduction_factor)
    
    # Start the output sequence with a mel spectrum that is all zeros.
    max_decoder_len = next_power_of_two(maxlen)


    # If the encoder length and max decoder length are not the same, we must pad the encoder attention mask.
    if encoder_len < max_decoder_len:
        encoder_attention_mask = nn.functional.pad(encoder_attention_mask.to("cpu"), (0, max_decoder_len - encoder_len)).to(device)
    elif encoder_len > max_decoder_len:
        raise ValueError(f"Encoder length {encoder_len} is greater than max decoder length {max_decoder_len}")

    output_sequence = torch.zeros(bsz, max_decoder_len, model.config.num_mel_bins, dtype=encoder_last_hidden_state.dtype)
    
    spectrogram = []
    cross_attentions = []

    # Setup an encoder-decoder cache using static caches.
    past_key_values = EncoderDecoderCache(StaticCache(config=model.config, max_cache_len=max_decoder_len, max_batch_size=bsz), StaticCache(config=model.config, max_cache_len=max_decoder_len, max_batch_size=bsz))

    #Re-assign the update method to use the workaround.
    apply_cache_workaround(past_key_values)

    cache_position = torch.tensor([0], dtype=torch.int32, device=device)
    idx = 0
    result_spectrogram = {}
    while True and (not is_precompile or idx < 2):
        idx += 1

        # Run the decoder prenet on the entire output sequence.
        # Move outputs to cpu so the slice we perform on `decoder_hidden_states` doesn't cause a graph break. If we do not do this we will generate a slicing program for every `idx`
        decoder_hidden_states = model.speecht5.decoder.prenet(output_sequence.to(torch.float32).to(device), speaker_embeddings).to("cpu").to(torch.bfloat16)
        attention_mask = torch.ones(1, max_decoder_len, dtype=torch.int32)
        attention_mask[0, idx:] = 0

        decoder_out = model.speecht5.decoder.wrapped_decoder(
            hidden_states=decoder_hidden_states[:, idx-1:idx].to(device),
            attention_mask=attention_mask.to(device),
            encoder_hidden_states=encoder_last_hidden_state,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            output_attentions=output_cross_attentions,
            return_dict=True,
            cache_position=cache_position,
        )

        cache_position = cache_position + 1

        if output_cross_attentions:
            cross_attentions.append(torch.cat(decoder_out.cross_attentions, dim=0))

        last_decoder_output = decoder_out.last_hidden_state.squeeze(1)
        past_key_values = decoder_out.past_key_values

        # Predict the new mel spectrum for this step in the sequence.
        spectrum = model.speech_decoder_postnet.feat_out(last_decoder_output).to("cpu")
        spectrum = spectrum.view(bsz, model.config.reduction_factor, model.config.num_mel_bins)
        # The amount of sepctra we append to the spectrogram is determined dynamically by the number of generation loops.
        # Before the spectrogram is passed to the postnet, these tensors are stacked. This means the stack operation has
        # a dynamic shape as well. To avoid generating a program for the `stack` for every possible number of spectra,
        # we store the spectra on cpu and stack them there.
        spectrogram.append(spectrum)

        # Extend the output sequence with the new mel spectrum.
        new_spectrogram = spectrum[:, -1, :].view(bsz, 1, model.config.num_mel_bins).to("cpu")

        output_sequence[:, idx, :] = new_spectrogram

        # Predict the probability that this is the stop token.
        prob = torch.sigmoid(model.speech_decoder_postnet.prob_out(last_decoder_output))

        original_spectrogram_length = len(spectrogram) # to be populated below
        if idx < minlen:
            continue
        else:
            # If the generation loop is less than maximum length time, check the ones in the batch that have met
            # the prob threshold. Otherwise, assume all have met thresholds and fill other spectrograms for the batch.
            if idx < maxlen:
                meet_thresholds = torch.sum(prob, dim=-1) >= threshold
                meet_indexes = torch.where(meet_thresholds)
                if meet_indexes[0].numel() > 0:
                    meet_indexes = meet_indexes[0].tolist()
                else:
                    meet_indexes = []
            else:
                meet_indexes = range(len(prob))
            meet_indexes = [i for i in meet_indexes if i not in result_spectrogram]
            if len(meet_indexes) > 0:
                # As stated in an earlier comment, the number of spectra in `spectrogram` is dynamic.
                # We must stack the spectra on cpu to avoid generation of many graphs for `stack`.
                spectrograms = torch.stack(spectrogram)
                spectrograms = spectrograms.transpose(0, 1).flatten(1, 2)

                # Similar to the reasoning for stacking the spectra on cpu, we will also pad the spectrogram height
                # to the nearest power of two. This prevents compiling a new graph for `postnet` for every possible
                # spectrogram height. This way, we only compile a program for every power of two.
                original_spectrogram_length = spectrograms.shape[1]
                spectrograms = pad_spectrograms_to_nearest_power_of_two(spectrograms)
                # Run postnet on device

                spectrograms = model.speech_decoder_postnet.postnet(spectrograms.to(device))
                for meet_index in meet_indexes:
                    result_spectrogram[meet_index] = spectrograms[meet_index]
            if len(result_spectrogram) >= bsz:
                break
    
    # The vocoder outputs a waveform in the form of a 1D tensor,
    # This waveform is always 256x the height of the spectrogram.
    # Since the spectrogram is padded, we want to truncate the 
    # waveform as well to cut out the noise caused by the padding.
    # The output waveform is directly converted to a `.wav` file,
    # so this tensor must end up on host regardless. Thus we place
    # the waveform on host and truncate it, this part should have
    # a minimal impact on performance.
    true_waveform_length = original_spectrogram_length * 256
    spectrograms = [result_spectrogram[i] for i in range(len(result_spectrogram))]
    if not return_output_lengths:
        spectrogram = spectrograms[0] if bsz == 1 else torch.nn.utils.rnn.pad_sequence(spectrograms, batch_first=True)
        if vocoder is not None:
            outputs = vocoder(spectrogram.to("cpu")).to("cpu")[:true_waveform_length]
        else:
            outputs = spectrogram
        if output_cross_attentions:
            cross_attentions = torch.cat(cross_attentions, dim=2)
            if bsz > 1:
                cross_attentions = cross_attentions.view(
                    bsz, int(cross_attentions.size(0) / bsz), *cross_attentions.size()[-3:]
                )
            outputs = (outputs, cross_attentions)
    else:
        # batched return values should also include the spectrogram/waveform lengths
        spectrogram_lengths = []
        for i in range(bsz):
            spectrogram_lengths.append(spectrograms[i].size(0))
        if vocoder is None:
            spectrograms = torch.nn.utils.rnn.pad_sequence(spectrograms, batch_first=True)
            outputs = (spectrograms, spectrogram_lengths)
        else:
            waveforms = []
            spectrograms = torch.nn.utils.rnn.pad_sequence(spectrograms, batch_first=True)
            waveforms = vocoder(spectrograms.to("cpu")).to("cpu")[:true_waveform_length]
            waveform_lengths = [int(waveforms.size(1) / max(spectrogram_lengths)) * i for i in spectrogram_lengths]
            outputs = (waveforms, waveform_lengths)
        if output_cross_attentions:
            cross_attentions = torch.cat(cross_attentions, dim=2)
            cross_attentions = cross_attentions.view(
                bsz, int(cross_attentions.size(0) / bsz), *cross_attentions.size()[-3:]
            )
            outputs = (*outputs, cross_attentions)
    return outputs


def initialize_tts():
    os.environ["TT_RUNTIME_ENABLE_PROGRAM_CACHE"] = "1"
    torch._dynamo.config.cache_size_limit = 1024
    xr.set_device_type("TT")

    # We disable consteval to avoid OOM errors as for every precompiled graph, the consteval graphs end up copying the model weights.
    # This is not a problem for the single test case, however for serving, we will likely want o have precompild graphs for numerous
    # sequence lengths.
    torch_xla.set_custom_compile_options(
        {
            "enable_const_eval": False,
        }
    )

    device = xm.xla_device()
    # device = "cpu"

    model = get_model().eval()
    model = model.to(torch.bfloat16).to(device)

    # Run prenet in f32 as bf16 causes accuracy issues
    model.speecht5.decoder.prenet.to(torch.float32)

    model.speecht5.encoder.eval()
    model.speecht5.encoder.compile(backend="tt")
    model.speecht5.decoder.prenet.eval()
    model.speecht5.decoder.prenet.compile(backend="tt")
    model.speecht5.decoder.wrapped_decoder.eval()
    model.speecht5.decoder.wrapped_decoder.compile(backend="tt")
    model.speech_decoder_postnet.eval()
    model.speech_decoder_postnet.compile(backend="tt")
    # The `postnet` function of `speech_decoder_postnet` a separate function from `forward`, so we must explicitly compile this function.
    model.speech_decoder_postnet.postnet = torch.compile(model.speech_decoder_postnet.postnet, backend="tt")

    # The `feat_out` function of `speech_decoder_postnet` is a separate function from `forward`, so we must explicitly compile this function.
    model.speech_decoder_postnet.feat_out = torch.compile(model.speech_decoder_postnet.feat_out, backend="tt")

    # The `prob_out` function of `speech_decoder_postnet` is a separate function from `forward`, so we must explicitly compile this function.
    model.speech_decoder_postnet.prob_out = torch.compile(model.speech_decoder_postnet.prob_out, backend="tt")

    vocoder = get_vocoder()

    speaker_embeddings = get_speaker_embeddings().to(device)

    return device, model, vocoder, speaker_embeddings

def run_on_tt():
    device, model, vocoder, speaker_embeddings = initialize_tts()

    inputs = get_input("My name is Quietbox. I am here to help.", device)

    speech = _generate_speech(model, **inputs, speaker_embeddings=speaker_embeddings, vocoder=vocoder)

    sf.write("speech.wav", speech.detach().cpu().numpy(), samplerate=16000)

def run_vocoder_tt():
    xr.set_device_type("TT")
    vocoder = get_vocoder()

    # Wrap with fixed version to avoid graph breaks

    # torch inference first
    spectrogram = torch.randn(1, 100, 80)
    '''with torch.no_grad():
        speech = vocoder(spectrogram)'''


    vocoder.compile(backend="tt")
    device = xm.xla_device()
    vocoder = vocoder.to(device)
    spectrogram = spectrogram.to(device)


    torch_xla.set_custom_compile_options(
        {
            "export_path": EXPORT_PATH,
            "backend": "codegen_py",
        }
    )

    speech = vocoder(spectrogram)
    print(speech)

def validate_encoder():
    xr.set_device_type("TT")
    model = get_model().speecht5.encoder
    model.eval()
    model_inputs = get_input()

    with torch.no_grad():
        torch_output = model(model_inputs["input_ids"])
    torch_output = torch_output.last_hidden_state

    device = xm.xla_device()
    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    model = model.to(device)

    output = model(model_inputs["input_ids"])
    output = output.last_hidden_state.cpu()
    print(output.shape)
    print(torch_output.shape)

# --------------------------------
# main
# --------------------------------
if __name__ == "__main__":
    run_on_tt()