# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import base64
import io

import pytest
import vllm
from conftest import assert_output_coherent, check_host_memory
from PIL import Image, ImageDraw


@pytest.mark.nightly
@pytest.mark.bhqb
def test_generation_single_device_gemma4_e4b():
    model_name = "google/gemma-4-E4B-it"
    messages = [[{"role": "user", "content": "Describe Tenstorrent in one sentence."}]]
    sampling_params = vllm.SamplingParams(temperature=0.0, top_p=1.0, max_tokens=32)
    llm_args = {
        "model": model_name,
        # Text-only path on a multimodal model: zero every modality so the
        # mm-encoder graph doesn't compile the vision tower at all.
        "limit_mm_per_prompt": {"image": 0, "video": 0, "audio": 0},
        # Gemma-4 mm enforces a floor from MultiModalBudget regardless of
        # limit_mm_per_prompt; 2560 clears the video-frame floor of 2496.
        "max_num_batched_tokens": 2560,
        "max_num_seqs": 1,
        "max_model_len": 128,
        "gpu_memory_utilization": 0.002,
        "additional_config": {
            "enable_const_eval": True,
            "min_context_len": 32,
            "enable_tensor_parallel": False,
            "use_2d_mesh": False,
            "cpu_sampling": False,
            "flat_model_io": True,
        },
    }
    llm = vllm.LLM(**llm_args)

    output_text = llm.chat(messages, sampling_params)[0].outputs[0].text
    print(f"output: {output_text}")
    assert_output_coherent(output_text)

    check_host_memory(model_name)


@pytest.mark.nightly
@pytest.mark.bhqb
def test_generation_single_device_gemma4_e4b_image():
    model_name = "google/gemma-4-E4B-it"

    # Synthetic image: a red square on a blue background.
    image = Image.new("RGB", (256, 256), (20, 80, 200))
    ImageDraw.Draw(image).rectangle([64, 64, 192, 192], fill=(220, 40, 40))
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    image_url = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"

    messages = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_url}},
                    {"type": "text", "text": "Describe this image in one sentence."},
                ],
            }
        ]
    ]
    sampling_params = vllm.SamplingParams(temperature=0.0, top_p=1.0, max_tokens=32)
    llm_args = {
        "model": model_name,
        "limit_mm_per_prompt": {"image": 1, "video": 0, "audio": 0},
        # KV-sharing layers gather K/V from the cache (sized by max_model_len),
        # and the prefill bucket rounds max_model_len up to a power of two; keep
        # max_model_len a power of two >= the image token count so q == kv.
        "max_num_batched_tokens": 512,
        "max_num_seqs": 1,
        "max_model_len": 512,
        "gpu_memory_utilization": 0.002,
        "additional_config": {
            "enable_const_eval": True,
            "min_context_len": 32,
            "enable_tensor_parallel": False,
            "use_2d_mesh": False,
            "cpu_sampling": True,
            "flat_model_io": True,
        },
    }
    llm = vllm.LLM(**llm_args)

    output_text = llm.chat(messages, sampling_params)[0].outputs[0].text
    print(f"output: {output_text}")
    assert_output_coherent(output_text)

    check_host_memory(model_name)
