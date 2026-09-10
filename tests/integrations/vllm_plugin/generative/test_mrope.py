# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import base64
import io

import pytest
import vllm
from conftest import assert_output_coherent, check_host_memory
from PIL import Image, ImageDraw


@pytest.mark.push
@pytest.mark.single_device
def test_mrope():
    prompts = [
        "Continue in English: I like taking walks in the",
    ]
    sampling_params = vllm.SamplingParams(temperature=0.8, top_p=0.95, max_tokens=32)
    model_name = "Qwen/Qwen2-VL-2B-Instruct"

    llm_args = {
        "model": model_name,
        "max_num_batched_tokens": 32,
        "max_num_seqs": 1,
        "max_model_len": 32,
        "gpu_memory_utilization": 0.002,
        "enforce_eager": True,
        "limit_mm_per_prompt": {"image": 0, "video": 0, "audio": 0},
        "additional_config": {
            "min_context_len": 32,
        },
    }
    llm = vllm.LLM(**llm_args)

    output_text = llm.generate(prompts, sampling_params)[0].outputs[0].text
    print(f"prompt: {prompts[0]}, output: {output_text}")
    assert_output_coherent(output_text)

    check_host_memory(model_name)


@pytest.mark.nightly
@pytest.mark.single_device
def test_mrope_qwen2vl_multimodal():
    model_name = "Qwen/Qwen2-VL-2B-Instruct"

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
                    {
                        "type": "image",
                        "image": image_url,
                    },
                    {"type": "text", "text": "Describe this image in one sentence."},
                ],
            }
        ]
    ]
    sampling_params = vllm.SamplingParams(temperature=0.0, top_p=1.0, max_tokens=32)
    llm_args = {
        "model": model_name,
        "max_num_batched_tokens": 16384,
        "max_num_seqs": 1,
        "max_model_len": 512,
        "gpu_memory_utilization": 0.002,
        "limit_mm_per_prompt": {"image": 1, "video": 0, "audio": 0},
        "additional_config": {
            "min_context_len": 512,
            "enable_tensor_parallel": True,
            "cpu_sampling": True,
        },
    }
    llm = vllm.LLM(**llm_args)

    output_text = llm.chat(messages, sampling_params)[0].outputs[0].text
    print(f"output: {output_text}")
    assert_output_coherent(output_text)

    check_host_memory(model_name)
