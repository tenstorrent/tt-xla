# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import vllm
from conftest import check_host_memory


@pytest.mark.push
@pytest.mark.tensor_parallel
@pytest.mark.dual_chip
@pytest.mark.parametrize("model_name", ["meta-llama/Llama-3.2-3B"])
@pytest.mark.parametrize("use_2d_mesh", [True, False])
def test_tensor_parallel_generation_n300(model_name: str, use_2d_mesh: bool):
    prompts = [
        "I like taking walks in the",
    ]
    sampling_params = vllm.SamplingParams(temperature=0.8, top_p=0.95, max_tokens=32)
    llm_args = {
        "model": model_name,
        "max_num_batched_tokens": 32,
        "max_num_seqs": 1,
        "max_model_len": 32,
        "gpu_memory_utilization": 0.002,
        "additional_config": {
            "enable_const_eval": False,
            "min_context_len": 32,
            "enable_tensor_parallel": True,
            "use_2d_mesh": use_2d_mesh,
        },
    }
    llm = vllm.LLM(**llm_args)

    output_text = llm.generate(prompts, sampling_params)[0].outputs[0].text
    print(f"prompt: {prompts[0]}, output: {output_text}")


@pytest.mark.push
@pytest.mark.tensor_parallel
@pytest.mark.llmbox
@pytest.mark.parametrize(
    ["model_name", "enable_const_eval", "experimental_weight_dtype"],
    [
        pytest.param("Qwen/Qwen3-0.6B", False, ""),
    ],
)
@pytest.mark.parametrize("use_2d_mesh", [True, False])
def test_tensor_parallel_generation_llmbox_small(
    model_name: str,
    enable_const_eval: bool,
    experimental_weight_dtype: str,
    use_2d_mesh: bool,
):
    prompts = [
        "I like taking walks in the",
    ]
    sampling_params = vllm.SamplingParams(temperature=0.8, top_p=0.95, max_tokens=32)
    llm_args = {
        "model": model_name,
        "max_num_batched_tokens": 32,
        "max_num_seqs": 1,
        "max_model_len": 32,
        "gpu_memory_utilization": 0.002,
        "additional_config": {
            "enable_const_eval": enable_const_eval,
            "min_context_len": 32,
            "enable_tensor_parallel": True,
            "experimental_weight_dtype": experimental_weight_dtype,
            "use_2d_mesh": use_2d_mesh,
        },
    }
    llm = vllm.LLM(**llm_args)

    output_text = llm.generate(prompts, sampling_params)[0].outputs[0].text
    print(f"prompt: {prompts[0]}, output: {output_text}")

    check_host_memory(model_name)


@pytest.mark.nightly
@pytest.mark.tensor_parallel
@pytest.mark.llmbox
@pytest.mark.parametrize(
    [
        "model_name",
        "enable_const_eval",
        "experimental_weight_dtype",
        "use_2d_mesh",
        "cpu_sampling",
    ],
    [
        pytest.param("Qwen/Qwen3-32B", False, "", True, True),
        pytest.param("Qwen/Qwen2.5-32B", False, "", False, False),
        pytest.param("meta-llama/Llama-3.1-70B", True, "bfp_bf8", True, False),
    ],
)
def test_tensor_parallel_generation_llmbox_large(
    model_name: str,
    enable_const_eval: bool,
    experimental_weight_dtype: str,
    use_2d_mesh: bool,
    cpu_sampling: bool,
):
    prompts = [
        "I like taking walks in the",
    ]
    sampling_params = vllm.SamplingParams(temperature=0.8, top_p=0.95, max_tokens=32)
    llm_args = {
        "model": model_name,
        "max_num_batched_tokens": 32,
        "max_num_seqs": 1,
        "max_model_len": 32,
        "gpu_memory_utilization": 0.002,
        "additional_config": {
            "enable_const_eval": enable_const_eval,
            "min_context_len": 32,
            "enable_tensor_parallel": True,
            "experimental_weight_dtype": experimental_weight_dtype,
            "use_2d_mesh": use_2d_mesh,
            "cpu_sampling": cpu_sampling,
        },
    }
    llm = vllm.LLM(**llm_args)

    output_text = llm.generate(prompts, sampling_params)[0].outputs[0].text
    print(f"prompt: {prompts[0]}, output: {output_text}")

    check_host_memory(model_name)


@pytest.mark.nightly
@pytest.mark.tensor_parallel
@pytest.mark.bhqb
@pytest.mark.parametrize(
    ["enable_const_eval", "experimental_weight_dtype", "use_2d_mesh"],
    [
        pytest.param(True, "", False),
    ],
)
def test_tensor_parallel_generation_bhqb_multimodal_31b(
    enable_const_eval: bool,
    experimental_weight_dtype: str,
    use_2d_mesh: bool,
):
    import os
    from pathlib import Path

    from PIL import Image, ImageDraw

    model_name = "google/gemma-4-31B-it"

    # Image source order (same as e4b mm test):
    # 1. If `TT_TEST_IMAGE_PATH` is set AND the file exists, use that file.
    # 2. Otherwise, fall back to a synthetic red-square-on-blue image.
    default_image_path = (
        Path(__file__).resolve().parents[3]
        / "gemma4_logs"
        / "multimodal_test_image.png"
    )
    image_path = Path(os.environ.get("TT_TEST_IMAGE_PATH", default_image_path))
    image_path.parent.mkdir(parents=True, exist_ok=True)
    if not image_path.exists():
        image = Image.new("RGB", (256, 256), color=(20, 80, 200))
        ImageDraw.Draw(image).rectangle([64, 64, 192, 192], fill=(220, 40, 40))
        image.save(image_path)
        print(f"Generated synthetic test image at {image_path}")
    else:
        print(f"Using existing test image at {image_path}")
    image_for_vllm = Image.open(image_path)

    prompt_text = "Describe this image in detail."
    messages = [
        [
            {
                "role": "user",
                "content": [
                    {"image_pil": image_for_vllm},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
    ]
    sampling_params = vllm.SamplingParams(temperature=0.0, top_p=1.0, max_tokens=512)
    llm_args = {
        "model": model_name,
        # Image-only path: forcing video/audio limits to 0 makes vLLM's
        # `MultiModalBudget.get_modality_with_max_tokens()` resolve to
        # "image" (280 tokens) instead of "video" (2496 tokens, batched as
        # 32 frames). That keeps profile_run's dummy batch ~32x smaller and
        # fits the vision encoder graph in DRAM. Drop these caps once
        # video / audio are actually supported on TT.
        "limit_mm_per_prompt": {"image": 1, "video": 0, "audio": 0},
        # max_num_batched_tokens still has to clear vLLM's internal floor
        # but no longer needs to hold a full 32-frame video batch.
        "max_num_batched_tokens": 2560,
        "max_num_seqs": 1,
        "max_model_len": 1024,
        "gpu_memory_utilization": 0.1,
        "additional_config": {
            "enable_const_eval": enable_const_eval,
            "min_context_len": 32,
            "enable_tensor_parallel": True,
            "experimental_weight_dtype": experimental_weight_dtype,
            "use_2d_mesh": use_2d_mesh,
            "cpu_sampling": False,
        },
    }
    llm = vllm.LLM(**llm_args)

    output_text = llm.chat(messages, sampling_params)[0].outputs[0].text
    print(f"prompt: {prompt_text}, output: {output_text}")

    check_host_memory(model_name)


@pytest.mark.nightly
@pytest.mark.tensor_parallel
@pytest.mark.bhqb
def test_generation_single_device_multimodal_e4b():
    import os
    from pathlib import Path

    from PIL import Image, ImageDraw

    model_name = "google/gemma-4-E4B-it"
    # Image source order:
    # 1. If `TT_TEST_IMAGE_PATH` is set AND the file exists, use that file
    #    as-is (good for trying real photos).
    # 2. Otherwise, fall back to a synthetic red-square-on-blue image written
    #    to `gemma4_logs/multimodal_test_image.png` so the test stays
    #    network-free.
    default_image_path = (
        Path(__file__).resolve().parents[3]
        / "gemma4_logs"
        / "multimodal_test_image.png"
    )
    image_path = Path(os.environ.get("TT_TEST_IMAGE_PATH", default_image_path))
    image_path.parent.mkdir(parents=True, exist_ok=True)
    if not image_path.exists():
        image = Image.new("RGB", (256, 256), color=(20, 80, 200))
        ImageDraw.Draw(image).rectangle([64, 64, 192, 192], fill=(220, 40, 40))
        image.save(image_path)
        print(f"Generated synthetic test image at {image_path}")
    else:
        print(f"Using existing test image at {image_path}")

    # Re-open from disk so the test mirrors the real flow (file -> PIL -> vLLM).
    image_for_vllm = Image.open(image_path)

    prompt_text = "Describe this image in detail."
    # vLLM's chat content parser accepts a typeless dict with `image_pil` —
    # see MM_PARSER_MAP in vllm/entrypoints/chat_utils.py. The fallback path
    # (no `type` field) routes this directly to parse_image_pil() which
    # registers the image under the "image" modality.
    messages = [
        [
            {
                "role": "user",
                "content": [
                    {"image_pil": image_for_vllm},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]
    ]
    sampling_params = vllm.SamplingParams(temperature=0.0, top_p=1.0, max_tokens=512)
    llm_args = {
        "model": model_name,
        # Image-only path: forcing video/audio limits to 0 makes vLLM's
        # `MultiModalBudget.get_modality_with_max_tokens()` resolve to
        # "image" (280 tokens) instead of "video" (2496 tokens, batched as
        # 32 frames). That keeps profile_run's dummy batch ~32x smaller and
        # fits the vision encoder graph in DRAM. Drop these caps once
        # video / audio are actually supported on TT.
        "limit_mm_per_prompt": {"image": 1, "video": 0, "audio": 0},
        # max_num_batched_tokens still has to clear vLLM's internal floor
        # but no longer needs to hold a full 32-frame video batch.
        "max_num_batched_tokens": 2560,
        "max_num_seqs": 1,
        "max_model_len": 1024,
        "gpu_memory_utilization": 0.1,
        # BHQB 1D mesh (4, 1): all 4 chips on the "batch" axis. Embedding
        # / LM-head hidden-dim sharding distributes the PLE weight across
        # those chips (262144x10752 -> 262144x2688 per chip), keeping the
        # tt-mlir EmbeddingOpPadHiddenDim 12288 padding affordable.
        "additional_config": {
            "enable_const_eval": True,
            "min_context_len": 512,
            "enable_tensor_parallel": False,
            "use_2d_mesh": False,
            "cpu_sampling": False,
        },
    }
    llm = vllm.LLM(**llm_args)

    output_text = llm.chat(messages, sampling_params)[0].outputs[0].text
    print(f"prompt: {prompt_text}, output: {output_text}")

    check_host_memory(model_name)
