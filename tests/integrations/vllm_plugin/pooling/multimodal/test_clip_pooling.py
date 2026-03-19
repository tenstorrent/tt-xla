# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import numpy as np
import pytest
import vllm
from PIL import Image

MODEL = "openai/clip-vit-base-patch32"
# CLIP text encoder max sequence length
MAX_MODEL_LEN = 77


def cosine_similarity(a: list, b: list) -> float:
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def make_llm():
    llm_args = {
        "model": MODEL,
        "dtype": "bfloat16",
        "max_model_len": MAX_MODEL_LEN,
        "max_num_batched_tokens": MAX_MODEL_LEN,
        "max_num_seqs": 1,
        "additional_config": {
            "enable_const_eval": False,
            "min_context_len": 32,
        },
    }
    return vllm.LLM(**llm_args)


@pytest.mark.nightly
@pytest.mark.single_device
def test_clip_text_embedding():
    """Text inputs produce finite, non-zero embeddings of the expected size."""
    prompts = [
        "a photo of a cat",
        "a photo of a dog",
    ]
    llm = make_llm()
    outputs = llm.embed(prompts)

    assert len(outputs) == len(prompts)
    for prompt, output in zip(prompts, outputs):
        embeds = output.outputs.embedding
        # clip-vit-base-patch32 produces 512-dimensional embeddings
        assert len(embeds) == 512, f"Expected 512-dim embedding, got {len(embeds)}"
        assert all(
            np.isfinite(e) for e in embeds
        ), f"Embedding for {prompt!r} contains non-finite values"
        assert any(e != 0.0 for e in embeds), f"Embedding for {prompt!r} is all zeros"
        print(f"Prompt: {prompt!r}  embedding size={len(embeds)}")


@pytest.mark.nightly
@pytest.mark.single_device
def test_clip_image_embedding():
    """Image inputs produce finite, non-zero embeddings of the expected size."""
    images = [
        Image.new("RGB", (224, 224), color=(255, 0, 0)),  # red
        Image.new("RGB", (224, 224), color=(0, 0, 255)),  # blue
    ]
    llm = make_llm()

    inputs = [{"prompt": "", "multi_modal_data": {"image": img}} for img in images]
    outputs = llm.embed(inputs)

    assert len(outputs) == len(images)
    for idx, output in enumerate(outputs):
        embeds = output.outputs.embedding
        assert len(embeds) == 512, f"Expected 512-dim embedding, got {len(embeds)}"
        assert all(
            np.isfinite(e) for e in embeds
        ), f"Image {idx} embedding contains non-finite values"
        assert any(e != 0.0 for e in embeds), f"Image {idx} embedding is all zeros"
        print(f"Image {idx}  embedding size={len(embeds)}")


@pytest.mark.nightly
@pytest.mark.single_device
def test_clip_text_image_similarity():
    """Matching text-image pairs should have higher cosine similarity than
    mismatching pairs, which is CLIP's core retrieval property."""
    red_image = Image.new("RGB", (224, 224), color=(220, 30, 30))
    blue_image = Image.new("RGB", (224, 224), color=(30, 30, 220))

    llm = make_llm()

    text_outputs = llm.embed(["a red image", "a blue image"])
    image_outputs = llm.embed(
        [
            {"prompt": "", "multi_modal_data": {"image": red_image}},
            {"prompt": "", "multi_modal_data": {"image": blue_image}},
        ]
    )

    red_text_emb = text_outputs[0].outputs.embedding
    blue_text_emb = text_outputs[1].outputs.embedding
    red_img_emb = image_outputs[0].outputs.embedding
    blue_img_emb = image_outputs[1].outputs.embedding

    sim_red_match = cosine_similarity(red_text_emb, red_img_emb)
    sim_red_mismatch = cosine_similarity(red_text_emb, blue_img_emb)
    sim_blue_match = cosine_similarity(blue_text_emb, blue_img_emb)
    sim_blue_mismatch = cosine_similarity(blue_text_emb, red_img_emb)

    print(f"red text <-> red image  similarity: {sim_red_match:.4f}")
    print(f"red text <-> blue image similarity: {sim_red_mismatch:.4f}")
    print(f"blue text <-> blue image  similarity: {sim_blue_match:.4f}")
    print(f"blue text <-> red image similarity: {sim_blue_mismatch:.4f}")

    assert sim_red_match > sim_red_mismatch, (
        f"Expected red text to match red image better than blue image, "
        f"but got {sim_red_match:.4f} vs {sim_red_mismatch:.4f}"
    )
    assert sim_blue_match > sim_blue_mismatch, (
        f"Expected blue text to match blue image better than red image, "
        f"but got {sim_blue_match:.4f} vs {sim_blue_mismatch:.4f}"
    )
