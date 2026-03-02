# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""On-demand generation of .refpt reference files for LLM accuracy testing.

Loads a HuggingFace model on CPU, runs teacher-forced decoding on the
"Tale of Two Cities" text corpus, and saves reference tokens + top-5
predictions to a .refpt file.

Called automatically by init_accuracy_testing() when the .refpt file
is missing or was generated with different library versions.
"""

import bz2
import os
import tempfile
from pathlib import Path

import torch
import transformers
from llm_utils.decode_utils import (
    extract_topk,
    generate_and_benchmark,
    init_static_cache,
)
from loguru import logger
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# Directory containing reference_outputs/ (sibling of llm_utils/)
_REFERENCE_DIR = str(Path(__file__).resolve().parent.parent / "reference_outputs")
_CORPUS_FILE = os.path.join(_REFERENCE_DIR, "tale-of-two-cities.txt.bz2")


def generate_reference_outputs(total_length, output_file, model_name):
    """
    Generate reference outputs for accuracy testing using HuggingFace models.

    Args:
        total_length: Number of tokens to process from Tale of Two Cities.
                     WARNING: This value MUST match the max_cache_len (input_sequence_length)
                     used during accuracy testing. Context length mismatch causes accuracy
                     degradation even with teacher forcing.
        output_file: Path to save .refpt file
        model_name: HuggingFace model name (e.g., 'meta-llama/Llama-3.2-1B-Instruct')
    """
    device = torch.device("cpu")
    logger.info(f"Using device: {device} (CPU forced for reference generation)")

    # Load model and tokenizer from HuggingFace
    config = AutoConfig.from_pretrained(model_name)

    # Qwen only: add rope scaling to the config, for long context support.
    # https://huggingface.co/Qwen/Qwen2.5-7B-Instruct#processing-long-texts
    if "Qwen" in model_name:
        config.rope_scaling = {
            "factor": 4.0,
            "original_max_position_embeddings": 32768,
            "type": "yarn",
        }

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.bfloat16,
    )
    model.to(device)
    model.eval()

    # Load the book text
    if not os.path.exists(_CORPUS_FILE):
        raise FileNotFoundError(
            f"Tale of Two Cities text file not found: {_CORPUS_FILE}\n"
            f"Please ensure tale-of-two-cities.txt.bz2 exists in the reference_outputs directory."
        )

    logger.info(f"Loading text from {_CORPUS_FILE}")
    with bz2.open(_CORPUS_FILE, "rt", encoding="utf-8") as f:
        text = f.read()

    # Encode text to tokens
    encoded_tokens = tokenizer.encode(text, add_special_tokens=True)[:total_length]
    tokens_1d = torch.tensor(encoded_tokens, device=device)  # [total_length]

    logger.info(f"Processing {len(encoded_tokens)} tokens")
    logger.info(f"Model: {model_name}")
    logger.info(f"Output file: {output_file}")

    max_cache_len = total_length
    batch_size = 1
    split_point = total_length // 2

    static_cache = init_static_cache(
        config=config,
        batch_size=batch_size,
        max_cache_len=max_cache_len,
        device="cpu",
        dtype=torch.bfloat16,
    )

    # Construct input_args dict matching the format used by device benchmarks
    input_args = {
        "input_ids": tokens_1d[:split_point].unsqueeze(0),  # [1, split_point]
        "past_key_values": static_cache,
        "cache_position": torch.arange(0, split_point),
        "use_cache": True,
    }
    ground_truth_tokens = tokens_1d[split_point:]  # [decode_len]
    decode_len = len(ground_truth_tokens)

    logger.info(
        f"Generating reference topk with shared decode logic (split_point={split_point})"
    )
    output_logits, _ = generate_and_benchmark(
        model=model,
        input_args=input_args,
        device=device,
        max_tokens_to_generate=decode_len,
        read_logits_fn=lambda output: output.logits,
        ground_truth_tokens=ground_truth_tokens,
        verbose=True,
    )

    # Post-processing: extract top-k predictions from raw logits
    all_top1 = []
    all_top5 = []
    for logits in output_logits:
        top1, top5 = extract_topk(logits, k=5)
        all_top1.append(top1.squeeze(0).cpu())
        all_top5.append(top5.squeeze(0).cpu())

    top1_tokens = torch.cat(all_top1, dim=0)  # [total_length - 1]
    top5_tokens = torch.cat(all_top5, dim=0)  # [total_length - 1, 5]
    reference_tokens = tokens_1d.view(1, -1)  # [1, total_length]

    # Sanity check: predictions cover positions 0..total_length-2
    expected_predictions = total_length - 1
    if top1_tokens.shape[0] != expected_predictions:
        raise RuntimeError(
            f"Expected {expected_predictions} predictions, got {top1_tokens.shape[0]}"
        )

    data = {
        "top1_tokens": top1_tokens,
        "top5_tokens": top5_tokens,
        "reference_tokens": reference_tokens,
        "library_versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
    }

    # Atomic write: write to temp file then rename to avoid partial files
    # if parallel tests try to generate the same model concurrently
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(
        dir=os.path.dirname(output_file), suffix=".refpt.tmp"
    )
    os.close(tmp_fd)
    try:
        torch.save(data, tmp_path)
        os.replace(tmp_path, output_file)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise

    logger.info(f"Saved reference outputs to {output_file}")
    logger.info(
        f"Library versions: torch={torch.__version__}, transformers={transformers.__version__}"
    )
