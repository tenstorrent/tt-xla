# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Sensitivity score calculator for weight tensors.

S(T) = SUM_i [ Fii * (wi - Q(wi))^2 ]
Fii  = (1/D) * SUM_d [ g[d,i]^2 ]
"""

import argparse
import json
import time

import torch
import torch.nn as nn
import ttnn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

NUM_SAMPLES = 100
SEQ_LEN = 128


def parse_args():
    """Enable custom arguments for bfp4 quantization."""

    # TODO: Model llama 3.2-1b set for testing, remove later
    parser = argparse.ArgumentParser(
        description="Compute per-tensor BFP4 sensitivity scores."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--device-id", type=int, default=0, help="TT device ID (default: 0)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="sensitivity_scores.json",
        help="Output JSON file path",
    )
    return parser.parse_args()


def load_model_and_tokenizer(model_name):
    """Load model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model.eval()
    return model, tokenizer


def get_calibration_data(tokenizer):
    """Load calibration data from C4 dataset."""
    dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    samples = []
    for sample in dataset:
        tokens = tokenizer(
            sample["text"], return_tensors="pt", truncation=True, max_length=SEQ_LEN + 1
        )["input_ids"][0]
        if len(tokens) >= SEQ_LEN + 1:
            samples.append(tokens[: SEQ_LEN + 1])
        if len(samples) >= NUM_SAMPLES:
            break
    return samples


def collect_weights(model):
    """Return list of (name, param) for all weight tensors."""
    weights = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            weights.append((f"{name}.weight", module.weight))
    return weights


def compute_fisher(model, weight_params, calibration_data):
    """Accumulate squared gradients over calibration samples to approximate Fii."""
    for param in model.parameters():
        param.requires_grad_(False)
    for _, param in weight_params:
        param.requires_grad_(True)

    accumulators = {name: torch.zeros_like(param.data) for name, param in weight_params}

    for i, input_ids in enumerate(calibration_data):
        print(f"Fisher computation: sample {i + 1}/{len(calibration_data)}")
        input_ids = input_ids.unsqueeze(0)
        labels = input_ids[:, 1:].contiguous()
        inputs = input_ids[:, :-1].contiguous()
        model.zero_grad()

        logits = model(inputs).logits

        # Calculate Cross Entropy Loss
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )

        loss.backward()

        # TODO: Use GPU for larger models after testing
        # Accumulate squared gradients
        for name, param in weight_params:
            if param.grad is not None:
                accumulators[name] += param.grad.detach() ** 2

    fii = {name: acc / NUM_SAMPLES for name, acc in accumulators.items()}
    return fii


def compute_quant_error(weight_params, tt_device):
    """Compute (wi - Q(wi))^2 for each weight tensor via BFP4 quantization."""
    quant_errors = {}
    for name, param in weight_params:
        w = param.data.float()
        q = quantize_via_ttnn(w, ttnn.bfloat4_b, tt_device)
        quant_errors[name] = (w - q.float()) ** 2
    return quant_errors


def quantize_via_ttnn(original_torch, dtype, device):
    """Send tensor to device, typecast to dtype, return as torch tensor."""
    tt_tensor = ttnn.from_torch(
        original_torch, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    # Typecast to target (BFP4)
    tt_quantized = ttnn.typecast(tt_tensor, dtype)
    # Return back to bfloat16
    tt_dequantized = ttnn.typecast(tt_quantized, ttnn.bfloat16)

    result = ttnn.to_torch(tt_dequantized)
    result = result[: original_torch.shape[0], : original_torch.shape[1]]
    return result


def compute_sensitivity_scores(fii, quant_errors):
    """Compute sensitivity scores for each weight tensor."""
    scores = {}
    for name in fii:
        scores[name] = (fii[name] * quant_errors[name]).sum().item()
    return scores


def main():
    args = parse_args()
    t_start = time.perf_counter()

    model, tokenizer = load_model_and_tokenizer(args.model)
    model = torch.compile(model)

    calibration_data = get_calibration_data(tokenizer)
    weight_params = collect_weights(model)

    print("Loaded data, starting Fisher computation...")
    t0 = time.perf_counter()
    fii = compute_fisher(model, weight_params, calibration_data)
    print(f"Fisher computation done in {time.perf_counter() - t0:.1f}s")

    print("Starting quantization error computation...")
    t0 = time.perf_counter()
    tt_device = ttnn.open_device(device_id=args.device_id)
    try:
        quant_errors = compute_quant_error(weight_params, tt_device)
    finally:
        ttnn.close_device(tt_device)
    print(f"Quantization error computation done in {time.perf_counter() - t0:.1f}s")

    print("Starting sensitivity score computation...")
    scores = compute_sensitivity_scores(fii, quant_errors)
    scores_sorted = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    for name, score in scores_sorted.items():
        print(f"{score:.6f}  {name}")

    print("Saving scores to JSON file...")

    with open(args.output, "w") as f:
        json.dump(scores_sorted, f, indent=2)

    print(f"\nSaved to {args.output}")
    print(f"Total time: {time.perf_counter() - t_start:.1f}s")


if __name__ == "__main__":
    main()
