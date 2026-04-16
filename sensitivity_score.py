# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Sensitivity score calculator for weight tensors.

  S(T) = SUM_i [ Fii * (wi - Q(wi))^2 ]
  Fii  = (1/D) * SUM_d [ g[d,i]^2 ]

Two-stage workflow — model name is the only input needed on both sides:

  Stage 1 (GPU machine):
    python sensitivity_score.py --stage fisher --model meta-llama/Llama-3.2-1B
    → mixed_precision_experiments/fisher/Llama-3.2-1B/fii.pt

  Transfer fii.pt to the TT machine:
    scp mixed_precision_experiments/fisher/Llama-3.2-1B/fii.pt <tt-machine>:<same relative path>

  Stage 2 (TT machine):
    python sensitivity_score.py --stage scores --model meta-llama/Llama-3.2-1B
    → mixed_precision_experiments/sensitivity_scores/Llama-3.2-1B/sensitivity_Llama-3.2-1B.json
"""

import argparse
import json
import os
import time

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    import ttnn

    HAS_TTNN = True
except ImportError:
    HAS_TTNN = False

SEQ_LEN = 128
NUM_SAMPLES = 100
EXPERIMENTS_DIR = "mixed_precision_experiments"


# Helpers


def get_fii_path(model_name):
    """Path where Fisher information is saved/loaded."""
    model_short = model_name.split("/")[-1]
    return os.path.join(EXPERIMENTS_DIR, "fisher", model_short, "fii.pt")


def get_scores_path(model_name, normalize=False):
    """Path where sensitivity scores JSON is saved."""
    model_short = model_name.split("/")[-1]
    suffix = "_norm" if normalize else ""
    return os.path.join(
        EXPERIMENTS_DIR,
        "sensitivity_scores",
        model_short,
        f"sensitivity_{model_short}{suffix}.json",
    )


def parse_args():
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
        "--stage",
        type=str,
        choices=["fisher", "scores"],
        required=True,
        help="fisher: compute and save Fii (GPU). scores: load Fii and compute scores (TT device).",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        default=False,
        help="Normalize sensitivity scores by number of weight elements",
    )
    return parser.parse_args()


# Model and data loading


def load_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    model.eval()
    return model, tokenizer


def get_calibration_data(tokenizer, num_samples):
    """Load num_samples sequences of length SEQ_LEN from the C4 dataset."""
    dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    samples = []
    for sample in dataset:
        tokens = tokenizer(
            sample["text"], return_tensors="pt", truncation=True, max_length=SEQ_LEN + 1
        )["input_ids"][0]
        if len(tokens) >= SEQ_LEN + 1:
            samples.append(tokens[: SEQ_LEN + 1])
        if len(samples) >= num_samples:
            break
    return samples


def collect_weights(model):
    """Return [(name, param)] for all Linear weight tensors."""
    return [
        (f"{name}.weight", module.weight)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    ]


# Fisher computation (GPU stage)


def compute_fisher(model, weight_params, calibration_data, device, num_samples):
    """Approximate diagonal Fisher Fii via accumulated squared gradients."""
    for param in model.parameters():
        param.requires_grad_(False)
    for _, param in weight_params:
        param.requires_grad_(True)

    accumulators = {
        name: torch.zeros_like(param.data, device="cpu")
        for name, param in weight_params
    }

    for i, input_ids in enumerate(calibration_data):
        print(f"  Fisher: sample {i + 1}/{len(calibration_data)}")
        input_ids = input_ids.unsqueeze(0).to(device)
        inputs = input_ids[:, :-1].contiguous()
        labels = input_ids[:, 1:].contiguous()

        model.zero_grad()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(inputs).logits
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1)
        )
        loss.backward()

        for name, param in weight_params:
            if param.grad is not None:
                accumulators[name] += param.grad.detach().cpu() ** 2

    # Return CPU tensors needed for scores stage
    for acc in accumulators.values():
        acc /= num_samples
    return accumulators


# Quantization error computation (TT device stage)


def quantize_via_ttnn(tensor, dtype, device):
    """Roundtrip tensor through TT device at target dtype to get quantized values."""
    tt = ttnn.from_torch(
        tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    tt = ttnn.typecast(tt, dtype)
    tt = ttnn.typecast(tt, ttnn.bfloat16)
    result = ttnn.to_torch(tt)
    return result[: tensor.shape[0], : tensor.shape[1]]


def compute_quant_error(weight_params, tt_device):
    """Compute quantization error for each weight tensor."""
    quant_errors = {}
    for name, param in weight_params:
        w = param.data.cpu().float()
        q = quantize_via_ttnn(w, ttnn.bfloat4_b, tt_device)
        quant_errors[name] = (w - q.float()) ** 2
    return quant_errors


# Sensitivity score computation (CPU)


def compute_sensitivity_scores(fii, quant_errors, normalize=False):
    """Compute S(T) = sum(Fii * quant_error) for each weight tensor."""
    scores = {}
    for name in fii:
        score = (fii[name] * quant_errors[name]).sum().item()
        if normalize:
            score /= fii[name].numel()
        scores[name] = score
    return scores


def run_fisher_stage(args):
    gpu_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {gpu_device}")

    model, tokenizer = load_model_and_tokenizer(args.model)
    model.to(gpu_device)
    model = torch.compile(model)
    weight_params = collect_weights(model)
    calibration_data = get_calibration_data(tokenizer, NUM_SAMPLES)

    print(f"Computing Fisher over {NUM_SAMPLES} calibration samples...")
    t0 = time.perf_counter()
    fii = compute_fisher(
        model, weight_params, calibration_data, gpu_device, NUM_SAMPLES
    )
    del model

    fii_path = get_fii_path(args.model)
    os.makedirs(os.path.dirname(fii_path), exist_ok=True)
    torch.save(fii, fii_path)
    print(f"\nSaved: {fii_path}")


def run_scores_stage(args):
    if not HAS_TTNN:
        raise RuntimeError(
            "ttnn is not installed — scores stage requires a TT machine."
        )

    fii_path = get_fii_path(args.model)
    print(f"Loading Fisher information from {fii_path}...")
    fii = torch.load(fii_path, map_location="cpu")
    fii = {k.removeprefix("_orig_mod."): v for k, v in fii.items()}

    model, _ = load_model_and_tokenizer(args.model)
    weight_params = collect_weights(model)

    print("Computing quantization errors on TT device...")
    tt_device = ttnn.open_device(device_id=0)
    try:
        quant_errors = compute_quant_error(weight_params, tt_device)
    finally:
        ttnn.close_device(tt_device)

    scores = compute_sensitivity_scores(fii, quant_errors, normalize=args.normalize)
    scores_sorted = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    for name, score in scores_sorted.items():
        print(f"  {score:.6e}  {name}")

    scores_path = get_scores_path(args.model, normalize=args.normalize)

    os.makedirs(os.path.dirname(scores_path), exist_ok=True)
    with open(scores_path, "w") as f:
        json.dump(scores_sorted, f, indent=2)

    print(f"\nSaved: {scores_path}")


def main():
    args = parse_args()

    if args.stage == "fisher":
        run_fisher_stage(args)
    else:
        run_scores_stage(args)


if __name__ == "__main__":
    main()
