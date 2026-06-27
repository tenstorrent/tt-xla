# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Per-tensor BFP4 sensitivity scores: S(T) = SUM_i [ Fii * (wi - Q(wi))^2 ]

Stage 1 — GPU machine (compute Fisher):
  python sensitivity_score.py --stage fisher --model <model>
  --accum-device cpu   keep fp32 accumulators in host RAM
  --offload-layers     stream weights from disk one layer at a time (large models)

Stage 2 — TT machine (compute scores):
  python sensitivity_score.py --stage scores --model <model>
  Auto-detects fii.pt or per-layer chunk_*.pt directory from stage 1.
"""

import argparse
import json
import os
import threading

import torch
import torch.nn as nn

try:
    import ttnn

    HAS_TTNN = True
except ImportError:
    HAS_TTNN = False

from offload_fisher import (
    NUM_SAMPLES,
    _make_fisher_hook,
    fisher_thread_worker,
    load_model_shell,
)

SEQ_LEN = 128
EXPERIMENTS_DIR = "mixed_precision_experiments"


def collect_weights(model):
    """Return [(name, param)] for all quantizable weight tensors."""
    return [
        (f"{name}.{pname}" if name else pname, param)
        for name, module in model.named_modules()
        for pname, param in module.named_parameters(recurse=False)
        if param.ndim >= 2
        and not isinstance(module, nn.Embedding)
        and "norm" not in name
        and "router" not in name
        and pname != "bias"
    ]


def quantize_via_ttnn(tensor, dtype, device):
    """Roundtrip tensor through TT device at target dtype to get quantized values."""
    orig_shape = tensor.shape
    if tensor.ndim > 2:
        tensor = tensor.reshape(-1, tensor.shape[-1])
    tt = ttnn.from_torch(
        tensor, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device
    )
    tt = ttnn.typecast(tt, dtype)
    tt = ttnn.typecast(tt, ttnn.bfloat16)
    result = ttnn.to_torch(tt)
    result = result[: tensor.shape[0], : tensor.shape[1]]
    return result.reshape(orig_shape)


def iter_fisher(fii_source):
    """Yield (name, tensor) pairs from a Fisher output file or chunk directory."""
    if os.path.isfile(fii_source):
        data = torch.load(fii_source, map_location="cpu")
        yield from data.items()
    elif os.path.isdir(fii_source):
        chunk_files = [
            f
            for f in os.listdir(fii_source)
            if f.startswith("chunk_") and f.endswith(".pt")
        ]
        for fname in chunk_files:
            chunk = torch.load(os.path.join(fii_source, fname), map_location="cpu")
            yield from chunk.items()
    else:
        raise FileNotFoundError(
            f"Fisher info not found at {fii_source!r}. Run --stage fisher first."
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute per-tensor BFP4 sensitivity scores."
    )
    parser.add_argument(
        "--model",
        type=str,
        help="HuggingFace model name or local path",
    )
    parser.add_argument(
        "--stage",
        type=str,
        choices=["fisher", "scores"],
        required=True,
        help="'fisher': compute and save fisher scores (GPU). "
        "'scores': load fisher and compute sensitivity scores (TT device).",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=None,
        help="Override output path for fisher results",
    )
    parser.add_argument(
        "--accum-device",
        type=str,
        choices=["gpu", "cpu"],
        default="gpu",
        help="Where to keep the fp32 squared-gradient accumulators. "
        "'gpu' (default) avoids PCIe traffic but needs extra GPU memory; "
        "'cpu' falls back to host RAM when GPUs don't have enough space.",
    )
    parser.add_argument(
        "--offload-layers",
        action="store_true",
        default=False,
        help="Layer-wise CPU offloading for models too large to fit in GPU memory. "
        "Loads one transformer block at a time and iterates layer-wise."
        "Accumulators always live on CPU.",
    )
    return parser.parse_args()


def get_fii_path(model_name):
    model_short = model_name.split("/")[-1]
    return os.path.join(EXPERIMENTS_DIR, "fisher", model_short, "fii.pt")


def get_scores_path(model_name):
    model_short = model_name.split("/")[-1]
    return os.path.join(
        EXPERIMENTS_DIR,
        "sensitivity_scores",
        model_short,
        f"sensitivity_{model_short}.json",
    )


def load_model_and_tokenizer(model_name, device_map=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, device_map=device_map
    )
    model.eval()
    return model, tokenizer


def get_calibration_data(tokenizer, num_samples):
    from datasets import load_dataset

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


def compute_fisher(
    model, weight_params, calibration_data, device, num_samples, accum_device="gpu"
):
    """Approximate diagonal Fisher Fii via accumulated squared gradients.

    Squared gradients are reduced in place, each backward only needs g².
    Per parameter we keep one fp32 accumulator + one transient bf16 grad.

    Two placement modes:

      accum_device="gpu" (default): accumulator lives on the same device as
        each parameter. No PCIe traffic during the loop.

      accum_device="cpu": accumulator lives on host RAM; the squared grad is
        moved to CPU and added in place. Use when GPUs don't have enough space.
    """
    for param in model.parameters():
        param.requires_grad_(False)

    on_cpu = accum_device == "cpu"
    acc = {}
    handles = []
    for name, param in weight_params:
        param.requires_grad_(True)
        handles.append(
            param.register_post_accumulate_grad_hook(
                _make_fisher_hook(name, acc, on_cpu=on_cpu)
            )
        )

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

    for h in handles:
        h.remove()

    return {name: a.div_(num_samples).cpu() for name, a in acc.items()}


def compute_fisher_offloaded(
    model,
    weight_params,
    calibration_data,
    weight_map=None,
    model_path=None,
    out_dir=None,
):
    """Layer-wise Fisher with CPU offloading and multi-GPU sample parallelism.

    Each transformer layer is loaded to GPU once for the forward sweep and once for
    the backward sweep, total O(N_layers) disk reads independent of num_samples.

    Threads synchronize at a barrier after each layer's backward pass. Thread 0 reduces
    the per-GPU fp32 partials and writes a bf16 chunk to out_dir immediately. No large
    intermediate files are written, peak disk usage is the final output size.

    weight_map and model_path must come from load_model_shell. out_dir is the output
    directory where chunk_*.pt files are written.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("--offload-layers requires at least one CUDA GPU.")

    num_gpus = torch.cuda.device_count()
    if num_gpus < 1:
        raise RuntimeError("No CUDA GPUs available.")

    weight_names_set = {name for name, _ in weight_params}
    os.makedirs(out_dir, exist_ok=True)

    active_gpus = min(num_gpus, len(calibration_data))
    shared_partials = [None] * active_gpus
    barrier = threading.Barrier(active_gpus)

    chunk_size = max(1, len(calibration_data) // active_gpus)
    threads = []
    for gpu_id in range(active_gpus):
        start = gpu_id * chunk_size
        end = start + chunk_size if gpu_id < active_gpus - 1 else len(calibration_data)
        t = threading.Thread(
            target=fisher_thread_worker,
            args=(
                model,
                calibration_data[start:end],
                gpu_id,
                weight_names_set,
                weight_map,
                model_path,
                out_dir,
                shared_partials,
                barrier,
            ),
            daemon=True,
        )
        threads.append(t)

    for t in threads:
        t.start()
    for t in threads:
        t.join()


def _run_fisher_offload(args):
    print("Loading model shell for disk-based weight streaming...")
    model, tokenizer, weight_map, model_path = load_model_shell(args.model)
    weight_params = collect_weights(model)
    calibration_data = get_calibration_data(tokenizer, NUM_SAMPLES)
    fii_path = args.save_path if args.save_path else get_fii_path(args.model)
    out_dir = os.path.dirname(fii_path)
    print(
        f"Computing Fisher over {NUM_SAMPLES} samples (layer-wise offload, disk streaming)..."
    )
    compute_fisher_offloaded(
        model,
        weight_params,
        calibration_data,
        weight_map=weight_map,
        model_path=model_path,
        out_dir=out_dir,
    )
    del model
    print(f"\nSaved: {out_dir}/")


def _run_fisher_standard(args):
    device_map = "auto" if torch.cuda.is_available() else "cpu"

    model, tokenizer = load_model_and_tokenizer(args.model, device_map=device_map)
    model.gradient_checkpointing_enable()

    weight_params = collect_weights(model)
    calibration_data = get_calibration_data(tokenizer, NUM_SAMPLES)
    input_device = next(model.parameters()).device

    print(f"Computing Fisher over {NUM_SAMPLES} calibration samples...")
    fii = compute_fisher(
        model,
        weight_params,
        calibration_data,
        input_device,
        NUM_SAMPLES,
        accum_device=args.accum_device,
    )
    del model

    fii = {k: v.to(torch.bfloat16) for k, v in fii.items()}
    fii_path = args.save_path if args.save_path else get_fii_path(args.model)
    os.makedirs(os.path.dirname(fii_path), exist_ok=True)
    torch.save(fii, fii_path)
    print(f"\nSaved: {fii_path}")


def run_fisher_stage(args):
    if args.offload_layers:
        _run_fisher_offload(args)
    else:
        _run_fisher_standard(args)


def run_scores_stage(args):
    if not HAS_TTNN:
        raise RuntimeError("TTNN not found. Scores stage requires a TT device.")

    fii_path = get_fii_path(args.model)
    fii_dir = os.path.dirname(fii_path)

    has_chunks = os.path.isdir(fii_dir) and any(
        f.startswith("chunk_") and f.endswith(".pt") for f in os.listdir(fii_dir)
    )
    fii_source = fii_dir if has_chunks else fii_path

    print(f"Loading Fisher information from {fii_source}...")

    model, _ = load_model_and_tokenizer(args.model)
    weight_params_dict = {name: param for name, param in collect_weights(model)}

    print("Computing sensitivity scores on TT device...")
    tt_device = ttnn.open_device(device_id=0)
    scores = {}

    try:
        for name, fii_tensor in iter_fisher(fii_source):
            w = weight_params_dict[name].data.cpu().float()
            q = quantize_via_ttnn(w, ttnn.bfloat4_b, tt_device)
            quant_err = (w - q.float()) ** 2
            score = (fii_tensor.float() * quant_err).sum().item() / fii_tensor.numel()
            scores[name] = score
    finally:
        ttnn.close_device(tt_device)

    scores_sorted = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    scores_path = get_scores_path(args.model)
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
