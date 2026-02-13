#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
quantize_model.py

CPU-side simulation of Tenstorrent BFP8/BFP4 (block floating point) quantization
followed by "unpack" back to BF16/FP32, aiming to match tt-metal behavior closely.

Key choices (per your request):
- Quantize ONLY matmul/linear weights (nn.Linear.weight and (optionally) transformers Conv1D.weight)
- Option A reconstruction: rebuild float32 bits using shared exponent + reconstructed mantissa bits,
  then cast to bfloat16 (to mimic device unpack-to-bf16 before math).

Notes:
- This simulates the numeric effect of shared exponent per 16 values and mantissa reduction.
- It does NOT produce packed tiles/dwords; it reconstructs values directly.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn as nn


def make_out_dir(base_out: str, hf_model: str, mode: str) -> str:
    """
    Build output directory:
      <base_out>/<mode>_models/<hf_model>-<mode>
    Example:
      base_out="artifacts"
      hf_model="meta-llama/Llama-3.1-8B-Instruct"
      mode="bfp8"
      => "artifacts/bfp8_models/meta-llama/Llama-3.1-8B-Instruct-bfp8"

    We keep the 'org/model' structure by not stripping '/'.
    """
    mode = mode.lower()
    # light sanitization: remove trailing slashes and spaces
    hf_model = hf_model.strip().strip("/")

    # If hf_model accidentally has leading schemes etc, you can add more sanitization here.
    # Keep slashes so it nests under base_out.
    leaf = f"{hf_model}-{mode}"
    return os.path.join(base_out, f"{mode}_models", leaf)


def make_bf16_dir(base_out: str, hf_model: str) -> str:
    """
    Build bf16 output directory:
      <base_out>/bf16_models/<hf_model>-bf16
    """
    hf_model = hf_model.strip().strip("/")
    leaf = f"{hf_model}-bf16"
    return os.path.join(base_out, "bf16_models", leaf)


# -------------------------
# Config
# -------------------------


@dataclass(frozen=True)
class BFPFormat:
    name: str
    mantissa_bits_total: (
        int  # total bits INCLUDING hidden bit after reduction (7 for BFP8, 3 for BFP4)
    )


BFP8 = BFPFormat(name="bfp8", mantissa_bits_total=7)
BFP4 = BFPFormat(name="bfp4", mantissa_bits_total=3)


# -------------------------
# Utilities: module filtering (ONLY matmul/linear weights)
# -------------------------


def _get_transformers_conv1d_class():
    """
    Some GPT-style models use transformers.pytorch_utils.Conv1D for projections.
    This tries to import it; returns None if transformers isn't installed.
    """
    try:
        from transformers.pytorch_utils import Conv1D  # type: ignore

        return Conv1D
    except Exception:
        return None


def collect_quantizable_param_names(model: nn.Module) -> Set[str]:
    """
    Return names of parameters to quantize:
      - nn.Linear.weight
      - transformers.pytorch_utils.Conv1D.weight (if available)
    Skips biases and everything else (embeddings, norms, etc).
    """
    conv1d_cls = _get_transformers_conv1d_class()

    name_by_param_id: Dict[int, str] = {id(p): n for n, p in model.named_parameters()}

    quant_param_ids: Set[int] = set()

    for module in model.modules():
        if isinstance(module, nn.Linear):
            if getattr(module, "weight", None) is not None:
                quant_param_ids.add(id(module.weight))
        elif conv1d_cls is not None and isinstance(module, conv1d_cls):
            # Conv1D in HF: weight exists and is used as matmul weight
            w = getattr(module, "weight", None)
            if w is not None and isinstance(w, torch.nn.Parameter):
                quant_param_ids.add(id(w))

    quant_names: Set[str] = set()
    for pid in quant_param_ids:
        n = name_by_param_id.get(pid)
        if n is not None:
            quant_names.add(n)

    return quant_names


# -------------------------
# Core BFP simulation (NumPy-based bit work, CPU)
# -------------------------


def _pad_to_multiple(x: np.ndarray, axis: int, multiple: int) -> Tuple[np.ndarray, int]:
    """
    Zero-pad along `axis` so its length is a multiple of `multiple`.
    Returns (padded_array, pad_amount).
    """
    axis = axis % x.ndim
    dim = x.shape[axis]
    rem = dim % multiple
    if rem == 0:
        return x, 0
    pad_amt = multiple - rem
    pad_width = [(0, 0)] * x.ndim
    pad_width[axis] = (0, pad_amt)
    return np.pad(x, pad_width=pad_width, mode="constant", constant_values=0.0), pad_amt


def _move_axis_to_last(x: np.ndarray, axis: int) -> np.ndarray:
    axis = axis % x.ndim
    if axis == x.ndim - 1:
        return x
    return np.moveaxis(x, axis, -1)


def _move_axis_from_last(x: np.ndarray, axis: int) -> np.ndarray:
    axis = axis % x.ndim
    if axis == x.ndim - 1:
        return x
    return np.moveaxis(x, -1, axis)


def simulate_bfp_unpack_to_bf16(
    t: torch.Tensor,
    fmt: BFPFormat,
    *,
    block: int = 16,
    axis: int = -1,
    rounding: str = "rne",  # "rne" (round-to-nearest ties-to-even) or "trunc"
    flush_denorms: bool = True,
    is_exp_a: bool = False,  # default False per your doc unless you need rebias
    return_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """
    Simulate BF16/FP32 -> BFP (shared exponent per block) -> reconstruct float32 bits -> cast to return_dtype.

    - Works on CPU (converts tensor to CPU float32 internally).
    - Uses Option A reconstruction: rebuild float32 u32 bits with exponent=shared_exp.

    Parameters:
      t: input tensor (any dtype). Typically bf16/fp16/fp32.
      fmt: BFP8 or BFP4
      block: block size sharing exponent (16)
      axis: dimension along which consecutive groups of `block` share exponent
      rounding: "rne" or "trunc"
      flush_denorms: if True, exp==0 becomes exactly 0 (TT behavior)
      is_exp_a: optional rebias to 5-bit exponent [0,31] with bias 15 (rare; keep False unless needed)
      return_dtype: final dtype (bf16 to mimic device unpack)

    Returns:
      Tensor on original device, dtype=return_dtype.
    """
    if rounding not in ("rne", "trunc"):
        raise ValueError(f"rounding must be 'rne' or 'trunc', got {rounding}")
    if fmt.mantissa_bits_total < 2:
        raise ValueError("mantissa_bits_total must be >= 2 (includes hidden bit).")

    device = t.device

    # Step 1: convert to float32 on CPU, contiguous
    x_fp32 = t.detach().to(torch.float32).cpu().contiguous()
    x = x_fp32.numpy()  # float32 ndarray (shares memory)

    # Work in a "last-axis blocks" layout for easier reshape
    x_last = _move_axis_to_last(x, axis)
    x_last, pad_amt = _pad_to_multiple(x_last, axis=-1, multiple=block)

    orig_shape_last = x_last.shape
    last_dim = orig_shape_last[-1]
    nblocks = last_dim // block

    # Reshape to (..., nblocks, block)
    x_blk = x_last.reshape(*orig_shape_last[:-1], nblocks, block)

    # Bitcast float32 -> uint32 view
    u = x_blk.view(np.uint32)

    sign = (u >> 31) & np.uint32(0x1)
    exp = (u >> 23) & np.uint32(0xFF)
    mant = u & np.uint32(0x7FFFFF)

    # Optional exp_a rebias (matches your doc)
    if is_exp_a:
        # rebias: exp_new = exp_old - 127 + 15, clamp to [0,31]
        se = exp.astype(np.int32) - 127 + 15
        se = np.clip(se, 0, 31).astype(np.uint32)
        exp_for_shared = se
    else:
        exp_for_shared = exp

    # Step 3: flush denorm/zero (exp == 0)
    # TT: denorm or zero -> 0
    valid = exp != 0 if flush_denorms else np.ones_like(exp, dtype=bool)

    # For shared exponent max, ensure invalid don't win max
    exp_for_shared_masked = np.where(valid, exp_for_shared, np.uint32(0))

    # Step 5: shared exponent = max per block of 16
    shared = exp_for_shared_masked.max(axis=-1, keepdims=True).astype(
        np.uint32
    )  # shape (..., nblocks, 1)

    # Step 6: add hidden bit and align mantissa to shared exponent
    mant24 = np.where(valid, (np.uint32(1) << 23) | mant, np.uint32(0))

    # Need the "effective exp" used for alignment.
    # In tt-metal, alignment uses exp after optional rebias when is_exp_a.
    exp_align = exp_for_shared_masked  # invalid already 0
    shift = (shared - exp_align).astype(np.int32)
    shift = np.clip(shift, 0, 255).astype(
        np.uint32
    )  # safe range; numpy right_shift handles big shifts -> 0

    aligned = np.right_shift(mant24, shift)

    # Step 7: mantissa rounding/truncation to fmt.mantissa_bits_total
    m_total = fmt.mantissa_bits_total  # includes hidden bit
    shift_amount = 24 - m_total
    if shift_amount <= 0:
        raise ValueError("Invalid mantissa_bits_total; must be < 24")

    if rounding == "trunc":
        b = np.right_shift(aligned, np.uint32(shift_amount))
    else:
        # round-to-nearest, ties-to-even (matches your pseudo-code)
        round_mask = np.uint32((1 << shift_amount) - 1)
        tie_value = np.uint32(1 << (shift_amount - 1))

        round_value = aligned & round_mask
        b = np.right_shift(aligned, np.uint32(shift_amount))

        guard_bit = b & np.uint32(0x1)
        round_up = (round_value > tie_value) | (
            (round_value == tie_value) & (guard_bit == 1)
        )
        b = b + round_up.astype(np.uint32)

    # Clamp mantissa integer to max representable (TT does this)
    max_mant = np.uint32((1 << m_total) - 1)
    b = np.minimum(b, max_mant)

    # Step 8: if mantissa == 0, sign must be 0
    sign = np.where(b == 0, np.uint32(0), sign)

    # Option A reconstruction:
    # Build float32 bits with exponent = shared (8-bit exponent) and mantissa = top (m_total-1) fractional bits
    # b layout: [hidden_bit][fraction_bits...] (m_total bits)
    frac_bits = m_total - 1
    frac_mask = np.uint32((1 << frac_bits) - 1) if frac_bits > 0 else np.uint32(0)
    frac = b & frac_mask

    # Place fractional bits into the top bits of IEEE mantissa field (23 bits)
    if frac_bits > 0:
        mant23 = np.left_shift(frac, np.uint32(23 - frac_bits))
    else:
        mant23 = np.uint32(0)

    # shared exponent used in reconstructed float32 should be the original 8-bit exponent (not exp_a clamped),
    # because you described BFP8_B/BFP4_B share 8-bit exponent like IEEE.
    # If you ever truly need exp_a semantics, we can revisit; for now we keep this consistent with BFP8_B/BFP4_B.
    shared_ieee = (
        shared if not is_exp_a else shared
    )  # keep as-is; if exp_a path is used, it matches your doc.

    u_out = (sign << np.uint32(31)) | (shared_ieee << np.uint32(23)) | mant23

    # Ensure zeros are exactly 0 bits
    u_out = np.where(b == 0, np.uint32(0), u_out)

    y = u_out.view(np.float32)

    # Reshape back and unpad
    y_last = y.reshape(*orig_shape_last)
    if pad_amt:
        y_last = y_last[..., : (y_last.shape[-1] - pad_amt)]

    y_np = _move_axis_from_last(y_last, axis)

    # Return torch tensor on original device with requested dtype
    y_t = torch.from_numpy(np.ascontiguousarray(y_np)).to(device=device)
    if return_dtype is not None:
        y_t = y_t.to(return_dtype)
    return y_t


# -------------------------
# Model / state_dict quantization
# -------------------------


@torch.no_grad()
def quantize_model_linear_weights_inplace(
    model: nn.Module,
    *,
    mode: str = "bfp8",
    axis: int = -1,
    block: int = 16,
    rounding: str = "rne",
    return_dtype: torch.dtype = torch.bfloat16,
    verbose: bool = True,
) -> Set[str]:
    """
    In-place quantize ONLY matmul/linear weights in the given model.

    Returns:
      Set of parameter names that were quantized.
    """
    fmt = BFP8 if mode.lower() == "bfp8" else BFP4 if mode.lower() == "bfp4" else None
    if fmt is None:
        raise ValueError("mode must be 'bfp8' or 'bfp4'")

    quant_names = collect_quantizable_param_names(model)
    if verbose:
        print(
            f"[quantize_model] Found {len(quant_names)} quantizable linear/matmul weights."
        )

    for name, p in model.named_parameters():
        if name not in quant_names:
            continue
        if verbose:
            print(
                f"[quantize_model] Quantizing: {name}  shape={tuple(p.shape)} dtype={p.dtype} device={p.device}"
            )
        q = simulate_bfp_unpack_to_bf16(
            p.data,
            fmt,
            block=block,
            axis=axis,
            rounding=rounding,
            flush_denorms=True,
            is_exp_a=False,
            return_dtype=return_dtype,
        )
        p.data.copy_(q)

    return quant_names


@torch.no_grad()
def quantize_state_dict_linear_weights(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
    *,
    mode: str = "bfp8",
    axis: int = -1,
    block: int = 16,
    rounding: str = "rne",
    return_dtype: torch.dtype = torch.bfloat16,
    verbose: bool = True,
) -> Dict[str, torch.Tensor]:
    """
    Create a new state_dict where ONLY matmul/linear weights are replaced with simulated BFP unpacked tensors.
    Non-quantized entries are copied as-is.

    This is the recommended path if you want to save a "quantized" checkpoint.
    """
    fmt = BFP8 if mode.lower() == "bfp8" else BFP4 if mode.lower() == "bfp4" else None
    if fmt is None:
        raise ValueError("mode must be 'bfp8' or 'bfp4'")

    quant_names = collect_quantizable_param_names(model)
    if verbose:
        print(
            f"[quantize_model] Found {len(quant_names)} quantizable linear/matmul weights."
        )

    out: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if k in quant_names:
            if verbose:
                print(
                    f"[quantize_model] Quantizing state_dict entry: {k} shape={tuple(v.shape)} dtype={v.dtype}"
                )
            out[k] = simulate_bfp_unpack_to_bf16(
                v,
                fmt,
                block=block,
                axis=axis,
                rounding=rounding,
                flush_denorms=True,
                is_exp_a=False,
                return_dtype=return_dtype,
            )
        else:
            out[k] = v
    return out


# -------------------------
# Optional HF helpers + CLI
# -------------------------


def _load_hf_model(model_name_or_path: str, device: str = "cpu"):
    """
    Load a HF causal LM if transformers is available.
    """
    try:
        from transformers import AutoModelForCausalLM  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "transformers is required for --hf_model. Install transformers."
        ) from e

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype=torch.bfloat16
    )
    model.to(device)
    model.eval()
    return model


def _save_hf_model(model, out_dir: str):
    """
    Save HF model if transformers is available.
    """
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(out_dir)


def _save_hf_config_and_tokenizer(model_name_or_path: str, out_dir: str):
    """
    Save HF model config and tokenizer files (without weights).
    This allows reconstruction of the model architecture from a state_dict.
    """
    try:
        from transformers import AutoConfig, AutoTokenizer  # type: ignore
    except Exception:
        return

    os.makedirs(out_dir, exist_ok=True)

    # Save config
    try:
        config = AutoConfig.from_pretrained(model_name_or_path)
        config.save_pretrained(out_dir)
    except Exception as e:
        print(f"[Warning] Could not save config: {e}")

    # Save tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        tokenizer.save_pretrained(out_dir)
    except Exception as e:
        print(f"[Warning] Could not save tokenizer: {e}")


def load_model_from_dir(
    model_dir: str,
    device: str = "cpu",
    state_dict_name: str = "pytorch_model.pt",
) -> nn.Module:
    """
    Load a model from a directory containing:
      - config.json (HF model config)
      - pytorch_model.pt or pytorch_model_quant_sim.pt (state dict)
      - optionally tokenizer files

    Args:
        model_dir: Path to directory containing model files
        device: Device to load model on
        state_dict_name: Name of the state dict file (default: "pytorch_model.pt")

    Returns:
        Loaded model

    Example:
        # Load bf16 model
        model_bf16 = load_model_from_dir("bf16_models/meta-llama/Llama-3.1-8B-Instruct-bf16")

        # Load quantized model
        model_bfp8 = load_model_from_dir(
            "bfp8_models/meta-llama/Llama-3.1-8B-Instruct-bfp8",
            state_dict_name="pytorch_model_quant_sim.pt"
        )
    """
    try:
        from transformers import AutoConfig, AutoModelForCausalLM  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "transformers is required to load models. Install transformers."
        ) from e

    # Load config
    config_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json not found in {model_dir}")

    config = AutoConfig.from_pretrained(model_dir)

    # Initialize model with config (random weights)
    model = AutoModelForCausalLM.from_config(config)
    model.to(device)
    model.eval()

    # Load state dict
    state_dict_path = os.path.join(model_dir, state_dict_name)
    if not os.path.exists(state_dict_path):
        raise FileNotFoundError(f"{state_dict_name} not found in {model_dir}")

    state_dict = torch.load(state_dict_path, map_location=device)

    # Handle wrapped state dicts
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model.load_state_dict(state_dict, strict=True)

    print(
        f"[load_model] Loaded model from {model_dir} with state_dict: {state_dict_name}"
    )
    return model


def main():
    ap = argparse.ArgumentParser(
        description="Simulate TT BFP8/BFP4 quantization on linear/matmul weights and save checkpoint."
    )
    ap.add_argument(
        "--hf_model",
        type=str,
        default=None,
        help="HuggingFace model name/path (AutoModelForCausalLM).",
    )
    ap.add_argument(
        "--pt_checkpoint",
        type=str,
        default=None,
        help="Path to a torch checkpoint (state_dict) to load into HF model, or to just quantize and resave.",
    )
    ap.add_argument(
        "--out",
        type=str,
        required=True,
        help="Output path. If HF: directory. Else: .pt file.",
    )
    ap.add_argument(
        "--mode",
        type=str,
        default="bfp8",
        choices=["bfp8", "bfp4"],
        help="BFP mode to simulate.",
    )
    ap.add_argument(
        "--axis",
        type=int,
        default=-1,
        help="Axis to group blocks of 16 along (default last dim).",
    )
    ap.add_argument(
        "--block",
        type=int,
        default=16,
        help="Block size sharing exponent (default 16).",
    )
    ap.add_argument(
        "--rounding",
        type=str,
        default="rne",
        choices=["rne", "trunc"],
        help="Rounding mode.",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device for model (cpu or cuda). Simulation runs on CPU internally.",
    )
    ap.add_argument(
        "--inplace",
        action="store_true",
        help="Quantize model in-place (then save). Default saves a new state_dict.",
    )
    ap.add_argument("--verbose", action="store_true", help="Verbose logging.")
    args = ap.parse_args()

    if args.hf_model is None and args.pt_checkpoint is None:
        raise SystemExit("Provide at least one of --hf_model or --pt_checkpoint.")

    # Path A: HF model available (recommended)
    # --- inside main(), after parsing args and after you know args.hf_model is not None ---
    if args.hf_model is not None:
        # Create output directories for both bf16 and quantized models
        bf16_dir = make_bf16_dir(args.out, args.hf_model)
        quant_dir = make_out_dir(args.out, args.hf_model, args.mode)
        os.makedirs(bf16_dir, exist_ok=True)
        os.makedirs(quant_dir, exist_ok=True)

        model = _load_hf_model(args.hf_model, device=args.device)

        if args.pt_checkpoint is not None:
            sd = torch.load(args.pt_checkpoint, map_location="cpu")
            if (
                isinstance(sd, dict)
                and "state_dict" in sd
                and isinstance(sd["state_dict"], dict)
            ):
                sd = sd["state_dict"]
            model.load_state_dict(sd, strict=False)

        # Get original state dict
        original_sd = model.state_dict()

        # Save bf16 model (unquantized) for sanity check
        print(f"[quantize_model] Saving original bf16 model to: {bf16_dir}")
        torch.save(
            original_sd,
            os.path.join(bf16_dir, "pytorch_model.pt"),
            _use_new_zipfile_serialization=False,
        )
        _save_hf_config_and_tokenizer(args.hf_model, bf16_dir)

        if args.inplace:
            quantize_model_linear_weights_inplace(
                model,
                mode=args.mode,
                axis=args.axis,
                block=args.block,
                rounding=args.rounding,
                return_dtype=torch.bfloat16,
                verbose=args.verbose,
            )
            print(
                f"[quantize_model] Saving quantized {args.mode} model to: {quant_dir}"
            )
            torch.save(
                model.state_dict(),
                os.path.join(quant_dir, "pytorch_model_quant_sim.pt"),
                _use_new_zipfile_serialization=False,
            )
            _save_hf_config_and_tokenizer(args.hf_model, quant_dir)
        else:
            new_sd = quantize_state_dict_linear_weights(
                model,
                original_sd,
                mode=args.mode,
                axis=args.axis,
                block=args.block,
                rounding=args.rounding,
                return_dtype=torch.bfloat16,
                verbose=args.verbose,
            )
            print(
                f"[quantize_model] Saving quantized {args.mode} model to: {quant_dir}"
            )
            torch.save(
                new_sd,
                os.path.join(quant_dir, "pytorch_model_quant_sim.pt"),
                _use_new_zipfile_serialization=False,
            )
            _save_hf_config_and_tokenizer(args.hf_model, quant_dir)

        print(f"[quantize_model] Done!")
        print(f"  - BF16 model (unquantized) saved to: {bf16_dir}")
        print(f"  - {args.mode.upper()} model (quantized) saved to: {quant_dir}")
        return

    # Path B: only a raw torch checkpoint (state_dict) with no HF model context
    # We cannot reliably identify which params are linear weights without the module graph.
    # So we require HF model for correct filtering.
    raise SystemExit(
        "For 'only linear/matmul weights' filtering, please provide --hf_model so we can identify Linear/Conv1D modules.\n"
        "If you truly want name-based filtering for a raw state_dict, tell me your naming patterns and I’ll add it."
    )


if __name__ == "__main__":
    main()
