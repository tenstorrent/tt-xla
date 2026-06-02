# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

### Standalone xla.py reproducing tt-xla benchmark test_llms.py::test_lexora_lite_3b_v2.
### Resolved model: DeepMount00/Lexora-Lite-3B_v2 (HF) via AutoModelForCausalLM.
### No imports from tt-xla/third_party/tt_forge_models or tt-xla/tests/benchmark/*.
###
### NOTE: the lexora ModelLoader (third_party/tt_forge_models/lexora/causal_lm/pytorch/loader.py)
### is not present in the current tt_forge_models submodule checkout. LLM benchmark construction
### is fully standardized, so this script reproduces it from the public HF id directly:
###   - AutoModelForCausalLM.from_pretrained (causal_lm family) with torch_dtype at load time
###   - setup_model_and_tokenizer patches (benchmarks/llm_benchmark.py:73-79)
###   - construct_inputs() StaticCache path (benchmarks/llm_benchmark.py:85-169)
### The loader's optional get_weight_dtype_config_path() auto-discovery could not be consulted
### (loader absent); no per-tensor weight dtype overrides applied — matches the common case.

import os
from pathlib import Path

import torch
import torch_xla
import torch_xla.runtime as xr
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import StaticCache
from tt_torch import codegen_py

MODEL_ID = "DeepMount00/Lexora-Lite-3B_v2"
DATA_FORMAT = torch.bfloat16     # test_llms.py: DEFAULT_DATA_FORMAT = "bfloat16"
BATCH_SIZE = 32                  # test_llms.py: DEFAULT_BATCH_SIZE = 32
INPUT_SEQUENCE_LENGTH = 128      # test_llms.py: DEFAULT_INPUT_SEQUENCE_LENGTH = 128
DEFAULT_INPUT_PROMPT = (
    "Here is an exaustive list of the best practices for writing clean code:"
)

OUTPUT_DIR = str(Path(__file__).resolve().parent / "model")

# Compile options assembled from benchmarks/llm_benchmark.py:457-466.
# Per-test overrides (test_lexora_lite_3b_v2): optimization_level=2, trace_enabled=True.
# Benchmark-file DEFAULT_* (not overridden by the test):
#   experimental_weight_dtype="bfp_bf8"  (DEFAULT_EXPERIMENTAL_WEIGHT_DTYPE)
#   experimental_enable_permute_matmul_fusion=False  (DEFAULT_EXPERIMENTAL_ENABLE_PERMUTE_MATMUL_FUSION)
# Dropped runtime-instrumentation keys: export_path, export_model_name, ttnn_perf_metrics_*.
COMPILE_OPTIONS = {
    "optimization_level": 2,
    # "enable_trace": True,  # benchmark used trace_enabled=True
    "experimental_weight_dtype": "bfp_bf8",
    "experimental_enable_permute_matmul_fusion": False,
}


def load_pytorch_model():
    # causal_lm family -> AutoModelForCausalLM, HUGGING_FACE source: dtype passed at load time.
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DATA_FORMAT)

    # setup_model_and_tokenizer patches from benchmarks/llm_benchmark.py:73-79.
    if hasattr(model.config, "layer_types"):
        model.config.layer_types = ["full_attention"] * len(model.config.layer_types)
    if hasattr(model.config, "_experts_implementation"):
        model.config._experts_implementation = "dense"

    model.eval()
    return model


def load_input():
    """Replicates construct_inputs() from benchmarks/llm_benchmark.py:85-169 (StaticCache path)."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    prompts = [DEFAULT_INPUT_PROMPT] * BATCH_SIZE
    tokenized = tokenizer(
        prompts,
        return_tensors="pt",
        max_length=INPUT_SEQUENCE_LENGTH,
        truncation=True,
    )
    input_ids = tokenized["input_ids"]

    # StaticCache lives on CPU and is moved to the device explicitly later — see
    # https://github.com/tenstorrent/tt-xla/issues/1645 for why we don't construct
    # it directly on the device.
    model = load_pytorch_model()
    past_key_values = StaticCache(
        config=model.config,
        max_batch_size=BATCH_SIZE,
        max_cache_len=INPUT_SEQUENCE_LENGTH,
        device="cpu",
        dtype=DATA_FORMAT,
    )

    cache_position = torch.arange(0, input_ids.shape[1])

    return {
        "input_ids": input_ids,
        "past_key_values": past_key_values,
        "cache_position": cache_position,
        "use_cache": True,
    }


def _inputs_to_device(inputs, device):
    """Mirrors transfer_to_device() from benchmarks/llm_benchmark.py:179-204 for the
    StaticCache path (no MLA layers in this model)."""
    out = dict(inputs)
    out["input_ids"] = out["input_ids"].to(device)
    out["cache_position"] = out["cache_position"].to(device)
    for layer in out["past_key_values"].layers:
        layer.keys = layer.keys.to(device)
        layer.values = layer.values.to(device)
    return out


def run_pytorch_model():
    model = load_pytorch_model()
    inputs = load_input()

    with torch.no_grad():
        output = model(**inputs)

    return output.logits


def run_tt_model():
    device = torch_xla.device()

    torch_xla.set_custom_compile_options(COMPILE_OPTIONS)

    model = load_pytorch_model()
    model.compile(backend="tt", options={"tt_legacy_compile": True})
    model = model.to(device)
    inputs = _inputs_to_device(load_input(), device)

    with torch.no_grad():
        output = model(**inputs)

    return output.logits.cpu()


def codegen_model():
    os.environ["XLA_HLO_DEBUG"] = "1"

    model = load_pytorch_model()
    inputs = load_input()

    codegen_py(
        model,
        inputs,
        export_path=OUTPUT_DIR,
        export_tensors=True,
        compiler_options=COMPILE_OPTIONS,
    )


def compare_pytorch_and_tt_runs():
    # Capture exact PCC from first --golden run and paste here.
    exact_pcc = None

    pt_output = run_pytorch_model()
    tt_output = run_tt_model()

    assert pt_output.shape == tt_output.shape, (
        f"shape mismatch: {pt_output.shape} vs {tt_output.shape}"
    )
    assert pt_output.dtype == tt_output.dtype, (
        f"dtype mismatch: {pt_output.dtype} vs {tt_output.dtype}"
    )
    x, y = pt_output.flatten(), tt_output.flatten()
    vx, vy = x - x.mean(), y - y.mean()
    pcc = ((vx @ vy) / (vx.norm() * vy.norm())).item()
    print(f"PCC: {pcc:.6f}")
    assert pcc == exact_pcc, f"PCC {pcc} does not match expected {exact_pcc}"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="test_lexora_lite_3b_v2 codegen pipeline")

    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run-pt", action="store_true", help="Run PyTorch model on CPU")
    mode.add_argument("--run-tt", action="store_true", help="Run model on TT hardware")
    mode.add_argument("--codegen", action="store_true", help="Generate TTNN code")
    mode.add_argument(
        "--golden", action="store_true", help="Compare PyTorch and TTNN runs"
    )

    args = parser.parse_args()

    if args.run_pt:
        run_pytorch_model()
    if args.run_tt:
        run_tt_model()
    if args.codegen:
        codegen_model()
    if args.golden:
        compare_pytorch_and_tt_runs()
