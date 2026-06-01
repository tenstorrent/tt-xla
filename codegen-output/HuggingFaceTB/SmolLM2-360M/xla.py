# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

### Standalone xla.py reproducing tt-xla benchmark test_llms.py::test_smollm2_360m.
### Resolved model: HuggingFaceTB/SmolLM2-360M (HF) via AutoModelForCausalLM.
### No imports from tt-xla/third_party/tt_forge_models or tt-xla/tests/benchmark/*.

import os
from pathlib import Path

import torch
import torch_xla
import torch_xla.runtime as xr
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import StaticCache
from tt_torch import codegen_py
from tt_torch.weight_dtype import apply_weight_dtype_overrides

MODEL_ID = "HuggingFaceTB/SmolLM2-360M"
DATA_FORMAT = torch.bfloat16     # test_llms.py: DEFAULT_DATA_FORMAT = "bfloat16"
BATCH_SIZE = 32                  # test_llms.py: DEFAULT_BATCH_SIZE = 32 (--batch-size default None)
INPUT_SEQUENCE_LENGTH = 128      # test_llms.py: DEFAULT_INPUT_SEQUENCE_LENGTH = 128
DEFAULT_INPUT_PROMPT = (
    "Here is an exaustive list of the best practices for writing clean code:"
)

# Per-test weight dtype overrides (test_smollm2_360m passes weight_dtype_overrides=...).
# Applied via tt_torch.weight_dtype.apply_weight_dtype_overrides before compiling.
WEIGHT_DTYPE_OVERRIDES = {"default": "bfp_bf8"}

OUTPUT_DIR = str(Path(__file__).resolve().parent / "model")

# Compile options (benchmarks/llm_benchmark.py:457-466). Per-test/DEFAULT precedence:
#   optimization_level=2                         (test override; also DEFAULT_OPTIMIZATION_LEVEL=2)
#   enable_trace=True                            (test override trace_enabled=True; emitted commented out)
#   experimental_weight_dtype="bfp_bf8"          (DEFAULT_EXPERIMENTAL_WEIGHT_DTYPE)
#   experimental_enable_permute_matmul_fusion=False (DEFAULT_EXPERIMENTAL_ENABLE_PERMUTE_MATMUL_FUSION)
# Runtime-instrumentation keys (export_path, export_model_name, ttnn_perf_metrics_*) dropped.
COMPILE_OPTIONS = {
    "optimization_level": 2,
    # "enable_trace": True,  # benchmark used trace_enabled=True
    "experimental_weight_dtype": "bfp_bf8",
    "experimental_enable_permute_matmul_fusion": False,
}


def load_pytorch_model():
    # Mirrors third_party/tt_forge_models/smollm2/causal_lm/pytorch/loader.py:load_model
    # (HUGGING_FACE branch): from_pretrained with torch_dtype passed at load time.
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DATA_FORMAT)

    # setup_model_and_tokenizer patches from benchmarks/llm_benchmark.py:73-78.
    if hasattr(model.config, "layer_types"):
        model.config.layer_types = ["full_attention"] * len(model.config.layer_types)
    if hasattr(model.config, "_experts_implementation"):
        model.config._experts_implementation = "dense"

    model.eval()
    return model


def load_input():
    """Replicates construct_inputs() from benchmarks/llm_benchmark.py:85-169."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    # SmolLM2 uses a GPT2-style tokenizer without a dedicated pad token (loader.py:92-94).
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
    config = model.config
    past_key_values = StaticCache(
        config=config,
        max_batch_size=BATCH_SIZE,
        max_cache_len=INPUT_SEQUENCE_LENGTH,
        device="cpu",
        dtype=DATA_FORMAT,
    )
    # Mirror init_static_cache() (llm_utils/decode_utils.py:113-145): preallocate the
    # per-layer key/value tensors so transfer_to_device can move them.
    if getattr(config, "head_dim", None):
        head_dim = config.head_dim
    else:
        head_dim = config.hidden_size // config.num_attention_heads
    num_key_value_heads = getattr(
        config, "num_key_value_heads", config.num_attention_heads
    )
    past_key_values.early_initialization(
        batch_size=BATCH_SIZE,
        num_heads=num_key_value_heads,
        head_dim=head_dim,
        dtype=DATA_FORMAT,
        device="cpu",
    )

    cache_position = torch.arange(0, input_ids.shape[1])

    return {
        "input_ids": input_ids,
        "past_key_values": past_key_values,
        "cache_position": cache_position,
        "use_cache": True,
    }


def _inputs_to_device(inputs, device):
    """Mirrors transfer_to_device() from benchmarks/llm_benchmark.py:179-205 for the
    StaticCache path (no MLA layers in SmolLM2)."""
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
    # Per-tensor weight dtype overrides (benchmarks/llm_benchmark.py:475-477).
    apply_weight_dtype_overrides(model, WEIGHT_DTYPE_OVERRIDES)
    model.compile(backend="tt", options={"tt_legacy_compile": True})
    model = model.to(device)
    inputs = _inputs_to_device(load_input(), device)

    with torch.no_grad():
        output = model(**inputs)

    return output.logits.cpu()


def codegen_model():
    os.environ["XLA_HLO_DEBUG"] = "1"

    model = load_pytorch_model()
    apply_weight_dtype_overrides(model, WEIGHT_DTYPE_OVERRIDES)

    # codegen_py forwards only torch.Tensor args/kwargs to the model (codegen.py:43-44),
    # so the StaticCache + dict used in run_tt/golden can't be threaded through. Emit the
    # representative prefill graph from the same tokenized input_ids; transformers builds
    # a default cache internally for this single forward.
    inputs = load_input()
    input_ids = inputs["input_ids"]

    codegen_py(
        model,
        input_ids=input_ids,
        export_path=OUTPUT_DIR,
        export_tensors=True,
        compiler_options=COMPILE_OPTIONS,
    )


def compare_pytorch_and_tt_runs():
    # Capture exact PCC from first --golden run and paste here.
    exact_pcc = 0.968750

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

    parser = argparse.ArgumentParser(description="test_smollm2_360m codegen pipeline")

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
