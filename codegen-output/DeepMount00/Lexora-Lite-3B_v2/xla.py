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


def load_input(model, device="cpu"):
    """Replicates construct_inputs() from benchmarks/llm_benchmark.py:85-169 (StaticCache path).

    The StaticCache is built directly on `device` (not on CPU then transferred): the
    transformers 5.2.0 StaticLayer caches its own `device`/`cumulative_length` internally
    and uses them inside `update()` / `get_seq_length()`, so a partial keys/values transfer
    leaves the layer inconsistent (mixes cpu and xla tensors during the forward). Building on
    the device keeps every cache tensor consistent with the model and inputs. Issue
    https://github.com/tenstorrent/tt-xla/issues/1645 (the reason the benchmark builds on CPU)
    only affects traced runs; trace is disabled here.
    """
    config = model.config
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    prompts = [DEFAULT_INPUT_PROMPT] * BATCH_SIZE
    tokenized = tokenizer(
        prompts,
        return_tensors="pt",
        max_length=INPUT_SEQUENCE_LENGTH,
        truncation=True,
    )
    input_ids = tokenized["input_ids"].to(device)

    past_key_values = StaticCache(
        config=config,
        max_batch_size=BATCH_SIZE,
        max_cache_len=INPUT_SEQUENCE_LENGTH,
        device=device,
        dtype=DATA_FORMAT,
    )
    # Mirror init_static_cache() from tests/benchmark/llm_utils/decode_utils.py:113-145:
    # eagerly allocate per-layer keys/values (the raw constructor leaves them lazily None).
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
        device=device,
    )

    cache_position = torch.arange(0, input_ids.shape[1], device=device)

    return {
        "input_ids": input_ids,
        "past_key_values": past_key_values,
        "cache_position": cache_position,
        "use_cache": True,
    }


def run_pytorch_model():
    model = load_pytorch_model()
    inputs = load_input(model, device="cpu")

    with torch.no_grad():
        output = model(**inputs)

    return output.logits


def run_tt_model():
    device = torch_xla.device()

    torch_xla.set_custom_compile_options(COMPILE_OPTIONS)

    model = load_pytorch_model()
    model.compile(backend="tt", options={"tt_legacy_compile": True})
    model = model.to(device)
    inputs = load_input(model, device=device)

    with torch.no_grad():
        output = model(**inputs)

    return output.logits.cpu()


def codegen_model():
    os.environ["XLA_HLO_DEBUG"] = "1"

    model = load_pytorch_model()
    inputs = load_input(model, device="cpu")

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
