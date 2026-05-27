# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Full transformer layer forward tests for GPT-OSS 120B on TT device.

Loads a 1-layer model skeleton and injects weights from the target layer
directly from safetensors shards (no full model load). Runs the full
layer forward (attention + MoE) on TT device and compares to CPU golden.

Note: MoE expert weights are stored in a custom block-quantized format
(gate_up_proj_blocks/scales/bias). If TT device cannot execute the custom
dequantization kernels, the MoE sub-module will fail. In that case, use
test_matmul_gpt_oss_120b.py which tests individual matmul ops in isolation.

Required: GPT-OSS 120B must be cached in the HuggingFace hub cache.

Optional env var:
    GPT_OSS_120B_MODEL_DIR — explicit path to the cached model snapshot.
                             If not set, auto-detected via huggingface_hub.
    REL_L2_OUTPUT          — path to JSONL file for metric recording
                             (see test_matmul_gpt_oss_120b.py for details).

Run:
    pytest tests/operators/test_layer_gpt_oss_120b.py -k "layer_0 and opt0_bf16" -s -v
"""

import json
import os
import shutil
import tempfile
from itertools import product
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from infra import Framework, run_op_test
from tests.infra.testers.compiler_config import CompilerConfig

MODEL_NAME = "openai/gpt-oss-120b"
_HIDDEN_SIZE = 2880
_SEQ_LEN = 128
_LAYERS = [0, 18, 19]

# Full compiler config matrix (same as test_matmul_gpt_oss_120b.py)
_WEIGHT_DTYPE_FIDELITY = [
    ("",        ["hifi4", "hifi3", "hifi2", "lofi"]),
    ("bfp_bf8", ["hifi4", "hifi3", "hifi2", "lofi"]),
    ("bfp_bf4", ["hifi4", "hifi3", "hifi2", "lofi"]),
]
_DTYPE_LABEL = {"": "bf16", "bfp_bf8": "bfp8", "bfp_bf4": "bfp4"}

_COMPILER_CONFIGS = [
    CompilerConfig(
        optimization_level=opt,
        experimental_weight_dtype=dtype,
        math_fidelity=fidelity,
        fp32_dest_acc_en=fp32,
    )
    for opt, (dtype, fidelities), fp32 in product([0, 2], _WEIGHT_DTYPE_FIDELITY, [True, False])
    for fidelity in fidelities
]
_COMPILER_IDS = [
    f"opt{c.optimization_level}_{_DTYPE_LABEL[c.experimental_weight_dtype]}_{c.math_fidelity}_fp32{'true' if c.fp32_dest_acc_en else 'false'}"
    for c in _COMPILER_CONFIGS
]


# ---------------------------------------------------------------------------
# Model loading + weight injection
# ---------------------------------------------------------------------------


def _find_model_dir() -> Path:
    explicit = os.environ.get("GPT_OSS_120B_MODEL_DIR")
    if explicit:
        return Path(explicit)
    try:
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(MODEL_NAME, local_files_only=True))
    except Exception as e:
        pytest.skip(f"GPT-OSS 120B not found in HuggingFace cache. "
                    f"Set GPT_OSS_120B_MODEL_DIR or cache the model first.\n{e}")


def _create_remapped_checkpoint(model_dir: Path, target_layer: int) -> Path:
    """
    Creates a temporary checkpoint directory where model.layers.0.* keys
    point to target_layer's weights. from_pretrained on this directory will
    load target_layer's weights (including proper MXFP4 dequantization) into
    the single-layer model's layer 0.

    Strategy:
    - Read all model.layers.{N}.* tensors from safetensors shards
    - Rename keys to model.layers.0.* and save as a new shard
    - Write a modified index that maps layer 0 keys to the new shard
    - Symlink all other shards + copy config files
    """
    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError:
        pytest.skip("safetensors not installed: pip install safetensors")

    with open(model_dir / "model.safetensors.index.json") as f:
        orig_index = json.load(f)
    weight_map = orig_index["weight_map"]

    # Collect target layer weights
    src_prefix = f"model.layers.{target_layer}."
    dst_prefix = "model.layers.0."
    shard_to_keys: dict = {}
    for key, shard in weight_map.items():
        if key.startswith(src_prefix):
            shard_to_keys.setdefault(shard, []).append(key)

    if not shard_to_keys:
        pytest.skip(f"No keys found for layer {target_layer} in safetensors index")

    # Read and remap
    remapped: dict = {}
    for shard_name, keys in shard_to_keys.items():
        with safe_open(str(model_dir / shard_name), framework="pt", device="cpu") as f:
            for key in keys:
                new_key = key.replace(src_prefix, dst_prefix)
                remapped[new_key] = f.get_tensor(key).clone()

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"gpt_oss_layer{target_layer}_"))

    # Save remapped weights as a new shard
    remapped_shard = "model-layer-remapped.safetensors"
    save_file(remapped, str(tmp_dir / remapped_shard))

    # Build new index: layer 0 keys → remapped shard; everything else unchanged
    new_weight_map = {}
    remapped_keys = set(remapped.keys())
    for key, shard in weight_map.items():
        if key.startswith(dst_prefix):
            if key in remapped_keys:
                new_weight_map[key] = remapped_shard
            # keys in layer 0 not present for target layer are dropped
        else:
            new_weight_map[key] = shard

    new_index = {"metadata": orig_index.get("metadata", {}), "weight_map": new_weight_map}
    with open(tmp_dir / "model.safetensors.index.json", "w") as f:
        json.dump(new_index, f)

    # Symlink all original shards (needed for embeddings, lm_head, etc.)
    for shard_file in model_dir.glob("model-*.safetensors"):
        link = tmp_dir / shard_file.name
        if not link.exists():
            os.symlink(shard_file, link)

    # Copy config files
    for fname in model_dir.iterdir():
        if fname.suffix in (".json", ".jinja") and fname.name != "model.safetensors.index.json":
            dst = tmp_dir / fname.name
            if not dst.exists():
                shutil.copy2(fname, dst)

    return tmp_dir


def _build_single_layer_model(model_dir: Path, target_layer: int) -> nn.Module:
    """
    Loads a 1-layer model with target_layer's weights by creating a temporary
    remapped checkpoint. from_pretrained handles MXFP4 dequantization correctly
    for any target layer.
    """
    from transformers import AutoConfig, AutoModelForCausalLM

    if target_layer == 0:
        checkpoint_dir = model_dir
        tmp_dir = None
    else:
        tmp_dir = _create_remapped_checkpoint(model_dir, target_layer)
        checkpoint_dir = tmp_dir

    try:
        config = AutoConfig.from_pretrained(str(checkpoint_dir), trust_remote_code=True)
        config.num_hidden_layers = 1
        model = AutoModelForCausalLM.from_pretrained(
            str(checkpoint_dir),
            config=config,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation="eager",
        )
        model.eval()
    finally:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return model


# ---------------------------------------------------------------------------
# Layer wrapper
# ---------------------------------------------------------------------------


class _LayerWrapper(nn.Module):
    """
    Wraps a single HuggingFace decoder layer.

    GPT-OSS uses the newer HF pattern where RoPE embeddings are precomputed
    at the model level (model.rotary_emb) and passed as position_embeddings=(cos, sin)
    to each layer. position_ids are used only for indexing into the RoPE cache.

    Input:  hidden_states (B, S, H), position_ids (B, S)
    Output: hidden_states (B, S, H)  — first element of layer output tuple
    """

    def __init__(self, layer: nn.Module, rotary_emb: nn.Module):
        super().__init__()
        self.layer = layer
        self.rotary_emb = rotary_emb

    def forward(self, hidden_states: torch.Tensor, position_ids: torch.Tensor) -> torch.Tensor:
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        out = self.layer(hidden_states, position_embeddings=position_embeddings)
        return out[0] if isinstance(out, (tuple, list)) else out


# ---------------------------------------------------------------------------
# Metrics comparator (same logic as test_matmul_gpt_oss_120b.py)
# ---------------------------------------------------------------------------


def _make_comparator(test_id: str, request, pcc_threshold: float = 0.98):
    def _comparator(tt_res, cpu_res, args, kwargs):
        tt_f64 = tt_res.cpu().to(torch.float64).flatten()
        cpu_f64 = cpu_res.cpu().to(torch.float64).flatten()

        diff_norm = torch.linalg.vector_norm(tt_f64 - cpu_f64).item()
        golden_norm = torch.linalg.vector_norm(cpu_f64).item()
        if golden_norm == 0.0:
            rel_l2 = 0.0 if diff_norm == 0.0 else float("inf")
        else:
            rel_l2 = diff_norm / golden_norm

        stacked = torch.stack([tt_f64.float(), cpu_f64.float()])
        pcc = float(torch.corrcoef(stacked)[0, 1])

        print(f"\n[METRICS] rel_l2={rel_l2:.6f}  pcc={pcc:.6f}  test={test_id}")

        if request is not None:
            request.node.user_properties.append(("rel_l2", rel_l2))
            request.node.user_properties.append(("pcc", pcc))

        output_path = os.environ.get("REL_L2_OUTPUT")
        if output_path:
            entry = json.dumps({"test_id": test_id, "rel_l2": rel_l2, "pcc": pcc})
            with open(output_path, "a") as f:
                f.write(entry + "\n")

        assert pcc >= pcc_threshold, f"PCC {pcc:.6f} < {pcc_threshold}"

    return _comparator


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.single_device
@pytest.mark.parametrize("compiler_config", _COMPILER_CONFIGS, ids=_COMPILER_IDS)
@pytest.mark.parametrize("layer_idx", _LAYERS, ids=[f"layer_{i}" for i in _LAYERS])
def test_layer_gpt_oss_120b(layer_idx, compiler_config, request):
    """Full transformer layer forward (attention + MoE) with GPT-OSS 120B weights."""
    model_dir = _find_model_dir()
    model = _build_single_layer_model(model_dir, layer_idx)
    wrapper = _LayerWrapper(model.model.layers[0], model.model.rotary_emb)

    torch.manual_seed(42)
    hidden_states = torch.randn(1, _SEQ_LEN, _HIDDEN_SIZE, dtype=torch.bfloat16)
    position_ids = torch.arange(_SEQ_LEN, dtype=torch.long).unsqueeze(0)

    run_op_test(
        wrapper,
        [hidden_states, position_ids],
        framework=Framework.TORCH,
        compiler_config=compiler_config,
        request=request,
        custom_comparator=_make_comparator(request.node.nodeid, request),
    )
