
#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Save weight tensors from GPT-OSS 120B transformer layers for use in
single-op matmul tests.

Loads each target layer via a remapped 1-layer model checkpoint so that
from_pretrained handles MXFP4 dequantization correctly. Weights are then
extracted from the loaded model and saved as individual .pt files.

Only nn.Linear weight matrices are saved (q_proj, k_proj, v_proj, o_proj,
mlp router). MoE gate_up/down_proj excluded — 3D weight, not isolatable.

Output (--output-dir):
  layer_<N>/<op>/weight.pt    — weight tensor (bfloat16, dequantized)

Usage:
  # Default: layers 0/18/19, model must already be cached
  python scripts/extract_gpt_oss_matmul_activations.py

  # Custom layers
  python scripts/extract_gpt_oss_matmul_activations.py --layers 0 18 19

  # Custom output directory
  python scripts/extract_gpt_oss_matmul_activations.py --output-dir /tmp/weights

After running, set GPT_OSS_120B_WEIGHTS_DIR to the output directory and run:
  pytest tests/operators/test_matmul_gpt_oss_120b.py  # in tt-xla
"""

import argparse
import json
import os
import shutil
import tempfile
import torch
from pathlib import Path

MODEL_NAME = "openai/gpt-oss-120b"
TARGET_LAYERS = [0, 18, 19]

# Maps op_subdir → attribute path on layer object
_OP_ATTRS = [
    ("self_attn_q_proj", lambda layer: layer.self_attn.q_proj.weight),
    ("self_attn_k_proj", lambda layer: layer.self_attn.k_proj.weight),
    ("self_attn_v_proj", lambda layer: layer.self_attn.v_proj.weight),
    ("self_attn_o_proj", lambda layer: layer.self_attn.o_proj.weight),
    ("mlp_router",       lambda layer: layer.mlp.router.weight),
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=Path("gpt_oss_120b_weights"),
                        help="Directory to save weight tensors (default: gpt_oss_120b_weights)")
    parser.add_argument("--layers", type=int, nargs="+", default=TARGET_LAYERS,
                        help=f"Layer indices to extract (default: {TARGET_LAYERS})")
    parser.add_argument("--model-dir", type=Path, default=None,
                        help="Path to cached model directory. If not set, auto-detected via huggingface_hub.")
    return parser.parse_args()


def _find_model_dir(model_dir: Path = None) -> Path:
    if model_dir is not None:
        return model_dir
    try:
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(MODEL_NAME, local_files_only=True))
    except Exception as e:
        raise RuntimeError(
            f"Could not locate cached model. Pass --model-dir or cache the model first.\n{e}"
        )


def _create_remapped_checkpoint(model_dir: Path, target_layer: int) -> Path:
    """
    Creates a temporary checkpoint directory where model.layers.0.* keys
    point to target_layer's weights, so from_pretrained loads the correct
    layer (including MXFP4 dequantization) into the single-layer model.
    """
    from safetensors import safe_open
    from safetensors.torch import save_file

    with open(model_dir / "model.safetensors.index.json") as f:
        orig_index = json.load(f)
    weight_map = orig_index["weight_map"]

    src_prefix = f"model.layers.{target_layer}."
    dst_prefix = "model.layers.0."

    shard_to_keys: dict = {}
    for key, shard in weight_map.items():
        if key.startswith(src_prefix):
            shard_to_keys.setdefault(shard, []).append(key)

    remapped: dict = {}
    for shard_name, keys in shard_to_keys.items():
        with safe_open(str(model_dir / shard_name), framework="pt", device="cpu") as f:
            for key in keys:
                new_key = key.replace(src_prefix, dst_prefix)
                remapped[new_key] = f.get_tensor(key).clone()

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"gpt_oss_layer{target_layer}_"))

    remapped_shard = "model-layer-remapped.safetensors"
    save_file(remapped, str(tmp_dir / remapped_shard))

    new_weight_map = {}
    remapped_keys = set(remapped.keys())
    for key, shard in weight_map.items():
        if key.startswith(dst_prefix):
            if key in remapped_keys:
                new_weight_map[key] = remapped_shard
        else:
            new_weight_map[key] = shard

    new_index = {"metadata": orig_index.get("metadata", {}), "weight_map": new_weight_map}
    with open(tmp_dir / "model.safetensors.index.json", "w") as f:
        json.dump(new_index, f)

    for shard_file in model_dir.glob("model-*.safetensors"):
        link = tmp_dir / shard_file.name
        if not link.exists():
            os.symlink(shard_file, link)

    for fname in model_dir.iterdir():
        if fname.suffix in (".json", ".jinja") and fname.name != "model.safetensors.index.json":
            dst = tmp_dir / fname.name
            if not dst.exists():
                shutil.copy2(fname, dst)

    return tmp_dir


def _load_layer(model_dir: Path, target_layer: int):
    """Loads a 1-layer model with target_layer's weights (including dequantization)."""
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


def extract_weights(layers: list, output_dir: Path, model_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    for layer_idx in layers:
        print(f"\nLoading layer {layer_idx}...")
        model = _load_layer(model_dir, layer_idx)
        layer = model.model.layers[0]

        for op_subdir, get_weight in _OP_ATTRS:
            weight = get_weight(layer).detach().cpu().clone()

            op_dir = output_dir / f"layer_{layer_idx}" / op_subdir
            op_dir.mkdir(parents=True, exist_ok=True)
            path = op_dir / "weight.pt"
            torch.save(weight, path)
            print(f"  Saved layer_{layer_idx}/{op_subdir}/weight.pt  shape={list(weight.shape)}  dtype={weight.dtype}")
            saved += 1

        del model

    return saved


def run(
    output_dir: Path = Path("gpt_oss_120b_weights"),
    layers: list = TARGET_LAYERS,
    model_dir: Path = None,
) -> int:
    resolved_model_dir = _find_model_dir(model_dir)
    print(f"Model dir: {resolved_model_dir}")
    print(f"Extracting layers {layers}...")

    saved = extract_weights(layers, output_dir, resolved_model_dir)

    print(f"\nDone. Saved {saved} weight tensors to {output_dir}/")
    print(f"Set GPT_OSS_120B_WEIGHTS_DIR={output_dir.resolve()} before running tests.")
    return saved


def test_extract_gpt_oss_120b_weights():
    """Pytest entry point — runs with default settings (layers 0/18/19)."""
    saved = run()
    assert saved == len(TARGET_LAYERS) * len(_OP_ATTRS)


def main():
    args = parse_args()
    run(
        output_dir=args.output_dir,
        layers=args.layers,
        model_dir=args.model_dir,
    )


if __name__ == "__main__":
    main()
