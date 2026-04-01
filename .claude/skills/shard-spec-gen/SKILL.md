---
name: shard-spec-gen
description: Generates shard specs and minimal pytest tests for HuggingFace model components on Tenstorrent hardware (tt-xla). Use this skill whenever the user wants to shard a model, generate a shard spec, distribute a layer across devices, write a multi-chip test, or asks about tensor parallelism, Megatron sharding, mesh configuration, or how to split attention/MLP/MoE weights across TT devices. Trigger even if the user just names a model and mentions devices or sharding without being explicit.
allowed-tools: Bash Read Grep Glob Write Edit
---

# Shard Spec Generator for tt-xla

You help users generate **shard specs** and **minimal pytest tests** for HuggingFace model components running on Tenstorrent hardware via the tt-xla PJRT backend.

## Step 1: Gather information

**STOP. Do not proceed to any later step until you have explicit answers to all four questions below.**

If any of the following are missing from the user's request, ask for all missing ones at once and wait for their reply before doing anything else — no build checks, no file searches, nothing:

1. **Model**: Which HuggingFace model? (e.g. `meta-llama/Llama-3.1-8B`)
2. **Component**: Which part?
   - `attention` — just the attention layer
   - `mlp` — just the MLP/FFN layer
   - `moe` — mixture-of-experts block
   - `decoder_layer` — full decoder layer (attention + MLP)
3. **Strategy**: `megatron`, `data parallel`, etc
4. **Target hardware / mesh shape**: Which hardware are they targeting?
   - `single_device` — 1 chip, no sharding needed (mesh=None)
   - `llmbox` — 8 chips; mesh `(1, 8)` with axes `("batch", "model")`
   - `galaxy` — 32 chips; mesh `(4, 8)` with axes `("batch", "model")`
   - If they give a raw device count, recommend: 2 → `(1, 2)`; 4 → `(1, 4)`; 8 → `(1, 8)`; 32 → `(4, 8)`
   - Fall back to batch splitting if heads aren't divisible by the model-axis size
   - Include all available targets in `@parametrize_arch` (e.g. `["single_device", "llmbox", "galaxy"]` if galaxy is relevant)

Only proceed to Step 2 once you have confirmed answers for all four items above.

## Step 2: Check the build

Before anything else, verify the environment is ready to run tests:

```bash
source venv/activate

# Check a build exists and is usable
if [ ! -f build/CMakeCache.txt ]; then
  echo "NO BUILD FOUND — run: cmake -G Ninja -B build && cmake --build build"
elif ! python3 -c "import torch_xla" 2>/dev/null; then
  echo "torch_xla not importable — venv may need rebuilding"
else
  echo "Build OK"
  grep CMAKE_BUILD_TYPE build/CMakeCache.txt
fi
```

If no build exists, tell the user and stop — the tests cannot run without it. The build command is:
```bash
source venv/activate
cmake -G Ninja -B build
cmake --build build
```

## Step 3: Find the ModelLoader (or fall back to AutoConfig)

Check `third_party/tt_forge_models/` for a matching ModelLoader:
```bash
find third_party/tt_forge_models -name "loader.py" | xargs grep -l "ModelLoader" | head -20
```

**If a ModelLoader exists** — use it. Check whether it accepts `num_layers` (e.g. `GPTOSSModelLoader(num_layers=1)`) by reading the loader's `__init__` signature, since this matters for MoE. Also list available variants:
```bash
python3 -c "
from third_party.tt_forge_models.<model>.<task>.pytorch.loader import ModelLoader, ModelVariant
print([v for v in ModelVariant])
"
```

**If no ModelLoader exists** — fall back to HuggingFace directly:
```python
from transformers import AutoConfig
config = AutoConfig.from_pretrained("<hf-model-id>")
```
If you get a `GatedRepoError` or 401 when calling `AutoConfig.from_pretrained()`, stop and ask the user to set their HuggingFace token:

```
Please set your HuggingFace token so we can access the model config:

  export HF_TOKEN=<your-token>

You can find your token at https://huggingface.co/settings/tokens
Once set, let me know and I'll continue.
```

Wait for the user to confirm, then re-run with the token in the environment. Do not proceed to the manual config fallback unless the user explicitly says they don't have a token or don't want to use one.

For a completely offline fallback (no token available), construct a minimal config manually:
```python
from transformers import LlamaConfig  # or whichever config class
config = LlamaConfig(
    hidden_size=2048,
    intermediate_size=8192,
    num_attention_heads=32,
    num_key_value_heads=8,
    num_hidden_layers=1,
)
```
Look up the correct config class and values from the model card on HuggingFace (use WebFetch if needed). When using a manual config, note in the generated test that the config values should be updated to match the actual model checkpoint.

## Step 4: Verify layer names by loading the model interactively

**Before writing any shard spec**, confirm the actual attribute paths and config values. Run a quick Python snippet:

```bash
source venv/activate
python3 - <<'EOF'
from third_party.tt_forge_models.<model>.<task>.pytorch.loader import ModelLoader, ModelVariant
import torch

loader = ModelLoader(variant=ModelVariant.<VARIANT>)
config = loader.load_config()

# For attention/MLP: instantiate the layer directly from config (fast)
from transformers.models.<arch>.modeling_<arch> import <LayerClass>
layer = <LayerClass>(config).to(torch.bfloat16)
print("Layer attributes:", [n for n, _ in layer.named_parameters()])
print("hidden_size:", config.hidden_size)
print("num_attention_heads:", getattr(config, "num_attention_heads", "N/A"))
print("num_key_value_heads:", getattr(config, "num_key_value_heads", "N/A"))
print("intermediate_size:", getattr(config, "intermediate_size", "N/A"))

# For MoE: need to load the full model (use minimal layers)
# config.num_hidden_layers = 1  # override before loading
# model = loader.load_model(dtype_override=torch.bfloat16)  (or pass num_layers=1)
# mlp = model.model.layers[0].mlp
# print("MoE block type:", type(mlp))
# print("MoE attrs:", [n for n, _ in mlp.named_parameters()][:20])
EOF
```

Use the output to confirm:
- Exact attribute paths (`attention.q_proj.weight` vs `attention.c_attn.weight` etc.)
- Whether it's MoE and what the expert structure looks like
- Config values needed for input tensor shapes

**For MoE models**, load the full model but minimize it first:
```python
config = loader.load_config()
config.num_hidden_layers = 1   # only one transformer layer — much faster
# If loader supports num_layers: ModelLoader(num_layers=1)
model = loader.load_model(dtype_override=torch.bfloat16)
mlp = model.model.layers[0].mlp
print(type(mlp))
print([n for n, _ in mlp.named_parameters()][:30])
# Check for shared_expert:
print("has shared_expert:", hasattr(mlp, "shared_expert"))
print("has experts:", hasattr(mlp, "experts"))
if hasattr(mlp, "experts"):
    print("num experts:", len(mlp.experts))
    print("expert type:", type(mlp.experts[0]))
```

## Step 5: Generate the shard spec

### Megatron tensor parallel rules

Weight shape is `[out_features, in_features]`. Mesh axis name is typically `"model"`.

**Attention (standard):**
```python
shard_specs[attention.q_proj.weight] = ("model", None)   # column-parallel
shard_specs[attention.k_proj.weight] = ("model", None)   # column-parallel
shard_specs[attention.v_proj.weight] = ("model", None)   # column-parallel
shard_specs[attention.o_proj.weight] = (None, "model")   # row-parallel
```

**MLP (standard gate/up/down):**
```python
shard_specs[mlp.gate_proj.weight] = ("model", None)
shard_specs[mlp.up_proj.weight]   = ("model", None)
shard_specs[mlp.down_proj.weight] = (None, "model")
```

**MoE (per-expert, iterate over experts):**
```python
for expert in mlp.experts:
    shard_specs[expert.gate_proj.weight] = ("model", None)
    shard_specs[expert.up_proj.weight]   = ("model", None)
    shard_specs[expert.down_proj.weight] = (None, "model")
# If shared_expert exists (e.g. Qwen3-MoE):
if hasattr(mlp, "shared_expert") and mlp.shared_expert is not None:
    shard_specs[mlp.shared_expert.gate_proj.weight] = ("model", None)
    shard_specs[mlp.shared_expert.up_proj.weight]   = ("model", None)
    shard_specs[mlp.shared_expert.down_proj.weight] = (None, "model")
```

**When heads aren't divisible by num_devices** — fall back to batch sharding on inputs:
```python
shard_specs[args[0]] = ("batch", None, None)   # hidden_states
# plus all weight specs as above
```

Use the actual attribute names confirmed in Step 3 — don't assume.

## Step 6: Generate the test file

Follow the conventions from `tests/torch/graphs/test_attention.py` and `test_mlp.py`.

Save to: `tests/torch/graphs/test_<model>_<component>.py`

```python
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import torch
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.evaluators.evaluation_config import ComparisonConfig, PccConfig
from infra.utilities.torch_multichip_utils import enable_spmd
from torch_xla.distributed.spmd import Mesh
from transformers.models.<arch>.modeling_<arch> import <LayerClass>

from tests.utils import parametrize_arch
from third_party.tt_forge_models.<model>.<task>.pytorch.loader import (
    ModelLoader, ModelVariant,
)


@pytest.mark.nightly
@parametrize_arch(["llmbox"])
@pytest.mark.parametrize("seq_len", [1024])
def test_<model>_<component>_sharded(seq_len, arch):
    xr.set_device_type("TT")

    loader = ModelLoader(variant=ModelVariant.<VARIANT>)
    config = loader.load_config()
    # For attention:
    config._attn_implementation = "sdpa"
    layer = <LayerClass>(config, layer_idx=0).to(torch.bfloat16)

    # For MoE — load minimal model:
    # config.num_hidden_layers = 1
    # model = loader.load_model(dtype_override=torch.bfloat16)
    # layer = model.model.layers[0].mlp

    enable_spmd()
    num_devices = xr.global_runtime_device_count()
    # llmbox: (1, 8); galaxy: (4, 8)
    mesh_shape = (1, num_devices) if arch == "llmbox" else (4, num_devices // 4)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

    hidden_states = torch.randn(
        (mesh_shape[0], seq_len, config.hidden_size), dtype=torch.bfloat16
    )
    # Attention also needs:
    # cos_sin = torch.rand(mesh_shape[0], seq_len, config.head_dim, dtype=torch.bfloat16)
    # position_embeddings = (cos_sin, cos_sin)
    # attention_mask = torch.rand(mesh_shape[0], 1, seq_len, seq_len, dtype=torch.bfloat16)
    # past_key_states = None

    def get_shard_spec(layer, args, kwargs):
        shard_specs = {}
        # --- insert shard specs confirmed in Step 4/5 ---
        return shard_specs

    comparison_config = ComparisonConfig(pcc=PccConfig(required_pcc=0.98))

    run_graph_test(
        layer,
        [hidden_states],   # replace with actual inputs
        framework=Framework.TORCH,
        comparison_config=comparison_config,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )
```

### Key conventions
- `xr.set_device_type("TT")` first, always
- `enable_spmd()` before mesh creation
- `torch.bfloat16` throughout
- `@parametrize_arch` includes `llmbox` and/or `galaxy` based on the user's target — do NOT include `single_device`
- Batch size should match `mesh_shape[0]` (the batch axis of the mesh)
- One layer only — no stacking, no full forward pass through the model
- `config.num_hidden_layers = 1` when you need to load a full model (MoE) to keep it fast
- When using `AutoConfig` or a manual config instead of a ModelLoader, add a comment in the test explaining this and what the correct HF model ID is

## Step 7: Run the test

```bash
source venv/activate
TEST=tests/torch/graphs/test_<model>_<component>.py
```

Run with `TTXLA_LOGGER_LEVEL=DEBUG` and log to a file so the user can inspect the compiled graph and verify the shard spec is applied correctly. Replace `llmbox` with `galaxy` as appropriate:
```bash
ARCH=llmbox   # or galaxy
LOGFILE="${TEST%.py}_${ARCH}.log"
TTXLA_LOGGER_LEVEL=DEBUG pytest -svv $TEST \
  -k "$ARCH" --no-header 2>&1 | tee "$LOGFILE"
echo "Full log saved to: $LOGFILE"
```

After the run, show the user the key section of the log that shows the sharded TTNN graph. Search for shard annotations:
```bash
grep -E "shard|mesh|replicated|device_ids" "$LOGFILE" | head -40
```

Tell the user: "Check `$LOGFILE` to see the full compiled graph. Look for the tensor shard annotations to confirm your weights are split across devices as expected."

## Step 8: Deliver

Show the user:
1. The verified shard spec dict (standalone snippet)
2. The complete test file path in `tests/torch/graphs/`
3. Test run output summary
4. A brief explanation of why each tensor is sharded the way it is
