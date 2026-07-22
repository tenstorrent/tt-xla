<!-- SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# vLLM on Tenstorrent

These examples run a standard `vllm serve` OpenAI-compatible server with models
executing on Tenstorrent hardware. This works through the **`vllm_tt`** out-of-tree
vLLM platform plugin (in [`integrations/vllm_plugin`](../../integrations/vllm_plugin)),
which routes vLLM's compiled graphs through `torch_xla` → tt-xla's PJRT plugin →
tt-mlir → tt-metal.

## Prerequisites

Activate the tt-xla environment and install the plugin and its pinned dependencies:

```bash
source venv/activate

# vllm==0.22.1, transformers==5.5.1, fastapi (see requirements file)
uv pip install -r integrations/vllm_plugin/requirements-vllm-plugin.txt

# The plugin itself (registers the "tt" platform via entry points)
uv pip install ./integrations/vllm_plugin
```

> These use `uv pip`, matching the rest of the project (the CI images install
> `uv`, and `venv/activate` aliases `PIP` to `uv pip` when it is available).
> Plain `pip install ...` works too if you prefer.

Verify the device is visible:

```bash
python -c "import torch_xla.core.xla_model as xm; print(xm.get_xla_supported_devices())"
```

## How it fits together

- `vllm serve` picks up `vllm_tt` automatically via its entry point (`tt = vllm_tt:register`),
  which selects `TTPlatform` (`device_type="xla"`, `simple_compile_backend="tt"`).
- TT-specific knobs are passed through vLLM's `--additional-config` as JSON, e.g.
  `{"enable_const_eval": "False", "min_context_len": 32}`. The plugin converts these
  into PJRT compile options.
- Environment variables tune the tt-xla backend: `TTXLA_LOGGER_LEVEL=DEBUG` for verbose
  logs, `TT_RUNTIME_ENABLE_PROGRAM_CACHE=1` to cache compiled programs.

## Examples

Each example ships a `service.sh` (starts the server on `http://localhost:8000`)
and a `client.py`. Start the server in one terminal, run the client in another.

### Chat / generation

| Example | Model | Notes |
| --- | --- | --- |
| [`TinyLlama-1.1B-Chat-v1.0/`](TinyLlama-1.1B-Chat-v1.0) | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | Smallest generative demo. Includes `responses_client.py` for the OpenAI `/v1/responses` API. |
| [`Qwen3-0.6B-codegen/`](Qwen3-0.6B-codegen) | `Qwen/Qwen3-0.6B` | Serves from **pre-emitted TTNN codegen** instead of compiling at startup (see below). |

```bash
# Terminal 1
bash examples/vllm/TinyLlama-1.1B-Chat-v1.0/service.sh

# Terminal 2 (interactive chat over /v1/chat/completions, streamed)
python examples/vllm/TinyLlama-1.1B-Chat-v1.0/client.py
```

### Embeddings

| Example | Model |
| --- | --- |
| [`BGE-M3/`](BGE-M3) | `BAAI/bge-m3` |
| [`Qwen3-Embedding-4B/`](Qwen3-Embedding-4B) | `Qwen/Qwen3-Embedding-4B` |

```bash
# Terminal 1
bash examples/vllm/BGE-M3/service.sh

# Terminal 2 (posts to /v1/embeddings, prints the embedding vector)
python examples/vllm/BGE-M3/client.py
```

## Codegen load mode (Qwen3-0.6B)

The `Qwen3-0.6B-codegen` example demonstrates emitting TTNN Python codegen once,
then serving from it to skip SHLO→TTIR→TTNN compilation on subsequent runs:

```bash
# 1. Emit once (offline) — compiles and writes per-graph Python to ./qwen_codegen
python examples/vllm/Qwen3-0.6B-codegen/qwen.py --emit

# 2. Serve from that dir (params in service.sh MUST match the emit run)
bash examples/vllm/Qwen3-0.6B-codegen/service.sh

# 3. Chat
python examples/vllm/Qwen3-0.6B-codegen/client.py
```

The plugin matches each graph by StableHLO hash against the saved subdirs and runs
the (optionally edited) `main.py`. See the header comments in
[`qwen.py`](Qwen3-0.6B-codegen/qwen.py) and
[`service.sh`](Qwen3-0.6B-codegen/service.sh) for details.

## Notes

- The `--additional-config` flags and server parameters (`--max-model-len`, etc.)
  are tuned per model; keep them in sync if you copy a `service.sh` as a starting point.
- Clients maintain conversation history themselves — vLLM keeps no server-side state.
- If a client reports a connection error, the server is likely still warming up
  (compiling graphs); wait for it to become ready and retry.
