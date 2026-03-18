# vLLM Serving Examples

Examples for running LLM inference on Tenstorrent hardware via vLLM.

## Available Models

| Model | Directory | Parameters | Notes |
|-------|-----------|------------|-------|
| TinyLlama-1.1B-Chat-v1.0 | `TinyLlama-1.1B-Chat-v1.0/` | 1.1B | Smallest, fastest to compile |
| Llama-3.2-3B-Instruct | `Llama-3.2-3B-Instruct/` | 3B | Gated model (requires HF access) |
| Llama-3.1-8B-Instruct | `Llama-3.1-8B-Instruct/` | 8B | Gated model (requires HF access) |
| BGE-M3 | `BGE-M3/` | 568M | Embedding model |
| Qwen3-Embedding-4B | `Qwen3-Embedding-4B/` | 4B | Embedding model |

## Prerequisites

- TT-XLA environment activated (`source venv/activate`)
- vLLM TT plugin installed (`pip install` the wheel from `integrations/vllm_plugin/`)
- For gated models (Llama): set `HF_TOKEN` or run `huggingface-cli login`

## Quick Start

Each model directory contains:
- `service.sh` — starts the vLLM OpenAI-compatible server
- `client.py` — interactive streaming chat client (`/v1/chat/completions`)
- `responses_client.py` — interactive streaming chat client (`/v1/responses`)

### 1. Start the server

```bash
bash examples/vllm/Llama-3.2-3B-Instruct/service.sh
```

Wait for the server to print `Uvicorn running on http://0.0.0.0:8000` before sending requests.

### 2. Chat interactively (separate terminal)

```bash
python examples/vllm/Llama-3.2-3B-Instruct/client.py
```

The client prints a stats line after each response:
```
[87 tokens, TTFT: 1.234s, 12.50 tok/s]
```

### 3. Benchmark throughput (separate terminal)

```bash
python examples/vllm/benchmark_serving.py --num-prompts 5 --max-tokens 128
```

The benchmark script auto-detects the served model via `/v1/models`. Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--base-url` | `http://localhost:8000` | Server base URL |
| `--model` | auto-detected | Override model name |
| `--num-prompts` | 5 | Number of prompts to run |
| `--max-tokens` | 128 | Max tokens per completion |
| `--warmup` | 1 | Warmup requests (excluded from metrics) |

## Llama Model Sizes

- **Llama 3.2**: 1B, 3B (text only); 11B, 90B (vision)
- **Llama 3.1**: 8B, 70B, 405B
- **Instruct** variants are fine-tuned for chat/instruction-following (same architecture and perf as base models)

## Server Configuration Notes

Key `service.sh` flags:
- `--max-model-len` — max sequence length the server will accept
- `--max-num-batched-tokens` — total tokens per batch
- `--max-num-seqs` — max concurrent requests (1 for single-chip demos)
- `--gpu-memory-utilization` — fraction of device memory for KV cache (increase if you get KV cache OOM errors)
- `--no-enable-prefix-caching` — required for TT backend
- `--additional-config` — TT-specific options:
  - `enable_const_eval` — set `"False"` to avoid OOM when pre-compiling multiple context lengths
  - `min_context_len` — minimum padded context length (32 is typical)

If you see an error like:
```
ValueError: To serve at least one request ... KV cache is needed, which is larger than the available KV cache memory
```
Increase `--gpu-memory-utilization` (e.g. from `0.01` to `0.05`).
