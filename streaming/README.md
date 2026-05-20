# Streaming Inference

Layer-streaming inference for large MoE models that don't fit on host as a
whole. Each layer is loaded from HuggingFace, sparse-MLP-rewritten, and
shipped to device one at a time so peak host RAM stays bounded.

## Layout

| File | Role |
| --- | --- |
| `core.py` | `run_streaming(adapter, config)` — the model-agnostic runner. |
| `run.py` | CLI entry point (`python -m streaming.run`). |
| `config.py` | `StreamingConfig` dataclass + env-var parsing. |
| `result.py` | `StreamingResult` return value. |
| `streaming_loader.py` | Handle-only XLA upload + per-block ship helpers. |
| `_helpers.py` | Skeleton builder, top-level ship, buffer plumbing, host-mem logging. |
| `adapters/base.py` | `ModelAdapter` protocol. |
| `adapters/deepseek_v4_flash.py` | DeepSeek-V4-Flash adapter. |
| `weight_loaders/deepseek_v4_flash.py` | HF weight reader + dequant. |

## Quick start

```bash
source venv/activate
python -m streaming.run
```

Default: DeepSeek-V4-Flash, 43 layers, bsz=8, prompt_len=128, 3 decode tokens,
bf16 weights.

## Environment overrides

| Var | Default | Purpose |
| --- | --- | --- |
| `STREAM_MODE` | `whole_graph` | `whole_graph` (single `torch.compile`) or `layer_eager` (per-layer compile). |
| `STREAM_NUM_LAYERS` | `43` | Truncate the layer stack. Useful for smoke tests. |
| `STREAM_BATCH_SIZE` | `8` | Inference batch. |
| `STREAM_PROMPT_LEN` | `128` | Padded prompt length (left-pad). |
| `STREAM_MAX_NEW_TOKENS` | `3` | Decode steps. |
| `STREAM_EXPERT_DTYPE` | `bf16` | MoE expert pack dtype: `bf16` / `bfp_bf8` / `bfp_bf4`. |
| `STREAM_ATTN_DTYPE` | `bf16` | Attention weight pack dtype (same values). |

## Device DRAM trade-off

`run_streaming` sets the compile option
`enable_const_eval_inputs_to_system_memory=False` so that const-eval inputs
stay in device DRAM instead of getting bounced back to host. Without this
the streaming guarantee on host RAM is broken — every const-eval re-pull
would repopulate the host shadow we just freed.

The cost is that const-eval inputs remain device-resident for the lifetime
of the program. If device DRAM headroom is tight, streaming can OOM on
device. Plan capacity assuming each layer's const-eval input lives on
device until the program ends.

## Adding a new model

Implement `streaming.adapters.base.ModelAdapter` and pass an instance to
`run_streaming`. See `adapters/deepseek_v4_flash.py` for a worked example.
The adapter exposes the model's:

- skeleton constructor + per-layer weight loader
- sharding spec for top-level params and per-block params
- forward signatures (`call_model`, `forward_pre_layers`, `forward_layer`,
  `forward_post_layers`)
- dummy block-input shape for the per-layer flush
- optional MoE swap hook + weight-dtype overrides
