# Qwen3.6-27B bringup — overview

Status of running **Qwen/Qwen3.6-27B** (`Qwen3NextForCausalLM`, hybrid Gated
DeltaNet + Gated Attention + vision encoder) under the TT vLLM plugin.

- **Branch:** `vzeljkovic/qwen3.6` (commit `gated-delta-attention`).
- **Reference machine:** `bh-glx-110-c01u08` (Blackhole Galaxy), mesh `[4, 8]`.
- **Run harness:** `service.sh` (server) + `client.py` (chat client) in this dir.
- **Deep dive on DeltaNet:** see [`DELTA_NET_BRINGUP.md`](./DELTA_NET_BRINGUP.md).

The model has **three independent blockers** for full multimodal serving:
**(1) Gated Delta (linear) attention**, **(2) M-RoPE**, **(3) the vision
encoder / multimodal plumbing**. Only (1) has been worked on so far.

---

## What's done

The DeltaNet spec in `DELTA_NET_BRINGUP.md` has been implemented (first pass):

| Area | File | What |
|---|---|---|
| Pure-PyTorch DeltaNet override | `vllm_tt/tt_gated_delta_net.py` (new, ~310 lines) | `TTGatedDeltaNetAttention` replacing vLLM's `gdn_attention_core` custom op — causal conv1d, gating, delta-rule recurrence in plain ops so dynamo shape-prop is correct. |
| Register override | `vllm_tt/overrides.py` | `(GatedDeltaNetAttention, tt_gated_delta_net_module)` added to `ISINSTANCE_OVERRIDES`. |
| Mamba KV-cache spec + allocation | `vllm_tt/model_runner.py` | `MambaSpec` branch in `get_kv_cache_spec` and `initialize_kv_cache` to allocate `(conv_state, ssm_state)` per DeltaNet layer. |
| Platform fix | `vllm_tt/platform.py` | `current_device()` override so `MambaBase.__init__` doesn't crash on `None()`. |
| M-RoPE (partial) | `vllm_tt/overrides.py` | `TTRotaryEmbedding` now accepts `RotaryEmbeddingBase`, so `MRotaryEmbedding` is handled — **but treated as 1-D** (correct for text-only, not for image/video positions). |
| Run scripts | `service.sh`, `client.py` | `max-model-len 128`, `max-num-seqs 1`, TP on 8 chips. Tiny seq len because the naive token-by-token recurrence unrolls into a huge FX graph at full length. |

**Current state:** the latest run (`vllm_5_3.log`) gets through model load and
compile (reaches `embed_tokens`), then engine-core init fails with a PJRT
execute error. The newest machine logs also show hardware-topology
`TT_FATAL: ... connects to a remote mmio device` errors — needs confirming
whether that's a machine/mesh config issue vs. our code.

---

## What's left

### 1. Gated Delta attention (closest to working)
- Debug the current PJRT execute / engine-core init failure on a tiny prompt.
- **Verify numerics** of the naive recurrence against the FLA reference on a
  CUDA box (bf16 tolerance, not bit-exact).
- Replace the naive `for t in range(T)` loop with the **chunkwise-parallel**
  formulation so `max-model-len` can go back up (the naive loop is why it's
  pinned at 128 — see note in `service.sh`).
- Scale up: prefill+decode, then multiple concurrent requests (needs real
  `GDNAttentionMetadata` instead of the single-request shortcut).

### 2. M-RoPE
- Current override flattens positions to 1-D — fine for text-only, **wrong for
  image/video**. Needs a real `TTMRotaryEmbedding` that handles 3-D
  (temporal/height/width) position ids before multimodal works.

### 3. Multimodal / vision encoder
- Not started. Vision encoder compile through the `tt` backend + multimodal
  input plumbing (image preprocessing, embeddings merge). Largest remaining
  chunk of work.

---

## Realistic milestones
1. **Text-only chat** on Qwen3.6-27B — needs (1) finished + numerics verified.
   Est. ~1–2 weeks focused (per `DELTA_NET_BRINGUP.md §10`).
2. **Multimodal** — needs (2) M-RoPE 3-D + (3) vision encoder on top. Larger,
   separate effort.
