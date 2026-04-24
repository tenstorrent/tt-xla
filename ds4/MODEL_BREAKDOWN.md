# DeepSeek V4 Flash Model Architecture Breakdown

This document breaks down the DeepSeek V4 Flash model into its core components for modular testing and incremental bring-up on TT hardware.

## Table of Contents

1. [High-Level Pipeline](#high-level-pipeline)
2. [Module Dependency Graph](#module-dependency-graph)
3. [Core Components](#core-components)
  - [Embedding Layer](#1-embedding-layer)
  - [RMSNorm](#2-rmsnorm)
  - [Linear Layers](#3-linear-layers)
  - [Rotary Position Embeddings](#4-rotary-position-embeddings-rope)
  - [Attention Mechanisms](#5-attention-mechanisms)
  - [KV Compression](#6-kv-compression)
  - [Mixture of Experts (MoE)](#7-mixture-of-experts-moe)
  - [Hyper-Connections (HC)](#8-hyper-connections-hc)
  - [Transformer Block](#9-transformer-block)
  - [Output Head](#10-output-head)
  - [Multi-Token Prediction (MTP)](#11-multi-token-prediction-mtp)
4. [Testing Strategy](#testing-strategy)
5. [Module Shapes Reference](#module-shapes-reference)

---

## High-Level Pipeline

```
                                    DeepSeek V4 Flash Forward Pass
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                                                                             │
│  input_ids [B, S]                                                                           │
│       │                                                                                     │
│       ▼                                                                                     │
│  ┌─────────────────┐                                                                        │
│  │ ParallelEmbedding│  vocab_size → dim                                                     │
│  └────────┬────────┘                                                                        │
│           │ [B, S, dim]                                                                     │
│           ▼                                                                                 │
│  ┌─────────────────┐                                                                        │
│  │  HC Expand      │  Replicate to hc_mult copies: [B, S, dim] → [B, S, hc_mult, dim]       │
│  └────────┬────────┘                                                                        │
│           │                                                                                 │
│           ▼                                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────────────────────┐    │
│  │                           N × Transformer Blocks                                    │    │
│  │  ┌───────────────────────────────────────────────────────────────────────────────┐  │    │
│  │  │                                                                               │  │    │
│  │  │   ┌──────────┐   ┌──────────┐   ┌───────────┐   ┌──────────┐   ┌──────────┐   │  │    │
│  │  │   │ HC Pre   │──▶│ AttnNorm │──▶│ Attention │──▶│ HC Post  │──▶│ Residual │   │  │    │
│  │  │   └──────────┘   └──────────┘   └───────────┘   └──────────┘   └──────────┘   │  │    │
│  │  │        │                              │                              │        │  │    │
│  │  │        ▼                              ▼                              ▼        │  │    │
│  │  │   ┌──────────┐   ┌──────────┐   ┌───────────┐   ┌──────────┐   ┌──────────┐   │  │    │
│  │  │   │ HC Pre   │──▶│ FFN Norm │──▶│    MoE    │──▶│ HC Post  │──▶│ Residual │   │  │    │
│  │  │   └──────────┘   └──────────┘   └───────────┘   └──────────┘   └──────────┘   │  │    │
│  │  │                                                                               │  │    │
│  │  └───────────────────────────────────────────────────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────────────────────────────┘    │
│           │                                                                                 │
│           ▼                                                                                 │
│  ┌─────────────────┐                                                                        │
│  │  ParallelHead   │  HC reduce + RMSNorm + Linear projection                               │
│  └────────┬────────┘                                                                        │
│           │                                                                                 │
│           ▼                                                                                 │
│      logits [B, vocab_size]                                                                 │
│                                                                                             │
└─────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Module Dependency Graph

```
                           Module Dependencies (Bottom-Up Testing Order)

Level 0 (No dependencies - test first):
├── RMSNorm
├── Linear / ColumnParallelLinear / RowParallelLinear
├── precompute_freqs_cis
└── apply_rotary_emb

Level 1 (Depends on Level 0):
├── ParallelEmbedding (uses F.embedding)
├── act_quant / fp4_act_quant (quantization kernels)
├── rotate_activation (Hadamard transform)
└── get_window_topk_idxs / get_compress_topk_idxs

Level 2 (Depends on Level 1):
├── Compressor (RMSNorm, Linear, apply_rotary_emb, act_quant)
├── Expert (Linear × 3, SwiGLU activation)
└── Gate (Linear, score functions)

Level 3 (Depends on Level 2):
├── Indexer (Compressor, Linear, rotate_activation)
├── MoE (Gate, Expert × n_routed + shared)
└── sparse_attn kernel

Level 4 (Depends on Level 3):
├── Attention (Compressor, Indexer, sparse_attn, all projections)
└── hc_split_sinkhorn (Sinkhorn normalization)

Level 5 (Depends on Level 4):
├── Block (Attention, MoE, HC pre/post)
└── ParallelHead (hc_head, RMSNorm, Linear)

Level 6 (Full model):
├── Transformer (Block × n_layers, embed, head)
└── MTPBlock (Block + embed/head projections)
```

---

## Core Components

### 1. Embedding Layer

**Class:** `ParallelEmbedding`
**Location:** `model.py:83-105`

**Purpose:** Converts input token IDs to dense embeddings, sharded across tensor-parallel ranks.

**Signature:**

```python
ParallelEmbedding(vocab_size: int, dim: int)
# Forward: [B, S] → [B, S, dim]
```

**Key Details:**

- Vocabulary sharded: each rank holds `vocab_size // world_size` rows
- Out-of-range indices masked to zero before `all_reduce`
- Default: vocab_size=129,280, dim=4,096

**Test Independently:**

```python
embed = ParallelEmbedding(vocab_size=1024, dim=512)
x = torch.randint(0, 1024, (2, 16))
out = embed(x)  # [2, 16, 512]
```

---

### 2. RMSNorm

**Class:** `RMSNorm`
**Location:** `model.py:183-196`

**Purpose:** Root Mean Square Layer Normalization (no mean centering).

**Signature:**

```python
RMSNorm(dim: int, eps: float = 1e-6)
# Forward: [*, dim] → [*, dim]
```

**Formula:**

```
x_norm = x / sqrt(mean(x²) + eps) * weight
```

**Key Details:**

- Weight stored in FP32 for numerical stability
- Computation done in FP32, output cast back to input dtype

**Test Independently:**

```python
norm = RMSNorm(dim=512)
x = torch.randn(2, 16, 512, dtype=torch.bfloat16)
out = norm(x)  # [2, 16, 512]
```

---

### 3. Linear Layers

**Classes:** `Linear`, `ColumnParallelLinear`, `RowParallelLinear`
**Location:** `model.py:123-180`

**Purpose:** Linear projections with FP8/FP4/BF16 weight support.


| Class                  | Sharding   | All-Reduce   |
| ---------------------- | ---------- | ------------ |
| `Linear`               | None       | No           |
| `ColumnParallelLinear` | Output dim | No           |
| `RowParallelLinear`    | Input dim  | Yes (output) |


**Weight Formats:**

- **BF16:** Standard `torch.bfloat16` weights
- **FP8:** `torch.float8_e4m3fn` with block-wise scales
- **FP4:** `torch.float4_e2m1fn_x2` with per-32-element scales

**Test Independently:**

```python
linear = Linear(512, 1024)
x = torch.randn(2, 16, 512, dtype=torch.bfloat16)
out = linear(x)  # [2, 16, 1024]
```

---

### 4. Rotary Position Embeddings (RoPE)

**Functions:** `precompute_freqs_cis`, `apply_rotary_emb`
**Location:** `model.py:199-244`

**Purpose:** Position encoding via rotation in complex plane.

**Key Details:**

- Uses YaRN scaling when `original_seq_len > 0`
- Frequency interpolation with smooth ramp between `beta_fast` and `beta_slow`
- Applied only to last `rope_head_dim` dimensions of Q/K

**Signature:**

```python
freqs_cis = precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow)
# freqs_cis: [seqlen, dim//2] complex

apply_rotary_emb(x: [B, S, H, head_dim], freqs_cis, inverse=False)
# Rotates last rope_head_dim dims in-place
```

**Test Independently:**

```python
freqs = precompute_freqs_cis(64, 128, 0, 10000.0, 40, 32, 1)
x = torch.randn(2, 128, 8, 64, dtype=torch.bfloat16)
apply_rotary_emb(x[..., -64:], freqs)  # In-place rotation
```

---

### 5. Attention Mechanisms

**Class:** `Attention`
**Location:** `model.py:436-543` (original), `modified_model.py:601-728`

**Purpose:** Multi-head Latent Attention (MLA) with sliding window and optional KV compression.

#### Attention Types (per layer, controlled by `compress_ratios`):


| Type    | `compress_ratio` | Description                   | Has Indexer |
| ------- | ---------------- | ----------------------------- | ----------- |
| **SWA** | 0                | Sliding Window Attention only | No          |
| **CSA** | 4                | Compressed Sparse Attention   | Yes         |
| **HSA** | 128              | Heavily Compressed Attention  | No          |


**Default layer pattern:** `(0, 0, 4, 128, 4, 128, 4, 0)`

#### Internal Flow:

```
              Attention Forward Pass
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  x [B, S, dim]                                               │
│       │                                                      │
│       ├────────────────────────────────────────┐             │
│       ▼                                        ▼             │
│  ┌─────────┐                              ┌─────────┐        │
│  │  wq_a   │ dim → q_lora_rank            │   wkv   │        │
│  └────┬────┘                              └────┬────┘        │
│       │                                        │             │
│       ▼                                        ▼             │
│  ┌─────────┐                              ┌─────────┐        │
│  │ q_norm  │                              │ kv_norm │        │
│  └────┬────┘                              └────┬────┘        │
│       │                                        │             │
│       ▼                                        ▼             │
│  ┌─────────┐                              ┌─────────┐        │
│  │  wq_b   │ q_lora_rank → n_heads×head   │  RoPE   │        │
│  └────┬────┘                              └────┬────┘        │
│       │                                        │             │
│       ├───────── Q normalization ──────────────┤             │
│       │                                        │             │
│       ▼                                        ▼             │
│  ┌─────────┐                         ┌─────────────────┐     │
│  │  RoPE   │                         │  Window KV Cache│     │
│  └────┬────┘                         └────────┬────────┘     │
│       │                                       │              │
│       │    ┌─────────────────────────────┐    │              │
│       │    │ Compressed KV (if ratio > 0)│    │              │
│       │    │  ┌───────────────────────┐  │    │              │
│       │    │  │    Compressor         │  │    │              │
│       │    │  │    (gated pooling)    │  │    │              │
│       │    │  └───────────────────────┘  │    │              │
│       │    │           │                 │    │              │
│       │    │  ┌───────────────────────┐  │    │              │
│       │    │  │ Indexer (CSA only)    │  │    │              │
│       │    │  │ (top-k selection)     │  │    │              │
│       │    │  └───────────────────────┘  │    │              │
│       │    └─────────────┬───────────────┘    │              │
│       │                  │                    │              │
│       ▼                  ▼                    ▼              │
│  ┌───────────────────────────────────────────────────┐       │
│  │               sparse_attn kernel                  │       │
│  │  (Q × K^T × scale + attn_sink) → softmax → × V    │       │
│  └───────────────────────┬───────────────────────────┘       │
│                          │                                   │
│                          ▼                                   │
│                    ┌───────────┐                             │
│                    │Inverse RoPE│                            │
│                    └─────┬─────┘                             │
│                          │                                   │
│                          ▼                                   │
│  ┌─────────────────────────────────────────────────────┐     │
│  │        Grouped Low-Rank Output Projection           │     │
│  │   reshape → wo_a (einsum) → wo_b (all_reduce)       │     │
│  └─────────────────────────┬───────────────────────────┘     │
│                            │                                 │
│                            ▼                                 │
│                      output [B, S, dim]                      │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**Key Projections:**


| Layer  | Shape                                              | Purpose                       |
| ------ | -------------------------------------------------- | ----------------------------- |
| `wq_a` | [dim, q_lora_rank]                                 | Q low-rank down-projection    |
| `wq_b` | [q_lora_rank, n_heads×head_dim]                    | Q low-rank up-projection      |
| `wkv`  | [dim, head_dim]                                    | Shared KV projection (latent) |
| `wo_a` | [n_groups, o_lora_rank, head_dim×n_heads/n_groups] | O grouped low-rank            |
| `wo_b` | [n_groups×o_lora_rank, dim]                        | O final projection            |


**Test Independently (CSA):**

```python
args = ModelArgs(dim=512, n_heads=8, head_dim=64, q_lora_rank=128,
                 compress_ratios=(0, 0, 4))
attn = Attention(layer_id=2, args=args)  # CSA layer
x = torch.randn(2, 64, 512, dtype=torch.bfloat16)
out = attn(x, start_pos=0)  # [2, 64, 512]
```

---

### 6. KV Compression

**Class:** `Compressor`
**Location:** `model.py:279-377`

**Purpose:** Compresses KV cache via learned gated pooling over consecutive tokens.

#### Compression Modes:


| Ratio | Overlap | Description                                     |
| ----- | ------- | ----------------------------------------------- |
| 4     | Yes     | CSA - overlapping windows for smooth boundaries |
| 128   | No      | HSA - non-overlapping, aggressive compression   |


**Internal Flow:**

```
x [B, S, dim]
    │
    ├──▶ wkv [B, S, coff×head_dim]  ──┐
    │                                 │
    └──▶ wgate [B, S, coff×head_dim] ─┼──▶ softmax(score + APE) × kv
                                      │
                                      ▼
                               [B, S//ratio, head_dim]
                                      │
                                      ▼
                                   RMSNorm
                                      │
                                      ▼
                                    RoPE
                                      │
                                      ▼
                          (optional) Hadamard rotation
                                      │
                                      ▼
                               KV cache update
```

**Key Parameters:**

- `ape`: Absolute position embedding for compression [ratio, coff×head_dim]
- `wkv`: Computes compressed KV values
- `wgate`: Computes gating scores

**Test Independently:**

```python
args = ModelArgs(dim=512, head_dim=64, rope_head_dim=16)
comp = Compressor(args, compress_ratio=4, head_dim=64)
comp.kv_cache = torch.zeros(4, 256, 64)  # Assign cache
comp.freqs_cis = precompute_freqs_cis(16, 256, ...)
x = torch.randn(2, 128, 512, dtype=torch.bfloat16)
kv_compressed = comp(x, start_pos=0)  # [2, 32, 64]
```

---

### 7. Mixture of Experts (MoE)

**Classes:** `Gate`, `Expert`, `MoE`
**Location:** `model.py:546-645`

**Purpose:** Route tokens to specialized expert FFNs.

#### MoE Flow:

```
x [B×S, dim]
    │
    ▼
┌─────────────────┐
│      Gate       │  scores = score_func(x @ weight + bias)
│                 │  indices = topk(scores)
│  Hash routing:  │  weights = gather(scores, indices) * route_scale
│  tid → expert   │
└────────┬────────┘
         │
         ▼
    ┌────────────────────────────────────────────────┐
    │           Expert Selection & Computation        │
    │                                                 │
    │  ┌──────────┐  ┌──────────┐      ┌──────────┐  │
    │  │ Expert 0 │  │ Expert 1 │ ...  │ Expert N │  │
    │  │ (SwiGLU) │  │ (SwiGLU) │      │ (SwiGLU) │  │
    │  └────┬─────┘  └────┬─────┘      └────┬─────┘  │
    │       │             │                 │        │
    │       └─────────────┼─────────────────┘        │
    │                     ▼                          │
    │             weighted sum                       │
    └────────────────────┬───────────────────────────┘
                         │
                         ▼
                   + shared_expert(x)
                         │
                         ▼
                   output [B×S, dim]
```

#### Expert (SwiGLU FFN):

```python
# Expert computation
gate = w1(x)            # [B, dim] → [B, inter_dim]
up = w3(x)              # [B, dim] → [B, inter_dim]
x = silu(gate) * up     # SwiGLU activation
x = w2(x)               # [B, inter_dim] → [B, dim]
```

#### Gating Score Functions:


| Function     | Formula                  |
| ------------ | ------------------------ |
| softmax      | `softmax(scores)`        |
| sigmoid      | `sigmoid(scores)`        |
| sqrtsoftplus | `sqrt(softplus(scores))` |


**Key Parameters:**

- `n_routed_experts`: 8 (default)
- `n_activated_experts`: 2 (top-k)
- `n_shared_experts`: 1 (always activated)
- `route_scale`: 1.0

**Test Expert Independently:**

```python
expert = Expert(dim=512, inter_dim=2048)
x = torch.randn(2, 16, 512, dtype=torch.bfloat16)
out = expert(x)  # [2, 16, 512]
```

**Test Gate Independently:**

```python
args = ModelArgs(dim=512, n_routed_experts=8, n_activated_experts=2)
gate = Gate(layer_id=0, args=args)
x = torch.randn(32, 512)  # [B×S, dim]
weights, indices = gate(x)  # weights: [32, 2], indices: [32, 2]
```

---

### 8. Hyper-Connections (HC)

**Location:** `Block.hc_pre`, `Block.hc_post` in `model.py:674-687`

**Purpose:** Replace simple residuals with learned multi-copy mixing.

#### HC Mechanism:

Instead of `x = x + f(x)` (standard residual), HC maintains `hc_mult` copies:

```
                    Hyper-Connections Flow
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  x [B, S, hc_mult, dim]                                      │
│         │                                                    │
│         ▼                                                    │
│  ┌──────────────────────────────────────────────────────┐    │
│  │                   hc_pre                              │    │
│  │                                                       │    │
│  │  1. Flatten: [B, S, hc_mult × dim]                    │    │
│  │  2. Compute mixes via hc_fn projection                │    │
│  │  3. Split into pre, post, comb via Sinkhorn           │    │
│  │  4. Weighted sum: y = sum(pre_i × x_i)                │    │
│  │                                                       │    │
│  │  Output: y [B, S, dim], post, comb                    │    │
│  └───────────────────────┬──────────────────────────────┘    │
│                          │                                   │
│                          ▼                                   │
│                   Attention / MoE                            │
│                          │                                   │
│                          ▼                                   │
│  ┌──────────────────────────────────────────────────────┐    │
│  │                   hc_post                             │    │
│  │                                                       │    │
│  │  y = post × f(x) + sum(comb_ij × residual_j)          │    │
│  │                                                       │    │
│  │  Output: [B, S, hc_mult, dim]                         │    │
│  └───────────────────────┬──────────────────────────────┘    │
│                          │                                   │
│                          ▼                                   │
│              output [B, S, hc_mult, dim]                     │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

**Key Parameters:**

- `hc_mult`: 4 (number of hidden state copies)
- `hc_sinkhorn_iters`: 20 (Sinkhorn normalization iterations)
- `hc_eps`: 1e-6

**HC Parameters per Block:**


| Parameter       | Shape                              | Purpose                      |
| --------------- | ---------------------------------- | ---------------------------- |
| `hc_attn_fn`    | [(2+hc_mult)×hc_mult, hc_mult×dim] | Attention HC mixing weights  |
| `hc_ffn_fn`     | [(2+hc_mult)×hc_mult, hc_mult×dim] | FFN HC mixing weights        |
| `hc_attn_base`  | [(2+hc_mult)×hc_mult]              | Attention HC bias            |
| `hc_ffn_base`   | [(2+hc_mult)×hc_mult]              | FFN HC bias                  |
| `hc_attn_scale` | [3]                                | Attention HC scaling factors |
| `hc_ffn_scale`  | [3]                                | FFN HC scaling factors       |


---

### 9. Transformer Block

**Class:** `Block`
**Location:** `model.py:648-701`

**Purpose:** Single transformer layer with HC-wrapped attention and MoE.

**Structure:**

```
x [B, S, hc_mult, dim]
    │
    ├──▶ residual
    │
    ▼
hc_pre(x, hc_attn_fn, hc_attn_scale, hc_attn_base)
    │
    ▼
attn_norm (RMSNorm)
    │
    ▼
Attention(x, start_pos)
    │
    ▼
hc_post(x, residual, post, comb)
    │
    ├──▶ residual
    │
    ▼
hc_pre(x, hc_ffn_fn, hc_ffn_scale, hc_ffn_base)
    │
    ▼
ffn_norm (RMSNorm)
    │
    ▼
MoE(x, input_ids)
    │
    ▼
hc_post(x, residual, post, comb)
    │
    ▼
output [B, S, hc_mult, dim]
```

**Test Independently:**

```python
args = ModelArgs(dim=512, n_heads=8, head_dim=64, ...)
block = Block(layer_id=0, args=args)
x = torch.randn(2, 64, args.hc_mult, 512, dtype=torch.bfloat16)
input_ids = torch.randint(0, 1000, (2, 64))
out = block(x, start_pos=0, input_ids=input_ids)  # [2, 64, 4, 512]
```

---

### 10. Output Head

**Class:** `ParallelHead`
**Location:** `model.py:704-736`

**Purpose:** Convert hidden states to logits over vocabulary.

**Flow:**

```
x [B, S, hc_mult, dim]
    │
    ▼
hc_head (HC reduction to single copy)
    │
    ▼
RMSNorm
    │
    ▼
Linear (dim → vocab_size)
    │
    ▼
all_gather (if world_size > 1)
    │
    ▼
logits [B, vocab_size]
```

**Test Independently:**

```python
head = ParallelHead(vocab_size=1024, dim=512)
x = torch.randn(2, 64, 4, 512, dtype=torch.bfloat16)
norm = RMSNorm(512)
hc_fn = torch.randn(4, 4*512)
hc_scale = torch.randn(1)
hc_base = torch.randn(4)
logits = head(x, hc_fn, hc_scale, hc_base, norm)  # [2, 1024]
```

---

### 11. Multi-Token Prediction (MTP)

**Class:** `MTPBlock`
**Location:** `model.py:739-767`

**Purpose:** Speculative decoding - predict multiple future tokens.

**Flow:**

```
x [B, S, hc_mult, dim]  (from main transformer)
input_ids [B, S]
    │
    ├──▶ embed(input_ids) → enorm → e_proj
    │
    └──▶ hnorm → h_proj
         │
         ▼
    e_proj + h_proj
         │
         ▼
    Block.forward (attention + MoE)
         │
         ▼
    ParallelHead → logits
```

**Additional Parameters:**


| Layer    | Purpose                      |
| -------- | ---------------------------- |
| `e_proj` | Project embedding for MTP    |
| `h_proj` | Project hidden state for MTP |
| `enorm`  | Normalize embeddings         |
| `hnorm`  | Normalize hidden states      |


---

## Testing Strategy

### Phase 1: Primitive Operations

Test these first as they have no dependencies:

1. `RMSNorm` - numerical stability test
2. `Linear` variants - shape and dtype verification
3. `precompute_freqs_cis` - rotary embedding generation
4. `apply_rotary_emb` - rotation correctness

### Phase 2: Basic Components

Test with random inputs:

1. `ParallelEmbedding` - lookup and sharding
2. `Expert` (SwiGLU FFN) - activation and projection
3. `Gate` - routing score computation

### Phase 3: Compression & Indexing

Test compression logic:

1. `Compressor` - pooling and cache update
2. `Indexer` - top-k selection
3. `sparse_attn` kernel - attention computation

### Phase 4: Attention Variants

Test each attention type separately:

1. SWA (compress_ratio=0) - sliding window only
2. CSA (compress_ratio=4) - with Indexer
3. HSA (compress_ratio=128) - without Indexer

### Phase 5: Full Blocks

Test assembled components:

1. `MoE` - gating + expert routing
2. `Block` - attention + MoE + HC
3. `ParallelHead` - logit computation

### Phase 6: Full Model

1. `Transformer` - end-to-end forward
2. `MTPBlock` - speculative decoding

---

## Module Shapes Reference

Default configuration (`ModelArgs` defaults):


| Parameter             | Default Value                |
| --------------------- | ---------------------------- |
| `vocab_size`          | 129,280                      |
| `dim`                 | 4,096                        |
| `n_layers`            | 7                            |
| `n_heads`             | 64                           |
| `head_dim`            | 512                          |
| `rope_head_dim`       | 64                           |
| `q_lora_rank`         | 1,024                        |
| `o_lora_rank`         | 1,024                        |
| `o_groups`            | 8                            |
| `moe_inter_dim`       | 4,096                        |
| `n_routed_experts`    | 8                            |
| `n_activated_experts` | 2                            |
| `window_size`         | 128                          |
| `hc_mult`             | 4                            |
| `compress_ratios`     | (0, 0, 4, 128, 4, 128, 4, 0) |


### Tensor Shapes (B=batch, S=seq_len):


| Location              | Shape                             |
| --------------------- | --------------------------------- |
| Input IDs             | [B, S]                            |
| After embed           | [B, S, dim]                       |
| After HC expand       | [B, S, hc_mult, dim]              |
| Q before reshape      | [B, S, n_heads × head_dim]        |
| Q after reshape       | [B, S, n_heads, head_dim]         |
| KV latent             | [B, S, head_dim]                  |
| KV cache (window)     | [B, window_size, head_dim]        |
| KV cache (compressed) | [B, max_seq_len//ratio, head_dim] |
| Attention output      | [B, S, n_heads, head_dim]         |
| After wo_b            | [B, S, dim]                       |
| After block           | [B, S, hc_mult, dim]              |
| Logits                | [B, vocab_size]                   |


---

## Quick Reference: Module to Test File Location


| Module              | Original File  | Modified File           |
| ------------------- | -------------- | ----------------------- |
| `ParallelEmbedding` | `model.py:83`  | `modified_model.py:222` |
| `Linear`            | `model.py:123` | `modified_model.py:254` |
| `RMSNorm`           | `model.py:183` | `modified_model.py:307` |
| `Compressor`        | `model.py:279` | `modified_model.py:416` |
| `Indexer`           | `model.py:380` | `modified_model.py:529` |
| `Attention`         | `model.py:436` | `modified_model.py:601` |
| `CSAAttention`      | N/A            | `modified_model.py:731` |
| `HSAAttention`      | N/A            | `modified_model.py:752` |
| `Gate`              | `model.py:546` | N/A                     |
| `Expert`            | `model.py:587` | N/A                     |
| `MoE`               | `model.py:609` | N/A                     |
| `Block`             | `model.py:648` | N/A                     |
| `ParallelHead`      | `model.py:704` | N/A                     |
| `MTPBlock`          | `model.py:739` | N/A                     |
| `Transformer`       | `model.py:770` | N/A                     |


