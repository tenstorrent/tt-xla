# Matmul Compute Kernel Config Sweep for Z-Image Transformer

## Approach

Systematically evaluated different `WormholeComputeKernelConfig` settings for matmul and
normalization ops in the Z-Image DiT transformer. The model has 34 transformer blocks, each
containing:
- 4 attention matmuls (Q/K/V projections + output projection)
- 3 FFN matmuls (w1 with fused SiLU, w3, w2)
- 4 RMS norms (pre-attention, post-attention, pre-FFN, post-FFN)
- 1 adaLN linear (for modulated blocks)

Total: ~238 matmuls + ~136 norms per forward pass.

### Config Knobs Tested

| Knob | Description | Values tested |
|------|-------------|---------------|
| `math_fidelity` | Precision of multiply-accumulate operations | HiFi4 (most precise), HiFi3, HiFi2, LoFi (fastest) |
| `fp32_dest_acc_en` | Accumulate partial sums in FP32 instead of BF16 | True (default), False |
| `packer_l1_acc` | Keep packer intermediate data in L1 instead of DRAM | True, False (default for matmul) |
| `math_approx_mode` | Use approximate math for transcendental functions | Only relevant for norms |

### Methodology

- Single Blackhole p150 device, 13x10 core grid
- Model weights loaded once; configs monkey-patched at module level between runs
- Each config: 1 warm-up iteration + 3 timed iterations, average of timed runs reported
- PCC measured against PyTorch FP32 CPU reference output
- Two config categories tested independently: COMPUTE_CONFIG (matmuls, adaLN linears) and
  COMPUTE_CONFIG_NORM (rms_norm ops)

## Results

| # | Config | Matmul fidelity | fp32_acc | packer_l1 | Norm fidelity | Avg (ms) | PCC | Delta |
|---|--------|----------------|----------|-----------|---------------|----------|-----|-------|
| 1 | **baseline** | HiFi4 | True | False | HiFi4 | 15335.8 | 0.998898 | 0.0% |
| 2 | HiFi4_no_fp32acc | HiFi4 | **False** | False | HiFi4 (no fp32acc) | 15245.9 | 0.995713 | -0.6% |
| 3 | HiFi2_fp32acc | **HiFi2** | True | False | HiFi2 | 15190.5 | 0.997606 | -0.9% |
| 4 | HiFi2_no_fp32acc | **HiFi2** | **False** | False | HiFi2 (no fp32acc) | 15093.1 | 0.995185 | -1.6% |
| 5 | HiFi2mm_HiFi4norm | **HiFi2** | True | False | HiFi4 | 15190.1 | 0.998575 | -1.0% |
| 6 | HiFi2mm_noacc_HiFi4norm | **HiFi2** | **False** | False | HiFi4 | 15099.6 | 0.996066 | -1.5% |
| 7 | LoFi_fp32acc | **LoFi** | True | False | LoFi | 15157.2 | 0.991366 | -1.2% |
| 8 | LoFi_no_fp32acc | **LoFi** | **False** | False | LoFi (no fp32acc) | 15029.0 | 0.991548 | -2.0% |
| 9 | LoFi_mm_HiFi4norm | **LoFi** | True | False | HiFi4 | 15160.6 | 0.995633 | -1.1% |
| 10 | HiFi4_packer_l1 | HiFi4 | True | **True** | HiFi4 | 15191.4 | **0.999423** | -0.9% |
| **11** | **HiFi2_packer_l1_HiFi4n** | **HiFi2** | True | **True** | HiFi4 | **15092.1** | **0.999289** | **-1.6%** |
| 12 | HiFi3_fp32acc | HiFi3 | True | False | HiFi3 | 15270.5 | 0.998845 | -0.4% |
| 13 | HiFi3mm_HiFi4norm | HiFi3 | True | False | HiFi4 | 15271.4 | 0.998866 | -0.4% |

## Key Findings

### 1. Best overall: HiFi2 matmul + packer_l1 + HiFi4 norm (#11)

**15092ms, PCC 0.999289 — 1.6% faster with BETTER accuracy than baseline.**

```python
COMPUTE_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)
COMPUTE_CONFIG_NORM = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)
```

This config works because:
- **HiFi2 on matmuls** reduces precision in individual multiply-accumulate operations but
  the FP32 accumulation (`fp32_dest_acc_en=True`) compensates by keeping the running sum accurate.
- **packer_l1_acc=True** keeps packer data in L1, reducing DRAM bandwidth pressure. This
  also explains the PCC improvement: L1 has lower latency and no quantization, so intermediate
  results are more accurate.
- **HiFi4 on norms** preserves numerical stability where it matters most (normalization involves
  mean, variance, rsqrt — operations sensitive to precision).

### 2. packer_l1_acc is the surprising winner

Enabling `packer_l1_acc=True` on matmuls (#10, #11) consistently improved both speed AND accuracy:
- #10 (HiFi4 + packer_l1): PCC 0.999423 — best PCC of all configs
- #11 (HiFi2 + packer_l1): PCC 0.999289 — second best PCC, fastest with good accuracy

This is a free improvement with no downside.

### 3. math_fidelity impact: HiFi4 > HiFi3 >> HiFi2 >> LoFi

| Fidelity | Speed gain | PCC impact |
|----------|-----------|------------|
| HiFi4 (baseline) | 0% | 0.998898 |
| HiFi3 | -0.4% | -0.005% (negligible) |
| HiFi2 | -0.9% | -0.13% (acceptable) |
| LoFi | -1.2% | -0.75% (significant) |

HiFi3 provides almost no speed benefit over HiFi4 on Blackhole. HiFi2 provides a modest 0.9%
speedup. LoFi provides the most speedup but with significant PCC degradation.

### 4. fp32_dest_acc_en: significant PCC impact

Disabling FP32 accumulation saves ~0.6% latency but drops PCC by 0.3-0.4%:
- HiFi4: 0.998898 → 0.995713 (fp32 off)
- HiFi2: 0.997606 → 0.995185 (fp32 off)

This is expected: BF16 accumulation truncates each partial sum, and with K=3840 (120 tiles),
rounding errors accumulate over many additions.

### 5. Mixed configs recover PCC cheaply

Using HiFi4 for norms while reducing matmul fidelity recovers most of the PCC loss:
- HiFi2 uniform: PCC 0.997606
- HiFi2 matmul + HiFi4 norm: PCC 0.998575 (+0.1%)
- LoFi uniform: PCC 0.991366
- LoFi matmul + HiFi4 norm: PCC 0.995633 (+0.4%)

Norm precision matters more than matmul precision for overall accuracy because normalization
errors propagate multiplicatively through every subsequent operation.

### 6. LoFi is not worth it

LoFi saves only 0.3% more than HiFi2 but loses 0.6% PCC. The Pareto front clearly favors
HiFi2 over LoFi.

## Recommended Configuration

**For best perf/accuracy trade-off (config #11):**

```python
COMPUTE_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi2,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)
COMPUTE_CONFIG_NORM = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    math_approx_mode=False,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)
```

- **1.6% faster** than baseline (15092ms vs 15336ms)
- **Better PCC** than baseline (0.999289 vs 0.998898)
- Combined with `nlp_create_qkv_heads` + `addcmul` from the tt_dit ops report, total
  improvement would be approximately **3%** (additive, as they affect different operations)

**For maximum accuracy (config #10):**

```python
COMPUTE_CONFIG = ttnn.WormholeComputeKernelConfig(
    math_fidelity=ttnn.MathFidelity.HiFi4,
    fp32_dest_acc_en=True,
    packer_l1_acc=True,
)
```

- **0.9% faster** with PCC 0.999423 (best accuracy of all configs)
- The only change from baseline is adding `packer_l1_acc=True` to matmuls

## Pareto Front (speed vs accuracy)

```
PCC
0.9994 |  *#10 (HiFi4+packer_l1)
0.9993 |      *#11 (HiFi2+packer_l1+HiFi4n) ← RECOMMENDED
0.9989 |  *baseline
0.9989 |            *#13 (HiFi3mm+HiFi4n)
0.9986 |            *#5 (HiFi2mm+HiFi4n)
0.9976 |            *#3 (HiFi2)
0.9961 |                *#6 (HiFi2mm noacc+HiFi4n)
0.9957 |  *#2 (HiFi4 no acc)
0.9956 |            *#9 (LoFi+HiFi4n)
0.9952 |                *#4 (HiFi2 no acc)
0.9915 |            *#8 (LoFi no acc)
0.9914 |            *#7 (LoFi)
       +----+----+----+----+----+
       0   -0.5  -1.0 -1.5 -2.0  % speedup
```

Configs #10 and #11 dominate the Pareto front — they are faster AND more accurate than baseline.
