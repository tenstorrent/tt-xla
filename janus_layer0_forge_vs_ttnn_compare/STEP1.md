# Step 1: Forge vs TTNN vs CPU (saved tensors)

Save three `[3, B, S, H]` stacks and compare offline.

| Artifact | Script | Where it runs |
|----------|--------|----------------|
| `cpu_stacked_pro1b.pt` | `capture_cpu.py` | tt-xla venv |
| `forge_stacked_pro1b.pt` | `capture_forge.py` | tt-xla venv + TT device |
| `ttnn_stacked_pro1b.pt` | `capture_ttnn.py` | tt-metal `python_env` + TT device |

## One-shot (recommended)

```bash
cd /proj_sw/user_dev/ctr-akannan/31_may_yyz/tt-xla
bash janus_layer0_forge_vs_ttnn_compare/run_step1_forge_vs_ttnn.sh
```

Log: `janus_layer0_step1_forge_vs_ttnn.log`

## Manual (three captures + compare)

### 0) Env

```bash
export TTXLA_ROOT=/proj_sw/user_dev/ctr-akannan/31_may_yyz/tt-xla
export TTMETAL_ROOT=/proj_sw/user_dev/ctr-akannan/31_may_tt_metal/tt-metal
export JANUS_LAYER0_FIXTURE_DIR=${TTXLA_ROOT}/janus_logs/layer0_tensors/Pro_1B
export JANUS_TTNN_GRAPH_DIR=${TTMETAL_ROOT}/janus_layer0_ln_attn_no_dep_codegen/graph_0
export JANUS_COMPARE_ARTIFACTS_DIR=${TTXLA_ROOT}/janus_layer0_forge_vs_ttnn_compare/artifacts
```

### 1) CPU (tt-xla venv)

```bash
cd $TTXLA_ROOT && source venv/bin/activate
python janus_layer0_forge_vs_ttnn_compare/capture_cpu.py
```

### 2) Forge live (tt-xla venv, device)

```bash
cd $TTXLA_ROOT && source venv/bin/activate
python janus_layer0_forge_vs_ttnn_compare/capture_forge.py
```

### 3) TTNN export (tt-metal venv, device)

```bash
cd $TTMETAL_ROOT
$TTMETAL_ROOT/python_env/bin/python \
  $TTXLA_ROOT/janus_layer0_forge_vs_ttnn_compare/capture_ttnn.py \
  --graph-dir $JANUS_TTNN_GRAPH_DIR
```

### 4) Compare (no device)

```bash
cd $TTXLA_ROOT && source venv/bin/activate
python janus_layer0_forge_vs_ttnn_compare/compare.py
```

## Read the matrix

| Report section | Meaning |
|----------------|---------|
| **Forge vs TTNN** | Step 1 primary — does export match live Forge? |
| **CPU vs Forge** | Should match Experiment A (~0.77 on `self_attn`) |
| **CPU vs TTNN** | Should match `cpu_vs_tt.log` (~0.99 on `self_attn`) |

**`self_attn` Forge vs TTNN ~1.0** → export ≈ live Forge; focus on **why Forge ≠ CPU** on tt-xla.

**`self_attn` Forge vs TTNN low** → **live Forge ≠ TTNN**; fix graph/inputs/trace before attn op debug.
