---
name: shard-spec-gen
description: Generates shard specs and minimal pytest tests for HuggingFace model components on Tenstorrent hardware (tt-xla). Use this skill whenever the user wants to shard a model, generate a shard spec, distribute a layer across devices, write a multi-chip test, or asks about tensor parallelism, Megatron sharding, mesh configuration, or how to split attention/MLP/MoE weights across TT devices. Trigger even if the user just names a model and mentions devices or sharding without being explicit.
allowed-tools: Bash Read Grep Glob Write Edit Fetch
---

# Shard Spec Generator for tt-xla

Generates shard specs and pytest tests for HuggingFace model components on Tenstorrent hardware.

## Modes

Infer from the user's request — don't ask unless ambiguous.

- **Verified** (default): runs smoke test, loads layer minimally, generates and runs the test.
- **Theoretical**: no execution. Use when user says "just generate", "no hardware", "theoretical", etc. Output is clearly marked unverified.

---

## Step 1: Gather information

**Do not proceed until 1, 2, and 4 are answered. Ask for all missing ones at once. Strategy (3) is optional.**

1. **Model** — HuggingFace model ID (e.g. `meta-llama/Llama-3.1-8B`)
2. **Component** — `attention`, `mlp`, `moe`, `full model`, etc.
3. **Strategy** *(optional)* — `megatron`, `fsdp`, `data parallel`, etc. If not provided, auto-select per the strategy selection logic below.
4. **Hardware**
   - `llmbox` — 8 chips, mesh `(1, 8)`, axes `("batch", "model")`
   - `galaxy` — 32 chips, mesh `(4, 8)`, axes `("batch", "model")`
   - `single_device` — 1 chip, no sharding needed
   - Raw count: 2→`(1,2)`, 4→`(1,4)`, 8→`(1,8)`, 32→`(4,8)`
   - Fall back to batch splitting if heads aren't divisible by the model-axis size

---

## Strategy Selection Logic

When the user does **not** specify a strategy, apply this rule:

> **Default to Megatron** (tensor parallelism). It produces fewer collective communication operations (CCLs) than FSDP, making it more efficient when it fits in memory. **Warn the user** that Megatron shards weights across the model axis and may cause OOM on very large models — if they hit OOM, switch to FSDP.

Decision tree:
1. User provided a strategy → use it as-is, skip auto-selection.
2. User mentioned OOM using **Megatron** → try **FSDP**.
3. Component is `moe` or model has MoE layers → see **MoE / Expert Parallelism note** below, then apply rule 4.
4. Otherwise → use **Megatron**, and emit the following note to the user:

> **Strategy chosen: Megatron** (tensor parallelism) — selected because it generates fewer CCL ops than FSDP. If you hit OOM when running the full model, ask me to regenerate with FSDP.

**MoE / Expert Parallelism (EP) note:** When a model has MoE layers, Expert Parallelism is an additional strategy to consider alongside Megatron and FSDP. EP assigns whole experts to devices (each device owns `num_experts / num_devices` experts) rather than sharding each expert's weight matrices. This avoids intra-expert weight communication but introduces All-to-All token routing between devices.

- If `num_experts` is cleanly divisible by `num_devices` → flag EP as viable and mention it to the user as an alternative worth exploring.
- Default is still Megatron (it also works on MoE by applying column/row-parallel rules to each expert's weights).
- See `references/expert_parallel_example.py` for the EP shard-spec pattern (to be filled in).

Always state the chosen strategy and the reason **before** generating the spec.

---

## Step 2: Verify the environment

**Skip entirely in theoretical mode.**

```bash
[ -d venv ] && source venv/activate
python3 .claude/skills/shard-spec-gen/references/smoke_test.py
```

Passing output: `OK — <N> devices, mesh shape (1, N)` → proceed in verified mode.

**If the smoke test fails**, show the error and ask the user:
> 1. **Fix the environment** — I'll re-run when ready
> 2. **Provide access instructions** — Docker command, remote machine, build steps
> 3. **Generate a theoretical spec** — no execution, attribute paths estimated from docs

Re-run on options 1/2 until passing. Switch to theoretical mode on option 3. **Do not proceed until resolved.**

---

## Step 3: Find the ModelLoader

Search locally first (works in all modes):
```bash
find third_party/tt_forge_models -name "loader.py" | xargs grep -l "ModelLoader" | head -20
```

**ModelLoader found**: check if it accepts `num_layers` (read `__init__` signature) and list variants:
```bash
python3 -c "from third_party.tt_forge_models.<model>.<task>.pytorch.loader import ModelLoader, ModelVariant; print(list(ModelVariant))"
```

**No ModelLoader**: use `AutoConfig.from_pretrained("<hf-model-id>")`. On `GatedRepoError`/401, ask the user to `export HF_TOKEN=<token>` and wait before retrying. If unsuccessful ask user for instructions to load the model.

---

## Step 4: Verify layer names

Always load minimally. Confirm exact attribute paths and config values before writing any spec.

Fill in the placeholders in `references/inspect_layer.py` and run it:
```bash
[ -d venv ] && source venv/activate
python3 .claude/skills/shard-spec-gen/references/inspect_layer.py
```

**Theoretical mode**: use Fetch on `https://huggingface.co/<org>/<model>/blob/main/modeling_<arch>.py` to extract attribute names. Note estimated paths in the test.

---

## Step 5: Generate the shard spec

Apply rules from `references/sharding_rules.md` using the attribute names confirmed in Step 4.

---

## Step 6: Generate the test file

Use `references/test_template.py` as the template. Save to `tests/torch/graphs/test_<model>_<component>.py`. Fill in the confirmed attribute paths, config values, and shard spec from Steps 4–5.

---

## Step 7: Run the test

**Verified mode only.**

```bash
[ -d venv ] && source venv/activate
TEST=tests/torch/graphs/test_<model>_<component>.py
ARCH=llmbox  # or galaxy
LOGFILE="${TEST%.py}_${ARCH}.log"
TTXLA_LOGGER_LEVEL=DEBUG pytest -svv $TEST -k "$ARCH" --no-header 2>&1 | tee "$LOGFILE"
grep -E "shard|mesh|replicated|device_ids" "$LOGFILE" | head -40
```

Show the user the grep output and tell them: "Check `$LOGFILE` for the full compiled graph. Look for tensor shard annotations to confirm weights are split across devices as expected."

---

## Step 8: Deliver

Always print `get_shard_spec` as a fenced code block so the user can copy it without opening the test file.

**Verified**: shard spec snippet · test file path · test run summary · brief explanation of each sharding choice

**Theoretical**: prepend `⚠ unverified — estimated from docs` · shard spec snippet · test file path · explanation · what to grep for when they run it themselves
