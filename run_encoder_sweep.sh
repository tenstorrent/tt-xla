#!/bin/bash
# Uniform encoder coverage sweep across the 4 input cases.
#
# Order is deliberate: the two cheap probes for all 4 cases first (~1 h total, so
# there is a full picture early), then the long Chisel sweep behind them.
# Everything serialises anyway -- RequirementsManager holds an exclusive flock.
#
# Single-instance lock: duplicate watchers have caused device contention twice.
exec 9>/tmp/.diffgemma_sweep.lock
flock -n 9 || { echo "[sweep] another instance is already running; exiting"; exit 1; }

cd /proj_sw/user_dev/ctr-akannan/2_sep_yyz/tt-xla || exit 1
source venv/activate >/dev/null 2>&1
export PYTHONPATH="$PWD:$PWD/tests:$PYTHONPATH"   # append! venv/activate puts build/python_packages (chisel) here
export TTXLA_LOGGER_LEVEL=INFO
mkdir -p sweep_logs chisel_results_archive

# case | variant | prompt tokens (0 = loader default)
CASES=(
  "a_text15:ENCODER:0"
  "b_text277:ENCODER:277"
  "c_imageonly:ENCODER_IMAGE_ONLY:0"
  "d_imagetext:ENCODER_IMAGE:0"
)

say() { echo "[sweep $(date +%H:%M:%S)] $*"; }

# ---------- stage 1: per-layer cumulative curve (independent CPU reference) ----------
for c in "${CASES[@]}"; do
  IFS=: read -r name variant toks <<< "$c"
  log="sweep_logs/${name}_layers.log"
  [ -s "$log" ] && { say "stage1 $name already done, skipping"; continue; }
  say "stage1 $name: per-layer curve ($variant, ${toks} tok)"
  PROBE_VARIANT="$variant" PROBE_TARGET_TOKENS="$toks" \
    timeout 5400 python diffusiongemma_vision_logs/probes/probe_len.py > "$log" 2>&1
  say "stage1 $name exit=$?"
done

# ---------- stage 2: MoE block + routing (covers the ops Chisel cannot measure) ----------
for c in "${CASES[@]}"; do
  IFS=: read -r name variant toks <<< "$c"
  log="sweep_logs/${name}_moe.log"
  [ -s "$log" ] && { say "stage2 $name already done, skipping"; continue; }
  say "stage2 $name: MoE block + routing"
  DIFFGEMMA_MOE_VARIANT="$variant" DIFFGEMMA_PROMPT_TOKENS="$toks" \
    timeout 5400 pytest -svv --capture=tee-sys \
    tests/torch/models/diffusiongemma/test_diffusiongemma_moe_block.py > "$log" 2>&1
  say "stage2 $name exit=$?"
done

# ---------- stage 3: per-op Chisel isolation ----------
# case (d) is already complete (encoder_image_isolation.jsonl, 89,494 records)
for c in "${CASES[@]}"; do
  IFS=: read -r name variant toks <<< "$c"
  [ "$name" = "d_imagetext" ] && { say "stage3 $name: reusing completed run"; continue; }
  out="chisel_results_archive/${name}_isolation.jsonl"
  [ -s "$out" ] && { say "stage3 $name already done, skipping"; continue; }
  log="sweep_logs/${name}_chisel.log"
  say "stage3 $name: Chisel isolation ($variant, ${toks} tok)"
  rm -rf chisel_results
  CHISEL_MODES=iso DIFFGEMMA_CHISEL_VARIANT="$variant" DIFFGEMMA_PROMPT_TOKENS="$toks" \
    timeout 21600 pytest --enable-chisel -p chisel_fixups -svv --capture=tee-sys \
    tests/torch/models/diffusiongemma/test_diffusiongemma_chisel.py::test_chisel_encoder_image \
    > "$log" 2>&1
  say "stage3 $name exit=$?"
  cp chisel_results/*.jsonl "$out" 2>/dev/null && say "  archived $(wc -l < "$out") records"
done

# ---------- verdicts ----------
say "verdicts:"
for c in "${CASES[@]}"; do
  IFS=: read -r name variant toks <<< "$c"
  f="chisel_results_archive/${name}_isolation.jsonl"
  [ "$name" = "d_imagetext" ] && f="chisel_results_archive/encoder_image_isolation.jsonl"
  [ -s "$f" ] && python analyze_chisel.py "$f" > "sweep_logs/${name}_verdict.txt" 2>&1 \
    && say "  $name: $(grep VERDICT "sweep_logs/${name}_verdict.txt")"
done
say "SWEEP DONE"
