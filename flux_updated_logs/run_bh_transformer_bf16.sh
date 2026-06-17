#!/usr/bin/env bash
# Blackhole QB (4x p150b, 32GB/chip) — full FLUX.2 transformer, bf16, (1,4) 1-D mesh.
# Token is passed via the environment (HF_TOKEN); do NOT hardcode it here.
set -uo pipefail
cd /localdev/ctr-akannan/17_jun_sjc/tt-xla

export HF_HOME=/localdev/ctr-akannan/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
export TT_METAL_CACHE=/localdev/ctr-akannan/.cache/tt_metal
: "${HF_TOKEN:?HF_TOKEN must be set in the environment}"
export HF_TOKEN HF_HUB_ENABLE_HF_TRANSFER=1

FLUX_NL=999 FLUX_NS=999 FLUX_SHARDED=1 \
  timeout 7200 python -m pytest -svv \
  tests/torch/models/flux2/test_transformer_realwt_isolate.py::test_realwt
echo "EXIT_CODE=$?"
