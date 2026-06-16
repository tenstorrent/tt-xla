#!/bin/bash
# Sequential FLUX.2 component test runner (reset between each)
cd /proj_sw/user_dev/ctr-akannan/15_june_yyz/tt-xla
source .flux_env.sh
source venv/activate 2>/dev/null
LOGDIR=flux_updated_logs
mkdir -p $LOGDIR

run() {
  local name="$1"; shift
  echo "=================== RESET before $name ==================="
  tt-smi -r >/dev/null 2>&1
  echo "=================== RUN $name ==================="
  python -m pytest -svv "$@" 2>&1 | tee "$LOGDIR/${name}.log"
  echo "EXIT[$name]=${PIPESTATUS[0]}"
}

run test_vae_decoder            tests/torch/models/flux2/test_vae_decoder.py
run test_text_encoder_single    tests/torch/models/flux2/test_text_encoder.py::test_text_encoder
run test_transformer_single     tests/torch/models/flux2/test_transformer.py::test_transformer
run test_text_encoder_sharded   tests/torch/models/flux2/test_text_encoder.py::test_text_encoder_sharded
run test_transformer_sharded    tests/torch/models/flux2/test_transformer.py::test_transformer_sharded
echo "=================== ALL DONE ==================="
