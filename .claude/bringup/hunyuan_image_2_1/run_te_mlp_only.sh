#!/bin/bash
set -o pipefail
cd /proj_sw/user_dev/ctr-akannan/3_jun_yyz/tt-xla
export TT_XLA_ARCH=n300-llmbox
export TT_VISIBLE_DEVICES=0,1
export TT_XLA_SPMD=1
export CONVERT_SHLO_TO_SHARDY=1
export HUNYUAN_TE_MLP_ONLY=1
echo "ENV MLP_ONLY=$HUNYUAN_TE_MLP_ONLY ARCH=$TT_XLA_ARCH DEVICES=$TT_VISIBLE_DEVICES"
python -m pytest -svv \
  tests/torch/models/HunyuanImage_2_1/test_text_encoder.py::test_text_encoder_sharded \
  2>&1
echo "PYTEST_EXIT=$?"
