#!/bin/bash

cd venv
rm -rf bin/ include/ lib/ lib64 pyvenv.cfg share/ .lock

cd ..
rm -rf build/ third_party/tt-mlir/ third_party/loguru/

source venv/activate
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build

uv pip install -e integrations/vllm_plugin
