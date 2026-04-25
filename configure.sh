#!/bin/bash

git submodule update --init --recursive

cd tt-xla
source venv/activate
cmake -G Ninja -B build -DCMAKE_BUILD_TYPE=Debug -DTTMLIR_SOURCE_DIR_OVERRIDE=../tt-mlir -DTTMLIR_TTMETAL_SOURCE_DIR=../tt-metal
