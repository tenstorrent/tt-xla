# Chisel Integration

## Overview

Chisel is a runtime inspection tool from **[tt-mlir](https://github.com/tenstorrent/tt-mlir)** that allows you to debug MLIR graph.
It provides hooks to observe operation level execution and check the intermediate states of the graph.

## Build Configuration

To enable Chisel support in `tt-xla`, configure the build with:

```bash
cmake -G Ninja -B build ... -DTTXLA_ENABLE_CHISEL=ON
cmake --build build
```

## Running Chisel

After building, the Chisel entry point is available at:

```bash
python third_party/tt-mlir/src/tt-mlir/runtime/tools/chisel/chisel/main.py <args>
```

Examples:

```bash
python third_party/tt-mlir/src/tt-mlir/runtime/tools/chisel/chisel/main.py --help
```

```bash
python third_party/tt-mlir/src/tt-mlir/runtime/tools/chisel/chisel/main.py -i ttir.mlir -o output --report-path report.csv -f main --flatbuffer-path fb.ttnn
```


## Further Information

For detailed usage, command-line options, and examples, see the official [Chisel README in the tt-mlir repository](https://github.com/tenstorrent/tt-mlir/blob/main/runtime/tools/chisel/README.md).
