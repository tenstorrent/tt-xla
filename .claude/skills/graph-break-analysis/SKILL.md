---
name: graph-break-analysis
description: Analyzes and debugs graph breaks in PyTorch/XLA model compilation. Use when a model generates more graphs than expected during compilation, the user mentions "graph break", or when debugging excessive graph generation in tt-xla pipelines.
allowed-tools: Bash Read Grep Glob Write Edit Task Fetch
---

# Graph Break Analysis

Graph breaks occur when a model, pipeline, or script is split into more graphs than necessary during compilation.
Graph breaks in TT compilation happen either as a result of torch dynamo tracing, torch_xla tracing or in rare cases as byproduct of torch.export.
Common misconception is that different mlir modules are graph breaks. This is not true, when compiler starts going into mlir (starting with vhlo) those are just versions of the same graph.


## Context

- Each graph goes through the compile phase: `vhlo -> stablehlo -> ttir -> ttnn`
- After each module, the log contains the string: `"------------------ END OF MLIR MODULE ------------------"`
- Each graph compilation produces 7 or 8 of these MLIR module strings: 5-6 for vhlo/stablehlo, 1 for ttir, and 1 for ttnn
- If the log file contains N of these strings, then the script generated N//7 or N//8  different graphs (you will see in runtime what is the case).

## Requirements for analysis
- If user's log doesn't meet these criteria, don't proceed with analysis and first run the actual model script. Ask the user for the concrete script that he used and optionally arguments.
- The user must run a model script and dump outputs into a single log file <file>.log
- The user must have used tt-xla repo that is built in debug mode. Use `grep CMAKE_BUILD_TYPE build/CMakeCache.txt` and see if you get "Debug". If empty or "Release", you can't use that log.
- The user must run (has ran) the script with following flags `TORCH_LOGS="+dynamo" XLA_HLO_DEBUG=1`. Go through the log file and find out if that is the case. 
- `TORCH_LOGS="+dynamo"`: example of the lines you will see in the log: "venv/lib/python3.12/site-packages/torch/_dynamo/convert_frame.py", "venv/lib/python3.12/site-packages/torch/_dynamo/", "symbolic_convert.py", "Step 1: torchdynamo start tracing forward /root/tt-xla/graph_break_demo.py", "TRACE starts_line". If these are present you have this variable enabled, if not, then not.
- `XLA_HLO_DEBUG=1`: example of the lines you will see in the log: loc("/path/to/tt-xla/venv/lib/python3.12/site-packages/path/to/file.py:lines")

## Running user's model script
1. Do `source venv/activate` (not source venv/bin/activate)
2. If the build type is not debug, do `cmake --preset debug && cmake --build build`
3. Run the user's script in format `TTXLA_LOGGER_LEVEL=DEBUG TTMLIR_RUNTIME_LOGGER_LEVEL=DEBUG XLA_HLO_DEBUG=1 TORCH_LOGS="+dynamo python userscript.py &> userscript.log`

## Steps

1. Search for `"------------------ END OF MLIR MODULE ------------------"` strings in the log to determine the number of graphs
2. Identify each graph and link them to the source model that caused them (location markers in the log can help)
3. Find the Python/PyTorch implementation of all models used in the script — search locally or on the web (e.g., HuggingFace, similar libraries) to identify the culprits. Always first search locally, looking at the 1. imports that lead to custom implementation 2. imports that lead to third party implementation (e.g. /path/to/tt-xla/venv/lib/python3.12/site-packages/path/to/[transformers/diffusers]). Only then if you don't find locally, then search the web. This decreases the chance of having discrepancy in model impl. 
4. Use 5 research agents in parallel for analysis

## Deliverables

Produce a detailed markdown report of what is causing the graph breaks, sorted by most important/frequent (descending).

For each graph break, provide a Python script that reproduces it. Use this format:

```python
import torch_xla
import torch_xla.runtime as xr
# other imports
# add any global patches, functions, variables, or env vars from the original file

def main():
    xr.set_device_type("TT")
    # add any custom compile options, e.g.:
    # torch_xla.set_custom_compile_options({"optimization_level": 1})

    full_model = load_the_original_full_model()
    model = full_model.submodule.where.we.want.graphbreak.repro

    inputs = ...
    inputs = inputs.to(torch_xla.device())
    model.compile(backend="tt")
    model = model.to(torch_xla.device())
    out = model(inputs)
    torch_xla.sync()

if __name__ == "__main__":
    main()
```

Think carefully and produce a thorough graph break report as instructed.
