---
name: graph-break-analysis
description: Analyzes and debugs graph breaks in PyTorch/XLA model compilation. Use when a model generates more graphs than expected during compilation, the user mentions "graph break", or when debugging excessive graph generation in tt-xla pipelines.
allowed-tools: Bash Read Grep Glob Write Edit Task Fetch
---

# Graph Break Analysis

Graph breaks occur when a model, pipeline, or script is split into more graphs than necessary during compilation.
Graph breaks in TT compilation happen either as a result of torch dynamo tracing, torch_xla tracing or in rare cases as byproduct of torch.export.

## When to Use

- User says "I have graph break" or "my model generated many graphs"
- Debugging excessive graph generation during compilation

## Context

- Each graph goes through the compile phase: `vhlo -> stablehlo -> ttir -> ttnn`
- After each module, the log contains the string: `"------------------ END OF MLIR MODULE ------------------"`
- Each graph compilation produces 7 of these MLIR module strings: 5 for vhlo/stablehlo, 1 for ttir, and 1 for ttnn
- If the log file contains 7N of these strings, then the script generated N different graphs

## Steps

1. Search for `"------------------ END OF MLIR MODULE ------------------"` strings in the log to determine the number of graphs
2. Identify each graph and link them to the source model that caused them (location markers in the log can help)
3. If dynamo logs are not present, instruct the user to enable them:
   ```bash
   TORCH_LOGS="+dynamo" TORCHDYNAMO_VERBOSE=1
   ```
   Then re-run the model execution
4. If location information is missing from the compiler output (example format: `loc("/root/tt-xla/venv/lib/python3.12/site-packages/diffusers/models/embeddings.py":1813:0)`), instruct the user to enable locations:
   ```bash
   XLA_HLO_DEBUG=1
   ```
   Then re-run the model
5. Find the Python/PyTorch implementation of all models used in the script — search locally or on the web (e.g., HuggingFace, similar libraries) to identify the culprits
6. Use 5 research agents in parallel for analysis

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
