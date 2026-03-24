---
name: graph-break-analysis
description: Guide for doing graph break analysis for some model/script and the log file search analysis.
allowed-tools: Bash, Read, Grep, Glob, Write, Edit, Task, Fetch
---

Graph breaks are term used when some model/pipeline/script is broken into more graphs during compilation than  needed.

Use Case: Debugging many generated graphs
Trigger: User says "I have graph break" or "my model generated many graphs"

Context:
- each graph goes through compile phase vhlo->stablehlo->ttir->ttnn
- after each module there is a string in the log "------------------ END OF MLIR MODULE ------------------"
- each graph compilation has 7 of these mlir module strings: 5 for vhlo/stablehlo, 1 ttir and 1 ttnn
- that means that if the log file contains 7N of these strings, then the whole script generated N different graphs.

Steps to follow:
- You should search for these strings to find out what are these graphs.
- When you find what are these graphs, try to link them to the source model which caused them (location markers can help with that, look them up in the log)
- If you don't see dynamo logs, instruct the user to turn on dynamo logs with `TORCH_LOGS="+dynamo" TORCHDYNAMO_VERBOSE=1` and run the model execution again for easier debugging
- If you don't see locations in the compiler (example format loc("/root/tt-xla/venv/lib/python3.12/site-packages/diffusers/models/embeddings.py":1813:0)), then instruct the user to turn on locations in tt-xla with `XLA_HLO_DEBUG=1` and run the model again
- Ideally, find the python/pytorch implementation of all models used in the given script - this will help you find the culprits (either search for local implementations or search web for huggingface or similar libraries)
- for analysis, use 5 research agents in parallel

Deliverables:
A detailed markdown report of what is causing the graph breaks (sorted by most important/frequent decreasingly)
For each one of them, ideally provide a python script that gives a repro for that graph break. Repro should be in format similar to this:
```
import torch_xla
import torch_xla.runtime as xr
# other imports
# add any global patches or functions or variables or env vars that were present in original file, to ensure the reproduction

def main():
    xr.set_device_type("TT")
    # add any custom compile options, e.g. torch_xla.set_custom_compile_options({"optimization_level": 1})

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
Please think hard and make this graph break report like instructed
