# TTNN Graph Capture

TT-XLA can record the TTNN graph of each execution as a JSON report for
[ttnn-visualizer](https://github.com/tenstorrent/ttnn-visualizer), letting you
inspect the ops, buffers and memory usage of a model as it actually ran on
device.

## Usage

Set `TTXLA_GRAPH_CAPTURE_DIR` to a directory and run your model as usual:

```bash
TTXLA_GRAPH_CAPTURE_DIR=./graph_reports python my_model.py
```

One report is written per execution, named
`graph_<executable_name>_<fingerprint>_<index>.json`. Import a report into a
visualizer database and open it:

```bash
python -m ttnn.graph_report ./graph_reports/graph_tt_executable_a91c15d2e2a8b9bd_0.json ./visualizer_db/
```

### How reports are separated

A model usually compiles to more than one graph — an LLM has separate prefill
and decode graphs, for example — and each is a separate PJRT executable that
gets its own reports.

Graphs are told apart by `<fingerprint>`, a hash of the graph's MLIR and its
compile options. It is stable across runs, so the same graph keeps the same
fingerprint from one run to the next, and prefill and decode land in clearly
different files. The `<index>` counts executions **of that graph**, so a decode
loop produces `..._0.json`, `..._1.json` and so on for the decode graph while
prefill keeps its own numbering.

`<executable_name>` is currently the fixed string `tt_executable` for every
graph, so the fingerprint is what carries the identity.

### Limiting the number of reports

Each call into an executable produces its own report, so a generative loop can
emit hundreds. Cap them with:

```bash
TTXLA_GRAPH_CAPTURE_DIR=./graph_reports TTXLA_GRAPH_CAPTURE_LIMIT=1 python my_model.py
```

The limit is **per graph**, not per process: `TTXLA_GRAPH_CAPTURE_LIMIT=1` gives
one report for prefill *and* one for decode, rather than a single report from
whichever graph happened to run first. Reaching the limit is logged once per
graph, so a truncated set of reports is never mistaken for a complete one.

Leave `TTXLA_GRAPH_CAPTURE_LIMIT` unset to capture every execution.

## Cost

Reports for real models are large. A Falcon3-1B benchmark run captured with
`TTXLA_GRAPH_CAPTURE_LIMIT=1` produces four reports totalling roughly 280 MB:
about 12 MB for each decode graph (~4,300 TTNN ops) and about 126 MB for each
prefill graph (~46,000 TTNN ops). Point `TTXLA_GRAPH_CAPTURE_DIR` somewhere with
room, and prefer a low limit on generative models.

Capture also instruments every TTNN op, recording buffer and circular-buffer
activity and building the graph in memory as execution proceeds. It is not free,
so do not capture while measuring performance — run benchmarks and captures
separately.

## Notes

- Capture covers both the flatbuffer and the compiled-shared-object (EmitC)
  execution paths.
- Real device execution still happens; capture observes it rather than
  replacing it.
- A report is written even when an execution fails part way through, which is
  usually the report worth having.
- Only the local TTNN runtime is supported. Under the TTMetal or distributed
  runtimes a warning is logged once and no reports are written.

## Capturing from a flatbuffer instead

If you only need the graph of a compiled model and not a record of a specific
in-process run, you can export the flatbuffer from TT-XLA and use `ttrt`, which
has the same capability:

```bash
ttrt run --graph-capture report.json model.ttnn
```

This runs the model outside TT-XLA with generated inputs, so prefer
`TTXLA_GRAPH_CAPTURE_DIR` when the real weights, mesh configuration or an
actual failure are what you need to see.
