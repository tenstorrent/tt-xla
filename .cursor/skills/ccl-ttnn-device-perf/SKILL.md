---
name: ccl-ttnn-device-perf
description: Analyze TTNN collective communication ops from TTNN MLIR and correlate them with Tracy device performance CSVs. Use when the user asks about all_gather, reduce_scatter, all_reduce, CCL shapes/configs, mesh topology, or average runtime for collectives in TTNN modules.
---

# CCL TTNN Device Perf

Use this skill when analyzing TTNN collectives from a generated TTNN MLIR module and, optionally, matching them to Tracy device perf CSV data.

## Environment

Prefer running inside the TT-XLA Docker container because that is where the repo, Python env, and profiling tools are expected to exist.

If starting from the host, determine:
- the TT-XLA container name
- the user inside the container
- the repo path inside the container

Generic host template:

```bash
docker exec --user <CONTAINER_USER> <CONTAINER_NAME> /bin/bash -lc '
cd <REPO_PATH_IN_CONTAINER>
source venv/activate
<COMMAND>
'
```

If already inside the container:

```bash
cd <REPO_PATH_IN_CONTAINER>
source venv/activate
<COMMAND>
```

## Inputs

Typical inputs are:
- one TTNN MLIR file under `modules/irs/ttnn_*.mlir`
- one Tracy ops CSV such as `.tracy_artifacts/reports/<timestamp>/ops_perf_results_<timestamp>.csv`

The TTNN MLIR defines the logical collective ops and mesh topology.

The Tracy CSV provides runtime op names and device FW durations.

## What To Extract From TTNN MLIR

From the TTNN MLIR, collect:
- mesh shape
- mesh topology
- all `ttnn.all_gather`
- all `ttnn.reduce_scatter`
- all `ttnn.all_reduce`

For each collective, record:
- input shape
- output shape
- op-specific config
- repeat count in the module

Keep shapes readable:
- remove `#ttnn_layout...` suffixes from shape columns
- keep just logical dimensions and dtype, for example `1x1x32x256xbf16`

## Config To Preserve

For `all_gather`, preserve:
- `all_gather_dim`
- `cluster_axis`
- `topology`

For `reduce_scatter`, preserve:
- `scatter_dim`
- `cluster_axis`
- `reduce_type`
- `topology`
- `compute_config`

For `all_reduce`, preserve its full config block.

## Mesh Topology

Always include the mesh topology from the TTNN module header near the top of the summary, for example:

```text
meshShape = 4x8
meshTopology = [ring, ring]
```

## Matching To Tracy CSV

When a Tracy CSV is provided, add average runtime per unique CCL shape/config case.

Use:
- `DEVICE FW DURATION [ns]`
- convert to microseconds
- average across matching runtime rows

Match runtime rows by:
- op kind
- dtype
- cluster axis when available
- dimension attribute when available
- corresponding runtime shape pattern

Runtime names may differ from TTNN MLIR names. Common runtime op names include:
- `AllGatherDeviceOperation`
- `ReduceScatterDeviceOperation`
- `AllReduceDeviceOperation`
- `FastReduceNCDeviceOperation`

## Lowered Runtime Patterns

Do not assume every TTNN collective appears as a single runtime collective op.

If a TTNN collective has no clear one-to-one runtime match, check whether it lowers into multiple runtime ops.

Known useful pattern:
- some TTNN `reduce_scatter` cases may appear as `all_gather` plus a reduction op in the runtime CSV
- for large head/output cases, `all_gather + FastReduceNCDeviceOperation` is a likely pattern

If a lowered multi-op pattern is strongly supported by the CSV:
- sum the average device FW times of those component runtime ops
- report the summed value in the summary
- explain the decomposition in notes

If the reduction partner is not uniquely attributable:
- leave the runtime average as `n/a`
- explain why in notes

## Output Format

Write a Markdown summary next to the TTNN MLIR, for example:

```text
modules/irs/<ttnn-module-stem>_ccl_shapes_and_configs.md
```

Recommended structure:

```markdown
# CCL Shape And Config Inventory

Source: `<ttnn module path>`

Mesh topology: `meshShape = ...`, `meshTopology = [...]`

## `all_gather`

Observed ops: `<count>`

| Repeats | Input Shape | Output Shape | Avg Device FW Time [us] | Config |
| --- | --- | --- | --- | --- |
| ... | ... | ... | ... | ... |

## `reduce_scatter`

...

## `all_reduce`

...

## Notes

- explanation of any `n/a`
- explanation of any summed multi-op lowering
```

## Workflow

1. Read the TTNN MLIR.
2. Extract mesh shape and mesh topology.
3. Enumerate `all_gather`, `reduce_scatter`, and `all_reduce`.
4. Group by unique shape plus config.
5. Count repeats for each unique case.
6. If a Tracy CSV is available, compute average device FW time in microseconds for each case.
7. For unmatched TTNN collectives, check for lowered runtime patterns such as `all_gather + FastReduceNCDeviceOperation`.
8. Sum component averages only when the lowering pattern is strongly supported.
9. Write the Markdown summary next to the source MLIR.

## Use With Layer Profiling

When layer profiling reveals communication-heavy behavior, use this skill on the corresponding TTNN IR module to inventory collective cases and correlate them with device runtime.

## References

- `tests/benchmark/LAYER_PROFILING_PLAN.md`
- `tests/benchmark/PROFILING.md`
