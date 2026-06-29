# Codegen emit/load — known issue

Scratch notes for the `svuckovic/vllm-codegen` work. Covers the codegen
emit → (edit) → load workflow (`TTXLA_CODEGEN_EXPORT_DIR` /
`TTXLA_CODEGEN_LOAD_DIR`, and the `codegen_py` / `codegen_load_py` backends).

## Known issue: emit-only produces non-loadable code

**Symptom.** Emitting with `TTXLA_CODEGEN_EXPORT_DIR` and then loading the same
directory with `TTXLA_CODEGEN_LOAD_DIR` fails at execution time:

```
Codegen load: graph 71e6252d1b9863c2 -> .../qwen_codegen/graph_0   # match OK
...
RuntimeError: AttributeError: module 'main' has no attribute 'forward'
```

The graph is matched by hash correctly; the failure is that the saved
`graph_N/main.py` has no `forward(inputs, device)` entrypoint, so the loader
cannot run it.

**Why.** Load executes the saved code via `PythonModelRunner`, which imports
module `main` and calls `forward`:

- `pjrt_implementation/src/api/so_loaded_executable_instance.cc` —
  `MODULE_NAME = "main"`, `ENTRYPOINT_NAME = "forward"`.

But emission only asks tt-alchemist to generate that entrypoint when the graph
is *not* a dry run:

- `pjrt_implementation/src/api/module_builder/module_builder.cc`, `performCodegen`:
  `target-module = !dry_run` (i.e. the `forward` wrapper is emitted only when
  `dry_run == false`).

With the current `dry_run` defaults, env-triggered emit (`codegen_py`) runs as
`dry_run == true` (emit-only — it generates code but does not execute it):

- `pjrt_implementation/src/api/compile_options.cc` — only `TTNNFlatbuffer` and
  `TTNNCodegenLoadPy` default to not-dry-run; `codegen_py` defaults to
  `dry_run == true`.

So emit-only writes a `main.py` with `_main()` / `main()` but **no `forward()`**,
and the later load run — which *does* execute (`codegen_load_py` is not a dry
run) — has nothing to call.

In short: the artifact emitted in dry-run mode is not a loadable artifact, yet
load assumes every emitted graph is loadable.

## Why the obvious one-liner is not the fix

Forcing `target-module=true` unconditionally in `performCodegen` makes emitted
code carry a `forward` wrapper, but that is necessary, not sufficient — the
correct fix here is more involved and is intentionally **left as a follow-up**.
The real fix needs to reconcile the emit/load contract end to end:

- Decide where the emit/load split sits relative to `dry_run`: emit-only should
  still produce a *loadable* artifact (forward wrapper + whatever the runner
  expects), independent of whether the emit run also executes.
- Make sure the emitted `forward` entrypoint is correct for every graph kind the
  run produces (prefill / decode buckets / sampling), and matches what
  `PythonModelRunner` + `SOLoadedExecutableInstance::prepareInputTensor` feed in
  (including the multi-device host-tensor re-wrap for sharded inputs).
- Either guarantee emit always emits loadable graphs, or have load fail loudly
  and early (before execution) when a matched graph is not runnable.

Until that lands, treat **emit-only + load as not functional**; the emitted
directory is useful for inspecting generated code but cannot currently be
reloaded and executed.

## Repro

```bash
source venv/activate
python codegen_qwen.py qwen_codegen        # emit (dry_run -> no forward wrapper)
python codegen_qwen_load.py qwen_codegen   # load -> AttributeError: ... 'forward'
```

(`codegen_qwen.py` / `codegen_qwen_load.py` / `chat_qwen.py` are scratch driver
scripts at the repo root, not part of the build.)
