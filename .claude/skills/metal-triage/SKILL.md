---
name: metal-triage
description: Automate triage of a failing tt-xla backward-pass training test by extracting the offending TTNN op from runtime debug logs and producing a standalone single-op Python TTNN repro plus a draft tt-metal / tt-alchemist issue. Use this skill whenever the user asks to triage, reduce, reproduce, or "file an issue for" a backward / training test failure on Tenstorrent hardware (L1 OOM, DRAM OOM, TT_FATAL, op-level errors, conv2d kernel constraints, all_gather / all_reduce failures), or whenever they paste a tt-xla pytest test id in the context of a single-op repro, even if they do not say the word "triage". This skill assumes the env is already built in Debug mode and will refuse to build it.
---

# metal-triage

Reduces a single failing tt-xla backward-pass training test to a minimal
TTNN single-op repro and a draft downstream issue.

This skill implements the workflow specified in
[tenstorrent/tt-xla#4294](https://github.com/tenstorrent/tt-xla/issues/4294).

## What this skill does

Given **one** failing test id (e.g.
`tests/torch/training/test_basic.py::test_qwen3_backward`), the skill:

1. Verifies the env is built in Debug mode — if not, it aborts with an
   actionable message. **It does not build the env.**
2. Runs the test with full runtime-debug logging, piping the entire `pytest -svv`
   transcript to a temp file plus a junit XML for structured failure data.
3. Reads the dump end-to-end (no `tail`) and extracts the offending
   `RuntimeTTNN | DEBUG | Executing operation:` line and a classified error.
4. Builds a standalone Python TTNN repro from the op's MLIR signature.
5. Runs the repro and compares its error to the original.
6. Emits one of: a draft tt-metal issue, a draft tt-alchemist issue, or a
   triage log explaining what was tried and where it diverged.

All output goes to `/tmp/tt-triage-<test-name>/` for human review. The skill
never auto-files an issue.

## Phase 0 — Verify the env is in Debug mode

Run the precondition script:

```bash
bash .claude/skills/metal-triage/scripts/check_debug_build.sh
```

The script greps `git diff third_party/CMakeLists.txt` for both
`-DCMAKE_BUILD_TYPE=Debug` and `-DTT_RUNTIME_DEBUG=ON`. These overrides are
required because the runtime-debug op-execution log lines (
`RuntimeTTNN | DEBUG | Executing operation: ...`) are only emitted when the
runtime is built with `TT_RUNTIME_DEBUG=ON`, and the more verbose error stacks
require `Debug` build type. Without them the rest of the workflow has nothing
to extract.

If the script exits non-zero, **stop**. Print its output and tell the user to
apply the patch and rebuild themselves. Do not run pytest, do not run cmake,
do not modify the build.

## Phase 1 — Run the failing test with full debug logging

Compute a sanitized name and the output dir:

```bash
TEST_ID="<the-test-id-from-args>"
NAME="$(printf '%s' "${TEST_ID}" | tr '/:.' '___' | tr -s '_')"
OUT="/tmp/tt-triage-${NAME}"
mkdir -p "${OUT}"
```

Run the test. The exact invocation matters — every flag is here for a reason:

```bash
TTMLIR_RUNTIME_LOGGER_LEVEL=DEBUG \
TT_RUNTIME_MEMORY_LOG_LEVEL=operation \
TTXLA_LOGGER_LEVEL=DEBUG \
TTMLIR_LOGGER_LEVEL=DEBUG \
XLA_HLO_DEBUG=1 \
pytest -svv --force-run --runxfail \
  --junitxml="${OUT}/junit.xml" \
  "${TEST_ID}" > "${OUT}/run.log" 2>&1
echo "exit=$?" >> "${OUT}/run.log"
```

Why each piece:

- `TTMLIR_RUNTIME_LOGGER_LEVEL=DEBUG`, `TTMLIR_LOGGER_LEVEL=DEBUG` — emit the
  `RuntimeTTNN | DEBUG | Executing operation:` lines. This is the entire reason
  we required `TT_RUNTIME_DEBUG=ON` at build time.
- `TT_RUNTIME_MEMORY_LOG_LEVEL=operation` — annotate each runtime op with
  memory-state info, useful for L1/DRAM OOM classification.
- `TTXLA_LOGGER_LEVEL=DEBUG` — plugin-side logs (compilation stages, device
  open/close, errors at the PJRT boundary).
- `XLA_HLO_DEBUG=1` — preserves source-location metadata in the HLO so MLIR
  ops still carry useful `loc(...)` annotations we can read in the dump.
- `--force-run --runxfail` — xfail markers in this repo go stale; we want the
  test to run regardless of what the YAML / decorator says, otherwise we get
  back "xfailed" and never see the error.
- `--junitxml=...` — gives us a structured `<failure>` / `<error>` block we
  can read and grep cheaply, and cross-check against the unstructured log.
- Redirect to a file + capture exit code on the trailing line — the dump is
  the source of truth. Never use `tail` on it; read with the `Read` tool's
  `offset` / `limit` or with `grep`.

## Phase 2 — Read the dump end-to-end

Always analyze `${OUT}/run.log` and `${OUT}/junit.xml` regardless of exit
code. The underlying TT_FATAL / op-level message is in the log either way.

What to look for, in order:

1. **The last `RuntimeTTNN | DEBUG | Executing operation:` line** before the
   first crash marker. Crash markers: `TT_THROW`, `TT_FATAL`,
   `RuntimeError`, the start of a Python traceback, or `error:` from the
   MLIR verifier.
2. **The matching error pattern** — open
   `references/error-patterns.md` and walk the patterns in order, using the
   first match closest to the failing-op line. Capture the **class** label —
   it goes in the issue title and decides which downstream repo gets the bug.
3. **The junit.xml `<failure>` / `<error>` text** — sanity-check it tells the
   same story. If the junit text and the log disagree, prefer the log (junit
   is summary-only).

### When there is no `Executing operation:` line in the dump

Possible causes:

- **Pytest exited with a collection / usage error** (exit codes 2, 3, 4):
  conftest import failure, missing fixture, syntax error in the test file,
  collection error. The test never reached the runtime, so no op-execution
  lines were emitted. Read the top of `run.log` — pytest prints the import /
  collection error first.
- **The test failed before any tt-xla compilation happened** — e.g. an
  `ImportError`, a CPU-side numpy issue, or an assertion in a fixture.
- **The Debug-mode override is present in the diff but the build wasn't
  actually rebuilt** with it. The `RuntimeTTNN | DEBUG` log channel is
  enabled at compile time; without a rebuild, the runtime stays silent even
  with the env vars set.

In any of these cases, **skip Phase 3 (no op to extract) and Phase 4 (no
repro to build)**. Jump directly to Phase 5 and emit
`triage-log.md` describing what the dump actually shows. Never invent a
`RuntimeTTNN` op to fill the gap — the artifact would mislead the reviewer.

If the run exited with code 13, also load
`references/error-13-deepdive.md` and incorporate its guidance into the final
report. Do not branch the workflow on exit 13 — the dump still drives the
analysis.

Save the extracted op and the classified error to `${OUT}/extracted-op.txt`:

```
test: <test-id>
exit: <exit-code>
class: <error-class>
op: <full Executing operation: line>
error: <one-line summary of the matched pattern, including any captured byte counts>
```

## Phase 3 — Build the TTNN Python repro

Read `references/ttnn-repro-template.py`. Substitute its TODO markers using
values you parse out of the failing op's MLIR signature in the
`Executing operation:` line. The template handles the common case (a single
op over a sharded mesh tensor); for less common shapes adapt as needed but
keep the structure (open mesh device, build torch tensor, push to TTNN with a
mesh mapper, call the op, print).

Write the result to `${OUT}/repro.py`. Run it:

```bash
python "${OUT}/repro.py" > "${OUT}/repro.log" 2>&1
echo "exit=$?" >> "${OUT}/repro.log"
```

Read `repro.log` end-to-end. Classify its error using the same
`references/error-patterns.md` patterns. Compare its **class** to the class
from Phase 2.

If the error class did not match a single-op-reducible pattern (e.g.
`dram-oom`, `numpy-subtract-dtype`, `ttir-conv-transpose-channel-mismatch`),
skip the repro run — the issue's note in `error-patterns.md` already says
single-op repro doesn't apply.

## Phase 4 — Emit the artifact

Pick exactly one output file based on what happened:

### Case A: classes match → draft tt-metal (or tt-mlir) issue

Write `${OUT}/issue-tt-metal.md`. Title: `[<class>] <op-name>: <one-line error summary>`.
Body sections, in order:

- **Repro** — fenced Python block from `repro.py`.
- **Expected vs actual** — what we expected (op should succeed for these shapes
  per the test that exercises it) vs the error class observed.
- **Source** — the test id and the `Executing operation:` MLIR line that the
  repro was reduced from.
- **Environment** — output of `git -C . log -1 --oneline` for tt-xla and (if
  the submodules are present) tt-mlir / tt-metal. The developer can paste this
  block straight into the GitHub issue.

If the class points at tt-mlir (e.g. `ttir-conv-transpose-channel-mismatch`),
title it for tt-mlir and skip the Python-repro section in favor of the MLIR
snippet — the verifier error reproduces from MLIR, not Python.

### Case B: classes don't match, or repro doesn't trip the bug → triage log

Write `${OUT}/triage-log.md`. Sections:

- **What was tried** — the test invocation, env vars, output paths.
- **Original error** — class + `Executing operation:` line + ~30 lines of
  surrounding log context.
- **Repro outcome** — what `repro.py` produced (or "skipped because class is
  not single-op-reducible") and why it diverged.
- **Recommendation** — one line for the developer (e.g. "rerun on clean
  machine"; "investigate at model level — DRAM OOM not reducible";
  "MLIR-level bug, no Python repro needed, attach MLIR module").

### Stretch: tt-alchemist path

Only if `which tt-alchemist` succeeds, try
`tt-alchemist generate-python <mlir-of-failing-op> -o ${OUT}/alchemist-out`.
If the conversion fails in a way that itself looks like a tool bug, write
`${OUT}/issue-tt-alchemist.md` capturing the input MLIR and the alchemist
error. If `tt-alchemist` isn't on PATH, skip silently — this skill must not
build it.

You write **at most one** of the three artifacts per run, plus the always-
present `extracted-op.txt`, `run.log`, `repro.log`, and `junit.xml`.

## Phase 5 — Report back

Print a short summary to the chat:

- Path: `${OUT}/`
- Class: `<error-class>` (from `extracted-op.txt`)
- Artifact: `issue-tt-metal.md` | `issue-tt-alchemist.md` | `triage-log.md`
- One-line recommendation for the developer.

Never claim you filed an issue. The output is always for human review per the
issue #4294 protocol ("All output gets passed back to the developer for review
before submitting any issues").

## Things this skill must not do

- Build the env (no `cmake`, `ninja`, `pip install`, `make`).
- Modify any file outside `/tmp/tt-triage-*/` and the skill's own dir.
- `chmod` device files, kill other processes, or otherwise mutate machine
  state to "fix" exit-13 errors. Hand those back to the developer.
- Use `tail` on `run.log`. The full transcript is needed; read with
  `Read offset/limit` or `grep`.
- Trust xfail markers — always pass `--force-run --runxfail`.
- Auto-file issues. Output is always a draft for human review.
