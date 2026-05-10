# Error patterns for tt-xla backward-pass triage

The patterns below come from issue #4294 plus what the runtime actually emits.
Search `run.log` (the full pytest dump, never `tail`-ed) for these. Match in this
order — earlier entries are more specific. Use the **class** to title the draft
issue and to decide which downstream repo it belongs to.

When several patterns match, prefer the one closest (in the dump) to the last
`RuntimeTTNN | DEBUG | Executing operation:` line — that's the line we treat as
the offending op.

## Memory exhaustion (tt-metal)

### L1 OOM — circular buffer overflow
- Class: `l1-cb-overflow`
- Repo: tt-metal
- Regex: `Statically allocated circular buffers on core range \[\(.*?\) - \(.*?\)\] grow to (\d+) B which is beyond max L1 size of (\d+) B`
- Note: capture both byte counts; they go in the issue title and body.

### L1 OOM — bank allocation
- Class: `l1-bank-oom`
- Repo: tt-metal
- Regex: `Out of Memory: Not enough space to allocate (\d+) B L1 buffer across (\d+) banks`
- Note: capture the requested bytes and bank count.

### DRAM OOM
- Class: `dram-oom`
- Repo: tt-metal (BUT: per the issue, DRAM OOMs are usually NOT reducible to a
  single op. Set the artifact to `triage-log.md` and explain that single-op
  repro doesn't apply — the developer needs to look at the model-level
  allocation pattern.)
- Regex: `(DRAM|out of memory).*allocate (\d+) B.*DRAM`
- Note: do not attempt the single-op repro path for this class.

## Op-level errors (tt-metal or tt-mlir)

### TT_FATAL conv2d
- Class: `tt-fatal-conv2d`
- Repo: tt-metal
- Regex: `RuntimeError: TT_FATAL.*conv2d`
- Note: usually a kernel constraint or an unsupported shape. Repro is single-op.

### TTIR transpose-conv channel mismatch
- Class: `ttir-conv-transpose-channel-mismatch`
- Repo: tt-mlir
- Regex: `error: 'ttir\.conv_transpose2d' op Number of input channels from input tensor must match the first dimension of the weight tensor`
- Note: this is a verifier error, repro is the MLIR module — Python TTNN repro
  may not be possible. Fall back to attaching the MLIR.

### Generic TT_FATAL / TT_THROW
- Class: `tt-fatal-other`
- Repo: tt-metal
- Regex: `(TT_FATAL|TT_THROW)`
- Note: the catch-all. Use only if no more specific pattern matched.

## Numpy / dtype contract failures (tt-xla)

### Subtract dtype mismatch (training-loss path)
- Class: `numpy-subtract-dtype`
- Repo: tt-xla
- Regex: `ufunc 'subtract' did not contain a loop with signature matching types`
- Note: this is a CPU-side numpy issue, not a TTNN op. No single-op TTNN repro
  needed. Write a `triage-log.md` and recommend a tt-xla bug.

## Infrastructure failures (no TTNN op ever ran)

### Pytest collection / import error
- Class: `infra-pytest-collection-error`
- Repo: tt-xla (or env / venv issue, not a model bug)
- Regex: `ImportError while loading conftest|ERROR collecting|ERRORS  =+|ImportError: .*undefined symbol`
- Note: no `Executing operation:` lines will be in the dump because the test
  never ran. Pytest exit codes 2/3/4 also indicate this class. Output is
  always `triage-log.md` — do not attempt a Python repro.

### CPU-side / fixture failure before runtime
- Class: `infra-pre-runtime-failure`
- Repo: depends on the traceback (often the test or a helper)
- Regex: `(AssertionError|RuntimeError|ValueError) .*\n.*\n.*\n.*(?!RuntimeTTNN)`
  (heuristic: a Python exception with no preceding `RuntimeTTNN` lines)
- Note: same as above — no op repro possible.

## Generic catch-all

If none of the above match:
- Class: `unclassified`
- Repo: unknown
- Action: write `triage-log.md` with the last `Executing operation:` line (if
  any), the surrounding ~50 lines from the dump, and an explicit note that
  the failure did not match any known pattern and needs human classification.
