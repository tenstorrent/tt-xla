# MP Bringup Log — <model-id>

- **Model / pytest id:** <e.g. Qwen2.5-0.5B-Instruct / qwen2.5-0.5b-instruct>
- **Hardware:** <n150 / qb2-blackhole / ...>
- **Branch / commit:** dgolubovic/mp-agentic-bringup @ <sha>
- **Agent session date:** <YYYY-MM-DD>

## 1. Baseline
- Starting config (dtypes as-is): weights=<>, kv=<>, activations=<>
- `baseline_acc` (TOP1 p5): <%>   TOP5 p5: <%>
- `threshold`: <%> (source: user / 0.90×baseline)
- Command used: `pytest -svv "...::test_vllm_benchmark[<id>]" --accuracy-testing`

## 2. Runs (one row per accuracy run)

| # | weight_dtype | weight_overrides | kv_cache | activation_lowering | other knobs | TOP1 p5 | TOP5 p5 | notes |
|---|---|---|---|---|---|---|---|---|
| 1 |  |  |  |  |  |  |  | baseline |
| 2 |  |  |  |  |  |  |  |  |

## 3. Investigations
For each: what/why, IR-dump comparison (`modules/`), lit repro, chisel output,
single-op repro, compute-kernel-config changes, kernel debugging. Include the
exact commands and their results (pass/fail, before/after accuracy).

## 4. Pattern-not-triggering findings (if any)
- Pass: <TTNN... / TTIR...>   Lit repro path: <>   Reproduced miss? <y/n>
- tt-mlir fix: <summary>   Draft PR: <link>

## 5. Final result
- Final config kept: <>
- Features kept / dropped and WHY: <>
- Final TOP1 p5 / TOP5 p5 vs baseline: <>

## 6. Root-cause analysis — what went wrong
<What was surprising, what wasted time, what the skill should have told you.
This is the ONLY basis for a proposed skill edit — and only with user approval.>

## 7. Proposed skill edits (draft — needs user approval before applying)
<Concrete diff/summary of what to change in SKILL.md and why.>
