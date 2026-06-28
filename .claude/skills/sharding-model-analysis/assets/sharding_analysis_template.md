# Sharding analysis - <MODEL NAME>

> Living document. Built up across the skill's phases, finalized in phase 8.

## 1. Model & target
- Model / variant:
- Load script:
- dtype:
- Device count and target mesh shape(s): <e.g. (1, 4) on qb2 - see references/mesh_shapes.md>

## 2. Architecture
<Module tree with parameter shapes - see references/run_analysis.md. Mark which
consecutive layers form Megatron column→row pairs.>

## 3. Recommended strategy (summary)
<One paragraph: which parallelism (tensor / sequence / data) and which scheme under it
(e.g. Megatron column→row), plus the headline CCL total.>

## 4. Per-component design
Repeat per component:

### <component name>
- **Chosen sharding:** <e.g. `to_q/k/v ("tp", None)` column-parallel; `to_out (None, "tp")` row-parallel>
- **Why:** <rationale; cite a source - Megatron §, a PR, references/video_*.md>
- **Rejected alternatives:** <option → why worse: more/larger CCLs, or unsupported per references/compiler_support.md>
- **CCLs:** <count + approximate size, per references/ccl_cheatsheet.md>

## 5. CCL budget
| Component | Collective | Count | Approx size |
|---|---|---|---|
|  |  |  |  |
| **Total** |  |  |  |

## 6. Bottlenecks & compiler limitations
<Largest CCLs; anything hitting references/compiler_support.md; fallbacks taken and why.>

## 7. Sources
<Papers, GitHub issues/PRs, discussions, and code references cited above.>
