# FLUX.2 sharded tests — with fix branch tt-forge-models `akannan/fix_flux2_encoder_oom`

Submodule `third_party/tt_forge_models` @ `ac2212ffdf`. Machine: 8-chip LLMBox (wh-lb-46).
xfail/skip markers commented out → results are real (PASS = pass, not xpass).

## The fix
`flux2/pytorch/src/model_utils.py`:
1. `MESH_SHAPES` now puts every device on the "model" axis: `8: (1, 8)` (was `(2, 4)`).
   → 8-way weight sharding instead of 4-way (the batch axis only replicated, wasting half the chips).
2. `_resolve_text_transformer()` correctly descends `Mistral3ForConditionalGeneration`
   (`model.language_model`) so the text-encoder shard spec actually applies to the decoder stack.

## Results
| Test | Before fix | After fix |
|------|-----------|-----------|
| `test_text_encoder_sharded` | FAILED — DRAM OOM at input prep (320 MB) | **PASSED** (275 s) |
| `test_transformer_sharded`  | FAILED — DRAM OOM, tilize (162 MB) | **STILL FAILED** — DRAM OOM, `TilizeWithValPadding` output (56 MB); surfaced as `Bad StatusOr Error code 13` (954 s) |

Verified the transformer run used mesh `(1, 8)` (8-way), so the fix's mesh change was active.

### transformer_sharded remaining failure (real)
`TT_FATAL ... Out of Memory: cannot allocate 56623104 B (56 MB) DRAM buffer across 12 banks`
(allocated 1039 MB, free 31 MB) at `bank_manager.cpp:462`, during execution at
`ttnn::prim::tilize_with_val_padding` → `create_output_tensors`. The 32B transformer's sharded
weights still nearly fill per-bank DRAM (~1 GB of 1.07 GB), so a mid-graph activation tilize OOMs.
This 56 MB figure matches the model's original xfail reason.

## Bottom line
- ✅ The fix resolves **text_encoder_sharded** (OOM → pass).
- ❌ **transformer_sharded** still DRAM-OOM-bound even with 8-way sharding; needs further memory
  reduction (activation memory-config / on-the-fly weight handling / dtype), not just more
  weight-sharding.

## Logs
`flux_updated_logs/test_text_encoder_sharded_FIX.log`, `flux_updated_logs/test_transformer_sharded_FIX.log`
