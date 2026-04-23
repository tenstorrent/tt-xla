# Requested Functionality — shard-spec-gen skill

Sourced from context.txt and additional_context.txt (team discussion). Only items explicitly requested are listed.

---

# TODO EP 1, SP 7
## Sharding strategies

- [x] Megatron tensor parallelism (column/row-parallel for attention & MLP)
- [x] Data parallelism / batch splitting fallback
- [x] MoE sharding (iterate over experts, handle shared_expert)
- [x] FSDP sharding strategy
<!-- - [ ] Pipeline parallelism --> "So if we (for now justifiably) ignore pipeline parallelism"
- [ ] Sequence parallelism (SP) — for DiT image/video-gen models <!-- maybe - on their map but sub-mesh support for now... -->
- [ ] Expert parallelism (EP) — for large MoE models (e.g. 120B)

# TODO 1
## Strategy selection logic
"the recipe is pretty straightforward — either Megatron or FSDP (both with or without data parallelism) and if memory allows, 
use Megatron (fewer CCLs), and if not, then it has to be FSDP. I think you can get very far with this and I don’t think there’s 
a need (probably not even room) to optimize these two."

- [x] Try Megatron first; fall back to FSDP if memory doesn't allow it
- [x] Try options in order and stop at the first one that works

# TODO 6
## Compile-time validation / CCL feedback (Uros's suggestion)

- [ ] Compile with `num_hidden_layers=1` and report CCL counts per layer
<!-- - [ ] OOM / not-OOM detection  --> "besides OOM/not OOM - obviously for this it first has to run all hidden layers"
"It would be easy to add instructions there — e.g. in Megatron you must have 2 all_reduce per layer, minimize the number of all_to_all operations, 
for FSDP we expect such-and-such number of all_reduce per layer, etc. This matters because of the Python implementation of models (transposes of 
sharded matrices can generate additional CCLs, in which case you just need to swap axes), but anyway this is what skills would look like and this 
would definitely speed up the model bring-up process a lot."
- [ ] Assert 2 `all_reduce` per layer for Megatron
- [ ] Flag excess `all_to_all` operations (should be minimized)
- [ ] Expected `all_reduce` count per layer for FSDP

# TODO 8
## Device & topology knowledge
"The skill could contain some instructions about what types of sharding exist, how they’re done, if there are any 
specifics for our devices, information about memory per chip and similar, topology selection, knowledge about how 
we’ve sharded some previous models e.g. in metal, sharding for CFG in image/video-gen models"

- [x] Hardware topology info (llmbox `(1,8)`, galaxy `(4,8)`)
- [ ] Memory per chip information
- [ ] CFG sharding for image/video-gen models

# TODO 5
## Tribal knowledge capture

- [ ] Sharding patterns from prior metal model implementations  <!-- Requested in issue-->

# TODO 2
<!-- What we definitely need:

one megatron style LLM sharding reference
one conv-based model sharding reference -->
## Code references (as loadable scripts in `references/`)

- [ ] Megatron-style LLM sharding reference script
- [ ] Conv-based model sharding reference script

# TODO 4
## Explicit replication / sharding constraints

- [ ] Ability to mark tensors as explicitly replicated (no sharding propagated) <!-- Requested in issue-->

# TODO 3
## Normalization op awareness

- [ ] Warn when sharding normalization ops (norm statistics become local-only, not global)  <!-- Requested in issue-->

# TODO 3
## Head-count mismatch handling

- [ ] Detect when `num_attention_heads` is not divisible by mesh model-axis size
- [ ] Suggest head padding as a solution (e.g. pad 30 heads → 32 for a 4-chip mesh)
