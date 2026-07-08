# Fused-MoE Decode on Wormhole Galaxy — Current State (human summary)

*Branch: `hshah/moe-compute-path` (tt-xla) + `hshah/dmilinkovic/moe-decode-composite` (tt-mlir). Last updated 2026-07-08.*
*Detailed companion (for continuing the work): `CURRENT_STATE_CLAUDE.md`.*

## What this is

We're bringing up GPT-OSS fused-MoE **decode** on the 32-chip Wormhole galaxy (`(4,8)` mesh, EP-4 / TP-8,
`FABRIC_1D_RING`) through tt-xla → tt-mlir → tt-metal.

**Lowering pattern (high level).** In decode, the PyTorch/XLA model (with the `tt_moe_fused` backend) emits a single
StableHLO composite **`tt.moe_decode`** per layer. tt-mlir lowers it to a **fused two-op TTNN pipeline**:

1. `ttnn.all_to_all_dispatch_metadata` — routes each token to the device(s) owning its selected experts, over the
   `cluster_axis=0` (EP-4) 1D ring.
2. `ttnn.moe_compute` (full path, `compute_only=false`) — does the per-expert matmuls (`w0`/`w1`, activation, `w2`,
   optional bias) **and** an internal score-weighted A2A **combine** back to each token's originating device.

The composite is opaque to the sharding pass except for a custom rule that shards the **token** and **expert**
dimensions along the cluster axis and replicates the rest. Expert weights are `bfloat4_b` (host-quantized). Prefill
uses a separate dense-bmm path (not the fused decode op).

## Where we stand

- ✅ **Single-layer decode runs end-to-end and is numerically correct.** `test_gpt_oss_20b_moe_fused_galaxy_pcc`
  with `--num-layers 1` passes: **prefill PCC 0.999905, decode PCC 0.999237**, device output byte-identical to the CPU
  reference. Re-verified after all cleanup.
- ❌ **Two or more layers HANG.** With `--num-layers 2` (or the full 24) the run deadlocks ~10 min in, on the device,
  after both layers' MoE collectives have completed.

## The hang — what we know

Trace-instrumentation localized it precisely: **all four MoE collectives complete** (both layers' dispatch + combine
each return), and the process then **hangs at the very next fabric op — the LM-head logit `ttnn.all_gather` on
`cluster_axis=0`** (`8×1×201088 → 32×1×201088`, gathering the data-parallel batch). One host thread busy-polls forever;
all 32 chips sit at a flat ~35 W (parked on a fabric wait). A single layer runs the *identical* all_gather and it
completes — so it's depth-dependent.

## Potential sources / where to look next

We ran ~11 galaxy experiments and a fast pure-`ttnn` reproduction, which **ruled out** the obvious suspects:

- **Not the fused-MoE collectives themselves.** A standalone `ttnn` harness running the *same* real
  `moe_compute`+`dispatch`+`all_gather` — at the exact `(4,8)` topology, as one back-to-back trace program, with fresh
  per-layer semaphores, and even with the full 201088-wide gather — **never hangs**.
- **Not a cross-axis interaction.** Removing the attention `cluster_axis=1` all-gathers (verified in the graph) does not
  fix the 2-layer hang.
- **Not** the earlier "fabric mux / EDM credit-leak" theory (refuted by the negative control), and **not** an artifact
  of the test's lm-head sharding hook.

So the hang appears to be a property of the **full compiled decode graph**, not the fused ops in isolation. The
remaining candidates to investigate (in rough priority):

1. **Systemic on-device resource pressure** — L1 / DRAM / worker-core / buffer-allocation contention from the whole
   program (attention + MoE persistent buffers + per-op fabric mux cores + const-eval'd bf4 weights + paged KV cache)
   such that the large `cluster_axis=0` all_gather can't acquire a core/L1/connection it needs and blocks.
2. **The data-dependent LM-head chain** — the all_gather's input is produced by the big `2880→201088` lm-head matmul;
   some interaction of that matmul's allocation/completion with the gather.
3. **tt-mlir lowering / buffer placement specifics** in the full program vs the fast harness.

**The blocker:** we could not get device-level visibility — **Watcher is broken in this environment** (hard-throws on
fast-dispatch core (0,0); patching it segfaults in the ETH dump and strands the board). The single most valuable next
step is to instrument the actual hang with Watcher/tt-lens in a working environment (or have the tt-metal fabric team
do so) to see *which core/waypoint/semaphore* the all_gather is stuck on. An escalation with a runnable negative-control
harness is written up in `MOE_FUSED_DECODE_WH_HANG_TTMETAL_ESCALATION.md`.
