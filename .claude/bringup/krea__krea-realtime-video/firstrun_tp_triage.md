# FIRST_RUN_TP triage — krea/krea-realtime-video (8-chip n300-llmbox mesh [2,4])

Env: TT_XLA_SPMD=1 CONVERT_SHLO_TO_SHARDY=1 TT_VISIBLE_DEVICES=0,1,2,3

Mesh built, devices initialized, sharding applied, graphs compiled partway on
all three — i.e. NO mesh-config / partition-spec-rank error. All failures are
op-level / backend compiler issues, NOT shard-spec or mesh errors.

| component | verdict | root cause | class |
|-----------|---------|-----------|-------|
| vae_decoder_480p_sharded | FAILED (16m) | `aten.slice.Tensor(dim=2, start=-2, ...)` -> "Value out of range (expected [-1,0], got -2)" inside torch_xla `partition_fx_graph_for_cpu_fallback`. Negative-index temporal slice in the AutoencoderKLWan causal decoder the dynamo bridge mishandles. | framework/op (torch_xla) |
| umt5_sharded | FAILED (3m) | `TT_FATAL: Can only slice tilized tensor with width begin index aligned to tiles` (ttnn slice_device_operation.cpp:164; needs shape[-1] % 32 == 0 and slice_start[-1] % 32 == 0) + "unexpected run_mailbox" -> Bad StatusOr 13. Non-tile-aligned slice op. | TTNN op constraint |
| causal_wan_dit_480p_sharded | ERROR (~8m) | ncrisc/brisc RISC-V kernel build failure during TT-MLIR compile -> Segmentation fault (core dump). 28GB weights loaded fine; failure is in backend codegen. | backend codegen crash |

None are shard-spec/mesh -> per run directive, recorded for follow-up (not fixed
in this pass). Recommended next: minimal op-repro for each (slice-with-negative
index; tile-unaligned slice; the DiT kernel-build/segfault), then issue-create or
runtime-failure-debugger.
