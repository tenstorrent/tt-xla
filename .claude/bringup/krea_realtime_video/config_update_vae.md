# CONFIG_UPDATE — krea_realtime_video / vae (single_device)

- Surface        : pipeline component test (marker-only family; not in runner YAML)
- Test path      : tests/torch/models/krea_realtime/test_vae_decoder.py::test_vae_decoder
- Arch verified  : p150 (Blackhole single-chip host; TT_VISIBLE_DEVICES=0)
- Result         : PASSED — run_graph_test PCC 0.99 gate satisfied (879.82s)
- bringup_status : EXPECTED_PASSING (implicit; @nightly @model_test @single_device)
- runner_yaml    : unchanged (consistent with text_encoder/transformer component records)
- Repair landed  : src/model_utils.py _patch_wan_vae_causal_slice() (tt-xla#4465)

## Arch note
- Requested --arch n150 is not physically runnable on this Blackhole (p100a) host;
  used the documented p150 fallback. weight_fit shows VAE fits n150 budget too, but
  n150 (Wormhole) was not exercised on this machine. supported_archs recorded: [p150].
