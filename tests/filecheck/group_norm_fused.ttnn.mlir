// Non-tile-aligned group_norm must keep the fused kernel, not decompose.
// PCC and max abs error both pass on the decomposed path, so this is the only
// check that detects a routing regression. See tt-metal#51159 / #52924.
// CHECK: "ttnn.group_norm"
