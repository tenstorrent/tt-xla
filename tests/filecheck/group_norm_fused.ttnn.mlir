// At opt >= 1 all six GroupNorm(32, 1024) must fuse; CHECK-NOT pins the count to exactly six.
// CHECK-COUNT-6: "ttnn.group_norm"
// CHECK-NOT: "ttnn.group_norm"
