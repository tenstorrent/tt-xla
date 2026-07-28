// A rank-2 activation through a biased nn.Linear decomposes to aten::addmm, and
// PyTorch's addmm decomposition upcasts every operand to f32. The linear must stay
// bf16 on all three operands and the result. Refs: tt-xla #5756.
//CHECK: "ttnn.linear"
//CHECK-SAME: (tensor<{{[0-9x]+}}xbf16{{[^)]*}}, tensor<{{[0-9x]+}}xbf16{{[^)]*}}, tensor<{{[0-9x]+}}xbf16{{[^)]*}}) -> tensor<{{[0-9x]+}}xbf16
