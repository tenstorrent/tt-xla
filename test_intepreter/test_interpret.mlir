// RUN: /localdev/ajakovljevic/stablehlo/build/bin/stablehlo-translate --interpret %s

func.func @add_op_scalar() {
  %0 = stablehlo.constant dense<2> : tensor<i4>
  %1 = stablehlo.constant dense<3> : tensor<i4>
  %2 = stablehlo.add %0, %1 : tensor<i4>
  check.expect_eq_const %2, dense<5> : tensor<i4>
  func.return
}
