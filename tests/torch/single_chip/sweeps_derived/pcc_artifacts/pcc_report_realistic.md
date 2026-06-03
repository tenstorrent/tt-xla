# matmul_mp PCC report

Failures: **48**. Distinct PCC values: **3**.

## Summary by shape

| Shape | Fails | PCC min | PCC max |
|---|---|---|---|
| `(32,128,1024)x(1024,2048)` | 16 | 0.5002 | 0.5002 |
| `(32,128,2304)x(2304,1024)` | 16 | 0.5005 | 0.5005 |
| `(32,128,2560)x(2560,1024)` | 16 | 0.5004 | 0.5004 |

## `(32,128,1024)x(1024,2048)` (16 fails)

| opt | fp32acc | wd | mf | ANOTHER_OP | HOST |
|---|---|---|---|---|---|
| 2 | False | bf16 | hifi2 | 0.500189 | — |
| 2 | False | bf16 | hifi4 | 0.500189 | — |
| 2 | False | bf16 | lofi | 0.500189 | — |
| 2 | False | bfp4 | hifi2 | 0.500189 | — |
| 2 | False | bfp4 | lofi | 0.500189 | — |
| 2 | False | bfp8 | hifi2 | 0.500189 | — |
| 2 | False | bfp8 | hifi4 | 0.500189 | — |
| 2 | False | bfp8 | lofi | 0.500189 | — |
| 2 | True | bf16 | hifi2 | 0.500189 | — |
| 2 | True | bf16 | hifi4 | 0.500189 | — |
| 2 | True | bf16 | lofi | 0.500189 | — |
| 2 | True | bfp4 | hifi2 | 0.500189 | — |
| 2 | True | bfp4 | lofi | 0.500189 | — |
| 2 | True | bfp8 | hifi2 | 0.500189 | — |
| 2 | True | bfp8 | hifi4 | 0.500189 | — |
| 2 | True | bfp8 | lofi | 0.500189 | — |

## `(32,128,2304)x(2304,1024)` (16 fails)

| opt | fp32acc | wd | mf | ANOTHER_OP | HOST |
|---|---|---|---|---|---|
| 2 | False | bf16 | hifi2 | 0.500491 | — |
| 2 | False | bf16 | hifi4 | 0.500491 | — |
| 2 | False | bf16 | lofi | 0.500491 | — |
| 2 | False | bfp4 | hifi2 | 0.500491 | — |
| 2 | False | bfp4 | lofi | 0.500491 | — |
| 2 | False | bfp8 | hifi2 | 0.500491 | — |
| 2 | False | bfp8 | hifi4 | 0.500491 | — |
| 2 | False | bfp8 | lofi | 0.500491 | — |
| 2 | True | bf16 | hifi2 | 0.500491 | — |
| 2 | True | bf16 | hifi4 | 0.500491 | — |
| 2 | True | bf16 | lofi | 0.500491 | — |
| 2 | True | bfp4 | hifi2 | 0.500491 | — |
| 2 | True | bfp4 | lofi | 0.500491 | — |
| 2 | True | bfp8 | hifi2 | 0.500491 | — |
| 2 | True | bfp8 | hifi4 | 0.500491 | — |
| 2 | True | bfp8 | lofi | 0.500491 | — |

## `(32,128,2560)x(2560,1024)` (16 fails)

| opt | fp32acc | wd | mf | ANOTHER_OP | HOST |
|---|---|---|---|---|---|
| 2 | False | bf16 | hifi2 | 0.500380 | — |
| 2 | False | bf16 | hifi4 | 0.500380 | — |
| 2 | False | bf16 | lofi | 0.500380 | — |
| 2 | False | bfp4 | hifi2 | 0.500380 | — |
| 2 | False | bfp4 | lofi | 0.500380 | — |
| 2 | False | bfp8 | hifi2 | 0.500380 | — |
| 2 | False | bfp8 | hifi4 | 0.500380 | — |
| 2 | False | bfp8 | lofi | 0.500380 | — |
| 2 | True | bf16 | hifi2 | 0.500380 | — |
| 2 | True | bf16 | hifi4 | 0.500380 | — |
| 2 | True | bf16 | lofi | 0.500380 | — |
| 2 | True | bfp4 | hifi2 | 0.500380 | — |
| 2 | True | bfp4 | lofi | 0.500380 | — |
| 2 | True | bfp8 | hifi2 | 0.500380 | — |
| 2 | True | bfp8 | hifi4 | 0.500380 | — |
| 2 | True | bfp8 | lofi | 0.500380 | — |

