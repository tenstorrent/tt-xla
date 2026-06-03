# matmul_mp PCC report

Failures: **150**. Distinct PCC values: **13**.

## Summary by shape

| Shape | Fails | PCC min | PCC max |
|---|---|---|---|
| `(32,128,1024)x(1024,2048)` | 48 | 0.9661 | 0.9863 |
| `(32,128,2304)x(2304,1024)` | 48 | 0.8404 | 0.9374 |
| `(32,128,2560)x(2560,1024)` | 54 | 0.8072 | 0.9893 |

## `(32,128,1024)x(1024,2048)` (48 fails)

| opt | fp32acc | wd | mf | ANOTHER_OP | HOST |
|---|---|---|---|---|---|
| 0 | False | bf16 | hifi2 | 0.966150 | 0.966150 |
| 0 | False | bf16 | hifi4 | 0.966147 | 0.966147 |
| 0 | False | bf16 | lofi | 0.966568 | 0.966568 |
| 0 | False | bfp4 | hifi2 | 0.966150 | 0.966150 |
| 0 | False | bfp4 | lofi | 0.966568 | 0.966568 |
| 0 | False | bfp8 | hifi2 | 0.966150 | 0.966150 |
| 0 | False | bfp8 | hifi4 | 0.966147 | 0.966147 |
| 0 | False | bfp8 | lofi | 0.966568 | 0.966568 |
| 2 | False | bf16 | hifi2 | 0.986347 | 0.986347 |
| 2 | False | bf16 | hifi4 | 0.986347 | 0.986347 |
| 2 | False | bf16 | lofi | 0.986347 | 0.986347 |
| 2 | False | bfp4 | hifi2 | 0.986347 | 0.986347 |
| 2 | False | bfp4 | lofi | 0.986347 | 0.986347 |
| 2 | False | bfp8 | hifi2 | 0.986347 | 0.986347 |
| 2 | False | bfp8 | hifi4 | 0.986347 | 0.986347 |
| 2 | False | bfp8 | lofi | 0.986347 | 0.986347 |
| 2 | True | bf16 | hifi2 | 0.986347 | 0.986347 |
| 2 | True | bf16 | hifi4 | 0.986347 | 0.986347 |
| 2 | True | bf16 | lofi | 0.986347 | 0.986347 |
| 2 | True | bfp4 | hifi2 | 0.986347 | 0.986347 |
| 2 | True | bfp4 | lofi | 0.986347 | 0.986347 |
| 2 | True | bfp8 | hifi2 | 0.986347 | 0.986347 |
| 2 | True | bfp8 | hifi4 | 0.986347 | 0.986347 |
| 2 | True | bfp8 | lofi | 0.986347 | 0.986347 |

## `(32,128,2304)x(2304,1024)` (48 fails)

| opt | fp32acc | wd | mf | ANOTHER_OP | HOST |
|---|---|---|---|---|---|
| 0 | False | bf16 | hifi2 | 0.840404 | 0.840404 |
| 0 | False | bf16 | hifi4 | 0.840401 | 0.840401 |
| 0 | False | bf16 | lofi | 0.840990 | 0.840990 |
| 0 | False | bfp4 | hifi2 | 0.840404 | 0.840404 |
| 0 | False | bfp4 | lofi | 0.840990 | 0.840990 |
| 0 | False | bfp8 | hifi2 | 0.840404 | 0.840404 |
| 0 | False | bfp8 | hifi4 | 0.840401 | 0.840401 |
| 0 | False | bfp8 | lofi | 0.840990 | 0.840990 |
| 2 | False | bf16 | hifi2 | 0.937412 | 0.937412 |
| 2 | False | bf16 | hifi4 | 0.937412 | 0.937412 |
| 2 | False | bf16 | lofi | 0.937412 | 0.937412 |
| 2 | False | bfp4 | hifi2 | 0.937412 | 0.937412 |
| 2 | False | bfp4 | lofi | 0.937412 | 0.937412 |
| 2 | False | bfp8 | hifi2 | 0.937412 | 0.937412 |
| 2 | False | bfp8 | hifi4 | 0.937412 | 0.937412 |
| 2 | False | bfp8 | lofi | 0.937412 | 0.937412 |
| 2 | True | bf16 | hifi2 | 0.937412 | 0.937412 |
| 2 | True | bf16 | hifi4 | 0.937412 | 0.937412 |
| 2 | True | bf16 | lofi | 0.937412 | 0.937412 |
| 2 | True | bfp4 | hifi2 | 0.937412 | 0.937412 |
| 2 | True | bfp4 | lofi | 0.937412 | 0.937412 |
| 2 | True | bfp8 | hifi2 | 0.937412 | 0.937412 |
| 2 | True | bfp8 | hifi4 | 0.937412 | 0.937412 |
| 2 | True | bfp8 | lofi | 0.937412 | 0.937412 |

## `(32,128,2560)x(2560,1024)` (54 fails)

| opt | fp32acc | wd | mf | ANOTHER_OP | HOST |
|---|---|---|---|---|---|
| 0 | False | bf16 | hifi2 | 0.807190 | 0.807190 |
| 0 | False | bf16 | hifi4 | 0.807187 | 0.807187 |
| 0 | False | bf16 | lofi | 0.807641 | 0.807641 |
| 0 | False | bfp4 | hifi2 | 0.807190 | 0.807190 |
| 0 | False | bfp4 | lofi | 0.807641 | 0.807641 |
| 0 | False | bfp8 | hifi2 | 0.807190 | 0.807190 |
| 0 | False | bfp8 | hifi4 | 0.807187 | 0.807187 |
| 0 | False | bfp8 | lofi | 0.807641 | 0.807641 |
| 0 | True | bf16 | lofi | 0.989335 | 0.989335 |
| 0 | True | bfp4 | lofi | 0.989335 | 0.989335 |
| 0 | True | bfp8 | lofi | 0.989335 | 0.989335 |
| 2 | False | bf16 | hifi2 | 0.924953 | 0.924953 |
| 2 | False | bf16 | hifi4 | 0.924953 | 0.924953 |
| 2 | False | bf16 | lofi | 0.924953 | 0.924953 |
| 2 | False | bfp4 | hifi2 | 0.924953 | 0.924953 |
| 2 | False | bfp4 | lofi | 0.924953 | 0.924953 |
| 2 | False | bfp8 | hifi2 | 0.924953 | 0.924953 |
| 2 | False | bfp8 | hifi4 | 0.924953 | 0.924953 |
| 2 | False | bfp8 | lofi | 0.924953 | 0.924953 |
| 2 | True | bf16 | hifi2 | 0.924953 | 0.924953 |
| 2 | True | bf16 | hifi4 | 0.924953 | 0.924953 |
| 2 | True | bf16 | lofi | 0.924953 | 0.924953 |
| 2 | True | bfp4 | hifi2 | 0.924953 | 0.924953 |
| 2 | True | bfp4 | lofi | 0.924953 | 0.924953 |
| 2 | True | bfp8 | hifi2 | 0.924953 | 0.924953 |
| 2 | True | bfp8 | hifi4 | 0.924953 | 0.924953 |
| 2 | True | bfp8 | lofi | 0.924953 | 0.924953 |

