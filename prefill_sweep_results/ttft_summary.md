# Prefill TTFT Summary

All runs: `bs=1`, `sl=1024`. TTFT extracted from `ttft_s` (× 1000).

| Model | Current (ms) |
|------------------------------|--------------|
| 70b_bs1_sl1024_opt0          |       43.623 |
| 70b_bs1_sl1024_opt1          |       33.619 |
| llama_3_1_8b_bs1_sl1024_opt0 |      529.775 |
| llama_3_1_8b_bs1_sl1024_opt1 |      185.821 |
| llama_3_2_1b_bs1_sl1024_opt0 |      164.470 |
| llama_3_2_1b_bs1_sl1024_opt1 |       52.151 |
| phi1_bs1_sl1024_opt0         |      257.025 |
| phi1_bs1_sl1024_opt1         |      211.206 |
| qwen_3_8b_bs1_sl1024_opt0    |      556.249 |
| qwen_3_8b_bs1_sl1024_opt1    |      232.790 |

> Note: 70B ran with `num_layers=1` (a single-layer slice), so its TTFT is **not** comparable to the full-model runs of the other models.
