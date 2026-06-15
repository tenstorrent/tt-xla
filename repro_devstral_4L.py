# Devstral-2-123B DP+TP smoke test: mesh (4,8) -> batch on 4 (DP), model on 8 (TP).
# 4 layers / max_model_len 32 so only 4 layers' fp8 weights load+dequant (fast),
# while still exercising the full DP+TP decode path (paged_update_cache /
# paged_fill_cache sharding). Bump max_num_seqs (and gpu_memory_utilization) for
# the batch-128 run.
import os
import vllm


def main():
    bs = int(os.environ.get("BS", "32"))
    gmu = float(os.environ.get("GMU", "0.05"))
    if os.environ.get("SAME", "0") == "1":
        # Period-(bs/dp) prompts: each DP replica sees the SAME set of distinct
        # prompts, so user i and i+local_bs (next replica) must produce identical
        # tokens iff DP sharding is correct. Distinct-within-replica avoids the
        # prefix-cache dedup that identical prompts would trigger.
        period = int(os.environ.get("PERIOD", "8"))
        prompts = [f"Continue in English: item number {i % period} is" for i in range(bs)]
    else:
        prompts = [f"Continue in English: item number {i} is" for i in range(bs)]
    sampling_params = vllm.SamplingParams(temperature=0.0, max_tokens=16)
    llm_args = {
        "model": "mistralai/Devstral-2-123B-Instruct-2512",
        "max_num_batched_tokens": max(1024, 32 * bs),  # >= max_model_len * batch
        "max_num_seqs": bs,
        "max_model_len": 32,
        "gpu_memory_utilization": gmu,
        "enable_prefix_caching": False,  # per-user fill_cache must run; no dedup
        "additional_config": {
            "min_context_len": 32,
            "enable_trace": os.environ.get("TRACE", "1") == "1",
            "experimental_weight_dtype": "bfp_bf8",
            "enable_tensor_parallel": True,   # model on 8 (TP)
            "enable_data_parallel": os.environ.get("DP", "1") == "1",  # DP=0 -> TP-only reference
            "shard_weights_on_batch_axis": False,  # pure DP (replicate weights)
            "num_hidden_layers": 4,           # fast: only 4 layers load+dequant
            "cpu_sampling": True,             # 2D mesh on-device sampler is soup (#4440)
        },
    }
    print(f"=== Devstral 4x8 DP+TP | batch={bs} gmu={gmu} ===")
    llm = vllm.LLM(**llm_args)
    outputs = llm.generate(prompts, sampling_params)
    dp = os.environ.get("DP", "1")
    print(f"\n=== DEVSTRAL 4x8 REPRO OK: {len(outputs)} outputs (batch={bs}) DP={dp} ===")
    for i, out in enumerate(outputs):
        toks = list(out.outputs[0].token_ids)
        print(f"[{i:02d}] toks={toks} text={out.outputs[0].text!r}")


if __name__ == "__main__":
    main()
