# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


### NOTES FROM LEWIS ###
# When the max_tokens we allow to be generated is low (i.e. 32), generation will most likely stop for both users at the exact same time due to the fact they hit the token cap
# It seems as though when this happens, vLLm immediately picks up the next two users at the same time and generation begins without error.

# The issue seems to occur only when user 0 completes generation naturally (i.e. generates end token) while user 1 is still generating.
# User 2 then takes user 0's spot, and then user 1 starts generating bad tokens while user 2 ends up generating good tokens.
########################

import asyncio
import time
from typing import AsyncGenerator

import vllm
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.v1.engine.async_llm import AsyncLLM


async def get_and_print_output(
    llm: AsyncLLM,
    prompts: list[str],
    sampling_params: vllm.SamplingParams,
    request_id: str,
):
    output: AsyncGenerator[vllm.RequestOutput, None] = llm.generate(
        prompts, sampling_params, request_id=request_id
    )
    async for result in output:
        print(f"prompt: {prompts[0]}, output: {result.outputs[0].text}")


async def test_tinyllama_generation_multibatch():
    prompts = [
        "Hello, my name is",
        "Paris is the capital of",
        "I like taking walks in the",
        "Cheese is an excellent",
        "Gorrilas are the",
    ]
    sampling_params = vllm.SamplingParams(temperature=0.8, top_p=0.95, max_tokens=2048)
    llm_args = AsyncEngineArgs(
        **{
            "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "max_num_batched_tokens": 2048,
            "max_num_seqs": 2,
            "max_model_len": 2048,
            "gpu_memory_utilization": 0.15,
            "additional_config": {
                "enable_const_eval": False,
                "min_context_len": 512,
            },
        }
    )

    llm = AsyncLLM.from_engine_args(llm_args)
    routines = []
    for i in range(len(prompts)):
        routines.append(
            get_and_print_output(llm, prompts[i], sampling_params, f"request_{i}")
        )
        print("SLEEPING FOR 1 SECOND")
        time.sleep(1)
    await asyncio.gather(*routines)


if __name__ == "__main__":
    asyncio.run(test_tinyllama_generation_multibatch())
