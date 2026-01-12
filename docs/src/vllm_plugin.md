# Overview
The TT-XLA vLLM Plugin enables [vLLM](https://github.com/vllm-project/vllm) — a high-performance large-language-model serving system — to use TT-XLA as a backend for running inference on Tenstorrent hardware. The plugin integrates TT-XLA into vLLM’s plugin system, allowing models served by vLLM to be compiled and executed through TT-XLA’s PJRT and MLIR pipeline.

# Installation
vLLM plugin requires a working TT-XLA installation (built or wheel). Please follow [Getting started](./getting_started.md) for more information about TT-XLA. Please install TT-XLA and use the same virtual environment for vLLM plugin.

## Installation Options
vLLM plugin can be installed in two ways

### Building from wheel
1. Please activate TT-XLA virtual environment.

2.  Please set VLLM_TARGET_DEVICE as empty

```bash
export VLLM_TARGET_DEVICE="empty"
```

3. Install the wheel in TT-XLA activate virtual enviroment.

```bash
pip install vllm-tt --extra-index-url https://pypi.eng.aws.tenstorrent.com/
```

>**NOTE:** You can pull pre-releases (these may not be stable, so proceed with caution) by adding the `--pre` flag
> directly after `pip install`.
> You can also choose a wheel from the [nightly release page](https://github.com/tenstorrent/tt-xla/releases).

### Building from source
1. Please follow the instruction on building TT-XLA from [source](./getting_started_build_from_source.md). This will build the required TT-XLA dependencies for the plugin. There is no need to set VLLM_TARGET_DEVICE as TT-XLA virtual enviroment handles it in `venv/activate`.

2. Install vLLM plugin with

```bash
pip install -e integrations/vllm_plugin/
```

**Note:** vLLM plugin is installed as editable package. User can modify the existing codebase.

# Running an example
vLLM runs model in two models

## Online serve
This approach uses vLLM's HTTP server to start a model as service and a user can make HTTP request for inference.
1. Start HTTP service to host a model

```bash
vllm serve meta-llama/Llama-3.2-3B \
    --max-model-len 64 \
    --max-num-batched-tokens 64 \
    --max-num-seqs 1 \
    --gpu-memory-utilization 0.002 \
    --additional-config "{\"enable_const_eval\": \"False\", \"min_context_len\": 32}"
```

**Note:** It will take few minutes to downlaod the weights and start the model.

2. Once model is fully loaded; you can make HTTP request from another terminal using curl.

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-3B",
    "prompt": "I like taking walks in the",
    "max_tokens": 32,
    "temperature": 0.7
  }'
```

**Note:** More examples can be found in (tt-xla/examples/vllm/)[https://github.com/tenstorrent/tt-xla/tree/main/examples/vllm]

## Offline Inference
This approach is used to run model via python script as offline inference. An example script is given below

```
import vllm

def llama3_3b_generation():
    prompts = [
        "I like taking walks in the",
    ]
    sampling_params = vllm.SamplingParams(temperature=0.8, top_p=0.95, max_tokens=32)
    llm_args = {
        "model": "meta-llama/Llama-3.2-3B",
        "max_num_batched_tokens": 32,
        "max_num_seqs": 1,
        "max_model_len": 32,
        "gpu_memory_utilization": 0.002,
        "additional_config": {
            "enable_const_eval": False,
            "min_context_len": 32,
        },
    }
    llm = vllm.LLM(**llm_args)

    output_text = llm.generate(prompts, sampling_params)[0].outputs[0].text
    print(f"prompt: {prompts[0]}, output: {output_text}")

if __name__ == "__main__":
    llama3_3b_generation()
```

**Note:** More such python scripts can be found as pytest in (tt-xla/tests/integrations/vllm/)[https://github.com/tenstorrent/tt-xla/tree/main/tests/integrations/vllm_plugin]
