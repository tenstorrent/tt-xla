# 🧠 LLaMA 3.1–8B: Tensor Parallel JAX Implementation (Draft PR)

This draft PR introduces a tensor-parallel JAX implementation of Meta’s LLaMA 3.1–8B model using a 1×4 device mesh, built with Flax (linen API). The implementation supports both sharded (multi-chip) and unsharded (single-chip) execution.

The outputs of the JAX implementation were compared against:

    ✅ A single-chip JAX model built with Flax

    ✅ A multi-chip (1×4) JAX model using shard_map for tensor parallelism

    ✅ Meta’s official PyTorch implementation from meta-llama/llama3

    ✅ The Hugging Face PyTorch transformers version of LLaMA 3.1–8B (meta-llama/Meta-Llama-3.1-8B) - is not in PR

    ❌ The Hugging Face Flax implementation was not included in the comparison because it currently does not support safetensors, which makes it incompatible with direct loading from from_pretrained() using Meta’s released weights.


## Setup Instructions



### 🌿 Branch and Directory Setup

All changes for this draft PR are in the branch:

```
Llama3.1-8B-paralel
```

Clone the repo (if you haven’t), switch to the branch, and enter the main working directory:

```
cd tests/jax/multi_chip/bounties/Llama_3.1-8B
```



### 🐍 Create and Activate Virtual Environment

Make sure you're using Python ≥3.12 (tested on 3.12):

```
python3 -m venv llama_env
source llama_env/bin/activate
pip install -r requirements.txt
```


### 🤗 Authenticate with Hugging Face

You must log into Hugging Face to download the LLaMA 3.1 weights.

```
pip install huggingface_hub
huggingface-cli login
```
Make sure you've requested access to the Meta LLaMA 3 model: https://huggingface.co/meta-llama


### 📥 Download and Organize Model Files

```
huggingface-cli download meta-llama/Llama-3.1-8B original/tokenizer.model --local-dir llama3.1-8B/original
huggingface-cli download meta-llama/Llama-3.1-8B original/consolidated.00.pth --local-dir llama3.1-8B/8B
huggingface-cli download meta-llama/Llama-3.1-8B original/params.json --local-dir llama3.1-8B/8B
```


### ▶️ Running the Scripts

You can run any of the available generation scripts using:

python3 llama/generate_single_chip.py
python3 llama/generate_multi_chip.py
python3 llama/generate_hf.py

    generate_multi_chip.py: Runs the sharded tensor-parallel JAX model (1×4 mesh).

    generate_single_chip.py: Runs the unsharded JAX model.

    generate_hf.py: Runs the Hugging Face PyTorch reference model.

To compare the token outputs from all three models, run:

```
python3 check_outputs.py
```
