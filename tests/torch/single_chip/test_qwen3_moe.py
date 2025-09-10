import os
import torch
import torch.nn as nn
from infra.comparators.torch_comparator import TorchComparator
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer
from tests.infra.comparators.comparison_config import (
    AtolConfig,
    ComparisonConfig,
    PccConfig,
)
import pytest

from infra.comparators.torch_comparator import TorchComparator
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock


def config_modify(config):
    def set_if_has(o, k, v):
        if hasattr(o, k):
            print(f"set {o} to {k} to {v}")
            setattr(o, k, v)
    
    overrides = {
        "num_experts": 4,
        "num_experts_per_tok": 1,
        "moe_intermediate_size": 256,
    }
    for k, v in overrides.items():
        set_if_has(config, k, v)
    return config

def prune_experts_inplace(model, keep_idx=(0,1,2,3)):
    keep_idx = list(keep_idx)
    for layer in model.model.layers:
        mlp = layer.mlp
        old = mlp.experts
        new = nn.ModuleList([old[i] for i in keep_idx])
        mlp.experts = new
        old_gate = mlp.gate
        new_gate = nn.Linear(old_gate.in_features, len(keep_idx), bias=False,
                             dtype=old_gate.weight.dtype, device=old_gate.weight.device)
        with torch.no_grad():
            new_gate.weight.copy_(old_gate.weight[keep_idx, :])
        mlp.gate = new_gate

    if hasattr(model.config, "num_experts"): model.config.num_experts = len(keep_idx)

def test_qwen3_moe_block_only():
    model_id = "Qwen/Qwen3-30B-A3B"
    config = AutoConfig.from_pretrained(model_id)
    block = Qwen3MoeSparseMoeBlock(config)
    block = block.to(torch.bfloat16)
    block.eval()
    block = block.to("xla")
    # torch._dynamo.config.capture_dynamic_output_shape_ops = True

    # compiled = torch.compile(block, backend="tt", dynamic=False, fullgraph=True)

    x = torch.randn(1, 4, config.hidden_size, dtype=torch.bfloat16).to("xla")

    with torch.no_grad():
        # outputs = compiled(x,)
        outputs = block(x,)

    print("outputs: ", outputs)

def test_qwen3_moe_model():
    model_id = "Qwen/Qwen3-30B-A3B"
    config = AutoConfig.from_pretrained(model_id)
    # config = config_modify(config)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, config=config)
    # prune_experts_inplace(model, keep_idx=(0,1,2,3,4,5,6,7))
    model.eval()
    model = model.to("xla")
    torch._dynamo.config.capture_dynamic_output_shape_ops = True
    compiled = torch.compile(model.forward, backend="tt", dynamic=False)

    # generated_ids = compiled(
    #     **model_inputs)

    # print(generated_ids)
    
    prompt = "Hi what is your name?"
    messages = [
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to("xla")


    with torch.no_grad():
        outputs = compiled(**model_inputs, return_dict=True, use_cache=False)

    print("generated_ids: ", outputs)

    # generated_ids = model.generate(
    #     **model_inputs,
    #     max_new_tokens=16384
    # )
    # output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

    # content = tokenizer.decode(output_ids, skip_special_tokens=True)

    # print("content:", content)
