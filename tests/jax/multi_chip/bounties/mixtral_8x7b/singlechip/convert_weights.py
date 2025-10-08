# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
# from singlechip.flaxconfigmixtral import MixtralConfig
import jax
import jax.numpy as jnp
from singlechip.flaxmixtral import FlaxMixtralForCausalLM


def convert_weights(modelTorch, configTorch):
    embeddings = modelTorch.model.embed_tokens.weight.detach().cpu().numpy()
    # Save LM head
    lm_head = modelTorch.lm_head.weight.detach().cpu().numpy()
    newmodel = FlaxMixtralForCausalLM(configTorch)
    newmodel.model.embed_tokens.embedding.value = jnp.array(embeddings)
    newmodel.lm_head.kernel.value = jnp.array(lm_head.T)

    final_norm = modelTorch.model.norm.weight.detach().cpu().numpy()
    newmodel.model.norm.weight.value = jnp.array(final_norm)

    for i in range(configTorch.num_hidden_layers):
        layer = modelTorch.model.layers[i]
        layerJax = newmodel.model.layers[i]
        input_layernorm = layer.input_layernorm.weight.detach().cpu().numpy()
        post_attention_layernorm = (
            layer.post_attention_layernorm.weight.detach().cpu().numpy()
        )

        attn_q = layer.self_attn.q_proj.weight.detach().cpu().numpy()
        attn_k = layer.self_attn.k_proj.weight.detach().cpu().numpy()
        attn_v = layer.self_attn.v_proj.weight.detach().cpu().numpy()
        attn_o = layer.self_attn.o_proj.weight.detach().cpu().numpy()

        layerJax.input_norm.weight.value = jnp.array(input_layernorm)
        layerJax.attn_norm.weight.value = jnp.array(post_attention_layernorm)

        layerJax.attn.q_proj.kernel.value = jnp.array(attn_q.T)
        layerJax.attn.k_proj.kernel.value = jnp.array(attn_k.T)
        layerJax.attn.v_proj.kernel.value = jnp.array(attn_v.T)
        layerJax.attn.o_proj.kernel.value = jnp.array(attn_o.T)

        moe = layer.block_sparse_moe
        moe_gate = moe.gate.weight.detach().cpu().numpy()
        layerJax.block_sparse_moe.gate.kernel.value = jnp.array(moe_gate.T)
        num_experts = configTorch.num_local_experts

        for j in range(num_experts):
            w1 = moe.experts[j].w1.weight.detach().cpu().numpy()
            w2 = moe.experts[j].w2.weight.detach().cpu().numpy()
            w3 = moe.experts[j].w3.weight.detach().cpu().numpy()
            expert = getattr(layerJax.block_sparse_moe, f"experts_{j}")

            expert.gate_proj.kernel.value = jnp.array(w3.T)
            expert.up_proj.kernel.value = jnp.array(w1.T)
            expert.down_proj.kernel.value = jnp.array(w2.T)

    return newmodel
