# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
#
# Same run as test_mrope.py, but with python-level monkey-patches that remove the
# graph breaks found in logs/final_debug.log (see logs/graph_break_report.md).
#
# TWO confirmed true graph breaks, both aten._local_scalar_dense (data-dependent):
#
#   (1) DOMINANT, 197x, PREFILL — attention.py:237  mixed_qkv = mixed_qkv[:num_actual]
#       num_actual = md.num_actual_tokens is a leaked numpy.int32 (model_runner.py:654
#       sets num_actual_tokens=total_num_scheduled_tokens WITHOUT the int() cast its
#       sibling fields have). Dynamo tensorizes the numpy scalar, so the slice breaks.
#       Patch: wrap TTModelRunner._build_per_layer_attn_metadata to int()-coerce the
#       num_actual_tokens field on the built GDN metadata.
#
#   (2) 48x, DECODE — conv1d.py:175  slot = int(conv_state_indices[t]) -> Tensor.item()
#       (reported at torch_overrides.py:66 because a global TorchFunctionMode routes the
#       int() through its catch-all). The decode conv update never got the branchless
#       single-token fast path the prefill twin tt_causal_conv1d_fn already has.
#       Patch: replace tt_causal_conv1d_update with a branchless index_select /
#       index_copy_ implementation that consumes conv_state_indices as a tensor.
import pytest
import vllm
from conftest import assert_output_coherent, check_host_memory


def _install_num_actual_tokens_patch():
    """Fix break (1): coerce GDN metadata num_actual_tokens to a python int so the
    attention.py:237 `mixed_qkv[:num_actual]` slice bound is not a tensorized numpy
    scalar. Mirrors the missing int() cast at model_runner.py:654."""
    from vllm_tt.model_runner import TTModelRunner

    orig = TTModelRunner._build_per_layer_attn_metadata

    def wrapped(self, *args, **kwargs):
        md_dict = orig(self, *args, **kwargs)
        if isinstance(md_dict, dict):
            for md in md_dict.values():
                n = getattr(md, "num_actual_tokens", None)
                if n is not None and not isinstance(n, int):
                    md.num_actual_tokens = int(n)
        return md_dict

    TTModelRunner._build_per_layer_attn_metadata = wrapped


def _install_gdn_graphbreak_patches():
    """Monkey-patch the GDN decode conv1d update to be branchless (no .item())."""
    import torch
    from vllm_tt.layers.gdn import conv1d as _conv1d
    from vllm_tt.layers.gdn import attention as _attention
    from vllm_tt.layers import gdn as _gdn_pkg

    def tt_causal_conv1d_update(
        x, conv_state, weight, bias, activation, conv_state_indices
    ):
        """Branchless decode conv1d update — mirrors the prefill num_seqs==1 path.

        Reads conv_state_indices only via index_select / index_copy_ (tensor ops),
        never int(...), so dynamo does not break the graph. Works for any
        num_tokens (including the max_num_seqs=1 decode case of 1).
        """
        conv_dim, K = weight.shape
        # Gather [T, conv_dim, K-1] left context, one slot per token.
        state = conv_state.index_select(0, conv_state_indices).to(x.dtype)
        # Append the new token -> [T, conv_dim, K].
        window = torch.cat([state, x.unsqueeze(-1)], dim=-1)
        # Depthwise dot over the K window (weight broadcasts over the token dim).
        y = (weight.unsqueeze(0) * window).sum(dim=-1)  # [T, conv_dim]
        if bias is not None:
            y = y + bias
        out = _conv1d._apply_activation(y, activation)
        # Shift each window left by one and scatter back to the originating slots.
        if K > 1:
            conv_state.index_copy_(
                0, conv_state_indices, window[:, :, 1:].to(conv_state.dtype)
            )
        return out

    # Patch every namespace the symbol is looked up from. _core_decode in
    # attention.py resolves the name from the attention module globals, so that
    # binding is the one that actually matters; the others keep things consistent.
    _conv1d.tt_causal_conv1d_update = tt_causal_conv1d_update
    _attention.tt_causal_conv1d_update = tt_causal_conv1d_update
    _gdn_pkg.tt_causal_conv1d_update = tt_causal_conv1d_update


@pytest.mark.push
@pytest.mark.single_device
def test_mrope_fixed():
    _install_num_actual_tokens_patch()  # break (1): prefill, attention.py:237
    _install_gdn_graphbreak_patches()   # break (2): decode, conv1d.py:175

    prompts = [
        "Continue in English: I like taking walks in the",
    ]
    sampling_params = vllm.SamplingParams(temperature=0.8, top_p=0.95, max_tokens=32)
    model_name = "Qwen/Qwen3.6-27B"

    llm_args = {
        "model": model_name,
        "max_num_batched_tokens": 512,
        "max_num_seqs": 1,
        "max_model_len": 32,
        "gpu_memory_utilization": 0.2,
        "limit_mm_per_prompt": {"image": 0, "video": 0, "audio": 0},
        "additional_config": {
            "min_context_len": 32,
            "enable_tensor_parallel": True,
            "use_2d_mesh": True,
        },
    }
    llm = vllm.LLM(**llm_args)

    output_text = llm.generate(prompts, sampling_params)[0].outputs[0].text
    print(f"prompt: {prompts[0]}, output: {output_text}")
    assert_output_coherent(output_text)

    check_host_memory(model_name)
