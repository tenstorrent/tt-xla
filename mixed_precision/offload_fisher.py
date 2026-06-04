# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Layer-wise Fisher computation with CPU offloading for models too large to fit in GPU memory."""

import copy
import json
import os
import threading

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

NUM_SAMPLES = 100


def _make_fisher_hook(n, acc, on_cpu=True):
    """Return a post-accumulate-grad hook that accumulates squared gradients.

    on_cpu=True (default): moves gradient to CPU before accumulating.
    on_cpu=False: accumulates on the same device as the gradient.
    """

    def hook(p):
        if p.grad is not None:
            g2 = p.grad.detach().float().pow_(2)
            if on_cpu:
                g2 = g2.cpu()
            if n in acc:
                acc[n].add_(g2)
            else:
                acc[n] = g2
            p.grad = None

    return hook


def _resolve_model_path(model_name_or_path):
    """Resolve a HF model name or local path to a directory with safetensors files."""
    if os.path.isdir(model_name_or_path):
        return model_name_or_path
    from transformers.utils import cached_file

    try:
        config_file = cached_file(
            model_name_or_path, "config.json", local_files_only=True
        )
        return os.path.dirname(config_file)
    except Exception as e:
        raise RuntimeError(
            f"Cannot find model at '{model_name_or_path}'. "
            "Pass a local directory path or ensure the model is in the HF cache."
        ) from e


def _build_weight_map(model_path):
    """Return {tensor_name: shard_filename} for all tensors in the model."""
    from safetensors import safe_open

    index_path = os.path.join(model_path, "model.safetensors.index.json")
    if os.path.exists(index_path):
        with open(index_path) as f:
            return json.load(f)["weight_map"]

    single = os.path.join(model_path, "model.safetensors")
    with safe_open(single, framework="pt") as f:
        return {k: "model.safetensors" for k in f.keys()}


def _dequantize_mxfp4_inplace(raw, device):
    """Dequantize MXFP4 *_blocks/*_scales pairs to bf16 in place.

    Some MoE models (e.g. GPT-OSS-120B) store expert weights on disk in MXFP4
    format where each weight tensor is split into a packed-integer *_blocks
    tensor and a per-block *_scales tensor. Replace them with a single
    dequantized bf16 tensor.
    """
    from transformers.integrations.mxfp4 import convert_moe_packed_tensors

    for full_name in list(raw):
        if not full_name.endswith("_blocks"):
            continue
        base = full_name[: -len("_blocks")]
        scales_key = base + "_scales"
        if scales_key not in raw:
            continue
        raw[base] = convert_moe_packed_tensors(
            raw.pop(full_name).to(device),
            raw.pop(scales_key).to(device),
            dtype=torch.bfloat16,
        )


def load_tensors_to_layer(layer, prefix, weight_map, model_path, device):
    """Load all tensors whose name starts with prefix from safetensors shards to device.

    Tensors go disk → device without staging the full model in CPU RAM.
    Uses accelerate's set_module_tensor_to_device to convert meta tensors.
    """
    from accelerate.utils import set_module_tensor_to_device
    from safetensors import safe_open

    shards = {}
    for tensor_name, shard_file in weight_map.items():
        if tensor_name.startswith(prefix):
            shards.setdefault(shard_file, []).append(tensor_name)

    raw = {}
    for shard_file, tensor_names in shards.items():
        with safe_open(os.path.join(model_path, shard_file), framework="pt") as f:
            for full_name in tensor_names:
                raw[full_name] = f.get_tensor(full_name)

    _dequantize_mxfp4_inplace(raw, device)

    for full_name, tensor in raw.items():
        param_name = full_name[len(prefix) :]
        set_module_tensor_to_device(layer, param_name, device, value=tensor.to(device))


def load_model_shell(model_name_or_path):
    """Create an empty model shell for disk-based weight streaming.

    Returns (model, tokenizer, weight_map, model_path). The model has meta tensors,
    no weights are in RAM. Call load_tensors_to_layer per layer during the sweep
    to bring weights from disk to GPU one layer at a time.
    """
    from accelerate import init_empty_weights

    model_path = _resolve_model_path(model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    config = AutoConfig.from_pretrained(model_name_or_path)

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.bfloat16)
    model.eval()

    weight_map = _build_weight_map(model_path)
    return model, tokenizer, weight_map, model_path


def _run_layer(layer, hidden_states, position_ids, position_embeddings=None):
    """Run one transformer block, return the hidden_states tensor."""
    kwargs = {}
    if position_embeddings is not None:
        kwargs["position_embeddings"] = position_embeddings
    try:
        out = layer(hidden_states, position_ids=position_ids, **kwargs)
    except TypeError:
        out = layer(hidden_states, **kwargs)
    return out[0] if isinstance(out, (tuple, list)) else out


def _compute_position_embeddings(base_model, seq_len, position_ids, device):
    """Compute RoPE cos/sin for models with a model-level rotary_emb."""
    if not hasattr(base_model, "rotary_emb"):
        return None
    rotary_cls = type(base_model.rotary_emb)
    rotary_emb = rotary_cls(config=base_model.config, device=device)
    dummy = torch.zeros(1, seq_len, 1, dtype=torch.bfloat16, device=device)
    with torch.no_grad():
        position_embeddings = rotary_emb(dummy, position_ids)
    del rotary_emb, dummy
    return position_embeddings


def _barrier_reduce_and_save(shared_partials, out_dir, filename):
    """Sum fp32 partials from all GPU threads, normalize, convert to bf16, and save."""
    acc = {}
    for partial in shared_partials:
        if partial is None:
            continue
        for name, val in partial.items():
            if name in acc:
                acc[name].add_(val)
            else:
                acc[name] = val.clone()
    chunk = {n: v.div_(NUM_SAMPLES).to(torch.bfloat16) for n, v in acc.items()}
    torch.save(chunk, os.path.join(out_dir, filename))


def _sync_reduce_save(shared_partials, barrier, gpu_id, out_dir, filename, acc):
    """Barrier sync: all threads submit partials, thread 0 reduces and saves."""
    shared_partials[gpu_id] = acc
    barrier.wait()
    if gpu_id == 0:
        _barrier_reduce_and_save(shared_partials, out_dir, filename)
    barrier.wait()
    shared_partials[gpu_id] = None


def _run_forward_sweep(
    model,
    samples,
    seq_len,
    layer_list,
    weight_map,
    model_path,
    device,
    position_ids,
    position_embeddings,
    gpu_id,
):
    """Run embed + all transformer layers without gradients, return boundary activations.

    boundary_acts[s][k] is the CPU tensor output of layer k-1 (input to layer k).
    Index 0 is the embedding output; index N is the last layer's output.
    """
    N, num_s = len(layer_list), len(samples)
    boundary_acts = [[] for _ in range(num_s)]

    embed = copy.deepcopy(model.model.embed_tokens)
    load_tensors_to_layer(embed, "model.embed_tokens.", weight_map, model_path, device)
    with torch.no_grad():
        for s, input_ids in enumerate(samples):
            boundary_acts[s].append(
                embed(input_ids[:seq_len].unsqueeze(0).to(device)).cpu()
            )
    del embed
    torch.cuda.empty_cache()

    for k in range(N):
        layer_gpu = copy.deepcopy(layer_list[k])
        load_tensors_to_layer(
            layer_gpu, f"model.layers.{k}.", weight_map, model_path, device
        )
        with torch.no_grad():
            for s in range(num_s):
                act = boundary_acts[s][k].to(device)
                boundary_acts[s].append(
                    _run_layer(layer_gpu, act, position_ids, position_embeddings).cpu()
                )
        del layer_gpu
        torch.cuda.empty_cache()
        if gpu_id == 0:
            print(f"  Offload Fisher: layer {k + 1}/{N} forward", flush=True)

    return boundary_acts


def _run_lm_head_pass(
    model,
    boundary_acts,
    samples,
    seq_len,
    weight_names_set,
    weight_map,
    model_path,
    device,
):
    """Compute loss backward through norm+lm_head; return (lm_head_acc, grad_outs).

    lm_head_acc holds Fisher accumulators for lm_head weights.
    """
    N = len(boundary_acts[0]) - 1

    norm_gpu = copy.deepcopy(model.model.norm)
    load_tensors_to_layer(norm_gpu, "model.norm.", weight_map, model_path, device)
    lm_head_gpu = copy.deepcopy(model.lm_head)
    load_tensors_to_layer(lm_head_gpu, "lm_head.", weight_map, model_path, device)
    for p in lm_head_gpu.parameters():
        p.requires_grad_(True)

    lm_head_acc = {}
    handles = []
    for pname, param in lm_head_gpu.named_parameters():
        full_name = f"lm_head.{pname}"
        if full_name in weight_names_set:
            handles.append(
                param.register_post_accumulate_grad_hook(
                    _make_fisher_hook(full_name, lm_head_acc)
                )
            )

    grad_outs = []
    for s, input_ids in enumerate(samples):
        labels = input_ids[1 : seq_len + 1].unsqueeze(0).to(device)
        act = boundary_acts[s][N].to(device).requires_grad_(True)
        logits = lm_head_gpu(norm_gpu(act))
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1)
        )
        loss.backward()
        grad_outs.append(act.grad.detach().cpu())

    for h in handles:
        h.remove()
    del norm_gpu, lm_head_gpu

    return lm_head_acc, grad_outs


def _run_backward_sweep(
    layer_list,
    boundary_acts,
    grad_outs,
    position_ids,
    position_embeddings,
    weight_names_set,
    weight_map,
    model_path,
    device,
    shared_partials,
    barrier,
    out_dir,
    gpu_id,
):
    """Backward sweep layer-wise in reverse; sync and save Fisher chunk after each layer."""
    N, num_s = len(layer_list), len(boundary_acts)

    for k in range(N - 1, -1, -1):
        layer_gpu = copy.deepcopy(layer_list[k])
        load_tensors_to_layer(
            layer_gpu, f"model.layers.{k}.", weight_map, model_path, device
        )
        for p in layer_gpu.parameters():
            p.requires_grad_(True)

        # Gradients are moved to CPU immediately to avoid accumulating large GPU tensors
        layer_acc = {}
        handles = []
        for pname, param in layer_gpu.named_parameters():
            full_name = f"model.layers.{k}.{pname}"
            if full_name in weight_names_set:
                handles.append(
                    param.register_post_accumulate_grad_hook(
                        _make_fisher_hook(full_name, layer_acc)
                    )
                )

        new_grad_outs = []
        for s in range(num_s):
            act_in = boundary_acts[s][k].detach().to(device).requires_grad_(True)
            act_out = _run_layer(layer_gpu, act_in, position_ids, position_embeddings)
            act_out.backward(grad_outs[s].to(device))
            new_grad_outs.append(act_in.grad.detach().cpu())

        for h in handles:
            h.remove()

        grad_outs = new_grad_outs
        del layer_gpu
        torch.cuda.empty_cache()

        _sync_reduce_save(
            shared_partials,
            barrier,
            gpu_id,
            out_dir,
            f"chunk_layer_{k:04d}.pt",
            layer_acc,
        )
        if gpu_id == 0:
            print(f"  Offload Fisher: layer {k + 1}/{N} backward", flush=True)


def fisher_thread_worker(
    model,
    samples,
    gpu_id,
    weight_names_set,
    weight_map,
    model_path,
    out_dir,
    shared_partials,
    barrier,
):
    """Full layer-wise Fisher loop for one GPU's samples."""
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    layer_list = list(model.model.layers)

    if len(samples) == 0:
        return

    seq_len = samples[0].shape[0] - 1
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    position_embeddings = _compute_position_embeddings(
        model.model, seq_len, position_ids, device
    )

    boundary_acts = _run_forward_sweep(
        model,
        samples,
        seq_len,
        layer_list,
        weight_map,
        model_path,
        device,
        position_ids,
        position_embeddings,
        gpu_id,
    )
    lm_head_acc, grad_outs = _run_lm_head_pass(
        model,
        boundary_acts,
        samples,
        seq_len,
        weight_names_set,
        weight_map,
        model_path,
        device,
    )
    _sync_reduce_save(
        shared_partials, barrier, gpu_id, out_dir, "chunk_lmhead.pt", lm_head_acc
    )
    _run_backward_sweep(
        layer_list,
        boundary_acts,
        grad_outs,
        position_ids,
        position_embeddings,
        weight_names_set,
        weight_map,
        model_path,
        device,
        shared_partials,
        barrier,
        out_dir,
        gpu_id,
    )
