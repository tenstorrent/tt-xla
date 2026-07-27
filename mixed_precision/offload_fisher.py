# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Layer-wise Fisher computation with CPU offloading for models too large to fit in GPU memory.

Requires model.embed_tokens, model.layers, model.norm, lm_head (e.g. Llama, Qwen, GPT-OSS).
"""

import copy
import os
import threading

import torch
import torch.nn as nn
from utils import _compute_position_embeddings, _run_layer, load_tensors_to_layer

NUM_SAMPLES = 100


def _make_fisher_hook(param_name, fisher_accumulator, on_cpu=True):
    """Return a post-accumulate-grad hook that accumulates squared gradients.

    on_cpu=True (default): moves gradient to CPU before accumulating.
    on_cpu=False: accumulates on the same device as the gradient.
    """

    def hook(p):
        if p.grad is None:
            return
        grad_sq = p.grad.detach().float().pow_(2)
        if on_cpu:
            grad_sq = grad_sq.cpu()
        if param_name in fisher_accumulator:
            fisher_accumulator[param_name].add_(grad_sq)
        else:
            fisher_accumulator[param_name] = grad_sq
        p.grad = None

    return hook


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
    lm_base,
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
    load_tensors_to_layer(
        embed, f"{lm_base}embed_tokens.", weight_map, model_path, device
    )
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
            layer_gpu, f"{lm_base}layers.{k}.", weight_map, model_path, device
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
    lm_base,
    lm_head_base,
    device,
):
    """Compute loss backward through norm+lm_head; return (lm_head_acc, grad_outs).

    lm_head_acc holds Fisher accumulators for lm_head weights.
    """
    N = len(boundary_acts[0]) - 1

    norm_gpu = copy.deepcopy(model.model.norm)
    load_tensors_to_layer(norm_gpu, f"{lm_base}norm.", weight_map, model_path, device)
    lm_head_gpu = copy.deepcopy(model.lm_head)

    # Tied embeddings: lm_head.weight is not stored separately in safetensors.
    # Load embed_tokens weight instead so the forward pass works, but skip Fisher hooks.
    lm_head_tied = not any(k.startswith(f"{lm_head_base}lm_head.") for k in weight_map)
    if lm_head_tied:
        load_tensors_to_layer(
            lm_head_gpu, f"{lm_base}embed_tokens.", weight_map, model_path, device
        )
    else:
        load_tensors_to_layer(
            lm_head_gpu, f"{lm_head_base}lm_head.", weight_map, model_path, device
        )
    for p in lm_head_gpu.parameters():
        p.requires_grad_(True)

    lm_head_acc = {}
    handles = []
    if not lm_head_tied:
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
    lm_base,
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
            layer_gpu, f"{lm_base}layers.{k}.", weight_map, model_path, device
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
    lm_base,
    lm_head_base,
    out_dir,
    shared_partials,
    barrier,
    errors,
):
    """Full layer-wise Fisher loop for one GPU's samples."""
    try:
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
            lm_base,
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
            lm_base,
            lm_head_base,
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
            lm_base,
            device,
            shared_partials,
            barrier,
            out_dir,
            gpu_id,
        )
    except Exception as exc:
        errors[gpu_id] = exc
        try:
            barrier.abort()
        except Exception:
            pass
