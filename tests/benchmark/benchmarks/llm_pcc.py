# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Device-vs-host PCC correctness driver for LLMs (NOT a performance benchmark).

`run_llm_pcc_e2e` runs the same model end-to-end on the host (CPU, bf16
reference) and on the Tenstorrent device (torch-xla "tt" backend, with the same
mesh/sharding the perf path uses), then compares the **prefill** and
**first-decode** output logits via PCC and asserts they meet a threshold.

Unlike `benchmark_llm_torch_xla`, this does NO warmup pass, NO timed generation,
NO tokens/sec or TTFT, and emits no benchmark result — it exists purely to
validate numerical correctness (device output ≈ host output). It reuses the
benchmark's model-setup/sharding/decode helpers verbatim so the device execution
path is identical to the perf path.
"""

import os

import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from benchmarks.llm_benchmark import (
    DEFAULT_INPUT_PROMPT,
    MODULE_EXPORT_PATH,
    _shard_kv_cache,
    construct_inputs,
    get_mesh,
    setup_model_and_tokenizer,
    transfer_to_device,
)
from llm_utils import generate_and_benchmark
from llm_utils.decode_utils import LLMSamplingWrapper
from loguru import logger
from tt_torch.sharding import sharding_constraint_hook
from tt_torch.weight_dtype import apply_weight_dtype_overrides
from utils import build_xla_export_name, compute_pcc, compute_rel_l2


def _default_read_logits_fn(output):
    return output.logits


# [EXPERIMENT] TTXLA_DIVERSE_PREFILL: 32 distinct real prompts, varied topics so
# each user's KV context differs -> diverse expert routing at every decode step.
_DIVERSE_PREFILL_PROMPTS = [
    "Explain how photosynthesis converts sunlight into chemical energy in plants.",
    "Write a short story about a lighthouse keeper who discovers a message in a bottle.",
    "Describe the main causes and consequences of the French Revolution in detail.",
    "What are the key differences between machine learning and deep learning today?",
    "Give me a recipe for a traditional Italian margherita pizza from scratch.",
    "Summarize the plot of Shakespeare's Hamlet and its central themes clearly.",
    "How does the human immune system defend the body against viral infections?",
    "Compare the economic systems of capitalism and socialism with concrete examples.",
    "Explain the theory of general relativity and how it changed modern physics.",
    "Describe the life cycle of a star from formation to its eventual collapse.",
    "What strategies help a small business grow its customer base sustainably?",
    "Write a persuasive paragraph arguing for the importance of public libraries.",
    "Explain how a blockchain achieves consensus without a central authority today.",
    "Describe the process of protein synthesis from DNA transcription to translation.",
    "What are the ethical considerations around deploying autonomous vehicles widely?",
    "Give an overview of the water cycle and its role in regulating Earth's climate.",
    "Explain the difference between a stack and a queue with practical examples.",
    "Describe how vaccines train the immune system to recognize future pathogens.",
    "What factors led to the fall of the Roman Empire over several centuries?",
    "Write a brief guide on how to start learning to play the acoustic guitar.",
    "Explain how neural networks learn through backpropagation and gradient descent.",
    "Describe the geography and biodiversity of the Amazon rainforest ecosystem.",
    "What are the benefits and risks of nuclear energy compared to fossil fuels?",
    "Summarize the scientific method and why reproducibility matters in research.",
    "Explain how supply and demand determine prices in a competitive market.",
    "Describe the cultural significance of the ancient Silk Road trade routes.",
    "What are effective techniques for improving long-term memory and recall?",
    "Explain how encryption keeps online communications secure from eavesdroppers.",
    "Describe the stages of the software development life cycle in a typical team.",
    "What are the causes of ocean acidification and its impact on marine life?",
    "Write an introduction explaining why regular exercise improves mental health.",
    "Explain how a democratic government separates legislative and judicial powers.",
]


def _maybe_diverse_prefill(input_args, tokenizer, batch_size, max_cache_len):
    """[EXPERIMENT] TTXLA_DIVERSE_PREFILL: replace the identical tiled prefill with
    `batch_size` DISTINCT real prompts so every user's KV context differs (diverse
    routing at every decode step, vs TTXLA_DIVERSE_BATCH which only swaps the single
    decode token on an identical context). Real prompts keep routing natural,
    avoiding the 0-token-EP-device hang that garbage tokens could re-trigger.
    Truncated to the common minimum token length so there is NO padding (every row
    ends on a real token, preserving the "[-1] logit = last real token" PCC
    assumption). Mutates input_args in place. Deterministic (same prompts+tokenizer)
    so the CPU-reference and device paths get identical inputs -> PCC is
    like-for-like. No-op unless the env is set."""
    if not os.environ.get("TTXLA_DIVERSE_PREFILL"):
        return
    bs = int(batch_size)
    sel = [_DIVERSE_PREFILL_PROMPTS[i % len(_DIVERSE_PREFILL_PROMPTS)] for i in range(bs)]
    tok = [tokenizer(p, return_tensors="pt")["input_ids"][0] for p in sel]
    min_len = min(int(t.shape[0]) for t in tok)
    min_len = min(min_len, int(max_cache_len))
    ids = torch.stack([t[:min_len] for t in tok], dim=0).to(
        input_args["input_ids"].dtype
    )
    input_args["input_ids"] = ids
    input_args["cache_position"] = torch.arange(0, min_len)
    logger.warning(
        f"[EXPERIMENT] diverse PREFILL: {bs} distinct real prompts, "
        f"truncated to common len={min_len} tokens (no padding)"
    )


def run_llm_pcc_e2e(
    model_loader,
    model_variant,
    display_name,
    *,
    optimization_level,
    batch_size,
    input_sequence_length,
    mesh_config_fn,
    shard_spec_fn,
    required_pcc,
    experimental_weight_dtype="",
    experimental_kv_cache_dtype=None,
    experimental_enable_permute_matmul_fusion=False,
    read_logits_fn=_default_read_logits_fn,
    input_output_sharding_spec=None,
    kv_cache_sharding_spec=None,
    experts_implementation=None,
    weight_dtype_overrides=None,
    fp32_dest_acc_en=None,
    use_mla_cache=False,
    decode_only=False,
) -> dict:
    """Run the model on host and device and assert prefill/decode logit PCC.

    Returns a dict with the measured ``prefill_pcc`` / ``decode_pcc`` (and their
    rel-L2). Raises AssertionError if either PCC is below ``required_pcc``.
    """
    xr.set_device_type("TT")

    # [EXPERIMENT] TTXLA_ROUTING_ONLY is a CPU-only diagnostic (it prints per-layer
    # EP device coverage and returns before the device run). Skip ALL device/mesh
    # init so it can run without opening the galaxy (e.g. while another run is
    # tearing a hung mesh down).
    _routing_only = bool(os.environ.get("TTXLA_ROUTING_ONLY"))

    # Enable SPMD sharding for the multi-chip (mesh) path, exactly like the
    # perf benchmark does.
    is_multichip = False
    device = None
    if not _routing_only:
        if mesh_config_fn is not None and shard_spec_fn is not None:
            is_multichip = xr.global_runtime_device_count() > 1
            if is_multichip:
                os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
                xr.use_spmd()

    max_cache_len = input_sequence_length
    if not _routing_only:
        device = torch_xla.device()

    model, tokenizer = setup_model_and_tokenizer(
        model_loader,
        model_variant,
        experts_implementation=experts_implementation,
    )

    input_args = construct_inputs(
        tokenizer,
        model.config,
        batch_size,
        max_cache_len,
        use_mla_cache=use_mla_cache,
    )

    _maybe_diverse_prefill(input_args, tokenizer, batch_size, max_cache_len)

    # ------------------------------------------------------------------
    # HOST (CPU) reference: prefill logits + first-decode logits.
    # tt_* experts/attention backends auto-fall-back to HF builtins for CPU
    # tensors, so no backend swap is needed here.
    # ------------------------------------------------------------------
    cpu_wrapper = LLMSamplingWrapper(model, read_logits_fn, return_logits=True)
    cpu_wrapper.eval()

    # [EXPERIMENT] TTXLA_ROUTING_ONLY: hook each layer's MoE router to capture the
    # decode token's top-k expert selection (CPU), then (below) print EP-4 device
    # coverage and skip the device run. Confirms the zero-token-device hang: a
    # tiled-identical batch makes all tokens pick the same top-k experts; if those
    # don't span all 4 EP devices, the empty device(s) deadlock moe_compute combine.
    _routing_capture = {}
    if os.environ.get("TTXLA_ROUTING_ONLY"):
        _rbase = getattr(model, "model", model)
        def _mk_router_hook(li):
            def _h(mod, inp, out):
                _routing_capture[li] = (
                    out[0] if isinstance(out, (tuple, list)) else out
                ).detach()
            return _h
        for _li, _layer in enumerate(_rbase.layers):
            _layer.mlp.router.register_forward_hook(_mk_router_hook(_li))

    # Iter 0: prefill. Always run — besides providing the prefill PCC reference,
    # it populates the KV cache (decode_only_cache) that the decode step / the
    # decode-only device path reads from. Advances input_args to the post-prefill
    # decode state (input_ids = first predicted token).
    cpu_prefill_logits, _ = generate_and_benchmark(
        cpu_wrapper,
        input_args,
        torch.device("cpu"),
        1,
        verbose=False,
        collect_logits=True,
    )

    # Snapshot first-decode inputs before the CPU decode step advances them.
    first_decode_input_ids = input_args["input_ids"].clone()
    # [EXPERIMENT] TTXLA_FORCE_DECODE_TOKEN=<id>: overwrite the first-decode token
    # here at the SOURCE (in-place on the CPU snapshot) so it flows through the
    # normal device-transfer path (first_decode_input_ids.to(device)) without
    # creating a new graph input -> no recompile/fabric-reinit. Deconfounds the
    # 2-layer MoE hang: is it the (differing) decode token/routing (c) or the
    # static program state (a/b)? PCC will be garbage; we only watch for hang.
    _forced_tok = os.environ.get("TTXLA_FORCE_DECODE_TOKEN")
    if _forced_tok:
        first_decode_input_ids.fill_(int(_forced_tok))
        logger.warning(f"[EXPERIMENT] forced first-decode token -> {_forced_tok}")
    # [EXPERIMENT] TTXLA_DIVERSE_BATCH: replace the tiled-identical decode batch
    # with BATCH distinct tokens so their top-k experts span all EP devices (no
    # zero-token device) -> should NOT hang, proving the zero-token-device
    # mechanism. Applied to both first_decode_input_ids (device decode) and
    # input_args (CPU decode / routing check). PCC will be garbage.
    if os.environ.get("TTXLA_DIVERSE_BATCH") and not os.environ.get(
        "TTXLA_DIVERSE_PREFILL"
    ):
        _bs = first_decode_input_ids.shape[0]
        _vocab = int(getattr(model.config, "vocab_size", 201088))
        _step = max(1, _vocab // (_bs + 2))
        _div = ((torch.arange(_bs) + 1) * _step).remainder(_vocab)
        _div = _div.to(first_decode_input_ids.dtype).reshape(
            _bs, *first_decode_input_ids.shape[1:]
        )
        first_decode_input_ids.copy_(_div)
        input_args["input_ids"] = _div.clone()
        logger.warning(
            f"[EXPERIMENT] diverse batch: {_bs} distinct decode tokens "
            f"(e.g. {_div.flatten().tolist()[:6]}...)"
        )
    decode_only_cache_position = input_args["cache_position"].clone()
    decode_only_cache = input_args["past_key_values"]

    # Iter 1: first decode. Provides the PCC reference for the device decode.
    cpu_decode_logits, _ = generate_and_benchmark(
        cpu_wrapper,
        input_args,
        torch.device("cpu"),
        1,
        verbose=False,
        collect_logits=True,
    )

    # Layout: [prefill_logits, first_decode_logits]. [-1] is always the
    # first-decode step (used as the decode PCC reference in both modes).
    cpu_output_logits = cpu_prefill_logits + cpu_decode_logits

    if os.environ.get("TTXLA_ROUTING_ONLY"):
        _ne = int(getattr(model.config, "num_local_experts", 32))
        _k = int(getattr(model.config, "num_experts_per_tok", 4))
        _ep = 4  # cluster_axis=0 (EP) size on the (4,8) galaxy
        _per = _ne // _ep  # experts per device (contiguous EP shard)
        print(f"[ROUTING] num_experts={_ne} top_k={_k} EP={_ep} experts/device={_per}")
        for _li in sorted(_routing_capture):
            _lg = _routing_capture[_li].reshape(-1, _ne)  # [rows, experts]
            _topk = _lg.float().topk(_k, dim=-1).indices  # [rows, k]
            # device coverage = union over ALL batch rows (a device gets tokens if
            # ANY row selects an expert it owns). tiled batch -> == row 0.
            _covered = sorted(set((_topk // _per).flatten().tolist()))
            _missing = [d for d in range(_ep) if d not in _covered]
            _row0 = _topk[0].tolist()
            print(
                f"[ROUTING] layer {_li}: row0_top{_k}={_row0} "
                f"covered_devices={_covered} ZERO_TOKEN_DEVICES={_missing}"
            )
        logger.warning("[ROUTING] TTXLA_ROUTING_ONLY set; skipping device run.")
        return {"routing_only": True}

    # ------------------------------------------------------------------
    # DEVICE setup: transfer + shard the model identically to the perf path.
    # ------------------------------------------------------------------
    model = model.to(device, dtype=torch.bfloat16)

    mesh = None
    if is_multichip:
        shard_specs = shard_spec_fn(model_loader, model)
        mesh = get_mesh(model_loader, mesh_config_fn)
        # Register the mesh globally so mesh-aware experts backends (tt_moe,
        # tt_moe_fused) can read it via torch_xla get_global_mesh().
        xs.set_global_mesh(mesh)
        if shard_specs is not None:
            for tensor, shard_spec in shard_specs.items():
                xs.mark_sharding(tensor, mesh, shard_spec)

        # All-gather lm_head logits so the host-side comparison sees full logits.
        if hasattr(model, "lm_head") and model.lm_head is not None:
            hook = sharding_constraint_hook(model.lm_head, mesh, (None, None, None))
            model.lm_head.register_forward_hook(hook)

    export_model_name = build_xla_export_name(
        model_name=display_name + "_pcc",
        num_layers=getattr(model_loader, "num_layers", None),
        batch_size=batch_size,
        input_sequence_length=input_sequence_length,
    )
    options = {
        "optimization_level": optimization_level,
        # Correctness test: trace disabled, no perf metrics collection.
        "enable_trace": False,
        "export_path": MODULE_EXPORT_PATH,
        **({"dry_run": True} if os.environ.get("TTXLA_DRY_RUN") else {}),
        **({"export_tensors": True} if os.environ.get("TTXLA_EXPORT_TENSORS") else {}),
        # Diagnostic: disable tt-mlir const-eval so per-layer weight-prep runs
        # inline instead of being hoisted into cached const-eval buffers (tests
        # whether the >1-layer decode collapse is a const-eval buffer issue).
        **(
            {"enable_const_eval": False}
            if os.environ.get("TTXLA_DISABLE_CONST_EVAL")
            else {}
        ),
        "export_model_name": export_model_name,
        "experimental_weight_dtype": experimental_weight_dtype,
        "experimental_enable_permute_matmul_fusion": experimental_enable_permute_matmul_fusion,
    }
    if fp32_dest_acc_en is not None:
        options["fp32_dest_acc_en"] = fp32_dest_acc_en
    if experimental_kv_cache_dtype is not None:
        options["experimental-kv-cache-dtype"] = experimental_kv_cache_dtype
    torch_xla.set_custom_compile_options(options)

    if weight_dtype_overrides:
        apply_weight_dtype_overrides(model, weight_dtype_overrides)
    else:
        weight_dtype_config = model_loader.get_weight_dtype_config_path()
        if weight_dtype_config:
            apply_weight_dtype_overrides(model, weight_dtype_config)

    # ------------------------------------------------------------------
    # DEVICE run: prefill logits + first-decode logits (no warmup, no timing).
    # ------------------------------------------------------------------
    logits_wrapper = LLMSamplingWrapper(
        model,
        read_logits_fn,
        return_logits=True,
        mesh=mesh,
        output_sharding_spec=input_output_sharding_spec,
    )
    logits_wrapper.eval()
    compiled_logits = torch.compile(logits_wrapper, backend="tt")

    input_args = construct_inputs(
        tokenizer,
        model.config,
        batch_size,
        max_cache_len,
        past_key_values=decode_only_cache if decode_only else None,
        use_mla_cache=use_mla_cache,
    )
    if decode_only:
        input_args["input_ids"] = first_decode_input_ids.clone()
        input_args["cache_position"] = decode_only_cache_position.clone()
    else:
        # Apply the SAME diverse prefill as the CPU reference above so device and
        # CPU prefill on identical per-user inputs (PCC compares like-for-like).
        _maybe_diverse_prefill(input_args, tokenizer, batch_size, max_cache_len)

    input_args = transfer_to_device(input_args, device)
    if is_multichip:
        _shard_kv_cache(input_args["past_key_values"], mesh, kv_cache_sharding_spec)
    if input_output_sharding_spec:
        xs.mark_sharding(input_args["input_ids"], mesh, input_output_sharding_spec)

    device_prefill_logits = []
    if not decode_only:
        print("PCC test: running device prefill...")
        device_prefill_logits, _ = generate_and_benchmark(
            compiled_logits,
            input_args,
            device,
            1,
            verbose=False,
            collect_logits=True,
        )

        # Keep the first-decode input aligned with the CPU reference: if the
        # device prefill argmax diverged from CPU's, feed CPU's token so the
        # decode PCC compares like-for-like (a poor prefill PCC otherwise
        # compounds into the decode PCC — see tt-xla #4614).
        device_prefill_output_ids = input_args["input_ids"].to("cpu")
        if not torch.equal(device_prefill_output_ids, first_decode_input_ids.cpu()):
            logger.warning(
                "Device prefill produced different tokens than CPU prefill; "
                "using CPU prefill output as the decode PCC reference input."
            )
            input_args["input_ids"] = first_decode_input_ids.to(device)
            if input_output_sharding_spec:
                xs.mark_sharding(
                    input_args["input_ids"], mesh, input_output_sharding_spec
                )

    print("PCC test: running device first-decode...")
    device_decode_logits, _ = generate_and_benchmark(
        compiled_logits,
        input_args,
        device,
        1,
        verbose=False,
        collect_logits=True,
    )

    output_logits = (
        device_decode_logits
        if decode_only
        else device_prefill_logits + device_decode_logits
    )

    # ------------------------------------------------------------------
    # Compare prefill + decode logit PCC (device vs host) and assert.
    # ------------------------------------------------------------------
    results = {}
    if not decode_only:
        prefill_pcc = compute_pcc(output_logits[0][0], cpu_output_logits[0][0])
        prefill_rel_l2 = compute_rel_l2(cpu_output_logits[0][0], output_logits[0][0])
        results["prefill_pcc"] = prefill_pcc
        results["prefill_rel_l2"] = prefill_rel_l2
        print(
            f"Prefill PCC = {prefill_pcc:.6f} (required {required_pcc}), "
            f"rel_l2 = {prefill_rel_l2:.6e}"
        )

    # cpu_output_logits[-1] / output_logits[-1] is the first-decode step in both
    # the prefill+decode ([1]) and decode-only ([0]) layouts.
    decode_pcc = compute_pcc(output_logits[-1][0], cpu_output_logits[-1][0])
    decode_rel_l2 = compute_rel_l2(cpu_output_logits[-1][0], output_logits[-1][0])
    results["decode_pcc"] = decode_pcc
    results["decode_rel_l2"] = decode_rel_l2
    print(
        f"Decode PCC = {decode_pcc:.6f} (required {required_pcc}), "
        f"rel_l2 = {decode_rel_l2:.6e}"
    )

    # [DIAGNOSTIC] The PCC above is user-0-only (output_logits[-1][0]). Under
    # TTXLA_DIVERSE_BATCH each batch row is a DISTINCT input, so user 0 is NOT
    # representative — a fused-MoE bug that only corrupts some users' routing
    # would pass a user-0 check. Report per-user decode PCC (device vs CPU on the
    # SAME per-user input) across the whole batch: min/mean/pooled + the full list.
    _dev_dec = output_logits[-1]
    _ref_dec = cpu_output_logits[-1]
    _bs = _dev_dec.shape[0]
    _per_user = [compute_pcc(_dev_dec[u], _ref_dec[u]) for u in range(_bs)]
    _pooled = compute_pcc(_dev_dec, _ref_dec)
    _min_u = min(range(_bs), key=lambda u: _per_user[u])
    results["decode_pcc_per_user_min"] = _per_user[_min_u]
    results["decode_pcc_pooled"] = _pooled
    print(
        f"Decode PCC per-user: min={_per_user[_min_u]:.6f} (user {_min_u}), "
        f"mean={sum(_per_user)/_bs:.6f}, pooled(all {_bs} users)={_pooled:.6f}"
    )
    print(
        "  per-user PCC: "
        + " ".join(f"{u}:{_per_user[u]:.3f}" for u in range(_bs))
    )
    # per-user rel_l2 (scale-aware abs error) + top-1 next-token agreement. A low
    # PCC on a FLAT/ambiguous next-token distribution (short prompts) can coexist
    # with small rel_l2 and a matching argmax -> that is PCC sensitivity, NOT a
    # compute error. A real fused-MoE bug shows large rel_l2 AND argmax mismatch.
    _per_user_rl2 = [
        compute_rel_l2(_ref_dec[u], _dev_dec[u]) for u in range(_bs)
    ]
    _dev_top1 = _dev_dec[:, -1, :].to(torch.float32).argmax(dim=-1)
    _ref_top1 = _ref_dec[:, -1, :].to(torch.float32).argmax(dim=-1)
    _match = int((_dev_top1 == _ref_top1).sum())
    results["decode_top1_agreement"] = _match / _bs
    print(
        f"Decode top-1 next-token agreement (device argmax == CPU argmax): "
        f"{_match}/{_bs} users"
    )
    print(
        "  per-user rel_l2: "
        + " ".join(f"{u}:{_per_user_rl2[u]:.2f}" for u in range(_bs))
    )
    _low = [u for u in range(_bs) if _per_user[u] < required_pcc]
    if _low:
        print(
            "  users below required_pcc: "
            + " ".join(
                f"u{u}(pcc={_per_user[u]:.3f},rl2={_per_user_rl2[u]:.2f},"
                f"top1{'=' if _dev_top1[u]==_ref_top1[u] else '!'}CPU)"
                for u in _low
            )
        )

    # ------------------------------------------------------------------
    # Human-readable host vs device outputs (predicted tokens -> text).
    # The batch is the prompt tiled batch_size times (identical rows), so row 0
    # is representative. "next token" at a step = argmax of the last position.
    # Printed before the asserts so the outputs are visible even on a PCC failure.
    # ------------------------------------------------------------------
    def _next_token(step_logits):
        tid = int(step_logits[0, -1, :].to(torch.float32).argmax(dim=-1).item())
        return tid, tokenizer.decode([tid])

    print(
        f"[{display_name}] outputs (HOST=CPU reference, DEVICE=tt); "
        f"input prompt (tiled x{batch_size}): {DEFAULT_INPUT_PROMPT!r}"
    )
    host_prefill_txt = device_prefill_txt = ""
    if not decode_only:
        h_id, host_prefill_txt = _next_token(cpu_output_logits[0])
        d_id, device_prefill_txt = _next_token(output_logits[0])
        print(
            f"  PREFILL next-token  HOST: id={h_id} {host_prefill_txt!r}   "
            f"DEVICE: id={d_id} {device_prefill_txt!r}   match={h_id == d_id}"
        )
    h_id2, host_decode_txt = _next_token(cpu_output_logits[-1])
    d_id2, device_decode_txt = _next_token(output_logits[-1])
    print(
        f"  DECODE  next-token  HOST: id={h_id2} {host_decode_txt!r}   "
        f"DEVICE: id={d_id2} {device_decode_txt!r}   match={h_id2 == d_id2}"
    )
    print(
        "  HOST   text: "
        f"{(DEFAULT_INPUT_PROMPT + host_prefill_txt + host_decode_txt)!r}"
    )
    print(
        "  DEVICE text: "
        f"{(DEFAULT_INPUT_PROMPT + device_prefill_txt + device_decode_txt)!r}"
    )

    if not decode_only:
        assert (
            results["prefill_pcc"] >= required_pcc
        ), f"Prefill PCC failed: {results['prefill_pcc']:.6f} < {required_pcc}"
    assert (
        decode_pcc >= required_pcc
    ), f"Decode PCC failed: {decode_pcc:.6f} < {required_pcc}"

    print(
        f"[{display_name}] device-vs-host PCC PASSED "
        f"(prefill={results.get('prefill_pcc', 'n/a')}, decode={decode_pcc:.6f})"
    )
    return results
