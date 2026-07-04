import gc
import math
from unittest.mock import patch

import torch
from transformers import AutoProcessor, DiffusionGemmaForBlockDiffusion
from loguru import logger

from third_party.tt_forge_models.diffusiongemma.pytorch import ModelLoader

MODEL_ID = "google/diffusiongemma-26B-A4B-it"
MAX_NEW_TOKENS = 512


@torch.no_grad()
def manual_generate(model, input_ids, attention_mask, max_new_tokens, **model_kwargs):
    """With-cache CPU mimic of DiffusionGemma's block-diffusion ``generate()``.

    Faithfully replicates generate()'s driver -- encoder prefill -> KV cache ->
    per-canvas denoising loop -- and REUSES the model's own helper methods, so the
    only pieces that are "ours" (and swappable for a TT run) are the encoder/decoder
    forward calls. Mirrors the is_compiling=False (eager, DynamicCache) branch of
    generate(). See generation_diffusion_gemma.py::generate.
    """
    # 0. Setup (mirrors generate()'s section 0).
    gen_cfg, model_kwargs = model._prepare_generation_config(
        None, max_new_tokens=max_new_tokens, **model_kwargs
    )
    batch_size, cur_len = input_ids.shape
    max_length, max_new_tokens = model._prepare_generated_length(gen_cfg, cur_len)
    max_new_canvases = math.ceil(max_new_tokens / model.config.canvas_length)

    device = input_ids.device
    canvas_length = model.config.canvas_length
    finished_sequences = torch.zeros(batch_size, dtype=torch.bool, device=device)
    past_key_values = model._prepare_cache_for_generation(
        generation_config=gen_cfg,
        batch_size=batch_size,
        max_length=max_length - canvas_length,  # the last canvas isn't cached
    )
    eos_tensor = (
        torch.tensor(gen_cfg.eos_token_id, device=device)
        if gen_cfg.eos_token_id is not None
        else None
    )
    encoder_position_ids = torch.arange(
        cur_len - input_ids.shape[1], cur_len, dtype=torch.int32, device=device
    ).unsqueeze(0)
    decoder_position_ids = torch.arange(
        cur_len, cur_len + canvas_length, dtype=torch.int32, device=device
    ).unsqueeze(0)

    # is_compiling=False branch: full forward as the decoder, encoder module for the
    # prefill, canvas fully visible in the decoder mask. These two forwards are what
    # a TT run would compile.
    decoder_forward = model.forward
    encoder_forward_after_prefill = model.model.encoder
    decoder_attention_mask = torch.nn.functional.pad(
        attention_mask, (0, canvas_length), value=True
    )

    sampler = model._prepare_sampler(gen_cfg)
    logits_processor = model._prepare_logits_processor(gen_cfg, None)
    ar_stopping = model._prepare_ar_stopping_criteria(gen_cfg, None)
    diffusion_stopping = model._prepare_diffusion_stopping_criteria(gen_cfg)

    # 1. Autoregressive canvas (block) loop.
    is_prefill = True
    for block in range(max_new_canvases):
        # 1.a. Encode all previous tokens -> KV cache.
        unprocessed_input_ids, encoder_mask_mapping = model._prepare_encoder_inputs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_position_ids=encoder_position_ids,
            past_key_values=past_key_values,
            is_prefill=is_prefill,
            canvas_length=canvas_length,
            batch_size=batch_size,
            **model_kwargs,
        )
        encoder_forward = model.model.encoder if is_prefill else encoder_forward_after_prefill
        encoder_outputs = encoder_forward(
            input_ids=unprocessed_input_ids,
            attention_mask=encoder_mask_mapping,
            past_key_values=past_key_values,
            position_ids=encoder_position_ids,
            **model_kwargs,
        )
        past_key_values = encoder_outputs.past_key_values
        is_prefill = False

        # 1.b. Prepare denoiser inputs (initializes the canvas, builds mask_mapping).
        current_canvas, self_conditioning_logits, mask_mapping, finished_denoising = (
            model._prepare_denoiser_inputs(
                decoder_attention_mask=decoder_attention_mask,
                past_key_values=past_key_values,
                sampler=sampler,
                diffusion_stopping_criteria=diffusion_stopping,
                batch_size=batch_size,
                device=device,
                model_kwargs=model_kwargs,
            )
        )
        argmax_canvas = current_canvas

        # 1.c. Denoising loop (reverse diffusion: N..1).
        for cur_step in reversed(range(1, gen_cfg.max_denoising_steps + 1)):
            current_canvas, argmax_canvas, self_conditioning_logits, finished_denoising = (
                model._denoising_step(
                    decoder_forward=decoder_forward,
                    current_canvas=current_canvas,
                    argmax_canvas=argmax_canvas,
                    input_ids=input_ids,
                    decoder_position_ids=decoder_position_ids,
                    self_conditioning_logits=self_conditioning_logits,
                    mask_mapping=mask_mapping,
                    past_key_values=past_key_values,
                    finished_denoising=finished_denoising,
                    cur_step=cur_step,
                    sampler=sampler,
                    logits_processor=logits_processor,
                    diffusion_stopping_criteria=diffusion_stopping,
                    **model_kwargs,
                )
            )
            if torch.all(finished_denoising):
                break

        logger.info("block {}/{} done", block + 1, max_new_canvases)
        # 1.d. Append the denoised canvas.
        input_ids = torch.cat([input_ids, argmax_canvas], dim=-1)

        # 1.e. Autoregressive stopping; pad finished sequences.
        input_ids, finished_sequences = model._finalize_canvas(
            input_ids=input_ids,
            finished_sequences=finished_sequences,
            generation_config=gen_cfg,
            stopping_criteria=ar_stopping,
            canvas_length=canvas_length,
            eos_tensor=eos_tensor,
        )
        if torch.all(finished_sequences):
            break

        # 1.f. Prepare tensors for the next block.
        (
            cur_len,
            decoder_attention_mask,
            attention_mask,
            encoder_position_ids,
            decoder_position_ids,
        ) = model._prepare_kwargs_for_next_canvas(
            attention_mask=attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            decoder_position_ids=decoder_position_ids,
            past_key_values=past_key_values,
            canvas_length=canvas_length,
            cur_len=cur_len,
            is_compiling=False,
        )

    return input_ids


def test_gemma4():
    # ---- Reference path: model.generate() ----
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = DiffusionGemmaForBlockDiffusion.from_pretrained(MODEL_ID, dtype=torch.bfloat16)
    model.eval()

    # Architecture and the unique dtypes across all model parameters.
    logger.info("model={}", model)
    logger.info("model dtypes={}", {p.dtype for p in model.parameters()})

    message = [{"role": "user", "content": "Why is the sky blue?"}]
    enc = processor.apply_chat_template(
        message,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    # Convert floating-point inputs to bfloat16 (leave integer token ids/masks as-is).
    for key, value in enc.items():
        if torch.is_floating_point(value):
            enc[key] = value.to(torch.bfloat16)
            
    

    # dtype, shape and contents of each input.
    for key, value in enc.items():
        logger.info(
            "input key={}, dtype={}, shape={}, value={}",
            key, value.dtype, value.shape, value,
        )

    # Record every RNG draw (multinomial samples + randint canvases) generate()
    # makes, so the manual loop can replay the *same* draws. This removes RNG as a
    # variable: matching outputs on identical draws prove the loop mimics generate().
    rng_draws = []
    _orig_multinomial, _orig_randint = torch.multinomial, torch.randint

    def _record_multinomial(*args, **kwargs):
        out = _orig_multinomial(*args, **kwargs)
        rng_draws.append(("m", out.clone()))
        return out

    def _record_randint(*args, **kwargs):
        out = _orig_randint(*args, **kwargs)
        rng_draws.append(("r", out.clone()))
        return out

    torch.manual_seed(0)
    with patch("torch.multinomial", _record_multinomial), patch(
        "torch.randint", _record_randint
    ):
        ref_output = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS)
    logger.info("recorded {} rng draws during generate()", len(rng_draws))
    logger.info("reference raw output (token ids)={}", ref_output)
    ref_seq = getattr(ref_output, "sequences", ref_output)
    ref_text = processor.decode(ref_seq[0], skip_special_tokens=True)
    logger.info("reference generate() output={}", ref_text)

    # Free the reference model before loading the loader's copy (avoid 2x ~52GB).
    del model
    gc.collect()

    # ---- Loader path: manual no-cache denoising loop ----
    loader = ModelLoader()
    loader_model = loader.load_model(dtype_override=torch.bfloat16)
    loader_model.eval()

    # Architecture and param dtypes for the loader's model copy.
    # logger.info("loader model={}", loader_model)
    logger.info("loader model dtypes={}", {p.dtype for p in loader_model.parameters()})

    loader_inputs = loader.load_inputs(dtype_override=torch.bfloat16)

    for key, value in loader_inputs.items():
        logger.info(
            "loader input key={}, dtype={}, shape={}, value={}",
            key, value.dtype, value.shape, value,
        )

    # Replay generate()'s recorded draws into the manual loop, same order, so both
    # walk the identical trajectory. The guards separate a real wiring bug from a
    # benign RNG call-order/count skew (the paths may draw RNG at different points).
    _replay_iter = iter(rng_draws)
    _replayed = [0]

    def _replay(tag):
        try:
            drawn_tag, drawn = next(_replay_iter)
        except StopIteration:
            raise AssertionError(
                f"manual made more '{tag}' RNG calls than generate() recorded "
                f"({_replayed[0]}/{len(rng_draws)}) -> call-count skew, not a wiring bug"
            )
        assert drawn_tag == tag, (
            f"rng call-order skew at draw #{_replayed[0]}: manual asked '{tag}', "
            f"recorded '{drawn_tag}' -> paths draw RNG differently, not a wiring bug"
        )
        _replayed[0] += 1
        return drawn

    # Same extra inputs generate() received (e.g. mm_token_type_ids), but NOT the
    # loader's decoder_input_ids -> let the loop initialize its own canvas like generate().
    manual_kwargs = {
        k: v
        for k, v in loader_inputs.items()
        if k not in ("input_ids", "attention_mask", "decoder_input_ids")
    }
    with patch("torch.multinomial", lambda *a, **k: _replay("m")), patch(
        "torch.randint", lambda *a, **k: _replay("r")
    ):
        manual_output = manual_generate(
            loader_model,
            input_ids=loader_inputs["input_ids"],
            attention_mask=loader_inputs["attention_mask"],
            max_new_tokens=MAX_NEW_TOKENS,
            **manual_kwargs,
        )
    leftover = len(list(_replay_iter))
    assert leftover == 0, (
        f"{leftover} recorded draws unused ({_replayed[0]}/{len(rng_draws)} replayed) "
        f"-> call-count skew between generate() and manual loop, not a wiring bug"
    )
    logger.info("replayed all {} rng draws into manual loop", _replayed[0])
    logger.info("manual raw output (token ids)={}", manual_output)

    # Outputs may be a structured GenerationOutput (dict of tensors); compare the
    # `sequences` token ids from each path, not just the decoded text.
    manual_seq = getattr(manual_output, "sequences", manual_output)
    logger.info("ref sequences shape={}, value={}", tuple(ref_seq.shape), ref_seq)
    logger.info("manual sequences shape={}, value={}", tuple(manual_seq.shape), manual_seq)
    sequences_equal = (
        ref_seq.shape == manual_seq.shape and torch.equal(ref_seq, manual_seq)
    )
    logger.info("sequences equal (torch.equal) = {}", sequences_equal)
    if ref_seq.shape == manual_seq.shape and not sequences_equal:
        diff = torch.nonzero(ref_seq[0] != manual_seq[0], as_tuple=False)
        first = int(diff[0]) if diff.numel() else -1
        logger.info(
            "token mismatches = {}/{}, first differing index = {}",
            int((ref_seq != manual_seq).sum()),
            ref_seq.numel(),
            first,
        )

    manual_text = loader.processor.decode(manual_seq[0], skip_special_tokens=True)
    logger.info("manual loop output={}", manual_text)

    # With identical RNG draws replayed, a bit-exact match proves the manual loop
    # mimics generate() end-to-end (steps + loop wiring + block boundaries).
    assert sequences_equal, (
        "manual loop diverged from generate() on identical RNG draws -> loop-wiring "
        "bug (see 'first differing index' above to localize the step/block)"
    )
