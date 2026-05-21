# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import gc
import math
from functools import lru_cache
from typing import Dict, Tuple

import numpy as np
import pytest
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from infra.utilities.torch_multichip_utils import enable_spmd
from torch_xla.distributed.spmd import Mesh
from transformers import AutoTokenizer
from tt_torch.sharding import sharding_constraint_hook
from tt_torch.sparse_mlp import enable_sparse_mlp
from tt_torch.weight_dtype import apply_weight_dtype_overrides

from tests.benchmark.utils import compute_pcc
from third_party.tt_forge_models.deepseek_v4.modified_model import (
    model_decode_opt as mdo,
)

from . import utils, weight_loader

# All the prompts below are constructed to be exactly 128 tokens long
PROMPT_LEN = 128
PROMPTS = [
    "Yesterday I was hiking on a forested trail in the Cascade Mountains of Washington State at around fifteen hundred meters elevation when I spotted a small mammal with reddish brown fur, a bushy tail nearly as long as its body, large dark eyes, and prominent tufted ears, scampering up a Douglas fir tree only ten meters from me. It paused on a branch, chittered loudly for almost a minute, and then vanished into the upper canopy. Could you help me figure out what species it most likely was, what it eats during the colder months, whether it stays active through winter or hibernates underground, and whether it c",
    "I have been maintaining a sourdough starter for about three weeks using a mix of whole wheat and rye flour at fifty percent hydration kept on the kitchen counter near twenty two degrees Celsius. For the first ten days it bubbled vigorously and doubled within four hours of feeding, but over the last few days it has slowed dramatically and developed a sharp acetone smell along with a darker liquid layer pooling on top. What is going on biochemically with the yeast and bacterial populations, how should I rescue it without throwing everything out and starting again from scratch, and would shifting to a stiffer dough or a colder fermentation temperature help reset the microbial balance back",
    "While debugging a long running Python service in production I noticed that resident memory climbs steadily over many hours even though my heap profiler shows the live Python object set stays flat near four hundred megabytes. The process eventually grows past sixteen gigabytes and gets killed by the kernel oom reaper. I am calling into a third party C library through ctypes that returns large numpy arrays, and I suspect a leak somewhere along that boundary. What strategies do you recommend for narrowing down whether the leak lives in pure Python, the C extension code itself, or in glibc allocator fragmentation, and would tools like valgrind, jemalloc heap profiles",
    "I am forty years old, have never played piano before, and want to learn well enough within two years to perform a Chopin nocturne for friends at a small living room gathering. I can practice forty five minutes on weekdays and an hour each weekend day. I do not currently read sheet music at all. Should I take weekly lessons with a teacher from the very start, work through method books like Alfred or Faber by myself first, or use an app like Simply Piano. What pacing of theory drills, technique exercises, and actual repertoire would you suggest for me, and which specific nocturne in the canon would be a realistic",
    "The Perseid meteor shower peaks next week and I am driving out to Joshua Tree National Park to view it on Tuesday night around two in the morning when the radiant is high overhead and the moon has already set below the horizon. I will be bringing a full frame camera with a fast wide angle lens. What camera settings, focusing technique, exposure length, and shooting cadence would maximize my chances of capturing several bright meteor trails against the Milky Way without trailing the stars too obviously, should I shoot raw stills or use a video mode, and how should I plan a stacking workflow afterward to combine multiple frames into a single final image. I",
    "I want to paint a watercolor of a misty pine forest at dawn with warm shafts of light filtering through the trunks and a small deer standing in the middle distance. I have decent technique with wet on wet washes for skies, but my forests always end up looking flat and muddy because my greens turn dull whenever I layer them on top of each other. What pigment combinations would you recommend for a luminous range of greens, what order should I lay down the washes, and how can I preserve the glowing atmosphere through the trunks rather than killing every highlight by overworking the paper, and would lifting with a damp brush help once the underwash has",
    "I have been reading about ancient Roman aqueducts and I find it astonishing they used purely gravity fed channels across distances of more than a hundred kilometers with gradients as gentle as one part in a thousand. How did Roman engineers actually survey such precise gradients across hilly terrain without any modern optical instruments, what tools like the chorobates or the groma did they use to keep alignment correct over multiple generations of construction, and how did they handle the inevitable settling, leaks, and biological fouling once the lines were operating, so that water kept reaching cities reliably across many centuries of continuous service. I am most curious about the urban distribution at the",
    "I am trying to teach my twelve year old daughter why prime numbers matter beyond just being a definition we memorize in school. She finds them arbitrary and keeps asking why mathematicians care about them so much. What concrete examples would you suggest for showing her that primes are the fundamental building blocks of all integers, how they appear in surprising places like cicada life cycles or the distribution of energy levels in chaotic quantum systems, and how internet encryption depends on the fact that factoring a product of two large primes is computationally hard for any classical computer running today, even given many years of patient effort. She just learned modular arithmetic in school last month, so I can",
    "I am an English speaker who has been studying Mandarin for about six months using flashcards and a textbook, and I can recognize maybe four hundred characters and hold a basic conversation about food or weather. I am traveling to Chengdu for two months this fall and want to make rapid progress in spoken comprehension before I go. What daily routine combining listening practice, tutoring sessions, shadowing native speakers, and writing in a journal would you recommend so I can hold real conversations with locals in Sichuanese accented Mandarin by the end of my visit, and which specific apps, podcasts, or YouTube channels are worth a serious time commitment. I will be staying with",
    "Tardigrades are tiny microscopic animals that can survive temperatures near absolute zero, the vacuum of outer space, doses of ionizing radiation that would instantly kill any human, and dehydration for many years on end. What molecular and cellular mechanisms allow them to enter their cryptobiotic state and then revive afterward without losing their internal cellular structure, how is research into these mechanisms being applied to practical problems like preserving organs for transplant or shipping live vaccines without refrigeration into remote regions where reliable cold chains do not yet exist, and what is the ongoing scientific debate about how they evolved. I read that some lab strains have even been revived from museum samples decades old",
    "I want to make handmade soap from scratch in my own kitchen using olive oil and coconut oil with sodium hydroxide as the saponifying agent. I am nervous about handling lye safely and about getting the chemistry exactly right so the final bar ends up mild rather than still caustic after curing. Could you walk me through a beginner appropriate cold process recipe with safe oil to lye ratios, the order in which to combine ingredients, what the trace stage actually looks like in the bowl, how many weeks the bars need to cure before they are gentle on skin, and what protective equipment I should put on before opening the lye container. I",
    "Philosophers have argued about free will for centuries and I find the debate genuinely confusing because compatibilists, libertarians, and hard determinists all seem to be using the words free and choice in subtly different ways. Could you map out the main positions clearly, explain what each side actually claims about whether we could have done otherwise in a strict physical sense, describe how recent neuroscience experiments like the Libet readiness potential studies are interpreted very differently by each camp without ever quite resolving the underlying philosophical question one way or the other, and which thinkers would you recommend reading first. I am especially interested in whether the recent neuroscience evidence really should move me",
    "My partner and I are planning a three week trip to Japan next April for our very first visit and we want a balance of major cultural sites, deep food experiences, and quieter rural areas to escape the cherry blossom crowds. We will arrive in Tokyo and depart from Osaka. We love hiking, traditional crafts, hot springs, and small family run inns. Could you sketch a possible itinerary that hits the headline Tokyo, Kyoto, and Osaka highlights but also spends meaningful time in less touristed regions like Kyushu, Shikoku, or the Japan Sea coast, how should we move between them by rail or rental car, and what should",
    "I have been playing tennis recreationally for about ten years and my forehand is reliable, but my one handed backhand collapses under any real pressure, especially on high balls above shoulder height. I keep dumping the ball into the net or popping it up short for an easy putaway. Could you diagnose what the most common technical breakdowns tend to be at that contact point, what footwork or preparation changes would give me more time and stability, what drills could I practice alone against a wall to build confidence before applying it in a match situation, and would switching to a slightly heavier racket or different string tension help. My current racket",
    "My tomato plants in the backyard greenhouse are showing a strange wilting pattern where the lower leaves yellow and curl while the upper canopy still looks green and healthy, and there are dark concentric rings on some of the older leaves that look almost like archery target patterns. The fruits themselves seem fine so far. What disease are these symptoms most consistent with, how do I confirm the diagnosis without sending samples to a lab, what organic options do I have for treatment and for preventing a recurrence next season without resorting to synthetic fungicides, and should I rotate to a different bed. I have been watering by drip line at the soil level only, never",
    "Modern earthquake resistant skyscrapers in Tokyo and San Francisco use a combination of base isolation systems, tuned mass dampers, and specially designed structural frames to ride out strong ground shaking instead of resisting it rigidly with brute mass. Could you explain at an intuitive physics level what each of those three systems actually does to absorb or redirect seismic energy, why a counter intuitive flexible building outperforms a much stiffer one in a strong earthquake, how engineers tune the parameters of a damper to match the resonant frequencies of a specific structure, and what tradeoffs they accept in everyday wind loading. I would also love a sense of how this compares",
    "I love magical realism in Latin American literature, especially Garcia Marquez and Borges, where the fantastical is woven into the everyday without any narrative explanation or apology. Could you recommend half a dozen lesser known novels in that tradition from authors outside the very famous names, explain what specifically distinguishes magical realism from ordinary fantasy or surrealism in how it treats the impossible, suggest the best translations to read first if I do not yet read Spanish well enough to tackle the originals on my own, and identify any contemporary writers continuing the tradition today. I have already read One Hundred Years of Solitude, Pedro Paramo, and most of the major Borg",
    "My grandmother on a fixed income is worried that her savings are losing real purchasing power because of inflation, and she has asked me to explain what is actually causing prices to rise and whether she should buy gold or move into bonds. Could you give me a clear explanation that a non economist can follow about the difference between cost push and demand pull inflation, why central banks raise interest rates as a response to it, what investment strategies actually preserve real purchasing power for someone close to retirement age who cannot afford much risk, and whether treasury inflation protected securities are appropriate here. She has about four hundred thousand dollars in mostly cash and short term certificates of deposit at",
    "I have been getting only about five hours of sleep on average for the last six months because of work pressure and a young child at home, and I am noticing that my memory, mood, and motivation are all suffering even though I keep telling myself I am adapting fine. What is actually happening in my brain biochemically and structurally during chronic partial sleep deprivation, are these effects fully reversible if I get back to seven or eight hours nightly, what specific recovery strategies are most effective beyond just sleeping in a little more on weekends, and how long would the recovery realistically take to reach a reasonable baseline again. I tried melatonin earlier this year and it",
    "I have read that black holes have a finite entropy proportional to the area of their event horizon rather than their interior volume, and that this leads to the holographic principle suggesting our three dimensional universe could in principle be encoded on a two dimensional boundary surface. Could you walk me through why entropy scales with area instead of volume in this case, what that tells us about how information is fundamentally stored in a gravitational system, whether the holographic principle has any concrete experimental consequences beyond pure theoretical speculation, and how Hawking radiation interacts with the famous information paradox in this picture. I have a physics undergrad background, so you can use moderate equations rather than",
    "My five year old daughter has had a runny nose, a mild fever around thirty eight degrees Celsius, and a barking cough for two days now, and she is sleeping poorly because of the cough waking her at night. I want to avoid antibiotics unless they are truly needed because I know most childhood respiratory illnesses are viral. What specific symptoms or signs should make me bring her in to a pediatrician immediately rather than treating supportively at home with fluids and rest, which over the counter remedies are actually safe and effective for a child of her age, and is a humidifier or steam shower worth trying. She is fully vaccinated for her age and",
    "I read recently that a new pedestrian bridge in Norway is a cantilever extending almost a hundred meters out from a cliffside with no support underneath at all. Mechanically, what makes a long cantilever such a demanding structure to design compared to a more conventional beam supported at both ends, how do engineers actually handle the bending moments and tip deflections that grow with the cube of the cantilever length, what materials and cross sectional geometries make modern cantilevers possible at lengths that would have been completely unthinkable a century ago, and how do they manage wind induced oscillation. The bridge in question reportedly uses high strength weathering steel and a tapered",
    "I have always been fascinated by the relationships among the Indo European languages and how scholars reconstructed Proto Indo European purely from comparative evidence in dozens of daughter languages spoken thousands of years later. Could you walk me through the basic methods of historical linguistics that allow us to work backwards from modern Hindi, Greek, Latin, and English to a single common ancestor language, what kinds of evidence are most reliable for that reconstruction, where the limits of the comparative method break down so that we cannot reach back further than a few thousand years, and which proposed deeper macrofamilies remain controversial among specialists. I am especially curious about whether Anatolian languages like Hittite genuinely",
    "I am traveling through Iceland next month and I want to actually understand the volcanic geology I will be looking at on my drives around the ring road. The island sits on the slow spreading mid Atlantic ridge and is also over an active mantle plume, which together create an unusual variety of volcanic rock types in close proximity. Could you help me distinguish basalt, rhyolite, hyaloclastite, and palagonite tuff in the field, what each tells me about the eruption history of a given site, which iconic Icelandic landscapes correspond to particularly clear examples of each rock type, and what I should look for at lava tube sites. I",
    "I am directing my first short film and I want to use lighting to shape mood without relying heavily on overt color grading in post production. The key scene is a quiet conversation between two characters in a small kitchen at dusk that should feel intimate and warm, but with a thread of unease running underneath the surface. How would you approach the lighting setup with practical lamps versus dedicated film lights, what color temperatures and contrast ratios would you target, how do small adjustments in the height and softness of the key light register on the actors emotionally, and where should I place a subtle backlight to suggest tension. The kitchen has one practical pendant lamp above",
    "I want to build a wardrobe that is durable, ethically produced, and largely free of fast fashion. I have a moderate but not unlimited budget. Could you help me think through how to evaluate brands for actual sustainability claims rather than greenwashing marketing copy, which natural fibers like wool, hemp, and linen are most resilient and biodegradable in practice, how to build a small versatile capsule of around thirty pieces that can carry me through several years of weekday and weekend wear, where to find good secondhand options for higher end items, and how to care for these pieces so they truly last. I work in an office four days a week with a smart",
    "The thermohaline circulation in the North Atlantic Ocean has been weakening over recent decades according to several independent published measurements, and there are growing concerns this could shift European climate patterns or disrupt fisheries. Could you explain what physically drives the circulation, what role salinity and temperature gradients each play in setting it up, what feedback loops involving Greenland meltwater and Arctic sea ice could push the system toward an abrupt rather than a gradual change, and how confident climate scientists currently are about whether a complete shutdown is plausible within this century or remains a remote tail risk at this time. I would also love your view on whether recent paleoclimate proxy reconstructions back to",
    "I keep hearing that quantum computers will eventually break public key cryptography using Shor's algorithm and that we need post quantum schemes ready and deployed before that happens. Could you explain at a conceptual level how a quantum computer can factor large composite numbers exponentially faster than a classical machine, what physical platforms like superconducting qubits or trapped ions are most likely to scale up first to relevant problem sizes, which of the post quantum cryptographic candidates such as lattice based or hash based schemes are currently considered most promising for widespread deployment, and what realistic timelines we should plan around for migration. My organization runs a lot of long lived TLS certificates and signed firmware, so we are",
    "Octopuses have nine brains, copper based blue blood, and the ability to taste with their suckers, and there is growing experimental evidence that they can solve complex puzzles, recognize individual humans by face, and use tools in the wild. Could you describe what is unusual about their distributed nervous system compared to vertebrate brains, how they likely experience the world through their unique sensory apparatus spread across their arms, what ethical implications biologists and philosophers are drawing about how we should treat them in research labs and in the food industry given this evidence, and how solitary their daily lives actually are. I just watched the documentary My Octopus Teacher and have been thinking",
    "I have been told there are two distinct photosynthetic pathways called C three and C four that allow plants to fix carbon dioxide rather differently, and that crops like sugarcane and corn use the C four pathway while wheat and rice use C three. Could you explain what physical and biochemical differences make C four plants so much more efficient in hot dry conditions with limited water available, why most plants did not happen to evolve C four metabolism over geological time, what serious efforts are underway to engineer the more efficient pathway directly into rice plants, and how the CAM pathway used by succulents differs from both. I am especially interested in the C four rice project at the",
    "I am shopping for a new car and torn among a regular hybrid like a Toyota RAV4 Prime, a fully electric vehicle like a Hyundai Ioniq, and a plug in hybrid. My commute is about fifty kilometers each day and I take road trips of around eight hundred kilometers maybe four or five times a year. Charging at home overnight would be straightforward but rural fast charging on my road trip routes is still patchy. Given my use case what total cost of ownership, environmental, and practical considerations should drive my decision over a five to seven year horizon, and how should I think about long term battery degradation. I would also be",
    "Norse mythology features a complex pantheon with two warring families of gods called the Aesir and Vanir who eventually unite in peace, a cosmic ash tree called Yggdrasil holding nine separate worlds in its branches and roots, and a prophesied apocalyptic battle called Ragnarok in which most of the gods are slain. Could you walk through the main characters of the pantheon, the major myths that explain how the universe was created and how it is ultimately destined to end, how these stories were preserved through the Eddas after Christianization disrupted the oral tradition, and which retellings would you recommend for a modern",
]


def _tokenize_prompts(tokenizer, prompts: list) -> torch.Tensor:
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    prompt_rows = []
    for prompt in prompts:
        ids = tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        ).input_ids[0]
        if ids.shape[0] >= PROMPT_LEN:
            ids = ids[-PROMPT_LEN:]
        else:
            pad = torch.full((PROMPT_LEN - ids.shape[0],), pad_id, dtype=torch.long)
            ids = torch.cat([pad, ids], dim=0)
        prompt_rows.append(ids)
    return torch.stack(prompt_rows, dim=0).contiguous()


def _load_full_model(model_name: str, args: mdo.ModelArgs, mesh_shape: Tuple[int, ...]):
    print(
        f"[build] constructing Transformer skeleton (n_layers={args.n_layers}, "
        f"n_routed_experts={args.n_routed_experts}) ...",
        flush=True,
    )
    model = utils.make_transformer(args, True)
    print(f"[load] full state_dict for all {args.n_layers} layers ...", flush=True)
    sd = weight_loader.load_transformer_state_dict(
        model_name, range(args.n_layers), include_mtp=False
    )
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    print(f"[swap] enable_sparse_mlp on all {args.n_layers} blocks ...", flush=True)
    enable_sparse_mlp(
        model,
        mesh=mesh_shape,
        cluster_axis=0,
        config=args,
        verbose=False,
    )

    print("[build] done. CPU model assembled.", flush=True)
    return model


def _reset_attn_caches(model: mdo.Transformer) -> None:
    """Zero in-place attention/compressor cache buffers populated by a prior
    forward pass so the next prefill starts from the same blank state as a
    fresh model."""
    for block in model.layers:
        attn = block.attn
        attn.kv_cache.zero_()
        if attn.compress_ratio:
            attn.compressor.kv_cache.zero_()
            attn.compressor.kv_state.zero_()
            attn.compressor.score_state.fill_(float("-inf"))
            if hasattr(attn, "indexer") and attn.indexer is not None:
                attn.indexer.compressor.kv_cache.zero_()
                attn.indexer.compressor.kv_state.zero_()
                attn.indexer.compressor.score_state.fill_(float("-inf"))


def transformer_shard_spec(model: mdo.Transformer):
    shard_specs = {}
    for layer in model.layers:
        attn: mdo.Attention = layer.attn
        shard_specs[attn.wq_b.weight] = ("model", None)
        shard_specs[attn.wo_a.weight] = ("model", None)
        shard_specs[attn.wo_b.weight] = (None, "model")
        shard_specs[attn.kv_cache] = ("batch", None, None)

        if attn.compress_ratio:
            shard_specs[attn.compressor.kv_cache] = ("batch", None, None)
            shard_specs[attn.compressor.kv_state] = ("batch", None, None)
            shard_specs[attn.compressor.score_state] = ("batch", None, None)
        if hasattr(attn, "indexer") and attn.indexer is not None:
            shard_specs[attn.indexer.wq_b.weight] = ("model", None)
            shard_specs[attn.indexer.weights_proj.weight] = ("model", None)
            shard_specs[attn.indexer.compressor.kv_cache] = ("batch", None, None)
            shard_specs[attn.indexer.compressor.kv_state] = ("batch", None, None)
            shard_specs[attn.indexer.compressor.score_state] = ("batch", None, None)

        ffn = layer.ffn
        if hasattr(ffn, "mlp") and hasattr(ffn.mlp, "experts"):
            experts = ffn.mlp.experts
            compound = ("batch", "model")
            shard_specs[experts.gate_proj] = (compound, None, None)
            shard_specs[experts.up_proj] = (compound, None, None)
            shard_specs[experts.down_proj] = (compound, None, None)
    return shard_specs


@pytest.mark.nightly
@pytest.mark.bh_galaxy
@pytest.mark.parametrize("model_name", ["deepseek-ai/DeepSeek-V4-Flash"])
@pytest.mark.parametrize("num_layers", [10, 15, 20, 30, 43])
@pytest.mark.parametrize("num_iterations", [1, 2, 5, 10])
@pytest.mark.parametrize("use_cpu_decode_inputs", [True, False])
@torch.inference_mode()
def test_prefill_and_decode_pcc_e2e(
    model_name, num_layers, num_iterations, use_cpu_decode_inputs
):
    enable_spmd()
    xr.set_device_type("TT")

    mesh = utils.make_2d_mesh()
    bsz = len(PROMPTS)

    assert bsz == 32, "Batch size must be 32 (restriction comes from enable_sparse_mlp)"

    args = weight_loader.load_config_args(model_name, False)

    args.n_mtp_layers = 0  # Disable multi-token prediction
    args.max_batch_size = bsz
    args.max_seq_len = 2 * PROMPT_LEN

    if num_layers < args.n_layers:
        args.n_layers = num_layers
        args.compress_ratios = args.compress_ratios[:num_layers]

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompt_ids = _tokenize_prompts(tokenizer, PROMPTS)
    assert prompt_ids.shape == (bsz, PROMPT_LEN)

    model = _load_full_model(model_name, args, mesh.mesh_shape)

    # ---- CPU prefill + decodes ------------------------------------------
    # decode_inputs[k] is the [bsz, 1] token tensor fed into CPU decode
    # step k (decode_inputs[0] is argmax of prefill logits;
    # decode_inputs[k>0] is argmax of decode step k-1's logits).
    # cpu_generated_tokens[k] is the token CPU produced *at* decode step k
    # (= argmax of step k's logits = decode_inputs[k+1] when k+1 exists).
    decode_inputs: list[torch.Tensor] = []
    cpu_generated_tokens: list[torch.Tensor] = []

    print("[cpu] prefill ...", flush=True)
    sp_prefill = torch.tensor(
        PROMPT_LEN, dtype=torch.long
    )  # start_pos tensor for prefill
    h_pf, logits_pf = model(prompt_ids, sp_prefill, return_hidden_states=True)
    cpu_prefill_h = h_pf.detach().to(torch.float32).cpu().clone()
    cpu_prefill_logits = logits_pf.detach().to(torch.float32).cpu().clone()
    next_token = logits_pf.detach().cpu().argmax(dim=-1, keepdim=True).to(torch.long)

    print(
        f"[cpu] prefill hidden_states shape={tuple(cpu_prefill_h.shape)} "
        f"first_ids[:8]={next_token[:8, 0].tolist()}",
        flush=True,
    )

    # List to store the hidden state tensors after each decode step
    cpu_decode_hs: list[torch.Tensor] = []
    cpu_decode_logits: list[torch.Tensor] = []
    for step in range(num_iterations):
        decode_inputs.append(next_token)
        sp_step = torch.tensor(PROMPT_LEN + step, dtype=torch.long)
        h_d, logits_d = model(next_token, sp_step, return_hidden_states=True)
        h = h_d.detach().to(torch.float32).cpu().clone()
        logits = logits_d.detach().to(torch.float32).cpu().clone()
        cpu_decode_hs.append(h)
        cpu_decode_logits.append(logits)
        next_token = logits_d.detach().cpu().argmax(dim=-1, keepdim=True).to(torch.long)
        cpu_generated_tokens.append(next_token)
        print(
            f"[cpu] decode step {step} sp={PROMPT_LEN + step} "
            f"shape={tuple(h.shape)} next_ids[:8]={next_token[:8, 0].tolist()}",
            flush=True,
        )

    assert len(decode_inputs) == num_iterations
    assert len(cpu_generated_tokens) == num_iterations
    # CPU prefill+decodes mutated kv_cache + compressor / indexer state in
    # place. Reset so the device run starts from the same fresh state.
    _reset_attn_caches(model)

    # ---- Move to device + shard ----------------------------------------
    print("[device] moving model to TT (this releases CPU storage) ...", flush=True)
    device = torch_xla.device()
    model = model.to(device)
    # gc.collect()
    for tensor, spec in transformer_shard_spec(model).items():
        xs.mark_sharding(tensor, mesh, spec)

    hook = sharding_constraint_hook(model.head, mesh, (None, None))
    model.head.register_forward_hook(hook)

    compiled = torch.compile(model, backend="tt")

    # ---- Device prefill + decodes --------------------------------------
    prompt_ids_tt = prompt_ids.to(device)
    xs.mark_sharding(prompt_ids_tt, mesh, ("batch", None))
    sp_prefill_tt = torch.tensor(PROMPT_LEN, dtype=torch.long).to(device)
    print("[device] compiling + running prefill ...", flush=True)
    h_pf_dev, logits_pf_dev = compiled(
        prompt_ids_tt, sp_prefill_tt, return_hidden_states=True
    )
    device_prefill_h = h_pf_dev.detach().to("cpu").to(torch.float32).clone()
    device_prefill_logits = logits_pf_dev.detach().to("cpu").to(torch.float32).clone()

    next_token_dev = (
        logits_pf_dev.detach().to("cpu").argmax(dim=-1, keepdim=True).to(torch.long)
    )

    print(
        f"[device] prefill hidden_states shape={tuple(device_prefill_h.shape)} "
        f"first_ids[:8]={next_token_dev[:8, 0].tolist()}",
        flush=True,
    )

    device_decode_hs: list[torch.Tensor] = []
    device_decode_logits: list[torch.Tensor] = []

    for step in range(num_iterations):
        if use_cpu_decode_inputs:
            # Replay the CPU-derived token sequence so CPU and device decode steps see
            # identical inputs.
            token_in = decode_inputs[step]
        else:
            token_in = next_token_dev

        decode_token_tt = token_in.to(device)  # [bsz, 1]
        xs.mark_sharding(decode_token_tt, mesh, ("batch", None))
        sp_step_tt = torch.tensor(PROMPT_LEN + step, dtype=torch.long).to(device)
        h_d_dev, logits_d_dev = compiled(
            decode_token_tt, sp_step_tt, return_hidden_states=True
        )
        h = h_d_dev.detach().to("cpu").to(torch.float32).clone()
        logits = logits_d_dev.detach().to("cpu").to(torch.float32).clone()
        device_decode_hs.append(h)
        device_decode_logits.append(logits)
        dev_argmax = (
            logits_d_dev.detach().to("cpu").argmax(dim=-1, keepdim=True).to(torch.long)
        )
        next_token_dev = dev_argmax

        print(
            f"[device] decode step {step} sp={PROMPT_LEN + step} "
            f"shape={tuple(h.shape)} "
            f"next_ids[:8]={dev_argmax[:8, 0].tolist()}",
            flush=True,
        )

    # ---- PCC comparisons ------------------------------------------------
    pccs: list[tuple[str, float]] = []
    assert cpu_prefill_h.shape == device_prefill_h.shape, (
        f"prefill hidden states shape mismatch: cpu={tuple(cpu_prefill_h.shape)} "
        f"device={tuple(device_prefill_h.shape)}"
    )
    assert cpu_prefill_logits.shape == device_prefill_logits.shape, (
        f"prefill logits shape mismatch: cpu={tuple(cpu_prefill_logits.shape)} "
        f"device={tuple(device_prefill_logits.shape)}"
    )

    prefill_h_pcc = compute_pcc(cpu_prefill_h, device_prefill_h)
    prefill_logits_pcc = compute_pcc(cpu_prefill_logits, device_prefill_logits)

    pccs.append(("prefill hidden states pcc", prefill_h_pcc))
    pccs.append(("prefill logits pcc", prefill_logits_pcc))

    print(f"[pcc] prefill hidden states pcc={prefill_h_pcc:.6f}", flush=True)
    print(f"[pcc] prefill logits pcc={prefill_logits_pcc:.6f}", flush=True)

    assert len(cpu_decode_hs) == len(device_decode_hs)
    assert len(cpu_decode_logits) == len(device_decode_logits)
    assert len(cpu_decode_logits) == len(cpu_decode_hs)
    for i in range(len(cpu_decode_hs)):
        cpu_h = cpu_decode_hs[i]
        dev_h = device_decode_hs[i]
        cpu_logits = cpu_decode_logits[i]
        dev_logits = device_decode_logits[i]
        assert cpu_h.shape == dev_h.shape, (
            f"decode[{i}] hidden states shape mismatch: cpu={tuple(cpu_h.shape)} "
            f"device={tuple(dev_h.shape)}"
        )
        assert cpu_logits.shape == dev_logits.shape, (
            f"decode[{i}] logits shape mismatch: cpu={tuple(cpu_logits.shape)} "
            f"device={tuple(dev_logits.shape)}"
        )
        pcc_h = compute_pcc(cpu_h, dev_h)
        pcc_logits = compute_pcc(cpu_logits, dev_logits)
        pccs.append((f"decode[{i}] hidden states pcc", pcc_h))
        pccs.append((f"decode[{i}] logits pcc", pcc_logits))
        print(f"[pcc] decode step {i} hidden states pcc={pcc_h:.6f}", flush=True)
        print(f"[pcc] decode step {i} logits pcc={pcc_logits:.6f}", flush=True)

    all_pass = True
    for desc, pcc in pccs:
        if pcc < 0.95:
            all_pass = False
            print(
                f"[pcc validation] {desc} failed validation with pcc={pcc}", flush=True
            )
    assert all_pass, "Expected all PCCs to pass"
