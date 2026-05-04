# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""End-to-end prefill+decode generation for DeepSeek-V4-Flash on TT.

Loads the full real published weights (43 layers, 256 routed experts,
n_mtp_layers=0) into `model_decode_opt.Transformer`, swaps each block's
MoE for an A2aSparseMLP via `tt_torch.sparse_mlp.enable_sparse_mlp`,
runs one prefill on the tokenized prompt and a fixed number of decode
steps, and prints the detokenized continuation.

No PCC; no CPU oracle. The only contract is that the loop produces tokens.

Compile budget: 4 tt-backend compiles total (2 const-eval + 2 main forward,
one prefill + one decode), via the same sp_buffer / one_buffer trick as
test_deepseek_v4_prefill_decode_loop_no_int.py:151-158.

Memory: assumes the host has enough RAM to hold the full dequantized
checkpoint (~600 GB BF16) plus per-block stacked expert copies during
sparse-MLP swap, before `.to(torch_xla.device())` ships shards out.
"""

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
from tt_torch.sparse_mlp import enable_sparse_mlp
from tt_torch.weight_dtype import apply_weight_dtype_overrides

from tt_torch.sharding import sharding_constraint_hook

from third_party.tt_forge_models.deepseek_v4.modified_model import (
    model_decode_opt as mdo,
)

from . import weight_loader

# Prefill length must be a multiple of args.window_size (=128) so the
# attention's circular kv_cache update lands on a window boundary; otherwise
# the cache contents become inconsistent for the first decode step. It also
# must be >= max(args.compress_ratios) so the Compressor's
# `rope_idx = start_pos + 1 - ratio` stays non-negative
# (model_decode_opt.py:414). 128 satisfies both.
PROMPT_LEN = 128
MAX_NEW_TOKENS = 64

# Batch size is a hard requirement of the A2aSparseMLP op; on the (4, 8)
# mesh this shards to 8 rows per _axis_0 device.
BATCH_SIZE = 32

# Number of transformer blocks to load + run. Real config has 43; truncating
# to a prefix is a memory dial. Layer subset is sliced from
# real_args.compress_ratios so the included layers stay consistent with the
# real-checkpoint layout (mirrors transformer_args in
# test_deepseek_v4_tp_no_int.py:163).
NUM_LAYERS = 43

# see issue https://github.com/tenstorrent/tt-xla/issues/4444
torch._dynamo.config.cache_size_limit = 100

# One distinct prompt per batch row; len(PROMPTS) must equal BATCH_SIZE.
# Each prompt is a unique, content-rich question on its own topic, sliced in
# token space so it tokenizes to exactly 128 DeepSeek-V4 tokens (verified by
# test_prompts via get_num_tokens). This matches PROMPT_LEN, so the left-pad
# in _tokenize_prompts is a no-op for these rows. Some prompts end mid-word
# because the slice lands inside a multi-token unit; that is intentional and
# avoids the boilerplate-loop behavior we saw with repeated padding text.
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
assert len(PROMPTS) == BATCH_SIZE, (
    f"PROMPTS has {len(PROMPTS)} entries, must equal BATCH_SIZE={BATCH_SIZE}"
)


# Longer-content variant of PROMPTS used by the OOD-margin diagnostic. The
# fixed prefix bulks each prompt up to ~55-65 content tokens after
# tokenization, so that under the standard PROMPT_LEN=128 left-pad each row
# ends up with ~60 content tokens and ~68 pad tokens — the regime in which
# the user empirically observed the logits PCC improving relative to the
# default short-prompt 10/118 split.
LONG_CONTENT_PREFIX = (
    "In a recent thoughtful conversation held at an academic gathering "
    "about science, technology, history, philosophy, and a wide variety "
    "of other interesting modern topics that often come up in everyday "
    "discussions among curious people, one particularly enthusiastic "
    "participant raised the following important question: "
)
LONGER_PROMPTS = [LONG_CONTENT_PREFIX + p for p in PROMPTS]
assert len(LONGER_PROMPTS) == BATCH_SIZE


# ----------------------------------------------------------------------------
# All-at-once weight loader. Builds the Transformer skeleton, loads every
# real-checkpoint param in a single pass, then swaps every block's MoE for
# A2aSparseMLP.
#
# No gate / ffn patches needed: model_decode_opt.Gate.forward already has
# the _ambient_input_ids fallback and the SPMD-safe one_hot * scores form
# (model_decode_opt.py:678-714), and Block.forward already calls
# self.ffn(x, input_ids) (model_decode_opt.py:839), which the post-swap
# A2aSparseMLPWithSharedExperts accepts via its
# forward(hidden_states, *extra_args, **extra_kwargs) signature
# (sparse_mlp.py:993). RouterAdapter (sparse_mlp.py:753) flattens the 2D
# input_ids to 1D and threads them straight into Gate.forward, so the gate
# stash is also redundant.
# ----------------------------------------------------------------------------


def _build_and_load_full_model(args: mdo.ModelArgs, mesh_shape: Tuple[int, int]):
    """Construct Transformer, load all real weights in one call, then swap
    every block's MoE in one enable_sparse_mlp call. Returns the CPU-side
    model; caller is expected to .to(torch_xla.device()).
    """
    print(f"[build] constructing Transformer skeleton (n_layers={args.n_layers}, "
          f"n_routed_experts={args.n_routed_experts}) ...", flush=True)
    # Transformer relies on torch.get_default_dtype() for ParallelEmbedding
    # and the bf16 Linear weights; without this override they default to fp32
    # and mismatch the bf16 activations / checkpoint. Same construction
    # pattern as make_real_transformer in test_deepseek_v4_tp_no_int.py.
    prev_default = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = mdo.Transformer(args).eval()
    finally:
        torch.set_default_dtype(prev_default)

    print(f"[load] full state_dict for all {args.n_layers} layers ...", flush=True)
    sd = weight_loader.load_transformer_state_dict(
        range(args.n_layers), include_mtp=False
    )
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}", flush=True)
    del sd
    gc.collect()

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


# ----------------------------------------------------------------------------
# SPMD shard spec for the full model post-swap. Mesh axes are
# ("_axis_0", "_axis_1") to stay compatible with A2aSparseMLP's expert
# dispatch convention; attention specs map model->axis_1, batch->axis_0.
# ----------------------------------------------------------------------------


def make_mesh() -> Tuple[Mesh, Tuple[int, int]]:
    # Hardcoded 4x8 mesh; assumes a 32-device llmbox host.
    mesh_shape = (4, 8)
    num_devices = mesh_shape[0] * mesh_shape[1]
    assert xr.global_runtime_device_count() == num_devices, (
        f"Expected {num_devices} TT devices for mesh {mesh_shape}, got "
        f"{xr.global_runtime_device_count()}"
    )
    device_ids = np.array(range(num_devices))
    return Mesh(device_ids, mesh_shape, ("_axis_0", "_axis_1")), mesh_shape


def transformer_shard_spec(model: mdo.Transformer) -> Dict[torch.Tensor, Tuple]:
    """SPMD shard spec for every parameter and cache buffer in the post-swap
    Transformer. Mirrors `_attn_shard_spec` from
    test_deepseek_v4_tp_no_int.py:296 (with axis renames) for attention, and
    `get_shard_spec` from test_deepseek_v4_flash_moe.py:59 for MoE.
    """
    specs: Dict[torch.Tensor, Tuple] = {}

    # Top-level: replicate. Embed [vocab, dim] is large but sharding it would
    # need careful integration with ParallelEmbedding (out of scope here).
    specs[model.embed.weight] = (None, None)
    specs[model.norm.weight] = (None,)
    specs[model.head.weight] = (None, None)
    specs[model.hc_head_fn] = (None, None)
    specs[model.hc_head_base] = (None,)
    specs[model.hc_head_scale] = (None,)

    compound = ("_axis_0", "_axis_1")  # for compound-sharded experts

    for block in model.layers:
        # ---- Attention ----
        attn = block.attn
        specs[attn.wq_b.weight] = ("_axis_1", None)
        specs[attn.wo_a.weight] = ("_axis_1", None)
        specs[attn.wo_b.weight] = (None, "_axis_1")
        specs[attn.kv_cache] = ("_axis_0", None, None)
        if attn.compress_ratio:
            specs[attn.compressor.kv_cache] = ("_axis_0", None, None)
            specs[attn.compressor.kv_state] = ("_axis_0", None, None)
            specs[attn.compressor.score_state] = ("_axis_0", None, None)
            if attn.indexer is not None:
                specs[attn.indexer.wq_b.weight] = ("_axis_1", None)
                specs[attn.indexer.weights_proj.weight] = ("_axis_1", None)
                specs[attn.indexer.compressor.kv_cache] = ("_axis_0", None, None)
                specs[attn.indexer.compressor.kv_state] = ("_axis_0", None, None)
                specs[attn.indexer.compressor.score_state] = (
                    "_axis_0", None, None,
                )

        # ---- MoE (post-swap) ----
        # block.ffn is A2aSparseMLPWithSharedExperts; .mlp is A2aSparseMLP;
        # .shared_experts is the original Expert.
        a2a_with_shared = block.ffn
        mlp = a2a_with_shared.mlp
        # Replicate gate.weight (sharding it changes fp32 reduction order
        # and flips topk choices; see test_deepseek_v4_flash_moe.py:140-145).
        specs[mlp.router.gate.weight] = (None, None)
        # tid2eid (hash) or bias (non-hash): replicate.

        # Stacked experts: compound shard along both mesh axes.
        specs[mlp.experts.gate_proj] = (compound, None, None)
        specs[mlp.experts.up_proj] = (compound, None, None)
        specs[mlp.experts.down_proj] = (compound, None, None)

        shared = a2a_with_shared.shared_experts
        if shared is not None:
            specs[shared.w1.weight] = (None, None)
            specs[shared.w2.weight] = (None, None)
            specs[shared.w3.weight] = (None, None)

    return specs


# ----------------------------------------------------------------------------
# The test.
# ----------------------------------------------------------------------------


@pytest.mark.nightly
@pytest.mark.llmbox
@pytest.mark.parametrize("expert_dtype", ["bfp_bf4", "bfp_bf8"])
@torch.inference_mode()
def test_e2e_prefill_decode_full_real(expert_dtype: str) -> None:
    enable_spmd()
    xr.set_device_type("TT")
    torch.manual_seed(0)

    # --- Mesh + fixed batch ----------------------------------------------
    # bsz=32 is required by A2aSparseMLP's dispatch/combine op; on a (4, 8)
    # mesh that's 8 rows per _axis_0 device so the
    # ("_axis_0", None, None) kv/cache shardings stay valid. The same prompt
    # is tiled across the batch — every row generates the same answer.
    mesh, mesh_shape = make_mesh()
    bsz = BATCH_SIZE
    # assert bsz % mesh_shape[0] == 0, (
    #     f"BATCH_SIZE={bsz} must be divisible by mesh_shape[0]={mesh_shape[0]}"
    # )

    # --- Real config, MTP off, layer-prefix truncation --------------------
    args = weight_loader.load_config_args()
    args.n_mtp_layers = 0
    args.max_batch_size = bsz
    if NUM_LAYERS < args.n_layers:
        args.n_layers = NUM_LAYERS
        args.compress_ratios = args.compress_ratios[:NUM_LAYERS]
    # max_seq_len must cover both the prefill window (PROMPT_LEN) and all
    # decode positions. Round up to the next multiple of the largest
    # compress_ratio so every compressor's kv_cache has room for at least one
    # extra compressed slot beyond the prefill fill.
    max_compress_ratio = max(args.compress_ratios) if any(args.compress_ratios) else 1
    args.max_seq_len = math.ceil((PROMPT_LEN + MAX_NEW_TOKENS) / max_compress_ratio) * max_compress_ratio
    print(f"[args] n_layers={args.n_layers}, n_routed_experts={args.n_routed_experts}, "
          f"n_activated_experts={args.n_activated_experts}, "
          f"bsz={bsz}, max_seq_len={args.max_seq_len}, "
          f"compress_ratios={args.compress_ratios}", flush=True)

    # --- Tokenizer + 32 distinct prompts ----------------------------------
    # Each row gets its own prompt, left-padded to PROMPT_LEN so the actual
    # text sits at the right edge (most-recent context is intact for the
    # first decode token). Truncate from the left if a prompt overflows.
    from transformers import AutoTokenizer  # noqa: WPS433

    tokenizer = AutoTokenizer.from_pretrained(weight_loader.REPO_ID)
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    prompt_rows = []
    for prompt in PROMPTS:
        ids = tokenizer(
            prompt, return_tensors="pt", add_special_tokens=False
        ).input_ids[0]
        if ids.shape[0] >= PROMPT_LEN:
            ids = ids[-PROMPT_LEN:]
        else:
            pad = torch.full(
                (PROMPT_LEN - ids.shape[0],), pad_id, dtype=torch.long
            )
            ids = torch.cat([pad, ids], dim=0)
        prompt_rows.append(ids)
    prompt_ids = torch.stack(prompt_rows, dim=0).contiguous()
    assert prompt_ids.shape == (bsz, PROMPT_LEN)
    print(f"[tokenize] prompt_ids[0][-8:]={prompt_ids[0][-8:].tolist()}", flush=True)

    # --- Build model on CPU, load real weights, swap MoE per layer --------
    model = _build_and_load_full_model(args, mesh_shape)

    # --- Move to TT device ------------------------------------------------
    print("[device] moving model to TT (this releases CPU storage) ...", flush=True)
    device = torch_xla.device()
    model = model.to(device)
    gc.collect()

    for tensor, spec in transformer_shard_spec(model).items():
        xs.mark_sharding(tensor, mesh, spec)

    # --- Apply weight dtype overrides ------------------------------------
    # MoE routed experts + router: bfp4 
    weight_dtype_overrides = {
        # MoE routed experts (compound stacked across the 256 experts).
        "layers.*.ffn.mlp.experts.gate_proj": expert_dtype,
        "layers.*.ffn.mlp.experts.up_proj":   expert_dtype,
        "layers.*.ffn.mlp.experts.down_proj": expert_dtype,
        # Router gate: keep at bf16 (quantizing to bfp4/8 flips topk choices).
        "layers.*.ffn.mlp.router.gate.weight": "bf16",
    }
    applied = apply_weight_dtype_overrides(model, weight_dtype_overrides)
    print(f"[wdtype] applied {len(applied)} weight dtype overrides", flush=True)
    for path, dtype_str in applied:
        print(f"[wdtype]   {path} -> {dtype_str}", flush=True)

    hook = sharding_constraint_hook(model.head, mesh, (None, None))
    model.head.register_forward_hook(hook)
    # --- Compile (one handle, reused for prefill and every decode step) --
    # Compiled-mode pattern (mirrors test_prefill_decode_cache_coherence_compiled
    # in test_deepseek_v4_prefill_decode_loop_no_int.py:243-251): pass start_pos
    # as a fresh CPU->device tensor argument every call. dynamo turns it into a
    # symbolic graph input, so its value does NOT constant-fold per step. The
    # lazy-mode sp_buffer/one_buffer trick (lines 142-158 of that same file) is
    # NOT needed here and actively breaks the dynamo bridge's input-update path
    # (Bad StatusOr access during torch_xla.sync).
    #compile_options = {"tt_legacy_compile": True}
    compiled = torch.compile(model, backend="tt")

    # Per-row generation tracker. generated[i] is the list of token ids
    # produced for row i across prefill + every decode step.
    generated: list[list[int]] = [[] for _ in range(bsz)]

    # =================== PREFILL =====================================
    print("[prefill] compiling + running ...", flush=True)
    prompt_ids_tt = prompt_ids.to(device)
    xs.mark_sharding(prompt_ids_tt, mesh, ("_axis_0", None))
    sp_tt = torch.tensor(PROMPT_LEN, dtype=torch.long).to(device)
    prefill_logits = compiled(prompt_ids_tt, sp_tt)  # [bsz, vocab]
    torch_xla.sync(wait=True)
    next_ids = prefill_logits.detach().to("cpu").argmax(dim=-1)  # [bsz]
    for i in range(bsz):
        generated[i].append(int(next_ids[i].item()))
    print(f"[prefill] first ids[:8]={next_ids[:8].tolist()}", flush=True)

    # =================== DECODE LOOP =================================
    # All rows step together; we don't early-exit on per-row EOS so the
    # graph stays shape-stable. EOS trimming happens at print time.
    prev_token = next_ids.unsqueeze(1)  # [bsz, 1]
    for step in range(MAX_NEW_TOKENS - 1):
        start_pos = PROMPT_LEN + step
        prev_token_tt = prev_token.to(device)
        xs.mark_sharding(prev_token_tt, mesh, ("_axis_0", None))
        sp_tt = torch.tensor(start_pos, dtype=torch.long).to(device)
        decode_logits = compiled(prev_token_tt, sp_tt)
        #torch_xla.sync()

        next_ids = decode_logits.detach().to("cpu").argmax(dim=-1)  # [bsz]
        for i in range(bsz):
            generated[i].append(int(next_ids[i].item()))
        print(f"[decode {step + 1:>2}] sp={start_pos}: "
              f"ids[:8]={next_ids[:8].tolist()}", flush=True)
        prev_token = next_ids.unsqueeze(1)

    _print_decoded(tokenizer, PROMPTS, generated)


def _print_decoded(tokenizer, prompts: list, generated: list) -> None:
    eos = tokenizer.eos_token_id
    bar = "=" * 72
    print(f"\n{bar}", flush=True)
    for i, (prompt, ids) in enumerate(zip(prompts, generated)):
        # Trim the per-row continuation at the first EOS (if any) so
        # decoded text doesn't include junk past end-of-sequence.
        cut = next((k for k, t in enumerate(ids) if t == eos), len(ids))
        ids = ids[:cut]
        cont = tokenizer.decode(ids, skip_special_tokens=True)
        print(f"[row {i:02d}] prompt={prompt!r}")
        print(f"         ids={ids}")
        print(f"         cont={cont!r}")
    print(f"{bar}\n", flush=True)


# ----------------------------------------------------------------------------
# CPU-vs-device prefill PCC test.
# ----------------------------------------------------------------------------


PREFILL_PCC_REQUIRED = 0.95


def _pcc(x: torch.Tensor, y: torch.Tensor) -> float:
    """Pearson correlation. Mirrors _pcc in
    test_deepseek_v4_prefill_decode_loop_no_int.py: float64 conversion,
    allclose short-circuit (PCC is ill-conditioned when tensors are nearly
    identical), Pearson on flattened tensors."""
    x = x.detach().to(torch.float64).flatten()
    y = y.detach().to(torch.float64).flatten()
    if torch.allclose(x, y, rtol=1e-2, atol=1e-2):
        return 1.0
    if x.numel() <= 1:
        return 0.0
    vx = x - x.mean()
    vy = y - y.mean()
    denom = vx.norm() * vy.norm()
    if denom == 0:
        return float("nan")
    return float((vx @ vy) / denom)


def _reset_attn_caches(model: mdo.Transformer) -> None:
    """Zero in-place attention/compressor cache buffers populated by a prior
    forward pass so the next prefill starts from the same blank state as a
    fresh model. Mirrors the buffer set asserted in transformer_shard_spec."""
    for block in model.layers:
        attn = block.attn
        attn.kv_cache.zero_()
        if attn.compress_ratio:
            attn.compressor.kv_cache.zero_()
            attn.compressor.kv_state.zero_()
            attn.compressor.score_state.fill_(float("-inf"))
            if attn.indexer is not None:
                attn.indexer.compressor.kv_cache.zero_()
                attn.indexer.compressor.kv_state.zero_()
                attn.indexer.compressor.score_state.fill_(float("-inf"))


def _emulate_post_prefill_caches(
    model: mdo.Transformer, generator: torch.Generator, std: float = 0.1
) -> None:
    """Fill kv / compressor / indexer cache buffers with deterministic
    random values, emulating a non-empty post-prefill state. Same generator
    seed -> identical fills, so a CPU run and a device run can share bit-
    identical starting cache state without paying for a real CPU prefill.

    score_state would normally hold a mix of real scores and -inf for slots
    not yet computed; the model only uses it via softmax(dim=1) so finite
    random values are mathematically valid here (compressor logic at
    model_decode_opt.py:414,424). std=0.1 keeps softmax in-distribution."""
    for block in model.layers:
        attn = block.attn
        attn.kv_cache.normal_(mean=0.0, std=std, generator=generator)
        if attn.compress_ratio:
            attn.compressor.kv_cache.normal_(mean=0.0, std=std, generator=generator)
            attn.compressor.kv_state.normal_(mean=0.0, std=std, generator=generator)
            attn.compressor.score_state.normal_(mean=0.0, std=std, generator=generator)
            if attn.indexer is not None:
                attn.indexer.compressor.kv_cache.normal_(
                    mean=0.0, std=std, generator=generator
                )
                attn.indexer.compressor.kv_state.normal_(
                    mean=0.0, std=std, generator=generator
                )
                attn.indexer.compressor.score_state.normal_(
                    mean=0.0, std=std, generator=generator
                )


@lru_cache(maxsize=1)
def _get_deepseek_v4_tokenizer():
    from transformers import AutoTokenizer  # noqa: WPS433

    return AutoTokenizer.from_pretrained(weight_loader.REPO_ID)


def get_num_tokens(text: str) -> int:
    """Return the number of tokens DeepSeek-V4's tokenizer produces for `text`.

    Matches the encoding used by `_tokenize_prompts` (add_special_tokens=False).
    """
    tokenizer = _get_deepseek_v4_tokenizer()
    ids = tokenizer(text, add_special_tokens=False).input_ids
    return len(ids)


def test_prompts() -> None:
    """Sanity-check the structure of PROMPTS:

    - len(PROMPTS) == BATCH_SIZE (one prompt per batch row).
    - Every prompt tokenizes to exactly PROMPT_LEN (=128) DeepSeek-V4 tokens,
      so the left-pad in _tokenize_prompts is a no-op for these rows.
    - No tokenized prompt contains the EOS id, which would otherwise
      short-circuit the decode loop on the very first step.
    """
    tokenizer = _get_deepseek_v4_tokenizer()
    eos_id = tokenizer.eos_token_id

    assert len(PROMPTS) == BATCH_SIZE, (
        f"PROMPTS has {len(PROMPTS)} entries, must equal BATCH_SIZE={BATCH_SIZE}"
    )

    bad_len: list[tuple[int, int]] = []
    bad_eos: list[int] = []
    for i, prompt in enumerate(PROMPTS):
        ids = tokenizer(prompt, add_special_tokens=False).input_ids
        n = len(ids)
        assert n == get_num_tokens(prompt), "get_num_tokens disagrees with tokenizer"
        if n != PROMPT_LEN:
            bad_len.append((i, n))
        if eos_id is not None and eos_id in ids:
            bad_eos.append(i)

    assert not bad_len, (
        f"prompts not at {PROMPT_LEN} tokens (idx, len): {bad_len}"
    )
    assert not bad_eos, (
        f"prompts containing eos_id={eos_id}: {bad_eos}"
    )


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
            pad = torch.full(
                (PROMPT_LEN - ids.shape[0],), pad_id, dtype=torch.long
            )
            ids = torch.cat([pad, ids], dim=0)
        prompt_rows.append(ids)
    return torch.stack(prompt_rows, dim=0).contiguous()


@pytest.mark.nightly
@pytest.mark.llmbox
@pytest.mark.parametrize("num_layers", [1, 10, 15, 20, 30, 43])
@torch.inference_mode()
def test_e2e_prefill_pcc(num_layers: int) -> None:
    """Run prefill on CPU and on the TT device with a `num_layers`-deep
    swapped Transformer and PCC-compare both the final hidden-states output
    and the post-head logits.

    Both paths share one model object: the A2aSparseMLP swap routes through
    `_cpu_forward` -> original MoE while `hidden_states.device.type == "cpu"`
    (sparse_mlp.py:540) and through the all_to_all dispatch/combine kernels
    once moved to the TT device. CPU prefill runs first; its in-place kv /
    compressor / indexer cache mutations are zeroed before the device move
    so the device prefill starts from the same blank state.

    `Transformer.forward(..., return_hidden_states=True)` returns
    `(hidden_states, logits)` (model_decode_opt.py:972-981). hidden_states
    is the post-block, pre-head activation `h` of shape
    `[bsz, seq, hc_mult, dim]`; logits is `[bsz, vocab_size]` at the last
    position. Both are PCC-compared and both must clear PREFILL_PCC_REQUIRED.
    """
    enable_spmd()
    xr.set_device_type("TT")
    torch.manual_seed(0)

    mesh, mesh_shape = make_mesh()
    bsz = BATCH_SIZE

    args = weight_loader.load_config_args()
    args.n_mtp_layers = 0
    args.max_batch_size = bsz
    args.max_seq_len = 128
    if num_layers < args.n_layers:
        args.n_layers = num_layers
        args.compress_ratios = args.compress_ratios[:num_layers]
    print(
        f"[args] n_layers={args.n_layers}, "
        f"n_routed_experts={args.n_routed_experts}, "
        f"n_activated_experts={args.n_activated_experts}, bsz={bsz}, "
        f"max_seq_len={args.max_seq_len}, "
        f"compress_ratios={args.compress_ratios}",
        flush=True,
    )

    from transformers import AutoTokenizer  # noqa: WPS433

    tokenizer = AutoTokenizer.from_pretrained(weight_loader.REPO_ID)
    prompt_ids = _tokenize_prompts(tokenizer, PROMPTS)
    assert prompt_ids.shape == (bsz, PROMPT_LEN)
    print(
        f"[tokenize] prompt_ids[0][-8:]={prompt_ids[0][-8:].tolist()}",
        flush=True,
    )

    model = _build_and_load_full_model(args, mesh_shape)

    # --- CPU prefill ------------------------------------------------------
    print("[cpu] prefill ...", flush=True)
    sp_cpu = torch.tensor(PROMPT_LEN, dtype=torch.long)
    cpu_h, cpu_logits = model(prompt_ids, sp_cpu, return_hidden_states=True)
    cpu_out = cpu_h.detach().to(torch.float32).cpu().clone()
    cpu_logits_out = cpu_logits.detach().to(torch.float32).cpu().clone()
    print(
        f"[cpu] hidden_states shape={tuple(cpu_out.shape)} "
        f"logits shape={tuple(cpu_logits_out.shape)}",
        flush=True,
    )

    # CPU prefill mutated kv_cache / kv_state / score_state in place. Reset
    # them so the device prefill sees the same fresh-model starting state.
    _reset_attn_caches(model)

    # --- Move to device + shard ------------------------------------------
    print("[device] moving model to TT (this releases CPU storage) ...", flush=True)
    device = torch_xla.device()
    model = model.to(device)
    gc.collect()
    for tensor, spec in transformer_shard_spec(model).items():
        xs.mark_sharding(tensor, mesh, spec)

    # head is computed even under return_hidden_states=True (logits are the
    # second tuple element) — re-attach the sharding hook so head's output
    # carries the standard `(None, None)` constraint, matching
    # test_e2e_prefill_decode_full_real.
    hook = sharding_constraint_hook(model.head, mesh, (None, None))
    model.head.register_forward_hook(hook)

    compiled = torch.compile(model, backend="tt")

    prompt_ids_tt = prompt_ids.to(device)
    xs.mark_sharding(prompt_ids_tt, mesh, ("_axis_0", None))
    sp_tt = torch.tensor(PROMPT_LEN, dtype=torch.long).to(device)

    # --- Device prefill ---------------------------------------------------
    print("[device] compiling + running prefill ...", flush=True)
    device_h, device_logits = compiled(
        prompt_ids_tt, sp_tt, return_hidden_states=True
    )
    torch_xla.sync(wait=True)
    device_out = device_h.detach().to("cpu").to(torch.float32).clone()
    device_logits_out = device_logits.detach().to("cpu").to(torch.float32).clone()
    print(
        f"[device] hidden_states shape={tuple(device_out.shape)} "
        f"logits shape={tuple(device_logits_out.shape)}",
        flush=True,
    )

    assert cpu_out.shape == device_out.shape, (
        f"hidden_states shape mismatch: cpu={tuple(cpu_out.shape)} "
        f"device={tuple(device_out.shape)}"
    )
    assert cpu_logits_out.shape == device_logits_out.shape, (
        f"logits shape mismatch: cpu={tuple(cpu_logits_out.shape)} "
        f"device={tuple(device_logits_out.shape)}"
    )

    pcc_h = _pcc(cpu_out, device_out)
    pcc_logits = _pcc(cpu_logits_out, device_logits_out)
    print(
        f"[pcc] cpu vs device prefill hidden_states pcc={pcc_h:.6f} "
        f"logits pcc={pcc_logits:.6f} (required >= {PREFILL_PCC_REQUIRED})",
        flush=True,
    )
    failures = []
    if pcc_h < PREFILL_PCC_REQUIRED:
        failures.append(f"hidden_states={pcc_h:.6f}")
    if pcc_logits < PREFILL_PCC_REQUIRED:
        failures.append(f"logits={pcc_logits:.6f}")
    assert not failures, (
        f"PCC failed for: {failures} (required >= {PREFILL_PCC_REQUIRED})"
    )


@pytest.mark.nightly
@pytest.mark.llmbox
@pytest.mark.parametrize("num_layers", [10, 15, 20, 30, 43])
@torch.inference_mode()
def test_e2e_single_decode_pcc(num_layers: int) -> None:
    """Run one decode step (seqlen=1) on CPU and on the TT device starting
    from a deterministically-emulated post-prefill cache state, and
    PCC-compare the post-block, pre-head hidden states.

    Strategy: build one swapped Transformer on CPU. Fill kv / compressor /
    indexer caches with random values from a seeded generator, run a single
    CPU decode at start_pos=PROMPT_LEN. Re-fill the caches with the same
    seed (CPU decode mutated them in place), then move the model to the TT
    device — `.to(device)` carries the freshly re-emulated cache contents
    over so the device decode sees bit-identical starting state. Run a
    single device decode and compare.

    The "final hidden states" tensor compared here is the post-block,
    pre-head activation `h` returned by
    `Transformer.forward(..., return_hidden_states=True)`
    (model_decode_opt.py:972-981).

    Real prefill is not run because a 43-layer / 256-expert CPU prefill of
    a 128-token prompt is prohibitively slow; this test isolates decode-
    forward equivalence — what matters is that both sides start the decode
    step from identical caches.
    """
    enable_spmd()
    xr.set_device_type("TT")
    torch.manual_seed(0)

    mesh, mesh_shape = make_mesh()
    bsz = BATCH_SIZE

    args = weight_loader.load_config_args()
    args.n_mtp_layers = 0
    args.max_batch_size = bsz
    # max_seq_len > PROMPT_LEN: a decode at start_pos = PROMPT_LEN reads
    # `freqs_cis[start_pos]` directly (model_decode_opt.py:644), so the
    # precomputed table needs at least one entry past PROMPT_LEN. The
    # prefill+decode test gets away with max_seq_len = PROMPT_LEN = 128
    # because it only exercises decode under torch.compile on device, where
    # an OOB index_select doesn't raise the same way it does on CPU.
    args.max_seq_len = 2 * PROMPT_LEN
    if num_layers < args.n_layers:
        args.n_layers = num_layers
        args.compress_ratios = args.compress_ratios[:num_layers]
    print(
        f"[args] n_layers={args.n_layers}, "
        f"n_routed_experts={args.n_routed_experts}, "
        f"n_activated_experts={args.n_activated_experts}, bsz={bsz}, "
        f"max_seq_len={args.max_seq_len}, "
        f"compress_ratios={args.compress_ratios}",
        flush=True,
    )

    model = _build_and_load_full_model(args, mesh_shape)

    # Single-token decode input: id 0 per batch row. Deterministic and the
    # actual content doesn't affect the CPU-vs-device equivalence check.
    next_token = torch.zeros((bsz, 1), dtype=torch.long)
    cache_seed = 12345

    # --- CPU decode -------------------------------------------------------
    _emulate_post_prefill_caches(model, torch.Generator().manual_seed(cache_seed))
    print("[cpu] single decode (start_pos=PROMPT_LEN) ...", flush=True)
    sp_cpu = torch.tensor(PROMPT_LEN, dtype=torch.long)
    cpu_h, _ = model(next_token, sp_cpu, return_hidden_states=True)
    cpu_out = cpu_h.detach().to(torch.float32).cpu().clone()
    print(f"[cpu] hidden_states shape={tuple(cpu_out.shape)}", flush=True)

    # CPU decode mutated kv_cache + compressor / indexer state in place.
    # Re-emulate from the same seed so the device run starts from identical
    # cache contents.
    _emulate_post_prefill_caches(model, torch.Generator().manual_seed(cache_seed))

    # --- Move to device + shard ------------------------------------------
    print("[device] moving model to TT (this releases CPU storage) ...", flush=True)
    device = torch_xla.device()
    model = model.to(device)
    gc.collect()
    for tensor, spec in transformer_shard_spec(model).items():
        xs.mark_sharding(tensor, mesh, spec)

    hook = sharding_constraint_hook(model.head, mesh, (None, None))
    model.head.register_forward_hook(hook)

    compiled = torch.compile(model, backend="tt")

    next_token_tt = next_token.to(device)
    xs.mark_sharding(next_token_tt, mesh, ("_axis_0", None))
    sp_tt = torch.tensor(PROMPT_LEN, dtype=torch.long).to(device)

    # --- Device decode ----------------------------------------------------
    print("[device] compiling + running single decode ...", flush=True)
    device_h, _ = compiled(next_token_tt, sp_tt, return_hidden_states=True)
    torch_xla.sync(wait=True)
    device_out = device_h.detach().to("cpu").to(torch.float32).clone()
    print(f"[device] hidden_states shape={tuple(device_out.shape)}", flush=True)

    assert cpu_out.shape == device_out.shape, (
        f"shape mismatch: cpu={tuple(cpu_out.shape)} "
        f"device={tuple(device_out.shape)}"
    )

    pcc = _pcc(cpu_out, device_out)
    print(
        f"[pcc] cpu vs device single-decode hidden_states pcc={pcc:.6f} "
        f"(required >= {PREFILL_PCC_REQUIRED})",
        flush=True,
    )
    assert pcc >= PREFILL_PCC_REQUIRED, (
        f"PCC failed: got {pcc:.6f}, required >= {PREFILL_PCC_REQUIRED}"
    )


@pytest.mark.nightly
@pytest.mark.llmbox
@pytest.mark.parametrize("num_layers", [10, 15, 20, 30, 43])
@pytest.mark.parametrize("num_iterations", [1, 2, 5, 10])
@pytest.mark.parametrize("use_cpu_decode_inputs", [True, False])
@torch.inference_mode()
def test_prefill_and_decode_pcc(
    num_layers: int,
    num_iterations: int,
    use_cpu_decode_inputs: bool,
) -> None:
    """Run prefill + `num_iterations` decode steps on CPU, then on the TT
    device, capturing the post-block hidden states tensor at every step.
    PCC-compare CPU vs device for the prefill output and for each decode
    step.

    Decode is driven by real logits: the first decode token comes from
    argmax of the prefill logits, and each subsequent decode token comes
    from argmax of the previous decode's logits — autoregressive
    generation.

    `use_cpu_decode_inputs` selects how the device decode loop is fed:
      - True: the device replays the CPU-derived token sequence verbatim,
        so CPU and device see identical inputs at every decode step and
        the per-step PCC isolates numeric divergence rather than diverging
        argmax choices. This is the path whose decode-step PCCs are
        asserted.
      - False: the device derives its own next-token sequence from its
        own logits. At the end of every decode step we record the per-row
        match count between the token the device just argmaxed and the
        token CPU argmaxed at the same step, and print the list at the
        end (no assert — the user wants to inspect divergence). Decode-
        step PCCs are still computed and printed but not asserted, since
        once argmaxes diverge the inputs to subsequent steps differ and
        decode-PCC stops being equivalence-meaningful.

    `Transformer.forward(..., return_hidden_states=True)` returns
    `(hidden_states, logits)` (model_decode_opt.py:972-981). hidden_states
    is `[bsz, seqlen, hc_mult, dim]` (seqlen=PROMPT_LEN for prefill, 1 for
    each decode); logits is `[bsz, vocab_size]` at the last position.

    Caches mutate in place across prefill+decodes; CPU caches are zeroed
    before the device move so the device run starts from a fresh-model
    state matching the CPU run.
    """
    enable_spmd()
    xr.set_device_type("TT")
    torch.manual_seed(0)

    mesh, mesh_shape = make_mesh()
    bsz = BATCH_SIZE

    args = weight_loader.load_config_args()
    args.n_mtp_layers = 0
    args.max_batch_size = bsz
    # Extend the max_seq_len to allow kv_cache tensors to store all prefill+decode tokens
    args.max_seq_len = 2 * PROMPT_LEN
    assert PROMPT_LEN + num_iterations - 1 < args.max_seq_len, (
        f"PROMPT_LEN+num_iterations-1 ({PROMPT_LEN + num_iterations - 1}) "
        f">= max_seq_len ({args.max_seq_len})"
    )
    if num_layers < args.n_layers:
        args.n_layers = num_layers
        args.compress_ratios = args.compress_ratios[:num_layers]
    print(
        f"[args] n_layers={args.n_layers}, "
        f"n_routed_experts={args.n_routed_experts}, "
        f"n_activated_experts={args.n_activated_experts}, bsz={bsz}, "
        f"max_seq_len={args.max_seq_len}, num_iterations={num_iterations}, "
        f"compress_ratios={args.compress_ratios}",
        flush=True,
    )

    from transformers import AutoTokenizer  # noqa: WPS433

    tokenizer = AutoTokenizer.from_pretrained(weight_loader.REPO_ID)
    prompt_ids = _tokenize_prompts(tokenizer, PROMPTS)
    assert prompt_ids.shape == (bsz, PROMPT_LEN)

    model = _build_and_load_full_model(args, mesh_shape)

    # ---- CPU prefill + decodes ------------------------------------------
    # decode_inputs[k] is the [bsz, 1] token tensor fed into CPU decode
    # step k (decode_inputs[0] is argmax of prefill logits;
    # decode_inputs[k>0] is argmax of decode step k-1's logits).
    # cpu_generated_tokens[k] is the token CPU produced *at* decode step k
    # (= argmax of step k's logits = decode_inputs[k+1] when k+1 exists).
    decode_inputs: list[torch.Tensor] = []
    cpu_generated_tokens: list[torch.Tensor] = []

    print("[cpu] prefill ...", flush=True)
    sp_prefill = torch.tensor(PROMPT_LEN, dtype=torch.long)
    # h_pf is the hidden states tensor returned after prefill
    # logits_pf is the logits tensor returned after prefill
    h_pf, logits_pf = model(prompt_ids, sp_prefill, return_hidden_states=True)
    cpu_prefill_h = h_pf.detach().to(torch.float32).cpu().clone()
    next_token = logits_pf.detach().cpu().argmax(dim=-1, keepdim=True).to(torch.long)
    print(
        f"[cpu] prefill hidden_states shape={tuple(cpu_prefill_h.shape)} "
        f"first_ids[:8]={next_token[:8, 0].tolist()}",
        flush=True,
    )

    cpu_decode_hs: list[torch.Tensor] = []
    for step in range(num_iterations):
        decode_inputs.append(next_token)
        sp_step = torch.tensor(PROMPT_LEN + step, dtype=torch.long)
        h_d, logits_d = model(next_token, sp_step, return_hidden_states=True)
        h = h_d.detach().to(torch.float32).cpu().clone()
        cpu_decode_hs.append(h)
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
    gc.collect()
    for tensor, spec in transformer_shard_spec(model).items():
        xs.mark_sharding(tensor, mesh, spec)

    hook = sharding_constraint_hook(model.head, mesh, (None, None))
    model.head.register_forward_hook(hook)

    compiled = torch.compile(model, backend="tt")

    # ---- Device prefill + decodes --------------------------------------
    prompt_ids_tt = prompt_ids.to(device)
    xs.mark_sharding(prompt_ids_tt, mesh, ("_axis_0", None))
    sp_prefill_tt = torch.tensor(PROMPT_LEN, dtype=torch.long).to(device)
    print("[device] compiling + running prefill ...", flush=True)
    h_pf_dev, logits_pf_dev = compiled(
        prompt_ids_tt, sp_prefill_tt, return_hidden_states=True
    )
    torch_xla.sync(wait=True)
    device_prefill_h = h_pf_dev.detach().to("cpu").to(torch.float32).clone()
    # Device's first decode input when running self-driven (False): the
    # argmax of device's own prefill logits. Materialize now via .to("cpu")
    # so the lazy-graph result for prefill is snapshotted before the
    # decode loop mutates caches.
    next_token_dev = (
        logits_pf_dev.detach().to("cpu").argmax(dim=-1, keepdim=True).to(torch.long)
    )
    print(
        f"[device] prefill hidden_states shape={tuple(device_prefill_h.shape)} "
        f"first_ids[:8]={next_token_dev[:8, 0].tolist()}",
        flush=True,
    )

    device_decode_hs: list[torch.Tensor] = []
    # When use_cpu_decode_inputs=False, store per-step (matches, total) of
    # device-vs-cpu argmax tokens. Empty in the True path.
    token_match_results: list[tuple[int, int, int]] = []

    for step in range(num_iterations):
        if use_cpu_decode_inputs:
            # Replay the CPU-derived token sequence so CPU and device
            # decode steps see identical inputs.
            token_in = decode_inputs[step]
        else:
            # Self-driven: feed device its own previous-step argmax.
            token_in = next_token_dev

        decode_token_tt = token_in.to(device)
        xs.mark_sharding(decode_token_tt, mesh, ("_axis_0", None))
        sp_step_tt = torch.tensor(PROMPT_LEN + step, dtype=torch.long).to(device)
        h_d_dev, logits_d_dev = compiled(
            decode_token_tt, sp_step_tt, return_hidden_states=True
        )
        # `.to("cpu")` forces lazy-graph materialization, snapshotting the
        # i-th decode's hidden states before the next step mutates caches.
        h = h_d_dev.detach().to("cpu").to(torch.float32).clone()
        device_decode_hs.append(h)
        dev_argmax = (
            logits_d_dev.detach().to("cpu").argmax(dim=-1, keepdim=True).to(torch.long)
        )
        next_token_dev = dev_argmax  # used as next-step input when not replaying CPU

        if not use_cpu_decode_inputs:
            matches = int((dev_argmax == cpu_generated_tokens[step]).sum().item())
            token_match_results.append((step, matches, bsz))

        print(
            f"[device] decode step {step} sp={PROMPT_LEN + step} "
            f"shape={tuple(h.shape)} "
            f"next_ids[:8]={dev_argmax[:8, 0].tolist()}",
            flush=True,
        )

    # ---- PCC comparisons ------------------------------------------------
    pccs: list[tuple[str, float]] = []

    assert cpu_prefill_h.shape == device_prefill_h.shape, (
        f"prefill shape mismatch: cpu={tuple(cpu_prefill_h.shape)} "
        f"device={tuple(device_prefill_h.shape)}"
    )
    pcc_prefill = _pcc(cpu_prefill_h, device_prefill_h)
    pccs.append(("prefill", pcc_prefill))
    print(f"[pcc] prefill pcc={pcc_prefill:.6f}", flush=True)

    decode_pccs: list[tuple[str, float]] = []
    for i, (cpu_h, dev_h) in enumerate(zip(cpu_decode_hs, device_decode_hs)):
        assert cpu_h.shape == dev_h.shape, (
            f"decode[{i}] shape mismatch: cpu={tuple(cpu_h.shape)} "
            f"device={tuple(dev_h.shape)}"
        )
        p = _pcc(cpu_h, dev_h)
        decode_pccs.append((f"decode[{i}]", p))
        print(f"[pcc] decode step {i} pcc={p:.6f}", flush=True)

    # ---- Token-sequence match (False-path only) ------------------------
    if not use_cpu_decode_inputs:
        print(
            "[match] device-vs-cpu argmax tokens per decode step "
            f"(use_cpu_decode_inputs=False, total={bsz} rows):",
            flush=True,
        )
        for step, matches, total in token_match_results:
            print(
                f"  step {step}: {matches}/{total} rows match",
                flush=True,
            )

    # ---- Assertions ----------------------------------------------------
    # Prefill PCC is always asserted (prefill inputs are identical for both
    # paths). Decode PCCs are asserted only when use_cpu_decode_inputs=True
    # — once self-driven argmaxes diverge, decode-step inputs differ between
    # CPU and device and the PCC stops being equivalence-meaningful.
    failures = []
    if pcc_prefill < PREFILL_PCC_REQUIRED:
        failures.append(f"prefill={pcc_prefill:.6f}")
    if use_cpu_decode_inputs:
        for label, p in decode_pccs:
            if p < PREFILL_PCC_REQUIRED:
                failures.append(f"{label}={p:.6f}")
    assert not failures, (
        f"PCC failed for: {failures} (required >= {PREFILL_PCC_REQUIRED})"
    )


# ----------------------------------------------------------------------------
# Diagnostic: pinpoint *which* sub-step inside ParallelHead causes the logits
# PCC to fall well below hidden-states PCC. Runs prefill with
# `return_head_intermediates=True` so both CPU and device emit
#   (h, logits, hc_out, norm_out, last_token)
# and PCC-compares each pair. The first stage that drops below a threshold
# localizes the responsible op inside the head pipeline.
# ----------------------------------------------------------------------------


@pytest.mark.nightly
@pytest.mark.llmbox
@pytest.mark.parametrize("num_layers", [1])
@torch.inference_mode()
def test_e2e_prefill_head_intermediates_pcc(num_layers: int) -> None:
    enable_spmd()
    xr.set_device_type("TT")
    torch.manual_seed(0)

    mesh, mesh_shape = make_mesh()
    bsz = BATCH_SIZE

    args = weight_loader.load_config_args()
    args.n_mtp_layers = 0
    args.max_batch_size = bsz
    args.max_seq_len = 128
    if num_layers < args.n_layers:
        args.n_layers = num_layers
        args.compress_ratios = args.compress_ratios[:num_layers]

    from transformers import AutoTokenizer  # noqa: WPS433

    tokenizer = AutoTokenizer.from_pretrained(weight_loader.REPO_ID)
    prompt_ids = _tokenize_prompts(tokenizer, PROMPTS)
    assert prompt_ids.shape == (bsz, PROMPT_LEN)

    model = _build_and_load_full_model(args, mesh_shape)

    # --- CPU prefill, returning all head intermediates --------------------
    print("[cpu] prefill (with head intermediates) ...", flush=True)
    sp_cpu = torch.tensor(PROMPT_LEN, dtype=torch.long)
    cpu_h, cpu_logits, cpu_hc_out, cpu_norm_out, cpu_last_token = model(
        prompt_ids, sp_cpu, return_head_intermediates=True
    )
    cpu_h_f32 = cpu_h.detach().to(torch.float32).cpu().clone()
    cpu_hc_f32 = cpu_hc_out.detach().to(torch.float32).cpu().clone()
    cpu_norm_f32 = cpu_norm_out.detach().to(torch.float32).cpu().clone()
    cpu_last_f32 = cpu_last_token.detach().to(torch.float32).cpu().clone()
    cpu_logits_f32 = cpu_logits.detach().to(torch.float32).cpu().clone()
    print(
        f"[cpu] shapes: h={tuple(cpu_h_f32.shape)} hc={tuple(cpu_hc_f32.shape)} "
        f"norm={tuple(cpu_norm_f32.shape)} last={tuple(cpu_last_f32.shape)} "
        f"logits={tuple(cpu_logits_f32.shape)}",
        flush=True,
    )

    _reset_attn_caches(model)

    # --- Move to device + shard ------------------------------------------
    print("[device] moving model to TT ...", flush=True)
    device = torch_xla.device()
    model = model.to(device)
    gc.collect()
    for tensor, spec in transformer_shard_spec(model).items():
        xs.mark_sharding(tensor, mesh, spec)

    # Replicate the (None, None) constraint that test_e2e_prefill_pcc applies
    # to the head's logits output, but apply it only to the logits element of
    # the (logits, hc_out, norm_out, last_token) tuple that head returns when
    # `return_intermediates=True`. Otherwise the default
    # sharding_constraint_hook wraps the whole tuple and torch.compile rejects
    # it (sharding_constraint expects a Tensor).
    base_hook = sharding_constraint_hook(model.head, mesh, (None, None))

    def _logits_only_hook(mod, input, output):
        if isinstance(output, tuple):
            logits = base_hook(mod, input, output[0])
            return (logits,) + output[1:]
        return base_hook(mod, input, output)

    model.head.register_forward_hook(_logits_only_hook)

    compiled = torch.compile(model, backend="tt")

    prompt_ids_tt = prompt_ids.to(device)
    xs.mark_sharding(prompt_ids_tt, mesh, ("_axis_0", None))
    sp_tt = torch.tensor(PROMPT_LEN, dtype=torch.long).to(device)

    print("[device] compiling + running prefill ...", flush=True)
    dev_h, dev_logits, dev_hc_out, dev_norm_out, dev_last_token = compiled(
        prompt_ids_tt, sp_tt, return_head_intermediates=True
    )
    torch_xla.sync(wait=True)
    dev_h_f32 = dev_h.detach().to("cpu").to(torch.float32).clone()
    dev_hc_f32 = dev_hc_out.detach().to("cpu").to(torch.float32).clone()
    dev_norm_f32 = dev_norm_out.detach().to("cpu").to(torch.float32).clone()
    dev_last_f32 = dev_last_token.detach().to("cpu").to(torch.float32).clone()
    dev_logits_f32 = dev_logits.detach().to("cpu").to(torch.float32).clone()
    print(
        f"[device] shapes: h={tuple(dev_h_f32.shape)} hc={tuple(dev_hc_f32.shape)} "
        f"norm={tuple(dev_norm_f32.shape)} last={tuple(dev_last_f32.shape)} "
        f"logits={tuple(dev_logits_f32.shape)}",
        flush=True,
    )

    stages = [
        ("h            (post-blocks, pre-head)", cpu_h_f32, dev_h_f32),
        ("hc_out       (post hc_head, pre norm)", cpu_hc_f32, dev_hc_f32),
        ("norm_out     (post RMSNorm, pre slice)", cpu_norm_f32, dev_norm_f32),
        ("last_token   (post slice [:, -1].float())", cpu_last_f32, dev_last_f32),
        ("logits       (post head matmul + gather)", cpu_logits_f32, dev_logits_f32),
    ]

    print("\n[head-pcc] CPU vs device per-stage PCC:", flush=True)
    for label, cpu_t, dev_t in stages:
        assert cpu_t.shape == dev_t.shape, (
            f"shape mismatch for {label}: cpu={tuple(cpu_t.shape)} "
            f"device={tuple(dev_t.shape)}"
        )
        diff = (cpu_t - dev_t).abs()
        rel = diff / (cpu_t.abs().clamp_min(1e-9))
        p = _pcc(cpu_t, dev_t)
        print(
            f"  {label:48s} pcc={p:.6f}  "
            f"max_abs_diff={float(diff.max()):.4e}  "
            f"mean_abs_diff={float(diff.mean()):.4e}  "
            f"max_rel_diff={float(rel.max()):.4e}",
            flush=True,
        )

    # --- Per-position PCC localization ----------------------------------
    # The slice [:, -1] cliff (norm_out 0.998 -> last_token 0.91) means
    # position 127 specifically diverges far more than the average
    # position. Confirm by computing PCC at each individual seq position
    # of norm_out and h.
    def _per_pos_pcc(label: str, cpu_3d: torch.Tensor, dev_3d: torch.Tensor):
        # accept both [B, S, D] (norm/hc) and [B, S, hc, D] (h) by flattening
        # the trailing dims so we always end up with one PCC per seq position.
        assert cpu_3d.shape == dev_3d.shape
        S = cpu_3d.shape[1]
        cpu_flat = cpu_3d.reshape(cpu_3d.shape[0], S, -1)
        dev_flat = dev_3d.reshape(dev_3d.shape[0], S, -1)
        per_pos = [_pcc(cpu_flat[:, s, :], dev_flat[:, s, :]) for s in range(S)]
        worst_idx = int(min(range(S), key=lambda i: per_pos[i]))
        best_idx = int(max(range(S), key=lambda i: per_pos[i]))
        print(
            f"  {label:24s} per-pos PCC: pos{worst_idx:>3d}(worst)={per_pos[worst_idx]:.6f}  "
            f"pos{best_idx:>3d}(best)={per_pos[best_idx]:.6f}  "
            f"pos127(last)={per_pos[127]:.6f}  pos0={per_pos[0]:.6f}  "
            f"pos64={per_pos[64]:.6f}",
            flush=True,
        )
        # surface the bottom 5 positions for full visibility
        bottom = sorted(range(S), key=lambda i: per_pos[i])[:5]
        print(
            f"  {label:24s} bottom 5 positions: "
            + ", ".join(f"pos{i}={per_pos[i]:.4f}" for i in bottom),
            flush=True,
        )

    print("\n[per-pos-pcc] per-sequence-position PCC:", flush=True)
    _per_pos_pcc("h", cpu_h_f32, dev_h_f32)
    _per_pos_pcc("hc_out", cpu_hc_f32, dev_hc_f32)
    _per_pos_pcc("norm_out", cpu_norm_f32, dev_norm_f32)


@pytest.mark.nightly
@pytest.mark.llmbox
@pytest.mark.parametrize("num_layers", [1])
@pytest.mark.parametrize("content_variant", ["short", "long"])
@torch.inference_mode()
def test_e2e_prefill_block_intermediates_pcc(num_layers: int, content_variant: str) -> None:
    """Block-level diagnostic: capture (h, attn_out, post_attn, ffn_out) from
    layer 0 on both CPU and device, then compute per-sequence-position PCC for
    each tensor. Localizes whether the boundary divergence (positions 125-127
    in `h` for prefill_len=128) lives in the attention sub-block or the
    MoE/FFN sub-block.

    `content_variant`:
      - "short": uses the default `PROMPTS` (4-10 content tokens per row,
        ~118 left-pad). The original 10/118-split diagnostic.
      - "long": uses `LONGER_PROMPTS` (≈55-65 content tokens per row,
        ≈63-73 left-pad). Used to test the OOD-router-margin prediction
        that pos127 PCC improves when content tokens dominate the
        attention context.

    Only valid for num_layers=1.
    """
    enable_spmd()
    xr.set_device_type("TT")
    torch.manual_seed(0)

    mesh, mesh_shape = make_mesh()
    bsz = BATCH_SIZE

    args = weight_loader.load_config_args()
    args.n_mtp_layers = 0
    args.max_batch_size = bsz
    args.max_seq_len = 128
    if num_layers < args.n_layers:
        args.n_layers = num_layers
        args.compress_ratios = args.compress_ratios[:num_layers]

    from transformers import AutoTokenizer  # noqa: WPS433

    tokenizer = AutoTokenizer.from_pretrained(weight_loader.REPO_ID)
    prompts_source = PROMPTS if content_variant == "short" else LONGER_PROMPTS
    print(f"[content_variant] {content_variant!r} ({len(prompts_source)} prompts)", flush=True)
    prompt_ids = _tokenize_prompts(tokenizer, prompts_source)

    # Print per-position pad-token-count so we can correlate position vs
    # content distribution. For short: positions 124-127 are content-heavy
    # across batch rows; for long: positions ~68-127 are content-heavy.
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    pad_count_per_pos = (prompt_ids == pad_id).sum(dim=0).tolist()
    content_per_row = (prompt_ids != pad_id).sum(dim=-1).tolist()
    print(
        f"[content] per-row content-token counts: {content_per_row}",
        flush=True,
    )
    # Find the first all-content position (where every row is non-pad).
    all_content_positions = [
        p for p in range(PROMPT_LEN)
        if pad_count_per_pos[p] == 0
    ]
    first_all_content = all_content_positions[0] if all_content_positions else PROMPT_LEN
    print(
        f"[content] first all-content position: {first_all_content}",
        flush=True,
    )
    sample_positions = sorted(set([
        0, 32, 64, first_all_content - 4, first_all_content - 2,
        first_all_content - 1, first_all_content, first_all_content + 2,
        120, 124, 125, 126, 127,
    ]))
    sample_positions = [p for p in sample_positions if 0 <= p < PROMPT_LEN]
    print(
        "[content] non-pad rows at sampled positions: "
        + ", ".join(
            f"pos{p}={bsz - pad_count_per_pos[p]}/{bsz}"
            for p in sample_positions
        ),
        flush=True,
    )

    model = _build_and_load_full_model(args, mesh_shape)

    # --- CPU prefill, returning all block + head intermediates ---------
    print("[cpu] prefill (with block intermediates) ...", flush=True)
    sp_cpu = torch.tensor(PROMPT_LEN, dtype=torch.long)
    (
        cpu_h, cpu_logits, cpu_hc_out, cpu_norm_out, cpu_last_token,
        cpu_attn_out, cpu_post_attn, cpu_ffn_out,
    ) = model(prompt_ids, sp_cpu, return_block_intermediates=True)

    cpu_h_f32 = cpu_h.detach().to(torch.float32).cpu().clone()
    cpu_attn_f32 = cpu_attn_out.detach().to(torch.float32).cpu().clone()
    cpu_post_attn_f32 = cpu_post_attn.detach().to(torch.float32).cpu().clone()
    cpu_ffn_f32 = cpu_ffn_out.detach().to(torch.float32).cpu().clone()
    print(
        f"[cpu] shapes: attn_out={tuple(cpu_attn_f32.shape)} "
        f"post_attn={tuple(cpu_post_attn_f32.shape)} "
        f"ffn_out={tuple(cpu_ffn_f32.shape)}",
        flush=True,
    )

    _reset_attn_caches(model)

    # --- Move to device + shard --------------------------------------
    print("[device] moving model to TT ...", flush=True)
    device = torch_xla.device()
    model = model.to(device)
    gc.collect()
    for tensor, spec in transformer_shard_spec(model).items():
        xs.mark_sharding(tensor, mesh, spec)

    base_hook = sharding_constraint_hook(model.head, mesh, (None, None))

    def _logits_only_hook(mod, input, output):
        if isinstance(output, tuple):
            logits = base_hook(mod, input, output[0])
            return (logits,) + output[1:]
        return base_hook(mod, input, output)

    model.head.register_forward_hook(_logits_only_hook)

    compiled = torch.compile(model, backend="tt")

    prompt_ids_tt = prompt_ids.to(device)
    xs.mark_sharding(prompt_ids_tt, mesh, ("_axis_0", None))
    sp_tt = torch.tensor(PROMPT_LEN, dtype=torch.long).to(device)

    print("[device] compiling + running prefill ...", flush=True)
    (
        dev_h, dev_logits, dev_hc_out, dev_norm_out, dev_last_token,
        dev_attn_out, dev_post_attn, dev_ffn_out,
    ) = compiled(prompt_ids_tt, sp_tt, return_block_intermediates=True)
    torch_xla.sync(wait=True)

    dev_h_f32 = dev_h.detach().to("cpu").to(torch.float32).clone()
    dev_attn_f32 = dev_attn_out.detach().to("cpu").to(torch.float32).clone()
    dev_post_attn_f32 = dev_post_attn.detach().to("cpu").to(torch.float32).clone()
    dev_ffn_f32 = dev_ffn_out.detach().to("cpu").to(torch.float32).clone()

    def _per_pos_pcc(label: str, cpu_t: torch.Tensor, dev_t: torch.Tensor):
        assert cpu_t.shape == dev_t.shape, (
            f"shape mismatch for {label}: cpu={tuple(cpu_t.shape)} dev={tuple(dev_t.shape)}"
        )
        S = cpu_t.shape[1]
        cpu_flat = cpu_t.reshape(cpu_t.shape[0], S, -1)
        dev_flat = dev_t.reshape(dev_t.shape[0], S, -1)
        per_pos = [_pcc(cpu_flat[:, s, :], dev_flat[:, s, :]) for s in range(S)]
        bottom = sorted(range(S), key=lambda i: per_pos[i])[:6]
        # Print PCC at the same sampled positions used for the content
        # histogram, so the cliff position is visible regardless of variant.
        spline = "  ".join(f"pos{p}={per_pos[p]:.4f}" for p in sample_positions)
        print(
            f"  {label:18s} aggregate={_pcc(cpu_t, dev_t):.6f}",
            flush=True,
        )
        print(f"  {label:18s} {spline}", flush=True)
        print(
            f"  {label:18s} bottom 6: "
            + ", ".join(f"pos{i}={per_pos[i]:.4f}" for i in bottom),
            flush=True,
        )

    print("\n[per-pos-pcc] block-level per-position PCC:", flush=True)
    _per_pos_pcc("attn_out", cpu_attn_f32, dev_attn_f32)
    _per_pos_pcc("post_attn", cpu_post_attn_f32, dev_post_attn_f32)
    _per_pos_pcc("ffn_out", cpu_ffn_f32, dev_ffn_f32)
    _per_pos_pcc("h (block out)", cpu_h_f32, dev_h_f32)


# ----------------------------------------------------------------------------
# CPU-only prefill: sanity-check model output without any device involvement.
# Complements the cross-implementation PCC tests by isolating the model
# behavior from CPU-vs-device numerical disagreement — useful when
# diagnosing whether a "low PCC" failure is a model-quality issue (would
# show up here too) or just a router-tiebreak disagreement (would not).
# ----------------------------------------------------------------------------


@pytest.mark.nightly
@pytest.mark.llmbox
@pytest.mark.parametrize("num_layers", [1, 10, 15, 20, 30, 43])
@torch.inference_mode()
def test_cpu_only_prefill(num_layers: int) -> None:
    """Run prefill on CPU only with a `num_layers`-deep swapped Transformer
    and print the next-token prediction (decoded to text) for each prompt
    in PROMPTS.

    No `.to(device)` call: the model and all its weights stay on the host;
    the A2aSparseMLP swap routes through `_cpu_forward` -> original MoE
    while `hidden_states.device.type == "cpu"`. Output of one prefill is
    `[bsz, vocab_size]`; the printed token is `argmax(dim=-1)` for each
    row.

    Mesh setup mirrors the PCC tests so the build_and_load path is
    identical (the swap needs a `mesh_shape` tuple), but the mesh is never
    actually used for sharding since nothing crosses the device boundary.
    """
    # enable_spmd()
    # xr.set_device_type("TT")
    torch.manual_seed(0)

    _, mesh_shape = make_mesh()
    bsz = BATCH_SIZE

    args = weight_loader.load_config_args()
    args.n_mtp_layers = 0
    args.max_batch_size = bsz
    args.max_seq_len = 128
    if num_layers < args.n_layers:
        args.n_layers = num_layers
        args.compress_ratios = args.compress_ratios[:num_layers]
    print(
        f"[args] n_layers={args.n_layers}, "
        f"n_routed_experts={args.n_routed_experts}, "
        f"n_activated_experts={args.n_activated_experts}, bsz={bsz}, "
        f"max_seq_len={args.max_seq_len}, "
        f"compress_ratios={args.compress_ratios}",
        flush=True,
    )

    from transformers import AutoTokenizer  # noqa: WPS433

    tokenizer = AutoTokenizer.from_pretrained(weight_loader.REPO_ID)
    prompt_ids = _tokenize_prompts(tokenizer, PROMPTS)
    assert prompt_ids.shape == (bsz, PROMPT_LEN)

    # Print per-position pad/content distribution so the reader can see
    # how heavily left-padded these prompts are. Most PROMPTS in this
    # file are 5-10 tokens, so positions 0-117 are pad-dominated and the
    # actual content sits at the right edge.
    pad_id = (
        tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None
        else tokenizer.eos_token_id
    )
    content_per_row = (prompt_ids != pad_id).sum(dim=-1).tolist()
    print(
        f"[content] per-row content-token counts (out of {PROMPT_LEN}): "
        f"{content_per_row}",
        flush=True,
    )

    model = _build_and_load_full_model(args, mesh_shape)

    print("[cpu] running prefill ...", flush=True)
    sp_cpu = torch.tensor(PROMPT_LEN, dtype=torch.long)
    logits = model(prompt_ids, sp_cpu)  # [bsz, vocab_size]
    next_ids = logits.detach().cpu().argmax(dim=-1)  # [bsz]
    print(f"[cpu] logits shape={tuple(logits.shape)}", flush=True)

    bar = "=" * 72
    print(f"\n{bar}", flush=True)
    print(
        f"[cpu_only_prefill] num_layers={num_layers} — next-token "
        f"prediction per prompt (CPU reference, no device):",
        flush=True,
    )
    print(f"{bar}", flush=True)
    for i, prompt in enumerate(PROMPTS):
        next_tok_id = int(next_ids[i].item())
        # Print both the special-tokens-included form (so EOS / BOS show
        # up if the model predicted them) and the user-readable form.
        next_tok_raw = tokenizer.decode([next_tok_id], skip_special_tokens=False)
        next_tok_clean = tokenizer.decode([next_tok_id], skip_special_tokens=True)
        print(f"[row {i:02d}] prompt={prompt!r}")
        print(
            f"         content_tokens={content_per_row[i]}  "
            f"next_id={next_tok_id}  "
            f"next_tok_raw={next_tok_raw!r}  "
            f"next_tok={next_tok_clean!r}"
        )
    print(f"{bar}\n", flush=True)
