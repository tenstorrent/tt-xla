# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Realistic inputs for DeepSeek-V4-Flash MoE tests.

Provides cached `(input_ids, hidden_states)` pairs derived from a fixed
natural-language passage and CPU forward pass through the actual model
prefix. Replaces the historical `torch.randn / torch.randint` placeholders
in the FFN/single-layer tests so the gate sees a realistic activation
distribution and routes to non-uniform experts.

`hidden_states` corresponds to the input that goes directly into
`Block(layer_id=LAYER_ID).ffn` — i.e. the value right before the MoE gate
matmul. To keep the CPU prefix tractable, `LAYER_ID` defaults to
`args.n_hash_layers` (= 3), the first score-routed layer; the prefix run
covers `embed -> hc_expand -> layers[0..2] (hash MoE) -> layers[LAYER_ID]
(hc_pre + attn_norm + attn + hc_post + hc_pre + ffn_norm)`.

First-run cost: ~12-15 GB HF download + several minutes of CPU forward.
The result is cached as a single `.pt` checked into the repo so subsequent
runs (and other contributors) just `torch.load` a ~16 MB file.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

import torch

from . import weight_loader

# Cache file lives next to the tests so it ships with the repo. Versioned
# field in the saved dict lets us invalidate when generation logic changes.
_CACHE_DIR = os.path.join(os.path.dirname(__file__), "_cached_inputs")
_CACHE_VERSION = 3  # bumped: _CACHE_SEQ raised to 128 for prefill+decode use.

# Largest shape we materialize. Tests with smaller (batch, seq) slice this.
# Pinned so the cache shape is deterministic across machines.
# seq=128 was picked to be ≥ max(compress_ratios)=128 in the V4-Flash config,
# which is the minimum prompt length for decode steps to satisfy the
# Compressor's `start_pos + 1 - ratio >= 0` rope_idx constraint.
_CACHE_BATCH = 64
_CACHE_SEQ = 128

# A passage of natural English, long enough to tokenize to >= _CACHE_BATCH *
# _CACHE_SEQ tokens with the DeepSeek tokenizer. Keep it deterministic — the
# cache is invariant to environment as long as this string and the model
# weights don't change. Multi-topic on purpose so different rows of the
# reshaped [batch, seq] tensor see unrelated content.
_TEXT_SAMPLE = (
    # MoE / transformer architecture (model-relevant)
    "Mixture-of-Experts transformers route each token through a small subset "
    "of feed-forward sub-networks chosen by a learned gate. The advantage is "
    "that the model parameter count grows much faster than the per-token "
    "compute cost, so capacity scales without proportional latency. In "
    "practice this introduces challenges around load balancing across "
    "experts, communication patterns when experts are split over devices, "
    "and numerical sensitivity at the gate's top-k boundary. DeepSeek-V4 "
    "Flash uses 256 routed experts with six activated per token plus one "
    "shared expert that every token sees. The first three layers use a "
    "static hash routing table that maps token id directly to expert ids, "
    "bypassing the gate matmul; subsequent layers learn a score-based gate "
    "with a sqrt-softplus activation. Hyper-Connections maintain four "
    "residual streams that are reduced via a Sinkhorn step before each "
    "sub-layer and re-expanded after, replacing the usual single-stream "
    "residual. Multi-head Latent Attention compresses the key/value cache "
    "via a low-rank projection so the inference KV footprint stays small "
    "even with many heads. Together these choices let the same architecture "
    "scale from a few billion parameters to hundreds of billions while "
    "keeping a fixed token-level cost. Evaluating these designs on real "
    "hardware requires faithful inputs at every layer, since the gate's "
    "routing decisions depend on the activation distribution, and a random "
    "Gaussian input does not reproduce the load-imbalance and near-tie "
    "patterns observed when hidden states have flowed through several real "
    "attention and feed-forward sub-layers. "
    # History / general knowledge
    "The Roman Republic gave way to the Empire after a long period of civil "
    "war, with Augustus taking the title of princeps in twenty-seven before "
    "the common era. The institutions that survived from the Republic were "
    "gradually emptied of independent power even though their forms persisted "
    "for centuries. Provincial administration relied on a mix of local "
    "elites, Roman governors, and a standing army stationed along frontiers "
    "from Britain to the Euphrates. Trade networks moved Egyptian grain to "
    "Italy, Spanish silver across the Mediterranean, and Indian spices "
    "through the Red Sea. The eventual fragmentation of the western half of "
    "the Empire in the fifth century did not end Roman law, language, or "
    "religious institutions, all of which carried forward into successor "
    "kingdoms and then into the medieval European order. "
    # Cooking
    "A reliable approach to cooking dried beans starts with an overnight "
    "soak in salted water. The salt softens the seed coat enough that the "
    "beans cook evenly without bursting. After draining, the beans go into "
    "fresh water with aromatics — a halved onion, a few cloves of garlic, a "
    "bay leaf, and a piece of pork or a parmesan rind for depth. Bring to a "
    "gentle simmer, never a hard boil, and skim the foam in the first "
    "fifteen minutes. The cooking time depends on the variety and age of "
    "the beans, but most kinds finish in about an hour. Season with salt "
    "near the end, taste a few beans for doneness, and let them rest in "
    "their cooking liquid for at least ten minutes before serving. "
    # Programming / systems
    "When debugging a multi-process pipeline, the first instinct is often to "
    "add print statements, but it pays to set up structured logging early. "
    "A correlation identifier passed through every stage lets you reconstruct "
    "the path of a single request even when log lines are interleaved across "
    "many workers. Sampling expensive logs at a lower rate reduces overhead "
    "in the hot path. Tracing tools that span process and machine boundaries "
    "make it easier to attribute latency to specific stages. The hardest "
    "bugs come not from a single failing component but from interactions "
    "between components, where each part is doing something defensible in "
    "isolation but the combined behavior violates an invariant nobody wrote "
    "down. Writing the invariant down — even informally as a comment — is "
    "often most of the work; once you know what should be true, finding the "
    "place where it isn't true is comparatively quick. "
    # Travel / nature
    "The road north from the city follows a river valley for the first hour, "
    "then climbs through a series of switchbacks above the tree line. Wind "
    "moves across the high meadows in long visible waves, and the only "
    "sounds are sheep bells and the occasional vehicle. Past the summit the "
    "road descends into a wider valley dotted with stone-walled fields and "
    "a single church visible from a great distance. Late summer brings the "
    "first cold mornings and the first dustings of snow on the highest "
    "peaks. The villages along the route are connected to the capital by a "
    "single bus that runs in each direction once a day; outside that "
    "schedule, the only way through is on foot or by hitching a ride with a "
    "farmer driving home from market. Travelers who linger find that the "
    "rhythms of the place are set by the seasons rather than the clock, and "
    "that an afternoon spent watching clouds cross the ridge is its own "
    "form of progress. "
    # Science
    "A semiconductor device works because the periodic potential of the "
    "crystal lattice imposes an energy band structure on the electrons that "
    "fills it. The gap between the highest occupied band and the lowest "
    "unoccupied band determines whether the material conducts at room "
    "temperature, and dopants added at parts-per-million concentrations can "
    "shift the Fermi level into either band. Putting an n-type and a "
    "p-type region next to each other forms a junction whose built-in "
    "potential rectifies current. Stacking such junctions, sometimes with "
    "an insulating gate above a thin channel, gives the field-effect "
    "transistor that underlies all of modern digital logic. Each generation "
    "of process technology shrinks features by about one-and-a-half times, "
    "with corresponding improvements in switching energy and density, "
    "though the gains have been slowing as the channel approaches the size "
    "of individual atoms. "
    # Music
    "A simple twelve-bar blues moves through three chords in a fixed pattern "
    "and yet supports an enormous range of styles. Country, rock, jazz, and "
    "soul all draw on this skeleton, decorating it with different rhythms, "
    "instrumentation, and vocal phrasing. The pattern's strength is that it "
    "leaves space for improvisation while keeping a clear harmonic "
    "framework that listeners can follow without knowing the form by name. "
    "A guitarist starting out can play along with hundreds of recordings by "
    "knowing only the key and the basic rhythm, and a listener with no "
    "training can hum the next chord change before it arrives. "
    # Linguistics
    "Languages change continuously, and what appears as a sudden shift in "
    "the historical record is usually the result of many small changes "
    "accumulating over centuries. Phonological shifts often follow regular "
    "patterns: the Great Vowel Shift in English, the High German consonant "
    "shift, and Grimm's Law all describe systematic changes that affected "
    "whole classes of sounds rather than individual words. Vocabulary "
    "borrows freely from contact languages — English alone contains layers "
    "from Old Norse, Norman French, Latin, and dozens of other sources, "
    "each visible in the etymology of common words. Syntax tends to change "
    "more slowly than vocabulary or sound, but it does change, and the "
    "comparative method allows linguists to reconstruct features of "
    "languages with no surviving written records. "
    # Sports
    "A long-distance runner builds a season around a mix of easy aerobic "
    "miles, tempo runs at the threshold of comfortable breathing, and "
    "shorter intervals at speeds well above race pace. Most of the volume "
    "is easy on purpose; the hard sessions get their effect because they "
    "are bracketed by enough recovery to allow adaptation. Coaches argue "
    "endlessly about the right ratio of these elements, and the answers "
    "depend on the athlete, the event, and the time of year, but the "
    "underlying logic is consistent across approaches. Strength training, "
    "drills, and mobility work fill out the week and reduce injury risk "
    "without adding much fatigue. "
    # Mathematics
    "A continuous function on a closed interval attains its maximum and "
    "minimum somewhere on that interval. The proof is short once the right "
    "definitions are in place: the image of a closed interval under a "
    "continuous map is itself closed and bounded, so it contains its "
    "supremum and infimum. The same argument generalizes to any compact "
    "topological space, replacing closed-and-bounded with the more abstract "
    "notion of compactness. Many results in analysis can be reduced to "
    "compactness arguments, which is why first courses in topology spend "
    "so much time on it. "
    # Architecture
    "A traditional courtyard house in northern China is organized around a "
    "rectangular open space with rooms on each side, each of which has a "
    "specific role for the family that lives there. The northern building "
    "is the main hall, used by the eldest generation; eastern and western "
    "wings house younger members and storage; the southern side, opposite "
    "the main hall, often contains the entrance gate and ancillary rooms. "
    "The walls present a blank face to the street, so the inner courtyard "
    "is a private world entered through a single decorated gate. The form "
    "is a response to climate as much as social organization, with thick "
    "walls and small external openings to insulate against summer heat and "
    "winter cold. "
    # Astronomy
    "The cosmic microwave background is the most distant electromagnetic "
    "signal that can be observed, dating from the moment when the universe "
    "first cooled enough for neutral hydrogen to form and photons to travel "
    "freely. Its temperature today is just under three kelvins, and its "
    "spectrum is the most perfect blackbody ever measured. The tiny "
    "anisotropies in the temperature, at the level of one part in one "
    "hundred thousand, encode information about the geometry, composition, "
    "and history of the universe at a much earlier epoch than any direct "
    "observation could probe. Mapping these anisotropies has been the work "
    "of multiple generations of space telescopes, each pushing the angular "
    "resolution and sensitivity higher and revealing finer detail in the "
    "fluctuation spectrum. "
    # Biology
    "A cell's energy budget is dominated by the production of adenosine "
    "triphosphate in the mitochondrion, where electrons donated by reduced "
    "carriers move down a chain of membrane-bound complexes that pump "
    "protons across the inner membrane. The resulting electrochemical "
    "gradient drives a rotary enzyme that synthesizes ATP from ADP and "
    "inorganic phosphate. Each glucose molecule passing through glycolysis "
    "and the citric-acid cycle yields a small amount of ATP directly and a "
    "much larger amount through oxidative phosphorylation. The whole "
    "machinery is ancient, predating the split between plants, animals, "
    "and fungi, which is why mitochondria carry their own small genome and "
    "divide on a schedule loosely coupled to the host cell. "
    # Photography
    "A useful first lens for portraits is a fast prime in the fifty to "
    "eighty-five millimeter range on a full-frame body, opened up to "
    "isolate the subject from a distracting background. Soft window light "
    "from a north-facing window, with a white card opposite to fill the "
    "shadows, produces flattering results at almost no cost. The eye "
    "nearest the camera should be in sharp focus; the rest of the face "
    "can fall off gently into a shallower depth of field. Conversation "
    "during the shoot relaxes the subject more reliably than any technical "
    "trick, and the best frame is usually one taken between the planned "
    "exposures rather than during them. "
)


@dataclass
class _CacheEntry:
    version: int
    layer_id: int
    n_hash_layers: int
    vocab_size: int
    dim: int
    batch: int
    seq: int
    input_ids: torch.Tensor  # [batch, seq] int64
    hidden_states: torch.Tensor  # [batch, seq, dim] bf16


def _cache_path(layer_id: int) -> str:
    return os.path.join(_CACHE_DIR, f"layer{layer_id}_real_inputs.pt")


def _is_compatible(entry: dict, layer_id: int, args) -> bool:
    """Cache hit only if the saved entry was generated with the same model
    config. A mismatch (e.g. dim or vocab changed) forces regeneration."""
    return (
        entry.get("version") == _CACHE_VERSION
        and entry.get("layer_id") == layer_id
        and entry.get("n_hash_layers") == args.n_hash_layers
        and entry.get("vocab_size") == args.vocab_size
        and entry.get("dim") == args.dim
        and entry.get("batch") == _CACHE_BATCH
        and entry.get("seq") == _CACHE_SEQ
    )


def _tokenize_passage(args, batch: int, seq: int) -> torch.Tensor:
    """Tokenize `_TEXT_SAMPLE` with the model's tokenizer and reshape the
    flat token stream to `[batch, seq]`. Drops trailing tokens that don't
    fill the last row.

    If the passage is shorter than `batch * seq`, the token stream is
    looped (concatenated with itself) until it covers the requested
    shape, then truncated. Different rows still see different content
    because the slicing offset advances per row; the only artifact is
    that distant rows wrap back to passage-start material rather than
    seeing fresh text. This is fine for activation-distribution
    purposes (gate routing, attention pattern) which is what the
    cached input is used for."""
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    tok_path = hf_hub_download(repo_id=weight_loader.REPO_ID, filename="tokenizer.json")
    tok = Tokenizer.from_file(tok_path)
    encoding = tok.encode(_TEXT_SAMPLE)
    ids = torch.tensor(encoding.ids, dtype=torch.long)
    needed = batch * seq
    if ids.numel() < needed:
        reps = (needed + ids.numel() - 1) // ids.numel()
        ids = ids.repeat(reps)
    ids = ids[:needed].view(batch, seq)
    # Clip into valid vocab range (DeepSeek tokenizer should produce ids
    # within vocab_size, but guard against any out-of-range artifacts so the
    # embed lookup never indexes off the end).
    return ids.clamp_(min=0, max=args.vocab_size - 1)


def _build_cpu_prefix_model(args, ds_model, layer_id: int):
    """Build a Transformer with `n_layers = layer_id + 1` and load real
    weights for embed + layers 0..layer_id. The MTP and head paths aren't
    needed for capturing the FFN input, so we skip loading them — but the
    Transformer constructor still allocates them, so leave their random
    init alone (forward never visits them in our truncated pipeline).

    Dtype handling: the model relies on `torch.get_default_dtype()` for any
    parameter not constructed under an explicit `set_dtype(...)` context —
    notably `ParallelEmbedding.weight` and the per-layer Linear weights via
    `default_dtype`. The upstream `__main__` sets the default to bf16 before
    construction; we do the same here (and restore after) so the embed and
    Linear params come out bf16 to match the bf16 activations and HF
    checkpoint. Params with explicit fp32 dtype (RMSNorm.weight, hc_*_fn,
    attn_sink) are unaffected by the default. We also fix `kv_cache`, which
    `torch.zeros` creates as fp32, to bf16 to match what attention writes
    into it."""
    # Override n_layers to the minimum we actually run; keep n_hash_layers
    # at the real config value so layers 0..n_hash_layers-1 stay hash-routed.
    args.n_layers = layer_id + 1
    args.n_mtp_layers = 0  # skip MTP weights/build entirely
    # `kv_cache` is sized at construction as [max_batch_size, ...]; the config
    # default is tuned for inference and may be smaller than _CACHE_BATCH. Bump
    # it so the prefill slice `kv_cache[:bsz, :seqlen] = kv` fits.
    if args.max_batch_size < _CACHE_BATCH:
        args.max_batch_size = _CACHE_BATCH
    prev_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        model = ds_model.Transformer(args).eval()
    finally:
        torch.set_default_dtype(prev_default_dtype)

    # Load real weights for embed + each layer up to and including layer_id.
    embed_state = weight_loader.load_embed_state_dict()
    model.embed.load_state_dict(embed_state, strict=True)

    for li in range(layer_id + 1):
        block_state = weight_loader.load_block_state_dict(li)
        # Block state-dict from HF may not include freqs_cis (registered as
        # non-persistent buffer) — strict=False to allow the existing buffer
        # to remain. Same for attn.kv_cache.
        model.layers[li].load_state_dict(block_state, strict=False)

    # kv_cache (and any compressor caches) are non-persistent buffers
    # constructed via `torch.zeros(...)` without explicit dtype, so they
    # default to torch.float32 even though the surrounding activations are
    # bf16. Cast them in-place to match. Skip complex (freqs_cis) and
    # integer (tid2eid, indexer index buffers) — they're already correct.
    for _, b in model.named_buffers():
        if b.dtype == torch.float32:
            b.data = b.data.to(torch.bfloat16)

    return model


def _capture_pre_ffn(model, layer_id: int, input_ids: torch.Tensor) -> torch.Tensor:
    """Run the prefix forward up to (but not including) `layers[layer_id].ffn`
    and return the pre-FFN activation that would feed the MoE gate. Mirrors
    `Block.forward` for layers 0..layer_id-1 and the first half of
    `Block.forward` for layer_id."""
    layers = model.layers
    h = model.embed(input_ids)
    h = h.unsqueeze(2).repeat(1, 1, model.hc_mult, 1)

    # Full forward through layers preceding `layer_id` (typically the hash
    # MoE prefix). start_pos=0 — we're doing a single-shot prefill, no decode.
    for li in range(layer_id):
        h = layers[li](h, 0, input_ids)

    # Layer `layer_id`: run only up to ffn_norm. Replicates the first half
    # of Block.forward.
    block = layers[layer_id]
    residual = h
    h, post, comb = block.hc_pre(
        h, block.hc_attn_fn, block.hc_attn_scale, block.hc_attn_base
    )
    h = block.attn_norm(h)
    h = block.attn(h, 0)
    h = block.hc_post(h, residual, post, comb)
    residual = h
    h, post, comb = block.hc_pre(
        h, block.hc_ffn_fn, block.hc_ffn_scale, block.hc_ffn_base
    )
    h = block.ffn_norm(h)
    return h


def _generate_cache(layer_id: int) -> _CacheEntry:
    """Heavy first-time path: download HF weights, build CPU model, run
    prefix forward, capture pre-FFN activation. Saves the .pt cache.

    Uses `modified_model` (TT-friendly variant): pure-torch hc/sparse_attn
    kernels live in its own `.kernel` package, and Attention.forward has the
    QAT-simulation calls + Hadamard rotation stripped, so the CPU prefix
    runs with no CUDA-only dependencies and no monkey-patching."""
    from third_party.tt_forge_models.deepseek_v4.modified_model import model as ds_model

    args = weight_loader.load_config_args()
    input_ids = _tokenize_passage(args, _CACHE_BATCH, _CACHE_SEQ)

    model = _build_cpu_prefix_model(args, ds_model, layer_id)
    with torch.no_grad():
        hidden_states = _capture_pre_ffn(model, layer_id, input_ids)
    hidden_states = hidden_states.to(torch.bfloat16).contiguous()

    entry = _CacheEntry(
        version=_CACHE_VERSION,
        layer_id=layer_id,
        n_hash_layers=args.n_hash_layers,
        vocab_size=args.vocab_size,
        dim=args.dim,
        batch=_CACHE_BATCH,
        seq=_CACHE_SEQ,
        input_ids=input_ids.contiguous(),
        hidden_states=hidden_states,
    )

    os.makedirs(_CACHE_DIR, exist_ok=True)
    torch.save(entry.__dict__, _cache_path(layer_id))
    return entry


def _load_or_generate(layer_id: int) -> _CacheEntry:
    args = weight_loader.load_config_args()
    path = _cache_path(layer_id)
    if os.path.exists(path):
        raw = torch.load(path, map_location="cpu", weights_only=False)
        if _is_compatible(raw, layer_id, args):
            return _CacheEntry(**raw)
    return _generate_cache(layer_id)


def get_realistic_inputs(
    layer_id: int, batch_size: int, seq_len: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Public API: return (input_ids[batch_size, seq_len],
    hidden_states[batch_size, seq_len, dim]) sliced from the cached layer
    `layer_id` snapshot. Generates the cache on first call.

    The cache is sized once at `_CACHE_BATCH x _CACHE_SEQ`; smaller test
    parametrizations get the leading slice. Larger asks fail loudly so we
    never silently feed a tiled / repeated tensor."""
    if batch_size > _CACHE_BATCH or seq_len > _CACHE_SEQ:
        raise ValueError(
            f"Requested ({batch_size}, {seq_len}) exceeds cached "
            f"({_CACHE_BATCH}, {_CACHE_SEQ}). Bump _CACHE_BATCH/_CACHE_SEQ "
            f"and regenerate {_cache_path(layer_id)}."
        )
    entry = _load_or_generate(layer_id)
    return (
        entry.input_ids[:batch_size, :seq_len].clone(),
        entry.hidden_states[:batch_size, :seq_len].clone(),
    )
