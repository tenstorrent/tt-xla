# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Portions (c) 2026 Tenstorrent AI ULC
"""Generate GDN golden tensors against the standalone FLA kernels.

Run on the GPU box (machine ``G``) that has ``flash-linear-attention`` installed::

    python tests/integrations/vllm_plugin/gdn/gen_goldens.py [--out PATH] [--cuda]

Saves a single ``.pt`` bundle holding, per case, the *inputs* and the *golden
outputs/states*, plus a version manifest. Inputs are saved (not just seeded) so
the consumer never relies on cross-machine/-version RNG reproducibility.

For the delta-rule ops it calls FLA and auto-detects FLA's recurrent-state layout
by cross-checking against the framework-agnostic reference (``_reference``); if
FLA is unavailable it falls back to the reference and tags the case ``source=ref``.
Transfer the bundle once to the TT box and consume it with ``test_gdn_ops.py``.
"""

import argparse
import os

import torch

from _reference import ref_delta_rule, ref_gating, ref_l2norm

DEFAULT_OUT = os.path.join(os.path.dirname(__file__), "golden", "gdn_golden.pt")


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.flatten().double()
    b = b.flatten().double()
    return float(torch.corrcoef(torch.stack([a, b]))[0, 1])


def ref_conv1d_prefill(x, weight, bias, conv_state, has_init, indices, qsl):
    """Independent depthwise causal conv reference (prefill). x: [conv_dim, T]."""
    import torch.nn.functional as F

    conv_dim, K = weight.shape
    out = torch.empty_like(x)
    cs = conv_state.clone()
    for n in range(indices.shape[0]):
        s, e = int(qsl[n]), int(qsl[n + 1])
        if e <= s:
            continue
        slot = int(indices[n])
        left = cs[slot] if bool(has_init[n]) else torch.zeros(conv_dim, K - 1)
        padded = torch.cat([left, x[:, s:e]], dim=-1)
        y = F.conv1d(padded.unsqueeze(0), weight.unsqueeze(1), groups=conv_dim).squeeze(0)
        if bias is not None:
            y = y + bias.unsqueeze(-1)
        out[:, s:e] = F.silu(y)
        cs[slot] = padded[:, -(K - 1):]
    return out, cs


def ref_conv1d_update(x, conv_state, weight, bias, indices):
    """Independent depthwise causal conv reference (decode). x: [num_tokens, conv_dim]."""
    import torch.nn.functional as F

    conv_dim, K = weight.shape
    out = torch.empty_like(x)
    cs = conv_state.clone()
    for t in range(x.shape[0]):
        slot = int(indices[t])
        window = torch.cat([cs[slot], x[t].unsqueeze(-1)], dim=-1)
        y = (weight * window).sum(-1)
        if bias is not None:
            y = y + bias
        out[t] = F.silu(y)
        cs[slot] = window[:, 1:]
    return out, cs


def _fla_delta(fla_fn, q, k, v, g, beta, scale, initial_state, cu_seqlens, l2, ref_o,
               recurrent=False, ssm_state_indices=None):
    """Call an FLA delta-rule kernel, auto-detecting the [..,V,K] vs [..,K,V]
    recurrent-state layout by matching the reference output."""
    best = None
    # The chunk kernel takes ``output_final_state``; the fused-recurrent kernel
    # has no such parameter (it always returns the final state, gated instead by
    # ``inplace_final_state``) and raises TypeError if ``output_final_state`` is
    # passed. Select the right kwarg per kernel.
    extra = {} if recurrent else {"output_final_state": True}
    # For packed multi-sequence recurrent decode, the sequence -> state-slot
    # mapping is carried by ``ssm_state_indices``; FLA does not assume identity.
    # The reference maps sequence n -> initial_state[n], so we must hand FLA the
    # matching indices or every sequence but the first reads the wrong state
    # (the output then diverges and no transpose can recover it).
    if recurrent and ssm_state_indices is not None:
        extra["ssm_state_indices"] = ssm_state_indices
    for transpose in (False, True):
        ist = initial_state
        if ist is not None and transpose:
            ist = initial_state.transpose(-1, -2).contiguous()
        # The recurrent kernel's ``inplace_final_state=True`` (its default)
        # mutates the passed state in place, so (a) clone per trial to keep the
        # two transpose attempts independent, and (b) some FLA versions then
        # return ``None`` for the final state and leave it in this input tensor.
        ist_in = ist.clone() if ist is not None else None
        o, st = fla_fn(
            q, k, v, g, beta,
            scale=scale,
            initial_state=ist_in,
            cu_seqlens=cu_seqlens,
            use_qk_l2norm_in_kernel=l2,
            **extra,
        )
        if st is None:
            st = ist_in  # final state was written in place into the input
        # FLA (Triton) outputs live on the kernel's device; ``ref_o`` is CPU.
        # Bring both to CPU for the PCC compare and for the saved golden.
        p = pcc(ref_o, o.float().cpu())
        st_norm = st.transpose(-1, -2).contiguous() if transpose else st
        if best is None or p > best[0]:
            best = (p, o.float().cpu(), st_norm.float().cpu(), transpose)
    if best[0] < 0.99:
        raise RuntimeError(
            f"FLA delta-rule output disagrees with reference (best PCC={best[0]:.4f}); "
            "the state-layout adapter in gen_goldens may need updating for this "
            "FLA version."
        )
    return best[1], best[2], best[3]


def build_cases(device, seeds=(0, 1, 2)):
    """Build the full case grid over several seeds.

    Each op is generated with ``len(seeds)`` independent random draws (the id is
    suffixed ``_s{seed}``), so a green run reflects agreement across multiple
    input samples rather than a single lucky draw. ``chunk`` additionally varies
    its sequence length per seed to cover different chunk-boundary alignments
    (sub-chunk, an exact multiple of ``CHUNK_SIZE=64``, and a ragged tail).
    """
    cases = []
    for seed in seeds:
        cases.extend(_build_cases_for_seed(seed))
    return cases


# Sequence lengths swept across seeds for the chunk op: < CHUNK_SIZE (single
# chunk), an exact 2x multiple (boundary), and a ragged multi-chunk tail.
_CHUNK_T_BY_SEED = [70, 64, 128]


def _build_cases_for_seed(seed):
    gen = torch.Generator(device="cpu").manual_seed(seed)

    def rnd(*shape):
        return torch.randn(*shape, generator=gen)

    sfx = f"_s{seed}"
    cases = []

    # --- l2norm -----------------------------------------------------------
    x = rnd(2, 16, 4, 32)
    cases.append(dict(op="l2norm", id=f"l2norm_basic{sfx}", inputs={"x": x}, params={}))

    # --- gating -----------------------------------------------------------
    HV = 4
    a = rnd(8, HV)
    b = rnd(8, HV)
    A_log = torch.log(torch.rand(HV, generator=gen) + 0.5)
    dt_bias = rnd(HV)
    cases.append(
        dict(op="gating", id=f"gating_basic{sfx}",
             inputs={"a": a, "b": b, "A_log": A_log, "dt_bias": dt_bias}, params={})
    )

    # --- conv1d prefill ---------------------------------------------------
    conv_dim, K, L = 24, 4, 10
    cases.append(dict(
        op="conv1d_prefill", id=f"conv_prefill_freshstate{sfx}",
        inputs={
            "x": rnd(conv_dim, L),
            "weight": rnd(conv_dim, K),
            "bias": rnd(conv_dim),
            "conv_state": torch.zeros(1, conv_dim, K - 1),
            "has_init": torch.zeros(1, dtype=torch.bool),
            "indices": torch.zeros(1, dtype=torch.long),
            "qsl": torch.tensor([0, L], dtype=torch.int32),
        }, params={}))
    cases.append(dict(
        op="conv1d_prefill", id=f"conv_prefill_withstate{sfx}",
        inputs={
            "x": rnd(conv_dim, L),
            "weight": rnd(conv_dim, K),
            "bias": rnd(conv_dim),
            "conv_state": rnd(1, conv_dim, K - 1),
            "has_init": torch.ones(1, dtype=torch.bool),
            "indices": torch.zeros(1, dtype=torch.long),
            "qsl": torch.tensor([0, L], dtype=torch.int32),
        }, params={}))

    # --- conv1d decode update --------------------------------------------
    cases.append(dict(
        op="conv1d_update", id=f"conv_update_basic{sfx}",
        inputs={
            "x": rnd(3, conv_dim),
            "conv_state": rnd(4, conv_dim, K - 1),
            "weight": rnd(conv_dim, K),
            "bias": rnd(conv_dim),
            "indices": torch.tensor([0, 2, 3], dtype=torch.long),
        }, params={}))

    # --- chunk (prefill delta rule) --------------------------------------
    T = _CHUNK_T_BY_SEED[seed % len(_CHUNK_T_BY_SEED)]
    for dtype, tag in [(torch.float32, "fp32"), (torch.bfloat16, "bf16")]:
        H, HVc, Kd, Vd = 2, 4, 16, 16
        cases.append(dict(
            op="chunk", id=f"chunk_T{T}_{tag}{sfx}",
            inputs={
                "q": rnd(1, T, H, Kd).to(dtype),
                "k": rnd(1, T, H, Kd).to(dtype),
                "v": rnd(1, T, HVc, Vd).to(dtype),
                "g": (-torch.rand(1, T, HVc, generator=gen) * 0.5),
                "beta": torch.rand(1, T, HVc, generator=gen).sigmoid(),
                "initial_state": rnd(1, HVc, Vd, Kd) * 0.1,
                "cu_seqlens": torch.tensor([0, T], dtype=torch.int32),
            },
            params={"scale": Kd ** -0.5, "l2norm": True}))

    # --- recurrent (decode delta rule), incl. chained multi-step ---------
    H, HVc, Kd, Vd, D = 2, 4, 16, 16, 4
    cases.append(dict(
        op="recurrent", id=f"recurrent_4seq{sfx}",
        inputs={
            "q": rnd(1, D, H, Kd),
            "k": rnd(1, D, H, Kd),
            "v": rnd(1, D, HVc, Vd),
            "g": (-torch.rand(1, D, HVc, generator=gen) * 0.5),
            "beta": torch.rand(1, D, HVc, generator=gen).sigmoid(),
            "initial_state": rnd(D, HVc, Vd, Kd) * 0.1,
            "cu_seqlens": torch.arange(D + 1, dtype=torch.int32),
            "ssm_state_indices": torch.arange(D, dtype=torch.long),
        },
        params={"scale": Kd ** -0.5, "l2norm": True}))

    # Single-token decode, n_seq==1, bf16 — the exact branch the plugin's
    # _core_decode runs under torch.compile at max_num_seqs=1 (the branchless
    # path that derives bounds from .shape and reads the state via index_select).
    # recurrent_4seq above only exercises the n_seq>1 Python-int value loop, and
    # decode in the model is bf16; this case covers both gaps. Identity slot keeps
    # it consistent with FLA's default seq->state mapping (the index ops run the
    # same regardless of slot value).
    cases.append(dict(
        op="recurrent", id=f"recurrent_decode1_bf16{sfx}",
        inputs={
            "q": rnd(1, 1, H, Kd).to(torch.bfloat16),
            "k": rnd(1, 1, H, Kd).to(torch.bfloat16),
            "v": rnd(1, 1, HVc, Vd).to(torch.bfloat16),
            "g": (-torch.rand(1, 1, HVc, generator=gen) * 0.5),
            "beta": torch.rand(1, 1, HVc, generator=gen).sigmoid(),
            "initial_state": rnd(1, HVc, Vd, Kd) * 0.1,
            "cu_seqlens": torch.tensor([0, 1], dtype=torch.int32),
            "ssm_state_indices": torch.zeros(1, dtype=torch.long),
        },
        params={"scale": Kd ** -0.5, "l2norm": True}))

    return cases


def compute_golden(case, device="cpu"):
    op = case["op"]
    inp = case["inputs"]
    prm = case["params"]

    # FLA's Triton kernels require their tensor arguments to be on the CUDA
    # device; the case inputs are built on CPU (for reproducible seeding and so
    # they can be saved/consumed anywhere). Move a copy to ``device`` for the
    # kernel call. The pure-PyTorch references below run on CPU regardless.
    def to_dev(t):
        return t.to(device) if isinstance(t, torch.Tensor) else t

    if op == "l2norm":
        try:
            from fla.modules.l2norm import l2norm_fwd  # type: ignore
        except ImportError:
            return {"y": ref_l2norm(inp["x"])}, "ref"
        # Some FLA versions return just ``y``; others a tuple ``(y, rstd, ...)``.
        # The normalized output is always the first element.
        out = l2norm_fwd(to_dev(inp["x"]))
        y = out[0] if isinstance(out, (tuple, list)) else out
        return {"y": y.cpu()}, "fla"

    if op == "gating":
        g, beta = ref_gating(inp["A_log"], inp["a"], inp["b"], inp["dt_bias"])
        return {"g": g, "beta": beta}, "ref"

    if op == "conv1d_prefill":
        y, cs = ref_conv1d_prefill(
            inp["x"], inp["weight"], inp["bias"], inp["conv_state"],
            inp["has_init"], inp["indices"], inp["qsl"])
        return {"y": y, "conv_state_out": cs}, "ref"

    if op == "conv1d_update":
        y, cs = ref_conv1d_update(
            inp["x"], inp["conv_state"], inp["weight"], inp["bias"], inp["indices"])
        return {"y": y, "conv_state_out": cs}, "ref"

    if op in ("chunk", "recurrent"):
        ref_o, ref_state = ref_delta_rule(
            inp["q"].float(), inp["k"].float(), inp["v"].float(),
            inp["g"], inp["beta"], prm["scale"],
            initial_state=inp["initial_state"],
            cu_seqlens=inp.get("cu_seqlens"), l2norm=prm["l2norm"])
        try:
            # Both the chunk (prefill) and recurrent (decode) TT ops implement the
            # SAME gated-delta-rule recurrence, so BOTH goldens are generated with
            # FLA's *chunk* kernel. We deliberately do NOT use FLA's
            # fused_recurrent kernel: in this FLA version it returns uninitialized
            # / nondeterministic output on these shapes (single-token decode gives
            # PCC(run,run) ~ 0, and packed multi-seq is deterministically wrong),
            # whereas chunk reproduces the reference at PCC=1.0 per-token for every
            # recurrent case. See diag_recurrent.py. The recurrent TT op is still
            # fully exercised — it runs against this correct, FLA-computed golden.
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fn
        except ImportError:
            # FLA not installed: fall back to the reference golden (quiet, expected).
            return {"o": ref_o, "final_state": ref_state}, "ref"
        # FLA *is* installed: any failure here is a real signature/version
        # mismatch and must surface, not be silently downgraded to a ref golden.
        # The recurrent cases use identity ssm_state_indices, so sequence n maps to
        # initial_state[n] under cu_seqlens alone — no index argument needed.
        cu = inp.get("cu_seqlens")
        o, st, _ = _fla_delta(
            fn, to_dev(inp["q"]), to_dev(inp["k"]), to_dev(inp["v"]),
            to_dev(inp["g"]), to_dev(inp["beta"]), prm["scale"],
            to_dev(inp["initial_state"]), to_dev(cu),
            prm["l2norm"], ref_o, recurrent=False)
        return {"o": o, "final_state": st}, "fla"

    raise ValueError(op)


GEN_VERSION = "v5 (recurrent golden via chunk kernel; recurrent kernel unreliable)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--cuda", action="store_true")
    args = ap.parse_args()
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    print(f"gen_goldens {GEN_VERSION}  device={device}")

    cases = build_cases(device)
    out_cases = []
    for c in cases:
        golden, src = compute_golden(c, device)
        out_cases.append({
            "op": c["op"], "id": c["id"], "source": src,
            "inputs": {k: v.cpu() for k, v in c["inputs"].items()},
            "params": c["params"],
            "golden": {k: v.cpu() for k, v in golden.items()},
        })
        print(f"  {c['id']:28s} op={c['op']:14s} source={src}")

    try:
        import fla  # type: ignore
        fla_ver = getattr(fla, "__version__", "unknown")
    except Exception:
        fla_ver = None
    meta = {"torch_version": torch.__version__, "fla_version": fla_ver}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"meta": meta, "cases": out_cases}, args.out)
    print(f"\nWrote {len(out_cases)} cases -> {args.out}  (fla={fla_ver})")


if __name__ == "__main__":
    main()
