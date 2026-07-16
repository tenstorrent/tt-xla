# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Draw the FIBO transformer (``BriaFiboTransformer2DModel``, briaai/FIBO) as a
*graph* instead of the flat module listing that ``print(model)`` produces.

FIBO is a FLUX-style MMDiT: a stack of double-stream (joint image+text) blocks
followed by single-stream blocks, driven by a SmolLM3 text encoder.  Its
signature feature is *per-block caption injection*: ``caption_projection`` is a
list of one Linear per block, and each block refreshes the second half of its
text stream with a fresh SmolLM3 layer's hidden state.

Three graphs are produced (each as ASCII + inline SVG):
  1. Top-level data flow (image / text / per-block captions / timestep).
  2. One double-stream ``BriaFiboTransformerBlock`` (two lanes, joint attention).
  3. One single-stream ``BriaFiboSingleTransformerBlock`` (parallel attn + MLP).

Outputs (both on by default):
  * ``fibo_arch.html`` - self-contained, theme-aware page with all three graphs
    as inline SVG (no network); open it in a browser.
  * ``fibo_arch.mmd``  - Mermaid graph for GitHub / mermaid.live.

All numbers are read live from the loaded model.  The generic drawing helpers
(canvas, SVG nodes, CSS) are shared with ``draw_arch.py``.

Usage:
  python draw_arch_fibo.py                 # print ASCII, write .html + .mmd
  python draw_arch_fibo.py --no-html       # ASCII only
  python draw_arch_fibo.py --no-mermaid    # skip Mermaid
  python draw_arch_fibo.py --self-test     # baked-in dims (no model load)
"""

from __future__ import annotations

import argparse

from draw_arch import (
    Canvas, _boxw, _stack,
    _svg_esc, _svg_node, _svg_wrap,
    _HTML_CSS, _html_chip,
)

MODEL_ID = "briaai/FIBO"


# --------------------------------------------------------------------------- #
# Small ASCII helpers on top of the shared Canvas.                             #
# --------------------------------------------------------------------------- #
def _abox(cv, cx, top, lines):
    """Place a box centered on column ``cx`` with its top at row ``top``."""
    return cv.box(top, cx - _boxw(lines) // 2, lines)


def _adown(cv, a, b, arrow="▼"):
    """Vertical connector between two boxes sharing a center column."""
    x = a["cx"]
    cv.vline(x, a["S"][0] + 1, b["N"][0] - 1)
    cv.put(b["N"][0] - 1, x, arrow)


def _merge(cv, left, right, target, cxS):
    """Route two lanes (left, right) into ``target`` centered at column cxS."""
    cxL, cxR = left["cx"], right["cx"]
    rj = max(left["S"][0], right["S"][0]) + 2
    cv.vline(cxL, left["S"][0] + 1, rj)
    cv.vline(cxR, right["S"][0] + 1, rj)
    cv.put(rj, cxL, "└")
    cv.put(rj, cxR, "┘")
    cv.hline(rj, cxL + 1, cxR - 1)
    cv.put(rj, cxS, "┬")
    cv.vline(cxS, rj + 1, target["N"][0] - 1)
    cv.put(target["N"][0] - 1, cxS, "▼")


def _res_rail(cv, railx, src_port, dst_port, enter):
    """Residual skip: tap just below src, out to railx, down, into dst from side."""
    sr = src_port["S"][0] + 1
    cx = src_port["cx"]
    cv.put(sr, cx, "├" if railx > cx else "┤")
    cv.hline(sr, min(cx, railx) + 1, max(cx, railx) - 1)
    cv.put(sr, railx, "┐" if railx > cx else "┌")
    dr = dst_port[enter][0]
    cv.vline(railx, sr + 1, dr)
    cv.put(dr, railx, "┘" if railx > cx else "└")
    dx = dst_port[enter][1]
    cv.hline(dr, min(dx, railx) + 1, max(dx, railx) - 1)
    cv.put(dr, dx + (1 if enter == "W" else -1), "►" if enter == "W" else "◄")


def _bus_feed(cv, src, busx, targets, side):
    """Fan one source box out to several targets via a vertical bus on one side.
    side 'W' : bus sits left of the targets, arrows enter their west edge (►).
    side 'E' : bus sits right of the targets, arrows enter their east edge (◄)."""
    sx = src["cx"]
    tr = src["S"][0] + 2
    cv.vline(sx, src["S"][0] + 1, tr)
    lo, hi = sorted((sx, busx))
    cv.hline(tr, lo, hi)
    cv.put(tr, sx, "┘" if busx < sx else "└")
    cv.put(tr, busx, "┌" if busx < sx else "┐")
    rows = [t[side][0] for t in targets]
    last = max(rows)
    cv.vline(busx, tr + 1, last)
    for t in targets:
        er, ex = t[side]
        if side == "W":
            cv.hline(er, busx + 1, ex - 1)
            cv.put(er, busx, "└" if er == last else "├")
            cv.put(er, ex + 1, "►")
        else:
            cv.hline(er, ex + 1, busx - 1)
            cv.put(er, busx, "┘" if er == last else "┤")
            cv.put(er, ex - 1, "◄")


# --------------------------------------------------------------------------- #
# 1. Top-level data-flow graph (ASCII).                                        #
# --------------------------------------------------------------------------- #
def draw_pipeline(A):
    cv = Canvas()
    dim, inc = A["dim"], A["in_channels"]
    # timestep on the left, captions on the right, so both conditioning buses
    # feed the block stack from their own side without crossing the data lanes.
    cxTime, cxImg, cxCtx, cxCap = 12, 35, 66, 99
    cxS = (cxImg + cxCtx) // 2
    C = (cxTime + cxCap) // 2

    title = f"{A['name']}   ·   {A['params_total']:.2f}B params   ·   FLUX-style MMDiT"
    sub = (f"dim={dim} · {A['n_heads']} heads × {A['head_dim']} · {A['n_double']} double "
           f"+ {A['n_single']} single blocks · RoPE {A['axes']}")
    cv.text(0, C - len(title) // 2, title)
    cv.text(1, C - len(sub) // 2, sub)
    note = (f"VAE: AutoencoderKLWan (Wan-2.2, {inc}-ch latent · patch_size=1)   ·   "
            f"text encoder: SmolLM3 ({A['text_enc_dim']}-d, 36 layers)")
    cv.text(2, C - len(note) // 2, note)

    s = 4
    tim = _stack(cv, cxTime, s, [
        ["t : timestep", "(scalar)"],
        ["time_embed", "sinusoid[256] →", f"temb [B, {dim}]"],
    ])
    img = _stack(cv, cxImg, s, [
        ["x : image latent", f"[B, N_img, {inc}]"],
        ["x_embedder", f"Linear {inc}→{dim}", f"[B, N_img, {dim}]"],
    ])
    ctx = _stack(cv, cxCtx, s, [
        ["encoder_hidden (text)", f"[B, N_txt, {A['ctx_dim']}]"],
        ["context_embedder", f"Linear {A['ctx_dim']}→{dim}", f"[B, N_txt, {dim}]"],
    ])
    cap = _stack(cv, cxCap, s, [
        ["text_encoder_layers", f"{A['n_cap']}× [B, N_txt, {A['text_enc_dim']}]"],
        ["caption_projection", f"{A['n_cap']}× Linear {A['text_enc_dim']}→{A['cap_out']}",
         f"[B, N_txt, {A['cap_out']}]"],
    ])

    spine = _stack(cv, cxS, max(img[-1]["S"][0], ctx[-1]["S"][0]) + 5, [
        ["transformer_blocks", f"{A['n_double']}× BriaFiboTransformerBlock",
         f"img [B,N_img,{dim}]  ⇄  txt [B,N_txt,{dim}]"],
        ["single_transformer_blocks", f"{A['n_single']}× BriaFiboSingleTransformerBlock",
         f"seq [B, N_txt+N_img, {dim}]"],
        ["norm_out", "AdaLayerNormContinuous", f"[B, N_img, {dim}]"],
        ["proj_out", f"Linear {dim}→{inc}", f"[B, N_img, {inc}]"],
        ["output", f"[B, N_img, {inc}]", "→ Wan-2.2 VAE (48-ch)"],
    ])
    _merge(cv, img[-1], ctx[-1], spine[0], cxS)

    # conditioning buses: timestep (adaLN) feeds every block + norm_out from the
    # left; per-block caption injection feeds both block stacks from the right.
    t_targets = [spine[0], spine[1], spine[2]]
    c_targets = [spine[0], spine[1]]
    time_bus = min(t["W"][1] for t in t_targets) - 4
    cap_bus = max(t["E"][1] for t in c_targets) + 4
    _bus_feed(cv, tim[-1], time_bus, t_targets, "W")
    _bus_feed(cv, cap[-1], cap_bus, c_targets, "E")

    lr = spine[-1]["S"][0] + 2
    cv.text(lr, cxTime, "► left bus  = timestep temb → adaLN conditioning in every block + norm_out")
    cv.text(lr + 1, cxTime, "◄ right bus = per-block SmolLM3 caption injection (block i ← caption_proj[i]) — FIBO's signature")
    return cv.render()


# --------------------------------------------------------------------------- #
# 2. Double-stream block (ASCII).                                              #
# --------------------------------------------------------------------------- #
def draw_double_block(A):
    cv = Canvas()
    dim, hid = A["dim"], A["ff_hidden"]
    cxIr, cxI, cxM, cxT, cxTr = 6, 24, 55, 86, 104

    cv.text(0, 2, f"BriaFiboTransformerBlock   (×{A['n_double']} in transformer_blocks)   —  MMDiT double stream")
    cv.text(1, 2, "temb ◄c → AdaLayerNormZero → 6 params/stream: shift_msa, scale_msa (applied in norm1) · gate_msa · shift_mlp, scale_mlp, gate_mlp")

    # FIBO per-block caption injection rebuilds the text stream *before* the block.
    y = 4
    capi = _abox(cv, cxT, y, ["caption_proj[i]", f"SmolLM3 layer · {A['text_enc_dim']}→{A['cap_out']}"])
    y = capi["S"][0] + 2
    img_in = _abox(cv, cxI, y, ["hidden_states  (image)", "from prev block", f"[B, N_img, {dim}]"])
    txt_in = _abox(cv, cxT, y, ["encoder_hidden_states (text)", f"concat[ ctx[:{A['cap_out']}] ; cap[i] ]", f"[B, N_txt, {dim}]"])
    _adown(cv, capi, txt_in)
    y = max(img_in["S"][0], txt_in["S"][0]) + 2
    n1 = _abox(cv, cxI, y, ["norm1  (AdaLN-Zero)", "×(1+scale_msa)+shift_msa"])
    n1c = _abox(cv, cxT, y, ["norm1_context  (AdaLN-Zero)", "×(1+c_scale_msa)+c_shift_msa"])
    _adown(cv, img_in, n1)
    _adown(cv, txt_in, n1c)

    y = max(n1["S"][0], n1c["S"][0]) + 2
    attn = _abox(cv, cxM, y, [
        "BriaFiboAttention  —  JOINT self-attention",
        f"image q,k,v (to_q/k/v)  +  text q,k,v (add_q/k/v_proj)   Linear {dim}→{dim}",
        f"reshape → [B, {A['n_heads']}, N_txt+N_img, {A['head_dim']}] · RMSNorm(q,k) · RoPE {A['axes']}",
        f"concat[text,image] · SDPA · split → to_out (image) · to_add_out (text)",
    ])
    # two inbound (norm1 image/text -> attention top), two outbound (-> adds)
    for x, src in ((cxI, n1), (cxT, n1c)):
        cv.vline(x, src["S"][0] + 1, attn["N"][0] - 1)
        cv.put(attn["N"][0] - 1, x, "▼")

    y = attn["S"][0] + 2
    add1 = _abox(cv, cxI, y, ["⊕  hidden += gate_msa · attn"])
    add1c = _abox(cv, cxT, y, ["⊕  enc += c_gate_msa · attn"])
    for x, dst in ((cxI, add1), (cxT, add1c)):
        cv.vline(x, attn["S"][0] + 1, dst["N"][0] - 1)
        cv.put(dst["N"][0] - 1, x, "▼")

    y = max(add1["S"][0], add1c["S"][0]) + 2
    n2 = _abox(cv, cxI, y, ["norm2  (LayerNorm)", "×(1+scale_mlp)+shift_mlp"])
    n2c = _abox(cv, cxT, y, ["norm2_context  (LayerNorm)", "×(1+c_scale_mlp)+c_shift_mlp"])
    _adown(cv, add1, n2)
    _adown(cv, add1c, n2c)

    y = max(n2["S"][0], n2c["S"][0]) + 2
    ff = _abox(cv, cxI, y, ["ff  (GELU-approx MLP)", f"{dim} → {hid} → {dim}"])
    ffc = _abox(cv, cxT, y, ["ff_context  (GELU MLP)", f"{dim} → {hid} → {dim}"])
    _adown(cv, n2, ff)
    _adown(cv, n2c, ffc)

    y = max(ff["S"][0], ffc["S"][0]) + 2
    add2 = _abox(cv, cxI, y, ["⊕  hidden += gate_mlp · ff"])
    add2c = _abox(cv, cxT, y, ["⊕  enc += c_gate_mlp · ff"])
    _adown(cv, ff, add2)
    _adown(cv, ffc, add2c)

    y = max(add2["S"][0], add2c["S"][0]) + 2
    out_i = _abox(cv, cxI, y, ["hidden_states →", f"[B, N_img, {dim}]"])
    out_t = _abox(cv, cxT, y, ["encoder_hidden_states →", f"[B, N_txt, {dim}]"])
    _adown(cv, add2, out_i)
    _adown(cv, add2c, out_t)

    # residual skips (image on the left rail, text on the right rail)
    _res_rail(cv, cxIr, img_in, add1, "W")
    _res_rail(cv, cxIr, add1, add2, "W")
    _res_rail(cv, cxTr, txt_in, add1c, "E")
    _res_rail(cv, cxTr, add1c, add2c, "E")
    return cv.render()


# --------------------------------------------------------------------------- #
# 3. Single-stream block (ASCII).                                              #
# --------------------------------------------------------------------------- #
def draw_single_block(A):
    cv = Canvas()
    dim = A["dim"]
    mlp, pin = A["single_mlp"], A["single_proj_in"]
    cxA, cxS, cxB, cxR = 22, 44, 68, 92

    cv.text(0, 2, f"BriaFiboSingleTransformerBlock   (×{A['n_single']} in single_transformer_blocks)  —  FLUX single stream")
    cv.text(1, 2, "temb ◄c → AdaLayerNormZeroSingle → 3 params: shift, scale (applied in norm) · gate")

    # FIBO per-block caption injection refreshes the text half before the seq-concat.
    y = 3
    capi = _abox(cv, cxS, y, [f"caption_proj[i] · SmolLM3 {A['text_enc_dim']}→{A['cap_out']}",
                              f"text ← concat[ ctx[:{A['cap_out']}] ; cap[i] ]"])
    y = capi["S"][0] + 2
    h_in = _abox(cv, cxS, y, ["hidden_states = concat [ text ; image ]", f"[B, N_txt+N_img, {dim}]"])
    _adown(cv, capi, h_in)
    y = h_in["S"][0] + 2
    norm = _abox(cv, cxS, y, ["norm  (AdaLN-Zero-Single)", "×(1+scale)+shift  → gate"])
    _adown(cv, h_in, norm)

    # diverge to parallel attention + MLP
    y = norm["S"][0] + 3
    attn = _abox(cv, cxA, y, ["attn  (pre_only)", f"RMSNorm(q,k) · RoPE · SDPA", f"[B, {A['n_heads']}, N, {A['head_dim']}] → {dim}"])
    mlpb = _abox(cv, cxB, y, ["proj_mlp", f"Linear {dim}→{mlp}", "GELU (tanh)"])
    jr = norm["S"][0] + 1
    cv.put(jr, cxS, "┬")
    cv.hline(jr, cxA, cxB)
    cv.put(jr, cxA, "┌")
    cv.put(jr, cxB, "┐")
    cv.vline(cxA, jr + 1, attn["N"][0] - 1)
    cv.put(attn["N"][0] - 1, cxA, "▼")
    cv.vline(cxB, jr + 1, mlpb["N"][0] - 1)
    cv.put(mlpb["N"][0] - 1, cxB, "▼")

    # converge to concat
    y = max(attn["S"][0], mlpb["S"][0]) + 2
    cat = _abox(cv, cxS, y, [f"concat [ attn ; mlp ]", f"dim = {dim}+{mlp} = {pin}"])
    kr = y - 1
    cv.vline(cxA, attn["S"][0] + 1, kr)
    cv.vline(cxB, mlpb["S"][0] + 1, kr)
    cv.put(kr, cxA, "└")
    cv.put(kr, cxB, "┘")
    cv.hline(kr, cxA + 1, cxB - 1)
    cv.put(kr, cxS, "┬")
    cv.vline(cxS, kr + 1, cat["N"][0] - 1)
    cv.put(cat["N"][0] - 1, cxS, "▼")

    y = cat["S"][0] + 2
    proj = _abox(cv, cxS, y, ["proj_out", f"Linear {pin} → {dim}"])
    _adown(cv, cat, proj)
    y = proj["S"][0] + 2
    add = _abox(cv, cxS, y, ["⊕  hidden += gate · proj_out"])
    _adown(cv, proj, add)
    y = add["S"][0] + 2
    out = _abox(cv, cxS, y, ["hidden_states →", f"[B, N_txt+N_img, {dim}]", "(split back to text / image)"])
    _adown(cv, add, out)

    _res_rail(cv, cxR, h_in, add, "E")
    return cv.render()


# --------------------------------------------------------------------------- #
# SVG builders.                                                                #
# --------------------------------------------------------------------------- #
def _svg_line(pts, cls="edge", arrow=True, marker="arrow"):
    d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts)
    a = f' marker-end="url(#{marker})"' if arrow else ""
    return f'<path class="{cls}" d="{d}"{a}/>'


def _svg_pipeline(A):
    W, H = 1220, 822
    dim, inc = A["dim"], A["in_channels"]
    # timestep left, captions right — conditioning enters the spine from each side.
    cxTime, cxImg, cxCtx, cxCap = 150, 400, 700, 1000
    cxS = (cxImg + cxCtx) // 2
    y0, y1 = 40, 150

    t0, pT0 = _svg_node(cxTime, y0, 200, 60, "t : timestep", ["(scalar)"], "io")
    t1, pT1 = _svg_node(cxTime, y1, 220, 84, "time_embed", ["sinusoid[256] →", f"temb [B, {dim}]"], "cond")
    i0, pI0 = _svg_node(cxImg, y0, 230, 60, "x : image latent", [f"[B, N_img, {inc}]"], "io")
    i1, pI1 = _svg_node(cxImg, y1, 230, 84, "x_embedder", [f"Linear {inc} → {dim}", f"[B, N_img, {dim}]"])
    c0, pC0 = _svg_node(cxCtx, y0, 250, 60, "encoder_hidden (text)", [f"[B, N_txt, {A['ctx_dim']}]"], "io")
    c1, pC1 = _svg_node(cxCtx, y1, 250, 84, "context_embedder", [f"Linear {A['ctx_dim']} → {dim}", f"[B, N_txt, {dim}]"])
    p0, pP0 = _svg_node(cxCap, y0, 280, 60, "text_encoder_layers", [f"{A['n_cap']}× [B, N_txt, {A['text_enc_dim']}]"], "io")
    p1, pP1 = _svg_node(cxCap, y1, 280, 84, "caption_projection",
                        [f"{A['n_cap']}× Linear {A['text_enc_dim']}→{A['cap_out']}", f"[B, N_txt, {A['cap_out']}]"], "join")

    sy = [332, 452, 572, 672, 762]
    s0, pS0 = _svg_node(cxS, sy[0], 430, 82, "transformer_blocks",
                        [f"{A['n_double']}× BriaFiboTransformerBlock", f"img [B,N_img,{dim}]  ⇄  txt [B,N_txt,{dim}]"], "mod")
    s1, pS1 = _svg_node(cxS, sy[1], 450, 82, "single_transformer_blocks",
                        [f"{A['n_single']}× BriaFiboSingleTransformerBlock", f"seq [B, N_txt+N_img, {dim}]"], "mod")
    s2, pS2 = _svg_node(cxS, sy[2], 340, 66, "norm_out", ["AdaLayerNormContinuous", f"[B, N_img, {dim}]"], "mod")
    s3, pS3 = _svg_node(cxS, sy[3], 280, 46, "proj_out", [f"Linear {dim} → {inc}"])
    s4, pS4 = _svg_node(cxS, sy[4], 320, 46, f"output  [B, N_img, {inc}]", [], "io")

    edges = [_svg_line([pT0["S"], pT1["N"]]), _svg_line([pI0["S"], pI1["N"]]),
             _svg_line([pC0["S"], pC1["N"]]), _svg_line([pP0["S"], pP1["N"]]),
             _svg_line([pS0["S"], pS1["N"]]), _svg_line([pS1["S"], pS2["N"]]),
             _svg_line([pS2["S"], pS3["N"]]), _svg_line([pS3["S"], pS4["N"]])]

    # merge image + text context into the block stack
    jy = pI1["bot"] + 34
    merge = _svg_line([pI1["S"], (cxImg, jy), (cxS, jy), (cxCtx, jy), pC1["S"]], arrow=False)
    merge += _svg_line([(cxS, jy), (cxS, pS0["top"])])
    merge += f'<circle class="jdot" cx="{cxS}" cy="{jy:.1f}" r="3.4"/>'

    # timestep conditioning: dashed amber, from the LEFT into each block's west edge
    def cond(target):
        sx, sy_ = pT1["S"]
        tx, ty = target["W"]
        return _svg_line([(sx, sy_), (sx, ty), (tx - 6, ty)], cls="cond-edge", marker="arrow-c")
    conds = cond(pS0) + cond(pS1) + cond(pS2)

    # per-block caption injection: dashed violet, from the RIGHT into each east edge
    def cap(target):
        sx, sy_ = pP1["S"]
        tx, ty = target["E"]
        return _svg_line([(sx, sy_), (sx, ty), (tx + 6, ty)], cls="cap-edge", marker="arrow-t")
    caps = cap(pS0) + cap(pS1)

    body = "\n".join([merge, conds, caps] + edges +
                     [t0, t1, i0, i1, c0, c1, p0, p1, s0, s1, s2, s3, s4])
    return _svg_wrap(W, H, body)


def _svg_double(A):
    W, H = 940, 1060
    dim, hid = A["dim"], A["ff_hidden"]
    cxIr, cxI, cxM, cxT, cxTr = 60, 250, 460, 670, 880

    a0, pA = _svg_node(cxM, 20, 680, 66, "temb ◄ c  →  AdaLayerNormZero  (norm1 · norm1_context)",
                       ["6 params / stream: shift_msa, scale_msa (in norm1) · gate_msa · shift_mlp, scale_mlp, gate_mlp"], "cond")

    # per-block caption injection rebuilds the text stream before the block
    cap, pCAP = _svg_node(cxT, 116, 260, 58, "caption_proj[i]", [f"SmolLM3 layer · {A['text_enc_dim']}→{A['cap_out']}"], "join")

    yi = 214
    ii, pII = _svg_node(cxI, yi, 250, 74, "hidden_states (image)", ["from prev block", f"[B, N_img, {dim}]"], "io")
    ti, pTI = _svg_node(cxT, yi, 260, 74, "encoder_hidden_states (text)", [f"concat[ ctx[:{A['cap_out']}] ; cap[i] ]", f"[B, N_txt, {dim}]"], "io")
    yn = 324
    n1, pN1 = _svg_node(cxI, yn, 250, 56, "norm1  (AdaLN-Zero)", ["×(1+scale_msa)+shift_msa"], "norm")
    n1c, pN1c = _svg_node(cxT, yn, 260, 56, "norm1_context", ["×(1+c_scale_msa)+c_shift_msa"], "norm")

    ya = 434
    attn, pAT = _svg_node(cxM, ya, 640, 96, "BriaFiboAttention — JOINT self-attention",
                          [f"image q,k,v  +  text add_q,k,v   Linear {dim}→{dim}",
                           f"→ [B, {A['n_heads']}, N_txt+N_img, {A['head_dim']}] · RMSNorm(q,k) · RoPE · SDPA",
                           "concat[text,image] · split → to_out (image) · to_add_out (text)"], "data")
    yd = 594
    d1, pD1 = _svg_node(cxI, yd, 240, 46, "⊕  += gate_msa·attn", [], "add")
    d1c, pD1c = _svg_node(cxT, yd, 240, 46, "⊕  += c_gate_msa·attn", [], "add")
    y2 = 684
    m2, pM2 = _svg_node(cxI, y2, 240, 60, "norm2 (LayerNorm)", ["×(1+scale_mlp)+shift_mlp"], "norm")
    m2c, pM2c = _svg_node(cxT, y2, 240, 60, "norm2_context", ["×(1+c_scale_mlp)+c_shift_mlp"], "norm")
    yf = 794
    ff, pFF = _svg_node(cxI, yf, 240, 60, "ff (GELU MLP)", [f"{dim} → {hid} → {dim}"], "data")
    ffc, pFFc = _svg_node(cxT, yf, 240, 60, "ff_context", [f"{dim} → {hid} → {dim}"], "data")
    yg = 904
    g2, pG2 = _svg_node(cxI, yg, 240, 46, "⊕  += gate_mlp·ff", [], "add")
    g2c, pG2c = _svg_node(cxT, yg, 240, 46, "⊕  += c_gate_mlp·ff", [], "add")
    yo = 994
    o1, pO1 = _svg_node(cxI, yo, 240, 44, "hidden_states → [B,N_img,%d]" % dim, [], "io")
    o1c, pO1c = _svg_node(cxT, yo, 240, 44, "enc_hidden → [B,N_txt,%d]" % dim, [], "io")

    E = []
    E.append(_svg_line([pCAP["S"], pTI["N"]]))
    E.append(_svg_line([pII["S"], pN1["N"]]))
    E.append(_svg_line([pTI["S"], pN1c["N"]]))
    # norm1 -> joint attention (into the wide box at each lane's x)
    E.append(_svg_line([pN1["S"], (cxI, pAT["top"])]))
    E.append(_svg_line([pN1c["S"], (cxT, pAT["top"])]))
    # joint attention -> adds
    E.append(_svg_line([(cxI, pAT["bot"]), pD1["N"]]))
    E.append(_svg_line([(cxT, pAT["bot"]), pD1c["N"]]))
    for a, b in [(pD1, pM2), (pM2, pFF), (pFF, pG2), (pG2, pO1),
                 (pD1c, pM2c), (pM2c, pFFc), (pFFc, pG2c), (pG2c, pO1c)]:
        E.append(_svg_line([a["S"], b["N"]]))

    # residual skips on the outer rails
    def res(railx, src, dst):
        sx, sy_ = src["S"]
        dx, dy = (dst["W"] if railx < src["cx"] else dst["E"])
        return (f'<circle class="jdot" cx="{sx:.1f}" cy="{sy_ + 8:.1f}" r="3"/>'
                + _svg_line([(sx, sy_ + 8), (railx, sy_ + 8), (railx, dy), (dx, dy)], cls="edge res"))
    R = (res(cxIr, pII, pD1) + res(cxIr, pD1, pG2)
         + res(cxTr, pTI, pD1c) + res(cxTr, pD1c, pG2c))

    body = "\n".join(E + [R] + [a0, cap, ii, ti, n1, n1c, attn, d1, d1c, m2, m2c, ff, ffc, g2, g2c, o1, o1c])
    return _svg_wrap(W, H, body)


def _svg_single(A):
    W, H = 820, 900
    dim, mlp, pin = A["dim"], A["single_mlp"], A["single_proj_in"]
    cxA, cxS, cxB, cxR = 220, 410, 600, 740

    a0, pA = _svg_node(cxS, 20, 500, 58, "temb ◄ c → AdaLayerNormZeroSingle", ["3 params: shift, scale (in norm) · gate"], "cond")
    # per-block caption injection refreshes the text half before the seq-concat
    cap, pCAP = _svg_node(cxS, 116, 440, 60, f"caption_proj[i] · SmolLM3 {A['text_enc_dim']}→{A['cap_out']}",
                          [f"text ← concat[ ctx[:{A['cap_out']}] ; cap[i] ]"], "join")
    yh = 216
    hi, pH = _svg_node(cxS, yh, 380, 56, "hidden_states = concat [ text ; image ]", [f"[B, N_txt+N_img, {dim}]"], "io")
    yn = 306
    nm, pNM = _svg_node(cxS, yn, 300, 56, "norm  (AdaLN-Zero-Single)", ["×(1+scale)+shift  → gate"], "norm")
    yp = 416
    at, pAT = _svg_node(cxA, yp, 240, 66, "attn  (pre_only)", ["RMSNorm(q,k) · RoPE", f"[B,{A['n_heads']},N,{A['head_dim']}] · SDPA"], "data")
    ml, pML = _svg_node(cxB, yp, 240, 66, "proj_mlp", [f"Linear {dim} → {mlp}", "GELU (tanh)"], "data")
    yc = 536
    ct, pCT = _svg_node(cxS, yc, 320, 56, "concat [ attn ; mlp ]", [f"{dim} + {mlp} = {pin}"], "join")
    ypo = 636
    po, pPO = _svg_node(cxS, ypo, 260, 46, "proj_out", [f"Linear {pin} → {dim}"])
    yg = 726
    gd, pGD = _svg_node(cxS, yg, 300, 46, "⊕  += gate · proj_out", [], "add")
    yo = 816
    oo, pOO = _svg_node(cxS, yo, 360, 46, f"hidden_states → [B,N_txt+N_img,{dim}] (split)", [], "io")

    E = [_svg_line([pCAP["S"], pH["N"]]), _svg_line([pH["S"], pNM["N"]])]
    # diverge
    jy = pNM["bot"] + 22
    E.append(_svg_line([pNM["S"], (cxS, jy)], arrow=False))
    E.append(_svg_line([(cxS, jy), (cxA, jy), (cxA, pAT["top"])]))
    E.append(_svg_line([(cxS, jy), (cxB, jy), (cxB, pML["top"])]))
    E.append(f'<circle class="jdot" cx="{cxS}" cy="{jy:.1f}" r="3.2"/>')
    # converge
    ky = pAT["bot"] + 22
    E.append(_svg_line([pAT["S"], (cxA, ky), (cxS, ky)], arrow=False))
    E.append(_svg_line([pML["S"], (cxB, ky), (cxS, ky)], arrow=False))
    E.append(_svg_line([(cxS, ky), pCT["N"]]))
    E.append(f'<circle class="jdot" cx="{cxS}" cy="{ky:.1f}" r="3.2"/>')
    E.append(_svg_line([pCT["S"], pPO["N"]]))
    E.append(_svg_line([pPO["S"], pGD["N"]]))
    E.append(_svg_line([pGD["S"], pOO["N"]]))
    # residual
    sx, sy_ = pH["S"]
    E.append(f'<circle class="jdot" cx="{sx:.1f}" cy="{sy_ + 8:.1f}" r="3"/>')
    E.append(_svg_line([(sx, sy_ + 8), (cxR, sy_ + 8), (cxR, pGD["E"][1]), (pGD["E"][0], pGD["E"][1])], cls="edge res"))

    body = "\n".join(E + [a0, cap, hi, nm, at, ml, ct, po, gd, oo])
    return _svg_wrap(W, H, body)


# --------------------------------------------------------------------------- #
# HTML page.                                                                   #
# --------------------------------------------------------------------------- #
_EXTRA_CSS = """
.cap-edge{fill:none;stroke:var(--capsig);stroke-width:1.6;stroke-dasharray:2 4;opacity:.95}
.ah-t{fill:var(--capsig)}
.node-join .ns{fill:var(--muted)}
:root{--capsig:#7A5AF0}
@media (prefers-color-scheme:dark){:root{--capsig:#A78BFA}}
:root[data-theme="light"]{--capsig:#7A5AF0}
:root[data-theme="dark"]{--capsig:#A78BFA}
.legend .sw.cap{border-color:var(--capsig);border-top-style:dashed}
.legend .box.join{background:var(--trace)}
"""


def _wrap_markers(svg):
    """Add the extra caption-signal arrow marker to an SVG's <defs>."""
    extra = ('<marker id="arrow-t" markerWidth="9" markerHeight="9" refX="7" refY="3.2" orient="auto">'
             '<path d="M0,0 L7,3.2 L0,6.4 Z" class="ah-t"/></marker>')
    return svg.replace("</defs>", extra + "</defs>", 1)


def build_html(A):
    axes = "·".join(str(x) for x in A["axes"])
    chips = "".join([
        _html_chip("params", f"<em>{A['params_total']:.2f}</em>B"),
        _html_chip("hidden dim", f"{A['dim']}"),
        _html_chip("double blk", f"{A['n_double']}"),
        _html_chip("single blk", f"{A['n_single']}"),
        _html_chip("heads × dim", f"{A['n_heads']} × {A['head_dim']}"),
        _html_chip("ffn", f"{A['ff_hidden']}"),
        _html_chip("text ctx", f"{A['ctx_dim']}"),
        _html_chip("rope axes", axes),
        _html_chip("in / out ch", f"{A['in_channels']}"),
    ])
    pipe = _wrap_markers(_svg_pipeline(A))
    dbl = _wrap_markers(_svg_double(A))
    sgl = _wrap_markers(_svg_single(A))
    body = f"""<div class="wrap">
<header>
  <p class="eyebrow">FLUX-style MMDiT · data-flow schematic</p>
  <h1>{_svg_esc(A['name'])} <span class="dim">/ briaai · FIBO</span></h1>
  <p class="lede">A text-to-image diffusion transformer over <strong>Wan-2.2 VAE</strong> latents
  ({A['in_channels']}-ch, <span class="mono">patch_size=1</span>). Image latents and a text stream are
  fused through {A['n_double']} double-stream (joint-attention) blocks, then {A['n_single']} single-stream
  blocks, conditioned on the diffusion timestep. FIBO's signature: a <strong>per-block caption
  injection</strong> &mdash; each block rebuilds its text stream as
  <span class="mono">concat[ ctx[:{A['cap_out']}] ; caption_proj[i]&nbsp;]</span> from a fresh SmolLM3
  ({A['text_enc_dim']}-d) layer.</p>
  <div class="chips">{chips}</div>
</header>
<section>
  <div class="slabel"><span class="num mono">01</span><h2>Top-level data flow</h2></div>
  <p class="scap">Image latents (<span class="mono">x_embedder</span>) and text
  (<span class="mono">context_embedder</span>) enter the block stack. Every block also pulls a fresh
  SmolLM3 layer through its own <span class="mono">caption_projection[i]</span> (dashed violet, <span class="mono">◄txt</span>),
  and the timestep conditions each block via adaLN (dashed amber, <span class="mono">◄c</span>).</p>
  <div class="frame">{pipe}</div>
  <div class="legend">
    <div><span class="sw data"></span>token stream</div>
    <div><span class="sw cap"></span>per-block caption (◄txt)</div>
    <div><span class="sw cond"></span>timestep conditioning (◄c)</div>
    <div><span class="box mod"></span>adaLN-modulated block</div>
    <div><span class="box join"></span>caption path</div>
  </div>
</section>
<section>
  <div class="slabel"><span class="num mono">02</span><h2>Double-stream block <span class="mono">·</span> BriaFiboTransformerBlock</h2></div>
  <p class="scap">Before the block runs, the text stream is rebuilt by concatenating its kept first half
  with <span class="mono">caption_proj[i]</span> (the per-block SmolLM3 injection). Two residual lanes &mdash;
  image (<span class="mono">hidden_states</span>) and text (<span class="mono">encoder_hidden_states</span>) &mdash;
  then share one <em>joint</em> attention (q,k,v concatenated, attended together, split back), each with its own
  gated MLP. <span class="mono">AdaLayerNormZero</span> emits <strong>6</strong> params per stream: shift/scale
  applied inside <span class="mono">norm1</span> before attention, plus gate_msa and shift/scale/gate_mlp.</p>
  <div class="frame">{dbl}</div>
</section>
<section>
  <div class="slabel"><span class="num mono">03</span><h2>Single-stream block <span class="mono">·</span> BriaFiboSingleTransformerBlock</h2></div>
  <p class="scap">Text (again refreshed with <span class="mono">caption_proj[i]</span>) and image run as one
  concatenated sequence. Attention and a wide GELU MLP run in <em>parallel</em> on the same normalized input,
  their outputs are concatenated (<span class="mono">{A['dim']}+{A['single_mlp']}={A['single_proj_in']}</span>),
  projected back, gated, and added to the residual. <span class="mono">AdaLayerNormZeroSingle</span> emits 3
  params (shift/scale in-norm, one gate).</p>
  <div class="frame">{sgl}</div>
</section>
<p class="foot">Generated from the live model by <code>draw_arch_fibo.py</code>. Dimensions read directly
from the checkpoint (<code>{MODEL_ID}</code>). Self-contained inline SVG &mdash; open in any browser.</p>
</div>"""
    css = _HTML_CSS + _EXTRA_CSS
    return ("<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
            f"<title>{_svg_esc(A['name'])} · architecture</title>\n<style>{css}</style>\n</head>\n"
            f"<body>\n{body}\n</body>\n</html>\n")


# --------------------------------------------------------------------------- #
# Mermaid.                                                                      #
# --------------------------------------------------------------------------- #
def mermaid(A):
    dim = A["dim"]
    L = [
        "%% BriaFiboTransformer2DModel data-flow — render at https://mermaid.live",
        "flowchart TD",
        f'  x["x: image latent<br/>[B,N_img,{A["in_channels"]}]"] --> xe["x_embedder<br/>Linear {A["in_channels"]}→{dim}"]',
        f'  ctx["encoder_hidden (text)<br/>[B,N_txt,{A["ctx_dim"]}]"] --> ce["context_embedder<br/>Linear {A["ctx_dim"]}→{dim}"]',
        f'  tel["text_encoder_layers<br/>{A["n_cap"]}× SmolLM3 [{A["text_enc_dim"]}]"] --> cp["caption_projection<br/>{A["n_cap"]}× Linear {A["text_enc_dim"]}→{A["cap_out"]}"]',
        '  t["t: timestep"] --> te["time_embed<br/>→ temb"]',
        f'  xe --> db["transformer_blocks<br/>{A["n_double"]}× double (joint attn)"]',
        "  ce --> db",
        f'  db --> sb["single_transformer_blocks<br/>{A["n_single"]}× single"]',
        f'  sb --> no["norm_out<br/>AdaLayerNormContinuous"] --> po["proj_out<br/>Linear {dim}→{A["in_channels"]}"] --> out["output [B,N_img,{A["in_channels"]}]"]',
        "  cp -.->|per-block ◄txt| db",
        "  cp -.->|per-block ◄txt| sb",
        "  te -.->|◄c| db",
        "  te -.->|◄c| sb",
        "  te -.->|◄c| no",
    ]
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# Introspection.                                                               #
# --------------------------------------------------------------------------- #
def build_arch_info(model) -> dict:
    cfg = model.config
    inner = cfg["num_attention_heads"] * cfg["attention_head_dim"]
    db = model.transformer_blocks[0]
    sb = model.single_transformer_blocks[0]
    ff_hidden = db.ff.net[0].proj.out_features
    return {
        "name": type(model).__name__,
        "params_total": sum(p.numel() for p in model.parameters()) / 1e9,
        "dim": inner,
        "n_heads": cfg["num_attention_heads"],
        "head_dim": cfg["attention_head_dim"],
        "n_double": len(model.transformer_blocks),
        "n_single": len(model.single_transformer_blocks),
        "in_channels": cfg["in_channels"],
        "ctx_dim": cfg["joint_attention_dim"],
        "text_enc_dim": cfg["text_encoder_dim"],
        "cap_out": model.caption_projection[0].linear.out_features,
        "n_cap": len(model.caption_projection),
        "ff_hidden": ff_hidden,
        "single_mlp": sb.proj_mlp.out_features,
        "single_proj_in": sb.proj_out.in_features,
        "time_in": model.time_embed.time_proj.num_channels,
        "axes": list(cfg["axes_dims_rope"]),
        "patch": cfg["patch_size"],
    }


_SELF_TEST_INFO = {
    "name": "BriaFiboTransformer2DModel", "params_total": 8.286,
    "dim": 3072, "n_heads": 24, "head_dim": 128, "n_double": 8, "n_single": 38,
    "in_channels": 48, "ctx_dim": 4096, "text_enc_dim": 2048, "cap_out": 1536,
    "n_cap": 46, "ff_hidden": 12288, "single_mlp": 12288, "single_proj_in": 15360,
    "time_in": 256, "axes": [16, 56, 56], "patch": 1,
}


def _emit(A, html_path, mermaid_path):
    bar = "=" * 110
    for label, fn in [("TOP-LEVEL DATA-FLOW GRAPH", draw_pipeline),
                      ("DOUBLE-STREAM BLOCK  (BriaFiboTransformerBlock)", draw_double_block),
                      ("SINGLE-STREAM BLOCK  (BriaFiboSingleTransformerBlock)", draw_single_block)]:
        print(bar)
        print(label)
        print(bar)
        print(fn(A))
        print()
    if html_path:
        with open(html_path, "w") as f:
            f.write(build_html(A))
        print(f"HTML/SVG graph written to {html_path}  (open in a browser)")
    if mermaid_path:
        with open(mermaid_path, "w") as f:
            f.write(mermaid(A) + "\n")
        print(f"Mermaid graph written to {mermaid_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Draw BriaFiboTransformer2DModel (briaai/FIBO) as data-flow graphs.")
    ap.add_argument("--self-test", action="store_true",
                    help="render from a baked-in dims snapshot (no model load)")
    ap.add_argument("--html", default="fibo_arch.html",
                    help="path for the local HTML/SVG export (default: fibo_arch.html)")
    ap.add_argument("--no-html", action="store_true", help="skip the HTML/SVG export")
    ap.add_argument("--mermaid", metavar="PATH", default="fibo_arch.mmd",
                    help="path for the Mermaid .mmd export (default: fibo_arch.mmd)")
    ap.add_argument("--no-mermaid", action="store_true", help="skip the Mermaid export")
    args = ap.parse_args()

    html_path = None if args.no_html else args.html
    mermaid_path = None if args.no_mermaid else args.mermaid

    if args.self_test:
        _emit(_SELF_TEST_INFO, html_path, mermaid_path)
        return

    import torch
    from diffusers import BriaFiboTransformer2DModel

    print(f"Loading transformer from {MODEL_ID}/transformer ...")
    model = BriaFiboTransformer2DModel.from_pretrained(
        MODEL_ID, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    _emit(build_arch_info(model), html_path, mermaid_path)


if __name__ == "__main__":
    main()
