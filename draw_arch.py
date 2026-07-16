# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Draw the ZImageTransformer2DModel architecture as a *graph* instead of the flat
module listing that ``print(model)`` (and ``print_arch.py``) produce.

Two ASCII renderings are emitted:

  1. A top-level data-flow graph  (image / caption / timestep paths -> refiners
     -> joint transformer -> final layer -> output).
  2. A single ``ZImageTransformerBlock`` data-flow graph  (sandwich-norm adaLN
     block: the residual stream through attention and the SwiGLU feed-forward).

Two files are written alongside by default:
  * ``z_image_arch.html`` - a self-contained, theme-aware page with both graphs as
    inline SVG (no network, no external assets); open it directly in a browser.
  * ``z_image_arch.mmd``  - a Mermaid graph for GitHub / mermaid.live.

All numbers are read live from the loaded model, so the picture tracks the real
weights rather than hard-coded constants.

Usage:
  python draw_arch.py                   # print ASCII, write .html + .mmd
  python draw_arch.py --html arch.html  # choose the HTML output path
  python draw_arch.py --no-html         # skip the HTML/SVG export
  python draw_arch.py --no-mermaid      # skip the Mermaid export
  python draw_arch.py --self-test       # render with baked-in dims (no model load)
"""

from __future__ import annotations

import argparse

MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"


# --------------------------------------------------------------------------- #
# Canvas: a sparse 2D character grid you paste boxes / lines onto.            #
# --------------------------------------------------------------------------- #
class Canvas:
    def __init__(self):
        self._g: dict[tuple[int, int], str] = {}
        self.maxr = 0
        self.maxc = 0

    def put(self, r, c, ch, over=True):
        if ch == " ":
            return
        if not over and (r, c) in self._g:
            return
        self._g[(r, c)] = ch
        self.maxr = max(self.maxr, r)
        self.maxc = max(self.maxc, c)

    def text(self, r, c, s):
        for i, ch in enumerate(s):
            self.put(r, c + i, ch)

    def box(self, r, c, lines, pad=1):
        """Top-left corner at (r, c). Returns a dict of useful port coords."""
        w = _boxw(lines, pad)
        h = len(lines) + 2
        rb, cr = r + h - 1, c + w - 1
        self.put(r, c, "┌")
        self.put(r, cr, "┐")
        self.put(rb, c, "└")
        self.put(rb, cr, "┘")
        for j in range(c + 1, cr):
            self.put(r, j, "─")
            self.put(rb, j, "─")
        for i, l in enumerate(lines):
            rr = r + 1 + i
            self.put(rr, c, "│")
            self.put(rr, cr, "│")
            self.text(rr, c + 1 + pad, l)
        cx = c + w // 2
        return {
            "r": r, "c": c, "w": w, "h": h, "cx": cx,
            "N": (r, cx), "S": (rb, cx),
            "W": (r + h // 2, c), "E": (r + h // 2, cr),
        }

    def vline(self, c, r1, r2, over=False):
        for r in range(min(r1, r2), max(r1, r2) + 1):
            self.put(r, c, "│", over=over)

    def hline(self, r, c1, c2, over=False):
        for c in range(min(c1, c2), max(c1, c2) + 1):
            self.put(r, c, "─", over=over)

    def render(self):
        rows = []
        for r in range(self.maxr + 1):
            rows.append(
                "".join(self._g.get((r, c), " ") for c in range(self.maxc + 1)).rstrip()
            )
        return "\n".join(rows)


def _boxw(lines, pad=1):
    """Total drawn width of a box (borders + padding) for a set of text lines."""
    return max((len(l) for l in lines), default=0) + 2 * pad + 2


def _stack(cv: Canvas, cx, start_r, blocks, gap=1):
    """Stack boxes vertically centered on column ``cx`` and connect them with
    ``│`` + ``▼``. ``blocks`` is a list of line-lists. Returns list of ports."""
    ports = []
    r = start_r
    prev = None
    for lines in blocks:
        w = _boxw(lines)
        b = cv.box(r, cx - w // 2, lines)
        if prev is not None:
            cv.vline(cx, prev["S"][0] + 1, b["N"][0] - 1)
            cv.put(b["N"][0] - 1, cx, "▼")
        ports.append(b)
        prev = b
        r = b["r"] + b["h"] + gap
    return ports


# --------------------------------------------------------------------------- #
# Top-level data-flow graph.                                                   #
# --------------------------------------------------------------------------- #
def draw_pipeline(A) -> str:
    cv = Canvas()

    dim = A["dim"]
    # Column anchors (character columns of each vertical lane).
    cxL = 17            # image path
    cxR = 62            # caption / text path
    cxS = (cxL + cxR) // 2   # merged spine (concat / layers / final / output)
    cxT = 92            # timestep / conditioning path (detached, far right)

    # Title.
    title = f"{A['name']}   ·   {A['params_total']:.2f}B params"
    sub = (f"dim={dim} · {A['n_heads']} heads × {A['head_dim']} · "
           f"{A['n_layers']} layers · SwiGLU {A['ffn_hidden']} · RoPE {A['axes']}")
    cv.text(0, cxS - len(title) // 2, title)
    cv.text(1, cxS - len(sub) // 2, sub)

    start = 4
    # Every row carries the same line-count across the three columns so the
    # lanes stay aligned (unequal heights would stagger the arrows and merge).
    # Image path -------------------------------------------------------------
    img = _stack(cv, cxL, start, [
        ["x : noisy latent", f"patchify → {A['patch_in']}-d/token"],
        ["all_x_embedder", "Linear", f"{A['patch_in']} → {dim}"],
        [f"noise_refiner           ◄─ c", f"{A['n_refiner']}× ZImageBlock", "adaLN-modulated"],
    ])
    # Text path --------------------------------------------------------------
    txt = _stack(cv, cxR, start, [
        ["cap_feats : text cond", f"[B, N_txt, {A['cap_dim']}]"],
        ["cap_embedder", "RMSNorm + Linear", f"{A['cap_dim']} → {dim}"],
        ["context_refiner", f"{A['n_refiner']}× ZImageBlock", "no modulation"],
    ])
    # Timestep / conditioning path (detached lane on the right) --------------
    _stack(cv, cxT, start, [
        ["t : timestep", f"→ sinusoid[{A['t_in']}]"],
        ["t_embedder", f"Linear {A['t_in']}→{A['t_mid']}", f"SiLU, Linear→{A['t_out']}"],
        ["c : adaLN cond [256]", "timestep signal", "→ every ◄─ c block"],
    ])

    # Merge image + caption into the joint sequence --------------------------
    ref_bottom = max(img[-1]["S"][0], txt[-1]["S"][0])
    rj = ref_bottom + 2                       # junction row
    cv.vline(cxL, img[-1]["S"][0] + 1, rj)
    cv.vline(cxR, txt[-1]["S"][0] + 1, rj)
    cv.put(rj, cxL, "└")
    cv.put(rj, cxR, "┘")
    cv.hline(rj, cxL + 1, cxR - 1)
    cv.put(rj, cxS, "┬")

    spine = _stack(cv, cxS, rj + 2, [
        [f"concat  [ x ; cap ]", f"[B, N_img+N_txt, {dim}]", "image tokens first"],
        [f"layers                   ◄─ c", f"{A['n_layers']}× ZImageBlock",
         f"adaLN · self-attn {A['n_heads']}×{A['head_dim']} · RoPE"],
        ["all_final_layer          ◄─ c", "LayerNorm → × scale(c)", f"Linear {dim} → {A['patch_out']}"],
        [f"output  [B, N_img, {A['patch_out']}]", f"unpatchify → {A['in_channels']} channels"],
    ])
    # connect junction -> concat
    cv.vline(cxS, rj + 1, spine[0]["N"][0] - 1)
    cv.put(spine[0]["N"][0] - 1, cxS, "▼")

    legend = "◄─ c : timestep conditioning applied via adaLN modulation inside the block"
    cv.text(spine[-1]["S"][0] + 2, cxL, legend)
    return cv.render()


# --------------------------------------------------------------------------- #
# Single ZImageTransformerBlock data-flow graph (adaLN / sandwich-norm).       #
# --------------------------------------------------------------------------- #
def draw_block(A) -> str:
    cv = Canvas()
    dim, hid = A["dim"], A["ffn_hidden"]
    cxB = 26            # main residual-stream spine
    railc = 62          # residual bypass rail (to the right of the spine)

    title = "ZImageTransformerBlock   (modulation=True → layers[×%d] & noise_refiner[×%d])" % (
        A["n_layers"], A["n_refiner"])
    cv.text(0, 2, title)

    # adaLN conditioning header.
    cv.text(2, 2, f"c[256] ─► adaLN_modulation: Linear 256 → {A['adaln_out']}  ─►  chunk 4")
    cv.text(3, 11, "scale_msa · gate_msa · scale_mlp · gate_mlp     (gate = tanh·,  scale = 1 + ·)")

    start = 5
    # gap=2 so a residual tap (├) and the arrowhead (▼) into the next box get
    # their own rows instead of overwriting each other.
    nodes = _stack(cv, cxB, start, [
        ["x  (residual stream)"],
        ["attention_norm1  (RMSNorm)", "× scale_msa"],
        ["Attention  —  self-attn", f"to_q/k/v : Linear {dim}→{dim}",
         f"{A['n_heads']} heads × {A['head_dim']} · QK-RMSNorm · RoPE",
         "SDPA → to_out"],
        ["attention_norm2  (RMSNorm)", "× gate_msa"],
        ["⊕  residual add"],
        ["ffn_norm1  (RMSNorm)", "× scale_mlp"],
        ["FeedForward  (SwiGLU)", f"w2( SiLU(w1·x) ⊙ w3·x )", f"{dim} → {hid} → {dim}"],
        ["ffn_norm2  (RMSNorm)", "× gate_mlp"],
        ["⊕  residual add"],
        ["output"],
    ], gap=2)
    x_in, norm1, attn, norm2, add1, fnorm1, ffn, fnorm2, add2, out = nodes

    # Residual bypass 1: split just below x -> add1 (E side).
    _bypass(cv, split_r=x_in["S"][0] + 1, add=add1, spine_c=cxB, railc=railc)
    # Residual bypass 2: split just below add1 -> add2.
    _bypass(cv, split_r=add1["S"][0] + 1, add=add2, spine_c=cxB, railc=railc)

    return cv.render()


def _bypass(cv: Canvas, split_r, add, spine_c, railc):
    """Draw a residual skip: tap the spine at ``split_r`` and route right, down,
    and back into ``add`` from the east."""
    cv.put(split_r, spine_c, "├")
    cv.hline(split_r, spine_c + 1, railc)
    cv.put(split_r, railc, "┐")
    add_r = add["E"][0]
    cv.vline(railc, split_r + 1, add_r)
    cv.put(add_r, railc, "┘")
    cv.hline(add_r, add["E"][1] + 1, railc - 1)
    cv.put(add_r, add["E"][1] + 1, "◄")


# --------------------------------------------------------------------------- #
# Mermaid export.                                                              #
# --------------------------------------------------------------------------- #
def mermaid(A) -> str:
    dim = A["dim"]
    L = [
        "%% ZImageTransformer2DModel data-flow — render at https://mermaid.live",
        "flowchart TD",
        '  x["x: noisy latent<br/>patchify → %d-d"] --> xe["all_x_embedder<br/>Linear %d→%d"]'
        % (A["patch_in"], A["patch_in"], dim),
        '  xe --> nr["noise_refiner<br/>%d× ZImageBlock (adaLN)"]' % A["n_refiner"],
        '  cap["cap_feats: text<br/>[B,N_txt,%d]"] --> ce["cap_embedder<br/>RMSNorm+Linear %d→%d"]'
        % (A["cap_dim"], A["cap_dim"], dim),
        '  ce --> cr["context_refiner<br/>%d× ZImageBlock (plain)"]' % A["n_refiner"],
        '  t["t: timestep"] --> te["t_embedder<br/>MLP 256→1024→256"]',
        '  te -->|c| c(["c: adaLN cond [256]"])',
        '  nr --> cat["concat [x ; cap]<br/>[B,N_img+N_txt,%d]"]' % dim,
        "  cr --> cat",
        '  cat --> lay["layers<br/>%d× ZImageBlock (adaLN, RoPE)"]' % A["n_layers"],
        '  lay --> fin["all_final_layer<br/>LayerNorm→scale(c)→Linear %d→%d"]'
        % (dim, A["patch_out"]),
        '  fin --> out["output → unpatchify<br/>%d channels"]' % A["in_channels"],
        "  c -.->|adaLN| nr",
        "  c -.->|adaLN| lay",
        "  c -.->|adaLN| fin",
    ]
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# Local HTML / inline-SVG export (theme-aware, self-contained, no network).    #
# --------------------------------------------------------------------------- #
def _svg_esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _svg_node(cx, ytop, w, h, title, subs=(), kind="data"):
    """Return (svg_fragment, ports). Box centered on ``cx``; ``ytop`` is top edge."""
    x = cx - w / 2
    frags = [f'<g class="node node-{kind}">',
             f'<rect x="{x:.1f}" y="{ytop:.1f}" width="{w}" height="{h}" rx="9"/>']
    if kind == "mod":
        frags.append(f'<rect class="stripe" x="{x:.1f}" y="{ytop:.1f}" width="4" height="{h}" rx="2"/>')
        frags.append(f'<text class="tag" x="{x + w - 9:.1f}" y="{ytop + 15:.1f}" text-anchor="end">◄ c</text>')
    n = 1 + len(subs)
    total = 18 + (n - 1) * 16
    ty = ytop + (h - total) / 2 + 14
    frags.append(f'<text class="nt" x="{cx:.1f}" y="{ty:.1f}" text-anchor="middle">{_svg_esc(title)}</text>')
    for i, s in enumerate(subs):
        frags.append(f'<text class="ns" x="{cx:.1f}" y="{ty + 18 + i * 16:.1f}" text-anchor="middle">{_svg_esc(s)}</text>')
    frags.append("</g>")
    ports = dict(N=(cx, ytop), S=(cx, ytop + h), W=(x, ytop + h / 2),
                 E=(x + w, ytop + h / 2), cx=cx, top=ytop, bot=ytop + h)
    return "\n".join(frags), ports


def _svg_vedge(a, b, cls="edge"):
    x1, y1 = a["S"]
    x2, y2 = b["N"]
    return f'<path class="{cls}" d="M {x1:.1f} {y1:.1f} L {x2:.1f} {y2:.1f}" marker-end="url(#arrow)"/>'


def _svg_wrap(w, h, body):
    return (f'<svg viewBox="0 0 {w} {h}" role="img" preserveAspectRatio="xMidYMin meet" '
            f'style="width:100%;height:auto;min-width:{min(w, 720)}px">\n'
            '<defs>'
            '<marker id="arrow" markerWidth="9" markerHeight="9" refX="7" refY="3.2" orient="auto">'
            '<path d="M0,0 L7,3.2 L0,6.4 Z" class="ah"/></marker>'
            '<marker id="arrow-c" markerWidth="9" markerHeight="9" refX="7" refY="3.2" orient="auto">'
            '<path d="M0,0 L7,3.2 L0,6.4 Z" class="ah-c"/></marker>'
            '</defs>\n' + body + '\n</svg>')


def _svg_pipeline(A):
    W, H = 1000, 912
    dim = A["dim"]
    cxI, cxT, cxC = 160, 470, 810
    cxS = (cxI + cxT) // 2
    ys = [40, 150, 278]

    i0, pI0 = _svg_node(cxI, ys[0], 210, 64, "x : noisy latent", [f"patchify → {A['patch_in']}-d / token"], "io")
    i1, pI1 = _svg_node(cxI, ys[1], 210, 76, "all_x_embedder", ["Linear", f"{A['patch_in']} → {dim}"])
    i2, pI2 = _svg_node(cxI, ys[2], 210, 82, "noise_refiner", [f"{A['n_refiner']}× ZImageBlock", "adaLN-modulated"], "mod")
    t0, pT0 = _svg_node(cxT, ys[0], 210, 64, "cap_feats : text", [f"[B, N_txt, {A['cap_dim']}]"], "io")
    t1, pT1 = _svg_node(cxT, ys[1], 210, 76, "cap_embedder", ["RMSNorm + Linear", f"{A['cap_dim']} → {dim}"])
    t2, pT2 = _svg_node(cxT, ys[2], 210, 82, "context_refiner", [f"{A['n_refiner']}× ZImageBlock", "no modulation"])
    c0, pC0 = _svg_node(cxC, ys[0], 200, 64, "t : timestep", [f"→ sinusoid [{A['t_in']}]"], "io")
    c1, pC1 = _svg_node(cxC, ys[1], 200, 76, "t_embedder", [f"MLP {A['t_in']} → {A['t_mid']} → {A['t_out']}"])
    c2, pC2 = _svg_node(cxC, ys[2], 200, 82, "c : adaLN cond", [f"[{A['t_out']}] · broadcast"], "cond")

    s0, pS0 = _svg_node(cxS, 470, 260, 82, "concat  [ x ; cap ]", [f"[B, N_img + N_txt, {dim}]", "image tokens first"], "join")
    s1, pS1 = _svg_node(cxS, 592, 300, 82, "layers", [f"{A['n_layers']}× ZImageBlock",
                        f"adaLN · self-attn {A['n_heads']}×{A['head_dim']} · RoPE"], "mod")
    s2, pS2 = _svg_node(cxS, 714, 300, 82, "all_final_layer", ["LayerNorm → × scale(c)", f"Linear {dim} → {A['patch_out']}"], "mod")
    s3, pS3 = _svg_node(cxS, 836, 260, 60, "output → unpatchify", [f"[B, N_img, {A['patch_out']}] → {A['in_channels']} ch"], "io")

    edges = [_svg_vedge(pI0, pI1), _svg_vedge(pI1, pI2), _svg_vedge(pT0, pT1), _svg_vedge(pT1, pT2),
             _svg_vedge(pC0, pC1), _svg_vedge(pC1, pC2), _svg_vedge(pS0, pS1), _svg_vedge(pS1, pS2), _svg_vedge(pS2, pS3)]

    jy = pI2["bot"] + 40
    mx1 = pI2["S"][0]
    mx2 = pT2["S"][0]
    merge = (f'<path class="edge" d="M {mx1:.1f} {pI2["S"][1]:.1f} L {mx1:.1f} {jy:.1f} '
             f'L {cxS} {jy:.1f} L {mx2:.1f} {jy:.1f} L {mx2:.1f} {pT2["S"][1]:.1f}"/>'
             f'<path class="edge" d="M {cxS} {jy:.1f} L {cxS} {pS0["top"]:.1f}" marker-end="url(#arrow)"/>'
             f'<circle class="jdot" cx="{cxS}" cy="{jy:.1f}" r="3.2"/>')

    def cond_curve(a, target):
        sx, sy = a["S"]
        tx, ty = target["E"]
        midx = (sx + tx) / 2 + 60
        return (f'<path class="cond-edge" d="M {sx:.1f} {sy:.1f} '
                f'C {sx:.1f} {sy + 60:.1f}, {midx:.1f} {ty:.1f}, {tx + 6:.1f} {ty:.1f}" '
                f'marker-end="url(#arrow-c)"/>')
    cond = cond_curve(pC2, pS1) + cond_curve(pC2, pS2)

    body = "\n".join([merge, cond] + edges + [i0, i1, i2, t0, t1, t2, c0, c1, c2, s0, s1, s2, s3])
    return _svg_wrap(W, H, body)


def _svg_block(A):
    W, H = 760, 1060
    dim, hid = A["dim"], A["ffn_hidden"]
    cx, rail = 300, 588

    a0, pA0 = _svg_node(cx, 30, 540, 74, f"adaLN_modulation(c) : Linear {A['t_out']} → {A['adaln_out']}",
                        ["→ chunk 4 →  scale_msa · gate_msa · scale_mlp · gate_mlp",
                         "gate = tanh(·)      scale = 1 + (·)"], "cond")

    y = 150
    def nxt(h, gap=34):
        nonlocal y
        top = y
        y = top + h + gap
        return top

    specs = [
        ("x", ["residual stream"], "io", 52),
        ("attention_norm1  (RMSNorm)", ["× scale_msa"], "norm", 60),
        ("Attention — self-attention",
         [f"to_q / to_k / to_v : Linear {dim} → {dim}",
          f"{A['n_heads']} heads × {A['head_dim']} · QK-RMSNorm · RoPE · SDPA → to_out"], "data", 76),
        ("attention_norm2  (RMSNorm)", ["× gate_msa"], "norm", 60),
        ("⊕   residual add", [], "add", 46),
        ("ffn_norm1  (RMSNorm)", ["× scale_mlp"], "norm", 60),
        ("FeedForward  (SwiGLU)", ["w2( SiLU(w1·x) ⊙ w3·x )", f"{dim} → {hid} → {dim}"], "data", 76),
        ("ffn_norm2  (RMSNorm)", ["× gate_mlp"], "norm", 60),
        ("⊕   residual add", [], "add", 46),
        ("output", [], "io", 46),
    ]
    widths = {"data": 420, "norm": 300, "add": 220, "io": 200}
    nodes, ports = [], []
    for title, subs, kind, h in specs:
        top = nxt(h)
        frag, p = _svg_node(cx, top, widths.get(kind, 300), h, title, subs, kind)
        nodes.append(frag)
        ports.append(p)
    x_p, n1, attn, n2, add1, f1, ffn, f2, add2, out = ports

    edges = [_svg_vedge(a, b) for a, b in
             [(x_p, n1), (n1, attn), (attn, n2), (n2, add1), (add1, f1),
              (f1, ffn), (ffn, f2), (f2, add2), (add2, out)]]

    def bypass(split_port, add_port):
        sx, sy = split_port["S"]
        sply = sy + 17
        ax, ay = add_port["E"]
        return (f'<circle class="jdot" cx="{sx:.1f}" cy="{sply:.1f}" r="3.2"/>'
                f'<path class="edge res" d="M {sx:.1f} {sply:.1f} L {rail} {sply:.1f} '
                f'L {rail} {ay:.1f} L {ax + 6:.1f} {ay:.1f}" marker-end="url(#arrow)"/>')
    res = bypass(x_p, add1) + bypass(add1, add2)

    busx = 52
    taps = (n1, n2, f1, f2)
    bus_bot = max(p["W"][1] for p in taps)
    seg = [f'<path class="cond-edge" d="M {busx} {pA0["S"][1]:.1f} L {busx} {bus_bot:.1f}"/>']
    for p in taps:
        tx, ty = p["W"]
        seg.append(f'<path class="cond-edge" d="M {busx} {ty:.1f} L {tx - 6:.1f} {ty:.1f}" '
                   f'marker-end="url(#arrow-c)"/>')
    cond = "".join(seg)

    body = "\n".join([cond] + edges + [res, a0] + nodes)
    return _svg_wrap(W, H, body)


_HTML_CSS = """
:root{
  --ground:#F4F7F9; --surface:#FFFFFF; --panel:#FBFCFD; --ink:#0F1720;
  --muted:#586573; --line:#E2E8ED; --trace:#0E8F91; --trace-soft:#0e8f9122;
  --signal:#C0791A; --grid:#e6eef1; --shadow:0 1px 2px #0f172010,0 8px 24px #0f172014;
}
@media (prefers-color-scheme:dark){
  :root{
    --ground:#0B131B; --surface:#131F2A; --panel:#0F1a24; --ink:#E7EEF4;
    --muted:#8FA3B3; --line:#233544; --trace:#34CFC9; --trace-soft:#34cfc922;
    --signal:#F0AE4E; --grid:#152430; --shadow:0 1px 2px #00000040,0 10px 30px #00000055;
  }
}
:root[data-theme="light"]{
  --ground:#F4F7F9; --surface:#FFFFFF; --panel:#FBFCFD; --ink:#0F1720;
  --muted:#586573; --line:#E2E8ED; --trace:#0E8F91; --trace-soft:#0e8f9122;
  --signal:#C0791A; --grid:#e6eef1; --shadow:0 1px 2px #0f172010,0 8px 24px #0f172014;
}
:root[data-theme="dark"]{
  --ground:#0B131B; --surface:#131F2A; --panel:#0F1a24; --ink:#E7EEF4;
  --muted:#8FA3B3; --line:#233544; --trace:#34CFC9; --trace-soft:#34cfc922;
  --signal:#F0AE4E; --grid:#152430; --shadow:0 1px 2px #00000040,0 10px 30px #00000055;
}
*{box-sizing:border-box}
html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--ground);color:var(--ink);
  font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;line-height:1.55;
  -webkit-font-smoothing:antialiased}
.wrap{max-width:1080px;margin:0 auto;padding:clamp(28px,5vw,64px) clamp(18px,4vw,40px) 80px}
header{border-bottom:1px solid var(--line);padding-bottom:30px;margin-bottom:38px}
.eyebrow{font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-size:12px;
  letter-spacing:.22em;text-transform:uppercase;color:var(--trace);margin:0 0 14px}
h1{font-size:clamp(30px,5vw,46px);line-height:1.05;letter-spacing:-.02em;margin:0;
  font-weight:680;text-wrap:balance}
h1 .dim{color:var(--muted);font-weight:500}
.lede{color:var(--muted);font-size:clamp(15px,2vw,17px);max-width:64ch;margin:16px 0 0}
.chips{display:flex;flex-wrap:wrap;gap:10px;margin-top:26px}
.chip{display:flex;flex-direction:column;gap:2px;background:var(--surface);border:1px solid var(--line);
  border-radius:10px;padding:9px 14px;box-shadow:var(--shadow)}
.chip .k{font-family:ui-monospace,Menlo,monospace;font-size:10.5px;letter-spacing:.14em;
  text-transform:uppercase;color:var(--muted)}
.chip .v{font-family:ui-monospace,Menlo,monospace;font-size:16px;font-weight:600;
  color:var(--ink);font-variant-numeric:tabular-nums}
.chip .v em{color:var(--trace);font-style:normal}
section{margin-top:46px}
.slabel{display:flex;align-items:baseline;gap:12px;margin:0 0 4px}
.slabel .num{font-family:ui-monospace,Menlo,monospace;font-size:12px;color:var(--trace);letter-spacing:.1em}
.slabel h2{font-size:19px;margin:0;font-weight:640;letter-spacing:-.01em}
.scap{color:var(--muted);font-size:14px;margin:0 0 18px;max-width:72ch}
.scap .mono,.lede .mono{font-family:ui-monospace,Menlo,monospace;font-size:.92em;color:var(--ink)}
.frame{background:var(--panel);border:1px solid var(--line);border-radius:16px;
  padding:22px clamp(14px,3vw,30px);box-shadow:var(--shadow);overflow-x:auto;
  background-image:linear-gradient(var(--grid) 1px,transparent 1px),
    linear-gradient(90deg,var(--grid) 1px,transparent 1px);
  background-size:26px 26px;background-position:-1px -1px}
.node rect{fill:var(--surface);stroke:var(--line);stroke-width:1.4}
.node .nt{fill:var(--ink);font-family:ui-monospace,"SF Mono",Menlo,Consolas,monospace;font-size:14.5px;font-weight:600}
.node .ns{fill:var(--muted);font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12px}
.node-io rect{fill:transparent;stroke:var(--muted);stroke-dasharray:3 4}
.node-io .nt{fill:var(--muted)}
.node-join rect{stroke:var(--trace);stroke-width:1.7}
.node-mod .stripe{fill:var(--signal);stroke:none}
.node-mod .tag{fill:var(--signal);font-family:ui-monospace,Menlo,monospace;font-size:11px;font-weight:600}
.node-cond rect{fill:var(--trace-soft);stroke:var(--signal);stroke-width:1.5}
.node-add rect{fill:var(--surface);stroke:var(--trace);stroke-width:1.7}
.node-add .nt{fill:var(--trace);font-size:16px}
.edge{fill:none;stroke:var(--trace);stroke-width:1.7}
.edge.res{stroke:var(--trace);opacity:.62;stroke-width:1.5}
.cond-edge{fill:none;stroke:var(--signal);stroke-width:1.6;stroke-dasharray:5 5;opacity:.9}
.ah{fill:var(--trace)}
.ah-c{fill:var(--signal)}
.jdot{fill:var(--trace)}
.legend{display:flex;flex-wrap:wrap;gap:20px 28px;margin-top:20px;padding:16px 18px;
  border:1px solid var(--line);border-radius:12px;background:var(--surface)}
.legend div{display:flex;align-items:center;gap:9px;font-size:13px;color:var(--muted)}
.legend .sw{width:26px;height:0;border-top-width:2px;border-top-style:solid;flex:none}
.legend .sw.data{border-color:var(--trace)}
.legend .sw.cond{border-color:var(--signal);border-top-style:dashed}
.legend .box{width:15px;height:15px;border-radius:4px;flex:none}
.legend .box.mod{background:var(--signal)}
.legend .box.io{border:1.4px dashed var(--muted)}
.legend .mono{font-family:ui-monospace,Menlo,monospace}
.foot{margin-top:52px;padding-top:24px;border-top:1px solid var(--line);color:var(--muted);font-size:13.5px}
.foot code{font-family:ui-monospace,Menlo,monospace;background:var(--surface);border:1px solid var(--line);
  border-radius:5px;padding:1.5px 6px;color:var(--ink);font-size:12.5px}
"""


def _html_chip(k, v):
    return f'<div class="chip"><span class="k">{k}</span><span class="v">{v}</span></div>'


def build_html(A):
    """Self-contained, theme-aware HTML page with both graphs as inline SVG."""
    axes = "·".join(str(x) for x in A["axes"])
    chips = "".join([
        _html_chip("params", f"<em>{A['params_total']:.2f}</em>B"),
        _html_chip("hidden dim", f"{A['dim']}"),
        _html_chip("layers", f"{A['n_layers']}"),
        _html_chip("heads × dim", f"{A['n_heads']} × {A['head_dim']}"),
        _html_chip("ffn (swiglu)", f"{A['ffn_hidden']}"),
        _html_chip("refiners", f"{A['n_refiner']} + {A['n_refiner']}"),
        _html_chip("rope axes", axes),
        _html_chip("in / out ch", f"{A['in_channels']}"),
    ])
    body = f"""<div class="wrap">
<header>
  <p class="eyebrow">Diffusion Transformer · data-flow schematic</p>
  <h1>{_svg_esc(A['name'])} <span class="dim">/ Z-Image-Turbo</span></h1>
  <p class="lede">A text-to-image flow transformer. Three input streams &mdash; noisy image
  latents, text-conditioning features, and the diffusion timestep &mdash; are embedded,
  refined, fused into one token sequence, and processed by a stack of sandwich-norm adaLN
  blocks. This is the tensor flow, not the module listing.</p>
  <div class="chips">{chips}</div>
</header>
<section>
  <div class="slabel"><span class="num mono">01</span><h2>Top-level data flow</h2></div>
  <p class="scap">Image tokens are refined with timestep conditioning (<span class="mono">noise_refiner</span>);
  text tokens are refined without it (<span class="mono">context_refiner</span>). They are concatenated
  <span class="mono">[image ; text]</span>, run through {A['n_layers']} joint transformer blocks, and the
  image span is projected back to latent channels.</p>
  <div class="frame">{_svg_pipeline(A)}</div>
  <div class="legend">
    <div><span class="sw data"></span>data / token stream</div>
    <div><span class="sw cond"></span>timestep conditioning (adaLN)</div>
    <div><span class="box mod"></span>adaLN-modulated block <span class="mono">◄ c</span></div>
    <div><span class="box io"></span>input / output tensor</div>
  </div>
</section>
<section>
  <div class="slabel"><span class="num mono">02</span><h2>Inside one ZImageTransformerBlock</h2></div>
  <p class="scap">The repeated unit ({A['n_layers']}× in <span class="mono">layers</span>,
  {A['n_refiner']}× in <span class="mono">noise_refiner</span>). A sandwich-norm residual:
  normalize &rarr; <em>scale</em> &rarr; sublayer &rarr; normalize &rarr; <em>gate</em> &rarr; add.
  The four modulation signals come from <span class="mono">adaLN(c)</span>;
  <span class="mono">context_refiner</span> runs the same graph with modulation disabled.</p>
  <div class="frame">{_svg_block(A)}</div>
</section>
<p class="foot">Generated from the live model by <code>draw_arch.py</code>. Dimensions read directly
from the loaded checkpoint (<code>{MODEL_ID}</code>). Open this file in any browser &mdash; it is
fully self-contained (inline SVG, no network).</p>
</div>"""
    return f"<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n" \
           f"<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n" \
           f"<title>{_svg_esc(A['name'])} · architecture</title>\n<style>{_HTML_CSS}</style>\n</head>\n" \
           f"<body>\n{body}\n</body>\n</html>\n"


# --------------------------------------------------------------------------- #
# Introspection: pull live dims out of the loaded model.                       #
# --------------------------------------------------------------------------- #
def build_arch_info(model) -> dict:
    cfg = model.config
    key = next(iter(model.all_x_embedder.keys()))
    x_emb = model.all_x_embedder[key]
    final = model.all_final_layer[key]
    blk = model.layers[0]
    ff = blk.feed_forward
    te = model.t_embedder.mlp
    total = sum(p.numel() for p in model.parameters())
    return {
        "name": type(model).__name__,
        "params_total": total / 1e9,
        "dim": cfg["dim"],
        "n_layers": len(model.layers),
        "n_refiner": len(model.noise_refiner),
        "n_heads": cfg["n_heads"],
        "head_dim": cfg["dim"] // cfg["n_heads"],
        "ffn_hidden": ff.w1.out_features,
        "patch_in": x_emb.in_features,
        "patch_out": final.linear.out_features,
        "cap_dim": model.cap_embedder[1].in_features,
        "adaln_out": blk.adaLN_modulation[0].out_features,
        "axes": list(cfg["axes_dims"]),
        "in_channels": cfg["in_channels"],
        "t_in": te[0].in_features,
        "t_mid": te[0].out_features,
        "t_out": te[2].out_features,
    }


# A frozen snapshot (from the real model) so the drawing can be iterated / shown
# without loading 6 GB of weights.
_SELF_TEST_INFO = {
    "name": "ZImageTransformer2DModel",
    "params_total": 6.15,
    "dim": 3840,
    "n_layers": 30,
    "n_refiner": 2,
    "n_heads": 30,
    "head_dim": 128,
    "ffn_hidden": 10240,
    "patch_in": 64,
    "patch_out": 64,
    "cap_dim": 2560,
    "adaln_out": 15360,
    "axes": [32, 48, 48],
    "in_channels": 16,
    "t_in": 256,
    "t_mid": 1024,
    "t_out": 256,
}


def _emit(A, html_path, mermaid_path):
    bar = "=" * 100
    print(bar)
    print("TOP-LEVEL DATA-FLOW GRAPH")
    print(bar)
    print(draw_pipeline(A))
    print("\n" + bar)
    print("ZImageTransformerBlock  (repeated unit)")
    print(bar)
    print(draw_block(A))
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
        description="Draw ZImageTransformer2DModel as a data-flow graph "
                    "(ASCII to stdout + a local self-contained HTML/SVG file).")
    ap.add_argument("--self-test", action="store_true",
                    help="render from a baked-in dims snapshot (no model load)")
    ap.add_argument("--html", default="z_image_arch.html",
                    help="path for the local HTML/SVG export (default: z_image_arch.html)")
    ap.add_argument("--no-html", action="store_true", help="skip the HTML/SVG export")
    ap.add_argument("--mermaid", metavar="PATH", default="z_image_arch.mmd",
                    help="path for the Mermaid .mmd export (default: z_image_arch.mmd)")
    ap.add_argument("--no-mermaid", action="store_true", help="skip the Mermaid export")
    args = ap.parse_args()

    html_path = None if args.no_html else args.html
    mermaid_path = None if args.no_mermaid else args.mermaid

    if args.self_test:
        _emit(_SELF_TEST_INFO, html_path, mermaid_path)
        return

    import torch
    from diffusers import ZImageTransformer2DModel

    print(f"Loading transformer from {MODEL_ID}/transformer ...")
    model = ZImageTransformer2DModel.from_pretrained(
        MODEL_ID, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    _emit(build_arch_info(model), html_path, mermaid_path)


if __name__ == "__main__":
    main()
