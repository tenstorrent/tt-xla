#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Render a TTNN codegen graph's main.py as an interactive dataflow graph.

Produces a single self-contained HTML file (no external libraries, no CDN, works
offline and inside sandboxed viewers) with an SVG dataflow DAG:

  * plumbing (to_device / to_layout / from_device / to_memory_config / typecast /
    reshape) is collapsed by default so the compute backbone stands out; edges are
    rewired through the folded nodes. Disable with --no-collapse.
  * left-to-right flow (longest-path depth) with weight/input leaves stacked above
    their consumer, so the backbone stays compact instead of a tall single column;
  * labeled stage bands (conv / pool / linear / norm / attention / reshape /
    elementwise / embed) behind the graph. Disable with --no-stages.
  * each node shows op + var + result shape (shapes from the sibling ttnn.mlir);
  * pan (drag), zoom (wheel), hover to highlight a node's edges, click to pin its
    full ancestor+descendant lineage, and a live op/var filter.

Usage:
    codegen_trace_graph.py <path>/main.py [-o out.html] [--mlir ttnn.mlir]
                           [--no-collapse] [--no-stages]
"""

import argparse
import html
import json
import os

from codegen_trace_table import build_trace, load_template

# op-family -> fill color (aligned with the table's accent colors)
OP_COLORS = {
    "matmul": "#0aa77a",
    "linear": "#0aa77a",
    "rms_norm": "#c58a1a",
    "conv2d": "#2d7dd2",
    "max_pool2d": "#2467b0",
    "embedding": "#8a2be2",
    "permute": "#7d8b99",
    "concat": "#7d8b99",
    "argmax": "#c0392b",
    "max": "#c0392b",
    "sum": "#c0392b",
    "relu": "#16a085",
    "silu": "#16a085",
    "exp": "#1f9e8f",
    "log": "#1f9e8f",
    "add": "#4b8b3b",
    "subtract": "#4b8b3b",
    "multiply": "#4b8b3b",
    "sin": "#4b8b3b",
    "cos": "#4b8b3b",
    "paged_scaled_dot_product_attention_decode": "#d35400",
    "paged_update_cache": "#e67e22",
    # plumbing (only shown with --no-collapse)
    "to_device": "#9aa0a6",
    "to_layout": "#9aa0a6",
    "from_device": "#9aa0a6",
    "to_memory_config": "#9aa0a6",
    "typecast": "#b0b6bb",
    "reshape": "#aeb6bd",
    "slice": "#aeb6bd",
}
DEFAULT_COLOR = "#5b6b7b"
INPUT_COLOR = "#b03a5b"

# ops folded away when --collapse (default). Shape/layout/movement only.
PLUMBING = {
    "to_device",
    "to_layout",
    "from_device",
    "to_memory_config",
    "typecast",
    "reshape",
}

# op -> logical stage family (for the background bands)
FAMILY = {
    "conv2d": "conv",
    "max_pool2d": "pool",
    "linear": "linear",
    "matmul": "linear",
    "rms_norm": "norm",
    "embedding": "embed",
    "paged_scaled_dot_product_attention_decode": "attention",
    "paged_update_cache": "attention",
    "permute": "reshape",
    "reshape": "reshape",
    "concat": "reshape",
    "slice": "reshape",
}
BAND_TINT = {
    "conv": "#e9f1fb",
    "pool": "#e0eafa",
    "linear": "#e7f7ef",
    "norm": "#faf1e0",
    "attention": "#fdece0",
    "embed": "#f1eafb",
    "reshape": "#eef1f3",
    "elementwise": "#f5f6f7",
}

COL_W = 210  # x spacing between depth layers
NODE_W = 178
NODE_H = 58
ROW_GAP = 84  # cross-axis spacing for branches at the same depth
IN_FIRST = NODE_H + 30  # gap from backbone up to the first stacked input
IN_GAP = NODE_H + 14  # gap between stacked inputs


def bare_op(op):
    return op.rsplit(".", 1)[-1] if op else "?"


def family_of(bare):
    return FAMILY.get(bare, "elementwise")


def build_graph(tr, collapse=True):
    """Return (node_list, edges, bands). Positions are baked into each node."""
    rows = tr["rows"]
    arg_info = tr["arg_info"]
    out_set = set(tr["outputs"])

    nodes = {}  # key -> dict

    def ensure_input(key):
        if key in nodes:
            return
        label, kind, shape = key, "input", ""
        if key.startswith("input["):
            idx = key[key.find("[") + 1 : key.find("]")]
            if idx.isdigit() and int(idx) < len(arg_info):
                info = arg_info[int(idx)]
                label = info["label"] or key
                kind = info["kind"] or "input"
                shape = info["shape"]
        nodes[key] = {
            "key": key,
            "bare": "input",
            "label": label,
            "kind": kind,
            "shape": shape,
            "is_input": True,
            "is_output": False,
            "preds": [],
        }

    order = []  # compute nodes in program order
    for step, out, op, inputs, dtype, layout, mem, shape in rows:
        for i in inputs:
            if i.startswith("input["):
                ensure_input(i)
        nodes[out] = {
            "key": out,
            "bare": bare_op(op),
            "label": out,
            "kind": "",
            "shape": shape,
            "is_input": False,
            "is_output": out in out_set,
            "preds": list(inputs),
        }
        order.append(out)

    def kept(n):
        return n["is_input"] or n["is_output"] or n["bare"] not in PLUMBING

    # fold plumbing: resolve each node's real (kept) predecessors in topo order
    if collapse:
        real = {}
        input_keys = [k for k in nodes if nodes[k]["is_input"]]
        for key in input_keys + order:  # sources first, then topo
            rp = []
            for p in nodes[key]["preds"]:
                if p not in nodes:
                    continue
                src = [p] if kept(nodes[p]) else real.get(p, [])
                for pp in src:
                    if pp not in rp:
                        rp.append(pp)
            real[key] = rp
        keep = [k for k in nodes if kept(nodes[k])]
        for k in keep:
            nodes[k]["preds"] = real[k]
        nodes = {k: nodes[k] for k in keep}
        order = [k for k in order if k in nodes]

    succ = {k: [] for k in nodes}
    for k in nodes:
        for p in nodes[k]["preds"]:
            if p in nodes:
                succ[p].append(k)

    # longest-path depth over non-input predecessors (program order is topo)
    for key in order:
        d = 0
        for p in nodes[key]["preds"]:
            if p in nodes and not nodes[p]["is_input"]:
                d = max(d, nodes[p].get("depth", 0) + 1)
        nodes[key]["depth"] = d

    by_depth = {}
    for key in order:
        by_depth.setdefault(nodes[key]["depth"], []).append(key)
    for d, keys in by_depth.items():
        c = len(keys)
        for i, key in enumerate(keys):
            nodes[key]["x"] = d * COL_W
            nodes[key]["y"] = (i - (c - 1) / 2) * ROW_GAP

    # place input leaves stacked above their (earliest) consumer
    slots = {}
    for key, n in nodes.items():
        if not n["is_input"]:
            continue
        cons = [s for s in succ[key] if not nodes[s]["is_input"]]
        if not cons:
            n["x"], n["y"] = 0, 0
            continue
        c = min(cons, key=lambda s: nodes[s]["x"])
        slot = slots.get(c, 0)
        slots[c] = slot + 1
        n["x"] = nodes[c]["x"]
        n["y"] = nodes[c]["y"] - IN_FIRST - slot * IN_GAP

    # ids: compute nodes first (in depth order), then inputs
    node_list, id_of = [], {}
    for d in sorted(by_depth):
        for key in by_depth[d]:
            nodes[key]["id"] = len(node_list)
            id_of[key] = len(node_list)
            node_list.append(nodes[key])
    for key, n in nodes.items():
        if n["is_input"]:
            n["id"] = len(node_list)
            id_of[key] = len(node_list)
            node_list.append(n)

    edges = []
    for key in nodes:
        for p in nodes[key]["preds"]:
            if p in nodes:
                edges.append((id_of[p], id_of[key]))

    # stage bands: merge consecutive depths sharing a family
    bands = []
    if by_depth:
        top = min(n["y"] for n in node_list)
        bottom = max(n["y"] + NODE_H for n in node_list)
        depths = sorted(by_depth)
        fam = {d: family_of(nodes[by_depth[d][0]]["bare"]) for d in depths}
        i = 0
        while i < len(depths):
            j = i
            while j + 1 < len(depths) and fam[depths[j + 1]] == fam[depths[i]]:
                j += 1
            d0, d1 = depths[i], depths[j]
            x0 = d0 * COL_W - 26
            x1 = d1 * COL_W + NODE_W + 26
            bands.append(
                {
                    "x": x0,
                    "w": x1 - x0,
                    "y": top - 30,
                    "h": (bottom - top) + 60,
                    "label": fam[depths[i]],
                    "tint": BAND_TINT.get(fam[depths[i]], "#f2f3f4"),
                }
            )
            i = j + 1

    return node_list, edges, bands


def color_for(n):
    if n["is_input"]:
        return INPUT_COLOR
    return OP_COLORS.get(n["bare"], DEFAULT_COLOR)


def edge_path(a, b):
    acx, acy = a["x"] + NODE_W / 2, a["y"] + NODE_H / 2
    bcx, bcy = b["x"] + NODE_W / 2, b["y"] + NODE_H / 2
    dx, dy = bcx - acx, bcy - acy
    if abs(dx) >= abs(dy):  # horizontal-ish
        if dx >= 0:
            x1, y1, x2, y2 = a["x"] + NODE_W, acy, b["x"], bcy
        else:
            x1, y1, x2, y2 = a["x"], acy, b["x"] + NODE_W, bcy
        mx = (x1 + x2) / 2
        return (
            f"M{x1:.0f},{y1:.0f} C{mx:.0f},{y1:.0f} {mx:.0f},{y2:.0f} {x2:.0f},{y2:.0f}"
        )
    if dy >= 0:  # vertical-ish
        x1, y1, x2, y2 = acx, a["y"] + NODE_H, bcx, b["y"]
    else:
        x1, y1, x2, y2 = acx, a["y"], bcx, b["y"] + NODE_H
    my = (y1 + y2) / 2
    return f"M{x1:.0f},{y1:.0f} C{x1:.0f},{my:.0f} {x2:.0f},{my:.0f} {x2:.0f},{y2:.0f}"


def trunc(s, n=24):
    return s if len(s) <= n else s[: n - 1] + "…"


def render(tr, main_path, out_path, collapse=True, stages=True):
    nodes, edges, bands = build_graph(tr, collapse=collapse)
    if not stages:
        bands = []
    esc = html.escape

    xs = [n["x"] for n in nodes] or [0]
    ys = [n["y"] for n in nodes] or [0]
    minx = min(xs) - 34
    top = min(ys)
    miny = top - 56
    w = (max(n["x"] + NODE_W for n in nodes) - minx) + 34 if nodes else 100
    h = (max(n["y"] + NODE_H for n in nodes) - miny) + 34 if nodes else 100

    band_svg = []
    for b in bands:
        band_svg.append(
            f'<g><rect class=band x="{b["x"]:.0f}" y="{b["y"]:.0f}" '
            f'width="{b["w"]:.0f}" height="{b["h"]:.0f}" fill="{b["tint"]}"/>'
            f'<text class=blab x="{b["x"] + 10:.0f}" y="{b["y"] + 16:.0f}">'
            f'{esc(b["label"].upper())}</text></g>'
        )

    edge_svg = []
    for s, t in edges:
        edge_svg.append(
            f'<path class=edge data-s="{s}" data-t="{t}" d="{edge_path(nodes[s], nodes[t])}"/>'
        )

    node_svg = []
    for n in nodes:
        title = esc(n["key"])
        if n["is_input"] and n["label"] != n["key"]:
            title += f'  ({esc(n["label"])})'
        if n["shape"]:
            title += f'  [{esc(n["shape"])}]'
        if n["is_input"]:
            l1, l2 = esc(trunc(n["label"], 22)), esc(n["key"])
        else:
            l1, l2 = esc(n["bare"]), esc(trunc(n["key"], 22))
        l3 = esc(n["shape"] or "")
        stroke = "#111" if n["is_output"] else "none"
        node_svg.append(
            f'<g class=node data-id="{n["id"]}" transform="translate({n["x"]:.0f},{n["y"]:.0f})">'
            f"<title>{title}</title>"
            f'<rect width="{NODE_W}" height="{NODE_H}" rx="7" fill="{color_for(n)}" '
            f'stroke="{stroke}" stroke-width="2.5"/>'
            f'<text class=nop x="10" y="19">{l1}</text>'
            f'<text class=nvar x="10" y="35">{l2}</text>'
            f'<text class=nshape x="10" y="50">{l3}</text>'
            f"</g>"
        )

    op_counts = {}
    for n in nodes:
        if not n["is_input"]:
            op_counts[n["bare"]] = op_counts.get(n["bare"], 0) + 1
    legend = " · ".join(
        f"{k}:{v}" for k, v in sorted(op_counts.items(), key=lambda x: -x[1])
    )
    mode = "collapsed" if collapse else "full"
    graph_name = os.path.basename(os.path.dirname(os.path.abspath(main_path)))

    doc = (
        load_template("codegen_trace_graph_template.html")
        .replace("__VIEWBOX__", f"{minx:.0f} {miny:.0f} {w:.0f} {h:.0f}")
        .replace("<!--__NAME__-->", esc(graph_name))
        .replace(
            "<!--__STATS__-->", f"{len(nodes)} nodes · {len(edges)} edges · {mode}"
        )
        .replace("<!--__LEGEND__-->", esc(legend))
        .replace("<!--__BANDS__-->", "".join(band_svg))
        .replace("<!--__EDGES__-->", "".join(edge_svg))
        .replace("<!--__NODES__-->", "".join(node_svg))
        .replace("/*__EDGES__*/", json.dumps(edges))
    )

    with open(out_path, "w") as f:
        f.write(doc)
    return len(nodes), len(edges)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("-o", "--out")
    ap.add_argument("--mlir")
    ap.add_argument(
        "--no-collapse",
        action="store_true",
        help="keep plumbing ops (to_device/to_layout/reshape/...) as nodes",
    )
    ap.add_argument(
        "--no-stages", action="store_true", help="omit the labeled stage bands"
    )
    args = ap.parse_args()

    tr = build_trace(args.path, args.mlir)
    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.path)), "graph.html"
    )
    n, e = render(
        tr,
        args.path,
        out_path,
        collapse=not args.no_collapse,
        stages=not args.no_stages,
    )
    note = "" if tr["has_shapes"] else "  (no ttnn.mlir -> shapes blank)"
    print(f"wrote {out_path}  ({n} nodes, {e} edges){note}")


if __name__ == "__main__":
    main()
