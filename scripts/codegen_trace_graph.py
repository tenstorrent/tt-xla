#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Render a TTNN codegen graph's main.py as an interactive dataflow graph.

Produces a single self-contained HTML file (no external libraries, no CDN, works
offline and inside sandboxed viewers) with an SVG dataflow DAG:

  * nodes are ttnn ops (and input[i] leaves), colored by op family, labeled with
    op name + result shape (shapes sourced from the sibling ttnn.mlir);
  * a top-down layered layout (longest-path depth) with weight/constant leaves
    pulled down next to their first consumer to keep edges short;
  * pan (drag), zoom (wheel), hover-to-highlight a node's direct edges, click to
    pin a node's full ancestor+descendant lineage, and a live op-name filter.

Usage:
    codegen_trace_graph.py <path/to/main.py> [-o out.html] [--mlir ttnn.mlir]
"""

import argparse
import html
import json
import os

from codegen_trace_table import build_trace


# op-family -> fill color (kept close to the table's accent colors)
OP_COLORS = {
    "matmul": "#0aa77a", "linear": "#0aa77a",
    "rms_norm": "#c58a1a", "conv2d": "#2d7dd2", "max_pool2d": "#2d7dd2",
    "embedding": "#8a2be2",
    "to_device": "#9aa0a6", "to_layout": "#9aa0a6", "from_device": "#9aa0a6",
    "to_memory_config": "#9aa0a6", "typecast": "#b0b6bb",
    "slice": "#7d8b99", "reshape": "#7d8b99", "permute": "#7d8b99",
    "concat": "#7d8b99",
    "argmax": "#c0392b", "max": "#c0392b", "sum": "#c0392b",
    "paged_scaled_dot_product_attention_decode": "#d35400",
    "paged_update_cache": "#e67e22",
}
DEFAULT_COLOR = "#5b6b7b"
INPUT_COLOR = "#b03a5b"

COL_W = 210     # horizontal spacing within a layer
ROW_H = 78      # vertical spacing between layers
NODE_W = 176
NODE_H = 44


def bare_op(op):
    return op.rsplit(".", 1)[-1] if op else "?"


def build_graph(tr):
    """Turn a trace dict into (nodes, edges) with computed x/y positions."""
    rows = tr["rows"]
    arg_info = tr["arg_info"]
    out_set = set(tr["outputs"])

    # node registry keyed by var name (and input[i] leaf keys)
    nodes = {}   # key -> dict

    def ensure_input_node(key):
        if key in nodes:
            return
        label, kind, shape = key, "input", ""
        m = key[key.find("[") + 1:key.find("]")] if key.startswith("input[") else None
        if m is not None and m.isdigit() and int(m) < len(arg_info):
            info = arg_info[int(m)]
            label = info["label"] or key
            kind = info["kind"] or "input"
            shape = info["shape"]
        nodes[key] = {
            "key": key, "op": "input", "bare": "input", "label": label,
            "kind": kind, "shape": shape, "is_input": True, "is_output": False,
            "preds": [], "depth": 0,
        }

    order = []  # keys in program order (compute nodes only)
    for step, out, op, inputs, dtype, layout, mem, shape in rows:
        for i in inputs:
            if i.startswith("input["):
                ensure_input_node(i)
        nodes[out] = {
            "key": out, "op": op, "bare": bare_op(op), "label": out,
            "kind": "", "shape": shape, "is_input": False,
            "is_output": out in out_set, "preds": list(inputs), "depth": 0,
        }
        order.append(out)

    # longest-path depth in program order (rows are already topologically sorted)
    for key in order:
        n = nodes[key]
        d = 0
        for p in n["preds"]:
            if p in nodes:
                d = max(d, nodes[p]["depth"] + 1)
        n["depth"] = d

    # collect successors, then pull leaf inputs down to (min successor depth - 1)
    succ = {k: [] for k in nodes}
    for key in order:
        for p in nodes[key]["preds"]:
            if p in nodes:
                succ[p].append(key)
    for n in nodes.values():
        if n["is_input"] and succ[n["key"]]:
            n["depth"] = max(0, min(nodes[s]["depth"] for s in succ[n["key"]]) - 1)

    # assign a column index within each depth (compute chain first, then leaves)
    by_depth = {}
    for key in order:                                   # compute nodes first
        by_depth.setdefault(nodes[key]["depth"], []).append(key)
    for key, n in nodes.items():                        # then input leaves
        if n["is_input"]:
            by_depth.setdefault(n["depth"], []).append(key)

    id_of = {}
    node_list = []
    for depth in sorted(by_depth):
        for col, key in enumerate(by_depth[depth]):
            n = nodes[key]
            nid = len(node_list)
            id_of[key] = nid
            n["id"] = nid
            n["x"] = col * COL_W
            n["y"] = depth * ROW_H
            node_list.append(n)

    edges = []
    for key in order:
        for p in nodes[key]["preds"]:
            if p in nodes:
                edges.append((id_of[p], id_of[key]))

    return node_list, edges


def color_for(n):
    if n["is_input"]:
        return INPUT_COLOR
    return OP_COLORS.get(n["bare"], DEFAULT_COLOR)


def render(tr, main_path, out_path):
    nodes, edges = build_graph(tr)
    esc = html.escape

    max_x = max((n["x"] for n in nodes), default=0) + NODE_W + 40
    max_y = max((n["y"] for n in nodes), default=0) + NODE_H + 40

    # SVG edges
    edge_svg = []
    for s, t in edges:
        a, b = nodes[s], nodes[t]
        x1, y1 = a["x"] + NODE_W / 2, a["y"] + NODE_H
        x2, y2 = b["x"] + NODE_W / 2, b["y"]
        my = (y1 + y2) / 2
        edge_svg.append(
            f'<path class=edge data-s="{s}" data-t="{t}" '
            f'd="M{x1:.0f},{y1:.0f} C{x1:.0f},{my:.0f} {x2:.0f},{my:.0f} {x2:.0f},{y2:.0f}"/>'
        )

    # SVG nodes
    node_svg = []
    for n in nodes:
        title = esc(n["key"])
        if n["is_input"] and n["label"] != n["key"]:
            title += f'  ({esc(n["label"])})'
        if n["shape"]:
            title += f'  [{esc(n["shape"])}]'
        stroke = "#111" if n["is_output"] else "none"
        label = esc(n["label"] if n["is_input"] else n["bare"])
        if len(label) > 24:
            label = label[:23] + "…"
        sub = esc(n["shape"] or "")
        node_svg.append(
            f'<g class=node data-id="{n["id"]}" transform="translate({n["x"]},{n["y"]})">'
            f'<title>{title}</title>'
            f'<rect width="{NODE_W}" height="{NODE_H}" rx="6" '
            f'fill="{color_for(n)}" stroke="{stroke}" stroke-width="2"/>'
            f'<text class=nlab x="8" y="18">{label}</text>'
            f'<text class=nsub x="8" y="35">{sub}</text>'
            f'</g>'
        )

    # adjacency for lineage highlighting (built in JS from these)
    edge_json = json.dumps(edges)

    op_counts = {}
    for n in nodes:
        if not n["is_input"]:
            op_counts[n["bare"]] = op_counts.get(n["bare"], 0) + 1
    legend = " · ".join(f"{k}:{v}" for k, v in sorted(op_counts.items(), key=lambda x: -x[1]))

    doc = f"""<!doctype html><meta charset=utf-8>
<title>Dataflow graph</title>
<style>
 html,body{{margin:0;height:100%;font:13px ui-monospace,Menlo,Consolas,monospace;background:#fff;color:#111}}
 #bar{{padding:6px 10px;border-bottom:1px solid #ddd;display:flex;gap:12px;align-items:center;flex-wrap:wrap}}
 #bar b{{font-weight:600}} #bar .leg{{color:#666;font-size:11px}}
 input{{font:inherit;padding:3px 6px;width:220px}}
 #wrap{{position:absolute;top:0;left:0;right:0;bottom:0;padding-top:38px;box-sizing:border-box}}
 svg{{width:100%;height:100%;cursor:grab;user-select:none;background:#fbfbfc}}
 svg.grab{{cursor:grabbing}}
 .edge{{fill:none;stroke:#bcc4cc;stroke-width:1.5}}
 .node text{{pointer-events:none;fill:#fff}}
 .nlab{{font-weight:600}} .nsub{{fill:#eef;font-size:10px;opacity:.85}}
 .edge.hi{{stroke:#111;stroke-width:2.5}}
 .edge.dim,.node.dim{{opacity:.12}}
 .node.hi rect{{stroke:#111;stroke-width:3}}
</style>
<div id=bar>
 <b>{esc(os.path.basename(os.path.dirname(os.path.abspath(main_path))))}</b>
 <span>{len(nodes)} nodes · {len(edges)} edges</span>
 <input id=f placeholder="filter by op / var…">
 <span id=hint style="color:#888">drag=pan · wheel=zoom · hover=edges · click=lineage · esc=reset</span>
 <span class=leg>{esc(legend)}</span>
</div>
<div id=wrap>
<svg id=svg viewBox="-20 -20 {max_x} {max_y}">
<g id=scene>
<g id=edges>{''.join(edge_svg)}</g>
<g id=nodes>{''.join(node_svg)}</g>
</g>
</svg>
</div>
<script>
const EDGES = {edge_json};
const svg = document.getElementById('svg');
const succ = {{}}, pred = {{}};
for (const [s,t] of EDGES) {{ (succ[s]=succ[s]||[]).push(t); (pred[t]=pred[t]||[]).push(s); }}
const nodeEls = [...document.querySelectorAll('.node')];
const edgeEls = [...document.querySelectorAll('.edge')];
const nodeById = {{}}; nodeEls.forEach(e => nodeById[e.dataset.id]=e);

// ---- pan / zoom via viewBox ----
let vb = svg.getAttribute('viewBox').split(' ').map(Number); // x,y,w,h
function setVB(){{ svg.setAttribute('viewBox', vb.join(' ')); }}
svg.addEventListener('wheel', e => {{
  e.preventDefault();
  const r = svg.getBoundingClientRect();
  const mx = vb[0] + (e.clientX-r.left)/r.width*vb[2];
  const my = vb[1] + (e.clientY-r.top)/r.height*vb[3];
  const k = e.deltaY>0 ? 1.1 : 1/1.1;
  vb[0]=mx-(mx-vb[0])*k; vb[1]=my-(my-vb[1])*k; vb[2]*=k; vb[3]*=k; setVB();
}}, {{passive:false}});
let drag=null;
svg.addEventListener('mousedown', e => {{ if(e.target.closest('.node'))return;
  drag={{x:e.clientX,y:e.clientY,vx:vb[0],vy:vb[1]}}; svg.classList.add('grab'); }});
addEventListener('mousemove', e => {{ if(!drag)return;
  const r=svg.getBoundingClientRect();
  vb[0]=drag.vx-(e.clientX-drag.x)/r.width*vb[2];
  vb[1]=drag.vy-(e.clientY-drag.y)/r.height*vb[3]; setVB(); }});
addEventListener('mouseup', ()=>{{ drag=null; svg.classList.remove('grab'); }});

// ---- highlight ----
function clear(){{ [...nodeEls,...edgeEls].forEach(e=>e.classList.remove('hi','dim')); }}
function highlightNeighbors(id){{
  clear();
  const keep=new Set([id]);
  (succ[id]||[]).forEach(t=>keep.add(t)); (pred[id]||[]).forEach(s=>keep.add(s));
  nodeEls.forEach(e=>e.classList.toggle('dim', !keep.has(+e.dataset.id)));
  nodeById[id].classList.add('hi');
  edgeEls.forEach(e=>{{ const s=+e.dataset.s,t=+e.dataset.t;
    if(s===id||t===id){{e.classList.add('hi');}} else {{e.classList.add('dim');}} }});
}}
function lineage(id){{
  const up=new Set(), dn=new Set(); let st=[id];
  while(st.length){{const n=st.pop(); (pred[n]||[]).forEach(p=>{{if(!up.has(p)){{up.add(p);st.push(p);}}}});}}
  st=[id]; while(st.length){{const n=st.pop(); (succ[n]||[]).forEach(s=>{{if(!dn.has(s)){{dn.add(s);st.push(s);}}}});}}
  const keep=new Set([id,...up,...dn]); clear();
  nodeEls.forEach(e=>{{const i=+e.dataset.id; if(!keep.has(i))e.classList.add('dim'); else if(i===id)e.classList.add('hi');}});
  edgeEls.forEach(e=>{{const s=+e.dataset.s,t=+e.dataset.t;
    if(keep.has(s)&&keep.has(t))e.classList.add('hi'); else e.classList.add('dim');}});
}}
let pinned=null;
nodeEls.forEach(e=>{{
  const id=+e.dataset.id;
  e.addEventListener('mouseenter',()=>{{ if(pinned===null) highlightNeighbors(id); }});
  e.addEventListener('mouseleave',()=>{{ if(pinned===null) clear(); }});
  e.addEventListener('click',ev=>{{ ev.stopPropagation();
    if(pinned===id){{pinned=null;clear();}} else {{pinned=id;lineage(id);}} }});
}});
svg.addEventListener('click',()=>{{ if(pinned!==null){{pinned=null;clear();}} }});
addEventListener('keydown',e=>{{ if(e.key==='Escape'){{pinned=null;clear();}} }});

// ---- filter ----
document.getElementById('f').addEventListener('input',function(){{
  const q=this.value.toLowerCase(); pinned=null;
  nodeEls.forEach(e=>{{
    const t=(e.querySelector('title').textContent+' '+e.textContent).toLowerCase();
    e.classList.toggle('dim', q!=='' && !t.includes(q));
  }});
  edgeEls.forEach(e=>e.classList.remove('hi','dim'));
}});
</script>"""

    with open(out_path, "w") as f:
        f.write(doc)
    return len(nodes), len(edges)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("-o", "--out")
    ap.add_argument("--mlir")
    args = ap.parse_args()

    tr = build_trace(args.path, args.mlir)
    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.path)), "graph.html")
    n, e = render(tr, args.path, out_path)
    note = "" if tr["has_shapes"] else "  (no ttnn.mlir -> shapes blank)"
    print(f"wrote {out_path}  ({n} nodes, {e} edges){note}")


if __name__ == "__main__":
    main()
