#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Render a TTNN codegen graph's main.py as a plain HTML table of computation steps.

Each row is one ttnn op in execution order, with its output tensor, inputs,
result shape (sourced from the sibling ttnn.mlir), dtype/layout/memory, and the
tensors it deallocates. Produces a self-contained, dependency-free HTML file with
a live filter box.

Usage:
    codegen_trace_table.py <path/to/main.py> [-o out.html] [--mlir ttnn.mlir]
                           [--no-plumbing]

Shapes are pulled from ttnn.mlir (auto-detected as a sibling of main.py). main.py
and ttnn.mlir are both linear traces of the same IR, so the k-th occurrence of an
op in one matches the k-th in the other; shape-preserving ops (to_device,
to_layout, to_memory_config, paged_update_cache) inherit their input's shape.
"""
import argparse
import ast
import html
import os
import re
from collections import defaultdict

# main.py op name -> MLIR op name
NAME_MAP = {
    "slice": "slice_static",
    "paged_scaled_dot_product_attention_decode": "paged_scaled_dot_product_attention_decode",
    "paged_update_cache": "paged_update_cache",
}
# ops whose result shape == shape of their first tensor operand (no MLIR result)
PASSTHROUGH = {"to_device", "to_layout", "to_memory_config", "paged_update_cache"}
# pure shape/layout/movement ops hidden by --no-plumbing (matches the graph tool)
PLUMBING = {
    "to_device",
    "to_layout",
    "from_device",
    "to_memory_config",
    "typecast",
    "reshape",
}


def pretty_arg_name(raw):
    """Turn a raw ttir.name into a short readable label."""
    if not raw:
        return ""
    if "kv_cache" in raw:
        lm = re.search(r"layers_(\d+)", raw)
        km = re.search(r"(kv_cache_\d+)", raw)
        if lm and km:
            return f"layers.{lm.group(1)}.{km.group(1)}"
    s = re.sub(r"^L__self___(model_)+", "", raw)  # drop L__self___model_model_
    s = re.sub(r"^model_", "", s)
    return s


def parse_mlir_shapes(path):
    """Return (arg_info, op_shapes) where op_shapes maps mlir_op_name -> list of
    result shape-strings in program order, and arg_info is an ordered list (by %argN)
    of dicts: {shape, name, label, kind}."""
    text = open(path).read()
    # func signature: capture ordered %argN with tensor type + attribute block
    sig = re.search(r"func\.func @main\((.*?)\)\s", text, re.S)
    arg_info = []
    if sig:
        for m in re.finditer(
            r"%arg\d+:\s*tensor<([0-9x]+)x([a-z0-9_]+),[^{]*\{([^}]*)\}", sig.group(1)
        ):
            attrs = m.group(3)
            nm = re.search(r'ttir\.name = "([^"]*)"', attrs)
            name = nm.group(1) if nm else ""
            kd = re.search(r"argument_type<(\w+)>", attrs)
            kind = kd.group(1) if kd else ""
            if "kv_cache" in attrs:
                kind = "kv_cache"
            arg_info.append(
                {
                    "shape": f"{m.group(1)}·{m.group(2)}",
                    "name": name,
                    "label": pretty_arg_name(name),
                    "kind": kind,
                }
            )

    op_shapes = defaultdict(list)
    # result-producing ops:  %N = "ttnn.NAME"(...) ... -> tensor<DIMSxDTYPE,
    for m in re.finditer(
        r'%\d+ = "ttnn\.([a-z_]+)"\(.*?->\s*tensor<([0-9x]+)x([a-z0-9_]+)[,>]',
        text,
    ):
        op_shapes[m.group(1)].append(f"{m.group(2)}·{m.group(3)}")
    return arg_info, op_shapes


def unparse_short(node, maxlen=48):
    try:
        s = ast.unparse(node)
    except Exception:
        s = "<?>"
    s = " ".join(s.split())
    return s if len(s) <= maxlen else s[: maxlen - 1] + "…"


def call_name(call):
    """Return e.g. 'ttnn.matmul' for a Call node, or None."""
    f = call.func
    parts = []
    while isinstance(f, ast.Attribute):
        parts.append(f.attr)
        f = f.value
    if isinstance(f, ast.Name):
        parts.append(f.id)
    return ".".join(reversed(parts)) if parts else None


def referenced_vars(node, defined):
    """Collect input references: prior variables and input[i] subscripts."""
    refs = []
    seen = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name):
            key = ast.unparse(n)
            if n.value.id == "input" and key not in seen:
                seen.add(key)
                refs.append(key)
        elif isinstance(n, ast.Name) and n.id in defined and n.id not in seen:
            seen.add(n.id)
            refs.append(n.id)
    return refs


def kw_summary(call):
    """Pull dtype / layout / memory kind out of keyword args."""
    dtype = layout = mem = ""
    for kw in call.keywords:
        val = ast.unparse(kw.value) if kw.value is not None else ""
        if kw.arg == "dtype":
            dtype = val.replace("ttnn.DataType.", "")
        elif kw.arg == "memory_config":
            if "L1" in val:
                mem = "L1"
            elif "DRAM" in val:
                mem = "DRAM"
    # layout appears as positional (ttnn.Layout.TILE) or keyword
    txt = ast.unparse(call)
    if "Layout.TILE" in txt:
        layout = "TILE"
    elif "Layout.ROW_MAJOR" in txt:
        layout = "ROW_MAJOR"
    return dtype, layout, mem


def build_trace(main_path, mlir_path=None):
    """Parse a codegen main.py (and its sibling ttnn.mlir) into an ordered op trace.

    Returns a dict with:
      rows       : list of [step, out, op, inputs, dtype, layout, mem, shape]
      arg_info   : ordered per-%argN metadata (shape/name/label/kind)
      outputs    : vars returned by forward()
      freed_at   : {step: [vars deallocated right after that step]}
      var_shapes : {var: shape-string}
      mlir_path, has_shapes, shape_hits, shape_misses
    """
    tree = ast.parse(open(main_path).read())
    fn = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "forward"
    )

    if mlir_path is None:
        mlir_path = os.path.join(
            os.path.dirname(os.path.abspath(main_path)), "ttnn.mlir"
        )
    arg_info, op_shapes = ([], {})
    if os.path.exists(mlir_path):
        arg_info, op_shapes = parse_mlir_shapes(mlir_path)
    op_cursor = defaultdict(int)  # mlir_op_name -> next occurrence index to consume
    var_shapes = {}  # main.py var -> shape string
    shape_hits = shape_misses = 0

    def bare_op(op):
        return op.rsplit(".", 1)[-1] if op else op

    def first_operand_shape(inputs):
        for i in inputs:
            m = re.fullmatch(r"input\[(\d+)\]", i)
            if m:
                idx = int(m.group(1))
                if idx < len(arg_info):
                    return arg_info[idx]["shape"]
            elif i in var_shapes:
                return var_shapes[i]
        return ""

    def resolve_shape(op, out, inputs):
        b = bare_op(op)
        if b in PASSTHROUGH:
            return first_operand_shape(inputs)
        mlir_name = NAME_MAP.get(b, b)
        lst = op_shapes.get(mlir_name)
        if lst is not None and op_cursor[mlir_name] < len(lst):
            s = lst[op_cursor[mlir_name]]
            op_cursor[mlir_name] += 1
            return s
        return first_operand_shape(inputs)  # fallback

    defined = set()
    rows = []
    deallocs = {}  # var -> step at which it is freed
    outputs = []

    step = 0
    for stmt in fn.body:
        if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
            op = call_name(stmt.value) or "?"
            target = stmt.targets[0]
            out = target.id if isinstance(target, ast.Name) else ast.unparse(target)
            inputs = referenced_vars(stmt.value, defined)
            dtype, layout, mem = kw_summary(stmt.value)
            shape = resolve_shape(op, out, inputs)
            if shape:
                shape_hits += 1
                var_shapes[out] = shape
            else:
                shape_misses += 1
            step += 1
            rows.append([step, out, op, inputs, dtype, layout, mem, shape])
            defined.add(out)
        elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            if call_name(stmt.value) == "ttnn.deallocate" and stmt.value.args:
                a = stmt.value.args[0]
                if isinstance(a, ast.Name):
                    deallocs[a.id] = step  # freed right after current step
        elif isinstance(stmt, ast.Return):
            outputs = referenced_vars(stmt.value, defined)

    freed_at = {}  # step -> [vars freed just after that step]
    for var, st in deallocs.items():
        freed_at.setdefault(st, []).append(var)

    return {
        "rows": rows,
        "arg_info": arg_info,
        "outputs": outputs,
        "freed_at": freed_at,
        "var_shapes": var_shapes,
        "mlir_path": mlir_path,
        "has_shapes": bool(op_shapes),
        "shape_hits": shape_hits,
        "shape_misses": shape_misses,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("-o", "--out")
    ap.add_argument(
        "--mlir",
        help="ttnn.mlir to source tensor shapes from " "(default: sibling ttnn.mlir)",
    )
    ap.add_argument(
        "--no-plumbing",
        action="store_true",
        help="hide plumbing ops (to_device/to_layout/from_device/"
        "to_memory_config/typecast/reshape); rewire kept ops' "
        "inputs through the folded ops",
    )
    args = ap.parse_args()

    tr = build_trace(args.path, args.mlir)
    rows = tr["rows"]
    arg_info = tr["arg_info"]
    outputs = tr["outputs"]
    freed_at = tr["freed_at"]
    mlir_path = tr["mlir_path"]
    op_shapes = tr["has_shapes"]
    shape_hits, shape_misses = tr["shape_hits"], tr["shape_misses"]
    out_set = set(outputs)

    # optional: hide plumbing ops, rewiring kept ops' inputs through the folded
    # plumbing to the nearest real (non-plumbing) producer var.
    def bare(op):
        return op.rsplit(".", 1)[-1] if op else op

    producer = {r[1]: r for r in rows}  # out var -> row
    plumb_vars = {r[1] for r in rows if bare(r[2]) in PLUMBING}
    display = rows
    if args.no_plumbing:
        memo = {}

        def kept_srcs(var, seen=None):
            if var in memo:
                return memo[var]
            seen = seen or set()
            if var in seen:
                return []
            seen.add(var)
            r = producer.get(var)
            if r is None or bare(r[2]) not in PLUMBING:
                res = [var]  # leaf (input[i]) or real producer
            else:
                res = []
                for inp in r[3]:
                    for s in kept_srcs(inp, seen):
                        if s not in res:
                            res.append(s)
            memo[var] = res
            return res

        display = []
        for step, out, op, inputs, dtype, layout, mem, shape in rows:
            if bare(op) in PLUMBING:
                continue
            new_inputs = []
            for inp in inputs:
                for s in kept_srcs(inp):
                    if s not in new_inputs:
                        new_inputs.append(s)
            display.append([step, out, op, new_inputs, dtype, layout, mem, shape])

    # ---- emit HTML ----
    op_counts = {}
    for r in display:
        op_counts[r[2]] = op_counts.get(r[2], 0) + 1

    def esc(x):
        return html.escape(str(x))

    parts = []
    parts.append(
        """<!doctype html><meta charset=utf-8>
<title>Compute trace</title>
<style>
 body{font:13px/1.4 ui-monospace,Menlo,Consolas,monospace;margin:1rem;color:#111;background:#fff}
 h1{font-size:16px} .meta{color:#555;margin-bottom:.6rem}
 input{font:inherit;padding:4px 6px;width:280px;margin-bottom:.6rem}
 table{border-collapse:collapse;width:100%}
 th,td{border:1px solid #ddd;padding:3px 6px;text-align:left;vertical-align:top}
 th{position:sticky;top:0;background:#f4f4f4;cursor:default}
 tbody tr:nth-child(even){background:#fafafa}
 td.num{color:#999;text-align:right}
 .op{font-weight:600}
 .in{color:#06c} .free{color:#c00} .out{background:#fff6d5}
 .argname{font-style:italic}
 .k-parameter{color:#a0740a} .k-kv_cache{color:#8a2be2} .k-constant{color:#0a8a8a}
 .k-input{color:#c0392b}
 .shape{color:#093;white-space:nowrap} .stype{color:#787} .noshape{color:#bbb}
 .t-matmul{color:#0a7} .t-rms_norm{color:#a60} .t-slice,.t-reshape{color:#777}
</style>
<h1>Computation trace</h1>"""
    )
    shape_note = (
        f" &middot; shapes: {shape_hits}/{len(rows)} resolved"
        + (f" ({shape_misses} unknown)" if shape_misses else "")
        + f" from {esc(os.path.basename(mlir_path))}"
        if op_shapes
        else " &middot; no MLIR shapes"
    )
    op_count_note = (
        f"{len(display)} ops (of {len(rows)}, plumbing hidden)"
        if args.no_plumbing
        else f"{len(rows)} ops"
    )
    parts.append(
        f"<div class=meta>{esc(os.path.abspath(args.path))} &middot; "
        f"{op_count_note} &middot; {len(outputs)} outputs{shape_note}<br>"
        + " &middot; ".join(
            f"{esc(k.replace('ttnn.',''))}:{v}"
            for k, v in sorted(op_counts.items(), key=lambda x: -x[1])
        )
        + "</div>"
    )
    parts.append(
        '<input id=f placeholder="filter rows (op name, var…)" '
        'oninput="var q=this.value.toLowerCase();'
        "for(var r of document.querySelectorAll('tbody tr'))"
        "r.style.display=r.textContent.toLowerCase().includes(q)?'':'none'\">"
    )
    parts.append(
        "<table><thead><tr>"
        "<th>#</th><th>inputs</th><th>op</th><th>layout</th><th>output</th>"
        "<th>dims</th><th>type</th>"
        "<th>dtype</th><th>mem</th><th>frees</th>"
        "</tr></thead><tbody>"
    )

    def render_input(i):
        m = re.fullmatch(r"input\[(\d+)\]", i)
        if m and int(m.group(1)) < len(arg_info):
            info = arg_info[int(m.group(1))]
            label, kind = info["label"], info["kind"]
            title = esc(info["name"] or "") + (f" [{esc(kind)}]" if kind else "")
            name_html = (
                f' <span class="argname k-{esc(kind)}">{esc(label)}</span>'
                if label
                else ""
            )
            return f'<span class=in title="{title}">{esc(i)}</span>{name_html}'
        return f"<span class=in>{esc(i)}</span>"

    for step, out, op, inputs, dtype, layout, mem, shape in display:
        opshort = op.replace("ttnn.", "")
        ins = ", ".join(render_input(i) for i in inputs)
        frees = ", ".join(
            f"<span class=free>{esc(v)}</span>"
            for v in freed_at.get(step, [])
            if not (args.no_plumbing and v in plumb_vars)
        )
        out_cls = " class=out" if out in out_set else ""
        if shape:
            dims, _, stype = shape.partition("·")
            dims_html = f"<span class=shape>{esc(dims)}</span>"
            stype_html = f"<span class=stype>{esc(stype)}</span>"
        else:
            dims_html = stype_html = "<span class=noshape>?</span>"
        parts.append(
            f"<tr><td class=num>{step}</td>"
            f"<td>{ins}</td>"
            f'<td class="op t-{esc(opshort)}">{esc(opshort)}</td>'
            f"<td>{esc(layout)}</td>"
            f"<td{out_cls}>{esc(out)}</td>"
            f"<td>{dims_html}</td><td>{stype_html}</td>"
            f"<td>{esc(dtype)}</td>"
            f"<td>{esc(mem)}</td><td>{frees}</td></tr>"
        )
    parts.append("</tbody></table>")

    out_path = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.path)), "trace.html"
    )
    with open(out_path, "w") as f:
        f.write("".join(parts))
    print(f"wrote {out_path}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
