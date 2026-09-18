#!/usr/bin/env python
"""Turn a Chisel results .jsonl into a PASS/FAIL verdict against a PCC floor.

    python analyze_chisel.py chisel_results/<file>.jsonl [--floor 0.99]

Reports worst isolated PCC per op, and separates the ops whose Chisel number is
NOT a valid measurement so they are never mistaken for failures:

  MoE chain   sparse_matmul / all_to_all_combine / all_to_all_dispatch /
              moe_expert_token_remap / topk -- the goldens model a dense matmul,
              the wrong output shape, a permutation-dependent layout, the wrong
              dtype, and categorical indices respectively. Covered instead by
              tests/torch/models/diffusiongemma/test_diffusiongemma_moe_block.py.
  ttnn.full   one fill_value=0x7FC00000 (NaN) sentinel; metrics.py:61 hard-returns
              0.0 for all-NaN vs non-NaN.

It also prints the mid-graph golden_promoted count. That number is the structural
limit of isolation mode: each op's golden is computed from that op's ACTUAL DEVICE
INPUTS, so a corrupted upstream value is inherited and the op still passes.
Isolation validates arithmetic given inputs, never the inputs themselves -- which
is why the per-layer cumulative probe (independent CPU reference) is required
alongside it, not merely as corroboration.
"""

import argparse
import json
from collections import Counter, defaultdict

MOE = {
    "ttnn.sparse_matmul",
    "ttnn.all_to_all_combine",
    "ttnn.all_to_all_dispatch",
    "ttnn.moe_expert_token_remap",
    "ttnn.topk",
}
ARTIFACT = {"ttnn.full"}
NON_ARITH = {
    "ttnn.deallocate",
    "ttnn.get_device",
    "func.call",
    "ttcore.load_cached",
    "ttnn.constant",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--floor", type=float, default=0.99)
    a = ap.parse_args()

    recs = [json.loads(l) for l in open(a.path)]
    worst, counts, status = {}, Counter(), defaultdict(Counter)
    for r in recs:
        op = r["op"]
        status[op][r["payload"].get("status")] += 1
        if r["check"] == "numerics" and r["payload"].get("pcc") is not None:
            counts[op] += 1
            worst[op] = min(worst.get(op, 1.0), r["payload"]["pcc"])

    checked = {k: v for k, v in worst.items() if k not in MOE | ARTIFACT}
    fails = {k: v for k, v in checked.items() if v < a.floor}

    print(f"{a.path}   {len(recs)} records, {len(worst)} ops with numerics\n")
    print(f"{'op':34}{'n':>8}{'worst PCC':>13}")
    for op, w in sorted(checked.items(), key=lambda kv: kv[1])[:12]:
        print(f"{op:34}{counts[op]:>8}{w:>13.8f}")

    print(f"\nnot validly measurable by Chisel (covered elsewhere):")
    for op in sorted(MOE | ARTIFACT):
        if op in worst:
            where = "MoE block test" if op in MOE else "NaN sentinel artifact"
            print(f"  {op:32} worst={worst[op]:>11.6f}   -> {where}")
    present = [o for o in NON_ARITH if o in status]
    if present:
        print(f"  non-arithmetic (no numerics possible): {', '.join(sorted(present))}")

    prom = [r for r in recs if r["check"] == "golden_promoted"]
    mid = [r for r in prom if not str(r.get("ssa", "")).startswith("%arg")]
    print(
        f"\ngolden_promoted: {len(prom)} total, {len(mid)} MID-GRAPH "
        f"(device value taken as truth -> isolation cannot see corrupted inputs;\n"
        f"                 the per-layer cumulative probe is what covers that)"
    )

    print(
        f"\nVERDICT: {'PASS' if not fails else 'FAIL'} "
        f"-- {len(checked)} ops checked against floor {a.floor}"
    )
    for op, w in sorted(fails.items(), key=lambda kv: kv[1]):
        print(f"  BELOW FLOOR: {op} worst={w:.8f}")


if __name__ == "__main__":
    main()
