"""Slice GPT-OSS middle-layer CSVs into attention-only and MoE-only by LayerNorm boundaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def device0_sorted_ops(path: Path) -> list[tuple[int, str]]:
    rows: list[tuple[int, str]] = []
    with path.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            if row["DEVICE ID"] != "0":
                continue
            gcc = int(row["GLOBAL CALL COUNT"])
            rows.append((gcc, row["OP CODE"]))
    rows.sort(key=lambda x: x[0])
    return rows


def gcc_ranges_from_layer_norms(ops: list[tuple[int, str]]) -> tuple[tuple[int, int], tuple[int, int]]:
    ln_idx = [i for i, (_, o) in enumerate(ops) if o == "LayerNormDeviceOperation"]
    if len(ln_idx) < 2:
        raise ValueError(
            f"expected >=2 LayerNormDeviceOperation on device 0, got {len(ln_idx)}"
        )
    ln0, ln1 = ln_idx[0], ln_idx[1]
    attn_lo = ops[ln0][0]
    attn_hi = ops[ln1 - 1][0]
    moe_lo = ops[ln1][0]
    moe_hi = ops[-1][0]
    return (attn_lo, attn_hi), (moe_lo, moe_hi)


def filter_by_gcc(
    src: Path,
    gcc_lo: int,
    gcc_hi: int,
    dst: Path,
    fieldnames: list[str],
) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with src.open(newline="", encoding="utf-8", errors="replace") as fin:
        reader = csv.DictReader(fin)
        with dst.open("w", newline="", encoding="utf-8") as fout:
            w = csv.DictWriter(fout, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for row in reader:
                g = row.get("GLOBAL CALL COUNT", "").strip()
                if not g:
                    continue
                gi = int(g)
                if gcc_lo <= gi <= gcc_hi:
                    w.writerow(row)
                    n += 1
    return n


def process_layer(
    src: Path,
    out_dir: Path,
    stem: str,
    fieldnames: list[str],
) -> dict:
    ops = device0_sorted_ops(src)
    (attn_lo, attn_hi), (moe_lo, moe_hi) = gcc_ranges_from_layer_norms(ops)
    attn_path = out_dir / f"{stem}_attention.csv"
    moe_path = out_dir / f"{stem}_moe.csv"
    n_attn = filter_by_gcc(src, attn_lo, attn_hi, attn_path, fieldnames)
    n_moe = filter_by_gcc(src, moe_lo, moe_hi, moe_path, fieldnames)
    return {
        "source": str(src),
        "device0_op_count": len(ops),
        "layer_norm_indices_device0": [
            i for i, (_, o) in enumerate(ops) if o == "LayerNormDeviceOperation"
        ],
        "attention_gcc_range": [attn_lo, attn_hi],
        "moe_gcc_range": [moe_lo, moe_hi],
        "attention_rows_all_devices": n_attn,
        "moe_rows_all_devices": n_moe,
        "attention_csv": str(attn_path),
        "moe_csv": str(moe_path),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--layer-profile-dir", type=Path, required=True)
    p.add_argument("--out-sublayers", type=Path, default=None)
    args = p.parse_args()
    lp = args.layer_profile_dir.resolve()
    out = (args.out_sublayers or (lp / "sublayers")).resolve()

    even = lp / "middle_layer_even.csv"
    odd = lp / "middle_layer_odd.csv"
    if not even.is_file() or not odd.is_file():
        raise SystemExit(f"missing {even} or {odd}")

    with even.open(newline="", encoding="utf-8", errors="replace") as f:
        fieldnames = list(csv.DictReader(f).fieldnames or [])

    meta = {
        "method": "device0_two_layernorm_gcc_window",
        "reference": "2026_03_25 hub + gpt-oss-layer-parsing",
        "even": process_layer(even, out, "middle_even", fieldnames),
        "odd": process_layer(odd, out, "middle_odd", fieldnames),
    }
    meta_path = out / "sublayer_slice_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
