# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Build GitHub Pages tt_perf bundle for GPT-OSS 120B Galaxy layer profile (2026-04-07).

Prereqs (sublayer HTML): from repo root run `python tests/benchmark/slice_gpt_oss_sublayers.py`,
then `tt-perf-report` on `layer_profile/sublayers/middle_*_{attention,moe}.csv` into
`layer_profile/reports/*_performance_report.txt` (+ summary PNG), as in the Docker workflow.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from build_tt_perf_html_from_txt import build_page, load_style_from_reference

REPO = Path(__file__).resolve().parents[2]
RUN_ID = "2026_04_07_07_11_39"
LP_REPORTS = REPO / ".tracy_artifacts/reports" / RUN_ID / "layer_profile" / "reports"
OUT_DIR = REPO / "github_pages" / "perf_reports" / RUN_ID / "tt_perf"
REF_STYLE = (
    REPO
    / "github_pages/perf_reports/2026_03_25_12_13_39/tt_perf/even_full_performance_report.html"
)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    style = load_style_from_reference(REF_STYLE)

    actual_even = LP_REPORTS / "middle_layer_even_summary.png"
    actual_odd = LP_REPORTS / "middle_layer_odd_summary.png"
    actual_dec = LP_REPORTS / "decode_2_only_summary.png"
    for src, renamed in [
        (actual_even, "even_full_summary.png"),
        (actual_odd, "odd_full_summary.png"),
        (actual_dec, "decode_2_only_summary.png"),
    ]:
        if not src.is_file():
            raise SystemExit(f"missing PNG {src}")
        shutil.copy2(src, OUT_DIR / renamed)

    jobs = [
        (
            LP_REPORTS / "middle_layer_even_performance_report.txt",
            "GPT-OSS 120B · Even full layer (middle, decode₂)",
            "Even Full Layer · GPT-OSS 120B",
            "even_full_summary.png",
            OUT_DIR / "even_full_performance_report.html",
        ),
        (
            LP_REPORTS / "middle_layer_odd_performance_report.txt",
            "GPT-OSS 120B · Odd full layer (middle, decode₂)",
            "Odd Full Layer · GPT-OSS 120B",
            "odd_full_summary.png",
            OUT_DIR / "odd_full_performance_report.html",
        ),
        (
            LP_REPORTS / "decode_2_performance_report.txt",
            "GPT-OSS 120B · Second decode₂ window (6 layers)",
            "Second decode₂ trace (6-layer slice)",
            "decode_2_only_summary.png",
            OUT_DIR / "decode_2_performance_report.html",
        ),
    ]
    for txt, title, rname, png, outp in jobs:
        if not txt.is_file():
            raise SystemExit(f"missing {txt}")
        build_page(txt, title, rname, png, style, outp)
        print("wrote", outp)

    sub_jobs = [
        (
            LP_REPORTS / "even_attention_performance_report.txt",
            "GPT-OSS 120B · Even attention sublayer",
            "Even Attention Sublayer · GPT-OSS 120B",
            "even_attention_summary.png",
            OUT_DIR / "even_attention_performance_report.html",
            LP_REPORTS / "even_attention_summary.png",
        ),
        (
            LP_REPORTS / "odd_attention_performance_report.txt",
            "GPT-OSS 120B · Odd attention sublayer",
            "Odd Attention Sublayer · GPT-OSS 120B",
            "odd_attention_summary.png",
            OUT_DIR / "odd_attention_performance_report.html",
            LP_REPORTS / "odd_attention_summary.png",
        ),
        (
            LP_REPORTS / "even_moe_performance_report.txt",
            "GPT-OSS 120B · Even MoE sublayer",
            "Even MoE Sublayer · GPT-OSS 120B",
            "even_moe_summary.png",
            OUT_DIR / "even_moe_performance_report.html",
            LP_REPORTS / "even_moe_summary.png",
        ),
        (
            LP_REPORTS / "odd_moe_performance_report.txt",
            "GPT-OSS 120B · Odd MoE sublayer",
            "Odd MoE Sublayer · GPT-OSS 120B",
            "odd_moe_summary.png",
            OUT_DIR / "odd_moe_performance_report.html",
            LP_REPORTS / "odd_moe_summary.png",
        ),
    ]
    for txt, title, rname, png, outp, png_src in sub_jobs:
        if not txt.is_file() or not png_src.is_file():
            raise SystemExit(f"missing sublayer artifact {txt} or {png_src}")
        shutil.copy2(png_src, OUT_DIR / png)
        build_page(txt, title, rname, png, style, outp)
        print("wrote", outp)


if __name__ == "__main__":
    main()
