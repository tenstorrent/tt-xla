# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
# SPDX-License-Identifier: Apache-2.0
"""Build colored tt-perf-report HTML previews from *performance_report.txt (no --no-advice tail)."""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

STYLE_END_MARK = "  </style>\n</head>\n"


def footer_html(png: str, alt: str) -> str:
    return f"""
    </pre>
    <div class="summary-chart">
      <img src="{html.escape(png)}" alt="{html.escape(alt)}" />
    </div>
  </div>
  <script>
    (function () {{
      const pre = document.querySelector("pre");
      if (!pre) return;

      const spans = Array.from(pre.querySelectorAll("span"));
      let inGapSection = false;

      for (const span of spans) {{
        const text = span.textContent.trim();

        if (text === "High Op-to-Op Gap") {{
          inGapSection = true;
          continue;
        }}

        if (inGapSection && text === "Matmul Optimization") {{
          break;
        }}

        if (!inGapSection) continue;

        if (!/^\\d+/.test(text) || !/DeviceOperation/.test(text)) continue;

        span.classList.remove("hi-us", "hi-us-low", "hi-us-med", "hi-us-high");

        const matches = [...span.textContent.matchAll(/(\\d[\\d,]*(?:\\.\\d+)?) μs/g)];
        if (matches.length < 2) continue;

        const gapUs = Number(matches[1][1].replace(/,/g, ""));
        if (Number.isNaN(gapUs)) continue;

        if (gapUs >= 400) {{
          span.classList.add("hi-gap", "hi-gap-400");
        }} else if (gapUs >= 200) {{
          span.classList.add("hi-gap", "hi-gap-200");
        }} else if (gapUs >= 100) {{
          span.classList.add("hi-gap", "hi-gap-100");
        }}
      }}
    }})();
  </script>
</body>
</html>
"""


def load_style_from_reference(ref_html: Path) -> str:
    text = ref_html.read_text(encoding="utf-8")
    i = text.index("<style>")
    j = text.index("</style>") + len("</style>")
    return text[i:j]


def escape_line(s: str) -> str:
    return html.escape(s)


def device_time_us(line: str) -> int | None:
    nums = re.findall(r"(\d[\d,]*)\s*μs", line)
    if not nums:
        return None
    return int(nums[0].replace(",", ""))


def op_category_class(line: str) -> str:
    if "MatmulDeviceOperation" in line:
        return "cat-matmul"
    if "FastReduceNCDeviceOperation" in line:
        return "cat-reduce"
    if re.search(
        r"(AllGather|ReduceScatter|AllBroadcast|Scatter)DeviceOperation", line
    ):
        return "cat-ccl"
    if re.search(
        r"(Softmax|SdpaDecode|TopK|MeshPartition|Embeddings|PagedUpdateCache|LayerNorm)DeviceOperation",
        line,
    ):
        return "cat-special"
    if re.search(r"(BinaryNg|Unary)DeviceOperation", line):
        return "cat-eltwise"
    if "ReduceDeviceOperation" in line:
        return "cat-reduce"
    if "DeviceOperation" in line:
        return "cat-tm"
    return "cat-tm"


def hi_device_classes(us: int | None) -> list[str]:
    if us is None:
        return []
    if us >= 200:
        return ["hi-us", "hi-us-high"]
    if us >= 100:
        return ["hi-us", "hi-us-med"]
    if us >= 50:
        return ["hi-us", "hi-us-low"]
    return []


def format_body_line(line: str, in_stacked: bool) -> tuple[str, bool]:
    """Return (html span line, new in_stacked)."""
    stripped = line.rstrip("\n")
    if "📊 Stacked" in stripped:
        in_stacked = True

    if stripped.strip() == "":
        return f"<span class=\"rule\">{escape_line(stripped)}</span>", in_stacked

    if "🚀 Performance Report" in stripped:
        return f"<span class=\"title\">{escape_line(stripped)}</span>", in_stacked
    if stripped.strip().startswith("==="):
        return f"<span class=\"rule-title\">{escape_line(stripped)}</span>", in_stacked
    if re.match(r"^[-=]{20,}", stripped):
        return f"<span class=\"rule\">{escape_line(stripped)}</span>", in_stacked
    if re.match(r"^ID\s+Total %", stripped) or re.match(
        r"^Total %\s+Op Code", stripped
    ):
        return f"<span class=\"header\">{escape_line(stripped)}</span>", in_stacked
    if stripped.startswith("- "):
        return f"<span class=\"bullet\">{escape_line(stripped)}</span>", in_stacked
    if stripped.startswith("These ops") or stripped.startswith("Alternatively"):
        return f"<span class=\"note\">{escape_line(stripped)}</span>", in_stacked
    if stripped in ("Matmul Optimization", "High Op-to-Op Gap"):
        return f"<span class=\"section\">{escape_line(stripped)}</span>", in_stacked

    stacked_data = re.match(
        r"^\s*(\d+\.?\d*)\s+%\s+(\S+)", stripped
    ) and in_stacked
    if stacked_data and "DeviceOperation" in stripped:
        parts = (
            ["stacked-row", op_category_class(stripped)]
            + hi_device_classes(device_time_us(stripped))
        )
        cls = " ".join(parts)
        return f"<span class=\"{cls}\">{escape_line(stripped)}</span>", in_stacked

    if "DeviceOperation" in stripped and re.match(r"^\s*\d+", stripped):
        cls_parts = [op_category_class(stripped)]
        if "SLOW" in stripped:
            cls_parts.append("slow")
        cls_parts.extend(hi_device_classes(device_time_us(stripped)))
        cls = " ".join(cls_parts)
        return f"<span class=\"{cls}\">{escape_line(stripped)}</span>", in_stacked

    if re.match(r"^\s*100\.0 %", stripped):
        return f"<span class=\"header\">{escape_line(stripped)}</span>", in_stacked

    if in_stacked and re.search(r"\d+\.?\d*\s+%", stripped):
        if "DeviceOperation" in stripped:
            parts = (
                ["stacked-row", op_category_class(stripped)]
                + hi_device_classes(device_time_us(stripped))
            )
            cls = " ".join(parts)
            return f"<span class=\"{cls}\">{escape_line(stripped)}</span>", in_stacked
        return (
            f"<span class=\"stacked-row\">{escape_line(stripped)}</span>",
            in_stacked,
        )

    return f"<span>{escape_line(stripped)}</span>", in_stacked


def trim_report_lines(lines: list[str]) -> list[str]:
    start = 0
    for i, ln in enumerate(lines):
        if "🚀 Performance Report" in ln:
            start = i
            break
    out: list[str] = []
    for ln in lines[start:]:
        if ln.startswith("Writing CSV") or ln.startswith("Plotting PNG"):
            break
        if ln.startswith("Warning: Unclassified"):
            continue
        out.append(ln.rstrip("\n"))
    return out


def build_page(
    txt_path: Path,
    page_title: str,
    report_name: str,
    png_name: str,
    style_block: str,
    out_path: Path,
) -> None:
    raw = txt_path.read_text(encoding="utf-8", errors="replace").splitlines()
    body_lines = trim_report_lines(raw)
    in_stacked = False
    spans: list[str] = []
    for ln in body_lines:
        span, in_stacked = format_body_line(ln, in_stacked)
        spans.append(span)

    head = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{html.escape(page_title)}</title>
{style_block}
</head>
<body>
  <div class="container">
    <div class="report-name">{html.escape(report_name)}</div>
    <pre>"""
    mid = "\n".join(spans)
    foot = footer_html(png_name, report_name)
    out_path.write_text(head + "\n" + mid + foot, encoding="utf-8")


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    ref = (
        repo
        / "github_pages/perf_reports/2026_03_25_12_13_39/tt_perf/even_full_performance_report.html"
    )
    style = load_style_from_reference(ref)
    lp = (
        repo
        / ".tracy_artifacts/reports/2026_03_27_12_21_16/layer_profile"
    )
    out_dir = repo / "github_pages/perf_reports/2026_03_27_12_21_16/tt_perf"
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs = [
        (
            lp / "g2_even_full_layer_performance_report.txt",
            "Even Full Layer (2026-03-27)",
            "Even Full Layer",
            "g2_even_full_layer_summary.png",
            out_dir / "even_full_performance_report.html",
        ),
        (
            lp / "g2_odd_full_layer_performance_report.txt",
            "Odd Full Layer (2026-03-27)",
            "Odd Full Layer",
            "g2_odd_full_layer_summary.png",
            out_dir / "odd_full_performance_report.html",
        ),
        (
            lp / "decode_2_only_performance_report.txt",
            "Second decode₂ trace (2026-03-27)",
            "Second decode₂ trace (full slice)",
            "decode_2_only_summary.png",
            out_dir / "decode_2_performance_report.html",
        ),
        (
            lp / "g2_even_full_layer_attention_only_performance_report.txt",
            "Even Attention Sublayer (2026-03-27)",
            "Even Attention Sublayer",
            "g2_even_full_layer_attention_only_summary.png",
            out_dir / "even_attention_performance_report.html",
        ),
        (
            lp / "g2_odd_full_layer_attention_only_performance_report.txt",
            "Odd Attention Sublayer (2026-03-27)",
            "Odd Attention Sublayer",
            "g2_odd_full_layer_attention_only_summary.png",
            out_dir / "odd_attention_performance_report.html",
        ),
        (
            lp / "g2_even_full_layer_moe_only_performance_report.txt",
            "Even MoE Sublayer (2026-03-27)",
            "Even MoE Sublayer",
            "g2_even_full_layer_moe_only_summary.png",
            out_dir / "even_moe_performance_report.html",
        ),
        (
            lp / "g2_odd_full_layer_moe_only_performance_report.txt",
            "Odd MoE Sublayer (2026-03-27)",
            "Odd MoE Sublayer",
            "g2_odd_full_layer_moe_only_summary.png",
            out_dir / "odd_moe_performance_report.html",
        ),
    ]

    for txt, title, rname, png, outp in jobs:
        if not txt.is_file():
            print(f"skip missing {txt}", file=sys.stderr)
            continue
        build_page(txt, title, rname, png, style, outp)
        print("wrote", outp)

    import shutil

    for png in [
        "g2_even_full_layer_summary.png",
        "g2_odd_full_layer_summary.png",
        "decode_2_only_summary.png",
        "g2_even_full_layer_attention_only_summary.png",
        "g2_odd_full_layer_attention_only_summary.png",
        "g2_even_full_layer_moe_only_summary.png",
        "g2_odd_full_layer_moe_only_summary.png",
    ]:
        src = lp / png
        if src.is_file():
            shutil.copy2(src, out_dir / png)


if __name__ == "__main__":
    main()
