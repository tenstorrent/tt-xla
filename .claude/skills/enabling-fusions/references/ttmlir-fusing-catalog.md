# tt-mlir Fusing Catalog (shared)

This skill reuses the **single canonical** fusing catalog maintained by the detection skill, to avoid divergence between detect and enable:

[`../../finding-missed-fusions/references/ttmlir-fusing-catalog.md`](../../finding-missed-fusions/references/ttmlir-fusing-catalog.md)

Read that file for the full list of TTIR (`ttir-fusing`) and TTNN (`ttnn-fusing`) patterns, the op each fuses to, the owning pass, source files, gating flags, and the TTIR-vs-TTNN level rule of thumb. Do not duplicate the catalog here — update the canonical copy and refresh it when the pinned tt-mlir commit (`third_party/CMakeLists.txt`) changes.
