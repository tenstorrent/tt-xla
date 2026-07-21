# ONNX Front-End Integration — Learning Roadmap

> Goal: integrate an **ONNX front-end** into tt-xla. The strategy is to convert
> ONNX models into the **StableHLO (SHLO)** dialect (via `onnx-mlir` or
> `torch-mlir`), then feed the *existing* tt-xla pipeline:
> `StableHLO → TTIR → TTNN → Tenstorrent hardware`.
>
> Audience: a dev proficient in C++/Python with AI basics, new to ONNX & MLIR.
> Use the checkboxes to track progress. You're "done" with a subtopic when you
> can explain it aloud or answer its self-check.

## The mental model (where we're headed)

```
ONNX model (.onnx file)
      │   ← onnx-mlir OR torch-mlir imports it
      ▼
MLIR (onnx dialect / torch dialect)
      │   ← conversion passes "lower" it
      ▼
StableHLO (SHLO) dialect   ◄── this is the join point with tt-xla
      │   ← tt-mlir compiler (already exists in your repo)
      ▼
TTIR dialect
      ▼
TTNN dialect
      ▼
Tenstorrent hardware
```

Key insight: **StableHLO is the "airlock."** If we can get an ONNX model into
StableHLO, the entire downstream pipeline already exists. So the whole problem
reduces to: *"how do I reliably turn ONNX into StableHLO?"* — and the two
candidate doors are `onnx-mlir` and `torch-mlir`.

---

## Module 0 — Framing (½ day)
Understand the problem before the tools.

- [x] **0.1** What tt-xla is today: a PJRT plugin ("glue in the middle"). JAX/PyTorch produce StableHLO → tt-mlir compiles it.
- [x] **0.2** What PJRT is — a *C API contract* (interface, not implementation); its `Compile` fn receives StableHLO; tt-xla implements it. C is chosen for ABI stability.
- [x] **0.3** The current data flow, end to end — 8 steps; PJRT boundary sits at Step 3 (`PJRT_Client_Compile`).
- [x] **0.4** Why ONNX is desirable: a *frozen, framework-neutral file* that widens the runnable-model set; price = op-coverage in conversion.
- [x] **0.5** The key realization: build a *front-end adapter* ending at StableHLO — not a new backend. Whole project reduces to "ONNX → StableHLO."
- Self-check: Can you draw the diagram above from memory and point at the "join point"? ✅ COMPLETE

---

## Module 1 — ONNX fundamentals (2–3 days)
The *format*, not the math.

- [ ] **1.1** What ONNX is — a serialized computation graph stored as protobuf.
- [ ] **1.2** Protobuf basics — how the graph is encoded on disk (`onnx.proto`).
- [ ] **1.3** The ONNX object model: `ModelProto`, `GraphProto`, `NodeProto`, `TensorProto`, `ValueInfoProto`.
- [ ] **1.4** Operators & opsets — the standard, versioned operator set.
- [ ] **1.5** Static vs dynamic shapes; shape inference.
- [ ] **1.6** Hands-on: export a tiny PyTorch model, load with `onnx`, inspect, run with `onnxruntime`.
- Self-check: Given a `.onnx` file, can you enumerate its nodes and identify the opset in ~10 lines of Python?

---

## Module 2 — Compiler & IR fundamentals (2 days)
Conceptual grounding before MLIR.

- [ ] **2.1** What an IR (Intermediate Representation) is.
- [ ] **2.2** SSA form (Static Single Assignment) — `%0 = op(%a, %b)`.
- [ ] **2.3** "Lowering" — progressively rewriting high-level to low-level.
- [ ] **2.4** Passes — a compiler as a pipeline of transformations.
- [ ] **2.5** Dialects as "vocabularies" (preview of MLIR).
- Self-check: Explain "lowering from dialect A to dialect B" in one sentence.

---

## Module 3 — MLIR core (4–5 days) ← heaviest module

- [ ] **3.1** What MLIR is — "Multi-Level IR," part of LLVM; dialects are its big idea.
- [ ] **3.2** Structural hierarchy: `Operation`/`Region`/`Block`/`Value`/`Type`/`Attribute`.
- [ ] **3.3** Dialects: `func`, `arith`, `tensor`, `linalg`, `stablehlo`, `onnx`, `torch`, `ttir`, `ttnn`.
- [ ] **3.4** Reading MLIR textual format.
- [ ] **3.5** Pass & Pass Manager system.
- [ ] **3.6** Dialect Conversion framework (`ConversionPattern`, `TypeConverter`, legal/illegal ops).
- [ ] **3.7** The `mlir-opt` tool.
- [ ] **3.8** Hands-on: write a trivial `.mlir`, run `mlir-opt --canonicalize`.
- Self-check: Read a small `.mlir` aloud and identify each op's dialect.

Resources: MLIR Toy Tutorial (ch. 1–4), MLIR Language Reference.

---

## Module 4 — StableHLO, the join point (2 days)

- [ ] **4.1** What StableHLO is — portable, stable high-level compute ops (from XLA/HLO).
- [ ] **4.2** Why tt-xla uses it — frameworks emit it, tt-mlir consumes it.
- [ ] **4.3** Read real StableHLO dumped from a JAX program.
- [ ] **4.4** The op-coverage question — does every ONNX op lower cleanly to StableHLO?
- Self-check: Point to where StableHLO enters the flow; name 5 StableHLO ops.

---

## Module 5 — The two candidate doors (3–4 days)

- [ ] **5.1** `onnx-mlir` — direct route (`onnx` dialect → StableHLO via `ONNXToStablehlo`).
- [ ] **5.2** `torch-mlir` — indirect route (ONNX → `torch` dialect → StableHLO).
- [ ] **5.3** Build a comparison matrix (op coverage, build complexity, vendoring, license, activity, dynamic shapes).
- [ ] **5.4** Prototype offline: take one small `.onnx` and produce a StableHLO `.mlir` on the command line via each route.
- Self-check: Can you produce a StableHLO `.mlir` from a `.onnx` on the command line?

---

## Module 6 — Integration into tt-xla (design phase)

- [ ] **6.1** Where PJRT expects input — how tt-xla receives StableHLO today (`src/`).
- [ ] **6.2** Front-end API shape — the user-facing entry for ONNX.
- [ ] **6.3** Dependency vendoring — onnx-mlir/torch-mlir under `third_party/` + CMake.
- [ ] **6.4** Testing strategy — per-op ONNX tests → StableHLO → device PCC checks.
- [ ] **6.5** Write a design doc — recommended route + integration points + risks.

---

### How to use this roadmap
- Modules **0–4 are pure learning**; **5 is investigation**; **6 is design**. Do them in order.
- Rough total: ~2.5–3 focused weeks to a confident design doc.
- Highest-value early experiment: **Module 5.4** — get *any* `.onnx` to a StableHLO `.mlir` on the command line. That validates the whole premise.

---

## Progress log
_(Add dated notes as you complete subtopics.)_

- 2026-07-21 — Roadmap created. Starting Module 0.
