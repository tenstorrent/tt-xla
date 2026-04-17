# tt-xla Commit Message Template

## Format

```
[Area] Short imperative description (#PR)
```

The `(#PR)` suffix is added automatically by GitHub on merge — omit it when writing manually.

---

## Area Prefix Table

Choose the **one** prefix that best matches the primary area touched. If no single area dominates, use the bare-verb style (no prefix).

| Prefix | Use when changes are primarily in… |
|---|---|
| `[vLLM plugin]` | `integrations/vllm_plugin/`, `tests/integrations/vllm_plugin/` |
| `[vLLM]` | vLLM-related but not plugin-specific (e.g. sampling, logprobs) |
| `[CI]` | `.github/workflows/`, `.github/actions/`, `.github/scripts/` |
| `[Test Infra]` | `tests/infra/`, `tests/runner/`, `pytest.ini`, `.test_durations` |
| `[pjrt]` | `pjrt_implementation/` |
| `[FX fusing]` | `python_package/tt_torch/` fusion passes |
| `[test]` | New/updated test files only (no source changes) |
| `[build]` | `CMakeLists.txt`, `CMakePresets.json`, `python_package/` build config |
| `[python-package]` | `python_package/` (non-build: deps, packaging) |
| `[tools]` | `scripts/` |

---

## Bare-Verb Style (no prefix)

Use for general changes that span areas or don't fit a prefix:

```
Add <thing>
Fix <thing>
Update <thing>
Enable <thing>
Remove <thing>
Disable <thing>
Uplift third_party/<name> to <hash> <date>
```

---

## Rules

1. **Title ≤ 72 characters**
2. **Imperative, sentence-case** — e.g. "Add support for sparse moe", not "Added" or "adding"
3. **No trailing period**
4. **No conventional-commit prefixes** (`feat:`, `fix:`, `chore:` etc.) — this repo does NOT use that convention
5. **SPDX copyright header** required on all new source files: `// SPDX-License-Identifier: Apache-2.0`

---

## Path-to-Prefix Lookup

| Changed path starts with… | Prefix |
|---|---|
| `.github/` | `[CI]` |
| `tests/integrations/vllm_plugin/` | `[vLLM plugin]` |
| `integrations/vllm_plugin/` | `[vLLM plugin]` |
| `tests/infra/` or `tests/runner/` | `[Test Infra]` |
| `tests/` (other) | `[test]` |
| `pjrt_implementation/` | `[pjrt]` |
| `python_package/tt_torch/` (fusion) | `[FX fusing]` |
| `python_package/` | `[build]` or `[python-package]` |
| `scripts/` | `[tools]` |
| `third_party/` (submodule bump) | bare `Uplift third_party/…` |

---

## Examples (from recent history)

```
[vLLM] Implement prompt_logprobs support
[CI] revert upgrade of checkout
[Test Infra] prevent ReferenceError during test teardown cleanup
[FX fusing] Expand rms_norm pattern for gpt_oss
[vLLM plugin] Implement allowed_token_ids and min_tokens sampler-level enforcement
[pjrt] PJRT_Buffer_ToHostBuffer size query fix
[build] local dev build fixes
[test] conv3d + mochi decoder tests improvement
Add support for sparse moe
Fix nightly
Enable dtype in CI runs log summary
Uplift third_party/tt_forge_models to 78e977e 2026-03-11
```
