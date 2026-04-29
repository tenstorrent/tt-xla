# Test Matrix Presets

Each `.json` file defines a list of test jobs consumed by `generate_test_matrix.py`.
Preset files can include other presets via `{ "include": ["file1.json", "file2.json"] }`.

## Entry fields

### Required

| Field | Type | Description |
|---|---|---|
| `runs-on` | string | Runner label (e.g. `n150`, `n300`, `n300-llmbox`, `wormhole_b0`, `galaxy-wh-6u`) |
| `name` | string | Job name shown in GitHub Actions UI |
| `dir` | string | Path passed to pytest (file, directory, or `file.py::test_fn`) |
| `test-mark` | string | Pytest `-m` expression used to filter tests |

### Optional — test selection

| Field | Type | Description |
|---|---|---|
| `args` | string | Extra arguments appended to the pytest command (e.g. `--emitpy`) |
| `contains` | string | Value passed to pytest `-k` for additional keyword filtering |
| `parallel-groups` | int | Split collected tests into N groups; the generator expands this entry into N jobs each with a `group-id` |

### Optional — runner behaviour

| Field | Type | Description |
|---|---|---|
| `shared-runners` | bool | Use a GitHub-hosted shared runner instead of a self-hosted one. The generator maps `runs-on` to the corresponding shared runner label and preserves the original value in `runs-on-original` |
| `require` | string | Special build artifact requirement. `"release"` forces `wheel_build=release`. `"alchemist"` forces `wheel_build=release` and downloads the alchemist compiler library |

### Optional — workflow hints

These flags tell `call-test.yml` to activate extra steps without hard-coding job names in the workflow.

| Field | Type | Description |
|---|---|---|
| `forge-models` | bool | Set `true` on any entry whose `dir` points to `tests/runner/test_models.py`. Triggers: system dep install (`libgl1`), test config validation, and passes `--arch` to pytest |
| `forked` | bool | Set `true` when tests must run in isolated processes (`--forked`). Required for torch and forge-models jobs due to test isolation issues (see [#795](https://github.com/tenstorrent/tt-xla/issues/795)) |
| `extra-wheel` | string | Name of an additional wheel bundle to download and install before running tests. Currently the only value is `"vllm"`, which downloads the `vllm-tt-whl-release` artifact and installs `requirements-vllm-plugin.txt` |

## Example entry

```json
{
  "runs-on": "n150",
  "name": "run_forge_models_llm",
  "dir": "./tests/runner/test_models.py::test_llms_torch",
  "test-mark": "n150 and nightly and expected_passing and single_device",
  "parallel-groups": 3,
  "forge-models": true,
  "forked": true
}
```

## Including other presets

A file can include other preset files instead of (or mixed with) direct entries:

```json
[
  { "include": ["basic-test.json", "model-test-push.json"] }
]
```

Includes are resolved recursively. An entry is either an `include` directive or a test entry — not both.
