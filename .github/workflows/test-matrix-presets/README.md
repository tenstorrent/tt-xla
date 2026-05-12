# Test Matrix Presets

Each `.json` file is a list of test job entries consumed by `generate_test_matrix.py`.
Defines a group of tests to run using call-test

## Fields

| Field | Required | Description |
|---|---|---|
| `runs-on` | ✓ | Runner label (`n150`, `p150`, `n300`, `n300-llmbox`, `wormhole_b0`, …) |
| `name` | ✓ | Job name in GitHub Actions UI |
| `dir` | ✓ | Path passed to pytest (directory, file, or `file.py::test_fn`) |
| `test-mark` | ✓ | Pytest `-m` expression |
| `parallel-groups` | | Split tests into N parallel jobs (generator expands to N entries with `group-id`) |
| `shared-runners` | | `true` to use a GitHub-hosted runner; generator remaps `runs-on` to the shared label |
| `require` | | `"release"` — force `wheel_build=release`. `"alchemist"` — same + download alchemist lib |
| `args` | | Extra pytest arguments (e.g. `--emitpy`) |
| `contains` | | Passed to pytest `-k` |
| `forge-models` | | `true` for `tests/runner/test_models.py` entries — installs system deps, validates config, passes `--arch` |
| `forked` | | `true` to run tests in isolated processes (`--forked`) |
| `extra-wheel` | | `"vllm"` — downloads and installs the vllm wheel |
