# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Generate the ``latest-forge-models-changed.json`` test-matrix preset for a
tt-forge-models uplift.

Given the previous (``--old``) and new (``--new``) tt-forge-models commit, this
computes which *models* changed between them and emits a test-matrix preset that
runs exactly those models, grouped by the arch(es) each model supports. The
result is composed with the existing ``model-test-extended.json`` suite via
``forge-models-uplift-qualification.json`` so an uplift runs
``(extended suite) UNION (changed models)``.

Detection (pure git + path + YAML parsing, no model imports):

1. ``git -C <repo> diff --name-only <old> <new>`` -> changed paths (relative to
   the tt-forge-models root).
2. Classify each changed path:
     - ignore: docs (``*.md``, ``LICENSE*``, ``docs/``);
     - global/shared: any root-level file (e.g. ``base.py``, ``config.py``,
       ``utils.py``) -> potentially affects every model, so we emit ``[]`` (add
       nothing beyond today's suite);
     - model-scoped otherwise.
3. Map each model-scoped file to the torch loader dir(s) it affects, by
   path-component prefix in either direction (file inside a model subtree, or a
   shared file above several loaders). Loader dirs are discovered by walking for
   ``**/pytorch/loader.py`` (same rule as tests/runner/utils/dynamic_loader.py);
   their ``<model_path>`` (relative dir, always ending in ``/pytorch``) is the
   leading segment of the pytest test id ``<model_path>-<variant>-...`` and is
   used verbatim as a ``-k`` term.
4. Resolve each changed model_path's arch set from ``supported_archs`` in the
   tests/runner/test_config/**/*.yaml configs (union over all config keys that
   start with ``<model_path>-``), defaulting to ``[n150, p150]`` for
   new/unconfigured models (matches tests/runner/conftest.py default_archs).
5. Group changed models by arch and emit one matrix entry per arch, each with an
   arch-only ``test-mark`` (so xfail/unspecified/new changed models are
   collectable) and a ``contains`` (``-k``) listing that arch's changed models.

NOTE (v1 scope):
  - Torch models only. The extended suite is torch-only (test_all_models_torch);
    JAX loaders (``**/jax/loader.py``) are intentionally not emitted here.
  - No runner-availability filtering: whatever arch a changed model declares is
    emitted, including scarce runners (galaxy-wh-6u, *blackhole, n300, p150).
  - The arch-only marker does not constrain parallelism, so all run-mode/
    parallelism param combos of a changed model on a supported arch are run.

Usage:
    python gen_changed_models_matrix.py \
        --old <sha> --new <sha> --repo <tt_forge_models checkout> \
        --config-dir tests/runner/test_config \
        --out .github/workflows/test-matrix-presets/latest-forge-models-changed.json
"""

import argparse
import glob
import json
import os
import subprocess
import sys

import yaml

# Default archs applied to a model that has no test_config entry (e.g. a model
# added in this uplift). Mirrors tests/runner/conftest.py default_archs.
DEFAULT_ARCHS = ["n150", "p150"]

# Archs whose torch tests run single-chip and are forked for process isolation,
# mirroring model-test-extended.json's n150 entry.
FORKED_ARCHS = {"n150", "p150"}

TEST_DIR = "./tests/runner/test_models.py::test_all_models_torch"


def _components(path: str):
    """Split a POSIX-style relative path into non-empty components."""
    return [c for c in path.strip("/").split("/") if c]


def is_prefix(a: str, b: str) -> bool:
    """True if path ``a`` is a component-wise prefix of (or equal to) ``b``."""
    ac, bc = _components(a), _components(b)
    return len(ac) <= len(bc) and bc[: len(ac)] == ac


def get_changed_files(repo: str, old: str, new: str):
    """Return files that differ between ``old`` and ``new`` (repo-relative)."""
    out = subprocess.check_output(
        ["git", "-C", repo, "diff", "--name-only", old, new],
        text=True,
    )
    return [line.strip() for line in out.splitlines() if line.strip()]


def discover_torch_model_paths(repo: str):
    """Discover torch loader dirs, returned as repo-relative model_paths.

    A model_path is ``relpath(dirname(pytorch/loader.py), repo)`` -- e.g.
    ``gpt2/pytorch``, ``llama/causal_lm/pytorch``. Matches the discovery rule in
    tests/runner/utils/dynamic_loader.py (basename == "pytorch" and loader.py).
    """
    model_paths = []
    for root, _dirs, files in os.walk(repo):
        if os.path.basename(root) == "pytorch" and "loader.py" in files:
            model_paths.append(os.path.relpath(root, repo).replace(os.sep, "/"))
    return sorted(set(model_paths))


def is_ignored(path: str) -> bool:
    """Docs / license files never affect model behavior."""
    base = os.path.basename(path)
    if base.endswith(".md") or base.startswith("LICENSE"):
        return True
    if _components(path)[:1] == ["docs"]:
        return True
    return False


def is_global(path: str) -> bool:
    """A root-level file is treated as shared -> potentially affects all models."""
    return "/" not in path.strip("/")


def affected_model_paths(changed_file: str, model_paths):
    """Loader dirs affected by ``changed_file`` (prefix relation either way)."""
    d = os.path.dirname(changed_file)
    return [mp for mp in model_paths if is_prefix(mp, d) or is_prefix(d, mp)]


def load_supported_archs(config_dir: str):
    """Map each test_config key -> its supported_archs list (or [] if absent)."""
    key_archs = {}
    for yaml_file in glob.glob(
        os.path.join(config_dir, "**", "*.yaml"), recursive=True
    ):
        with open(yaml_file, "r") as f:
            data = yaml.safe_load(f) or {}
        test_config = data.get("test_config") or {}
        for key, meta in test_config.items():
            archs = (meta or {}).get("supported_archs") or []
            # Normalize to a list of strings.
            if isinstance(archs, str):
                archs = [archs]
            key_archs.setdefault(key, [])
            key_archs[key].extend(str(a) for a in archs)
    return key_archs


def archs_for_model(model_path: str, key_archs):
    """Union of supported_archs across config keys for ``model_path``.

    Config keys have the form ``<model_path>-<variant>-<parallelism>-<runmode>``,
    so we match keys equal to or starting with ``<model_path>-``. Falls back to
    DEFAULT_ARCHS when the model has no config entry.
    """
    archs = set()
    for key, key_arch_list in key_archs.items():
        if key == model_path or key.startswith(model_path + "-"):
            archs.update(key_arch_list)
    return sorted(archs) if archs else list(DEFAULT_ARCHS)


def build_matrix(changed_model_paths, key_archs):
    """Group changed model_paths by arch and build matrix entries."""
    arch_to_models = {}
    for mp in sorted(changed_model_paths):
        for arch in archs_for_model(mp, key_archs):
            arch_to_models.setdefault(arch, []).append(mp)

    matrix = []
    for arch in sorted(arch_to_models):
        contains = " or ".join(sorted(arch_to_models[arch]))
        entry = {
            "runs-on": arch,
            "name": "changed_forge_models",
            "dir": TEST_DIR,
            "test-mark": arch,
            "contains": contains,
            "forge-models": True,
        }
        if arch in FORKED_ARCHS:
            entry["forked"] = True
        matrix.append(entry)
    return matrix


def compute_matrix(repo, old, new, config_dir):
    """Return the matrix list (possibly empty) for the given uplift range."""
    changed_files = get_changed_files(repo, old, new)

    model_scoped = []
    for path in changed_files:
        if is_ignored(path):
            continue
        if is_global(path):
            # A shared/general file changed -> add nothing beyond today's suite.
            print(f"Global/shared file changed ({path}); emitting []", file=sys.stderr)
            return []
        model_scoped.append(path)

    if not model_scoped:
        print("No model-scoped changes; emitting []", file=sys.stderr)
        return []

    model_paths = discover_torch_model_paths(repo)
    changed = set()
    for path in model_scoped:
        changed.update(affected_model_paths(path, model_paths))

    if not changed:
        print("No torch models affected by changes; emitting []", file=sys.stderr)
        return []

    key_archs = load_supported_archs(config_dir)
    return build_matrix(changed, key_archs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", required=True, help="Previous tt-forge-models SHA")
    parser.add_argument("--new", required=True, help="New tt-forge-models SHA")
    parser.add_argument(
        "--repo", required=True, help="Path to tt-forge-models checkout"
    )
    parser.add_argument(
        "--config-dir",
        default="tests/runner/test_config",
        help="Path to tests/runner/test_config",
    )
    parser.add_argument("--out", required=True, help="Output preset JSON path")
    args = parser.parse_args()

    matrix = compute_matrix(args.repo, args.old, args.new, args.config_dir)

    with open(args.out, "w") as f:
        json.dump(matrix, f, indent=2)
        f.write("\n")

    print(f"Wrote {len(matrix)} entr{'y' if len(matrix) == 1 else 'ies'} to {args.out}")
    print(json.dumps(matrix, indent=2))


if __name__ == "__main__":
    main()
