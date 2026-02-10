# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Standalone script to validate test_config YAML files against discovered model loaders.

Uses AST parsing (no torch/jax/transformers imports) to discover model variants
and cross-check them against YAML config entries.

Usage:
    python tests/runner/validate_test_config.py
"""

import ast
import difflib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# Allowed fields in test_config YAML entries (duplicated from config_loader.py)
_ALLOWED_FIELDS = {
    "required_pcc",
    "assert_pcc",
    "assert_atol",
    "required_atol",
    "assert_allclose",
    "allclose_rtol",
    "allclose_atol",
    "status",
    "reason",
    "bringup_status",
    "markers",
    "supported_archs",
    "batch_size",
    "arch_overrides",
    "execution_pass",
    "filechecks",
}

# Allowed architecture identifiers (duplicated from conftest.py)
ALLOWED_ARCHES = {"n150", "p150", "n300", "n300-llmbox"}

# Frameworks mapped to their config directory names
FRAMEWORKS = ("torch", "jax", "torch_llm")

# Parallelism values for test ID cross-product
PARALLELISMS_STANDARD = ("single_device", "data_parallel", "tensor_parallel")
PARALLELISMS_LLM = ("single_device", "tensor_parallel")

# Run modes
RUN_MODES_STANDARD = ("inference", "training")
RUN_MODES_LLM = ("inference",)

# LLM phases
LLM_PHASES = {"load_inputs_decode": "llm_decode", "load_inputs_prefill": "llm_prefill"}

# Models excluded from PyTorch discovery (matches dynamic_loader.py:416-418)
TORCH_EXCLUDED_MODEL_DIRS = {"suryaocr"}

# Single source of truth for the placeholders YAML filename
PLACEHOLDERS_FILENAME = "test_config_placeholders.yaml"

# Regex to strip pytest parametrization segments (e.g. "seq_1-", "batch_2-") from YAML keys
_PARAMETRIZATION_RE = re.compile(r"-(seq_\d+|batch_\d+)")


def _strip_parametrization(test_id: str) -> str:
    """Remove pytest parametrization noise (seq_X, batch_Y) from a test ID."""
    return _PARAMETRIZATION_RE.sub("", test_id)


@dataclass
class ValidationResult:
    """Container for validation output.

    Attributes:
        yaml_key_count: Number of test_config entries loaded from YAML files.
        structure_errors: YAML structure violations (unknown fields, bad arch_overrides, etc.).
        torch_model_count: Number of PyTorch model variants discovered via AST.
        jax_model_count: Number of JAX model variants discovered via AST.
        llm_model_count: Number of LLM model-phase pairs discovered via AST.
        discovered_id_count: Total expected test IDs generated from discovered models.
        discovered_ids: Full set of generated test IDs (used for close-match suggestions).
        unknown: YAML keys that don't match any discovered test ID (errors).
        unlisted: Discovered test IDs missing from YAML configs (warnings).
        parse_warnings: Loader files that could not be parsed (SyntaxError / OSError).
    """

    yaml_key_count: int
    structure_errors: list[str]
    torch_model_count: int
    jax_model_count: int
    llm_model_count: int
    discovered_id_count: int
    discovered_ids: set[str] = field(default_factory=set)
    unknown: set[str] = field(default_factory=set)
    unlisted: set[str] = field(default_factory=set)
    parse_warnings: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True when there are no structure errors and no unknown YAML keys."""
        return not self.structure_errors and not self.unknown


def _extract_model_variant_enum(tree: ast.Module) -> dict:
    """Extract ModelVariant StrEnum values from an AST.

    Returns:
        Dict mapping member names to string values, e.g. {"FALCON_1B": "tiiuae/Falcon3-1B-Base"}.
        Empty dict if no ModelVariant class found.
    """
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if node.name != "ModelVariant":
            continue
        # Check it inherits from StrEnum
        if not any(
            (isinstance(b, ast.Name) and b.id == "StrEnum")
            or (isinstance(b, ast.Attribute) and b.attr == "StrEnum")
            for b in node.bases
        ):
            continue

        members = {}
        for stmt in node.body:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                target = stmt.targets[0]
                if isinstance(target, ast.Name) and isinstance(
                    stmt.value, ast.Constant
                ):
                    members[target.id] = stmt.value.value
        return members

    return {}


def _extract_variants_from_dict_node(dict_node: ast.Dict, enum_members: dict) -> list:
    """Extract variant string values from an ast.Dict node's keys."""
    variants = []
    for key in dict_node.keys:
        # Keys are ModelVariant.MEMBER_NAME
        if isinstance(key, ast.Attribute) and isinstance(key.value, ast.Name):
            member_name = key.attr
            if member_name in enum_members:
                variants.append(enum_members[member_name])
            else:
                # Fallback: use the attribute name as lowercase
                variants.append(member_name.lower())
    return variants


def _extract_variants_dict_keys(tree: ast.Module, enum_members: dict) -> list:
    """Extract _VARIANTS dict keys from ModelLoader class, resolving ModelVariant.MEMBER references.

    Also checks module-level _VARIANTS if the class attribute references a name rather than a dict literal.

    Returns:
        List of variant string values. Empty list if no _VARIANTS found.
    """
    # First, collect any module-level _VARIANTS = {...} dict
    module_level_variants_dict = None
    for stmt in tree.body:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id == "_VARIANTS"
            and isinstance(stmt.value, ast.Dict)
        ):
            module_level_variants_dict = stmt.value
            break

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if node.name != "ModelLoader":
            continue

        for stmt in node.body:
            # Look for _VARIANTS = ...
            if not isinstance(stmt, ast.Assign):
                continue
            if not (
                len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and stmt.targets[0].id == "_VARIANTS"
            ):
                continue

            if isinstance(stmt.value, ast.Dict):
                return _extract_variants_from_dict_node(stmt.value, enum_members)

            # Handle _VARIANTS = _VARIANTS (name reference to module-level dict)
            if isinstance(stmt.value, ast.Name) and module_level_variants_dict:
                return _extract_variants_from_dict_node(
                    module_level_variants_dict, enum_members
                )

        # If ModelLoader exists but has no _VARIANTS assignment, return empty
        return []

    return []


def _has_llm_method(tree: ast.Module, method_name: str) -> bool:
    """Check if ModelLoader class defines a specific method (e.g. load_inputs_decode)."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if node.name != "ModelLoader":
            continue
        for stmt in node.body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if stmt.name == method_name:
                    return True
    return False


class TestConfigValidator:
    """Cross-validates test_config YAML entries against AST-discovered model loaders.

    Walks the YAML config directories and the tt_forge_models tree, then compares
    the two sets of test IDs to find mismatches.
    """

    def __init__(self, project_root: Path):
        """Initialise paths derived from *project_root*.

        Args:
            project_root: Repository root (contains ``tests/``, ``third_party/``, etc.).
        """
        self.project_root = project_root
        self._models_root = project_root / "third_party" / "tt_forge_models"
        self._config_dir = project_root / "tests" / "runner" / "test_config"
        self._filecheck_dir = project_root / "tests" / "filecheck"

    def validate(self) -> ValidationResult:
        """Run the full validation pipeline: load YAML, check structure, discover models, cross-validate."""
        yaml_keys, all_configs = self._load_yaml_keys()
        structure_errors = self._validate_yaml_structure(all_configs)
        torch_models, jax_models, llm_models, parse_warnings = self._discover_models()
        discovered_ids = self._generate_test_ids(torch_models, jax_models, llm_models)
        # Strip pytest parametrization noise (e.g. seq_X-batch_Y) from YAML keys
        # before comparing, so the validator doesn't need to replicate every
        # parametrization detail from test_models.py.
        normalized_yaml_keys = {_strip_parametrization(k) for k in yaml_keys}
        unknown = normalized_yaml_keys - discovered_ids
        unlisted = discovered_ids - normalized_yaml_keys

        return ValidationResult(
            yaml_key_count=len(yaml_keys),
            structure_errors=structure_errors,
            torch_model_count=len(torch_models),
            jax_model_count=len(jax_models),
            llm_model_count=len(llm_models),
            discovered_id_count=len(discovered_ids),
            discovered_ids=discovered_ids,
            unknown=unknown,
            unlisted=unlisted,
            parse_warnings=parse_warnings,
        )

    def _load_yaml_keys(self) -> tuple:
        """Walk test_config directories, parse YAML files, and collect all test_config keys.

        Returns:
            Tuple of (all_keys set, all_configs dict).
        """
        all_keys = set()
        all_configs = {}  # {test_id: (config_dict, framework, filename)}

        for framework in FRAMEWORKS:
            fw_dir = self._config_dir / framework
            if not fw_dir.exists():
                continue

            for yaml_file in sorted(fw_dir.glob("test_config_*.yaml")):
                if yaml_file.name == PLACEHOLDERS_FILENAME:
                    continue

                with open(yaml_file, "r") as fh:
                    data = yaml.safe_load(fh) or {}

                tests = data.get("test_config", {}) or {}
                for test_id, cfg in tests.items():
                    all_keys.add(test_id)
                    all_configs[test_id] = (cfg, framework, yaml_file.name)

        return all_keys, all_configs

    def _validate_yaml_structure(self, all_configs: dict) -> list:
        """Validate allowed fields, arch_overrides keys, and filecheck references.

        Returns:
            List of error strings. Empty means all valid.
        """
        errors = []

        for test_id, (cfg, framework, yaml_file) in all_configs.items():
            if not isinstance(cfg, dict):
                continue

            ctx = f"test '{test_id}' in {framework}/{yaml_file}"

            # Validate allowed fields at top level
            for key in cfg.keys():
                if key not in _ALLOWED_FIELDS:
                    errors.append(
                        f"Unknown field '{key}' in {ctx}. "
                        f"Allowed: {sorted(_ALLOWED_FIELDS)}"
                    )

            # Validate arch_overrides
            overrides = cfg.get("arch_overrides")
            if overrides is not None:
                if not isinstance(overrides, dict):
                    errors.append(f"arch_overrides is not a dict in {ctx}")
                else:
                    for arch_key, arch_cfg in overrides.items():
                        if arch_key not in ALLOWED_ARCHES:
                            errors.append(
                                f"Unknown arch '{arch_key}' in arch_overrides of {ctx}. "
                                f"Allowed: {sorted(ALLOWED_ARCHES)}"
                            )
                        if isinstance(arch_cfg, dict):
                            for key in arch_cfg.keys():
                                if key not in _ALLOWED_FIELDS:
                                    errors.append(
                                        f"Unknown field '{key}' in "
                                        f"arch_overrides['{arch_key}'] of {ctx}. "
                                        f"Allowed: {sorted(_ALLOWED_FIELDS)}"
                                    )

            # Validate filecheck references
            self._validate_filechecks(cfg, ctx, errors)

        return errors

    def _validate_filechecks(self, cfg, ctx, errors):
        """Verify that every ``filechecks`` path in *cfg* resolves to an existing file.

        Checks both top-level and per-arch_override filecheck lists.
        """

        def check_list(filechecks, where):
            if not filechecks:
                return
            if not isinstance(filechecks, list):
                errors.append(
                    f"'filechecks' should be a list in {where}. "
                    f"Found: {type(filechecks).__name__}"
                )
                return
            for pattern_file in filechecks:
                if not isinstance(pattern_file, str):
                    continue
                if not (self._filecheck_dir / pattern_file).exists():
                    errors.append(
                        f"Filecheck pattern file not found: {self._filecheck_dir / pattern_file} "
                        f"(referenced in {where})"
                    )

        if "filechecks" in cfg:
            check_list(cfg["filechecks"], ctx)

        overrides = cfg.get("arch_overrides")
        if isinstance(overrides, dict):
            for arch, arch_cfg in overrides.items():
                if isinstance(arch_cfg, dict) and "filechecks" in arch_cfg:
                    check_list(
                        arch_cfg["filechecks"],
                        f"arch_overrides['{arch}'] in {ctx}",
                    )

    def _discover_models(self) -> tuple:
        """Walk the models directory and use AST to discover all model variants.

        Returns:
            Tuple of (torch_models, jax_models, llm_models, parse_warnings) where
            each model list is a list of (rel_path, variant_or_none) tuples, llm_models
            additionally includes the phase string, and parse_warnings is a list of
            warning strings.
        """
        torch_models = []
        jax_models = []
        llm_models = []
        parse_warnings = []

        for root, _, files in os.walk(self._models_root):
            if "loader.py" not in files:
                continue

            basename = os.path.basename(root)
            if basename not in ("pytorch", "jax"):
                continue

            loader_path = os.path.join(root, "loader.py")
            # Get the relative path from models_root to the loader directory
            # e.g. "falcon/pytorch" or "albert/masked_lm/jax"
            rel_path = os.path.relpath(root, self._models_root)

            # Check torch exclusions
            is_torch = basename == "pytorch"
            if is_torch:
                # Check if any parent directory is in the exclusion list
                parent_dir = os.path.basename(os.path.dirname(root))
                if parent_dir in TORCH_EXCLUDED_MODEL_DIRS:
                    continue

            try:
                with open(loader_path, "r") as f:
                    source = f.read()
                tree = ast.parse(source, filename=loader_path)
            except (SyntaxError, OSError) as e:
                parse_warnings.append(f"Cannot parse {loader_path}: {e}")
                continue

            enum_members = _extract_model_variant_enum(tree)
            variants = _extract_variants_dict_keys(tree, enum_members)

            if is_torch:
                if variants:
                    for v in variants:
                        torch_models.append((rel_path, v))
                else:
                    torch_models.append((rel_path, None))

                # Check for LLM methods
                for method_name, phase_str in LLM_PHASES.items():
                    if _has_llm_method(tree, method_name):
                        if variants:
                            for v in variants:
                                llm_models.append((rel_path, v, phase_str))
                        else:
                            llm_models.append((rel_path, None, phase_str))
            else:
                # JAX
                if variants:
                    for v in variants:
                        jax_models.append((rel_path, v))
                else:
                    jax_models.append((rel_path, None))

        return torch_models, jax_models, llm_models, parse_warnings

    def _generate_test_ids(self, torch_models, jax_models, llm_models) -> set:
        """Generate the full set of expected YAML keys by cross-producting models with parametrization.

        Returns:
            Set of all expected test IDs.
        """
        ids = set()

        # test_all_models_torch: {rel_path}-{variant}-{parallelism}-{run_mode}
        for rel_path, variant in torch_models:
            base = f"{rel_path}-{variant}" if variant else rel_path
            for parallelism in PARALLELISMS_STANDARD:
                for run_mode in RUN_MODES_STANDARD:
                    ids.add(f"{base}-{parallelism}-{run_mode}")

        # test_all_models_jax: same format
        for rel_path, variant in jax_models:
            base = f"{rel_path}-{variant}" if variant else rel_path
            for parallelism in PARALLELISMS_STANDARD:
                for run_mode in RUN_MODES_STANDARD:
                    ids.add(f"{base}-{parallelism}-{run_mode}")

        # test_llms_torch: {rel_path}-{variant}-{phase}-{parallelism}-{run_mode}
        for rel_path, variant, phase in llm_models:
            base = f"{rel_path}-{variant}" if variant else rel_path
            for parallelism in PARALLELISMS_LLM:
                for run_mode in RUN_MODES_LLM:
                    ids.add(f"{base}-{phase}-{parallelism}-{run_mode}")

        return ids


def _print_result(result: ValidationResult) -> None:
    """Format and print every section of *result* to stdout.

    Prints structure errors, parse warnings, discovery counts, unknown-test
    errors (with close-match suggestions), and unlisted-test warnings.
    """
    print(f"Loaded {result.yaml_key_count} test_config entries from YAML files")

    if result.structure_errors:
        print("\nERROR: Found YAML structure issues:")
        for err in result.structure_errors:
            print(f"  - {err}")
        return

    print("All YAML structure checks passed (fields, arch_overrides, filechecks)")

    for warning in result.parse_warnings:
        print(f"WARNING: {warning}")
    print(
        f"Discovered {result.torch_model_count} torch models, "
        f"{result.jax_model_count} jax models, "
        f"{result.llm_model_count} LLM model-phase pairs"
    )
    print(f"Generated {result.discovered_id_count} expected test IDs")

    print(
        f"\nFound {len(result.unknown)} unknown tests "
        f"and {len(result.unlisted)} unlisted tests",
        flush=True,
    )

    if result.unlisted:
        print("\nWARNING: The following tests are missing from test_config yaml files:")
        for test_name in sorted(result.unlisted):
            print(f"  - {test_name}")
    else:
        print("\nAll discovered tests are properly defined in test_config yaml files")

    if result.unknown:
        print(
            "\nERROR: test_config yaml files contain entries "
            "not found in discovered tests."
        )
        for test_name in sorted(result.unknown):
            print(f"  - {test_name}")
            suggestion = difflib.get_close_matches(
                test_name, result.discovered_ids, n=1
            )
            if suggestion:
                print(f"    Did you mean: {suggestion[0]}?")
    else:
        print("\nAll test_config entries match discovered tests")


def main() -> int:
    """Entry point: auto-detect project root, run validation, print results.

    Returns:
        0 on success, 1 on failure.
    """
    # Auto-detect: this script is at tests/runner/validate_test_config.py
    project_root = Path(__file__).resolve().parent.parent.parent

    models_root = project_root / "third_party" / "tt_forge_models"
    if not models_root.exists():
        print(f"ERROR: Models directory not found: {models_root}")
        return 1

    print("=" * 60)
    print("VALIDATING TEST CONFIGURATIONS")
    print("=" * 60 + "\n")

    result = TestConfigValidator(project_root).validate()
    _print_result(result)

    print("\n" + "=" * 60)
    if result.passed:
        print("VALIDATION PASSED")
    print("=" * 60)
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
