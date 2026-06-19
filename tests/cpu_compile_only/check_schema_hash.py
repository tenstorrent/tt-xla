# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Compare the flatbuffer schema hash of two system descriptors (`*.ttsys`).

    python tests/cpu_compile_only/check_schema_hash.py a.ttsys b.ttsys
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple


def _binary_module():
    # Try, in order: installed ttrt package, top-level _ttmlir_runtime binding
    # (installed wheel), then the tt-mlir source build (dev checkout).
    try:
        import ttrt.binary as binary

        return binary
    except ModuleNotFoundError:
        pass

    try:
        import _ttmlir_runtime

        return _ttmlir_runtime.binary
    except ModuleNotFoundError:
        pass

    repo_root = Path(__file__).resolve().parents[2]
    runtime_python = (
        repo_root / "third_party/tt-mlir/src/tt-mlir/build/runtime/python"
    )
    if not runtime_python.exists():
        raise ModuleNotFoundError(
            "No flatbuffer binding found: install the pjrt_plugin_tt wheel or "
            "build tt-mlir."
        )
    sys.path.insert(0, str(runtime_python))
    import _ttmlir_runtime

    return _ttmlir_runtime.binary


def get_schema_hash(system_desc_path: str) -> str:
    fbb = _binary_module().load_system_desc_from_path(system_desc_path)
    return fbb.schema_hash


def schema_hashes_match(
    system_desc_path_a: str, system_desc_path_b: str
) -> Tuple[bool, str, str]:
    hash_a = get_schema_hash(system_desc_path_a)
    hash_b = get_schema_hash(system_desc_path_b)
    return hash_a == hash_b, hash_a, hash_b


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check whether two system descriptors share a schema hash."
    )
    parser.add_argument("descriptor_a", help="Path to the first .ttsys descriptor")
    parser.add_argument("descriptor_b", help="Path to the second .ttsys descriptor")
    args = parser.parse_args()

    match, hash_a, hash_b = schema_hashes_match(args.descriptor_a, args.descriptor_b)

    print(f"{args.descriptor_a}: {hash_a}")
    print(f"{args.descriptor_b}: {hash_b}")
    if match:
        print("MATCH: schema hashes are identical")
        return 0
    print("MISMATCH: schema hashes differ")
    return 1


if __name__ == "__main__":
    sys.exit(main())
