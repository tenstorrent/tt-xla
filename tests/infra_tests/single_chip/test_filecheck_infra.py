# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Tests for FileCheck infrastructure utilities.
Tests the basic mechanics of filecheck_utils with existing pattern files.
"""

import pytest
from tests.infra.utilities.filecheck_utils import (
    run_filecheck,
    validate_filecheck_results,
)


class TestFileCheckUtils:
    """Test filecheck utils with existing pattern files."""

    def test_matching_concatenate_heads(self, tmp_path):
        """Test that matching IR passes with concatenate_heads pattern."""
        ir_dir = tmp_path / "irs"
        ir_dir.mkdir()
        test_name = "test_concat"
        ir_file = ir_dir / f"{test_name}_ttnn.mlir"
        ir_file.write_text(
            'module {\n'
            '  func.func @main() {\n'
            '    "ttnn.concatenate_heads"() : () -> ()\n'
            '    return\n'
            '  }\n'
            '}\n'
        )

        results = run_filecheck(
            test_node_name=test_name,
            irs_filepath=str(ir_dir),
            pattern_files=["concatenate_heads.ttnn.mlir"],
        )

        assert results["concatenate_heads.ttnn"]["passed"] is True
        validate_filecheck_results(results)  # Should not raise

    def test_mismatching_concatenate_heads(self, tmp_path):
        """Test that non-matching IR fails with concatenate_heads pattern."""
        ir_dir = tmp_path / "irs"
        ir_dir.mkdir()
        test_name = "test_no_concat"
        ir_file = ir_dir / f"{test_name}_ttnn.mlir"
        ir_file.write_text(
            'module {\n'
            '  func.func @main() {\n'
            '    "ttnn.other_operation"() : () -> ()\n'
            '    return\n'
            '  }\n'
            '}\n'
        )

        results = run_filecheck(
            test_node_name=test_name,
            irs_filepath=str(ir_dir),
            pattern_files=["concatenate_heads.ttnn.mlir"],
        )

        assert results["concatenate_heads.ttnn"]["passed"] is False
        with pytest.raises(AssertionError, match="FileCheck failed"):
            validate_filecheck_results(results)

    def test_matching_split_query(self, tmp_path):
        """Test that matching IR passes with split_query pattern."""
        ir_dir = tmp_path / "irs"
        ir_dir.mkdir()
        test_name = "test_split"
        ir_file = ir_dir / f"{test_name}_ttnn.mlir"
        ir_file.write_text(
            'module {\n'
            '  func.func @main() {\n'
            '    "ttnn.split_query_key_value_and_split_heads"() : () -> ()\n'
            '    return\n'
            '  }\n'
            '}\n'
        )

        results = run_filecheck(
            test_node_name=test_name,
            irs_filepath=str(ir_dir),
            pattern_files=["split_query_key_value_and_split_heads.ttnn.mlir"],
        )

        assert results["split_query_key_value_and_split_heads.ttnn"]["passed"] is True
        validate_filecheck_results(results)

    def test_multiple_patterns(self, tmp_path):
        """Test checking multiple patterns at once."""
        ir_dir = tmp_path / "irs"
        ir_dir.mkdir()
        test_name = "test_multi"

        # Create IR with concatenate_heads
        ir_file = ir_dir / f"{test_name}_ttnn.mlir"
        ir_file.write_text('"ttnn.concatenate_heads"() : () -> ()\n')

        # Check both patterns (one should pass, one should fail)
        results = run_filecheck(
            test_node_name=test_name,
            irs_filepath=str(ir_dir),
            pattern_files=[
                "concatenate_heads.ttnn.mlir",
                "split_query_key_value_and_split_heads.ttnn.mlir",
            ],
        )

        # concatenate_heads should pass
        assert results["concatenate_heads.ttnn"]["passed"] is True
        # split_query should fail
        assert results["split_query_key_value_and_split_heads.ttnn"]["passed"] is False

    def test_missing_ir_file(self, tmp_path):
        """Test behavior when IR file doesn't exist for pattern."""
        ir_dir = tmp_path / "irs"
        ir_dir.mkdir()
        test_name = "test_missing"
        # Don't create any IR file

        results = run_filecheck(
            test_node_name=test_name,
            irs_filepath=str(ir_dir),
            pattern_files=["concatenate_heads.ttnn.mlir"],
        )

        assert results["concatenate_heads.ttnn"]["checked"] is False
        assert results["concatenate_heads.ttnn"]["passed"] is None
        assert "No IR files found" in results["concatenate_heads.ttnn"]["error"]

    def test_collect_multiple_failures(self, tmp_path):
        """Test collecting multiple failures without stopping at first."""
        ir_dir = tmp_path / "irs"
        ir_dir.mkdir()
        test_name = "test_failures"

        # Create IR that matches neither pattern
        ir_file = ir_dir / f"{test_name}_ttnn.mlir"
        ir_file.write_text('"ttnn.some_other_op"() : () -> ()\n')

        results = run_filecheck(
            test_node_name=test_name,
            irs_filepath=str(ir_dir),
            pattern_files=[
                "concatenate_heads.ttnn.mlir",
                "split_query_key_value_and_split_heads.ttnn.mlir",
            ],
        )

        # Both should fail
        assert results["concatenate_heads.ttnn"]["passed"] is False
        assert results["split_query_key_value_and_split_heads.ttnn"]["passed"] is False

        # Collect all errors without stopping
        errors = []
        for pattern, result in results.items():
            if not result["passed"]:
                errors.append(f"FileCheck failed for pattern '{pattern}': {result['error']}")

        # Should have both errors
        assert len(errors) == 2
        assert "concatenate_heads.ttnn" in errors[0]
        assert "split_query_key_value_and_split_heads.ttnn" in errors[1]
