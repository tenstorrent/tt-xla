# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

from infra.evaluators import EvaluatorFactory, QualityConfig, QualityResult
from infra.utilities import sanitize_test_name

from tests.infra.utilities.filecheck_utils import (
    run_filecheck,
    validate_filecheck_results,
)

FILECHECK_DIR = Path(__file__).parent.parent.parent.parent / "filecheck"


class QualityTester:
    """
    Abstract base class for quality metric-based testing.

    Does not inherit from BaseTester. Directly manages its own QualityEvaluator.
    """

    def __init__(
        self,
        quality_config: Optional[QualityConfig] = None,
        metric_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
        metric_names: Optional[List[str]] = None,
    ) -> None:
        self._last_result: Optional[QualityResult] = None
        self._quality_config = (
            quality_config if quality_config is not None else QualityConfig()
        )
        self._metric_names = metric_names or []
        self._metric_kwargs = metric_kwargs or {}
        self._evaluator = None

        if self._metric_names:
            self._evaluator = EvaluatorFactory.create_evaluator(
                evaluation_type="quality",
                quality_config=self._quality_config,
                metric_names=self._metric_names,
                metric_kwargs=self._metric_kwargs,
            )

    @abstractmethod
    def test(self, request=None) -> QualityResult:
        raise NotImplementedError("Subclasses must implement test()")

    @property
    def metrics(self) -> Dict[str, Any]:
        if self._last_result is None:
            return {}
        return self._last_result.metrics or {}

    @property
    def result(self) -> Optional[QualityResult]:
        return self._last_result

    @property
    def quality_config(self) -> QualityConfig:
        return self._quality_config

    @abstractmethod
    def serialize_on_device(self, workload=None, output_prefix: str = None) -> None:
        raise NotImplementedError("Subclasses must implement this method.")

    def serialize_compilation_artifacts(self, test_name: str, workload=None) -> None:
        clean_name = sanitize_test_name(test_name)
        output_prefix = f"output_artifact/{clean_name}"
        self.serialize_on_device(workload, output_prefix)

    def handle_filecheck_and_serialization(self, request, workload=None) -> None:
        if not request:
            return

        test_id = request.node.name

        serialize = request.config.getoption("--serialize", False)

        filecheck_marker = request.node.get_closest_marker("filecheck")
        pattern_files = (
            filecheck_marker.args[0]
            if filecheck_marker and filecheck_marker.args
            else None
        )

        if serialize or pattern_files:
            self.serialize_compilation_artifacts(test_name=test_id, workload=workload)

        if pattern_files:
            self._run_filecheck(pattern_files, test_id=test_id)

    def _run_filecheck(self, pattern_files: list, test_id: str) -> None:
        filecheck_results = run_filecheck(
            test_node_name=test_id,
            irs_filepath="output_artifact",
            pattern_files=pattern_files,
        )
        validate_filecheck_results(filecheck_results)
