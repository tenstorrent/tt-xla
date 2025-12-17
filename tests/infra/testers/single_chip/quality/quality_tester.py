# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, Optional

import torch_xla.runtime as xr


class QualityTester:
    """
    Abstract base class for quality metric-based testing.

    Unlike OpTester/GraphTester which compare CPU vs TT device outputs using PCC,
    QualityTester runs workloads on the target device and evaluates output quality
    using application-specific metrics (e.g., CLIP score, FID, BLEU, etc.).

    Subclasses must implement:
        - compute_metrics(): Run the workload and compute quality metrics
        - assert_quality(): Validate that computed metrics meet quality thresholds
    """

    def __init__(self, device_type: str = "TT") -> None:
        """
        Initialize the quality tester.

        Args:
            device_type: The device type to run on (default: "TT" for Tenstorrent)
        """
        self._device_type = device_type
        self._metrics: Dict[str, Any] = {}

    def _setup_device(self) -> None:
        """Sets up the XLA device for testing."""
        xr.set_device_type(self._device_type)

    @abstractmethod
    def compute_metrics(self) -> Dict[str, Any]:
        """
        Run the workload and compute quality metrics.

        This method should:
            1. Set up any necessary pipelines/models
            2. Run inference on the dataset/inputs
            3. Compute and return quality metrics

        Returns:
            Dictionary of metric names to their computed values.
        """
        raise NotImplementedError("Subclasses must implement compute_metrics()")

    @abstractmethod
    def assert_quality(self, metrics: Dict[str, Any]) -> None:
        """
        Validate that computed metrics meet quality thresholds.

        This method should raise AssertionError if quality thresholds are not met.

        Args:
            metrics: Dictionary of computed metrics from compute_metrics()
        """
        raise NotImplementedError("Subclasses must implement assert_quality()")

    def test(self) -> None:
        """
        Main test entry point.

        Sets up the device, computes metrics, and validates quality thresholds.
        """
        self._setup_device()
        self._metrics = self.compute_metrics()
        self.assert_quality(self._metrics)

    @property
    def metrics(self) -> Dict[str, Any]:
        """Returns the computed metrics after test() has been called."""
        return self._metrics
