# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import torch
from infra.connectors.torch_device_connector import TorchDeviceConnector
from infra.evaluators import (
    ComparisonConfig,
    EvaluatorFactory,
    EvaluatorType,
    QualityResult,
)
from infra.utilities import sanitize_test_name
from tt_torch import parse_compiled_artifacts_from_cache_to_disk

from .quality_tester import QualityTester


class StableDiffusionTester(QualityTester):
    """
    Quality tester for stable-diffusion-like image generation models.

    This definition assumes that the pipeline-like class will be provided. The benefit of assuming a StableDiffusion architecture
    is that we can assume that there will be a Unet model that will be used to generate images.
    Example usage:
        tester = StableDiffusionTester(
            pipeline_cls=SDXLPipeline,
            pipeline_config=SDXLConfig(width=512, height=512),
            dataset=CocoDataset(),
            metric="clip",
        )
        tester.test()
    """

    def __init__(
        self,
        pipeline_cls: Type[Any],
        pipeline_config: Any,
        dataset: Any,
        metric: str,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        warmup: bool = True,
        seed: int = 42,
    ) -> None:
        """
        Initialize the diffusion tester.

        Args:
            pipeline_cls: The pipeline class to instantiate (e.g., SDXLPipeline)
            pipeline_config: Configuration object for the pipeline (e.g., SDXLConfig)
            dataset: Dataset providing captions/prompts for generation
            metric: Metric name string for evaluation
            comparison_config: Configuration for quality thresholds
            warmup: Whether to run warmup inference (default: True)
            seed: Random seed for reproducibility (default: 42)
        """
        super().__init__(comparison_config=comparison_config)

        self._pipeline_cls = pipeline_cls
        self._pipeline_config = pipeline_config
        self._dataset = dataset
        self._seed = seed

        metric_kwargs = {
            "statistics_mean": self._dataset.statistics_mean,
            "statistics_cov": self._dataset.statistics_cov,
        }
        self._evaluator = EvaluatorFactory.create_evaluator(
            EvaluatorType.QUALITY,
            quality_config=comparison_config.quality,
            metric=metric,
            metric_kwargs=metric_kwargs,
        )

        # Will be set during compute_metrics
        self._pipeline: Optional[Any] = None
        self._images: Optional[torch.Tensor] = None
        self._quality_result: Optional[QualityResult] = None

    def compute_metrics(self) -> Dict[str, Any]:
        """
        Set up the pipeline, generate images, and compute quality metrics.

        Returns:
            Dictionary containing:
                - "clip_mean": Mean CLIP score across all samples (if CLIP metric)
                - "clip_min": Minimum CLIP score across all samples (if CLIP metric)
                - "fid_score": FID score (if FID metric)
                - "num_samples": Number of images generated
        """
        # Set up pipeline
        self._pipeline = self._pipeline_cls(config=self._pipeline_config)
        self._pipeline.setup(warmup=True)

        # Get captions from dataset
        captions: List[str] = self._dataset.captions

        # Generate images for each caption
        images = []
        for caption in captions:
            img = self._pipeline.generate(caption, seed=self._seed)
            images.append(img)

        # Concatenate all images into a single tensor
        self._images = torch.cat(images, dim=0)

        # Use the quality evaluator to compute metrics
        self._quality_result = self._evaluator.evaluate(self._images, prompts=captions)

        # Build return dict based on available metrics
        result: Dict[str, Any] = {
            "num_samples": self._quality_result.num_samples,
        }
        if self._quality_result.clip_mean is not None:
            result["clip_mean"] = self._quality_result.clip_mean
        if self._quality_result.clip_min is not None:
            result["clip_min"] = self._quality_result.clip_min
        if self._quality_result.fid_score is not None:
            result["fid_score"] = self._quality_result.fid_score

        return result

    def assert_quality(self, metrics: Dict[str, Any]) -> None:
        """
        Assert that quality metrics meet the configured thresholds.

        Args:
            metrics: Dictionary of computed metrics from compute_metrics()

        Raises:
            AssertionError: If quality check failed
        """
        if self._quality_result is None:
            raise RuntimeError("assert_quality called before compute_metrics")

        # Log metrics based on what's available
        if self._quality_result.clip_min is not None:
            logging.info(f"CLIP score (min): {self._quality_result.clip_min:.2f}")
            logging.info(
                f"CLIP threshold: {self._comparison_config.quality.min_clip_threshold:.2f}"
            )
        if self._quality_result.fid_score is not None:
            logging.info(f"FID score: {self._quality_result.fid_score:.2f}")
            logging.info(
                f"FID threshold: {self._comparison_config.quality.max_fid_threshold:.2f}"
            )

        # Use the evaluator's pass/fail determination
        if not self._quality_result.passed:
            raise AssertionError(self._quality_result.error_message)

    @property
    def images(self) -> Optional[torch.Tensor]:
        """Returns the generated images after compute_metrics() has been called."""
        return self._images

    def serialize_on_device(self, output_prefix: str) -> None:
        """
        Serialize the UNet compilation artifacts to disk.

        This clears the torch compile cache, sets up the pipeline,
        then extracts and saves the compilation artifacts.

        Args:
            output_prefix: Base path and filename prefix for output files
                           (creates {prefix}_ttir.mlir, {prefix}_ttnn.mlir, {prefix}.ttnn)
        """

        # Clear the torch compile cache to ensure clean serialization
        cache_dir = TorchDeviceConnector.get_cache_dir()
        cache_dir_path = Path(cache_dir)
        if cache_dir_path.exists():
            shutil.rmtree(cache_dir_path)
        cache_dir_path.mkdir(parents=True, exist_ok=True)

        if self._pipeline is None:
            self._pipeline = self._pipeline_cls(config=self._pipeline_config)
            self._pipeline.setup(warmup=True)

        # Extract and save compilation artifacts from cache
        parse_compiled_artifacts_from_cache_to_disk(cache_dir, output_prefix)

    def serialize_compilation_artifacts(self, test_name: str) -> None:
        """
        Serialize the pipeline's UNet compilation artifacts with a sanitized filename.

        Args:
            test_name: Test name to generate output prefix from
        """
        clean_name = sanitize_test_name(test_name)
        output_prefix = f"output_artifact/{clean_name}"
        self.serialize_on_device(output_prefix)
