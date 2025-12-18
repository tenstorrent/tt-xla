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
            min_threshold=25.0,
        )
        tester.test()
    """

    def __init__(
        self,
        pipeline_cls: Type[Any],
        pipeline_config: Any,
        dataset: Any,
        metric: str,
        min_threshold: float = 25.0,
        warmup: bool = True,
        seed: int = 42,
        device_type: str = "TT",
    ) -> None:
        """
        Initialize the diffusion tester.

        Args:
            pipeline_cls: The pipeline class to instantiate (e.g., SDXLPipeline)
            pipeline_config: Configuration object for the pipeline (e.g., SDXLConfig)
            dataset: Dataset providing captions/prompts for generation
            metric: Metric name string for evaluation ('clip', 'fid')
            min_threshold: Minimum acceptable metric score (default: 25.0 for CLIP)
            warmup: Whether to run warmup inference (default: True)
            seed: Random seed for reproducibility (default: 42)
            device_type: Device type to run on (default: "TT")
        """
        super().__init__(device_type=device_type)

        self._pipeline_cls = pipeline_cls
        self._pipeline_config = pipeline_config
        self._dataset = dataset
        self._metric = self._resolve_metric(metric)
        self._min_threshold = min_threshold
        self._seed = seed

        # Will be set during compute_metrics
        self._pipeline: Optional[Any] = None
        self._images: Optional[torch.Tensor] = None

    def _resolve_metric(self, metric: str) -> Any:
        """
        Resolve metric string to an instance.

        Args:
            metric: Metric name ('clip', 'fid')

        Returns:
            Instantiated metric object

        Raises:
            ValueError: If FID metric is requested but dataset lacks statistics
        """
        from tests.infra.metrics import get_metric

        metric_kwargs = {}

        # Auto-provide FID statistics from dataset if available
        if metric.lower() == "fid":
            if hasattr(self._dataset, "statistics_mean") and hasattr(
                self._dataset, "statistics_cov"
            ):
                metric_kwargs["statistics_mean"] = self._dataset.statistics_mean
                metric_kwargs["statistics_cov"] = self._dataset.statistics_cov
            else:
                raise ValueError(
                    f"FID metric requires dataset with statistics_mean and "
                    f"statistics_cov properties. Dataset {type(self._dataset).__name__} "
                    f"does not provide these."
                )

        return get_metric(metric, **metric_kwargs)

    def compute_metrics(self) -> Dict[str, Any]:
        """
        Set up the pipeline, generate images, and compute quality metrics.

        Returns:
            Dictionary containing:
                - "clip_mean": Mean CLIP score across all samples
                - "clip_min": Minimum CLIP score across all samples
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

        # Compute CLIP scores
        clip_mean, clip_min = self._metric.compute(self._images, captions)

        return {
            "clip_mean": clip_mean,
            "clip_min": clip_min,
            "num_samples": len(captions),
        }

    def assert_quality(self, metrics: Dict[str, Any]) -> None:
        """
        Assert that CLIP score meets the minimum threshold.

        Args:
            metrics: Dictionary of computed metrics from compute_metrics()

        Raises:
            AssertionError: If clip_min is below min_threshold
        """
        clip_min = metrics["clip_min"]
        logging.info(f"CLIP score: {clip_min:.2f}")
        logging.info(f"Minimum threshold: {self._min_threshold:.2f}")
        assert clip_min > self._min_threshold, (
            f"CLIP score regression detected: "
            f"clip_min={clip_min:.2f} < threshold={self._min_threshold:.2f}"
        )

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
