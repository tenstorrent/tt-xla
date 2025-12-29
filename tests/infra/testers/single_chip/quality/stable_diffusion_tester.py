# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import torch
from infra.connectors.torch_device_connector import TorchDeviceConnector
from infra.evaluators.quality_config import QualityConfig
from infra.utilities import sanitize_test_name
from tt_torch import parse_compiled_artifacts_from_cache_to_disk

from .quality_tester import QualityTester


class StableDiffusionTester(QualityTester):
    """
    Quality tester for stable-diffusion-like image generation models.

    This definition assumes that the pipeline-like class will be provided. The benefit
    of assuming a StableDiffusion architecture is that we can assume that there will
    be a Unet model that will be used to generate images.

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
        quality_config: Optional[QualityConfig] = None,
        warmup: bool = True,
        seed: int = 42,
    ) -> None:
        """
        Initialize the diffusion tester.

        Args:
            pipeline_cls: The pipeline class to instantiate (e.g., SDXLPipeline)
            pipeline_config: Configuration object for the pipeline (e.g., SDXLConfig)
            dataset: Dataset providing captions/prompts for generation
            metric: Metric name string for evaluation ('clip', 'fid')
            quality_config: Configuration for quality metrics
            warmup: Whether to run warmup inference (default: True)
            seed: Random seed for reproducibility (default: 42)
        """
        self._pipeline_cls = pipeline_cls
        self._pipeline_config = pipeline_config
        self._dataset = dataset
        self._metric_name = metric
        self._seed = seed
        self._warmup = warmup

        self._pipeline: Optional[Any] = None
        self._images: Optional[torch.Tensor] = None
        self._captions: Optional[List[str]] = None

        metric_kwargs = self._build_metric_kwargs()

        super().__init__(
            quality_config=quality_config,
            metric_kwargs=metric_kwargs,
        )

    def _build_metric_kwargs(self) -> Dict[str, Dict[str, Any]]:
        """
        Build metric kwargs, e.g., FID statistics from dataset.

        Returns:
            Dictionary mapping metric names to their initialization kwargs.
        """
        kwargs: Dict[str, Dict[str, Any]] = {}
        if self._metric_name.lower() == "fid":
            if hasattr(self._dataset, "statistics_mean") and hasattr(
                self._dataset, "statistics_cov"
            ):
                kwargs["fid"] = {
                    "statistics_mean": self._dataset.statistics_mean,
                    "statistics_cov": self._dataset.statistics_cov,
                }
            else:
                raise ValueError(
                    f"FID metric requires dataset with statistics_mean and "
                    f"statistics_cov properties. Dataset {type(self._dataset).__name__} "
                    f"does not provide these."
                )
        return kwargs

    def _get_metric_names(self) -> List[str]:
        """Return the configured metric name."""
        return [self._metric_name]

    def _generate_outputs(self) -> torch.Tensor:
        """
        Set up the pipeline and generate images.

        Returns:
            Tensor of generated images (N, C, H, W)
        """
        # Set up pipeline
        self._pipeline = self._pipeline_cls(config=self._pipeline_config)
        self._pipeline.setup(warmup=self._warmup)

        # Get captions from dataset
        self._captions = self._dataset.captions

        # Generate images for each caption
        images = []
        for caption in self._captions:
            img = self._pipeline.generate(caption, seed=self._seed)
            images.append(img)

        # Concatenate all images into a single tensor
        self._images = torch.cat(images, dim=0)
        return self._images

    def _get_prompts(self) -> Optional[List[str]]:
        """Return captions for CLIP metric."""
        return self._captions

    @property
    def images(self) -> Optional[torch.Tensor]:
        """Returns the generated images after _generate_outputs() has been called."""
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
