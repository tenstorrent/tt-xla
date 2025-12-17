# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import numpy as np
from typing import Any, List, Optional, Tuple, Union
from abc import ABC, abstractmethod
from scipy import linalg

from .models.clip_encoder import CLIPEncoder
from .models.inception_v3 import InceptionV3

class QualityMetric(ABC):
    """
    Abstract base class for quality metrics used in evaluating model outputs.

    Quality metrics measure application-specific output quality (e.g., CLIP score,
    FID, BLEU, etc.) rather than numeric accuracy comparisons (PCC, ATOL).

    Subclasses must implement:
        - compute(): Calculate and return metric value(s)
        - name: Property returning the metric name
    """

    @abstractmethod
    def compute(
        self, images: torch.Tensor, prompts: Optional[List[str]] = None, **kwargs
    ) -> Any:
        """
        Compute the quality metric.

        Args:
            images: Generated images tensor, typically (B, C, H, W)
            prompts: Optional text prompts used for generation
            **kwargs: Additional metric-specific parameters

        Returns:
            Metric value(s) - can be scalar, tuple, or dict depending on metric
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return metric name for logging/debugging."""
        pass


class CLIPMetric(QualityMetric):
    """
    CLIP score metric for measuring image-text alignment.

    Measures how well generated images match their text prompts using CLIP embeddings.
    Returns both mean and min scores across the batch for robust quality assessment.

    CLIP (Contrastive Language-Image Pre-training) provides a semantic similarity
    score between images and text, making it ideal for evaluating text-to-image
    generation quality without requiring reference images.
    """

    def __init__(self):
        """Initialize CLIP metric with pretrained CLIP encoder."""
        self._clip_model = CLIPEncoder()
        self._clip_model.eval()

    @torch.no_grad()
    def compute(
        self, images: torch.Tensor, prompts: List[str], **kwargs
    ) -> Tuple[float, float]:
        """
        Compute CLIP scores for images and prompts.

        Args:
            images: Tensor of shape (N, 3, H, W)
            prompts: List of N text prompts used to generate the images

        Returns:
            Tuple of (mean_score, min_score):
                - mean_score: Average CLIP score across all samples
                - min_score: Minimum CLIP score (worst-case quality indicator)

        Raises:
            AssertionError: If image tensor shape is invalid or prompt count mismatches
        """
        assert (
            images.ndim == 4 and images.shape[1] == 3
        ), "Images must be (N, 3, H, W)"
        assert len(prompts) == images.shape[0], (
            f"Number of prompts ({len(prompts)}) must match "
            f"number of images ({images.shape[0]})"
        )

        clip_scores = []
        for prompt, image in zip(prompts, images):
            score = self._clip_model.get_clip_score(prompt, image)
            clip_scores.append(100 * score.item())

        return np.mean(clip_scores), np.min(clip_scores)

    @property
    def name(self) -> str:
        """Return metric name."""
        return "clip"


class FIDMetric(QualityMetric):
    """
    Frechet Inception Distance (FID) metric for image generation quality.

    FID measures the distance between distributions of generated and real images
    using Inception-v3 activations. Lower FID scores indicate better quality and
    diversity of generated images.

    Unlike CLIP which measures prompt alignment, FID measures how statistically
    similar the generated images are to a reference distribution (e.g., COCO dataset).
    """

    MAX_BATCH_SIZE = 16

    def __init__(self, statistics_mean: np.ndarray, statistics_cov: np.ndarray):
        """
        Initialize FID metric with reference dataset statistics.

        Args:
            statistics_mean: Mean activations from reference dataset (shape: D,)
            statistics_cov: Covariance matrix from reference dataset (shape: D, D)
        """
        self.statistics_mean = statistics_mean
        self.statistics_cov = statistics_cov
        self._inception_model = InceptionV3()
        self._inception_model.eval()

    @torch.no_grad()
    def compute(
        self,
        images: torch.Tensor,
        prompts: Optional[List[str]] = None,
        **kwargs,
    ) -> float:
        """
        Compute FID score for generated images.

        Args:
            images: Tensor of shape (N, 3, H, W) in range [0, 1]
            prompts: Ignored for FID (not needed)
            **kwargs: Additional arguments (ignored)

        Returns:
            FID score (float) - lower is better

        Raises:
            AssertionError: If image tensor shape is invalid
        """
        assert (
            images.ndim == 4 and images.shape[1] == 3
        ), "Images must be a tensor of shape (N, 3, H, W)"

        # Batch processing if needed
        if images.shape[0] > self.MAX_BATCH_SIZE:
            batches = [
                images[i : min(i + self.MAX_BATCH_SIZE, images.shape[0])]
                for i in range(0, images.shape[0], self.MAX_BATCH_SIZE)
            ]
        else:
            batches = [images]

        # Extract activations
        activations = []
        for batch in batches:
            batch = self._normalize_images(batch)
            batch_activations = self._inception_model(batch)
            activations.extend(batch_activations)

        activations = torch.cat(activations, dim=0)
        generated_mean = activations.mean(dim=0)
        generated_cov = activations.T.cov()

        # Compute FID
        fid = self.calculate_frechet_distance(
            generated_mean,
            generated_cov,
            self.statistics_mean,
            self.statistics_cov,
        )
        return fid.item()

    def _normalize_images(self, images: torch.Tensor) -> torch.Tensor:
        """Min-max normalize images to range [0, 1]."""
        return (images - images.min()) / (images.max() - images.min())

    @staticmethod
    def calculate_frechet_distance(
        mu1: Union[np.ndarray, torch.Tensor],
        sigma1: Union[np.ndarray, torch.Tensor],
        mu2: Union[np.ndarray, torch.Tensor],
        sigma2: Union[np.ndarray, torch.Tensor],
        eps: float = 1e-6,
    ) -> float:
        """
        Compute Frechet distance between two Gaussian distributions.

        This function is adapted from the original implementation in tt-metal repository:
        tt-metal/models/experimental/stable_diffusion_xl_base/utils/fid_score.py

        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is:
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2))

        Stable version by Dougal J. Sutherland.

        Args:
            mu1: Mean activations for generated samples
            sigma1: Covariance matrix for generated samples
            mu2: Mean activations for reference dataset
            sigma2: Covariance matrix for reference dataset
            eps: Small epsilon for numerical stability

        Returns:
            Frechet Distance (float)
        """
        # Convert tensors to numpy
        if isinstance(mu1, torch.Tensor):
            mu1 = mu1.cpu().numpy()
        if isinstance(mu2, torch.Tensor):
            mu2 = mu2.cpu().numpy()
        if isinstance(sigma1, torch.Tensor):
            sigma1 = sigma1.cpu().numpy()
        if isinstance(sigma2, torch.Tensor):
            sigma2 = sigma2.cpu().numpy()

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert (
            mu1.shape == mu2.shape
        ), "Training and test mean vectors have different lengths"
        assert (
            sigma1.shape == sigma2.shape
        ), "Training and test covariances have different dimensions"

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError(f"Imaginary component {m}")
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean

    @property
    def name(self) -> str:
        """Return metric name."""
        return "fid"



metric_registry = {
    "clip": CLIPMetric,
    "fid": FIDMetric,
}

def get_metric(metric_name: str, **kwargs) -> QualityMetric:
    """
    Get a quality metric by name.

    Args:
        metric_name: Name of the metric to get
        **kwargs: Additional arguments for the metric
    """
    if metric_name not in metric_registry:
        raise ValueError(f"Metric {metric_name} not found in registry")
    return metric_registry[metric_name](**kwargs)