import numpy as np
import torch
from typing import List, Union
from .clip import CLIPEncoder
from .inception import InceptionV3

class FIDMetric:
    MAX_BATCH_SIZE = 16
    def __init__(self, statistics_mean: np.ndarray, statistics_cov: np.ndarray):
        self.statistics_mean = statistics_mean
        self.statistics_cov = statistics_cov
        self._inception_model = InceptionV3()
        self._inception_model.eval()

    @torch.no_grad()
    def compute(self, images: torch.Tensor) -> float:
        """
        A method that computes the FID score for a given set of images.
        FID score is calculated with respect to the statistics mean and covariance of the real images
        that is provided in the constructor.
        """
        assert images.ndim == 4 and images.shape[1] == 3, "Images must be a tensor of shape (N, 3, H, W)"

        # if number of images is greater than we can do in one batch, we chunk the batches
        if images.shape[0] > self.MAX_BATCH_SIZE:
            images = [images[i:min(i+self.MAX_BATCH_SIZE, images.shape[0])] for i in range(0, images.shape[0], self.MAX_BATCH_SIZE)]
        else:
            images = [images]
        
        activations = []
        for batch in images:
            batch = self._normalize_images(batch)
            batch_activations = self._inception_model(batch)
            activations.extend(batch_activations)
        
        activations = torch.cat(activations, dim=0)
        generated_mean = activations.mean(dim=0)
        generated_cov = activations.T.cov()

        fid = self.calculate_frechet_distance(generated_mean, generated_cov, self.statistics_mean, self.statistics_cov)
        return fid.item()

    
    def _normalize_images(self, images: torch.Tensor) -> torch.Tensor:
        """Min max normalize the images to the range [0, 1]"""
        return (images - images.min()) / (images.max() - images.min())

    @staticmethod
    def calculate_frechet_distance(mu1: Union[np.ndarray, torch.Tensor], sigma1: Union[np.ndarray, torch.Tensor], mu2: Union[np.ndarray, torch.Tensor], sigma2: Union[np.ndarray, torch.Tensor], eps: float = 1e-6) -> float:
        """Numpy implementation of the Frechet Distance.
        This function is a copy of the original implementation in tt-metal repository:
        tt-metal/models/experimental/stable_diffusion_xl_base/utils/fid_score.py
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

        Stable version by Dougal J. Sutherland.

        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                inception net (like returned by the function 'get_predictions')
                for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                representative data set.

        Returns:
        --   : The Frechet Distance.
        """

        if isinstance(mu1, torch.Tensor):
            mu1 = mu1.cpu().numpy()
        if isinstance(mu2, torch.Tensor):
            mu2 = mu2.cpu().numpy()
        if isinstance(sigma1, torch.Tensor):
            sigma1 = sigma1.cpu().numpy()
        if isinstance(sigma2, torch.Tensor):
            sigma2 = sigma2.cpu().numpy()

        from scipy import linalg

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
        assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            logger.info(f"fid calculation produces singular product; adding {eps} to diagonal of cov estimates")
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
class CLIPMetric:
    def __init__(self):
        self._clip_model = CLIPEncoder()
        self._clip_model.eval()

    @torch.no_grad()
    def compute(self, images: torch.Tensor, prompts: List[str]) -> float:
        clip_scores = []
        for prompt, image in zip(prompts, images):
            clip_scores.append(100 * self._clip_model.get_clip_score(prompt, image).item())
        return np.mean(clip_scores), np.min(clip_scores)