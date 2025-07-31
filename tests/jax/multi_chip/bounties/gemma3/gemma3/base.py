import logging
import os
import shutil
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp  # type: ignore
from flax import nnx
from huggingface_hub import snapshot_download  # type: ignore
from safetensors import safe_open

# Set up logging
logger = logging.getLogger(__name__)

DEFAULT_PARAMS_FILE = "jaxgarden_state"


@dataclass
class BaseConfig:
    """Base configuration for all the models implemented in the JAXgarden library.

    Each model implemented in JAXgarden should subclass this class for configuration management.
    """

    seed: int = 42
    log_level: str = "info"
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__

    def update(self, **kwargs: dict) -> None:
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            else:
                self.extra[k] = v


class BaseModel(nnx.Module):
    """Base class for all the models implemented in the JAXgarden library."""

    def __init__(
        self,
        config: BaseConfig,
        *,
        dtype: jnp.dtype | None = None,
        param_dtype: jnp.dtype = jnp.float32,
        precision: jax.lax.Precision | str | None = None,
        rngs: nnx.Rngs,
    ):
        """Initialize the model.

                        Args:
                    config: config class for this model.
                    dtype: Data type in which computation is performed.
        param_dtype: Data type in which params are stored.
                    precision: Numerical precision.
                    rngs: Random number generators for param initialization etc.
        """
        self.config = config
        self.dtype = dtype
        self.param_dtype = param_dtype
        self.precision = precision
        self.rngs = rngs

    @property
    def state(self) -> nnx.State:
        """Splits state from the graph and returns it"""
        return nnx.split(self, nnx.Param, ...)[1]  # type: ignore

    @property
    def state_dict(self) -> dict[str, jnp.ndarray]:
        """Splits state from the graph and returns it as a dictionary.

        It can be used for serialization with orbax."""
        state = self.state
        pure_dict_state = nnx.to_pure_dict(state)
        return pure_dict_state

    def save(self, path: str) -> None:
        """Saves the model state to a directory.

        Args:
            path: The directory path to save the model state to.
        """
        state = self.state_dict
        checkpointer = ocp.StandardCheckpointer()
        checkpointer.save(os.path.join(path, DEFAULT_PARAMS_FILE), state)
        checkpointer.wait_until_finished()

    def load(self, path: str) -> nnx.Module:
        """Loads the model state from a directory.

        Args:
            path: The directory path to load the model state from.
        """
        checkpointer = ocp.StandardCheckpointer()
        restored_pure_dict = checkpointer.restore(os.path.join(path, DEFAULT_PARAMS_FILE))
        abstract_model = nnx.eval_shape(lambda: self)
        graphdef, abstract_state = nnx.split(abstract_model)
        nnx.replace_by_pure_dict(abstract_state, restored_pure_dict)
        return nnx.merge(graphdef, abstract_state)

    @staticmethod
    def download_from_hf(
        repo_id: str, local_dir: str, token: str | None = None, force_download: bool = False
    ) -> None:
        """Downloads the model from the Hugging Face Hub.

        Args:
            repo_id: The repository ID of the model to download.
            local_dir: The local directory to save the model to.
            token: The hf auth token to download the model with.
              - If `True`, the token is read from the HuggingFace config
                folder.
              - If a string, it's used as the authentication token.
            force_download (`bool`, *optional*, defaults to `False`):
              Whether the file should be downloaded even if it already exists in the local cache.
        """
        logger.info(f"Attempting to download {repo_id} from Hugging Face Hub to {local_dir}.")
        try:
            snapshot_download(
                repo_id, local_dir=local_dir, token=token, force_download=force_download
            )
            logger.info(f"Successfully downloaded {repo_id} to {local_dir}.")
        except Exception as e:
            logger.error(f"Failed to download {repo_id}: {e}")
            raise

    @staticmethod
    def iter_safetensors(path_to_model_weights: str) -> Iterator[tuple[Any, Any]]:
        """Helper function to lazily load params from safetensors file.

        Use this static method to iterate over weights for conversion tasks.

        Args:
            path_to_model_weights: Path to directory containing .safetensors files."""
        if not os.path.isdir(path_to_model_weights):
            raise ValueError(f"{path_to_model_weights} is not a valid directory.")

        safetensors_files = Path(path_to_model_weights).glob("*.safetensors")

        for file in safetensors_files:
            with safe_open(file, framework="jax", device="cpu") as f:
                for key in f.keys():  # noqa: SIM118
                    yield key, f.get_tensor(key)

    def from_hf(
        self,
        model_repo_or_id: str,
        token: str | None = None,
        force_download: bool = False,
        save_in_orbax: bool = True,
        remove_hf_after_conversion: bool = True,
    ) -> None:
        """Downloads the model from the Hugging Face Hub and returns a new instance of the model.

        It can also save the converted weights in an Orbax checkpoint
            and removes the original HF checkpoint after conversion.

        Args:
            model_repo_or_id: The repository ID or name of the model to download.
            token: The token to use for authentication with the Hugging Face Hub.
            force_download: (`bool`, *optional*, defaults to `False`):
              Whether the file should be downloaded even if it already exists in the local cache.
            save_in_orbax: Whether to save the converted weights in an Orbax checkpoint.
            remove_hf_after_conversion: Whether to remove the downloaded HuggingFace checkpoint
                after conversion.
        """
        logger.info(f"Starting from_hf process for model: {model_repo_or_id}")
        local_dir = os.path.join(
            os.path.expanduser("~"), ".jaxgarden", "hf_models", *model_repo_or_id.split("/")
        )
        save_dir = local_dir.replace("hf_models", "models")
        if os.path.exists(save_dir):
            if force_download:
                logger.warning(f"Removing {save_dir} because force_download is set to True")
                shutil.rmtree(save_dir)
            else:
                raise RuntimeError(
                    f"Path {save_dir} already exists."
                    + " Set force_download to Tru to run conversion again."
                )

        logger.debug(f"Local Hugging Face model directory set to: {local_dir}")

        BaseModel.download_from_hf(
            model_repo_or_id, local_dir, token=token, force_download=force_download
        )
        logger.info(f"Initiating weight iteration from safetensors in {local_dir}")
        weights = BaseModel.iter_safetensors(local_dir)
        state = self.state
        logger.info("Running weight conversion...")
        self.convert_weights_from_hf(state, weights)
        logger.info("Weight conversion finished. Updating model state...")
        nnx.update(self, state)
        logger.warning("Model state successfully updated with converted weights.")

        if remove_hf_after_conversion:
            logger.warning(f"Removing HuggingFace checkpoint from {local_dir}...")
            shutil.rmtree(local_dir)

        if save_in_orbax:
            logger.warning(f")Saving Orbax checkpoint in {save_dir}.")
            self.save(save_dir)

        logger.warning(f"from_hf process completed for {model_repo_or_id}.")

    def convert_weights_from_hf(self, state: nnx.State, weights: Iterator[tuple[Any, Any]]) -> None:
        """Convert weights from Hugging Face Hub to the model's state.

        This method should be implemented in downstream classes
        to support conversion from HuggingFace format.
        """
        raise NotImplementedError("This model does not support conversion from HuggingFace yet.")
