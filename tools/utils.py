# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
import urllib.parse
import hashlib
import requests
import torch
from tabulate import tabulate
import json
from pathlib import Path
from torch.hub import load_state_dict_from_url
import yaml
from typing import Optional, Union, List, Callable, Any
from PIL import Image
from torchvision import models, transforms
from transformers import AutoImageProcessor
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


def get_file(path):
    """Get a file from local filesystem, cache, or URL.

    This function handles both local files and URLs, retrieving from cache when available
    or downloading/fetching as needed. For URLs, it creates a unique cached filename using
    a hash of the URL to prevent collisions.

    Args:
        path: Path to a local file or URL to download

    Returns:
        Path to the file in the cache
    """
    # Check if path is a URL - handle URLs and files differently
    path_is_url = path.startswith(("http://", "https://"))

    if path_is_url:
        # Create a hash from the URL to ensure uniqueness and prevent collisions
        url_hash = hashlib.md5(path.encode()).hexdigest()[:10]

        # Get filename from URL, or create one if not available
        file_name = os.path.basename(urllib.parse.urlparse(path).path)
        if not file_name:
            file_name = f"downloaded_file_{url_hash}"
        else:
            file_name = f"{url_hash}_{file_name}"

        rel_path = Path("url_cache")
        cache_dir_fallback = Path.home() / ".cache/url_cache"
    else:
        rel_dir, file_name = os.path.split(path)
        rel_path = Path("models/tt-ci-models-private") / rel_dir
        cache_dir_fallback = Path.home() / ".cache/lfcache" / rel_dir

    # Determine the base cache directory based on environment variables
    if (
        "DOCKER_CACHE_ROOT" in os.environ
        and Path(os.environ["DOCKER_CACHE_ROOT"]).exists()
    ):
        cache_dir = Path(os.environ["DOCKER_CACHE_ROOT"]) / rel_path
    elif "LOCAL_LF_CACHE" in os.environ:
        cache_dir = Path(os.environ["LOCAL_LF_CACHE"]) / rel_path
    else:
        cache_dir = cache_dir_fallback

    file_path = cache_dir / file_name

    # Support case where shared cache is read only and file not found. Can read files from it, but
    # fall back to home dir cache for storing downloaded files. Common w/ CI cache shared w/ users.
    cache_dir_rdonly = not os.access(cache_dir, os.W_OK)
    if not file_path.exists() and cache_dir_rdonly and cache_dir != cache_dir_fallback:
        print(
            f"Warning: {cache_dir} is read-only, using {cache_dir_fallback} for {path}"
        )
        cache_dir = cache_dir_fallback
        file_path = cache_dir / file_name

    cache_dir.mkdir(parents=True, exist_ok=True)

    # If file is not found in cache, download URL from web, or get file from IRD_LF_CACHE web server.
    if not file_path.exists():
        if path_is_url:
            try:
                print(f"Downloading file from URL {path} to {file_path}")
                response = requests.get(path, stream=True, timeout=(15, 60))
                response.raise_for_status()  # Raise exception for HTTP errors

                with open(file_path, "wb") as f:
                    f.write(response.content)

            except Exception as e:
                raise RuntimeError(f"Failed to download {path}: {str(e)}")
        elif "DOCKER_CACHE_ROOT" in os.environ:
            raise FileNotFoundError(
                f"File {file_path} is not available, check file path. If path is correct, DOCKER_CACHE_ROOT syncs automatically with S3 bucket every hour so please wait for the next sync."
            )
        else:
            if "IRD_LF_CACHE" not in os.environ:
                raise ValueError(
                    "IRD_LF_CACHE environment variable is not set. Please set it to the address of the IRD LF cache."
                )
            print(f"Downloading file from path {path} to {cache_dir}/{file_name}")
            exit_code = os.system(
                f"wget -nH -np -R \"indexg.html*\" -P {cache_dir} {os.environ['IRD_LF_CACHE']}/{path} --connect-timeout=15 --read-timeout=60 --tries=3"
            )
            # Check for wget failure
            if exit_code != 0:
                raise RuntimeError(
                    f"wget failed with exit code {exit_code} when downloading {os.environ['IRD_LF_CACHE']}/{path}"
                )

            # Ensure file_path exists after wget command
            if not file_path.exists():
                raise RuntimeError(
                    f"Download appears to have failed: File {file_name} not found in {cache_dir} after wget command"
                )

    return file_path


def load_class_labels(file_path):
    """Load class labels from a JSON or TXT file."""
    if file_path.endswith(".json"):
        with open(file_path, "r") as f:
            class_idx = json.load(f)
        return [class_idx[str(i)][1] for i in range(len(class_idx))]
    elif file_path.endswith(".txt"):
        with open(file_path, "r") as f:
            return [line.strip() for line in f if line.strip()]


def print_compiled_model_results(compiled_model_out, use_1k_labels: bool = True):
    if use_1k_labels:
        imagenet_class_index_path = str(
            get_file(
                "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
            )
        )
    else:
        imagenet_class_index_path = str(
            get_file(
                "https://raw.githubusercontent.com/mosjel/ImageNet_21k_Original_OK/main/imagenet_21k_original_OK.txt"
            )
        )

    class_labels = load_class_labels(imagenet_class_index_path)

    # Get top-1 class index and probability
    compiled_model_top1_probabilities, compiled_model_top1_class_indices = torch.topk(
        compiled_model_out[0].softmax(dim=1) * 100, k=1
    )
    compiled_model_top1_class_idx = compiled_model_top1_class_indices[0, 0].item()
    compiled_model_top1_class_prob = compiled_model_top1_probabilities[0, 0].item()

    # Get class label
    compiled_model_top1_class_label = class_labels[compiled_model_top1_class_idx]

    # Prepare results
    table = [
        ["Metric", "Compiled Model"],
        ["Top 1 Predicted Class Label", compiled_model_top1_class_label],
        ["Top 1 Predicted Class Probability", compiled_model_top1_class_prob],
    ]
    print(tabulate(table, headers="firstrow", tablefmt="grid"))


def get_state_dict(self, *args, **kwargs):
    kwargs.pop("check_hash")
    return load_state_dict_from_url(self.url, *args, **kwargs)


def generate_no_cache(max_new_tokens, model, inputs, seq_len, tokenizer):
    """
    Generates text autoregressively without using a KV cache, iteratively predicting one token at a time.
    The function stops generation if the maximum number of new tokens is reached or an end-of-sequence (EOS) token is encountered.

    Args:
        max_new_tokens (int): The maximum number of new tokens to generate.
        model (torch.nn.Module): The language model used for token generation.
        inputs (torch.Tensor): Input tensor of shape (batch_size, seq_len), representing tokenized text.
        seq_len (int): The current sequence length before generation starts.
        tokenizer: The tokenizer used to decode token IDs into text.

    Returns:
        str: The generated text after decoding the new tokens.
    """
    current_pos = seq_len

    for _ in range(max_new_tokens):
        logits = model(inputs)

        # Get only the logits corresponding to the last valid token
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        elif isinstance(logits, torch.Tensor):
            logits = logits
        else:
            raise TypeError(
                f"Expected logits to be a list or tuple or torch.Tensor, but got {type(logits)}"
            )
        next_token_logits = logits[:, current_pos - 1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1)
        # Stop if EOS token is encountered
        if next_token_id.item() == tokenizer.eos_token_id:
            break

        inputs[:, current_pos] = next_token_id

        current_pos += 1  # Move to next position

    # Decode valid tokens
    valid_tokens = inputs[:, seq_len:current_pos].view(-1).tolist()
    answer = tokenizer.decode(valid_tokens, skip_special_tokens=True)

    return answer


def pad_inputs(inputs, max_new_tokens=512):
    batch_size, seq_len = inputs.shape
    max_seq_len = seq_len + max_new_tokens
    padded_inputs = torch.zeros(
        (batch_size, max_seq_len), dtype=inputs.dtype, device=inputs.device
    )
    padded_inputs[:, :seq_len] = inputs
    return padded_inputs, seq_len


def cast_input_to_type(
    tensor: torch.Tensor, dtype_override: Optional[torch.dtype]
) -> torch.Tensor:
    """Cast tensor to dtype_override only if they are same numeric category.

    This applies casting when both tensor and dtype_override are floating-point, or both are non-floating types.
    If dtype_override is None or categories differ, returns tensor unchanged.
    """
    if dtype_override is None:
        return tensor

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected a torch.Tensor, got {type(tensor)}")

    tensor_is_float = tensor.dtype.is_floating_point
    override_is_float = getattr(dtype_override, "is_floating_point", None)
    if override_is_float is None:
        raise TypeError(
            "dtype_override must be a torch.dtype with is_floating_point attribute"
        )

    if tensor_is_float == override_is_float:
        return tensor.to(dtype_override)

    return tensor


# Vision utilities for image preprocessing and postprocessing
# These classes provide unified input/output processing for vision models


class VisionPreprocessor:
    """Generalized image preprocessor for vision models.

    This class handles image preprocessing for vision models from different sources
    (HuggingFace, TIMM, Torchvision, Custom) with configurable parameters.

    Supports:
    - Multiple input types: PIL Images, URLs, tensors, lists of images
    - Multiple model sources: HuggingFace, TIMM, Torchvision, Custom
    - High-resolution inputs via high_res_size parameter
    - Batch processing with different images
    - Dtype overrides for model inputs
    """

    def __init__(
        self,
        model_source,
        model_name: str,
        high_res_size: Optional[tuple] = None,
        default_image_url: str = "http://images.cocodataset.org/val2017/000000039769.jpg",
        weight_class_name_fn: Optional[Callable[[str], str]] = None,
        image_processor_kwargs: Optional[dict] = None,
        custom_preprocess_fn: Optional[Callable[[Image.Image], torch.Tensor]] = None,
    ):
        """Initialize the vision preprocessor.

        Args:
            model_source: Source of the model (HUGGING_FACE, TIMM, TORCHVISION, or CUSTOM)
            model_name: Name of the pretrained model (used for loading processors/weights)
            high_res_size: Optional tuple (width, height) for high-resolution input.
                         If None, uses default model input size.
            default_image_url: Default URL to use when image is None
            weight_class_name_fn: Required function to transform model_name to weight class name
                                for torchvision models (e.g., "resnet50" -> "ResNet50_Weights").
                                Must be provided when model_source is TORCHVISION.
            image_processor_kwargs: Optional dict of kwargs to pass to AutoImageProcessor.from_pretrained
            custom_preprocess_fn: Optional custom preprocessing function for CUSTOM source.
                                Must accept PIL.Image and return torch.Tensor.
                                Required when model_source is CUSTOM.
        """
        from ..config import ModelSource

        self.model_source = model_source
        self.model_name = model_name
        self.high_res_size = high_res_size
        self.default_image_url = default_image_url
        self.weight_class_name_fn = weight_class_name_fn
        self.image_processor_kwargs = image_processor_kwargs or {}
        self.custom_preprocess_fn = custom_preprocess_fn

        # Cache for image processor and model (for TIMM config)
        self._image_processor = None
        self._cached_model = None

    def set_cached_model(self, model):
        """Set a cached model instance (useful for TIMM models that need model for config).

        Args:
            model: The model instance to cache
        """
        self._cached_model = model

    def _load_image_from_source(
        self, image_source: Optional[Union[Image.Image, str]]
    ) -> Image.Image:
        """Load a PIL Image from various sources.

        Args:
            image_source: Can be PIL.Image.Image, URL string, or None

        Returns:
            PIL.Image.Image: Loaded and converted RGB image
        """
        if image_source is None:
            image_file = get_file(self.default_image_url)
            return Image.open(image_file).convert("RGB")
        elif isinstance(image_source, str):
            image_file = get_file(image_source)
            return Image.open(image_file).convert("RGB")
        elif isinstance(image_source, Image.Image):
            return image_source
        else:
            raise TypeError(
                f"image_source must be PIL.Image.Image, str (URL), or None, "
                f"got {type(image_source)}"
            )

    def _preprocess_huggingface(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for HuggingFace models.

        Args:
            image: PIL Image to preprocess

        Returns:
            torch.Tensor: Preprocessed tensor with batch dimension
        """
        if self._image_processor is None:
            self._image_processor = AutoImageProcessor.from_pretrained(
                self.model_name, **self.image_processor_kwargs
            )

        # Resize if high_res_size is specified
        if self.high_res_size is not None:
            image = image.resize(self.high_res_size)

        inputs = self._image_processor(
            images=image,
            return_tensors="pt",
            do_resize=False if self.high_res_size is not None else True,
        ).pixel_values

        return inputs

    def _preprocess_timm(
        self, image: Image.Image, model_for_config=None
    ) -> torch.Tensor:
        """Preprocess image for TIMM models.

        Args:
            image: PIL Image to preprocess
            model_for_config: Optional model instance to use for data config.
                            If None, uses cached model or loads one.

        Returns:
            torch.Tensor: Preprocessed tensor with batch dimension
        """
        # Use provided model, cached model, or load one
        if model_for_config is None:
            if self._cached_model is not None:
                model_for_config = self._cached_model
            else:
                # Load model temporarily for config
                model_for_config = timm.create_model(self.model_name, pretrained=True)

        data_config = resolve_data_config({}, model=model_for_config)

        if self.high_res_size is not None:
            data_config["crop_pct"] = 1.0  # Avoid center crop
            data_config["input_size"] = (
                3,
                self.high_res_size[1],  # height
                self.high_res_size[0],  # width
            )

        timm_transforms = create_transform(**data_config)
        inputs = timm_transforms(image).unsqueeze(0)
        return inputs

    def _preprocess_torchvision(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for Torchvision models.

        Args:
            image: PIL Image to preprocess

        Returns:
            torch.Tensor: Preprocessed tensor with batch dimension

        Raises:
            ValueError: If weight_class_name_fn is not provided for torchvision models
            AttributeError: If the weight class cannot be found in torchvision.models
        """
        from ..config import ModelSource

        # Determine weight class name
        if self.weight_class_name_fn is None:
            raise ValueError(
                f"weight_class_name_fn must be provided for torchvision models. "
                f"Model name: {self.model_name}. "
                f"Each torchvision model has a specific weight class naming convention "
                f"(e.g., 'resnet50' -> 'ResNet50_Weights', 'vit_b_16' -> 'ViT_B_16_Weights'). "
                f"Please provide a function that transforms the model name to the weight class name."
            )

        weight_class_name = self.weight_class_name_fn(self.model_name)

        # Try to get the weights class
        if not hasattr(models, weight_class_name):
            available_weights = [
                name
                for name in dir(models)
                if name.endswith("_Weights") and not name.startswith("_")
            ]
            raise AttributeError(
                f"Weight class '{weight_class_name}' not found in torchvision.models. "
                f"Model name: {self.model_name}. "
                f"Available weight classes: {sorted(available_weights)[:10]}..."
                f"(showing first 10 of {len(available_weights)}). "
                f"Please check your weight_class_name_fn implementation."
            )

        weights = getattr(models, weight_class_name).DEFAULT

        if self.high_res_size is not None:
            # Skip resize, just normalize
            preprocess = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=weights.transforms().mean,
                        std=weights.transforms().std,
                    ),
                ]
            )
        else:
            preprocess = weights.transforms()

        inputs = preprocess(image).unsqueeze(0)
        return inputs

    def preprocess_single_image(
        self,
        image: Optional[Union[Image.Image, str, torch.Tensor]],
        dtype_override: Optional[torch.dtype] = None,
        model_for_config=None,
    ) -> torch.Tensor:
        """Preprocess a single image.

        Args:
            image: Input image. Can be:
                  - PIL.Image.Image: A PIL Image object
                  - str: URL string to download and load image from
                  - torch.Tensor: Pre-processed tensor (will be used as-is)
                  - None: Uses default sample image
            dtype_override: Optional torch.dtype to override the inputs' default dtype
            model_for_config: Optional model instance (for TIMM models)

        Returns:
            torch.Tensor: Preprocessed input tensor with batch dimension
        """
        from ..config import ModelSource

        # Handle pre-processed tensor
        if isinstance(image, torch.Tensor):
            inputs = image.unsqueeze(0) if image.dim() == 3 else image
            if dtype_override is not None:
                inputs = inputs.to(dtype_override)
            return inputs

        # Load image from source
        img = self._load_image_from_source(image)

        # Resize if high_res_size is specified (before source-specific preprocessing)
        if (
            self.high_res_size is not None
            and self.model_source != ModelSource.HUGGING_FACE
        ):
            # HuggingFace handles resize in processor
            img = img.resize(self.high_res_size)

        # Preprocess based on source
        if self.model_source == ModelSource.HUGGING_FACE:
            inputs = self._preprocess_huggingface(img)
        elif self.model_source == ModelSource.TIMM:
            inputs = self._preprocess_timm(img, model_for_config)
        elif self.model_source == ModelSource.TORCHVISION:
            inputs = self._preprocess_torchvision(img)
        elif self.model_source == ModelSource.CUSTOM:
            # For custom models, allow custom preprocessing function
            if self.custom_preprocess_fn is None:
                raise ValueError(
                    f"Custom preprocessing function required for CUSTOM source. "
                    f"Provide custom_preprocess_fn parameter in __init__ or use a supported source."
                )
            inputs = self.custom_preprocess_fn(img)
            if not isinstance(inputs, torch.Tensor):
                raise TypeError(
                    f"custom_preprocess_fn must return torch.Tensor, got {type(inputs)}"
                )
            inputs = inputs.unsqueeze(0) if inputs.dim() == 3 else inputs
        else:
            raise ValueError(
                f"Unsupported model source: {self.model_source}. "
                f"Supported sources: HUGGING_FACE, TIMM, TORCHVISION, CUSTOM. "
                f"To add support for {self.model_source}, extend VisionPreprocessor."
            )

        # Apply dtype override if specified
        if dtype_override is not None:
            inputs = inputs.to(dtype_override)

        return inputs

    def preprocess(
        self,
        image: Optional[
            Union[Image.Image, str, torch.Tensor, List[Union[Image.Image, str]]]
        ] = None,
        dtype_override: Optional[torch.dtype] = None,
        batch_size: int = 1,
        model_for_config=None,
    ) -> torch.Tensor:
        """Preprocess input image(s) and return sample inputs for the model.

        Args:
            image: Optional input image(s). Can be:
                  - PIL.Image.Image: A PIL Image object to preprocess
                  - str: URL string to download and load image from
                  - torch.Tensor: Pre-processed tensor (will be used as-is after batch replication)
                  - List[Union[Image.Image, str]]: List of PIL Images or URLs for batched evaluation
                  - None: Uses default sample image
            dtype_override: Optional torch.dtype to override the inputs' default dtype
            batch_size: Batch size. If image is a list, batch_size is ignored and determined by list length.
                       If image is a single item, this replicates it batch_size times.
            model_for_config: Optional model instance (for TIMM models)

        Returns:
            torch.Tensor: Preprocessed input tensor suitable for the model
        """
        # Handle list of images for batched evaluation with different images
        if isinstance(image, list):
            if len(image) == 0:
                raise ValueError("image list cannot be empty")

            processed_inputs = []
            for img_item in image:
                if isinstance(img_item, torch.Tensor):
                    # For tensors in list, just use as-is
                    processed_inputs.append(
                        img_item.unsqueeze(0) if img_item.dim() == 3 else img_item
                    )
                else:
                    # Process single image
                    single_input = self.preprocess_single_image(
                        img_item, dtype_override, model_for_config
                    )
                    processed_inputs.append(single_input)

            # Stack all processed inputs into a batch
            return torch.cat(processed_inputs, dim=0)

        # Handle single image
        inputs = self.preprocess_single_image(image, dtype_override, model_for_config)

        # Replicate tensors for batch size (only if not already batched)
        if batch_size > 1 and not isinstance(image, list):
            inputs = inputs.repeat_interleave(batch_size, dim=0)

        return inputs


def create_vision_preprocessor(
    model_source,
    model_name: str,
    high_res_size: Optional[tuple] = None,
    default_image_url: str = "http://images.cocodataset.org/val2017/000000039769.jpg",
    weight_class_name_fn: Optional[Callable[[str], str]] = None,
    image_processor_kwargs: Optional[dict] = None,
    custom_preprocess_fn: Optional[Callable[[Image.Image], torch.Tensor]] = None,
) -> VisionPreprocessor:
    """Factory function to create a VisionPreprocessor instance.

    This is a convenience function for creating preprocessors with common configurations.

    Args:
        model_source: Source of the model (HUGGING_FACE, TIMM, TORCHVISION, or CUSTOM)
        model_name: Name of the pretrained model
        high_res_size: Optional tuple (width, height) for high-resolution input
        default_image_url: Default URL to use when image is None
        weight_class_name_fn: Required function to transform model_name to weight class name
                            for torchvision models (e.g., "resnet50" -> "ResNet50_Weights").
                            Must be provided when model_source is TORCHVISION.
        image_processor_kwargs: Optional dict of kwargs for AutoImageProcessor
        custom_preprocess_fn: Optional custom preprocessing function for CUSTOM source.
                            Required when model_source is CUSTOM.

    Returns:
        VisionPreprocessor: Configured preprocessor instance
    """
    return VisionPreprocessor(
        model_source=model_source,
        model_name=model_name,
        high_res_size=high_res_size,
        default_image_url=default_image_url,
        weight_class_name_fn=weight_class_name_fn,
        image_processor_kwargs=image_processor_kwargs,
        custom_preprocess_fn=custom_preprocess_fn,
    )


class VisionPostprocessor:
    """Generalized output postprocessor for vision classification models.

    This class handles postprocessing of model outputs to extract predictions,
    probabilities, and class labels for vision models from different sources.
    """

    def __init__(
        self,
        model_source,
        model_name: str,
        model_instance: Optional[Any] = None,
        use_1k_labels: bool = True,
        imagenet_class_index_url: Optional[str] = None,
        imagenet_21k_labels_url: Optional[str] = None,
    ):
        """Initialize the vision postprocessor.

        Args:
            model_source: Source of the model (HUGGING_FACE, TIMM, or TORCHVISION)
            model_name: Name of the pretrained model (used for loading model config if needed)
            model_instance: Optional model instance (for accessing config.id2label in HuggingFace models)
            use_1k_labels: Whether to use ImageNet-1k labels (True) or ImageNet-21k labels (False)
                          Only used for TIMM and Torchvision models
            imagenet_class_index_url: Optional custom URL for ImageNet-1k class index JSON
            imagenet_21k_labels_url: Optional custom URL for ImageNet-21k labels TXT
        """
        from ..config import ModelSource

        self.model_source = model_source
        self.model_name = model_name
        self.model_instance = model_instance
        self.use_1k_labels = use_1k_labels
        self.imagenet_class_index_url = imagenet_class_index_url or (
            "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
        )
        self.imagenet_21k_labels_url = imagenet_21k_labels_url or (
            "https://raw.githubusercontent.com/mosjel/ImageNet_21k_Original_OK/main/imagenet_21k_original_OK.txt"
        )

        # Cache for class labels
        self._class_labels_cache = None
        self._label_dict_cache = None

    def set_model_instance(self, model_instance: Any):
        """Set or update the model instance (useful for accessing config).

        Args:
            model_instance: The model instance to cache
        """
        self.model_instance = model_instance
        # Clear label dict cache if model changed
        self._label_dict_cache = None

    def _extract_logits(self, output: Union[torch.Tensor, Any]) -> torch.Tensor:
        """Extract logits tensor from various output formats.

        Args:
            output: Model output. Can be:
                  - torch.Tensor: Raw logits
                  - ModelOutput object with logits attribute (HuggingFace)
                  - list/tuple: First element is assumed to be logits

        Returns:
            torch.Tensor: Extracted logits tensor
        """
        if hasattr(output, "logits"):
            # HuggingFace ModelOutput object
            return output.logits.to("cpu")
        elif isinstance(output, (list, tuple)):
            # Some models return tuple/list of outputs
            if len(output) > 0:
                return output[0].to("cpu")
            else:
                raise ValueError("Empty output list/tuple")
        elif isinstance(output, torch.Tensor):
            # Already a tensor
            return output.to("cpu")
        else:
            raise TypeError(
                f"Unsupported output type: {type(output)}. "
                f"Expected torch.Tensor, ModelOutput, or list/tuple."
            )

    def _get_huggingface_labels(self) -> dict:
        """Get label dictionary for HuggingFace models.

        Returns:
            dict: Mapping from class index to label name
        """
        if self._label_dict_cache is not None:
            return self._label_dict_cache

        if self.model_instance is not None:
            # Use cached model instance
            if hasattr(self.model_instance, "config") and hasattr(
                self.model_instance.config, "id2label"
            ):
                self._label_dict_cache = self.model_instance.config.id2label
                return self._label_dict_cache

        # Try to load model temporarily to get config
        # This is a fallback if model_instance is not provided
        try:
            # Import here to avoid circular dependencies
            from transformers import AutoModelForImageClassification

            temp_model = AutoModelForImageClassification.from_pretrained(
                self.model_name
            )
            if hasattr(temp_model, "config") and hasattr(temp_model.config, "id2label"):
                self._label_dict_cache = temp_model.config.id2label
                return self._label_dict_cache
        except Exception as e:
            raise RuntimeError(
                f"Could not load model config for {self.model_name}. "
                f"Please provide model_instance or ensure model can be loaded. Error: {e}"
            )

        raise RuntimeError(
            f"Could not find id2label in model config for {self.model_name}"
        )

    def _get_imagenet_labels(self) -> List[str]:
        """Get ImageNet class labels.

        Returns:
            List[str]: List of class label names
        """
        if self._class_labels_cache is not None:
            return self._class_labels_cache

        if self.use_1k_labels:
            class_index_path = str(get_file(self.imagenet_class_index_url))
        else:
            class_index_path = str(get_file(self.imagenet_21k_labels_url))

        self._class_labels_cache = load_class_labels(class_index_path)
        return self._class_labels_cache

    def _get_label(self, class_idx: int) -> str:
        """Get class label for a given class index.

        Args:
            class_idx: Class index

        Returns:
            str: Class label name
        """
        from ..config import ModelSource

        if self.model_source == ModelSource.HUGGING_FACE:
            label_dict = self._get_huggingface_labels()
            return label_dict.get(class_idx, f"Unknown({class_idx})")
        else:
            # TIMM and Torchvision models use ImageNet class labels
            class_labels = self._get_imagenet_labels()
            if class_idx < len(class_labels):
                return class_labels[class_idx]
            else:
                return f"Unknown({class_idx})"

    def postprocess(
        self,
        output: Union[torch.Tensor, Any],
        top_k: int = 1,
        batch_idx: int = 0,
        return_dict: bool = True,
    ) -> Union[dict, tuple]:
        """Postprocess model output to get predictions.

        Args:
            output: Model output (tensor, ModelOutput, or list/tuple)
            top_k: Number of top predictions to return
            batch_idx: Batch index to use (if output has batch dimension)
            return_dict: If True, returns dict with 'label' and 'probability'.
                        If False, returns tuple (label, probability).

        Returns:
            dict or tuple: If return_dict=True, returns:
                          {
                              "label": str,  # Top-1 predicted class label
                              "probability": str  # Top-1 probability as percentage (e.g., "98.34%")
                          }
                          If return_dict=False, returns (label, probability) tuple.
                          For top_k > 1, returns lists of labels/probabilities.
        """
        # Extract logits
        logits = self._extract_logits(output)

        # Ensure logits is a tensor
        if not isinstance(logits, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(logits)}")

        # Check if input is already probabilities (values in [0,1] range and sum to ~1)
        # If the tensor is already probabilities, use it directly instead of applying softmax
        # This prevents double-softmax which would reduce confidence incorrectly
        logits_min = logits.min().item()
        logits_max = logits.max().item()
        # Check if values are in probability range [0,1]
        values_in_range = logits_min >= -0.01 and logits_max <= 1.01

        # Check if the tensor sums to ~1 along the last dimension (normalized probabilities)
        if values_in_range and logits.dim() > 0:
            if logits.dim() == 1:
                # 1D tensor: check if sum is close to 1
                sum_check = torch.allclose(
                    logits.sum(), torch.tensor(1.0, device=logits.device), atol=0.01
                )
            else:
                # 2D+ tensor: check if sum along last dim is close to 1 for each item
                sums = logits.sum(dim=-1)
                sum_check = torch.allclose(sums, torch.ones_like(sums), atol=0.01)
            is_already_prob = sum_check
        else:
            is_already_prob = False

        if is_already_prob:
            # Input is already probabilities, use directly
            probabilities = logits
        else:
            # Apply softmax to get probabilities
            probabilities = torch.softmax(logits, dim=-1)

        # Handle batch dimension
        if probabilities.dim() > 1:
            probs_for_batch = probabilities[batch_idx]
        else:
            probs_for_batch = probabilities

        # Get top-k predictions
        top_k = min(top_k, probs_for_batch.numel())
        top_probs, top_indices = torch.topk(probs_for_batch, k=top_k)

        # Convert to lists - ensure we extract scalar values from tensors
        # top_probs is a tensor, convert to percentage and extract values
        if top_probs.dim() > 0:
            top_probs_list = (top_probs * 100).tolist()  # Convert to percentage
        else:
            # Scalar tensor
            top_probs_list = [(top_probs * 100).item()]

        if top_indices.dim() > 0:
            top_indices_list = top_indices.tolist()
        else:
            # Scalar tensor
            top_indices_list = [top_indices.item()]

        # Get labels
        labels = [self._get_label(idx) for idx in top_indices_list]

        if top_k == 1:
            # Single prediction
            label = labels[0]
            prob_str = f"{top_probs_list[0]:.4f}%"
            if return_dict:
                return {"label": label, "probability": prob_str}
            else:
                return (label, prob_str)
        else:
            # Multiple predictions
            prob_strs = [f"{p:.4f}%" for p in top_probs_list]
            if return_dict:
                return {
                    "labels": labels,
                    "probabilities": prob_strs,
                    "indices": top_indices_list,
                }
            else:
                return (labels, prob_strs)

    def postprocess_batch(
        self,
        output: Union[torch.Tensor, Any],
        top_k: int = 1,
    ) -> List[dict]:
        """Postprocess model output for a batch of inputs.

        Args:
            output: Model output (tensor, ModelOutput, or list/tuple)
            top_k: Number of top predictions to return per batch item

        Returns:
            List[dict]: List of prediction dictionaries, one per batch item
        """
        logits = self._extract_logits(output)

        if not isinstance(logits, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(logits)}")

        # Ensure we have batch dimension
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        batch_size = logits.shape[0]
        results = []

        # Apply softmax to get probabilities
        probabilities = torch.softmax(logits, dim=-1)

        for batch_idx in range(batch_size):
            probs_for_batch = probabilities[batch_idx]

            # Get top-k predictions
            top_k_actual = min(top_k, probs_for_batch.numel())
            top_probs, top_indices = torch.topk(probs_for_batch, k=top_k_actual)

            # Convert to lists
            top_probs_list = (top_probs * 100).tolist()  # Convert to percentage
            top_indices_list = top_indices.tolist()

            # Get labels
            labels = [self._get_label(idx) for idx in top_indices_list]

            if top_k_actual == 1:
                # Single prediction
                label = labels[0]
                prob_str = f"{top_probs_list[0]:.4f}%"
                results.append({"label": label, "probability": prob_str})
            else:
                # Multiple predictions
                prob_strs = [f"{p:.4f}%" for p in top_probs_list]
                results.append(
                    {
                        "labels": labels,
                        "probabilities": prob_strs,
                        "indices": top_indices_list,
                    }
                )

        return results

    def print_results(
        self,
        co_out=None,
        framework_model=None,
        compiled_model=None,
        inputs=None,
        dtype_override=None,
        run_and_print_fn=None,
    ):
        """Print postprocessing results (legacy mode for backward compatibility).

        Args:
            co_out: Outputs from the compiled model
            framework_model: The original framework-based model
            compiled_model: The compiled version of the model
            inputs: A list of images to process and classify
            dtype_override: Optional torch.dtype to override the input's dtype
            run_and_print_fn: Optional custom function to run and print results
                            (for HuggingFace models with special handling)
        """
        from ..config import ModelSource

        if self.model_source == ModelSource.HUGGING_FACE:
            if run_and_print_fn is not None:
                run_and_print_fn(
                    framework_model, compiled_model, inputs, dtype_override
                )
            else:
                # Default behavior: extract and print
                if co_out is not None:
                    logits = self._extract_logits(co_out)
                    predicted_class_indices = logits.argmax(-1)

                    if predicted_class_indices.dim() == 0:
                        # Single prediction
                        idx = predicted_class_indices.item()
                        label = self._get_label(idx)
                        print(f"Predicted class: {label}")
                    else:
                        # Batch predictions
                        for i, idx in enumerate(predicted_class_indices):
                            label = self._get_label(idx.item())
                            print(f"Batch {i}: Predicted class: {label}")
        else:
            # TIMM and Torchvision models
            if co_out is not None:
                print_compiled_model_results(co_out, use_1k_labels=self.use_1k_labels)


def create_vision_postprocessor(
    model_source,
    model_name: str,
    model_instance: Optional[Any] = None,
    use_1k_labels: bool = True,
    imagenet_class_index_url: Optional[str] = None,
    imagenet_21k_labels_url: Optional[str] = None,
) -> VisionPostprocessor:
    """Factory function to create a VisionPostprocessor instance.

    This is a convenience function for creating postprocessors with common configurations.

    Args:
        model_source: Source of the model (HUGGING_FACE, TIMM, or TORCHVISION)
        model_name: Name of the pretrained model
        model_instance: Optional model instance (for accessing config.id2label)
        use_1k_labels: Whether to use ImageNet-1k labels (True) or ImageNet-21k labels (False)
        imagenet_class_index_url: Optional custom URL for ImageNet-1k class index JSON
        imagenet_21k_labels_url: Optional custom URL for ImageNet-21k labels TXT

    Returns:
        VisionPostprocessor: Configured postprocessor instance
    """
    return VisionPostprocessor(
        model_source=model_source,
        model_name=model_name,
        model_instance=model_instance,
        use_1k_labels=use_1k_labels,
        imagenet_class_index_url=imagenet_class_index_url,
        imagenet_21k_labels_url=imagenet_21k_labels_url,
    )
