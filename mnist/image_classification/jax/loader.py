# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
MNIST model loader implementation for image classification.
"""

from typing import Optional, Sequence
import inspect
import jax
import jax.numpy as jnp
import numpy as np

from ....base import ForgeModel
from ....config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from .mlp.model_implementation import MNISTMLPModel, MNISTMLPMultichipModel
from .cnn_batchnorm.model_implementation import MNISTCNNBatchNormModel
from .cnn_dropout.model_implementation import MNISTCNNDropoutModel


class ModelVariant(StrEnum):
    """Available MNIST model architectures."""

    MLP_CUSTOM = "mlp_custom"
    MLP_CUSTOM_1X2 = "mlp_custom_1x2"
    MLP_CUSTOM_1X4 = "mlp_custom_1x4"
    MLP_CUSTOM_1X8 = "mlp_custom_1x8"
    CNN_BATCHNORM = "cnn_batchnorm"
    CNN_DROPOUT = "cnn_dropout"


class ModelLoader(ForgeModel):
    """MNIST model loader implementation for image classification."""

    # Default random seed for parameter initialization
    DEFAULT_PARAMS_INIT_SEED = 42

    # Dictionary of available model configurations
    _VARIANTS = {
        ModelVariant.MLP_CUSTOM: ModelConfig(
            pretrained_model_name="mnist_mlp_custom",
        ),
        ModelVariant.MLP_CUSTOM_1X2: ModelConfig(
            pretrained_model_name="mnist_mlp_custom_1x2",
        ),
        ModelVariant.MLP_CUSTOM_1X4: ModelConfig(
            pretrained_model_name="mnist_mlp_custom_1x4",
        ),
        ModelVariant.MLP_CUSTOM_1X8: ModelConfig(
            pretrained_model_name="mnist_mlp_custom_1x8",
        ),
        ModelVariant.CNN_BATCHNORM: ModelConfig(
            pretrained_model_name="mnist_cnn_batchnorm_custom",
        ),
        ModelVariant.CNN_DROPOUT: ModelConfig(
            pretrained_model_name="mnist_cnn_dropout_custom",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.MLP_CUSTOM

    def __init__(
        self,
        variant: Optional[ModelVariant] = None,
        hidden_sizes: Sequence[int] = (256, 128, 64),
    ):
        """Initialize ModelLoader with specified variant and configuration.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
            hidden_sizes: Hidden layer sizes for MLP architecture.
        """
        super().__init__(variant)
        self._hidden_sizes = tuple(hidden_sizes)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """

        return ModelInfo(
            model="mnist",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.CUSTOM,
            framework=Framework.JAX,
        )

    def load_model(self, dtype_override=None):
        """Load and return the MNIST model instance for this instance's variant.

        Args:
            dtype_override: Optional dtype to override the default dtype.

        Returns:
            model: The loaded model instance
        """

        if self._variant in [
            ModelVariant.MLP_CUSTOM,
            ModelVariant.MLP_CUSTOM_1X2,
            ModelVariant.MLP_CUSTOM_1X4,
            ModelVariant.MLP_CUSTOM_1X8,
        ]:
            return MNISTMLPModel(self._hidden_sizes)
        elif self._variant == ModelVariant.CNN_BATCHNORM:
            return MNISTCNNBatchNormModel()
        elif self._variant == ModelVariant.CNN_DROPOUT:
            return MNISTCNNDropoutModel()
        else:
            raise ValueError(f"Unsupported variant: {self._variant}")

    def load_inputs(self, dtype_override=None, mesh=None):
        """Load and return sample inputs for the MNIST model.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.
            mesh: Optional JAX mesh object for multi-device configurations.

        Returns:
            inputs: Input tensors that can be fed to the model.
        """
        # Determine batch size based on mesh configuration
        if mesh is not None:
            # For multi-device, use a fixed batch size that's divisible by device count
            # This matches the original test which used batch_size=32
            num_devices = np.prod(list(mesh.shape.values())) if mesh.shape else 1
            batch_size = 32  # Fixed batch size, will be sharded across devices
            # Ensure batch size is divisible by number of devices
            if batch_size % num_devices != 0:
                batch_size = num_devices * (batch_size // num_devices + 1)
        else:
            # Default to 32 for single device too, for consistency
            batch_size = 32

        # Create random input like the original test
        # Using a fixed seed for reproducibility
        prng_key = jax.random.PRNGKey(37)
        # B, H, W, C
        # Channels is 1 as MNIST is in grayscale.
        inputs = jax.random.normal(prng_key, (batch_size, 28, 28, 1))

        # Apply dtype override if specified
        if dtype_override is not None:
            inputs = inputs.astype(dtype_override)

        return inputs

    def load_parameters(
        self,
        dtype_override=None,
        train=False,
        seed=None,
        inputs=None,
        # Multi-chip specific parameters (only needed for multi-chip variants)
        model_for_multichip=None,
        cpu_mesh=None,
        input_activations_partition_specs=None,
        input_parameters_partition_specs=None,
    ):
        """Load and return model parameters.

        Args:
            dtype_override: Optional dtype to override the default dtype.
            train: Whether to initialize for training mode (affects dropout).
            seed: Random seed for parameter initialization. If None, uses DEFAULT_PARAMS_INIT_SEED.
            inputs: Optional input tensors. If None, will load default inputs.
            model_for_multichip: Optional model instance for multi-chip initialization.
            cpu_mesh: Optional CPU mesh for multi-chip initialization.
            input_activations_partition_specs: Optional partition specs for input activations.
            input_parameters_partition_specs: Optional partition specs for parameters.

        Returns:
            PyTree: Model parameters initialized with random weights
        """
        # Use default seed if not provided
        if seed is None:
            seed = self.DEFAULT_PARAMS_INIT_SEED

        # Check if this is a multi-chip variant based on the variant name
        is_multichip_variant = self._variant in [
            ModelVariant.MLP_CUSTOM_1X2,
            ModelVariant.MLP_CUSTOM_1X4,
            ModelVariant.MLP_CUSTOM_1X8,
        ]

        if is_multichip_variant:
            # Multi-chip variants require special initialization
            if (
                model_for_multichip is None
                or cpu_mesh is None
                or input_activations_partition_specs is None
                or input_parameters_partition_specs is None
            ):
                raise ValueError(
                    f"Multi-chip variant {self._variant} requires model_for_multichip, cpu_mesh, "
                    "input_activations_partition_specs, and input_parameters_partition_specs parameters"
                )

            from infra.utilities import initialize_flax_linen_parameters_on_cpu

            # Use provided inputs or load default ones
            if inputs is None:
                inputs = self.load_inputs(dtype_override, mesh=cpu_mesh)

            return initialize_flax_linen_parameters_on_cpu(
                model_for_multichip,
                input_activations_partition_specs,
                inputs,
                input_parameters_partition_specs,
                cpu_mesh,
                seed,
            )
        else:
            # Single-chip variant uses standard initialization
            model = self.load_model(dtype_override)

            # Use provided inputs or load default ones
            if inputs is None:
                inputs = self.load_inputs(dtype_override, mesh=None)

            # Handle different model signatures
            if self._variant in [
                ModelVariant.MLP_CUSTOM,
                ModelVariant.MLP_CUSTOM_1X2,
                ModelVariant.MLP_CUSTOM_1X4,
                ModelVariant.MLP_CUSTOM_1X8,
            ]:
                return model.init(jax.random.PRNGKey(seed), inputs)
            else:
                # CNN models require train argument
                return model.init(jax.random.PRNGKey(seed), inputs, train=train)

    def load_parameters_partition_spec(
        self,
        model_for_multichip=None,
        cpu_mesh=None,
        input_activations_partition_specs=None,
        inputs=None,
        dtype_override=None,
    ):
        """Load and return parameter partition specifications for multi-chip configurations.

        Args:
            model_for_multichip: Model instance for multi-chip configurations.
            cpu_mesh: JAX Mesh object for CPU devices.
            input_activations_partition_specs: Partition specs for input activations.
            inputs: Optional input tensors. If None, will load default inputs.
            dtype_override: Optional dtype to override the default dtype.

        Returns:
            PyTree: Partition specifications for model parameters
        """
        # This is only for multi-chip variants
        is_multichip_variant = self._variant in [
            ModelVariant.MLP_CUSTOM_1X2,
            ModelVariant.MLP_CUSTOM_1X4,
            ModelVariant.MLP_CUSTOM_1X8,
        ]

        if not is_multichip_variant:
            raise ValueError(
                f"load_parameters_partition_spec is only for multi-chip variants, got {self._variant}"
            )

        if (
            model_for_multichip is None
            or cpu_mesh is None
            or input_activations_partition_specs is None
        ):
            raise ValueError(
                "Multi-chip partition spec requires model_for_multichip, cpu_mesh, "
                "and input_activations_partition_specs parameters"
            )

        from infra.utilities import make_flax_linen_parameters_partition_specs_on_cpu

        # Use provided inputs or load default ones for shape evaluation
        if inputs is None:
            inputs = self.load_inputs(dtype_override, mesh=cpu_mesh)

        return make_flax_linen_parameters_partition_specs_on_cpu(
            model_for_multichip,
            cpu_mesh,
            input_activations_partition_specs,
            inputs,
        )

    def load_multichip_model(
        self, axis_name="X", num_devices=2, train_mode=False, dtype_override=None
    ):
        """Load and return the MNIST multichip model instance.

        Args:
            axis_name: Name of the sharding axis.
            num_devices: Number of devices to use.
            train_mode: Whether to run in training mode.
            dtype_override: Optional dtype to override the model's default dtype.

        Returns:
            model: The loaded multichip model instance
        """
        # Apply dtype override if specified
        param_dtype = dtype_override if dtype_override is not None else jnp.bfloat16

        # Only MLP variants support multichip
        if self._variant not in [
            ModelVariant.MLP_CUSTOM_1X2,
            ModelVariant.MLP_CUSTOM_1X4,
            ModelVariant.MLP_CUSTOM_1X8,
        ]:
            raise ValueError(f"Variant {self._variant} does not support multichip mode")

        return MNISTMLPMultichipModel(
            hidden_sizes=self._hidden_sizes,
            axis_name=axis_name,
            num_devices=num_devices,
            train_mode=train_mode,
            param_dtype=param_dtype,
        )

    def get_input_activations_partition_spec(self, axis_name="X"):
        """Get partition specification for input activations.

        Args:
            axis_name: Name of the sharding axis.

        Returns:
            PartitionSpec for input activations (replicated for MNIST MLP)
        """
        from jax.sharding import PartitionSpec

        # No data parallelism utilized in MNIST MLP model - inputs are replicated
        return PartitionSpec()

    def get_forward_method_name(self):
        """Get the name of the forward method for the model.

        Returns:
            str: Name of the forward method (typically 'apply' for Flax models)
        """
        return "apply"

    def get_static_argnames(self):
        """Get static argument names for the forward method.

        For Flax models using apply, `train` needs to be static when the model
        uses it, because the model uses 'not train' which requires a concrete boolean value.
        Additionally, `mutable` must be static when using batch normalization
        because it contains string values that cannot be traced. `rngs` must be static for
        dropout models because it contains a random key that cannot be traced.

        Returns:
            list: List containing static argument names
        """
        static_args = []

        # Check if the model actually has a `train` parameter
        # Only CNN models have it, not the MLP models.
        if self._variant in [ModelVariant.CNN_BATCHNORM, ModelVariant.CNN_DROPOUT]:
            # `train` must be static because the model does boolean operations on it
            # (e.g., use_running_average=not train) which don't work with traced values.
            static_args.append("train")

        # `mutable` must be static for batch norm models because it contains strings
        # which cannot be traced by JAX.
        if self._variant == ModelVariant.CNN_BATCHNORM:
            static_args.append("mutable")

        # `rngs` must be static for dropout models because it contains a random key
        # which cannot be traced by JAX.
        if self._variant == ModelVariant.CNN_DROPOUT:
            static_args.append("rngs")

        return static_args

    def get_input_parameters(self, train=False, seed=None):
        """Get input parameters for the model.

        This method provides compatibility with DynamicJaxModelTester which expects
        a get_input_parameters method.

        Args:
            train: Whether to initialize for training mode (affects dropout).
            seed: Random seed for parameter initialization. If None, uses DEFAULT_PARAMS_INIT_SEED.

        Returns:
            PyTree: Model parameters initialized with random weights
        """
        return self.load_parameters(train=train, seed=seed)

    def get_forward_method_kwargs(self, train=False):
        """Get keyword arguments for the model's forward method.

        This method provides compatibility with DynamicJaxModelTester by returning
        the appropriate kwargs based on what the model's __call__ method accepts.

        Args:
            train: Whether the model is in training mode

        Returns:
            dict: Keyword arguments for the model's forward method
        """
        model = self.load_model()

        # Get the signature of the model's __call__ method.
        try:
            sig = inspect.signature(model.__call__)
            params = sig.parameters
        # Model might not have __call__ defined.
        except (AttributeError, ValueError):
            return {}

        kwargs = {}

        if "train" in params:
            kwargs["train"] = train

        # For Flax models using apply method, add RNG keys for models with dropout
        # We detect dropout by checking the model variant or by inspecting the model
        if self._variant == ModelVariant.CNN_DROPOUT and train:
            # Explicit dropout variant
            kwargs["rngs"] = {"dropout": jax.random.key(1)}

        # For models with batch normalization, make batch_stats mutable during training
        if self._variant == ModelVariant.CNN_BATCHNORM and train:
            kwargs["mutable"] = ("batch_stats",)

        return kwargs

    def wrapper_model(self, f):
        """Wrapper for model forward method to handle batch normalization outputs.

        When batch normalization is used with mutable=("batch_stats",) during training,
        the model returns a tuple of (output, updated_variables). This wrapper extracts
        just the output for compatibility with the testing framework.

        Args:
            f: The model forward function to wrap

        Returns:
            Wrapped function that handles batch norm outputs correctly
        """

        def model(args, kwargs):
            out = f(*args, **kwargs)
            # For CNN_BATCHNORM variant, when mutable is used (training mode),
            # the output is always a tuple (output, {"batch_stats": ...})
            # We need to extract just the first element
            if self._variant == ModelVariant.CNN_BATCHNORM:
                if isinstance(out, tuple) and len(out) == 2:
                    # Extract the actual model output from the tuple
                    out = out[0]
            return out

        return model
