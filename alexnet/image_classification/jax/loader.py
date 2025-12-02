# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
AlexNet model loader implementation for image classification.
"""

from typing import Optional
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
from .src import AlexNetModel, AlexNetMultichipModel


class ModelVariant(StrEnum):
    """Available AlexNet model variants."""

    CUSTOM = "custom"
    CUSTOM_1X2 = "custom_1x2"
    CUSTOM_1X4 = "custom_1x4"
    CUSTOM_1X8 = "custom_1x8"


class ModelLoader(ForgeModel):
    """AlexNet model loader implementation for image classification."""

    # Default random seed for parameter initialization
    DEFAULT_PARAMS_INIT_SEED = 42

    # Dictionary of available model variants using structured configs
    _VARIANTS = {
        ModelVariant.CUSTOM: ModelConfig(
            pretrained_model_name="custom",
        ),
        ModelVariant.CUSTOM_1X2: ModelConfig(
            pretrained_model_name="custom_1x2",
        ),
        ModelVariant.CUSTOM_1X4: ModelConfig(
            pretrained_model_name="custom_1x4",
        ),
        ModelVariant.CUSTOM_1X8: ModelConfig(
            pretrained_model_name="custom_1x8",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.CUSTOM

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Implementation method for getting model info with validated variant.

        Args:
            variant: Optional ModelVariant specifying which variant to get info for.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="alexnet",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.CV_IMAGE_CLS,
            source=ModelSource.CUSTOM,
            framework=Framework.JAX,
        )

    def load_model(self, dtype_override=None):
        """Load and return the AlexNet model instance for this instance's variant.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.

        Returns:
            model: The loaded model instance
        """
        # Apply dtype override if specified
        if dtype_override is not None:
            model = AlexNetModel(param_dtype=dtype_override)
        else:
            model = AlexNetModel()

        return model

    def load_inputs(self, dtype_override=None, mesh=None):
        """Load and return sample inputs for the AlexNet model with this instance's variant settings.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.
            mesh: Optional JAX mesh object for multi-device configurations.

        Returns:
            inputs: Input tensors that can be fed to the model.
        """
        # Determine batch size based on mesh configuration
        if mesh is not None:
            # For multi-device, use a fixed batch size that's divisible by device count
            # This matches the original test which used batch_size=8
            num_devices = np.prod(list(mesh.shape.values())) if mesh.shape else 1
            batch_size = 8  # Fixed batch size, will be sharded across devices
            # Ensure batch size is divisible by number of devices
            if batch_size % num_devices != 0:
                batch_size = num_devices * (batch_size // num_devices + 1)
        else:
            # Default to 8 for single device too, for consistency
            batch_size = 8

        # Create random input like the original test
        # Using a fixed seed for reproducibility
        prng_key = jax.random.PRNGKey(23)
        inputs = jax.random.randint(
            key=prng_key,
            # B, H, W, C
            shape=(batch_size, 224, 224, 3),
            # In the original paper inputs are normalized with individual channel
            # values learned from training set.
            minval=-128,
            maxval=128,
        )

        # Apply dtype override if specified
        if dtype_override is not None:
            inputs = inputs.astype(dtype_override)

        return inputs

    def load_parameters(
        self,
        dtype_override=None,
        train=False,
        seed=None,
        inputs=None,  # Add inputs parameter
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
            ModelVariant.CUSTOM_1X2,
            ModelVariant.CUSTOM_1X4,
            ModelVariant.CUSTOM_1X8,
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
            ModelVariant.CUSTOM_1X2,
            ModelVariant.CUSTOM_1X4,
            ModelVariant.CUSTOM_1X8,
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
        """Load and return the AlexNet multichip model instance.

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

        return AlexNetMultichipModel(
            axis_name=axis_name,
            num_devices=num_devices,
            train_mode=train_mode,
            param_dtype=param_dtype,
        )

    def get_input_activations_partition_spec(self, mesh, axis_name="X"):
        """Get partition specification for input activations.

        Args:
            axis_name: Name of the sharding axis.

        Returns:
            PartitionSpec for input activations (sharded on batch dimension)
        """
        from jax.sharding import PartitionSpec

        return PartitionSpec(axis_name)

    def get_input_parameters(self):
        """Get input parameters for the model.

        This method provides compatibility with DynamicJaxModelTester which expects
        a get_input_parameters method. Simply delegates to load_parameters.

        Returns:
            PyTree: Model parameters initialized with random weights
        """
        return self.load_parameters()

    def get_input_parameters(self, train=False):
        """Get input parameters for training or inference mode.

        This method is required by the DynamicJaxModelTester for training mode.
        It returns the model parameters that will be used in the backward pass.

        Args:
            train: Whether to initialize parameters for training mode (affects dropout, etc.)

        Returns:
            PyTree: Model parameters initialized with random weights
        """
        # Use the existing load_parameters method to get initialized parameters
        return self.load_parameters(train=train)

    def get_forward_method_kwargs(self, train=False):
        """Get keyword arguments for the model's forward method.

        Args:
            train: Whether the model is in training mode

        Returns:
            dict: Keyword arguments for the model's forward method
        """
        # AlexNet always requires train argument
        kwargs = {"train": train}

        # Add dropout RNG key when training (AlexNet has dropout layers)
        if train:
            kwargs["rngs"] = {"dropout": jax.random.key(1)}

        return kwargs

    def get_static_argnames(self):
        """Get static argument names for the model's forward method.

        For custom Flax models using apply, 'train' needs to be static because
        the model uses control flow based on the train value.

        Returns:
            list: List containing 'train' as a static argument
        """
        return ["train"]
