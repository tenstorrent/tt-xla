# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""BigBird model loader implementation for causal language modeling."""

from typing import Optional
import jax

from ....base import ForgeModel
from ....config import (
    LLMModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ....tools.jax_utils import cast_hf_model_to_type


class ModelVariant(StrEnum):
    """Available BigBird model variants for causal language modeling."""

    LARGE = "large"


class ModelLoader(ForgeModel):
    """BigBird model loader implementation for causal language modeling."""

    _VARIANTS = {
        ModelVariant.LARGE: LLMModelConfig(
            pretrained_model_name="google/bigbird-roberta-large",
            max_length=1472,
        ),
    }

    DEFAULT_VARIANT = ModelVariant.LARGE

    sample_text = """**JAX** is an open-source numerical computing library developed by Google, which is primarily used for high-performance machine learning research and other scientific computing tasks. It is built on top of **NumPy**, providing a familiar interface for those already comfortable with it, but with added features for automatic differentiation, GPU/TPU acceleration, and more.1. **NumPy-Compatible**: JAX is designed to be a drop-in replacement for **NumPy**, meaning that the code written in NumPy can often be used directly in JAX with minimal changes.JAX functions (such as `jax.numpy`) are compatible with NumPy syntax, so if you're familiar with NumPy, the learning curve for JAX is shallow.2. **Automatic Differentiation (Autograd)**: One of the core features of JAX is its ability to compute **gradients** (derivatives) of functions.The `jax.grad` function allows for automatic differentiation of scalar or vector-valued functions. This is useful in optimization, machine learning, and physics simulations.
        JAX can compute gradients for functions written in a **Pythonic** way, without requiring manual derivative computation, making it much more efficient for machine learning and other scientific applications.JAX uses **XLA**, which stands for **Accelerated Linear Algebra**, to perform optimized computation on CPUs, GPUs, and TPUs.With XLA, JAX can execute code much faster by optimizing the computation graph (just-in-time (JIT) compilation) and running it efficiently on hardware accelerators.**Just-In-Time (JIT) compilation** in JAX refers to the process of converting Python code into highly optimized machine code that runs faster. This is achieved through the `jax.jit` decorator, which compiles a function once and reuses the compiled version, speeding up the execution.The result is a substantial boost in performance, especially for large-scale machine learning models or iterative algorithms.JAX allows for the creation of custom gradients for user-defined operations, giving researchers and engineers flexibility
        to define their own backpropagation rules.It also offers function transformations such as `jax.jit`, `jax.grad`, `jax.vmap`, and `jax.pmap` that help with optimization, parallelization, and differentiation. **JAX** is an open-source numerical computing library developed by Google, which is primarily used for high-performance machine learning research and other scientific computing tasks. It is built on top of **NumPy**, providing a familiar interface for those already comfortable with it, but with added features for automatic differentiation, GPU/TPU acceleration, and more.1. **NumPy-Compatible**: JAX is designed to be a drop-in replacement for **NumPy**, meaning that the code written in NumPy can often be used directly in JAX with minimal changes.JAX functions (such as `jax.numpy`) are compatible with NumPy syntax, so if you're familiar with NumPy, the learning curve for JAX is shallow.2. **Automatic Differentiation (Autograd)**: One of the core features of JAX is its ability to compute **gradients** (derivatives)
        of functions.The `jax.grad` function allows for automatic differentiation of scalar or vector-valued functions. This is useful in optimization, machine learning, and physics simulations. JAX can compute gradients for functions written in a **Pythonic** way, without requiring manual derivative computation, making it much more efficient for machine learning and other scientific applications.JAX uses **XLA**, which stands for **Accelerated Linear Algebra**, to perform optimized computation on CPUs, GPUs, and TPUs.With XLA, JAX can execute code much faster by optimizing the computation graph (just-in-time (JIT) compilation) and running it efficiently on hardware accelerators.**Just-In-Time (JIT) compilation** in JAX refers to the process of converting Python code into highly optimized machine code that runs faster. This is achieved through the `jax.jit` decorator, which compiles a function once and reuses the compiled version, speeding up the execution.The result is a substantial boost in performance, especially for
        large-scale machine learning models or iterative algorithms.JAX allows for the creation of custom gradients for user-defined operations, giving researchers and engineers flexibility to define their own backpropagation rules.It also offers function transformations such as `jax.jit`, `jax.grad`, `jax.vmap`, and `jax.pmap` that help with optimization, parallelization, and differentiation. **JAX** is an open-source numerical computing library developed by Google, which is primarily used for high-performance machine learning research and other scientific computing tasks. It is built on top of **NumPy**, providing a familiar interface for those already comfortable with it, but with added features for automatic differentiation, GPU/TPU acceleration, and more.1. **NumPy-Compatible**: JAX is designed to be a drop-in replacement for **NumPy**, meaning that the code written in NumPy can often be used directly in JAX with minimal changes.JAX functions (such as `jax.numpy`) are compatible with NumPy syntax, so if you're familiar
        with NumPy, the learning curve for JAX is shallow.2. **Automatic Differentiation (Autograd)**: One of the core features of JAX is its ability to compute **gradients** (derivatives) of functions.The `jax.grad` function allows for automatic differentiation of scalar or vector-valued functions. This is useful in optimization, machine learning, and physics simulations. JAX can compute gradients for functions written in a **Pythonic** way, without requiring manual derivative computation, making it much more efficient for machine learning and other scientific applications.JAX uses **XLA**, which stands for **Accelerated Linear Algebra**, to perform optimized computation on CPUs, GPUs, and TPUs.With XLA, JAX can execute code much faster by optimizing the computation graph (just-in-time (JIT) compilation) and running it efficiently on hardware accelerators.**Just-In-Time (JIT) compilation** in JAX refers to the process of converting Python code into highly optimized machine code that runs faster. This is achieved through the
        `jax.jit` decorator, which compiles a function once and reuses the compiled version, speeding up the execution.The result is a substantial boost in performance, especially for large-scale machine learning models or iterative algorithms.JAX allows for the creation of custom gradients for user-defined operations, giving researchers and engineers flexibility to define their own backpropagation rules.It also offers function transformations such as `jax.jit`, `jax.grad`, `jax.vmap`, and `jax.pmap` that help with optimization, parallelization, and differentiation."""

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)
        self._tokenizer = None
        self._model_name = self._variant_config.pretrained_model_name

    @classmethod
    def _get_model_info(cls, variant_name: str = None):
        """Get model information.

        Args:
            variant_name: Optional variant name string.  If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """

        return ModelInfo(
            model="bigbird",
            variant=variant_name,
            group=ModelGroup.GENERALITY,
            task=ModelTask.NLP_CAUSAL_LM,
            source=ModelSource.HUGGING_FACE,
            framework=Framework.JAX,
        )

    def _load_tokenizer(self, dtype_override=None):
        """Load tokenizer for the current variant.

        Args:
            dtype_override: Optional dtype to override the tokenizer's default dtype.
                            If not provided, the tokenizer will use its default dtype (typically float32).

        Returns:
            The loaded tokenizer instance.
        """
        from transformers import AutoTokenizer

        tokenizer_kwargs = {}
        if dtype_override is not None:
            tokenizer_kwargs["dtype"] = dtype_override

        self._tokenizer = AutoTokenizer.from_pretrained(
            self._model_name, **tokenizer_kwargs
        )
        return self._tokenizer

    def load_model(self, dtype_override=None):
        """Load and return BigBird model for causal language modeling from Hugging Face.

        Args:
            dtype_override: Optional dtype to override the model's default dtype.
                            If not provided, the model will use its default dtype (typically float32).

        Returns:
            The loaded model instance.
        """
        from transformers import FlaxBigBirdForCausalLM

        model_kwargs = {}
        if dtype_override is not None:
            model_kwargs["dtype"] = dtype_override

        model = FlaxBigBirdForCausalLM.from_pretrained(self._model_name, **model_kwargs)

        # Cast the model to the dtype_override if provided
        if dtype_override is not None:
            model = cast_hf_model_to_type(model, dtype_override)

        return model

    def load_inputs(self, dtype_override=None):
        """Load inputs for the BigBird model.

        Args:
            dtype_override: Optional dtype to override the inputs' default dtype.
                            If not provided, the inputs will use its default dtype (typically float32).

        Returns:
            The loaded inputs instance.
        """

        if self._tokenizer is None:
            self._load_tokenizer(dtype_override=dtype_override)

        inputs = self._tokenizer(
            self.sample_text,
            return_tensors="jax",
            max_length=self._variant_config.max_length,
            truncation=True,
        )

        return inputs

    def get_forward_method_kwargs(self, train=False):
        """Get keyword arguments for the model's forward method.

        BigBird models require special RNG keys for training mode:
        - dropout_rng: for dropout layers
        - indices_rng: for sparse attention pattern generation

        Args:
            train: Whether the model is in training mode

        Returns:
            dict: Keyword arguments for the model's forward method
        """
        kwargs = {}

        # BigBird needs special RNG keys for training mode
        if train:
            kwargs["dropout_rng"] = jax.random.key(1)
            kwargs["indices_rng"] = jax.random.key(2)

        return kwargs

    def get_static_argnames(self):
        """Get static argument names for the model's forward method.

        HuggingFace models don't have static args at the top-level __call__.

        Returns:
            list: Empty list for HuggingFace models
        """
        return []
