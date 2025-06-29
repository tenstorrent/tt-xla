# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict

import jax
import pytest
from infra import ComparisonConfig, Framework, RunMode
from transformers import AutoTokenizer, FlaxBigBirdForCausalLM, FlaxPreTrainedModel
from utils import (
    BringupStatus,
    Category,
    ModelGroup,
    ModelSource,
    ModelTask,
    build_model_name,
    failed_ttmlir_compilation,
)

from ..tester import BigBirdTester

MODEL_PATH = "google/bigbird-roberta-large"
MODEL_NAME = build_model_name(
    Framework.JAX,
    "bigbird",
    "roberta_large",
    ModelTask.NLP_CAUSAL_LM,
    ModelSource.HUGGING_FACE,
)


class BigBirdLargeTester(BigBirdTester):
    def __init__(
        self,
        model_path: str,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
    ) -> None:
        super().__init__(model_path, comparison_config, run_mode)

    # @override
    def _get_model(self) -> FlaxPreTrainedModel:
        return FlaxBigBirdForCausalLM.from_pretrained(self._model_path)

    # @override
    def _get_input_activations(self) -> Dict[str, jax.Array]:
        tokenizer = AutoTokenizer.from_pretrained(self._model_path)
        inputs = tokenizer(
            "**JAX** is an open-source numerical computing library developed by Google, which is primarily used for high-performance machine learning research and other scientific computing tasks. It is built on top of **NumPy**, providing a familiar interface for those already comfortable with it, but with added features for automatic differentiation, GPU/TPU acceleration, and more.1. **NumPy-Compatible**: JAX is designed to be a drop-in replacement for **NumPy**, meaning that the code written in NumPy can often be used directly in JAX with minimal changes.JAX functions (such as `jax.numpy`) are compatible with NumPy syntax, so if you're familiar with NumPy, the learning curve for JAX is shallow.2. **Automatic Differentiation (Autograd)**: One of the core features of JAX is its ability to compute **gradients** (derivatives) of functions.The `jax.grad` function allows for automatic differentiation of scalar or vector-valued functions. This is useful in optimization, machine learning, and physics simulations."
            "JAX can compute gradients for functions written in a **Pythonic** way, without requiring manual derivative computation, making it much more efficient for machine learning and other scientific applications.JAX uses **XLA**, which stands for **Accelerated Linear Algebra**, to perform optimized computation on CPUs, GPUs, and TPUs.With XLA, JAX can execute code much faster by optimizing the computation graph (just-in-time (JIT) compilation) and running it efficiently on hardware accelerators.**Just-In-Time (JIT) compilation** in JAX refers to the process of converting Python code into highly optimized machine code that runs faster. This is achieved through the `jax.jit` decorator, which compiles a function once and reuses the compiled version, speeding up the execution.The result is a substantial boost in performance, especially for large-scale machine learning models or iterative algorithms.JAX allows for the creation of custom gradients for user-defined operations, giving researchers and engineers flexibility to define their own backpropagation rules.It also offers function transformations such as `jax.jit`, `jax.grad`, `jax.vmap`, and `jax.pmap` that help with optimization, parallelization, and differentiation."
            "**JAX** is an open-source numerical computing library developed by Google, which is primarily used for high-performance machine learning research and other scientific computing tasks. It is built on top of **NumPy**, providing a familiar interface for those already comfortable with it, but with added features for automatic differentiation, GPU/TPU acceleration, and more.1. **NumPy-Compatible**: JAX is designed to be a drop-in replacement for **NumPy**, meaning that the code written in NumPy can often be used directly in JAX with minimal changes.JAX functions (such as `jax.numpy`) are compatible with NumPy syntax, so if you're familiar with NumPy, the learning curve for JAX is shallow.2. **Automatic Differentiation (Autograd)**: One of the core features of JAX is its ability to compute **gradients** (derivatives) of functions.The `jax.grad` function allows for automatic differentiation of scalar or vector-valued functions. This is useful in optimization, machine learning, and physics simulations."
            "JAX can compute gradients for functions written in a **Pythonic** way, without requiring manual derivative computation, making it much more efficient for machine learning and other scientific applications.JAX uses **XLA**, which stands for **Accelerated Linear Algebra**, to perform optimized computation on CPUs, GPUs, and TPUs.With XLA, JAX can execute code much faster by optimizing the computation graph (just-in-time (JIT) compilation) and running it efficiently on hardware accelerators.**Just-In-Time (JIT) compilation** in JAX refers to the process of converting Python code into highly optimized machine code that runs faster. This is achieved through the `jax.jit` decorator, which compiles a function once and reuses the compiled version, speeding up the execution.The result is a substantial boost in performance, especially for large-scale machine learning models or iterative algorithms.JAX allows for the creation of custom gradients for user-defined operations, giving researchers and engineers flexibility to define their own backpropagation rules.It also offers function transformations such as `jax.jit`, `jax.grad`, `jax.vmap`, and `jax.pmap` that help with optimization, parallelization, and differentiation."
            "**JAX** is an open-source numerical computing library developed by Google, which is primarily used for high-performance machine learning research and other scientific computing tasks. It is built on top of **NumPy**, providing a familiar interface for those already comfortable with it, but with added features for automatic differentiation, GPU/TPU acceleration, and more.1. **NumPy-Compatible**: JAX is designed to be a drop-in replacement for **NumPy**, meaning that the code written in NumPy can often be used directly in JAX with minimal changes.JAX functions (such as `jax.numpy`) are compatible with NumPy syntax, so if you're familiar with NumPy, the learning curve for JAX is shallow.2. **Automatic Differentiation (Autograd)**: One of the core features of JAX is its ability to compute **gradients** (derivatives) of functions.The `jax.grad` function allows for automatic differentiation of scalar or vector-valued functions. This is useful in optimization, machine learning, and physics simulations."
            "JAX can compute gradients for functions written in a **Pythonic** way, without requiring manual derivative computation, making it much more efficient for machine learning and other scientific applications.JAX uses **XLA**, which stands for **Accelerated Linear Algebra**, to perform optimized computation on CPUs, GPUs, and TPUs.With XLA, JAX can execute code much faster by optimizing the computation graph (just-in-time (JIT) compilation) and running it efficiently on hardware accelerators.**Just-In-Time (JIT) compilation** in JAX refers to the process of converting Python code into highly optimized machine code that runs faster. This is achieved through the `jax.jit` decorator, which compiles a function once and reuses the compiled version, speeding up the execution.The result is a substantial boost in performance, especially for large-scale machine learning models or iterative algorithms.JAX allows for the creation of custom gradients for user-defined operations, giving researchers and engineers flexibility to define their own backpropagation rules.It also offers function transformations such as `jax.jit`, `jax.grad`, `jax.vmap`, and `jax.pmap` that help with optimization, parallelization, and differentiation.",
            return_tensors="jax",
            max_length=1472,
            truncation=True,
        )
        return inputs


# ----- Fixtures -----


@pytest.fixture
def inference_tester() -> BigBirdLargeTester:
    return BigBirdLargeTester(MODEL_PATH)


def training_tester() -> BigBirdLargeTester:
    return BigBirdLargeTester(MODEL_PATH, run_mode=RunMode.TRAINING)


# ----- Tests -----


@pytest.mark.model_test
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.FAILED_TTMLIR_COMPILATION,
)
@pytest.mark.xfail(
    reason=failed_ttmlir_compilation(
        "failed to legalize operation 'stablehlo.dynamic_slice' "
        "https://github.com/tenstorrent/tt-xla/issues/404"
    )
)
def test_bigbird_roberta_large_inference(inference_tester: BigBirdLargeTester):
    inference_tester.test()


@pytest.mark.nightly
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name=MODEL_NAME,
    model_group=ModelGroup.GENERALITY,
    run_mode=RunMode.TRAINING,
)
@pytest.mark.skip(reason="Support for training not implemented")
def test_bigbird_roberta_large_training(inference_tester: BigBirdLargeTester):
    training_tester.test()
