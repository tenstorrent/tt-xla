# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Shared constants for test configuration validation, loading, and discovery."""

# Allowed architecture identifiers for arch_overrides and --arch option
ALLOWED_ARCHES = {"n150", "p150", "n300", "n300-llmbox", "galaxy-wh-6u"}

# Allowed fields in test_config YAML entries
ALLOWED_FIELDS = {
    # Comparator controls
    "required_pcc",
    "assert_pcc",
    "assert_atol",
    "required_atol",
    "assert_allclose",
    "allclose_rtol",
    "allclose_atol",
    # Status/metadata
    "status",
    "reason",
    "bringup_status",
    "markers",
    "supported_archs",
    "batch_size",
    # Nested arch overrides
    "arch_overrides",
    # Needed for training tests
    "execution_pass",
    # FileCheck patterns list
    "filechecks",
}

# Single source of truth for the placeholders YAML filename
PLACEHOLDERS_FILENAME = "test_config_placeholders.yaml"

# Frameworks mapped to their config directory names
FRAMEWORKS = ("torch", "jax", "torch_llm")

# Parallelism values for test ID cross-product
PARALLELISMS_STANDARD = ("single_device", "data_parallel", "tensor_parallel")
PARALLELISMS_LLM = (
    "single_device",
    "tensor_parallel",
    "megatron_mesh_1x8-no_dp-tensor_parallel",
    "fsdp_mesh_2x4-no_dp-tensor_parallel",
    "fsdp_mesh_2x4-dp-tensor_parallel",
    "megatron_mesh_2x4-no_dp-tensor_parallel",
    "megatron_mesh_2x4-dp-tensor_parallel",
)

# Run modes
RUN_MODES_STANDARD = ("inference", "training")
RUN_MODES_LLM = ("inference",)

# LLM phases
LLM_PHASES = {"load_inputs_decode": "llm_decode", "load_inputs_prefill": "llm_prefill"}

# LLM parametrization values (mirrors test_models.py)
LLM_SEQUENCE_LENGTHS = (128, 1024, 2048, 4096, 8192)
LLM_BATCH_SIZES = (1, 2)

# Models excluded from PyTorch discovery (matches dynamic_loader.py)
TORCH_EXCLUDED_MODEL_DIRS = {"suryaocr"}
