# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Callable

import numpy as np
import pytest
import torch
import torch_xla
import torch_xla.runtime as xr
import tt_torch
from infra import Framework, run_graph_test
from infra.evaluators import ComparisonConfig, PccConfig
from torch_xla.distributed.spmd import Mesh
from transformers.models.falcon.modeling_falcon import FalconMLP
from transformers.models.gemma.modeling_gemma import GemmaMLP
from transformers.models.llama.modeling_llama import LlamaMLP
from transformers.models.mistral.modeling_mistral import MistralMLP
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP

from tests.utils import parametrize_arch
from third_party.tt_forge_models.falcon.pytorch.loader import (
    ModelLoader as FalconModelLoader,
)
from third_party.tt_forge_models.falcon.pytorch.loader import (
    ModelVariant as FalconModelVariant,
)
from third_party.tt_forge_models.gemma.pytorch.loader import (
    ModelLoader as GemmaModelLoader,
)
from third_party.tt_forge_models.gemma.pytorch.loader import (
    ModelVariant as GemmaModelVariant,
)
from third_party.tt_forge_models.gpt_oss.pytorch.loader import (
    ModelLoader as GPTOSSModelLoader,
)
from third_party.tt_forge_models.gpt_oss.pytorch.loader import (
    ModelVariant as GPTOSSModelVariant,
)
from third_party.tt_forge_models.llama.causal_lm.pytorch.loader import (
    ModelLoader as LlamaModelLoader,
)
from third_party.tt_forge_models.mistral.pytorch.loader import (
    ModelLoader as MistralModelLoader,
)
from third_party.tt_forge_models.mistral.pytorch.loader import (
    ModelVariant as MistralModelVariant,
)
from third_party.tt_forge_models.qwen_2_5.causal_lm.pytorch.loader import (
    ModelLoader as Qwen2_5ModelLoader,
)
from third_party.tt_forge_models.qwen_2_5.causal_lm.pytorch.loader import (
    ModelVariant as Qwen2_5ModelVariant,
)
from third_party.tt_forge_models.qwen_3.causal_lm.pytorch.loader import (
    ModelLoader as Qwen3ModelLoader,
)
from third_party.tt_forge_models.qwen_3.causal_lm.pytorch.loader import (
    ModelVariant as Qwen3ModelVariant,
)

MODEL_LOADER_MAP = {
    "llama": LlamaModelLoader,
    "qwen3": Qwen3ModelLoader,
    "qwen2_5": Qwen2_5ModelLoader,
    "gemma": GemmaModelLoader,
    "mistral": MistralModelLoader,
    "falcon": FalconModelLoader,
    "gpt_oss": GPTOSSModelLoader,
}

AVAILABLE_VARIANT_MAP = {
    "llama": [
        "llama_3_8b",
        "llama_3_1_8b",
        "llama_3_1_70b",
        "llama_3_2_1b",
        "llama_3_2_3b",
        "llama_3_3_70b_instruct",
        "huggyllama_7b",
        "TinyLlama_v1.1",
    ],
    "qwen3": ["0_6b", "1_7b", "4b", "8b", "14b", "32b"],
    "qwen2_5": [
        "0_5b",
        "1_5b",
        "3b",
        "7b",
        "14b",
        "32b_instruct",
        "72b_instruct",
        "math_7b",
    ],
    "gemma": [
        "google/gemma-1.1-2b-it",
        "google/gemma-1.1-7b-it",
        "google/gemma-2b",
        "google/gemma-2-2b-it",
        "google/gemma-2-9b-it",
        "google/gemma-2-27b-it",
    ],
    "mistral": [
        "7b",
        "7b_instruct_v03",
        "ministral_3b_instruct",
        "ministral_8b_instruct",
    ],
    "falcon": [
        "tiiuae/Falcon3-1B-Base",
        "tiiuae/Falcon3-3B-Base",
        "tiiuae/Falcon3-7B-Base",
        "tiiuae/Falcon3-10B-Base",
        "tiiuae/Falcon3-Mamba-7B-Base",
        "tiiuae/falcon-7b-instruct",
    ],
    "gpt_oss": ["gpt_oss_20b", "gpt_oss_120b"],
}


def get_available_variants(model_name):
    ModelLoader = MODEL_LOADER_MAP[model_name]
    available_variants = ModelLoader.query_available_variants()

    # Filter to only include variants that match names in AVAILABLE_VARIANT_MAP
    if model_name in AVAILABLE_VARIANT_MAP:
        allowed_variant_names = set(AVAILABLE_VARIANT_MAP[model_name])
        return {
            variant_key: variant_config
            for variant_key, variant_config in available_variants.items()
            if str(variant_key) in allowed_variant_names
        }

    return available_variants


"""Qwen3 MLP test"""


@pytest.mark.nightly
@parametrize_arch(["single_device", "llmbox"])
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("qwen3").items(),
    ids=[str(k) for k in get_available_variants("qwen3").keys()],
)
def test_qwen3_mlp(seq_len, variant, variant_config, arch):
    if not arch == "llmbox" and str(variant) == "32b":
        pytest.skip("Variant doesn't fit on a single device")

    xr.set_device_type("TT")

    loader = Qwen3ModelLoader(variant=variant)
    config = loader.load_config()
    mlp = Qwen3MLP(config).to(torch.bfloat16)

    batch_size = 1

    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )

    if arch == "llmbox":
        num_devices = xr.global_runtime_device_count()
        mesh_shape = (1, num_devices)
        device_ids = np.array(range(num_devices))
        mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

        def get_shard_spec(mlp, args, kwargs):
            shard_specs = {}
            # check if model is a MoE model (Qwen3-30B-A3B)
            if hasattr(mlp, "num_experts"):
                for expert in mlp.experts:
                    shard_specs[expert.gate_proj.weight] = ("model", None)
                    shard_specs[expert.up_proj.weight] = ("model", None)
                    shard_specs[expert.down_proj.weight] = (None, "model")
            else:
                shard_specs[mlp.gate_proj.weight] = ("model", None)
                shard_specs[mlp.up_proj.weight] = ("model", None)
                shard_specs[mlp.down_proj.weight] = (None, "model")

            return shard_specs

    else:
        mesh = None
        get_shard_spec = None

    run_graph_test(
        mlp,
        [hidden_states],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


"""Llama MLP test"""


@pytest.mark.nightly
@parametrize_arch(["single_device", "llmbox"])
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("llama").items(),
    ids=[str(k) for k in get_available_variants("llama").keys()],
)
def test_llama_mlp(seq_len, variant, variant_config, arch):
    if "70b" in str(variant) and not arch == "llmbox":
        pytest.skip("70B models don't fit on a single device")

    xr.set_device_type("TT")

    loader = LlamaModelLoader(variant=variant)
    config = loader.load_config()
    mlp = LlamaMLP(config).to(torch.bfloat16)

    batch_size = 1

    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )

    if arch == "llmbox":
        num_devices = xr.global_runtime_device_count()
        mesh_shape = (1, num_devices)
        device_ids = np.array(range(num_devices))
        mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

        def get_shard_spec(mlp, args, kwargs):
            shard_specs = {}
            shard_specs[mlp.gate_proj.weight] = ("model", None)
            shard_specs[mlp.up_proj.weight] = ("model", None)
            shard_specs[mlp.down_proj.weight] = (None, "model")
            return shard_specs

    else:
        mesh = None
        get_shard_spec = None

    run_graph_test(
        mlp,
        [hidden_states],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


"""Gemma MLP test"""


@pytest.mark.nightly
@parametrize_arch(["single_device", "llmbox"])
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("gemma").items(),
    ids=[str(k) for k in get_available_variants("gemma").keys()],
)
def test_gemma_mlp(seq_len, variant, variant_config, arch):
    if not arch == "llmbox" and (str(variant) == "google/gemma-2-27b-it"):
        pytest.skip("Variant doesn't fit on a single device")

    xr.set_device_type("TT")

    loader = GemmaModelLoader(variant=variant)
    config = loader.load_config()
    mlp = GemmaMLP(config).to(torch.bfloat16)

    batch_size = 1

    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )

    if arch == "llmbox":
        num_devices = xr.global_runtime_device_count()
        mesh_shape = (1, num_devices)
        device_ids = np.array(range(num_devices))
        mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

        def get_shard_spec(mlp, args, kwargs):
            shard_specs = {}
            shard_specs[mlp.gate_proj.weight] = ("model", None)
            shard_specs[mlp.up_proj.weight] = ("model", None)
            shard_specs[mlp.down_proj.weight] = (None, "model")
            return shard_specs

    else:
        mesh = None
        get_shard_spec = None

    run_graph_test(
        mlp,
        [hidden_states],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


"""Mistral MLP test"""


@pytest.mark.nightly
@parametrize_arch(["single_device", "llmbox"])
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("mistral").items(),
    ids=[str(k) for k in get_available_variants("mistral").keys()],
)
def test_mistral_mlp(seq_len, variant, variant_config, arch):

    xr.set_device_type("TT")

    loader = MistralModelLoader(variant=variant)
    config = loader.load_config()
    mlp = MistralMLP(config).to(torch.bfloat16)

    batch_size = 1

    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )

    if arch == "llmbox":
        num_devices = xr.global_runtime_device_count()
        mesh_shape = (1, num_devices)
        device_ids = np.array(range(num_devices))
        mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

        def get_shard_spec(mlp, args, kwargs):
            shard_specs = {}
            shard_specs[mlp.gate_proj.weight] = ("model", None)
            shard_specs[mlp.up_proj.weight] = ("model", None)
            shard_specs[mlp.down_proj.weight] = (None, "model")
            return shard_specs

    else:
        mesh = None
        get_shard_spec = None

    run_graph_test(
        mlp,
        [hidden_states],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


"""Qwen2_5 MLP test"""


@pytest.mark.nightly
@parametrize_arch(["single_device", "llmbox"])
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("qwen2_5").items(),
    ids=[str(k) for k in get_available_variants("qwen2_5").keys()],
)
def test_qwen2_5_mlp(seq_len, variant, variant_config, arch):
    if not arch == "llmbox" and (
        str(variant) == "72b_instruct" or str(variant) == "32b_instruct"
    ):
        pytest.skip("Variant doesn't fit on a single device")

    xr.set_device_type("TT")

    loader = Qwen2_5ModelLoader(variant=variant)
    config = loader.load_config()
    mlp = Qwen2MLP(config).to(torch.bfloat16)

    batch_size = 1

    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )

    if arch == "llmbox":
        num_devices = xr.global_runtime_device_count()
        mesh_shape = (1, num_devices)
        device_ids = np.array(range(num_devices))
        mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

        def get_shard_spec(mlp, args, kwargs):
            shard_specs = {}
            shard_specs[mlp.gate_proj.weight] = ("model", None)
            shard_specs[mlp.up_proj.weight] = ("model", None)
            shard_specs[mlp.down_proj.weight] = (None, "model")
            return shard_specs

    else:
        mesh = None
        get_shard_spec = None

    run_graph_test(
        mlp,
        [hidden_states],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


"""Falcon MLP test"""


@pytest.mark.nightly
@parametrize_arch(["single_device", "llmbox"])
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("falcon").items(),
    ids=[str(k) for k in get_available_variants("falcon").keys()],
)
def test_falcon_mlp(seq_len, variant, variant_config, arch):
    if variant != FalconModelVariant.FALCON_7B_INSTRUCT:
        if variant == FalconModelVariant.FALCON_MAMBA_7B:
            pytest.skip("FalconMamba has no MLP as it is a State Space Model.")
        else:
            pytest.skip(
                "Falcon3-Base models use Llama3 architecure, use Llama MLP test instead."
            )

    xr.set_device_type("TT")

    loader = FalconModelLoader(variant=variant)
    config = loader.load_config()
    mlp = FalconMLP(config).to(torch.bfloat16)

    batch_size = 1

    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )

    if arch == "llmbox":
        num_devices = xr.global_runtime_device_count()
        mesh_shape = (1, num_devices)
        device_ids = np.array(range(num_devices))
        mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))

        def get_shard_spec(mlp, args, kwargs):
            shard_specs = {}
            shard_specs[mlp.dense_h_to_4h.weight] = ("model", None)  # up_proj
            shard_specs[mlp.dense_4h_to_h.weight] = (None, "model")  # down_proj
            return shard_specs

    else:
        mesh = None
        get_shard_spec = None

    run_graph_test(
        mlp,
        [hidden_states],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


"""GPT-OSS MLP test"""


@pytest.mark.nightly
@parametrize_arch(["single_device", "llmbox"])
@pytest.mark.parametrize("mlp_type", ["original", "sparse", "a2a_sparse"])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("gpt_oss").items(),
    ids=[str(k) for k in get_available_variants("gpt_oss").keys()],
)
def test_gpt_oss_mlp(variant, variant_config, mlp_type, arch):
    xr.set_device_type("TT")

    loader = GPTOSSModelLoader(variant=variant, num_layers=1)
    model = loader.load_model()
    config = loader.load_config()
    inputs = loader.load_inputs()

    batch_size = inputs["input_ids"].shape[0]  # 1
    seq_len = inputs["input_ids"].shape[1]  # 128 with padding

    if mlp_type == "sparse":
        tt_torch.sparse_mlp.enable_sparse_mlp(model)
    elif mlp_type == "a2a_sparse":
        from tt_torch.sparse_mlp import A2aSparseMLP

        original_mlp = model.model.layers[0].mlp
        flat_device_order = [0, 1, 2, 3, 7, 6, 5, 4]  # T3K snake order for (2,4) → (1,8) flatten
        model.model.layers[0].mlp = A2aSparseMLP(
            original_mlp,
            num_experts=config.num_local_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            num_devices=8,
            cluster_axis=-1,
            config=config,
            flat_device_order=flat_device_order,
        )

    mlp = model.model.layers[0].mlp

    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )

    if arch == "llmbox":
        comparison_config = ComparisonConfig(pcc=PccConfig(required_pcc=0.85))
        num_devices = xr.global_runtime_device_count()

        if mlp_type == "a2a_sparse":
            # A2aSparseMLP: (1, 8) mesh, E sharded on "batch" axis (8-way EP)
            mesh_shape = (2, 4)
            device_ids = np.array(range(num_devices))
            mesh = Mesh(device_ids, mesh_shape, ("model", "batch"))

            def get_shard_spec(mlp, args, kwargs):
                shard_specs = {}
                shard_specs[mlp.router.weight] = (None, "batch")

                # weights [E, H, inter*2], E on "batch" (8-way EP, no TP)
                shard_specs[mlp.experts.gate_up_proj] = (("model", "batch"), None, None)
                shard_specs[mlp.experts.gate_up_proj_bias] = (("model", "batch"), None)
                shard_specs[mlp.experts.down_proj] = (("model", "batch"), None, None)
                shard_specs[mlp.experts.down_proj_bias] = (("model", "batch"), None)

                return shard_specs

        else:
            # Original/Sparse: (2, 4) mesh, EP on "model" + TP on "batch"
            mesh_shape = (2, 4)
            device_ids = np.array(range(num_devices))
            mesh = Mesh(device_ids, mesh_shape, ("model", "batch"))

            def get_shard_spec(mlp, args, kwargs):
                shard_specs = {}
                shard_specs[mlp.router.weight] = (None, "batch")

                shard_specs[mlp.experts.gate_up_proj] = ("model", "batch", None)
                shard_specs[mlp.experts.gate_up_proj_bias] = ("model", None)

                shard_specs[mlp.experts.down_proj] = ("model", None, "batch")
                shard_specs[mlp.experts.down_proj_bias] = ("model", "batch")

                return shard_specs

    else:
        comparison_config = ComparisonConfig(pcc=PccConfig(required_pcc=0.99))
        mesh = None
        get_shard_spec = None

    run_graph_test(
        mlp,
        [hidden_states],
        framework=Framework.TORCH,
        comparison_config=comparison_config,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


@pytest.mark.push
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("gpt_oss").items(),
    ids=[str(k) for k in get_available_variants("gpt_oss").keys()],
)
def test_sparse_mlp_cpu_parity(variant, variant_config):
    """Verify SparseMLP produces the same results as the original MLP on CPU."""
    from tt_torch.sparse_mlp import SparseMLP

    loader = GPTOSSModelLoader(variant=variant, num_layers=1)
    model = loader.load_model()
    config = loader.load_config()
    inputs = loader.load_inputs()

    batch_size = inputs["input_ids"].shape[0]
    seq_len = inputs["input_ids"].shape[1]

    original_mlp = model.model.layers[0].mlp
    sparse_mlp = SparseMLP(
        original_mlp,
        num_experts=config.num_local_experts,
        num_experts_per_tok=config.num_experts_per_tok,
        config=config,
    )

    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )

    with torch.no_grad():
        original_out, original_scores = original_mlp(hidden_states)
        sparse_out, sparse_scores = sparse_mlp(hidden_states)

    def compute_pcc(x, y):
        x_flat, y_flat = x.flatten().float(), y.flatten().float()
        vx, vy = x_flat - x_flat.mean(), y_flat - y_flat.mean()
        denom = vx.norm() * vy.norm()
        if denom == 0:
            return 1.0 if torch.allclose(x_flat, y_flat) else 0.0
        return float((vx @ vy) / denom)

    pcc_out = compute_pcc(original_out, sparse_out)
    pcc_scores = compute_pcc(original_scores, sparse_scores)

    print(f"PCC output: {pcc_out:.6f}")
    print(f"PCC router_scores: {pcc_scores:.6f}")

    assert pcc_out > 0.99, f"Output PCC too low: {pcc_out:.6f}"
    assert pcc_scores > 0.99, f"Router scores PCC too low: {pcc_scores:.6f}"


@pytest.mark.nightly
@parametrize_arch(["llmbox"])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("gpt_oss").items(),
    ids=[str(k) for k in get_available_variants("gpt_oss").keys()],
)
def test_a2a_sparse_mlp(variant, variant_config, arch):
    """Test A2aSparseMLP with EP compound sharding on (2,4) mesh.

    Expert weights [E, H, inter*2] are compound-sharded on E across both
    "model" and "batch" axes (2*4=8-way EP). cluster_axis=-1 enables
    all_to_all_dispatch/combine to route tokens across all 8 devices.
    """
    from tt_torch.sparse_mlp import A2aSparseMLP

    xr.set_device_type("TT")

    loader = GPTOSSModelLoader(variant=variant, num_layers=1)
    model = loader.load_model()
    config = loader.load_config()
    inputs = loader.load_inputs()

    batch_size = inputs["input_ids"].shape[0]
    seq_len = inputs["input_ids"].shape[1]

    original_mlp = model.model.layers[0].mlp

    # E=32 experts, mesh (2, 4), dispatch across all 8 devices
    # cluster_axis=-1 means route across all devices (both mesh dimensions)
    num_devices = 8
    # T3K snake order: reshape(2,4 → 1,8) gives chip order 0,1,2,3,7,6,5,4
    flat_device_order = [0, 1, 2, 3, 7, 6, 5, 4]

    mlp = A2aSparseMLP(
        original_mlp,
        num_experts=config.num_local_experts,
        num_experts_per_tok=config.num_experts_per_tok,
        num_devices=num_devices,
        cluster_axis=-1,
        flat_device_order=flat_device_order,
        config=config,
    )

    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )

    num_devices_total = xr.global_runtime_device_count()
    mesh_shape = (2, 4)
    device_ids = np.array(range(num_devices_total))
    mesh = Mesh(device_ids, mesh_shape, ("model", "batch"))

    def get_shard_spec(mlp, args, kwargs):
        shard_specs = {}
        shard_specs[mlp.router.weight] = (None, "batch")

        # EP sharding: weights [E, H, inter*2], E sharded on "batch" (8-way)
        # gate_up_proj [E, H, inter*2] - EP sharded on E along "batch"
        shard_specs[mlp.experts.gate_up_proj] = (("model", "batch"), None, None)
        # down_proj [E, inter, H] - EP sharded on E along "batch"
        shard_specs[mlp.experts.down_proj] = (("model", "batch"), None, None)
        shard_specs[mlp.experts.gate_up_proj_bias] = (("model", "batch"), None)
        shard_specs[mlp.experts.down_proj_bias] = (("model", "batch"), None)

        return shard_specs

    comparison_config = ComparisonConfig(pcc=PccConfig(required_pcc=0.99))

    run_graph_test(
        mlp,
        [hidden_states],
        framework=Framework.TORCH,
        comparison_config=comparison_config,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


@pytest.mark.nightly
@parametrize_arch(["llmbox"])
def test_all_to_all_dispatch_op(arch):
    """Unit test for full MoE forward (dispatch + gate_up/down sparse_matmul + activation + combine) on (2,4) mesh.

    Tests dispatch/combine with EP compound-sharded weights across all 8 devices.
    gate_up_proj [E, H, inter*2] and down_proj [E, inter, H] are sharded on E
    across ("model", "batch") axes (8-way EP).
    cluster_axis=-1 enables all-device routing via mesh flatten.
    """
    from tt_torch.sparse_mlp import build_expert_mapping

    # GPT-OSS 20B config
    batch_size = 1
    seq_len = 128
    hidden_size = 2880
    num_experts = 32
    num_experts_per_tok = 4

    xr.set_device_type("TT")

    num_devices = 8
    cluster_axis = -1  # all devices (mesh flatten for 2D mesh)

    intermediate_size = 7680  # GPT-OSS 20B intermediate size

    class DispatchSparseMatmulCombineModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            # Expert mapping with snake-order correction for (2,4) → (1,8) flatten.
            # On T3K, reshape(2,4 → 1,8) gives chip order: 0,1,2,3,7,6,5,4
            flat_device_order = [0, 1, 2, 3, 7, 6, 5, 4]
            mapping = build_expert_mapping(num_experts, num_devices)
            permuted_mapping = torch.zeros_like(mapping)
            for d in range(num_devices):
                permuted_mapping[:, :, :, flat_device_order[d]] = mapping[:, :, :, d]
            self.register_buffer("expert_mapping", permuted_mapping)

            self.alpha = 1.702
            self.limit = 7.0

            # gate_up_proj: [E, H, inter*2] (interleaved gate+up)
            self.gate_up_proj = torch.nn.Parameter(
                torch.randn(
                    num_experts,
                    hidden_size,
                    intermediate_size * 2,
                    dtype=torch.bfloat16,
                )
            )
            # down_proj: [E, inter, H]
            self.down_proj = torch.nn.Parameter(
                torch.randn(
                    num_experts,
                    intermediate_size,
                    hidden_size,
                    dtype=torch.bfloat16,
                )
            )
            # Biases
            self.gate_up_proj_bias = torch.nn.Parameter(
                torch.randn(num_experts, intermediate_size * 2, dtype=torch.bfloat16)
            )
            self.down_proj_bias = torch.nn.Parameter(
                torch.randn(num_experts, hidden_size, dtype=torch.bfloat16)
            )

        def forward(self, hidden_states, expert_indices):
            # hidden_states: [B, S, H], expert_indices: [B, S, K] (precomputed on CPU)
            B, S, H = hidden_states.shape
            K = expert_indices.shape[-1]

            # Reshape for dispatch: [B, 1, S, H] and [B, 1, S, K]
            x = hidden_states.view(B, 1, S, H)
            indices_4d = expert_indices.view(B, 1, S, K)

            # 1. Dispatch
            dispatched, metadata = torch.ops.tt.all_to_all_dispatch(
                x,
                indices_4d,
                self.expert_mapping,
                num_devices=num_devices,
                cluster_axis=cluster_axis,
            )
            BD = dispatched.shape[1]

            # 2. Sparsity mask from metadata
            metadata_indices = metadata[0]  # [B*D, S, K]
            sparsity = torch.zeros(
                BD,
                S,
                1,
                num_experts,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )
            topk_indices_unsqueezed = metadata_indices.unsqueeze(2)
            sparsity.scatter_(
                dim=-1,
                index=topk_indices_unsqueezed,
                src=torch.ones_like(topk_indices_unsqueezed, dtype=hidden_states.dtype),
            )

            # 3. Gate_up sparse_matmul
            hidden_4d = dispatched.view(BD, S, 1, H)
            gate_up_proj = self.gate_up_proj.unsqueeze(0)  # [1, E, H, inter*2]
            gate_up_out = torch.ops.tt.sparse_matmul(
                hidden_4d,
                gate_up_proj,
                sparsity,
                nnz=0,
                is_input_a_sparse=False,
                is_input_b_sparse=True,
            )
            # [BD, S, 1, E, 1, inter*2] → [BD, S, E, inter*2]
            gate_up_out = gate_up_out.view(BD, S, num_experts, intermediate_size * 2)
            gate_up_out = gate_up_out + self.gate_up_proj_bias

            # 4. Activation (interleaved split + clamped SiGLU)
            gate_out = gate_up_out[..., ::2]   # [BD, S, E, inter]
            up_out = gate_up_out[..., 1::2]    # [BD, S, E, inter]
            gate_out = gate_out.clamp(max=self.limit)
            up_out = up_out.clamp(-self.limit, self.limit)
            glu = gate_out * torch.sigmoid(gate_out * self.alpha)
            activated = (up_out + 1) * glu  # [BD, S, E, inter]

            # 5. Down sparse_matmul
            activated_reshaped = activated.view(
                BD * S, num_experts, 1, intermediate_size
            )
            sparsity_down = sparsity.view(1, 1, BD * S, num_experts)
            down_proj = self.down_proj.view(
                1, num_experts, intermediate_size, hidden_size
            )
            down_out = torch.ops.tt.sparse_matmul(
                activated_reshaped,
                down_proj,
                sparsity_down,
                nnz=0,
                is_input_a_sparse=True,
                is_input_b_sparse=False,
            )
            # [BD*S, E, 1, H] → [BD*S, E, H]
            down_out = down_out.squeeze(2)
            down_out = down_out + self.down_proj_bias

            # 6. Reshape for combine: [E, BD, S, H]
            down_out = down_out.view(BD, S, num_experts, H)
            out = down_out.permute(2, 0, 1, 3).contiguous()  # [E, BD, S, H]

            # 7. Combine
            combined = torch.ops.tt.all_to_all_combine(
                out,
                metadata,
                self.expert_mapping,
                num_devices=num_devices,
                cluster_axis=cluster_axis,
                num_experts_per_tok=K,
            )
            return combined.sum(dim=0)  # [B, S, H]

    module = DispatchSparseMatmulCombineModule()

    # Random input
    hidden_states = torch.randn(batch_size, seq_len, hidden_size, dtype=torch.bfloat16)

    # Precompute expert indices on CPU (avoids router TP numerical differences)
    torch.manual_seed(42)
    expert_indices = (
        torch.stack(
            [
                torch.randperm(num_experts)[:num_experts_per_tok]
                for _ in range(batch_size * seq_len)
            ]
        )
        .view(batch_size, seq_len, num_experts_per_tok)
        .to(torch.int64)
    )

    num_devices_total = xr.global_runtime_device_count()
    mesh_shape = (2, 4)
    device_ids = np.array(range(num_devices_total))
    mesh = Mesh(device_ids, mesh_shape, ("model", "batch"))

    def get_shard_spec(module, args, kwargs):
        shard_specs = {}
        shard_specs[args[0]] = (None, None, "batch")
        # EP compound-sharded on E across both mesh axes
        shard_specs[module.gate_up_proj] = (("model", "batch"), None, None)
        shard_specs[module.gate_up_proj_bias] = (("model", "batch"), None)
        shard_specs[module.down_proj] = (("model", "batch"), None, None)
        shard_specs[module.down_proj_bias] = (("model", "batch"), None)
        return shard_specs
    comparison_config = ComparisonConfig(pcc=PccConfig(required_pcc=1.0))
    run_graph_test(
        module,
        [hidden_states, expert_indices],
        comparison_config=comparison_config,
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


@pytest.mark.push
@pytest.mark.parametrize("num_devices", [8])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("gpt_oss").items(),
    ids=[str(k) for k in get_available_variants("gpt_oss").keys()],
)
def test_a2a_sparse_mlp_cpu_parity(variant, variant_config, num_devices):
    """Verify A2aSparseMLP produces the same results as the original MLP on CPU.

    Tests with num_devices=1 (no E0/E1 reshape) and num_devices>1 (E0/E1 reshape active).
    On CPU, dispatch/combine are no-ops regardless of num_devices, so outputs should match.
    """
    from tt_torch.sparse_mlp import A2aSparseStackedMlp

    loader = GPTOSSModelLoader(variant=variant, num_layers=1)
    model = loader.load_model()
    config = loader.load_config()
    inputs = loader.load_inputs()

    if config.num_local_experts % num_devices != 0:
        pytest.skip(
            f"num_experts ({config.num_local_experts}) not divisible by num_devices ({num_devices})"
        )

    batch_size = inputs["input_ids"].shape[0]
    seq_len = inputs["input_ids"].shape[1]

    original_mlp = model.model.layers[0].mlp

    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )

    # Run original BEFORE creating A2aSparseMLP (init reshapes shared experts)
    with torch.no_grad():
        original_out, original_scores = original_mlp(hidden_states)

    a2a_mlp = A2aSparseStackedMlp(
        original_mlp,
        num_experts=config.num_local_experts,
        num_experts_per_tok=config.num_experts_per_tok,
        num_devices=num_devices,
        cluster_axis=0,
        config=config,
    )

    with torch.no_grad():
        a2a_out, a2a_scores = a2a_mlp(hidden_states)

    def compute_pcc(x, y):
        x_flat, y_flat = x.flatten().float(), y.flatten().float()
        vx, vy = x_flat - x_flat.mean(), y_flat - y_flat.mean()
        denom = vx.norm() * vy.norm()
        if denom == 0:
            return 1.0 if torch.allclose(x_flat, y_flat) else 0.0
        return float((vx @ vy) / denom)

    pcc_out = compute_pcc(original_out, a2a_out)
    pcc_scores = compute_pcc(original_scores, a2a_scores)

    print(f"PCC output (original vs A2aSparseMLP, D={num_devices}): {pcc_out:.6f}")
    print(f"PCC router_scores: {pcc_scores:.6f}")

    assert pcc_out > 0.99, f"Output PCC too low: {pcc_out:.6f}"
    assert pcc_scores > 0.99, f"Router scores PCC too low: {pcc_scores:.6f}"
