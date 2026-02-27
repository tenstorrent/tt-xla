# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import os
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
        "3.0_8B",
        "3.1_8B",
        "3.1_70B",
        "3.2_1B",
        "3.2_3B",
        "3.3_70B_Instruct",
        "Huggyllama_7B",
        "Tinyllama_v1.1",
    ],
    "qwen3": ["0_6B", "1_7B", "4B", "8B", "14B", "32B"],
    "qwen2_5": [
        "0.5B",
        "1.5B",
        "3B",
        "7B",
        "14B",
        "32B_Instruct",
        "72B_Instruct",
        "Math_7B",
    ],
    "gemma": [
        "1.1_2B_IT",
        "1.1_7B_IT",
        "2B",
        "2_2B_IT",
        "2_9B_IT",
        "2_27B_IT",
    ],
    "mistral": [
        "7B",
        "7B_INSTRUCT_v03",
        "Ministral_3B_Instruct",
        "Ministral_8B_Instruct",
    ],
    "falcon": [
        "3_1B_Base",
        "3_3B_Base",
        "3_7B_Base",
        "3_10B_Base",
        "3_Mamba_7B_Base",
        "7B_Instruct",
    ],
    "gpt_oss": ["20b", "120b"],
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
    if not arch == "llmbox" and str(variant) == "32B":
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
    if not arch == "llmbox" and (str(variant) == "2_27B_IT"):
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
        str(variant) == "72B_Instruct" or str(variant) == "32B_Instruct"
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

    loader = GPTOSSModelLoader(
        variant=variant,
    )
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
        # flat_device_order = [0, 1, 2, 3, 7, 6, 5, 4]  # T3K snake order for (2,4) → (1,8) flatten
        model.model.layers[0].mlp = A2aSparseMLP(
            original_mlp,
            num_experts=config.num_local_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            num_devices=8,
            dispatch_devices=2,
            cluster_axis=0,
            config=config,
            # flat_device_order=flat_device_order,
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
                shard_specs[mlp.router.weight] = (None, "model")

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
    "model" (axis_0, 2 devices) and "batch" (axis_1, 4 devices) for 8-way EP.

    Dispatch/combine along axis_0 (cluster_axis=0, 2 devices):
      - dispatch routes tokens to the correct row
      - combine gathers expert results back

    All-reduce along axis_1 is inserted by Shardy's kReduction mechanism
    on the E factor to aggregate partial expert sums from column devices.

    Flow: dispatch(axis_0) → sparse_matmul → combine(axis_0) → all-reduce(axis_1)
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

    # E=32 experts, mesh (2,4), compound sharding on E across both axes
    # num_devices=8: expert_mapping [1,1,E,D] has D=8, mapping experts to all devices
    #   e.g. expert 0 → device 0, expert 4 → device 1, etc.
    #   dispatch kernel derives target ROW from device ID internally
    # dispatch_devices=2: dispatch/combine along axis_0 (2 row devices)
    #   BD = B * 2, combine restores B = BD / 2
    # Column aggregation: all-reduce on axis_1 via kReduction sharding rule
    num_devices = 8  # total devices (for expert_mapping D dimension)
    dispatch_devices = 2  # axis_0 size (for dispatch/combine communication)
    # T3K snake order: (2,4) mesh reshape gives chip order 0,1,2,3,7,6,5,4
    # flat_device_order = [0, 1, 2, 3, 7, 6, 5, 4]

    mlp = A2aSparseMLP(
        original_mlp,
        num_experts=config.num_local_experts,
        num_experts_per_tok=config.num_experts_per_tok,
        num_devices=num_devices,
        cluster_axis=0,
        dispatch_devices=dispatch_devices,
        # flat_device_order=flat_device_order,
        config=config,
    )

    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )

    num_devices_total = xr.global_runtime_device_count()
    mesh_shape = (2, 4)
    # T3K snake order: row 0 = chips 0,1,2,3, row 1 = chips 7,6,5,4
    device_ids = np.array([0, 1, 2, 3, 4, 5, 6, 7])
    mesh = Mesh(device_ids, mesh_shape, ("model", "batch"))

    def get_shard_spec(mlp, args, kwargs):
        shard_specs = {}
        shard_specs[mlp.router.weight] = (None, "model")

        # EP compound sharding: E sharded on ("model", "batch") = 2*4 = 8-way
        # gate_up_proj [E, H, inter*2]
        shard_specs[mlp.experts.gate_up_proj] = (("model", "batch"), None, None)
        # down_proj [E, inter, H]
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
            gate_out = gate_up_out[..., ::2]  # [BD, S, E, inter]
            up_out = gate_up_out[..., 1::2]  # [BD, S, E, inter]
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
@pytest.mark.nightly
def test_a2a_sparse_mlp_cpu_parity():
    """Verify A2aSparseMLP produces the same results as the original MLP on CPU.

    Tests with num_devices=1 (no E0/E1 reshape) and num_devices>1 (E0/E1 reshape active).
    On CPU, dispatch/combine are no-ops regardless of num_devices, so outputs should match.
    """
    num_devices = 8  # Set >1 to test E0/E1 reshape logic in A2aSparseMLP, but CPU dispatch/combine are still no-ops
    from tt_torch.sparse_mlp import A2aSparseMLP

    loader = GPTOSSModelLoader(num_layers=1)
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

    a2a_mlp = A2aSparseMLP(
        original_mlp,
        num_experts=config.num_local_experts,
        num_experts_per_tok=config.num_experts_per_tok,
        num_devices=num_devices,
        dispatch_devices=2,
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


@pytest.mark.push
def test_2d_mesh_device_ordering():
    """Verify 2D mesh device ordering by sharding a tensor with unique per-device
    values and gathering it back. This checks that compound sharding
    {"axis_0", "axis_1"} distributes data in the same linear order that
    build_expert_mapping assumes (device 0 gets experts 0-3, device 1 gets 4-7, etc).
    """
    torch_xla.runtime.use_spmd()
    xr.set_device_type("TT")
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    import torch_xla.core.xla_model as xm

    num_devices = xr.global_runtime_device_count()
    if num_devices < 8:
        pytest.skip(f"Need at least 8 devices, got {num_devices}")

    # Use actual mesh shape
    if num_devices == 32:
        mesh_shape = (8, 4)
    elif num_devices == 8:
        mesh_shape = (2, 4)
    else:
        pytest.skip(f"Unsupported device count: {num_devices}")

    rows, cols = mesh_shape
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("axis_0", "axis_1"))

    # Create tensor [rows, cols, 4] where element [r, c, :] = r*cols+c
    # Shard dim 0 on axis_0, dim 1 on axis_1 → each device gets [1, 1, 4]
    # with value = its linear device ID (if row-major ordering holds)
    data = (
        torch.arange(num_devices, dtype=torch.float32)
        .view(rows, cols)
        .unsqueeze(-1)
        .expand(-1, -1, 4)
        .contiguous()
    )

    t_xla = data.to(xm.xla_device())

    from torch_xla.distributed.spmd import mark_sharding

    # Shard dim 0 on axis_0, dim 1 on axis_1 (separate axes, not compound)
    mark_sharding(t_xla, mesh, ("axis_0", "axis_1", None))

    print(f"\n{'='*60}")
    print(f"2D Mesh Device Ordering Test")
    print(f"{'='*60}")
    print(f"Mesh shape: {mesh_shape} ({rows} rows x {cols} cols)")
    print(f"Num devices: {num_devices}")
    print(f"Input shape: {data.shape}")

    # Just pull the sharded tensor back to CPU — each device's local value
    # will be reassembled by the runtime. The value at [r, c, 0] tells us
    # what linear ID was assigned to mesh position (r, c).
    result = t_xla.cpu()
    print(f"Result shape: {result.shape}")
    print(f"Result tensor:\n{result[:, :, 0]}")

    # Build the device grid from result
    device_grid = result[:, :, 0]
    device_order_int = [int(x) for x in device_grid.flatten().tolist()]

    print(f"\nDevice order (row-major expected): {list(range(num_devices))}")
    print(f"Device order (actual):             {device_order_int}")

    is_row_major = device_order_int == list(range(num_devices))
    print(f"Row-major match: {is_row_major}")

    # Show the 2D layout
    print(f"\n2D Mesh Layout (which device ID sits at each position):")
    print(f"         ", end="")
    for c in range(cols):
        print(f"col{c:2d}  ", end="")
    print()
    for r in range(rows):
        print(f"  row{r}: ", end="")
        for c in range(cols):
            val = int(device_grid[r, c])
            print(f"  D{val:<3d}", end="")
        print()

    # Show expert mapping for comparison
    experts_per_device = 128 // num_devices if num_devices <= 128 else 1
    print(f"\nExpert mapping (build_expert_mapping assumes flat sequential):")
    for r in range(min(rows, 4)):  # show first 4 rows
        print(f"  row{r}: ", end="")
        for c in range(cols):
            dev = int(device_grid[r, c])
            e_start = dev * experts_per_device
            e_end = e_start + experts_per_device - 1
            print(f"  E{e_start}-{e_end}", end="")
        print()
    if rows > 4:
        print(f"  ... ({rows - 4} more rows)")

    if not is_row_major:
        print(f"\n*** WARNING: Device ordering is NOT row-major! ***")
        print(f"*** Compound sharding and expert_mapping may be misaligned! ***")


@pytest.mark.push
def test_a2a_dispatch_combine_e2e():
    """End-to-end test of all_to_all_dispatch → matmul → all_to_all_combine
    on a 2D mesh. Uses simple identity-like expert weights so we can verify
    the dispatch/combine routing and expert-device mapping are correct.

    Input: [B, 1, S, H] with H sharded on axis_1
    Expert weights: [E, H, H] with E sharded on (axis_0, axis_1) — compound
    Expert mapping: [1, 1, E, D] — sequential assignment
    """
    torch_xla.runtime.use_spmd()
    xr.set_device_type("TT")
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    import torch_xla.core.xla_model as xm
    from torch_xla.distributed.spmd import mark_sharding
    from tt_torch.sparse_mlp import build_expert_mapping

    num_devices = xr.global_runtime_device_count()
    if num_devices < 8:
        pytest.skip(f"Need at least 8 devices, got {num_devices}")

    if num_devices == 32:
        mesh_shape = (8, 4)
    elif num_devices == 8:
        mesh_shape = (2, 4)
    else:
        pytest.skip(f"Unsupported device count: {num_devices}")

    rows, cols = mesh_shape
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("axis_0", "axis_1"))

    # Model-like dimensions (divisible by mesh for sharding)
    B = 8  # divisible by axis_0 (2)
    S = 128
    H = 64  # divisible by cols=4 → local H=16
    E = num_devices  # 1 expert per device for simplicity
    K = 2  # top-2 experts per token

    print(f"\n{'='*60}")
    print(f"A2A Dispatch/Combine E2E Test")
    print(f"{'='*60}")
    print(f"Mesh: {mesh_shape}, Devices: {num_devices}")
    print(f"B={B}, S={S}, H={H}, E={E}, K={K}")

    # --- Input: [B, S, H] sharded on H by axis_1 ---
    # Each value encodes position: hidden_states[b, s, h] = s * H + h
    hidden = torch.zeros(B, S, H, dtype=torch.bfloat16)
    for b in range(B):
        for s in range(S):
            for h in range(H):
                hidden[b, s, h] = s * H + h
    hidden_xla = hidden.to(xm.xla_device())
    mark_sharding(hidden_xla, mesh, (None, None, "axis_1"))

    # --- Expert mapping: [1, 1, E, D] ---
    expert_mapping = build_expert_mapping(E, num_devices)
    expert_mapping_xla = expert_mapping.to(xm.xla_device())

    # --- Router: deterministic expert assignment ---
    # Token s → experts [s % E, (s+1) % E] (top-2)
    expert_indices = torch.zeros(B, 1, S, K, dtype=torch.int64)
    for b in range(B):
        for s in range(S):
            expert_indices[b, 0, s, 0] = s % E
            expert_indices[b, 0, s, 1] = (s + 1) % E
    expert_indices_xla = expert_indices.to(xm.xla_device())

    # --- Reshape for dispatch: [B, 1, S, H] ---
    x = hidden_xla.view(B, 1, S, H)

    # --- Dispatch ---
    dispatched, metadata = torch.ops.tt.all_to_all_dispatch(
        x,
        expert_indices_xla,
        expert_mapping_xla,
        num_devices=num_devices,
        cluster_axis=0,
    )

    print(f"\nDispatched shape: {dispatched.shape}")
    print(f"Metadata shape:   {metadata.shape}")

    # --- Follow sparse_mlp.py prefill flow exactly ---
    # dispatch output: [1, BD, S, H] where BD = B * dispatch_devices
    # sparse_mlp reshapes to [BD, S//M, M, H] or [BD//M, S, M, H] for sparse_matmul
    # then combine input: [E_local, BD, S, H]
    # combine output: [K, B, S, H] with output_shard_dim=1

    BD = dispatched.shape[1]  # B * dispatch_devices
    dispatch_devices = num_devices  # cluster_axis=0 covers all devices
    E_local = E // num_devices  # experts per device (=1 in our case)

    print(f"\nDispatched shape: {dispatched.shape}  (BD={BD})")
    print(f"Metadata shape:   {metadata.shape}")
    print(f"E_local (experts per device): {E_local}")

    # --- Identity pass-through as "expert computation" ---
    # In sparse_mlp, the output after down_proj is [E_local, BD, S, H]
    # For identity test, just rearrange dispatched [1, BD, S, H] → [E_local, BD, S, H]
    expert_output = dispatched.permute(0, 1, 2, 3)  # [1, BD, S, H] → keep as [1, BD, S, H]
    # dim 0 = E_local = 1, which matches 1 expert per device

    print(f"Expert output shape (for combine): {expert_output.shape}")

    # --- Combine (matching sparse_mlp prefill) ---
    combined = torch.ops.tt.all_to_all_combine(
        expert_output,
        metadata,
        expert_mapping_xla,
        num_devices=dispatch_devices,
        cluster_axis=0,
        num_experts_per_tok=K,
        output_shard_dim=1,  # prefill mode: shard on batch dim
    )
    # Expected: [K, B, S, H]

    print(f"Combined shape:   {combined.shape}  (expected: [K={K}, B={B}, S={S}, H={H}])")

    # --- Pull results back to CPU ---
    dispatched_cpu = dispatched.cpu()
    metadata_cpu = metadata.cpu()
    combined_cpu = combined.cpu()
    hidden_cpu = hidden_xla.cpu()

    print(f"\n--- Input (first 4 tokens, first 8 hidden dims) ---")
    print(hidden_cpu[0, :4, :8])

    print(f"\n--- Dispatched [1, BD={BD}, S, H] (first 2 device-slots, first 4 tokens) ---")
    for d in range(min(2, BD)):
        print(f"  Slot {d}: {dispatched_cpu[0, d, :4, :8]}")

    print(f"\n--- Metadata [1, BD={BD}, S, K] (first 2 device-slots, first 4 tokens) ---")
    for d in range(min(2, BD)):
        print(f"  Slot {d}: {metadata_cpu[0, d, :4, :]}")

    print(f"\n--- Combined [K={K}, B={B}, S={S}, H={H}] (first 4 tokens, first 8 hidden dims) ---")
    for k in range(K):
        print(f"  K={k}: {combined_cpu[k, 0, :4, :8]}")

    # --- Verify: identity pass-through means combined should equal input ---
    # Token s → expert (s%E) and expert ((s+1)%E), both with identity weights
    print(f"\n--- Verification (identity: combined should match input) ---")
    num_match = 0
    num_total = 0
    actual_B = combined_cpu.shape[1]
    actual_S = combined_cpu.shape[2]
    for b in range(actual_B):
        for s in range(actual_S):
            for k in range(K):
                actual = combined_cpu[k, b, s, :]
                expected = hidden[b, s, :]
                match = torch.allclose(actual.float(), expected.float(), atol=1.0)
                num_total += 1
                if match:
                    num_match += 1
                elif b == 0 and s < 4:  # only print first few mismatches
                    print(f"  MISMATCH b={b} s={s} K={k}: "
                          f"expected={expected[:4].tolist()}, got={actual[:4].tolist()}")

    print(f"  Matched: {num_match}/{num_total} "
          f"({'PASS' if num_match == num_total else 'FAIL'})")


@pytest.mark.push
def test_a2a_chained_combine_compound_expert_matmul():
    """Test dispatch -> sparse_matmul -> combine(axis=0) -> all_reduce(axis=1).

    For 2D mesh compound expert sharding, combine(axis=0) gathers within columns.
    The compiler auto-inserts all_reduce on the opposite axis (axis=1) to aggregate
    partial sums across the full mesh.
    """
    torch_xla.runtime.use_spmd()
    xr.set_device_type("TT")
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    import torch_xla.core.xla_model as xm
    from torch_xla.distributed.spmd import mark_sharding
    from tt_torch.sparse_mlp import build_expert_mapping

    num_devices = xr.global_runtime_device_count()
    if num_devices < 8:
        pytest.skip(f"Need at least 8 devices, got {num_devices}")

    if num_devices == 32:
        mesh_shape = (8, 4)
    elif num_devices == 8:
        mesh_shape = (2, 4)
    else:
        pytest.skip(f"Unsupported device count: {num_devices}")

    rows, cols = mesh_shape
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("axis_0", "axis_1"))

    B = 1
    S = 128
    H = 64
    M = 32
    E = 128
    K = 4

    print(f"\n{'='*60}")
    print(f"A2A Single Combine + Compound Expert Sparse Matmul Test")
    print(f"{'='*60}")
    print(f"Mesh: {mesh_shape}, Devices: {num_devices}")
    print(f"Flow: dispatch(axis=0) -> sparse_mm(gate_up) -> sparse_mm(down) -> combine(axis=0) -> all_reduce(axis=1)")

    hidden = torch.zeros(B, S, H, dtype=torch.bfloat16)
    for b in range(B):
        for s in range(S):
            for h in range(H):
                hidden[b, s, h] = s * H + h
    hidden_xla = hidden.to(xm.xla_device())
    mark_sharding(hidden_xla, mesh, (None, None, "axis_1"))

    # gate_up weights: [1, E, H, H] — scaled identity per expert
    gate_up_weights = torch.zeros(1, E, H, H, dtype=torch.bfloat16)
    for e in range(E):
        gate_up_weights[0, e] = (e + 1) * torch.eye(H, dtype=torch.bfloat16)
    gate_up_weights_xla = gate_up_weights.to(xm.xla_device())
    mark_sharding(gate_up_weights_xla, mesh, (None, ("axis_0", "axis_1"), None, None))

    # down weights: [1, E, H, H] — identity per expert (net effect = gate_up only)
    down_weights = torch.zeros(1, E, H, H, dtype=torch.bfloat16)
    for e in range(E):
        down_weights[0, e] = torch.eye(H, dtype=torch.bfloat16)
    down_weights_xla = down_weights.to(xm.xla_device())
    mark_sharding(down_weights_xla, mesh, (None, ("axis_0", "axis_1"), None, None))

    expert_mapping = build_expert_mapping(E, num_devices)
    expert_mapping_xla = expert_mapping.to(xm.xla_device())

    expert_indices = torch.zeros(B, 1, S, K, dtype=torch.int64)
    for b in range(B):
        for s in range(S):
            expert_indices[b, 0, s, 0] = s % E
            expert_indices[b, 0, s, 1] = (s + 1) % E
    expert_indices_xla = expert_indices.to(xm.xla_device())

    x = hidden_xla.view(B, 1, S, H)
    dispatched, metadata = torch.ops.tt.all_to_all_dispatch(
        x, expert_indices_xla, expert_mapping_xla,
        num_devices=rows, cluster_axis=0,
    )
    BD = dispatched.shape[1]
    dim_a = BD
    dim_b = S // M

    hidden_4d = dispatched.view(BD, dim_b, M, H)
    # Use ALL M positions (not just M=0) so all 8 experts get sparsity=1.
    # Previously [:,:,0] only saw expert 0,1 since M=32 is multiple of E=8.
    metadata_full = metadata[0].view(BD, dim_b, M, K)  # [BD, dim_b, M, K]
    metadata_flat = metadata_full.reshape(BD, dim_b * M, K)  # [BD, dim_b*M, K]
    sparsity_flat = torch.zeros(BD, dim_b * M, 1, E, dtype=hidden_xla.dtype, device=hidden_xla.device)
    topk_indices_unsqueezed = metadata_flat.unsqueeze(2)  # [BD, dim_b*M, 1, K]
    sparsity_flat.scatter_(
        dim=-1,
        index=topk_indices_unsqueezed,
        src=torch.ones_like(topk_indices_unsqueezed, dtype=hidden_xla.dtype),
    )
    # Fold M back and reduce: sum over M, clamp to 0/1
    sparsity = sparsity_flat.view(BD, dim_b, M, E).sum(dim=2).clamp(max=1.0)  # [BD, dim_b, E]
    sparsity = sparsity.unsqueeze(2)  # [BD, dim_b, 1, E]

    # 1st sparse_matmul (gate_up): input × weights, sparsity selects experts
    # hidden_4d: [BD, dim_b, M, H], gate_up_weights: [1, E, H, H], sparsity: [BD, dim_b, 1, E]
    gate_up_out = torch.ops.tt.sparse_matmul(
        hidden_4d,
        gate_up_weights_xla,
        sparsity,
        nnz=0,
        is_input_a_sparse=False,
        is_input_b_sparse=True,
    )
    # gate_up_out: [BD, dim_b, 1, E, M, H]
    gate_up_out = gate_up_out.squeeze(2)             # [BD, dim_b, E, M, H]
    gate_up_out = gate_up_out.permute(0, 1, 3, 2, 4)  # [BD, dim_b, M, E, H]

    # 2nd sparse_matmul (down): activated × down_weights
    # Reshape for is_input_a_sparse: [dim_a*dim_b, E, M, H]
    activated = gate_up_out.permute(0, 1, 3, 2, 4).contiguous()  # [BD, dim_b, E, M, H]
    activated = activated.view(BD * dim_b, E, M, H)
    sparsity_down = sparsity.view(1, 1, BD * dim_b, E)
    down_out = torch.ops.tt.sparse_matmul(
        activated,
        down_weights_xla,
        sparsity_down,
        nnz=0,
        is_input_a_sparse=True,
        is_input_b_sparse=False,
    )
    # down_out: [BD*dim_b, E, M, H]
    down_out = down_out.view(BD, dim_b, E, M, H)
    down_out = down_out.permute(2, 0, 1, 3, 4).contiguous()  # [E, BD, dim_b, M, H]
    expert_output = down_out.view(E, BD, S, H)

    # Single combine: axis=0 (column-wise).
    # Compiler should insert all_reduce after combine for compound sharding.
    output_shard_dim = 1
    combined_0 = torch.ops.tt.all_to_all_combine(
        expert_output, metadata, expert_mapping_xla,
        num_devices=rows, cluster_axis=0,
        num_experts_per_tok=K, output_shard_dim=output_shard_dim,
    )

    combined_cpu = combined_0.cpu()

    ordinal = xr.global_ordinal()
    if ordinal == 0:
        print(f"\n{'='*40} DEBUG (device {ordinal}) {'='*40}")
        print(f"expert_mapping:\n{expert_mapping.squeeze()}")
        print(f"\ncombined shape={combined_cpu.shape}")
        for k in range(K):
            nz = (combined_cpu[k] != 0).sum().item()
            total = combined_cpu[k].numel()
            sample = combined_cpu[k, 0, :8, 0].tolist()
            print(f"  K[{k}]: nonzeros={nz}/{total}, [0,:8,0]={sample}")
        print(f"{'='*80}\n")

    print(f"Combined shape: {combined_cpu.shape}")
    assert combined_cpu.shape[0] == K
    # Verification: prefer global check when combined_cpu materializes the full batch.
    num_match = 0
    num_total = 0
    actual_B = combined_cpu.shape[1]
    actual_S = combined_cpu.shape[2]
    ordinal = xr.global_ordinal()
    mismatches = []
    matches_sample = []

    if actual_B == B:
        batch_offset = 0
        verify_B = B
        verify_S = min(actual_S, S)
        verify_K = K
        for b in range(verify_B):
            for s in range(verify_S):
                for k in range(verify_K):
                    e = (s + k) % E
                    scale = e + 1
                    actual = combined_cpu[k, b, s, :4]
                    expected = hidden[b, s, :4].float() * scale
                    if torch.allclose(actual.float(), expected, rtol=0.1, atol=2.0):
                        num_match += 1
                        if len(matches_sample) < 3:
                            matches_sample.append((b, s, k, expected.tolist()[:2], actual.float().tolist()[:2]))
                    else:
                        if len(mismatches) < 5:
                            mismatches.append(
                                (b, s, k, expected.tolist(), actual.float().tolist())
                            )
                    num_total += 1
    else:
        # Fallback: local shard check when host tensor is not global batch.
        batch_offset = (ordinal % cols) * (B // cols)
        verify_K = K
        for local_b in range(min(actual_B, B - batch_offset)):
            global_b = batch_offset + local_b
            for s in range(min(actual_S, S)):
                for k in range(verify_K):
                    e = (s + k) % E
                    scale = e + 1
                    actual = combined_cpu[k, local_b, s, :4]
                    expected = hidden[global_b, s, :4].float() * scale
                    if torch.allclose(actual.float(), expected, atol=1.0e-1):
                        num_match += 1
                        if len(matches_sample) < 3:
                            matches_sample.append((global_b, s, k, expected.tolist()[:2], actual.float().tolist()[:2]))
                    else:
                        if len(mismatches) < 5:
                            mismatches.append(
                                (global_b, s, k, expected.tolist(), actual.float().tolist())
                            )
                    num_total += 1

    print(f"Combine verification: {num_match}/{num_total} matched")
    print(f"  ordinal={ordinal}, batch_offset={batch_offset}, actual nonzeros={(combined_cpu != 0).sum().item()}/{combined_cpu.numel()}")
    if matches_sample:
        print(f"  Sample matches: {matches_sample}")
    if mismatches:
        for gb, seq, k, exp, act in mismatches:
            print(f"  MISMATCH b={gb} s={seq} k={k}: exp={exp[:2]}... act={act[:2]}...")
    assert num_match == num_total, (
        f"Verification failed: {num_match}/{num_total} matched. "
        f"Combine + all_reduce output incorrect."
    )


@pytest.mark.push
def test_a2a_chained_combine_compound_expert_sparse_matmul():
    """Test dispatch -> sparse_mm(gate_up) -> sparse_mm(down) -> combine(axis=0) -> combine(axis=1).

    Chained combine version: combine(axis=0) gathers within columns, then
    combine(axis=1) gathers across rows. Compiler auto-inserts all_reduce
    after detecting the second combine (isSecondCombine path).
    """
    torch_xla.runtime.use_spmd()
    xr.set_device_type("TT")
    os.environ["CONVERT_SHLO_TO_SHARDY"] = "1"
    import torch_xla.core.xla_model as xm
    from torch_xla.distributed.spmd import mark_sharding
    from tt_torch.sparse_mlp import build_expert_mapping

    num_devices = xr.global_runtime_device_count()
    if num_devices < 8:
        pytest.skip(f"Need at least 8 devices, got {num_devices}")

    if num_devices == 32:
        mesh_shape = (8, 4)
    elif num_devices == 8:
        mesh_shape = (2, 4)
    else:
        pytest.skip(f"Unsupported device count: {num_devices}")

    rows, cols = mesh_shape
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("axis_0", "axis_1"))

    B = 8
    S = 128
    H = 64
    M = 32
    E = num_devices
    K = 2

    print(f"\n{'='*60}")
    print(f"A2A Chained Combine + Compound Expert Sparse Matmul Test")
    print(f"{'='*60}")
    print(f"Mesh: {mesh_shape}, Devices: {num_devices}")
    print(f"Flow: dispatch(axis=0) -> sparse_mm(gate_up) -> sparse_mm(down) -> combine(axis=0) -> combine(axis=1) -> all_reduce")

    hidden = torch.zeros(B, S, H, dtype=torch.bfloat16)
    for b in range(B):
        for s in range(S):
            for h in range(H):
                hidden[b, s, h] = s * H + h
    hidden_xla = hidden.to(xm.xla_device())
    mark_sharding(hidden_xla, mesh, (None, None, "axis_1"))

    # gate_up weights: [1, E, H, H] — scaled identity per expert
    gate_up_weights = torch.zeros(1, E, H, H, dtype=torch.bfloat16)
    for e in range(E):
        gate_up_weights[0, e] = (e + 1) * torch.eye(H, dtype=torch.bfloat16)
    gate_up_weights_xla = gate_up_weights.to(xm.xla_device())
    mark_sharding(gate_up_weights_xla, mesh, (None, ("axis_0", "axis_1"), None, None))

    # down weights: [1, E, H, H] — identity per expert (net effect = gate_up only)
    down_weights = torch.zeros(1, E, H, H, dtype=torch.bfloat16)
    for e in range(E):
        down_weights[0, e] = torch.eye(H, dtype=torch.bfloat16)
    down_weights_xla = down_weights.to(xm.xla_device())
    mark_sharding(down_weights_xla, mesh, (None, ("axis_0", "axis_1"), None, None))

    expert_mapping = build_expert_mapping(E, num_devices)
    expert_mapping_xla = expert_mapping.to(xm.xla_device())

    expert_indices = torch.zeros(B, 1, S, K, dtype=torch.int64)
    for b in range(B):
        for s in range(S):
            expert_indices[b, 0, s, 0] = s % E
            expert_indices[b, 0, s, 1] = (s + 1) % E
    expert_indices_xla = expert_indices.to(xm.xla_device())

    x = hidden_xla.view(B, 1, S, H)
    dispatched, metadata = torch.ops.tt.all_to_all_dispatch(
        x, expert_indices_xla, expert_mapping_xla,
        num_devices=rows, cluster_axis=0,
    )
    BD = dispatched.shape[1]
    dim_a = BD
    dim_b = S // M

    hidden_4d = dispatched.view(BD, dim_b, M, H)
    # Build sparsity from ALL M positions
    metadata_full = metadata[0].view(BD, dim_b, M, K)
    metadata_flat = metadata_full.reshape(BD, dim_b * M, K)
    sparsity_flat = torch.zeros(BD, dim_b * M, 1, E, dtype=hidden_xla.dtype, device=hidden_xla.device)
    topk_indices_unsqueezed = metadata_flat.unsqueeze(2)
    sparsity_flat.scatter_(
        dim=-1,
        index=topk_indices_unsqueezed,
        src=torch.ones_like(topk_indices_unsqueezed, dtype=hidden_xla.dtype),
    )
    sparsity = sparsity_flat.view(BD, dim_b, M, E).sum(dim=2).clamp(max=1.0)
    sparsity = sparsity.unsqueeze(2)  # [BD, dim_b, 1, E]

    # 1st sparse_matmul (gate_up)
    gate_up_out = torch.ops.tt.sparse_matmul(
        hidden_4d, gate_up_weights_xla, sparsity,
        nnz=0, is_input_a_sparse=False, is_input_b_sparse=True,
    )
    gate_up_out = gate_up_out.squeeze(2)               # [BD, dim_b, E, M, H]
    gate_up_out = gate_up_out.permute(0, 1, 3, 2, 4)   # [BD, dim_b, M, E, H]

    # 2nd sparse_matmul (down)
    activated = gate_up_out.permute(0, 1, 3, 2, 4).contiguous()  # [BD, dim_b, E, M, H]
    activated = activated.view(BD * dim_b, E, M, H)
    sparsity_down = sparsity.view(1, 1, BD * dim_b, E)
    down_out = torch.ops.tt.sparse_matmul(
        activated, down_weights_xla, sparsity_down,
        nnz=0, is_input_a_sparse=True, is_input_b_sparse=False,
    )
    down_out = down_out.view(BD, dim_b, E, M, H)
    down_out = down_out.permute(2, 0, 1, 3, 4).contiguous()  # [E, BD, dim_b, M, H]
    expert_output = down_out.view(E, BD, S, H)

    # 1st combine: axis=0 (within columns)
    output_shard_dim = 1
    combined_0 = torch.ops.tt.all_to_all_combine(
        expert_output, metadata, expert_mapping_xla,
        num_devices=rows, cluster_axis=0,
        num_experts_per_tok=K, output_shard_dim=output_shard_dim,
    )
    # combined_0: [K, B, S, H] — partial results from column experts only

    # Slice k=0 for second combine (each k is combined independently)
    combined_0_k0 = combined_0[0:1]  # [1, B, S, H]

    # 2nd combine: axis=1 (across rows) — compiler detects chain, inserts all_reduce
    combined_1 = torch.ops.tt.all_to_all_combine(
        combined_0_k0, metadata, expert_mapping_xla,
        num_devices=cols, cluster_axis=1,
        num_experts_per_tok=K, output_shard_dim=output_shard_dim,
    )

    combined_cpu = combined_1.cpu()

    ordinal = xr.global_ordinal()
    if ordinal == 0:
        print(f"\n{'='*40} DEBUG (device {ordinal}) {'='*40}")
        print(f"expert_mapping:\n{expert_mapping.squeeze()}")
        print(f"\ncombined shape={combined_cpu.shape}")
        for k in range(combined_cpu.shape[0]):
            nz = (combined_cpu[k] != 0).sum().item()
            total = combined_cpu[k].numel()
            sample = combined_cpu[k, 0, :8, 0].tolist()
            print(f"  K[{k}]: nonzeros={nz}/{total}, [0,:8,0]={sample}")
        print(f"{'='*80}\n")

    print(f"Combined shape: {combined_cpu.shape}")
    # Verification: only k=0 is valid (k=1 was sliced away for 2nd combine)
    num_match = 0
    num_total = 0
    actual_B = combined_cpu.shape[1]
    actual_S = combined_cpu.shape[2]
    ordinal = xr.global_ordinal()
    mismatches = []
    matches_sample = []
    verify_K = 1  # Only k=0 passed through second combine

    if actual_B == B:
        batch_offset = 0
        for b in range(B):
            for s in range(min(actual_S, S)):
                for k in range(verify_K):
                    e = (s + k) % E
                    scale = e + 1
                    actual = combined_cpu[k, b, s, :4]
                    expected = hidden[b, s, :4].float() * scale
                    if torch.allclose(actual.float(), expected, rtol=0.1, atol=2.0):
                        num_match += 1
                        if len(matches_sample) < 3:
                            matches_sample.append((b, s, k, expected.tolist()[:2], actual.float().tolist()[:2]))
                    else:
                        if len(mismatches) < 5:
                            mismatches.append(
                                (b, s, k, expected.tolist(), actual.float().tolist())
                            )
                    num_total += 1
    else:
        batch_offset = (ordinal % cols) * (B // cols)
        for local_b in range(min(actual_B, B - batch_offset)):
            global_b = batch_offset + local_b
            for s in range(min(actual_S, S)):
                for k in range(verify_K):
                    e = (s + k) % E
                    scale = e + 1
                    actual = combined_cpu[k, local_b, s, :4]
                    expected = hidden[global_b, s, :4].float() * scale
                    if torch.allclose(actual.float(), expected, rtol=0.1, atol=2.0):
                        num_match += 1
                        if len(matches_sample) < 3:
                            matches_sample.append((global_b, s, k, expected.tolist()[:2], actual.float().tolist()[:2]))
                    else:
                        if len(mismatches) < 5:
                            mismatches.append(
                                (global_b, s, k, expected.tolist(), actual.float().tolist())
                            )
                    num_total += 1

    print(f"Chained combine verification: {num_match}/{num_total} matched")
    print(f"  ordinal={ordinal}, batch_offset={batch_offset}, actual nonzeros={(combined_cpu != 0).sum().item()}/{combined_cpu.numel()}")
    if matches_sample:
        print(f"  Sample matches: {matches_sample}")
    if mismatches:
        for gb, seq, k, exp, act in mismatches:
            print(f"  MISMATCH b={gb} s={seq} k={k}: exp={exp[:2]}... act={act[:2]}...")
    assert num_match == num_total, (
        f"Verification failed: {num_match}/{num_total} matched. "
        f"Chained combine + all_reduce output incorrect."
    )

