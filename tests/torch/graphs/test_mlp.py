# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Callable

import numpy as np
import pytest
import torch
import torch_xla
import torch_xla.distributed.spmd as xs
import torch_xla.runtime as xr
from infra import Framework, run_graph_test
from infra.evaluators import ComparisonConfig, PccConfig
from torch_xla.distributed.spmd import Mesh
from transformers import AutoModelForCausalLM
from transformers.models.falcon.modeling_falcon import FalconMLP
from transformers.models.gemma.modeling_gemma import GemmaMLP
from transformers.models.llama.modeling_llama import LlamaMLP
from transformers.models.mistral.modeling_mistral import MistralMLP
from transformers.models.qwen2.modeling_qwen2 import Qwen2MLP
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP
from tt_torch import TT_DENSE_EXPERTS_BACKEND_NAME, TT_MOE_BACKEND_NAME
from tt_torch.sparse_mlp import (
    ACTIVATION_DEEPSEEK,
    A2aSparseMLP,
    _moe_activation,
    enable_sparse_mlp,
)

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
    "gpt_oss": ["20B", "120B"],
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


@pytest.mark.extended
@pytest.mark.nightly
@parametrize_arch(["single_device", "llmbox"])
@pytest.mark.parametrize("seq_len", [1024])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("llama").items(),
    ids=[str(k) for k in get_available_variants("llama").keys()],
)
@pytest.mark.filecheck(["matmul_with_activation_silu.ttnn.mlir"])
def test_llama_mlp(seq_len, variant, variant_config, arch, request):
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
        request=request,
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
@pytest.mark.filecheck(["matmul_with_activation_gelu.ttnn.mlir"])
def test_gemma_mlp(seq_len, variant, variant_config, arch, request):
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
        request=request,
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
@pytest.mark.parametrize(
    "mlp_type", [TT_DENSE_EXPERTS_BACKEND_NAME, TT_MOE_BACKEND_NAME]
)
@parametrize_arch(["single_device", "llmbox", "galaxy"])
@pytest.mark.parametrize(
    "variant,variant_config",
    get_available_variants("gpt_oss").items(),
    ids=[str(k) for k in get_available_variants("gpt_oss").keys()],
)
def test_gpt_oss_mlp(variant, variant_config, arch, mlp_type, request):
    if mlp_type == TT_MOE_BACKEND_NAME and arch != "llmbox":
        # tt_moe requires a multi-chip SPMD mesh with an EP axis larger than 1;
        # galaxy currently hangs (https://github.com/tenstorrent/tt-xla/issues/3941).
        pytest.skip("tt_moe HF backend requires llmbox arch")
    if variant == GPTOSSModelVariant.GPT_OSS_120B and arch == "single_device":
        pytest.skip("120B model too large for single device")
    if mlp_type == TT_DENSE_EXPERTS_BACKEND_NAME:
        request.node.add_marker(
            pytest.mark.filecheck(["matmul_with_activation_silu.ttnn.mlir"])
        )
    xr.set_device_type("TT")

    loader = GPTOSSModelLoader(variant=variant, num_layers=1)
    config = loader.load_config()

    batch_size = 1
    seq_len = 128

    model = AutoModelForCausalLM.from_config(
        config, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    # Route GptOssExperts through the chosen HF backend. tt_dense replaces the
    # legacy torch_overrides.py monkey patch; tt_moe replaces enable_sparse_mlp.
    model.config._experts_implementation = mlp_type
    model.eval()
    mlp = model.model.layers[0].mlp

    hidden_states = torch.randn(
        (batch_size, seq_len, config.hidden_size), dtype=torch.bfloat16
    )

    if arch in ("llmbox", "galaxy"):
        batch_size = 2 if arch == "llmbox" else 4
        num_devices = xr.global_runtime_device_count()
        mesh_shape = (batch_size, num_devices // batch_size)
        device_ids = np.array(range(num_devices))
        mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))
        if mlp_type == TT_MOE_BACKEND_NAME:
            # tt_moe HF backend reads the global SPMD mesh inside its forward to
            # pick the EP cluster axis; run_graph_test doesn't set it for us.
            xs.set_global_mesh(mesh)

        def get_shard_spec(mlp, args, kwargs):
            shard_specs = {}

            # Router weights (not sharded).
            shard_specs[mlp.router.weight] = (None, "batch")
            shard_specs[mlp.router.bias] = (None,)

            # Shard experts across devices.
            if mlp_type == TT_DENSE_EXPERTS_BACKEND_NAME:
                shard_specs[mlp.experts.gate_up_proj] = ("model", "batch", None)
                shard_specs[mlp.experts.gate_up_proj_bias] = ("model", None)
                shard_specs[mlp.experts.down_proj] = ("model", None, "batch")
                shard_specs[mlp.experts.down_proj_bias] = ("model", "batch")
            elif mlp_type == TT_MOE_BACKEND_NAME:
                shard_specs[mlp.experts.gate_up_proj] = (("batch", "model"), None, None)
                shard_specs[mlp.experts.gate_up_proj_bias] = (("batch", "model"), None)
                shard_specs[mlp.experts.down_proj] = (("batch", "model"), None, None)
                shard_specs[mlp.experts.down_proj_bias] = (("batch", "model"), None)

            return shard_specs

    else:
        mesh = None
        get_shard_spec = None

    comparison_config = ComparisonConfig()
    if (
        variant == "120B"
        and arch == "single_device"
        and mlp_type == TT_DENSE_EXPERTS_BACKEND_NAME
    ):
        comparison_config = ComparisonConfig(pcc=PccConfig(required_pcc=0.985))

    run_graph_test(
        mlp,
        [hidden_states],
        comparison_config=comparison_config,
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


"""GPT-OSS sparse MLP small-batch decode (tile-padding path)"""


@pytest.mark.nightly
@parametrize_arch(["llmbox"])
@pytest.mark.parametrize(
    "decode_batch,decode_seq_len",
    [(8, 1), (1, 1), (2, 8), (5, 5)],
    ids=["bsz8_sl1", "bsz1_sl1", "bsz2_sl8", "bsz5_sl5"],
)
def test_gpt_oss_sparse_mlp_small_batch(decode_batch, decode_seq_len, arch):
    """
    Exercises A2aSparseMLP's tile-padding path: when decode_batch * decode_seq_len
    is not a multiple of TILE (32), forward must pad the batch dim and slice it
    off the output. Mirrors small-batch decode (e.g. bsz=8 seq_len=1 -> 8 tokens,
    pad to 32). bsz5_sl5 covers the gcd(seq_len, TILE)=1 path where padded_batch
    must be a multiple of TILE itself.
    """
    xr.set_device_type("TT")

    loader = GPTOSSModelLoader(variant=GPTOSSModelVariant.GPT_OSS_20B, num_layers=1)
    config = loader.load_config()

    model = AutoModelForCausalLM.from_config(
        config, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    model.eval()
    mlp = model.model.layers[0].mlp

    hidden_states = torch.randn(
        (decode_batch, decode_seq_len, config.hidden_size), dtype=torch.bfloat16
    )

    # llmbox: (2, 4) mesh, cluster_axis=0 -> 2-way dispatch.
    num_devices = xr.global_runtime_device_count()
    mesh_dim0 = 2
    mesh_shape = (mesh_dim0, num_devices // mesh_dim0)
    device_ids = np.array(range(num_devices))
    mesh = Mesh(device_ids, mesh_shape, ("batch", "model"))
    mlp = enable_sparse_mlp(mlp, mesh=mesh_shape, cluster_axis=0)

    def get_shard_spec(mlp, args, kwargs):
        return {
            mlp.router.weight: (None, "batch"),
            mlp.router.bias: (None,),
            mlp.experts.gate_up_proj: (("batch", "model"), None, None),
            mlp.experts.gate_up_proj_bias: (("batch", "model"), None),
            mlp.experts.down_proj: (("batch", "model"), None, None),
            mlp.experts.down_proj_bias: (("batch", "model"), None),
        }

    # Small-batch sparse MoE in bf16 sees slightly worse PCC than the seq=128
    # case — relax the threshold to match the existing 120B/single_device entry.
    comparison_config = ComparisonConfig(pcc=PccConfig(required_pcc=0.98))

    run_graph_test(
        mlp,
        [hidden_states],
        comparison_config=comparison_config,
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=get_shard_spec,
    )


@pytest.mark.nightly
@pytest.mark.single_device
def test_a2a_sparse_mlp_cpu_parity():
    """Verify A2aSparseMLP produces the same results as the original MLP on CPU.

    Tests with num_devices=1 (no E0/E1 reshape) and num_devices>1 (E0/E1 reshape active).
    On CPU, dispatch/combine are no-ops regardless of num_devices, so outputs should match.
    PCC checks internally because we don't have support for cpu vs cpu comparison in our current test infra.
    """
    loader = GPTOSSModelLoader(num_layers=1)
    model = loader.load_model()
    config = loader.load_config()
    inputs = loader.load_inputs()

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
        num_devices=8,
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

    assert pcc_out > 0.99, f"Output PCC too low: {pcc_out:.6f}"
    assert pcc_scores > 0.99, f"Router scores PCC too low: {pcc_scores:.6f}"


@pytest.mark.nightly
@pytest.mark.single_device
def test_moe_activation_swiglu_limit():
    """Compile ``_moe_activation`` with the DeepSeek-V4 swiglu_limit clamp on
    TT and compare against the CPU reference. Inputs are scaled so a
    meaningful fraction of values lies outside ``[-swiglu_limit, swiglu_limit]``
    and exercises the clamp."""

    class _DeepseekSwigluActivation(torch.nn.Module):
        """Wraps ``_moe_activation`` in the DeepSeek path so the graph tester can
        compile it on TT and compare against CPU."""

        def __init__(self, swiglu_limit: float):
            super().__init__()
            self.swiglu_limit = swiglu_limit

        def forward(self, gate_up):
            return _moe_activation(
                gate_up, ACTIVATION_DEEPSEEK, swiglu_limit=self.swiglu_limit
            )

    xr.set_device_type("TT")

    model = _DeepseekSwigluActivation(swiglu_limit=2.0).to(torch.bfloat16)
    # Scale * 5 so most values exceed the clamp bound.
    gate_up = torch.randn(1, 64, 256, dtype=torch.bfloat16) * 5.0

    run_graph_test(model, [gate_up], framework=Framework.TORCH)
