# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import copy
import itertools
import json
import logging
import math
import os
import sys
import warnings
from abc import ABC, ABCMeta, abstractmethod
from collections import defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import FrozenInstanceError, asdict, dataclass, field, replace
from functools import partial
from os import mkdir
from os.path import exists, isdir, isfile, join
from typing import (
    Callable,
    Collection,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from packaging.version import Version
from torch import nn
from transformers import PreTrainedModel
from transformers.activations import get_activation
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import ModuleUtilsMixin
from transformers.models.xlm_roberta.modeling_xlm_roberta import (
    XLMRobertaOutput,
    XLMRobertaSdpaSelfAttention,
    XLMRobertaSelfAttention,
    XLMRobertaSelfOutput,
)
from transformers.pytorch_utils import Conv1D
from transformers.utils import cached_file
from transformers.utils.import_utils import is_torchvision_available

try:
    from bitsandbytes.nn import Int8Params, Linear4bit, Linear8bitLt, Params4bit

    bitsandbytes_available = True
except ImportError:
    bitsandbytes_available = False

try:
    from safetensors.torch import load_file, save_file

    safetensors_available = True
except ImportError:
    safetensors_available = False

from .utils import (
    ACTIVATION_RENAME,
    ADAPTERFUSION_CONFIG_NAME,
    ADAPTERFUSION_WEIGHTS_NAME,
    CONFIG_NAME,
    HEAD_CONFIG_NAME,
    HEAD_WEIGHTS_NAME,
    INTERFACE_CONFIG_NAME,
    SAFE_ADAPTERFUSION_WEIGHTS_NAME,
    SAFE_HEAD_WEIGHTS_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_NAME,
    AdapterSetup,
    AdapterType,
    ForwardContext,
    __version__,
    fix_seed,
    get_adapter_config_hash,
    inherit_doc,
    multigetattr,
    multihasattr,
    patch_forward,
    prefix_attention_mask,
    resolve_adapter_config,
    resolve_adapter_path,
)

logger = logging.getLogger(__name__)


class TaskSpecificSingularValue(nn.Module):
    def __init__(self, rank: int, dtype: Optional[str] = None):
        super().__init__()
        self.rank = rank
        self.dtype = dtype
        self.values = nn.Parameter(torch.ones(rank, dtype=dtype))

    def forward(self):
        return torch.diag(self.values)


class TaskSpecificLinear(nn.Module):
    def __init__(self, rank: int, dtype: Optional[str] = None):
        self.rank = (rank,)
        self.dtype = dtype
        self.values = nn.Parameter(torch.ones(rank, rank, dtype=dtype))

    def forward(self):
        return self.values


TASK_SPECIFIC_MATRIX_CLS = {
    "singular_values": TaskSpecificSingularValue,
    "linear": TaskSpecificLinear,
}


class AdapterConfig(Mapping):
    """
    Base class for all adaptation methods. This class does not define specific configuration keys, but only provides
    some common helper methods.

    Args:
        architecture (str, optional): The type of adaptation method defined by the configuration.
    """

    architecture: Optional[str] = None

    def __init__(self):
        raise TypeError(
            "AdapterConfig is an abstract class and cannot be instantiated."
        )

    # We want to emulate a simple form of immutability while keeping the ability to add custom attributes.
    # Therefore, we don't allow changing attribute values if set once.
    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise FrozenInstanceError()
        else:
            object.__setattr__(self, name, value)

    def __delattr__(self, name):
        raise FrozenInstanceError()

    def __getitem__(self, key):
        return self.__dict__[key]

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __eq__(self, other):
        return self.to_dict() == other.to_dict()

    def to_dict(self):
        """Converts the config class to a Python dict."""
        return asdict(self)

    def replace(self, **changes):
        """Returns a new instance of the config class with the specified changes applied."""
        return replace(self, **changes)

    @classmethod
    def from_dict(cls, config):
        """Creates a config class from a Python dict."""
        if isinstance(config, AdapterConfig):
            return config

        # the constructor does not accept additional kwargs, so add them separately
        defined_kwargs, new_kwargs = {}, {}
        for k, v in config.items():
            if k in cls.__dataclass_fields__.keys():
                defined_kwargs[k] = v
            else:
                new_kwargs[k] = v
        obj = cls(**defined_kwargs)
        for k, v in new_kwargs.items():
            setattr(obj, k, v)
        return obj

    @staticmethod
    def _get_config_class(config_dict):
        """
        Returns the matching config class for the given config dict based on its "architecture" key.
        """
        architecture = config_dict.get("architecture", None)
        arch_to_config = {
            "lora": LoRAConfig,
            "mtl_lora": MTLLoRAConfig,
            "prefix_tuning": PrefixTuningConfig,
            "union": ConfigUnion,
            None: BnConfig,
            "prompt_tuning": PromptTuningConfig,
            "reft": ReftConfig,
        }
        cls_new = arch_to_config[architecture]

        return cls_new

    @classmethod
    def load(cls, config: Union[dict, str], download_kwargs=None, **kwargs):

        if not config:
            return None
        # if force_download is set, skip the local map
        if download_kwargs and download_kwargs.get("force_download", False):
            local_map = None
        else:
            local_map = ADAPTER_CONFIG_MAP
        if download_kwargs:
            config_dict = resolve_adapter_config(
                config, local_map=local_map, **download_kwargs
            )
        else:
            config_dict = resolve_adapter_config(config, local_map=local_map)
        # convert back to dict to allow attr overrides
        if isinstance(config_dict, AdapterConfig):
            cls_new = config_dict.__class__
            config_dict = config_dict.to_dict()
        else:
            cls_new = cls._get_config_class(config_dict)
        # The check for "None" is necessary because of the example script flags.
        config_dict.update((k, v) for k, v in kwargs.items() if v is not None)
        return cls_new.from_dict(config_dict)


@dataclass(eq=False)
class ReftConfig(AdapterConfig):

    layers: Union[Literal["all"], List[int]] = "all"
    prefix_positions: int = 3
    suffix_positions: int = 0
    r: int = 1
    orthogonality: bool = True
    tied_weights: bool = False
    subtract_projection = True
    dropout: float = 0.05
    non_linearity: Optional[str] = None
    dtype: Optional[str] = None

    architecture: str = "reft"

    output_reft: bool = True
    init_weights_seed: Optional[int] = None


@dataclass(eq=False)
class LoReftConfig(ReftConfig):
    """
    Low-Rank Linear Subspace ReFT method proposed in Wu et al. (2024). See https://arxiv.org/pdf/2404.03592.
    """

    layers: Union[Literal["all"], List[int]] = "all"
    prefix_positions: int = 3
    suffix_positions: int = 0
    r: int = 1
    orthogonality: bool = True
    tied_weights: bool = False
    dtype: Optional[str] = None


@dataclass(eq=False)
class NoReftConfig(ReftConfig):
    """
    Variation of LoReft without orthogonality constraint.
    """

    layers: Union[Literal["all"], List[int]] = "all"
    prefix_positions: int = 3
    suffix_positions: int = 0
    r: int = 1
    orthogonality: bool = False
    tied_weights: bool = False
    dtype: Optional[str] = None


@dataclass(eq=False)
class DiReftConfig(ReftConfig):
    """
    Variation of LoReft without orthogonality constraint and projection subtraction as proposed in Wu et al. (2024). See https://arxiv.org/pdf/2404.03592.
    """

    layers: Union[Literal["all"], List[int]] = "all"
    prefix_positions: int = 3
    suffix_positions: int = 0
    r: int = 1
    orthogonality: bool = False
    tied_weights: bool = False
    subtract_projection = False
    dtype: Optional[str] = None


@dataclass(eq=False)
class BnConfig(AdapterConfig):

    # Required options
    mh_adapter: bool
    output_adapter: bool

    reduction_factor: Union[float, Mapping]
    non_linearity: str

    # Options with defaults
    original_ln_before: bool = False
    original_ln_after: bool = True
    ln_before: bool = False
    ln_after: bool = False
    init_weights: str = "bert"
    init_weights_seed: Optional[int] = None
    is_parallel: bool = False
    scaling: Union[float, str] = 1.0
    use_gating: bool = False
    residual_before_ln: Union[bool, str] = True
    adapter_residual_before_ln: bool = False
    inv_adapter: Optional[str] = None
    inv_adapter_reduction_factor: Optional[float] = None
    cross_adapter: bool = False
    leave_out: List[int] = field(default_factory=list)
    dropout: float = 0.0
    phm_layer: bool = False
    phm_dim: int = 4
    factorized_phm_W: Optional[bool] = True
    shared_W_phm: Optional[bool] = False
    shared_phm_rule: Optional[bool] = True
    factorized_phm_rule: Optional[bool] = False
    phm_c_init: Optional[str] = "normal"
    phm_init_range: Optional[float] = 0.0001
    learn_phm: Optional[bool] = True
    hypercomplex_nonlinearity: Optional[str] = "glorot-uniform"
    phm_rank: Optional[int] = 1
    phm_bias: Optional[bool] = True
    stochastic_depth: Optional[float] = 0.0

    # We want to emulate a simple form of immutability while keeping the ability to add custom attributes.
    # Therefore, we don't allow changing attribute values if set once.
    def __setattr__(self, name, value):
        if name in self.__dict__:
            raise FrozenInstanceError()
        elif name == "invertible_adapter":
            # This is for backwards compatibility. In v1, invertible adapters were specified in a nested config dict.
            # Now, we have two config keys directly in the adapter config.
            if value:
                object.__setattr__(self, "inv_adapter", value["block_type"])
                object.__setattr__(
                    self,
                    "inv_adapter_reduction_factor",
                    value["reduction_factor"],
                )
        else:
            object.__setattr__(self, name, value)


@dataclass(eq=False)
class SeqBnConfig(BnConfig):
    """
    The adapter architecture proposed by Pfeiffer et al. (2020). See https://arxiv.org/pdf/2005.00247.pdf.
    """

    original_ln_before: bool = True
    original_ln_after: bool = True
    residual_before_ln: Union[bool, str] = True
    adapter_residual_before_ln: bool = False
    ln_before: bool = False
    ln_after: bool = False
    mh_adapter: bool = False
    output_adapter: bool = True
    non_linearity: str = "relu"
    reduction_factor: Union[float, Mapping] = 16


@dataclass(eq=False)
class PrefixTuningConfig(AdapterConfig):

    architecture: Optional[str] = "prefix_tuning"

    encoder_prefix: bool = True
    cross_prefix: bool = True
    leave_out: List[int] = field(default_factory=list)

    flat: bool = False
    prefix_length: int = 30
    bottleneck_size: int = 512
    non_linearity: str = "tanh"
    dropout: float = 0.0
    use_gating: bool = False
    shared_gating: bool = True
    init_weights_seed: Optional[int] = None


@dataclass(eq=False)
class PromptTuningConfig(AdapterConfig):
    """
    The Prompt Tuning architecture proposed by Lester et al. (2021). See https://arxiv.org/pdf/2104.08691.pdf

    Args:
        prompt_length (int): The number of tokens in the prompt.
            Defaults to 10.
        prompt_init (str): The initialization method for the prompt. Can be either "random_uniform" or "from_string".
            Defaults to "random_uniform".
        prompt_init_text (str): The text to use for prompt initialization if prompt_init="from_string".
        random_uniform_scale (float): The scale of the random uniform initialization if prompt_init="random_uniform".
            Defaults to 0.5 as in the paper.
        combine (str):
            The method used to combine the prompt with the input. Can be either "prefix" or "prefix_after_bos".
            Defaults to "prefix".
        init_weights_seed (:obj:`int`, optional): The seed to use for the initialization of the adapter weights per layer.
            Important:  set, the seed will be reset for all adapter modules, meaning that all adapter modules will have the same
            initialization. If not set, the seed will be set once and each adapter module has random weights initialization. Defaults to None.
    """

    architecture: str = "prompt_tuning"

    prompt_length: int = 10
    prompt_init: str = "random_uniform"
    prompt_init_text: Optional[str] = None
    random_uniform_scale = 0.5
    combine: str = "prefix"
    init_weights_seed: Optional[int] = None


@dataclass(eq=False)
class LoRAConfig(AdapterConfig):

    architecture: Optional[str] = "lora"

    selfattn_lora: bool = True
    intermediate_lora: bool = False
    output_lora: bool = False
    leave_out: List[int] = field(default_factory=list)

    r: int = 8
    alpha: int = 8
    dropout: float = 0.0
    attn_matrices: List[str] = field(default_factory=lambda: ["q", "v"])
    composition_mode: str = "add"
    init_weights: str = "lora"
    init_weights_seed: Optional[int] = None
    use_gating: bool = False
    vera_d: Optional[float] = None
    vera_b: Optional[float] = None
    dtype: Optional[str] = None


@dataclass(eq=False)
class IA3Config(LoRAConfig):
    """
    The 'Infused Adapter by Inhibiting and Amplifying Inner Activations' ((IA)^3) architecture proposed by Liu et al.
    (2022). See https://arxiv.org/pdf/2205.05638.pdf. (IA)^3 builds on top of LoRA, however, unlike the additive
    composition of LoRA, it scales weights of a layer using an injected vector.
    """

    selfattn_lora: bool = True
    intermediate_lora: bool = True
    output_lora: bool = False
    leave_out: List[int] = field(default_factory=list)

    r: int = 1
    alpha: int = 1
    dropout: float = 0.0
    attn_matrices: List[str] = field(default_factory=lambda: ["k", "v"])
    composition_mode: str = "scale"
    init_weights: str = "ia3"
    use_gating: bool = False
    dtype: Optional[str] = None


@dataclass(eq=False)
class VeraConfig(LoRAConfig):
    """
    Lora Config that applies vector-based random matrix adaptation. It adds
    trainable matrices 'd' and 'b' while keeping the original LoRA matrices
    frozen, random, and shared across layers. See more through their paper:
    https://arxiv.org/pdf/2310.11454. Note that `r` will still be supplied
    since we are still initializing decomposition matrices A and B.
    The `composition_mode` parameter should also be set to `add`.
    """

    selfattn_lora: bool = True
    intermediate_lora: bool = False
    output_lora: bool = False

    r: int = 8
    vera_d: Optional[float] = 0.1
    vera_b: Optional[float] = 0.0
    init_weights: str = "vera"
    composition_mode: str = "add"
    dtype: Optional[str] = None


class MultiTaskConfig(AdapterConfig):
    """
    Flag class for all multi task adaptation methods.
    This class does not define specific configuration keys, but only provides
    some common helper methods.
    """

    ...


@dataclass(eq=False)
class MTLLoRAConfig(LoRAConfig, MultiTaskConfig):
    """
    The MTL-LoRA architecture, proposed by Yang et al. (2024), combine LoRA with multi-task learning. See https://arxiv.org/pdf/2410.09437.pdf.
    This configuration extends LoRA to support multi-task adaptation, allowing parameter-efficient fine-tuning across
    multiple tasks while leveraging low-rank reparameterization techniques.

    Args:
        n_up_projection (int, optional): The number of additional projection layers for task-specific adaptations.
            Defaults to 1.
        task_specific_matrix_type (Literal["singular_values", "linear"], optional): The type of task-specific matrix
            used in adaptation. Can be either "singular_values" (which adapts using singular value decomposition-based
            transformations) or "linear" (which applies a learned linear transformation). Defaults to "singular_values".
        weights_sharpness (float, optional): A scaling factor controlling the sharpness of the task-specific weight
            transformations, influencing how much task adaptation is applied. Defaults to 0.05.
    """

    architecture: Optional[str] = "mtl_lora"
    n_up_projection: int = 1
    task_specific_matrix_type: Literal["singular_values", "linear"] = "singular_values"
    weights_sharpness: float = 0.05


class LoRA(nn.Module):
    sharable_parameters = [
        "lora_A",
        "lora_B",
    ]

    def __init__(
        self,
        lora_A_shape,
        lora_B_shape,
        config: LoRAConfig,
        gating_heads: int = 1,
        name: str = None,
        **kwargs,
    ):
        super().__init__()
        assert (
            config.composition_mode == "add"
        ), "LoRA module only supports composition_mode='add'."
        self.r = config.r
        self.lora_alpha = config.alpha
        self.composition_mode = config.composition_mode
        self.attn_matrices = config.attn_matrices
        self.use_gating = config.use_gating
        self.name = name
        # Optional dropout
        if config.dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=config.dropout)
        else:
            self.lora_dropout = lambda x: x

        self.dtype = getattr(torch, config.dtype) if config.dtype else None
        # Actual trainable parameters
        self.lora_A = nn.Parameter(torch.zeros(lora_A_shape, dtype=self.dtype))

        self.lora_B = nn.Parameter(torch.zeros(lora_B_shape, dtype=self.dtype))
        self.scaling = self.lora_alpha / self.r

        # Set seed for reproducibility if specified in config
        fix_seed(config.init_weights_seed)

        # For compatibility with (IA)^3, allow all init_weights types here.
        # Usually should be "lora".
        if config.init_weights == "lora":
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        elif config.init_weights == "bert":
            nn.init.normal_(self.lora_A, std=0.02)
            nn.init.normal_(self.lora_B, std=0.02)
        elif config.init_weights == "ia3":
            nn.init.ones_(self.lora_A)
            nn.init.ones_(self.lora_B)
        elif config.init_weights == "vera":
            nn.init.kaiming_uniform_(self.lora_A)
            nn.init.kaiming_uniform_(self.lora_B)
        else:
            raise ValueError(
                "Unknown init_weights type: {}".format(config.init_weights)
            )

        if self.use_gating:
            self.gate = nn.Linear(lora_A_shape[-1], gating_heads)
            nn.init.normal_(self.gate.weight, std=0.02)

    @property
    def delta_w(self) -> torch.Tensor:
        return self.lora_B @ self.lora_A

    def com(
        self, weights: torch.Tensor, added: torch.Tensor, scaling=None
    ) -> torch.Tensor:
        """Performs the composition operation between existing and injected weights."""
        if scaling is None:
            scaling = self.scaling
        return weights + added * scaling

    def com_inv(self, weights: torch.Tensor, added: torch.Tensor) -> torch.Tensor:
        """Inverts the composition operation between existing and injected weights."""
        return weights - added * self.scaling

    def get_parameters(self, parameters_names=None):
        parameters_names = parameters_names or self.sharable_parameters
        return nn.ParameterDict(
            {
                param_name: deepcopy(getattr(self, param_name))
                for param_name in parameters_names
                if hasattr(self, param_name) and param_name in self.sharable_parameters
            }
        )

    def set_parameters(self, parameters):
        for name, param in parameters.items():
            if name in self.sharable_parameters:
                setattr(self, name, param)

    def forward(self, hidden_states: Optional[torch.Tensor], layer_input: torch.Tensor):
        if hidden_states is None:
            hidden_states = layer_input
        hidden_states = (
            self.lora_dropout(hidden_states)
            @ torch.t(self.lora_A)
            @ torch.t(self.lora_B)
        )
        if self.use_gating:
            gate = torch.sigmoid(self.gate(layer_input))
            gate = torch.mean(gate, dim=1).unsqueeze(-1)
            hidden_states = hidden_states * gate
        else:
            gate = None
            hidden_states = hidden_states * self.scaling

        return hidden_states, gate


class MTLLoRA(LoRA):
    def __init__(
        self,
        lora_A_shape,
        lora_B_shape,
        config: MTLLoRAConfig,
        gating_heads: int = 1,
        **kwargs,
    ):
        super().__init__(
            lora_A_shape,
            lora_B_shape,
            config,
            gating_heads,
        )
        self.lora_B = nn.Parameter(
            torch.zeros(config.n_up_projection, *lora_B_shape, dtype=self.dtype)
        )
        if config.init_weights == "lora":
            nn.init.zeros_(self.lora_B)
        elif config.init_weights == "bert":
            nn.init.normal_(self.lora_B, std=0.02)

        self.task_specific = TASK_SPECIFIC_MATRIX_CLS[config.task_specific_matrix_type](
            config.r
        )

        self.weights = nn.Parameter(torch.ones(config.n_up_projection))
        self.weights_sharpness = config.weights_sharpness

    def apply_weights(self, hidden_states: torch.Tensor):
        w = (self.weights / self.weights_sharpness).exp()
        # hidden_states : n_up_projection x bsz x seq_len x dim
        return torch.sum(((w[:, None, None, None] * hidden_states) / w.sum()), 0)

    def forward(
        self,
        hidden_states: Optional[torch.Tensor],
        layer_input: torch.Tensor,
    ):
        if hidden_states is None:
            hidden_states = layer_input

        hidden_states = (
            self.lora_dropout(hidden_states)
            @ torch.t(self.lora_A)
            @ self.task_specific()
            @ torch.transpose(self.lora_B.unsqueeze(1), -1, -2)
        )
        # apply weights
        hidden_states = self.apply_weights(hidden_states)

        if self.use_gating:
            gate = torch.sigmoid(self.gate(layer_input))
            gate = torch.mean(gate, dim=1).unsqueeze(-1)
            hidden_states = hidden_states * gate
        else:
            gate = None
            hidden_states = hidden_states * self.scaling

        return hidden_states, gate


class IA3(nn.Module):
    def __init__(
        self,
        lora_A_shape,
        lora_B_shape,
        config: LoRAConfig,
        gating_heads: int = 1,
        name: str = None,
        **kwargs,
    ):
        super().__init__()
        assert (
            config.composition_mode == "scale"
        ), "IA3 module only supports composition_mode='scale'."
        if config.r > 1:
            raise ValueError("Can only use composition_mode='scale' when r == 1.")
        self.r = config.r
        self.lora_alpha = config.alpha
        self.composition_mode = config.composition_mode
        self.attn_matrices = config.attn_matrices
        self.use_gating = config.use_gating
        self.name = name
        # Optional dropout
        if config.dropout > 0.0:
            raise ValueError("IA3 module does not support dropout.")

        # Actual trainable parameters
        self.lora_B = nn.Parameter(torch.zeros(lora_B_shape))
        self.scaling = self.lora_alpha

        # Set seed for reproducibility if specified in config
        fix_seed(config.init_weights_seed)

        # For compatibility with LoRA, allow all init_weights types here.
        # Usually should be "ia3".
        if config.init_weights == "lora":
            logger.warning(
                "(IA)^3 module initialized with LoRA zero init. Ignore if this is intended."
            )
            nn.init.zeros_(self.lora_B)
        elif config.init_weights == "bert":
            nn.init.normal_(self.lora_B, std=0.02)
        elif config.init_weights == "ia3":
            nn.init.ones_(self.lora_B)
        else:
            raise ValueError(
                "Unknown init_weights type: {}".format(config.init_weights)
            )

        if self.use_gating:
            self.gate = nn.Linear(lora_A_shape[-1], gating_heads)
            nn.init.normal_(self.gate.weight, std=0.02)

    @property
    def delta_w(self) -> torch.Tensor:
        return self.lora_B

    def com(
        self, weights: torch.Tensor, added: torch.Tensor, scaling=None
    ) -> torch.Tensor:
        """Performs the composition operation between existing and injected weights."""
        if scaling is None:
            scaling = self.scaling
        return weights * (added * scaling)

    def com_inv(self, weights: torch.Tensor, added: torch.Tensor) -> torch.Tensor:
        """Inverts the composition operation between existing and injected weights."""
        return weights / (added * self.scaling)

    def forward(self, hidden_states: Optional[torch.Tensor], layer_input: torch.Tensor):
        scaling_vector = self.lora_B.view(1, 1, -1).repeat(layer_input.shape[0], 1, 1)
        if hidden_states is None:
            hidden_states = scaling_vector
        else:
            hidden_states = hidden_states * scaling_vector
        if self.use_gating:
            gate = torch.sigmoid(self.gate(layer_input))
            gate = torch.mean(gate, dim=1).unsqueeze(-1)
            hidden_states = hidden_states * gate
        else:
            gate = None
            hidden_states = hidden_states * self.scaling

        return hidden_states, gate


class Vera(nn.Module):
    def __init__(
        self,
        lora_A_shape,
        lora_B_shape,
        config: LoRAConfig,
        gating_heads: int = 1,
        name: str = None,
        **kwargs,
    ):
        super().__init__()
        self.d = config.vera_d
        self.b = config.vera_b
        self.r = config.r
        self.alpha = config.alpha
        self.use_gating = config.use_gating
        self.name = name

        # check to make sure that the `composition_mode` is set to `add`
        self.composition_mode = config.composition_mode
        if self.composition_mode != "add":
            raise ValueError("Vera module only supports composition_mode='add'.")

        # Optional dropout
        if config.dropout > 0.0:
            self.lora_dropout = nn.Dropout(p=config.dropout)
        else:
            self.lora_dropout = lambda x: x

        self.lora_A_shape = lora_A_shape
        self.lora_B_shape = lora_B_shape
        self.d_shape = self.lora_A_shape[0]
        self.b_shape = self.lora_B_shape[0]

        # Actual trainable parameters
        self.vera_D = nn.Parameter(torch.diag(torch.ones(self.d_shape) * self.d))
        self.vera_B = nn.Parameter(torch.diag(torch.ones(self.b_shape) * self.b))
        self.scaling = self.alpha / self.r

        if self.use_gating:
            self.gate = nn.Linear(lora_A_shape[-1], gating_heads)
            nn.init.normal_(self.gate.weight, std=0.02)

    @property
    def delta_w(self) -> torch.Tensor:
        parameters = ForwardContext.get_context().shared_parameters[self.name]
        lora_A = parameters["lora_A"]
        lora_B = parameters["lora_B"]
        return self.vera_B @ lora_B @ self.vera_D @ lora_A

    def com(
        self, weights: torch.Tensor, added: torch.Tensor, scaling=None
    ) -> torch.Tensor:
        """Performs the composition operation between existing and injected weights."""
        if scaling is None:
            scaling = self.scaling
        return weights + added * scaling

    def com_inv(self, weights: torch.Tensor, added: torch.Tensor) -> torch.Tensor:
        """Inverts the composition operation between existing and injected weights."""
        return weights - added * self.scaling

    def forward(self, hidden_states: Optional[torch.Tensor], layer_input: torch.Tensor):
        parameters = ForwardContext.get_context().shared_parameters[self.name]
        lora_A = parameters["lora_A"]
        lora_B = parameters["lora_B"]

        if hidden_states is None:
            hidden_states = layer_input

        if getattr(self, "lora_dropout"):
            hidden_states = self.lora_dropout(hidden_states)

        hidden_states = hidden_states @ torch.t(
            self.vera_B @ lora_B @ self.vera_D @ lora_A
        )

        if self.use_gating:
            gate = torch.sigmoid(self.gate(layer_input))
            gate = torch.mean(gate, dim=1).unsqueeze(-1)
            hidden_states = hidden_states * gate
        else:
            gate = None
            hidden_states = hidden_states * self.scaling

        return hidden_states, gate


def init_shared_vera_parameters(lora_A_shape, lora_B_shape, adapter_config, device):
    """
    This function creates the shared random matrices A and B that are used across all Vera layers.
    These matrices are frozen and initialized according to the specified initialization strategy.

    Args:
        lora_A_shape: The shape of the A matrix
        lora_B_shape: The shape of the B matrix
        adapter_config (dict): The adapter configuration
        device: The device to place the parameters on

    Returns:
        nn.ParameterDict: Dictionary containing:
            - lora_A: Parameter of shape lora_A_shape
            - lora_B: Parameter of shape lora_B_shape
    """
    parameters = nn.ParameterDict()

    # Set seed for reproducibility if specified in config
    fix_seed(adapter_config.init_weights_seed)

    # initialize frozen, random tensors A, B
    dtype = getattr(torch, adapter_config.dtype) if adapter_config.dtype else None
    parameters["lora_A"] = nn.Parameter(
        torch.zeros(lora_A_shape, dtype=dtype).to(device), requires_grad=False
    )
    parameters["lora_B"] = nn.Parameter(
        torch.zeros(lora_B_shape, dtype=dtype).to(device), requires_grad=False
    )

    if adapter_config["init_weights"] == "lora":
        # initialize A the same way as the default for nn.Linear and B to zero
        nn.init.kaiming_uniform_(parameters["lora_A"], a=math.sqrt(5))
        nn.init.zeros_(parameters["lora_B"])
    elif adapter_config["init_weights"] == "bert":
        nn.init.normal_(parameters["lora_A"], std=0.02)
        nn.init.normal_(parameters["lora_B"], std=0.02)
    elif adapter_config["init_weights"] == "ia3":
        nn.init.ones_(parameters["lora_A"])
        nn.init.ones_(parameters["lora_B"])
    elif adapter_config["init_weights"] == "vera":
        nn.init.kaiming_uniform_(parameters["lora_A"])
        nn.init.kaiming_uniform_(parameters["lora_B"])
    else:
        raise ValueError(
            "Unknown init_weights type: {}".format(adapter_config["init_weights"])
        )

    return parameters


class AdapterCompositionBlock(Sequence):
    def __init__(self, *children):
        self.children = [parse_composition(b, None) for b in children]

    def __getitem__(self, key):
        return self.children[key]

    def __len__(self):
        return len(self.children)

    def __eq__(self, o: object) -> bool:
        if isinstance(o, type(self)):
            return all([c1 == c2 for c1, c2 in zip(self.children, o.children)])
        else:
            return False

    def __repr__(self):
        child_repr = ", ".join(map(str, self.children))
        return f"{self.__class__.__name__}[{child_repr}]"

    def first(self):
        if not isinstance(self.children[0], AdapterCompositionBlock):
            return self.children[0]
        else:
            return self.children[0].first()

    def last(self):
        if not isinstance(self.children[-1], AdapterCompositionBlock):
            return self.children[-1]
        else:
            return self.children[-1].last()

    @property
    def parallel_channels(self):
        return max(
            [
                (b.parallel_channels if isinstance(b, AdapterCompositionBlock) else 1)
                for b in self.children
            ]
        )

    def flatten(self) -> Set[str]:
        return set(
            itertools.chain(
                *[[b] if isinstance(b, str) else b.flatten() for b in self.children]
            )
        )

    def _get_save_kwargs(self):
        return None

    def to_dict(self):
        save_dict = {
            "type": self.__class__.__name__,
            "children": [
                (
                    c.to_dict()
                    if isinstance(c, AdapterCompositionBlock)
                    else {"type": "single", "children": [c]}
                )
                for c in self.children
            ],
        }
        if kwargs := self._get_save_kwargs():
            save_dict["kwargs"] = kwargs
        return save_dict

    @classmethod
    def from_dict(cls, data):
        children = []
        for child in data["children"]:
            if child["type"] == "single":
                children.append(child["children"][0])
            else:
                children.append(cls.from_dict(child))
        return getattr(sys.modules[__name__], data["type"])(
            *children, **data.get("kwargs", {})
        )


class Parallel(AdapterCompositionBlock):
    def __init__(self, *parallel_adapters: List[str]):
        """
        Can be used to perform inference for multiple tasks (i.e., adapters) in parallel (for the same input).

        See AdapterDrop https://arxiv.org/abs/2010.11918
        """
        super().__init__(*parallel_adapters)

    @property
    def parallel_channels(self):
        return len(self.children)


class Stack(AdapterCompositionBlock):
    def __init__(self, *stack_layers: List[Union[AdapterCompositionBlock, str]]):
        super().__init__(*stack_layers)


class Fuse(AdapterCompositionBlock):
    def __init__(
        self,
        *fuse_stacks: List[Union[AdapterCompositionBlock, str]],
        name: Optional[str] = None,
    ):
        super().__init__(*fuse_stacks)
        self._name = name

    # TODO-V2 pull this up to all block classes?
    @property
    def name(self):
        if self._name:
            return self._name
        else:
            return ",".join(
                [c if isinstance(c, str) else c.last() for c in self.children]
            )


class Split(AdapterCompositionBlock):
    def __init__(
        self,
        *split_adapters: List[Union[AdapterCompositionBlock, str]],
        splits: Union[List[int], int],
    ):
        super().__init__(*split_adapters)
        self.splits = (
            splits if isinstance(splits, list) else [splits] * len(split_adapters)
        )

    def _get_save_kwargs(self):
        return {"splits": self.splits}


class BatchSplit(AdapterCompositionBlock):
    def __init__(
        self,
        *split_adapters: List[Union[AdapterCompositionBlock, str]],
        batch_sizes: Union[List[int], int],
    ):
        super().__init__(*split_adapters)
        self.batch_sizes = (
            batch_sizes
            if isinstance(batch_sizes, list)
            else [batch_sizes] * len(split_adapters)
        )

    def _get_save_kwargs(self):
        return {"batch_sizes": self.batch_sizes}


class MultiTask(AdapterCompositionBlock):
    def __init__(self, *children):
        super().__init__(*children)


class Average(AdapterCompositionBlock):
    def __init__(
        self,
        *average_adapters: List[Union[AdapterCompositionBlock, str]],
        weights: Optional[List[float]] = None,
        normalize_weights: bool = True,
    ):
        super().__init__(*average_adapters)
        if weights is not None:
            # normalize weights
            if normalize_weights:
                sum_weights = sum(weights) if weights else 1
                self.weights = [w / sum_weights for w in weights]
            else:
                self.weights = weights
        else:
            self.weights = [1 / len(average_adapters)] * len(average_adapters)

    def _get_save_kwargs(self):
        return {"weights": self.weights}


class AdapterLayerBase(metaclass=ABCMeta):
    """
    Base class for all adaptation methods that require per-layer modules.

    Make sure the 'adapter_modules_name' attribute is overriden in derived classes.
    """

    adapter_modules_name = ""

    @property
    def adapter_modules(self) -> Collection:
        return getattr(self, self.adapter_modules_name)

    @property
    def layer_idx(self):
        return getattr(self, "_layer_idx", -1)

    @layer_idx.setter
    def layer_idx(self, layer_idx):
        idx = getattr(self, "_layer_idx", layer_idx)
        assert idx == layer_idx
        setattr(self, "_layer_idx", idx)

    def get_active_setup(self):
        if hasattr(self, "adapters_config"):
            # First check current context before falling back to defined setup
            context = AdapterSetup.get_context()
            if context is not None:
                adapter_setup = context.adapter_setup
            else:
                adapter_setup = self.adapters_config.active_setup
        else:
            adapter_setup = None
        skip_adapters = adapter_setup is None or (
            self.adapters_config.skip_layers is not None
            and self.layer_idx in self.adapters_config.skip_layers
        )
        if not skip_adapters and (
            len(set(self.adapter_modules.keys()) & adapter_setup.flatten()) > 0
        ):
            return adapter_setup
        else:
            return None

    def _store_gating_score(self, adapter_name, gating_score):
        context = ForwardContext.get_context()
        if context.output_adapter_gating_scores:
            gating_cache = context.adapter_gating_scores
            if self.layer_idx not in gating_cache[adapter_name]:
                gating_cache[adapter_name][self.layer_idx] = {}
            gating_score = gating_score.detach().squeeze().cpu().numpy()
            if len(gating_score.shape) == 0:
                gating_score = np.expand_dims(gating_score, axis=0)
            cache_score = gating_cache[adapter_name][self.layer_idx].get(
                self.location_key, None
            )
            if cache_score is not None:
                gating_cache[adapter_name][self.layer_idx][
                    self.location_key
                ] = np.column_stack((cache_score, gating_score))
            else:
                gating_cache[adapter_name][self.layer_idx][
                    self.location_key
                ] = gating_score

    def _store_fusion_attentions(self, fusion_name, attentions):
        context = ForwardContext.get_context()
        if context.output_adapter_fusion_attentions:
            attention_cache = context.adapter_fusion_attentions
            if self.layer_idx not in attention_cache[fusion_name]:
                attention_cache[fusion_name][self.layer_idx] = {}
            attention_cache[fusion_name][self.layer_idx][self.location_key] = attentions

    @abstractmethod
    def add_adapter(self, adapter_name: str, layer_idx: int) -> bool:
        """Adds a new adapter module to the layer.

        Args:
            adapter_name (str): The name of the new adapter to add.
            layer_idx (int):
                The index of the adapters layer (this should be set once by the first added adapter and the kept fix).

        Returns:
            bool: True if the adapter was added, False otherwise.
        """
        raise NotImplementedError()

    def average_adapter(
        self,
        adapter_name: str,
        input_adapters: Dict[str, float],
        combine_strategy,
        **kwargs,
    ) -> bool:
        """Averages a set of adapter modules into a new adapter module.

        Args:
            adapter_name (str): The name of the new (averaged) adapter module to add.
            input_adapters (Dict[str, float]): Dictionary of adapter names and their corresponding weights.
            combine_strategy (str): The strategy to combine the adapters. Available strategies depend on the used adapter method, see: https://docs.adapterhub.ml/adapter_composition.html#merging-adapters
            **kwargs: Additional arguments that are specific to the combine_strategy. E.g. svd_rank for LoRA.

        Returns:
            bool: True if the adapter was added, False otherwise.
        """
        # add new adapter
        if self.add_adapter(adapter_name, self.layer_idx):
            if combine_strategy != "linear":
                # You get the adapter type from the input adapters
                raise ValueError(
                    f"Combine strategy {combine_strategy} not supported for the chosen adapter methods."
                )

            # average weights linearly
            avg_state_dict = {}
            for name, weight in input_adapters.items():
                if name in self.adapter_modules:
                    module = self.adapter_modules[name]
                    for k, v in module.state_dict().items():
                        if k in avg_state_dict:
                            avg_state_dict[k] += weight * v
                        else:
                            avg_state_dict[k] = weight * v
                else:
                    self.delete_adapter(adapter_name)  # clean up before raising error
                    raise ValueError("Adapter {} not found.".format(name))

            # load averaged weights
            self.adapter_modules[adapter_name].load_state_dict(avg_state_dict)

            return True

        return False

    def delete_adapter(self, adapter_name: str):
        """Deletes an adapter module from the layer.

        Args:
            adapter_name (str): The name of the adapter to delete.
        """
        if adapter_name in self.adapter_modules:
            del self.adapter_modules[adapter_name]

    def share_parameters(
        self,
        name: str,
        adapter_names: List,
        reference_adapter_name: Optional[str],
    ):
        pass  # default implementation does nothing as multi task is not applicable to all methods

    def unshare_parameters(self, name: str):
        pass  # default implementation does nothing as multi task is not applicable to all methods

    def add_fusion_layer(self, adapter_names: Union[List, str]):
        pass  # default implementation does nothing as fusion is not applicable to all methods

    def delete_fusion_layer(self, adapter_names: Union[List, str]):
        pass  # default implementation does nothing as fusion is not applicable to all methods

    def enable_adapters(
        self,
        adapter_setup: AdapterCompositionBlock,
        unfreeze_adapters: bool,
        unfreeze_fusion: bool,
    ):
        """Enables/ disables a set of adapter modules within the layer.

        Args:
            adapter_setup (AdapterCompositionBlock): The adapter setup to enable/ disable.
            unfreeze_adapters (bool): Whether to unfreeze the adapters.
        """
        if unfreeze_adapters:
            for name in adapter_setup.flatten():
                if name in self.adapter_modules:
                    for param in self.adapter_modules[name].parameters():
                        param.requires_grad = True

    def freeze_adapter(self, adapter_name: str, freeze: bool = True):
        """Freezes/ unfreezes an adapter module.

        Args:
            adapter_name (str): The name of the adapter to freeze/ unfreeze.
            freeze (bool, optional): Whether to freeze the adapter. Defaults to True.
        """
        if adapter_name in self.adapter_modules:
            self.adapter_modules[adapter_name].train(not freeze)
            for param in self.adapter_modules[adapter_name].parameters():
                param.requires_grad = not freeze

    def get_adapter(self, adapter_name: str) -> nn.Module:
        """Returns the adapter module with the given name.

        Args:
            adapter_name (str): The name of the adapter module.
        """
        if adapter_name in self.adapter_modules:
            return self.adapter_modules[adapter_name]
        else:
            return None

    def pre_save_adapters(self):
        """Called before saving the adapters to disk."""
        pass


class ModelAdaptersConfig(Collection):
    """This class manages the setup and configuration of adapter modules in a pre-trained model."""

    def __init__(self, **kwargs):
        adapters_list = kwargs.pop("adapters", {})
        # this is for backwards compability: in v1.x, self.adapters values had shape (<type>, <config_name>)
        adapters_list = dict(
            map(
                lambda t: (
                    t[0],
                    t[1][1] or t[1][0] if isinstance(t[1], tuple) else t[1],
                ),
                adapters_list.items(),
            )
        )
        self.adapters: Mapping[str, str] = adapters_list
        self.config_map = kwargs.pop("config_map", {})

        self.fusions: Mapping[str, str] = kwargs.pop("fusions", {})
        self.fusion_config_map = kwargs.pop("fusion_config_map", {})
        self.fusion_name_map = kwargs.pop("fusion_name_map", {})

        # TODO-V2 Save this with config?
        self.active_setup: Optional[AdapterCompositionBlock] = None
        self.skip_layers = None

        self._vera_init_shapes = {}

    def __contains__(self, item):
        return item in self.adapters.keys()

    def __iter__(self):
        return iter(self.adapters)

    def __len__(self):
        return len(self.adapters)

    def get(self, adapter_name: str) -> Optional[dict]:
        """
        Gets the config dictionary for a given adapter.

        Args:
            adapter_name (str): The name of the adapter.

        Returns:
            Mapping: The adapter configuration.
        """
        if adapter_name in self.adapters:
            config_name = self.adapters[adapter_name]
            if config_name in self.config_map:
                config = self.config_map.get(config_name, None)
            else:
                config = ADAPTER_CONFIG_MAP.get(config_name, None)
            if isinstance(config, str):
                config = ADAPTER_CONFIG_MAP[config]
        else:
            config = None
        return config

    def match(
        self,
        adapter_name: str,
        config_type: type,
        layer_idx: Optional[int] = None,
        location_key: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Tries to match the given criteria to an existing adapter. Return the adapter config if a match is found,
        otherwise None.
        """
        config = self.get(adapter_name)
        if config is None:
            return None
        elif not isinstance(config, AdapterConfig):
            config = AdapterConfig.load(config)

        if isinstance(config, config_type):
            leave_out = config.get("leave_out", [])
            if layer_idx is None or layer_idx not in leave_out:
                if location_key is None or config.get(location_key, False):
                    return config
        # if we have a config union, match with all child configs
        elif isinstance(config, ConfigUnion):
            results = []
            for c in config.configs:
                if isinstance(c, config_type):
                    leave_out = c.get("leave_out", [])
                    if layer_idx is None or layer_idx not in leave_out:
                        if location_key is None or c.get(location_key, False):
                            results.append(c)
            if len(results) == 1:
                return results[0]
            elif len(results) > 1:
                raise ValueError(
                    "Multiple adapter definitions conflict for adapter '{}' in layer {}. "
                    "Please make sure there is only one adaptation block used per location and adapter.".format(
                        adapter_name, layer_idx
                    )
                )

        return None

    def add(self, adapter_name: str, config: Optional[Union[str, dict]] = None):
        """
        Adds a new adapter of the name to the model config.

        Args:
            adapter_name (str): The name of the adapter.
            config (Optional[Union[str, dict]], optional): The adapter config. Defaults to None.
        """
        if adapter_name in self.adapters:
            raise ValueError(
                f"An adapter with the name '{adapter_name}' has already been added."
            )
        if config is None:
            config = DEFAULT_ADAPTER_CONFIG
        if isinstance(config, str):
            if config not in ADAPTER_CONFIG_MAP and config not in self.config_map:
                raise ValueError(f"Invalid adapter config identifier '{config}'.")
            config_name = config
        # if it's a dict, compute it's hash and add a new entry to the config map
        elif isinstance(config, Mapping):
            config_name = get_adapter_config_hash(config)
            self.config_map[config_name] = AdapterConfig.load(config)
        else:
            raise ValueError("Invalid adapter config: {}".format(config))
        self.adapters[adapter_name] = config_name
        logger.info(f"Adding adapter '{adapter_name}'.")

    def get_fusion(
        self, fusion_name: Union[str, List[str]]
    ) -> Tuple[Optional[dict], Optional[list]]:
        """
        Gets the config dictionary for a given AdapterFusion.

        Args:
            fusion_name (Union[str, List[str]]): The name of the AdapterFusion or the adapters to fuse.

        Returns:
            Optional[dict]: The AdapterFusion configuration.
            Optional[list]: The names of the adapters to fuse.
        """
        if isinstance(fusion_name, list):
            fusion_name = ",".join(fusion_name)
        if fusion_name in self.fusions:
            config_name = self.fusions[fusion_name]
            if config_name in self.fusion_config_map:
                config = self.fusion_config_map.get(config_name, None)
            else:
                config = ADAPTERFUSION_CONFIG_MAP.get(config_name, None)

            if fusion_name in self.fusion_name_map:
                adapter_names = self.fusion_name_map[fusion_name]
            else:
                adapter_names = fusion_name.split(",")

            return config, adapter_names
        else:
            return None, None

    def add_fusion(
        self,
        adapter_names: List[str],
        config: Optional[Union[str, dict]] = None,
        fusion_name: Optional[str] = None,
    ):
        """
        Adds a new AdapterFusion.

        Args:
            adapter_names (List[str]): The names of the adapters to fuse.
            config (Optional[Union[str, dict]], optional): AdapterFusion config. Defaults to None.
            fusion_name (Optional[str], optional): The name of the AdapterFusion. If not specified, will default to comma-separated adapter names.
        """
        if fusion_name is None:
            fusion_name = ",".join(adapter_names)
        else:
            self.fusion_name_map[fusion_name] = adapter_names
        if fusion_name in self.fusions:
            raise ValueError(
                f"An AdapterFusion with the name '{fusion_name}' has already been added."
            )
        if config is None:
            config = DEFAULT_ADAPTERFUSION_CONFIG
        if isinstance(config, str):
            if (
                config not in ADAPTERFUSION_CONFIG_MAP
                and config not in self.fusion_config_map
            ):
                raise ValueError(f"Invalid AdapterFusion config identifier '{config}'.")
            config_name = config
        # if it's a dict, compute it's hash and add a new entry to the config map
        elif isinstance(config, Mapping):
            config_name = get_adapter_config_hash(config)
            self.fusion_config_map[config_name] = config
        else:
            raise ValueError("Invalid AdapterFusion config: {}".format(config))
        self.fusions[fusion_name] = config_name
        logger.info(f"Adding AdapterFusion '{fusion_name}'.")

    def common_config_value(self, adapter_names: list, attribute: str):
        """
        Checks whether all adapters in a list share the same config setting for a given attribute and returns the
        shared value.

        Args:
            adapter_names (list): The adapters to check.
            attribute (str): The config attribute to check.
        """
        common_value = None
        for i, name in enumerate(adapter_names):
            config = self.get(name)
            if not config:
                raise ValueError(
                    f"No adapter with name '{name}' found. Make sure that an adapter with this name is loaded."
                )
            config_value = config.get(attribute, None)
            if i > 0 and config_value != common_value:
                raise ValueError(
                    f"All given adapters must define the same value for config attribute {attribute}."
                )
            common_value = config_value
        return common_value

    def to_dict(self):
        output_dict = {}
        output_dict["adapters"] = copy.deepcopy(self.adapters)
        output_dict["config_map"] = {}
        for k, v in self.config_map.items():
            if isinstance(v, AdapterConfig):
                output_dict["config_map"][k] = v.to_dict()
            else:
                output_dict["config_map"][k] = copy.deepcopy(v)
        output_dict["fusions"] = copy.deepcopy(self.fusions)
        output_dict["fusion_config_map"] = {}
        for k, v in self.fusion_config_map.items():
            if isinstance(v, AdapterConfig):
                output_dict["fusion_config_map"][k] = v.to_dict()
            else:
                output_dict["fusion_config_map"][k] = copy.deepcopy(v)
        output_dict["fusion_name_map"] = copy.deepcopy(self.fusion_name_map)
        return output_dict

    def __eq__(self, other):
        return isinstance(other, ModelAdaptersConfig) and (
            self.__dict__ == other.__dict__
        )


class LoRALayer(AdapterLayerBase):
    adapter_modules_name = "loras"

    def __init__(
        self,
        location_key: str,
        model_config: PretrainedConfig,
        adapters_config: ModelAdaptersConfig,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.shared_parameters = nn.ModuleDict(dict())
        self.location_key = location_key + "_lora"
        self.model_config = model_config
        self.adapters_config = adapters_config
        self.loras = nn.ModuleDict(dict())

        self.merged = False

    def get_n_heads(self, lora: Union[LoRA, IA3, LoRAConfig]):
        return 1

    def _check_lora_location(self, config: LoRAConfig):
        return True

    def _get_lora_shapes(self, config: LoRAConfig):
        raise NotImplementedError()

    def add_adapter(self, adapter_name: str, layer_idx: int) -> bool:
        self.layer_idx = layer_idx

        lora_config = self.adapters_config.match(
            adapter_name,
            config_type=LoRAConfig,
            layer_idx=self.layer_idx,
            location_key=self.location_key,
        )
        if lora_config is not None and self._check_lora_location(lora_config):
            lora_args = {}
            if lora_config.composition_mode == "add":
                if isinstance(lora_config.vera_d, float) or isinstance(
                    lora_config.vera_b, float
                ):
                    lora_cls = Vera
                    if adapter_name not in self.adapters_config._vera_init_shapes:
                        lora_A_shape, lora_B_shape = self._get_lora_shapes(lora_config)
                        self.adapters_config._vera_init_shapes[adapter_name] = {
                            "lora_A_shape": lora_A_shape,
                            "lora_B_shape": lora_B_shape,
                        }
                else:
                    lora_cls = LoRA
                lora_cls = (
                    lora_cls if not isinstance(lora_config, MTLLoRAConfig) else MTLLoRA
                )
            elif lora_config.composition_mode == "scale":
                lora_cls = IA3
            else:
                raise ValueError(
                    f"Unknown composition_mode: {lora_config.composition_mode}"
                )

            lora = lora_cls(
                *self._get_lora_shapes(lora_config),
                config=lora_config,
                gating_heads=self.get_n_heads(lora_config),
                name=adapter_name,
                shared_parameters=self.shared_parameters,
                **lora_args,
            )

            lora.train(self.training)
            lora = lora.to(self.weight.device)
            self.loras[adapter_name] = lora
            return True

        return False

    def share_parameters(
        self,
        name: str,
        adapter_names: List,
        reference_adapter_name: Optional[str] = None,
    ):
        if all(name in self.loras for name in adapter_names):
            shared_params = self.loras[reference_adapter_name].get_parameters()
            self.shared_parameters[name] = shared_params
            for adapter_name in adapter_names:
                self.loras[adapter_name].set_parameters(shared_params)

    def unshare_parameters(self, name: str):
        if name in self.shared_parameters:
            del self.shared_parameters[name]

    def delete_adapter(self, adapter_name: str):
        if adapter_name in self.shared_parameters:
            del self.shared_parameters[adapter_name]
        else:
            super().delete_adapter(adapter_name)

    def average_adapter(
        self,
        adapter_name: str,
        input_adapters: Dict[str, float],
        combine_strategy: str,
        svd_rank: int = None,
        **kwargs,
    ) -> bool:
        # add new adapter
        if self.add_adapter(adapter_name, self.layer_idx, **kwargs):
            avg_state_dict = {}

            # First, check if all input adapters are present
            for name in input_adapters.keys():
                if name not in self.loras:
                    self.delete_adapter(adapter_name)  # clean up before raising error
                    raise ValueError("Adapter {} not found.".format(name))

            # VeRA only supports linear averaging.
            if isinstance(self.loras[list(input_adapters.keys())[0]], Vera):
                if combine_strategy != "linear":
                    raise ValueError(
                        "VeRA only supports linear averaging. The combine_strategy must be 'linear'. See https://docs.adapterhub.ml/merging_adapters.html for more information."
                    )

            # Now, combine the weights according to the strategy
            if combine_strategy == "linear":
                for name, weight in input_adapters.items():
                    module = self.loras[name]
                    for k, v in module.state_dict().items():
                        if k in avg_state_dict:
                            avg_state_dict[k] += weight * v
                        else:
                            avg_state_dict[k] = weight * v

            elif combine_strategy == "lora_linear_only_negate_b":
                # Same as linear but for negative weights only negate the B matrix and leave A positive
                # See Zhang et al. (2023) https://proceedings.neurips.cc/paper_files/paper/2023/hash/299a08ee712d4752c890938da99a77c6-Abstract-Conference.html
                for name, weight in input_adapters.items():
                    module = self.loras[name]
                    for k, v in module.state_dict().items():
                        if "lora_B" in k:
                            zhang_weight = weight
                        elif "lora_A" in k:
                            zhang_weight = abs(weight)
                        else:
                            # This should never happen as we only have lora_A and lora_B in the state_dict
                            raise ValueError(
                                f"Key must either contain 'lora_A' or 'lora_B' but is {k}. This should never"
                                " happen. Please open an issue on GitHub if you encounter this error."
                            )

                        if k in avg_state_dict:
                            avg_state_dict[k] += zhang_weight * v
                        else:
                            avg_state_dict[k] = zhang_weight * v

            elif combine_strategy == "lora_delta_w_svd":
                # Weight the delta_w matrices by the input weights and then use Singular Value Decomposition (SVD) to split them into A and B matrices.
                self._average_adapter_lora_delta_w_svd(
                    input_adapters, avg_state_dict, svd_rank
                )

            else:
                raise ValueError(
                    f"The combine_strategy '{combine_strategy}' is not supported for LoRA."
                )

            # load averaged weights
            self.loras[adapter_name].load_state_dict(avg_state_dict)
            return True

        return False

    def get_adapter(self, adapter_name: str) -> nn.Module:
        """Returns the adapter module with the given name.

        Args:
            adapter_name (str): The name of the adapter module.
        """
        if adapter_name in self.adapter_modules:
            return self.adapter_modules[adapter_name]
        elif adapter_name in self.shared_parameters:
            return self.shared_parameters[adapter_name]
        else:
            return None

    def _average_adapter_lora_delta_w_svd(
        self, input_adapters: Dict[str, float], avg_state_dict, svd_rank
    ):
        # Weight the delta_w matrices by the input weights and then use Singular Value Decomposition to split them into A and B matrices.
        if svd_rank is None:
            raise ValueError("svd_rank must be set when using 'lora_delta_w_svd'.")

        # Collect delta_w matrices. Shape of every delta_w matrix in the list: d×k
        delta_w = [
            self.loras[adapter_name].delta_w for adapter_name in input_adapters.keys()
        ]

        # If the lora has fan_in_fan_out, we need to transpose the matrices
        if self.fan_in_fan_out:
            delta_w = [torch.t(delta_w) for delta_w in delta_w]

        delta_w = torch.stack(delta_w, dim=0)  # shape: n×d×k

        # Weight the delta_w matrices
        weights = torch.tensor(
            list(input_adapters.values()), device=delta_w.device
        )  # shape: n
        weights = weights.view(-1, 1, 1)  # shape: n×1×1
        delta_w = delta_w * weights  # shape: n×d×k

        # Now bring down to d×k matrix
        delta_w = delta_w.sum(dim=0)  # shape: d×k

        # Perform SVD to split delta_w into A and B matrices
        U, S_diag, V = torch.linalg.svd(delta_w)

        # Reduce rank
        U = U[:, :svd_rank]  # U is 2D
        S_diag = S_diag[:svd_rank]  # S_diag is 1D
        V = V[:svd_rank, :]  # V is 2D

        # The SVD has decomposed delta_w into U, S, and V such that: delta_w = U @ S_diag @ V
        # In LoRA we have: delta_w = B @ A
        # Hence, we can set: A = V and B = U @ S_diag
        if self.fan_in_fan_out:
            avg_state_dict["lora_A"] = torch.t(V)
            avg_state_dict["lora_B"] = torch.t(U @ torch.diag(S_diag))
        else:
            avg_state_dict["lora_A"] = V
            avg_state_dict["lora_B"] = U @ torch.diag(S_diag)

    def _copy_hooks_from(self, module: nn.Module):
        for (
            k,
            v,
        ) in module.__dict__.items():
            if "_hooks" in k:
                setattr(self, k, v)


class LoRAState(NamedTuple):
    """Models the input and output states of a LoRA layer.

    Args:
        layer_input (torch.Tensor): The input states to the adapted layer.
        hidden_states (Optional[torch.Tensor]):
            The hidden states of the adaptation module. These can be None before passing through the first LoRA/ IA3
            module.
        layer_output (torch.Tensor): The output states of the original layer without adaptation.
        last (str, optional): Name of the last adapter applied in the composition.
    """

    layer_input: torch.Tensor
    hidden_states: Optional[torch.Tensor]
    layer_output: torch.Tensor
    last: Optional[str]


class ComposableAdapterLayerBase(AdapterLayerBase):
    """
    Base class for all adapter methods that support composition.

    Make sure the 'adapter_modules_name' and 'supported_compositions' attributes as well as all abstract methods are
    overriden in derived classes. 'allow_multi_parallelize' can be set to True to allow inputs to be parallelized
    independently multiple times. This is useful when there are multiple parallel input flows through an adapter layer
    (e.g. in LoRA).
    """

    supported_compositions = []
    allow_multi_parallelize = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._init_mapping()

    def _init_mapping(self):
        # Mapping between composition block types and names of composition functions
        self.composition_to_func_map = {
            Stack: "compose_stack",
            Fuse: "compose_fuse",
            Split: "compose_split",
            MultiTask: "compose_multi_task",
            BatchSplit: "compose_batch_split",
            Parallel: "compose_parallel",
            Average: "compose_average",
        }

    def _get_compose_func(self, composition_type: type) -> callable:
        """Retrieves the correct composition function based on the mapping in 'composition_to_func_map'."""
        return getattr(self, self.composition_to_func_map[composition_type])

    # START CUSTOMIZABLE METHODS #
    # The following methods should be implemented in derived classes.

    def _bsz(self, state: NamedTuple) -> int:
        """
        Returns the batch size of the given state.
        """
        return state[0].shape[0]

    def pre_block(
        self,
        adapter_setup: Union[AdapterCompositionBlock, str],
        state: NamedTuple,
    ) -> NamedTuple:
        """
        Optional state pre-processing method which is invoked before passing the state to the first child block of a
        composition. By default, this method does not contain any logic. E.g. used for bottleneck adapters to implement
        residuals and LNs.

        Args:
            adapter_setup (Union[AdapterCompositionBlock, str]): The current composition or single adapter.
            state (NamedTuple): The current state.

        Returns:
            NamedTuple: The pre-processed state.
        """
        return state

    def check_composition_valid(
        self,
        parent: AdapterCompositionBlock,
        child: AdapterCompositionBlock,
        lvl: int,
    ):
        """Checks whether the given composition is valid.

        Args:
            parent (AdapterCompositionBlock): The parent composition block.
            child (AdapterCompositionBlock): The child composition block.
            lvl (int): The composition depth.

        Raises:
            ValueError: If the composition is invalid.
        """
        # Break if setup is too deep
        if isinstance(parent, Stack) and lvl >= 1:
            raise ValueError(
                "Specified adapter setup is too deep. Cannot have {} at level {}".format(
                    child.__class__.__name__, lvl
                )
            )
        elif type(child) not in ALLOWED_NESTINGS[type(parent)]:
            raise ValueError(
                "Cannot nest {} inside {}. Only the following nestings are allowed: {}".format(
                    child.__class__.__name__,
                    parent.__class__.__name__,
                    ", ".join([t.__name__ for t in ALLOWED_NESTINGS[type(parent)]]),
                )
            )

    def compose_stack(
        self, adapter_setup: Stack, state: NamedTuple, lvl: int = 0
    ) -> NamedTuple:
        """
        For sequentially stacking multiple adapters.
        """
        for i, adapter_stack_layer in enumerate(adapter_setup):
            if isinstance(adapter_stack_layer, AdapterCompositionBlock):
                self.check_composition_valid(adapter_setup, adapter_stack_layer, lvl)
                composition_func = self._get_compose_func(type(adapter_stack_layer))
                state = composition_func(adapter_stack_layer, state, lvl=lvl + 1)
            elif adapter_stack_layer in self.adapter_modules:
                state = self.pre_block(adapter_stack_layer, state)
                state = self.compose_single(adapter_stack_layer, state, lvl=lvl + 1)
            else:
                pass

        return state

    def compose_fuse(self, adapter_setup: Fuse, state: NamedTuple, lvl: int = 0):
        """
        For fusing multiple adapters using adapter fusion. NOTE: This method has no default implementation.
        """
        # Fuse is currently only applicable to bottleneck adapters, thus don't provide a default implementation
        # If the adapter setup does not contain any of the adapter modules, return without doing anything
        if set(self.adapter_modules.keys()).isdisjoint(adapter_setup.flatten()):
            return state
        raise NotImplementedError()

    def compose_split(self, adapter_setup: Split, state: NamedTuple, lvl: int = 0):
        """
        For splitting to multiple adapters along the sequence length dimension. NOTE: This method has no default
        implementation.
        """
        # Split is currently only applicable to bottleneck adapters, thus don't provide a default implementation
        # If the adapter setup does not contain any of the adapter modules, return without doing anything
        if set(self.adapter_modules.keys()).isdisjoint(adapter_setup.flatten()):
            return state
        raise NotImplementedError()

    def compose_batch_split(
        self, adapter_setup: BatchSplit, state: NamedTuple, lvl: int = 0
    ):
        """
        For splitting to multiple adapters along the batch size dimension.
        """
        if sum(adapter_setup.batch_sizes) != self._bsz(state):
            raise IndexError(
                "The given batch has a size of {} which is not equal to the sum of batch_sizes {}".format(
                    self._bsz(state), adapter_setup.batch_sizes
                )
            )

        state = self.pre_block(adapter_setup, state)

        # sequentially feed different parts of the blown-up batch into different adapters
        children_states = []
        for i, child in enumerate(adapter_setup):
            # compute ids of sequences that should be passed to the ith adapter
            batch_idx = (
                sum(adapter_setup.batch_sizes[:i]),
                sum(adapter_setup.batch_sizes[: i + 1]),
            )
            if isinstance(child, AdapterCompositionBlock):
                self.check_composition_valid(adapter_setup, child, lvl)
                composition_func = self._get_compose_func(type(child))
                child_state = composition_func(
                    child,
                    self.vslice(state, slice(*batch_idx)),
                    lvl=lvl + 1,
                )
                children_states.append(child_state)
            elif child in self.adapter_modules:
                child_state = self.compose_single(
                    child,
                    self.vslice(state, slice(*batch_idx)),
                    lvl=lvl + 1,
                )
                children_states.append(child_state)
            else:
                children_states.append(self.vslice(state, slice(*batch_idx)))

        # concatenate all outputs and return
        state = self.pad_and_concat(children_states)
        return state

    def compose_multi_task(
        self, adapter_setup: MultiTask, state: NamedTuple, lvl: int = 0
    ):
        """
        For splitting to multiple adapters along the task_ids.
        """
        state = self.pre_block(adapter_setup, state)

        # sequentially feed different parts of the blown-up batch into different adapters
        context = ForwardContext.get_context()
        assert hasattr(context, "task_ids")
        task_ids = context.task_ids
        assert task_ids is not None
        if isinstance(task_ids, list) and isinstance(task_ids[0], str):
            children = adapter_setup.children
            task_ids = torch.tensor([children.index(task) for task in task_ids])
        ordering_idx = task_ids.argsort()
        batch_sizes = task_ids.bincount().tolist()
        inter_state = self.compose_batch_split(
            adapter_setup=BatchSplit(*adapter_setup.children, batch_sizes=batch_sizes),
            state=self.vslice(state, ordering_idx),
            lvl=lvl,
        )
        final_state = self.vslice(inter_state, ordering_idx.argsort())
        return final_state

    def compose_parallel(
        self, adapter_setup: Parallel, state: NamedTuple, lvl: int = 0
    ):
        """
        For parallel execution of the adapters on the same input. This means that the input is repeated N times before
        feeding it to the adapters (where N is the number of adapters).
        """

        context = ForwardContext.get_context()
        if not context.adapters_parallelized:
            orig_batch_size = self._bsz(state)
            state = self.repeat(state, adapter_setup.parallel_channels)
            context.adapters_parallelized = True
            context.original_batch_size = orig_batch_size
        else:
            bsz = self._bsz(state)
            # If the input was already parallelized, we can parallelize it again.
            # This is useful e.g. for LoRA, where attention matrices are parallelized independently.
            if self.allow_multi_parallelize and bsz == getattr(
                context, "original_batch_size", -1
            ):
                state = self.repeat(state, adapter_setup.parallel_channels)
                orig_batch_size = bsz
            # The base model should handle replication of input.
            # Therefore, we assume the (replicated) input batch to be divisible by the number of parallel channels.
            elif bsz % adapter_setup.parallel_channels != 0:
                raise ValueError(
                    "The total input batch size in a Parallel adapter block must be divisible by the number of"
                    " parallel channels."
                )
            else:
                orig_batch_size = bsz // adapter_setup.parallel_channels

        state = self.pre_block(adapter_setup, state)

        # sequentially feed different parts of the blown-up batch into different adapters
        children_states = []
        for i, child in enumerate(adapter_setup):
            if isinstance(child, AdapterCompositionBlock):
                self.check_composition_valid(adapter_setup, child, lvl)
                composition_func = self._get_compose_func(type(child))
                child_state = composition_func(
                    child,
                    self.vslice(
                        state,
                        slice(i * orig_batch_size, (i + 1) * orig_batch_size),
                    ),
                    lvl=lvl + 1,
                )
                children_states.append(child_state)
            elif child in self.adapter_modules:
                child_state = self.compose_single(
                    child,
                    self.vslice(
                        state,
                        slice(i * orig_batch_size, (i + 1) * orig_batch_size),
                    ),
                    lvl=lvl + 1,
                )
                children_states.append(child_state)
            else:
                children_states.append(
                    self.vslice(
                        state,
                        slice(i * orig_batch_size, (i + 1) * orig_batch_size),
                    )
                )

        # concatenate all outputs and return
        state = self.pad_and_concat(children_states)
        return state

    def compose_average(self, adapter_setup: Average, state: NamedTuple, lvl: int = 0):
        """
        For averaging the output representations of multiple adapters.
        """

        state = self.pre_block(adapter_setup, state)

        children_states = []
        for i, child in enumerate(adapter_setup):
            if isinstance(child, AdapterCompositionBlock):
                self.check_composition_valid(adapter_setup, child, lvl)
                composition_func = self._get_compose_func(type(child))
                child_state = composition_func(child, state, lvl=lvl + 1)
                children_states.append(child_state)
            elif child in self.adapter_modules:
                child_state = self.compose_single(child, state, lvl=lvl + 1)
                children_states.append(child_state)
            else:
                pass

        weights = torch.tensor(adapter_setup.weights)[:, None, None, None].to(
            state[0].device
        )
        state = self.mean(children_states, weights)

        return state

    def compose(
        self,
        adapter_setup: Union[AdapterCompositionBlock, str],
        state: NamedTuple,
    ) -> NamedTuple:
        """The main composition forward method which recursively calls the composition blocks forward methods.
        This method should be called by the forward method of the derived class.

        Args:
            adapter_setup (Union[AdapterCompositionBlock, str]): The adapter setup to be used.
            state (NamedTuple): The current state.

        Returns:
            NamedTuple: The state after forwarding through the adapter setup.
        """
        if isinstance(adapter_setup, AdapterCompositionBlock):
            composition_func = self._get_compose_func(type(adapter_setup))
            state = composition_func(adapter_setup, state, lvl=0)
        elif adapter_setup in self.adapter_modules:
            state = self.compose_single(adapter_setup, state, lvl=0)
        else:
            raise ValueError(
                "Invalid adapter setup: {} is not a valid adapter name or composition block.".format(
                    adapter_setup.__class__.__name__
                )
            )

        return state


class LoRALinear(LoRALayer, ComposableAdapterLayerBase):
    """
    LoRA implementation for Linear layer. This layer supports composition.

    Args:
        fan_in_fan_out (bool, optional):
            Set this to True if the layer to replace stores weight like (fan_in, fan_out). Defaults to False.
        no_init_bias (bool, optional): Use this to add a bias that is not initialized by PyTorch. Defaults to False.

    """

    supported_compositions = [
        Stack,
        BatchSplit,
        Average,
        Parallel,
        MultiTask,
    ]
    allow_multi_parallelize = True

    def __init__(
        self,
        in_features: int,
        out_features: int,
        location_key: str,
        model_config: PretrainedConfig,
        adapters_config: ModelAdaptersConfig,
        attn_key: str = None,
        fan_in_fan_out: bool = False,
        no_init_bias: bool = False,
        **kwargs,
    ):
        if no_init_bias and "bias" not in kwargs:
            kwargs["bias"] = False
        LoRALayer.__init__(
            self,
            location_key,
            model_config,
            adapters_config,
            in_features,
            out_features,
            **kwargs,
        )

        self.attn_key = attn_key
        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = torch.t(self.weight.data)
        if no_init_bias:
            self.bias = nn.Parameter(torch.empty(out_features))

    @classmethod
    def wrap(
        cls,
        module: Union[nn.Linear, Conv1D],
        location_key: str,
        model_config: PretrainedConfig,
        adapters_config: ModelAdaptersConfig,
        attn_key: str = None,
        **kwargs,
    ):
        if isinstance(module, Conv1D):
            new_module = LoRALinearTorch(
                module.weight.shape[0],
                module.weight.shape[1],
                location_key,
                model_config,
                adapters_config,
                attn_key=attn_key,
                **kwargs,
            )
        else:
            if bitsandbytes_available and isinstance(module, Linear4bit):
                cls = LoRALinear4bit
            elif bitsandbytes_available and isinstance(module, Linear8bitLt):
                cls = LoRALinear8bitLt
            else:
                cls = LoRALinearTorch
            # Make sure that the bias is not added if the original module does not have one
            if "bias" not in kwargs:
                kwargs["bias"] = hasattr(module, "bias") and module.bias is not None
            new_module = cls(
                module.in_features,
                module.out_features,
                location_key,
                model_config,
                adapters_config,
                attn_key=attn_key,
                **kwargs,
            )
        new_module.copy_from(module)
        new_module._copy_hooks_from(module)

        return new_module

    def copy_from(self, module: nn.Linear):
        self.weight = module.weight
        if module.bias is not None:
            self.bias = module.bias

    def _check_lora_location(self, config: LoRAConfig):
        return self.attn_key is None or self.attn_key in config.attn_matrices

    def _get_lora_shapes(self, config: LoRAConfig):
        return (config.r, self.in_features), (self.out_features, config.r)

    def maybe_t(self, w):
        return torch.t(w) if self.fan_in_fan_out else w

    def merge_adapter(self, name: str):
        if name in self.loras:
            if self.merged == name:
                return  # already merged
            elif not self.merged:
                lora = self.loras[name]
                if lora.use_gating:
                    raise ValueError("Cannot merge LoRA layer with gating.")
                delta_w = self.maybe_t(lora.delta_w)
                self.weight.data = lora.com(self.weight.data, delta_w)
                self.merged = name
            elif self.merged != name:
                raise ValueError(
                    "LoRALayer already has a merged LoRA module. Please reset it first."
                )

    def reset_adapter(self):
        if self.merged:
            lora = self.loras[self.merged]
            # Make sure that the weights are not merged
            delta_w = self.maybe_t(lora.delta_w)
            self.weight.data = lora.com_inv(self.weight.data, delta_w)
            self.merged = None

    def vslice(self, state: LoRAState, slice_obj: slice) -> LoRAState:
        return LoRAState(
            state.layer_input[slice_obj],
            (
                state.hidden_states[slice_obj]
                if state.hidden_states is not None
                else None
            ),
            state.layer_output[slice_obj],
            state.last,
        )

    def pad_and_concat(self, states: List[LoRAState]) -> LoRAState:
        return LoRAState(
            torch.cat([s.layer_input for s in states], dim=0),
            (
                torch.cat([s.hidden_states for s in states], dim=0)
                if states[0].hidden_states is not None
                else None
            ),
            torch.cat([s.layer_output for s in states], dim=0),
            states[-1].last,
        )

    def repeat(self, state: LoRAState, channels: int) -> LoRAState:
        return LoRAState(
            state.layer_input.repeat(channels, 1, 1),
            (
                state.hidden_states.repeat(channels, 1, 1)
                if state.hidden_states is not None
                else None
            ),
            state.layer_output.repeat(channels, 1, 1),
            state.last,
        )

    def mean(self, states: List[LoRAState], weights: torch.Tensor) -> LoRAState:
        return LoRAState(
            states[0].layer_input,
            (
                torch.mean(
                    torch.stack([s.hidden_states for s in states], dim=0) * weights,
                    dim=0,
                )
                if states[0].hidden_states is not None
                else None
            ),
            states[0].layer_output,
            states[-1].last,
        )

    def compose_single(
        self, adapter_setup: str, state: LoRAState, lvl: int = 0
    ) -> LoRAState:
        lora = self.loras[adapter_setup]
        hidden_states, gate = lora(state.hidden_states, state.layer_input)
        if gate is not None:
            self._store_gating_score(adapter_setup, gate)

        return state._replace(hidden_states=hidden_states, last=adapter_setup)

    def forward(self, input_states: torch.Tensor):
        if self.fan_in_fan_out:
            weight = (
                torch.transpose(self.weight, -2, -1)
                if self.fan_in_fan_out
                else self.weight
            )
            # result shape: <batch_size> x <seq_len> x <head_dim>
            layer_output = F.linear(input_states, weight, bias=self.bias)
        else:
            layer_output = super().forward(input_states)

        if not self.merged:
            adapter_setup = self.get_active_setup()
            if adapter_setup is not None:
                state = LoRAState(input_states, None, layer_output, None)
                state = self.compose(adapter_setup, state)
                _, hidden_states, layer_output, last = state

                last_lora = self.loras[last]
                layer_output = last_lora.com(
                    layer_output, hidden_states, scaling=1.0
                )  # scaling already applied in compose

        return layer_output


class LoRALinearTorch(LoRALinear, nn.Linear):
    pass


@dataclass(eq=False)
class AdapterFusionConfig(AdapterConfig):
    """Base class that models the architecture of an adapter fusion layer."""

    key: bool
    query: bool
    value: bool
    query_before_ln: bool
    regularization: bool
    residual_before: bool
    temperature: bool
    value_before_softmax: bool
    value_initialized: str
    dropout_prob: float

    @classmethod
    def load(cls, config: Union[dict, str], **kwargs):
        """
        Loads a given adapter fusion configuration specifier into a full AdapterFusionConfig instance.

        Args:
            config (Union[dict, str]): The configuration to load. Can be either:

                - a dictionary representing the full config
                - an identifier string available in ADAPTERFUSION_CONFIG_MAP
                - the path to a file containing a full adapter fusion configuration

        Returns:
            dict: The resolved adapter fusion configuration dictionary.
        """
        # currently storing AdapterFusion weights on AdapterHub is not supported.
        config_dict = resolve_adapter_config(config, local_map=ADAPTERFUSION_CONFIG_MAP)
        # convert back to dict to allow attr overrides
        if isinstance(config_dict, AdapterFusionConfig):
            config_dict = config_dict.to_dict()
        config_dict.update(kwargs)
        return AdapterFusionConfig.from_dict(config_dict)


@dataclass(eq=False)
class StaticAdapterFusionConfig(AdapterFusionConfig):
    """
    Static version of adapter fusion without a value matrix. See https://arxiv.org/pdf/2005.00247.pdf.
    """

    key: bool = True
    query: bool = True
    value: bool = False
    query_before_ln: bool = False
    regularization: bool = False
    residual_before: bool = False
    temperature: bool = False
    value_before_softmax: bool = True
    value_initialized: str = False
    dropout_prob: float = None


@dataclass(eq=False)
class DynamicAdapterFusionConfig(AdapterFusionConfig):
    """
    Dynamic version of adapter fusion with a value matrix and regularization. See https://arxiv.org/pdf/2005.00247.pdf.
    """

    key: bool = True
    query: bool = True
    value: bool = True
    query_before_ln: bool = False
    regularization: bool = True
    residual_before: bool = False
    temperature: bool = False
    value_before_softmax: bool = True
    value_initialized: str = True
    dropout_prob: float = None


ADAPTERFUSION_CONFIG_MAP = {
    "static": StaticAdapterFusionConfig(),
    "dynamic": DynamicAdapterFusionConfig(),
}

DEFAULT_ADAPTERFUSION_CONFIG = "dynamic"


class Activation_Function_Class(nn.Module):
    """
    Implementation of various activation function.
    """

    def __init__(self, hidden_act):
        super().__init__()
        if hidden_act is None:
            self.f = nn.Identity()
        elif hidden_act.lower() == "leakyrelu":
            self.f = nn.functional.leaky_relu
        else:
            self.f = get_activation(hidden_act.lower())

    def forward(self, x):
        return self.f(x)


# Single Adapter


class Adapter(nn.Module):
    """
    Implementation of a sequential bottleneck adapter block.
    """

    def __init__(
        self,
        adapter_name,
        input_size,
        down_sample,
        config: BnConfig,
    ):
        super().__init__()
        self.name = adapter_name
        self.input_size = input_size
        self.add_layer_norm_before = config["ln_before"]
        self.add_layer_norm_after = config["ln_after"]
        self.adapter_residual_before_ln = config["adapter_residual_before_ln"]
        self.use_gating = config["use_gating"]

        # Params related to input & output of adapter
        self.residual_before_ln = config["residual_before_ln"]
        self.original_ln_before = config["original_ln_before"]
        self.original_ln_after = config["original_ln_after"]

        # list for all modules of the adapter, passed into nn.Sequential()
        seq_list = []

        # If we want to have a layer norm on input, we add it to seq_list
        if self.add_layer_norm_before:
            self.adapter_norm_before = nn.LayerNorm(self.input_size)
            seq_list.append(self.adapter_norm_before)

        # if a downsample size is not passed, we just half the size of the original input
        self.down_sample = down_sample
        if down_sample is None:
            self.down_sample = self.input_size // 2

        # ensure that the down sample size is at least 1
        if self.down_sample < 1:
            self.down_sample = 1

        if config["phm_layer"]:
            # Linear down projection of the input
            seq_list.append(
                PHMLayer(
                    adapter_name, self.input_size, self.down_sample, "down", config
                )
            )
        else:
            seq_list.append(nn.Linear(self.input_size, self.down_sample))

        # select non-linearity
        self.non_linearity = Activation_Function_Class(config["non_linearity"].lower())

        seq_list.append(self.non_linearity)

        # sequential adapter, first downproject, then non-linearity then upsample. In the forward pass we include the
        # residual connection
        self.adapter_down = nn.Sequential(*seq_list)

        # Up projection to input size
        if config["phm_layer"]:
            # Linear down projection of the input
            self.adapter_up = PHMLayer(
                adapter_name, self.down_sample, self.input_size, "up", config
            )
        else:
            self.adapter_up = nn.Linear(self.down_sample, self.input_size)

        # Additional scaling factor (from He et al. (2021))
        if isinstance(config["scaling"], float):
            self.scaling = config["scaling"]
        elif config["scaling"] == "learned":
            self.scaling = nn.Parameter(torch.ones(1))
        elif config["scaling"] == "channel":
            self.scaling = nn.Parameter(torch.ones(input_size))
        else:
            raise ValueError("Unknown scaling type: {}".format(config["scaling"]))

        # If we want to have a layer norm on output, we apply it later after a separate residual connection
        # This means that we learn a new output layer norm, which replaces another layer norm learned in the bert layer
        if self.add_layer_norm_after:
            self.adapter_norm_after = nn.LayerNorm(self.input_size)

        if self.use_gating:
            self.gate = nn.Linear(self.input_size, 1)

        self.dropout = nn.Dropout(p=config["dropout"])

        # Set seed for reproducibility if specified in config
        fix_seed(config.init_weights_seed)

        # if we want to initialize with the bert strategy then this function is called for all the linear layers
        if config["init_weights"] == "bert":
            self.adapter_down.apply(self.init_bert_weights)
            self.adapter_up.apply(self.init_bert_weights)
            if self.use_gating:
                self.gate.apply(self.init_bert_weights)
        elif config["init_weights"] == "mam_adapter":
            with torch.no_grad():
                for layer in self.adapter_down:
                    if isinstance(layer, nn.Linear) or isinstance(layer, PHMLayer):
                        nn.init.kaiming_uniform_(layer.weight, a=math.sqrt(5))
                        nn.init.zeros_(layer.bias)
                nn.init.zeros_(self.adapter_up.weight)
                nn.init.zeros_(self.adapter_up.bias)
                if self.use_gating:
                    self.gate.apply(self.init_bert_weights)
        elif config["init_weights"] == "houlsby":
            for layer in self.adapter_down:
                if isinstance(layer, nn.Linear) or isinstance(layer, PHMLayer):
                    nn.init.trunc_normal_(
                        layer.weight, mean=0, std=1e-2, a=-2 * 1e-2, b=2 * 1e-2
                    )
                    nn.init.zeros_(layer.bias)

            nn.init.trunc_normal_(
                self.adapter_up.weight, mean=0, std=1e-2, a=-2 * 1e-2, b=2 * 1e-2
            )
            nn.init.zeros_(self.adapter_up.bias)
        else:
            raise ValueError(
                "Unknown init_weights type: {}".format(config["init_weights"])
            )

        if config["stochastic_depth"] > 0.0:
            if is_torchvision_available():
                from torchvision.ops.stochastic_depth import StochasticDepth

                self.DropPath = StochasticDepth(
                    p=config["stochastic_depth"], mode="row"
                )
            else:
                raise ImportError(
                    "stochastic_depth requires the package torchvision, but it is not installed"
                )

    def pre_forward(
        self,
        hidden_states,
        input_tensor,
        layer_norm,
        fusion_config=None,
    ):
        """
        Retrieves the hidden_states, query (for Fusion), and residual connection according to the set configuration.

        Args:
            adapter_config: config file according to what the parameters are passed
            hidden_states: output of previous layer
            input_tensor: residual connection before FFN

        Returns: hidden_states, query, residual

        """
        query = None

        if self.residual_before_ln is True:
            residual = hidden_states

        if fusion_config is not None and fusion_config["query_before_ln"]:
            query = hidden_states

        if self.original_ln_before:
            if layer_norm:
                hidden_states = hidden_states + input_tensor
                if self.residual_before_ln == "post_add":
                    residual = hidden_states
                hidden_states = layer_norm(hidden_states)
            else:
                # Some models like Phi use a parallel architecture where attention and FFN operate
                # independently on the same normalized hidden_states. In this case, the residual connection
                # is applied only once at the end by combining:
                #     output = original_input + attention_output + ffn_output
                #
                # In our standard adapter implementation, we expect input_tensor to contain intermediate
                # residuals between components. However, for parallel architectures like Phi, there are
                # no intermediate residuals, so input_tensor will be None when the adapter is attached
                # to the FFN. Therefore, this additional check is needed to prevent errors.
                if input_tensor is not None:
                    hidden_states = hidden_states + input_tensor

        if not self.residual_before_ln:
            residual = hidden_states

        if fusion_config is not None and not fusion_config["query_before_ln"]:
            query = hidden_states

        return hidden_states, query, residual

    def forward(self, x, residual_input, output_gating=False):
        down = self.adapter_down(x)

        up = self.adapter_up(down)
        if hasattr(self, "DropPath"):
            up = self.DropPath(up)
        up = up * self.scaling
        output = self.dropout(up)

        if self.use_gating:
            # x.shape = (batch_size, seq_len, hidden_size)
            gate = torch.sigmoid(self.gate(x))
            gate = torch.mean(gate, dim=1).unsqueeze(-1)
            output = output * gate

        # apply residual connection before layer norm if configured in this way
        if self.adapter_residual_before_ln:
            output = output + residual_input

        # apply layer norm if available
        if self.add_layer_norm_after:
            output = self.adapter_norm_after(output)

        # if residual should be applied after layer norm, apply it here
        if not self.adapter_residual_before_ln:
            output = output + residual_input

        if self.use_gating and output_gating:
            return output, down, up, gate
        return output, down, up

    def post_forward(
        self, hidden_states, input_hidden_states, input_tensor, layer_norm
    ):
        if self.original_ln_after:
            if layer_norm:
                hidden_states = layer_norm(hidden_states + input_tensor)
            else:
                hidden_states = hidden_states + input_tensor

        return hidden_states

    # This is copied from the BertPreTrainedModel class to make this a self containing class.
    @staticmethod
    def init_bert_weights(module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # std defaults to 0.02, this might need to be changed
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class ParallelAdapter(Adapter):
    """
    Implementation of a parallel bottleneck adapter block.
    """

    def __init__(self, adapter_name, input_size, down_sample, config: BnConfig):
        super().__init__(adapter_name, input_size, down_sample, config)

    def pre_forward(
        self,
        hidden_states,
        input_tensor,
        layer_norm,
        fusion_config=None,
    ):
        """
        Retrieves the hidden_states, query (for Fusion), and residual connection according to the set configuration.

        Args:
            adapter_config: config file according to what the parameters are passed
            hidden_states: output of previous layer
            input_tensor: residual connection before FFN

        Returns: hidden_states, query, residual

        """
        # In case of parallel adapter, return the input tensor as hidden states
        query = None
        if fusion_config is not None:
            query = input_tensor
        return input_tensor, query, input_tensor

    def forward(self, x, residual_input, output_gating=False):
        down = self.adapter_down(x)

        up = self.adapter_up(down)
        up = up * self.scaling

        output = self.dropout(up)

        if self.use_gating:
            # x.shape = (batch_size, seq_len, hidden_size)
            gate = torch.sigmoid(self.gate(x))
            gate = torch.mean(gate, dim=1).unsqueeze(-1)
            output = output * gate

        # apply layer norm if available
        if self.add_layer_norm_after:
            output = self.adapter_norm_after(output)

        if self.use_gating and output_gating:
            return output, down, up, gate
        return output, down, up

    def post_forward(
        self, hidden_states, input_hidden_states, input_tensor, layer_norm
    ):
        hidden_states = hidden_states + input_hidden_states

        if self.original_ln_after:
            if layer_norm:
                hidden_states = layer_norm(hidden_states + input_tensor)
            else:
                hidden_states = hidden_states + input_tensor

        return hidden_states


# Adapter Fusion


class BertFusion(nn.Module):
    """
    Implementation of an AdapterFusion block.
    """

    def __init__(
        self,
        config: AdapterFusionConfig,
        dense_size,
        attention_probs_dropout_prob,
    ):
        super(BertFusion, self).__init__()
        # if config.hidden_size % config.num_attention_heads != 0:
        #     raise ValueError(
        #         "The hidden size (%d) is not a multiple of the number of attention "
        #         "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.config = config

        self.dense_size = dense_size
        self.dropout = nn.Dropout(attention_probs_dropout_prob)

        if (
            not self.config["query"]
            and not self.config["key"]
            and not self.config["value"]
        ):
            self.dense = nn.Linear(self.dense_size, 1)

        if self.config["query"]:
            self.query = nn.Linear(self.dense_size, self.dense_size)
            self.query.apply(Adapter.init_bert_weights)

        if self.config["key"]:
            self.key = nn.Linear(self.dense_size, self.dense_size)
            self.key.apply(Adapter.init_bert_weights)

        if self.config["value"]:
            self.value = nn.Linear(self.dense_size, self.dense_size, bias=False)
            self.value.apply(Adapter.init_bert_weights)
            if self.config["value_initialized"]:
                init_tensor = (
                    torch.zeros(
                        self.dense_size,
                        self.dense_size,
                        device=self.value.weight.device,
                        dtype=self.value.weight.dtype,
                    )
                    + 0.000001
                )
                init_tensor.fill_diagonal_(1.0)
                self.value.weight.data = init_tensor
        if self.config["temperature"]:
            self.T = 50.0
        else:
            self.T = 1.0
        self.reduction = self.T / 1000.0

    def forward(self, query, key, value, residual, output_attentions: bool = False):
        if self.config["residual_before"]:
            value += residual[:, :, None, :].repeat(1, 1, value.size(2), 1)

        if self.config["query"]:
            query_layer = self.query(query)
        else:
            query_layer = query

        if self.config["key"]:
            key_layer = self.key(key)
        else:
            key_layer = key

        if self.config["value"] and self.config["value_before_softmax"]:
            # key/value have dims => batch, toks, number-of-adapters, feats
            value_layer = self.value(value)
        else:
            value_layer = value

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.squeeze(
            torch.matmul(query_layer.unsqueeze(2), key_layer.transpose(-2, -1)), dim=2
        )

        attention_scores = self.dropout(attention_scores)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores / self.T)
        self.T = max(self.T - self.reduction, 1.0)

        context_layer = torch.squeeze(
            torch.matmul(attention_probs.unsqueeze(2), value_layer), dim=2
        )

        if self.config["value"] and not self.config["value_before_softmax"]:
            # key/value have dims => batch, toks, number-of-adapters, feats
            context_layer = self.value(context_layer)
        else:
            context_layer = context_layer

        if not self.config["residual_before"]:
            context_layer += residual

        if output_attentions:
            attention_probs = attention_probs.detach().cpu().numpy()
            return context_layer, attention_probs
        else:
            return context_layer


class ConfigUnion(AdapterConfig):
    """
    Composes multiple adaptation method configurations into one. This class can be used to define complex adaptation
    method setups.
    """

    architecture: Optional[str] = "union"

    configs: List[AdapterConfig]

    def __init__(self, *configs: List[AdapterConfig]):
        self.validate(configs)
        self.configs = configs

    @staticmethod
    def validate(configs):
        # perform single config checks
        for config in configs:
            if not isinstance(config, AdapterConfig):
                raise TypeError(f"{config} is not an instance of AdapterConfig")
            elif isinstance(config, ConfigUnion):
                raise TypeError(
                    f"{config} of type {type(config)} is not supported in a config union."
                )
        # perform pairwise check
        for c_a, c_b in [
            (c_a, c_b)
            for i, c_a in enumerate(configs)
            for j, c_b in enumerate(configs)
            if i > j
        ]:
            if c_a.architecture != c_b.architecture:
                continue
            # if at least one config specifies a leave_out, we cannot make a final decision at this point
            elif c_a.get("leave_out", []) or c_b.get("leave_out", []):
                continue
            elif c_a.architecture is None or c_a.architecture == "bottleneck":
                is_valid = (
                    c_a.mh_adapter != c_b.mh_adapter
                    and c_a.output_adapter != c_b.output_adapter
                )
                if not is_valid:
                    raise ValueError(f"{c_a} and {c_b} cannot be combined.")
                else:
                    continue
            # at this point, we know that the architectures are the same
            raise ValueError(
                f"{c_a} and {c_b} have the same adapter architecture and cannot be combined."
            )

    def __getitem__(self, key):
        if isinstance(key, int):
            return self.configs[key]
        elif hasattr(self, key):
            return getattr(self, key)
        else:
            i, k = key.split(".")
            return self.configs[int(i)][k]

    def __iter__(self):
        for i, c in enumerate(self.configs):
            for k in iter(c):
                yield f"{i}.{k}"

    def __len__(self):
        return sum([len(c) for c in self.configs])

    def __eq__(self, other):
        return all([c_a == c_b for c_a, c_b in zip(self.configs, other.configs)])

    def to_dict(self):
        return {
            "architecture": self.architecture,
            "configs": [c.to_dict() for c in self.configs],
        }

    def replace(self, **changes):
        return ConfigUnion(*[c.replace(**changes) for c in self.configs])

    @classmethod
    def from_dict(cls, config):
        if isinstance(config, AdapterConfig):
            return config

        configs = []
        for c in config["configs"]:
            config_class = cls._get_config_class(c)
            configs.append(config_class.from_dict(c))

        return cls(*configs)


# IMPORTANT: When adding a new config here, also add it to docs/overview.md!
ADAPTER_CONFIG_MAP = {
    "lora": LoRAConfig(),
    "pfeiffer": SeqBnConfig(),
    "seq_bn": SeqBnConfig(),
    "prefix_tuning": PrefixTuningConfig(),
    "prefix_tuning_flat": PrefixTuningConfig(flat=True),
    "mtl_lora": MTLLoRAConfig(),
    "lora": LoRAConfig(),
    "ia3": IA3Config(),
    "vera": VeraConfig(),
    "prompt_tuning": PromptTuningConfig(),
    "reft": ReftConfig(),
    "loreft": LoReftConfig(),
    "noreft": NoReftConfig(),
    "direft": DiReftConfig(),
}

DEFAULT_ADAPTER_CONFIG = "seq_bn"


# Mapping each composition block type to the allowed nested types
ALLOWED_NESTINGS = {
    Stack: [str, Fuse, Split, Parallel, BatchSplit, Average, MultiTask],
    Fuse: [str, Stack],
    Split: [str, Split, Stack, BatchSplit, Average],
    Parallel: [str, Stack, BatchSplit, Average],
    MultiTask: [str, Stack, Average, Fuse],
    BatchSplit: [str, Stack, Split, BatchSplit, Average],
    Average: [str, Stack, Split, BatchSplit],
}

# Some composition blocks might not be supported by all models.
# Add a whitelist of models for those here.
SUPPORTED_MODELS = {
    Parallel: [
        "albert",
        "bert",
        "roberta",
        "distilbert",
        "deberta-v2",
        "deberta",
        "bart",
        "mbart",
        "mt5",
        "plbart",
        "gpt2",
        "gptj",
        "t5",
        "vit",
        "xlm-roberta",
        "bert-generation",
        "llama",
        "mistral",
        "electra",
        "whisper",
        "xmod",
    ],
}


def validate_composition(
    adapter_composition: AdapterCompositionBlock, level=0, model_type=None
):
    if level > 1 and not (
        isinstance(adapter_composition, Stack) or isinstance(adapter_composition, str)
    ):
        raise ValueError(
            f"Adapter setup is too deep. Cannot have {adapter_composition} at level {level}."
        )
    if isinstance(adapter_composition, AdapterCompositionBlock):
        block_type = type(adapter_composition)
        if model_type and block_type in SUPPORTED_MODELS:
            if model_type not in SUPPORTED_MODELS[block_type]:
                raise ValueError(
                    f"Models of type {model_type} don't support adapter composition using {block_type.__name__}."
                )
        for child in adapter_composition:
            if not type(child) in ALLOWED_NESTINGS[type(adapter_composition)]:
                raise ValueError(
                    f"Adapter setup is invalid. Cannot nest {child} in {adapter_composition}"
                )
            # recursively validate children
            validate_composition(child, level=level + 1)


def parse_composition(
    adapter_composition, level=0, model_type=None
) -> AdapterCompositionBlock:
    """
    Parses and validates a setup of adapters.

    Args:
        adapter_composition: The adapter setup to be parsed.
        level (int, optional): If set to none, disables validation. Defaults to 0.
    """
    if not adapter_composition:
        return None
    elif isinstance(adapter_composition, AdapterCompositionBlock):
        if level is not None:
            validate_composition(
                adapter_composition, level=level, model_type=model_type
            )
        return adapter_composition
    elif isinstance(adapter_composition, str):
        if level == 0:
            return Stack(adapter_composition)
        else:
            return adapter_composition
    elif isinstance(adapter_composition, Sequence):
        # Functionality of adapter-transformers v1.x
        warnings.warn(
            "Passing list objects for adapter activation is deprecated. Please use Stack or Fuse explicitly.",
            category=FutureWarning,
        )
        # for backwards compatibility
        if level == 1:
            block_class = Fuse
        else:
            block_class = Stack
        level = level + 1 if level is not None else None
        return block_class(*[parse_composition(b, level) for b in adapter_composition])
    else:
        raise TypeError(adapter_composition)


def parse_heads_from_composition(adapter_composition, reference_heads: list = None):
    """
    Parses a potential head configuration from a setup of adapters.

    Args:
        adapter_composition: The adapter setup to be parsed.
        reference_heads: The list of available to validate the retrieved head configuration against.
    """
    final_block = adapter_composition
    if isinstance(final_block, Stack):
        final_block = final_block.children[-1]

    if isinstance(final_block, str) and (
        reference_heads is None or final_block in reference_heads
    ):
        return final_block
    elif isinstance(final_block, Parallel):
        return [a if isinstance(a, str) else a.last() for a in final_block.children]
    elif isinstance(final_block, BatchSplit):
        # Convert BatchSplit of adapters to a BatchSplit of heads.
        blocks = [
            (block.last() if isinstance(block, AdapterCompositionBlock) else block)
            for block in final_block
        ]
        head_setup = BatchSplit(*blocks, batch_sizes=final_block.batch_sizes)
        if reference_heads is None or all(
            head in reference_heads for head in head_setup
        ):
            return head_setup
        else:
            raise ValueError(
                "Missing at least one head for the given BatchSplit setup. Expected heads: {}".format(
                    blocks
                )
            )
    else:
        return None


def adjust_tensors_for_parallel(hidden_states, *tensors):
    """
    Replicates a given list of tensors based on the shape of the reference tensor (first argument).
    """
    outputs = []
    for tensor in tensors:
        if tensor is not None and hidden_states.shape[0] > tensor.shape[0]:
            repeats = [1] * len(tensor.shape)
            repeats[0] = hidden_states.shape[0] // tensor.shape[0]
            new_tensor = tensor.repeat(*repeats)
            outputs.append(new_tensor)
        else:
            outputs.append(tensor)
    return tuple(outputs)


def adjust_tensors_for_parallel_(hidden_states, *tensors):
    """
    In-place version of adjust_tensors_for_parallel().
    """
    for tensor in tensors:
        if tensor is not None and hidden_states.shape[0] > tensor.shape[0]:
            repeats = [1] * len(tensor.shape)
            repeats[0] = hidden_states.shape[0] // tensor.shape[0]
            new_tensor = tensor.repeat(*repeats)
            tensor.set_(new_tensor)


def match_attn_matrices_for_parallel(
    query, key, value
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Matches the shapes of query, key and value matrices for parallel composition.
    """
    max_bsz = max(query.shape[0], key.shape[0], value.shape[0])

    query = query.repeat(max_bsz // query.shape[0], *([1] * len(query.shape[1:])))
    key = key.repeat(max_bsz // key.shape[0], *([1] * len(key.shape[1:])))
    value = value.repeat(max_bsz // value.shape[0], *([1] * len(value.shape[1:])))

    return query, key, value


class BottleneckState(NamedTuple):
    hidden_states: torch.Tensor
    input_tensor: torch.Tensor
    adapter_residual: torch.Tensor
    layer_norm: Optional[torch.nn.Module]
    bottleneck_up: Optional[torch.Tensor] = None
    last: Optional[str] = None


class BottleneckLayer(ComposableAdapterLayerBase, nn.Module):
    adapter_modules_name = "adapters"
    supported_compositions = [Stack, Fuse, Split, Parallel, BatchSplit, Average]

    def __init__(self, location_key: str, is_layer_hooked: bool = False):
        super().__init__()
        self.location_key = location_key
        self.is_layer_hooked = is_layer_hooked

    def init_adapters(self, model_config, adapters_config):
        self._init_mapping()
        self.model_config = model_config
        self.adapters_config = adapters_config
        self.adapters = nn.ModuleDict(dict())
        self.adapter_fusion_layer = nn.ModuleDict(dict())
        if not hasattr(self, "is_layer_hooked"):
            self.is_layer_hooked = False

    def add_adapter(self, adapter_name: str, layer_idx: int) -> bool:
        self.layer_idx = layer_idx
        adapter_config = self.adapters_config.match(
            adapter_name,
            config_type=BnConfig,
            layer_idx=self.layer_idx,
            location_key=self.location_key,
        )
        if adapter_config is not None:
            reduction_factor = adapter_config["reduction_factor"]
            if isinstance(reduction_factor, Mapping):
                if str(self.layer_idx) in reduction_factor:
                    reduction_factor = reduction_factor[str(self.layer_idx)]
                elif "default" in reduction_factor:
                    reduction_factor = reduction_factor["default"]
                else:
                    raise KeyError(
                        "The given reduction factor mapping does not give a default value and does not specify each "
                        "reduction factor individually. You need to provide a default value like this: "
                        '{"1": 16, "default": 16}'
                    )

            # check unsupported configurations for layer hooking mode
            if self.is_layer_hooked:
                for key, value in LAYER_HOOK_UNSUPPORTED:
                    if adapter_config.get(key, None) == value:
                        raise ValueError(
                            f"Unsupported configuration for bottleneck layer hooking mode: {key}={value}. "
                            "Please set this configuration to a supported value."
                        )

            if adapter_config.is_parallel:
                adapter_class = ParallelAdapter
            else:
                adapter_class = Adapter
            adapter = adapter_class(
                adapter_name=adapter_name,
                input_size=self.model_config.hidden_size,
                down_sample=int(self.model_config.hidden_size // reduction_factor),
                config=adapter_config,
            )
            # for adapters hooked via interface:
            # residual & LN are applied by model, so don't apply in adapters
            if self.is_layer_hooked:
                adapter.original_ln_after = False
            adapter.train(self.training)  # make sure training mode is consistent
            self.adapters[adapter_name] = adapter
            return True

        return False

    def add_fusion_layer(self, adapter_names: Union[List, str]):
        """See BertModel.add_fusion_layer"""
        fusion_name = (
            ",".join(adapter_names)
            if isinstance(adapter_names, list)
            else adapter_names
        )
        fusion_config, adapter_names = self.adapters_config.get_fusion(fusion_name)
        if self.adapters_config.common_config_value(adapter_names, self.location_key):
            dropout_prob = fusion_config.dropout_prob or getattr(
                self.model_config, "attention_probs_dropout_prob", 0
            )
            fusion = BertFusion(
                fusion_config,
                self.model_config.hidden_size,
                dropout_prob,
            )
            fusion.train(self.training)  # make sure training mode is consistent
            self.adapter_fusion_layer[fusion_name] = fusion

    def delete_fusion_layer(self, adapter_names: Union[List, str]):
        adapter_names = (
            adapter_names if isinstance(adapter_names, str) else ",".join(adapter_names)
        )
        if adapter_names in self.adapter_fusion_layer:
            del self.adapter_fusion_layer[adapter_names]

    def enable_adapters(
        self,
        adapter_setup: AdapterCompositionBlock,
        unfreeze_adapters: bool,
        unfreeze_fusion: bool,
    ):
        """
        Unfreezes a given list of adapters, the adapter fusion layer, or both

        Args:
            adapter_names: names of adapters to unfreeze (or names of adapters part of the fusion layer to unfreeze)
            unfreeze_adapters: whether the adapter weights should be activated
            unfreeze_fusion: whether the adapter fusion layer for the given adapters should be activated
        """
        if unfreeze_adapters:
            for adapter_name in adapter_setup.flatten():
                if adapter_name in self.adapters:
                    for param in self.adapters[adapter_name].parameters():
                        param.requires_grad = True
        if unfreeze_fusion:
            if isinstance(adapter_setup, Fuse):
                if adapter_setup.name in self.adapter_fusion_layer:
                    for param in self.adapter_fusion_layer[
                        adapter_setup.name
                    ].parameters():
                        param.requires_grad = True
            for sub_setup in adapter_setup:
                if isinstance(sub_setup, Fuse):
                    if sub_setup.name in self.adapter_fusion_layer:
                        for param in self.adapter_fusion_layer[
                            sub_setup.name
                        ].parameters():
                            param.requires_grad = True

    def get_adapter_fusion(self, adapter_names: Union[List, str]):
        adapter_names = (
            adapter_names if isinstance(adapter_names, str) else ",".join(adapter_names)
        )
        if adapter_names in self.adapter_fusion_layer:
            return self.adapter_fusion_layer[adapter_names]
        else:
            return None

    def pre_block(
        self,
        adapter_setup: Union[AdapterCompositionBlock, str],
        state: BottleneckState,
    ) -> BottleneckState:
        if isinstance(adapter_setup, AdapterCompositionBlock):
            adapter_name = adapter_setup.first()
        else:
            adapter_name = adapter_setup
        first_adapter = self.adapters[adapter_name]
        hidden_states, _, residual = first_adapter.pre_forward(
            state.hidden_states, state.input_tensor, state.layer_norm
        )

        return state._replace(hidden_states=hidden_states, adapter_residual=residual)

    def vslice(self, state: BottleneckState, slice_obj: slice) -> BottleneckState:
        return BottleneckState(
            state.hidden_states[slice_obj],
            state.input_tensor[slice_obj],
            state.adapter_residual[slice_obj],
            state.layer_norm,
            (
                state.bottleneck_up[slice_obj]
                if state.bottleneck_up is not None
                else None
            ),
            state.last,
        )

    def pad_and_concat(self, states: List[BottleneckState]) -> BottleneckState:
        return BottleneckState(
            torch.cat([state.hidden_states for state in states], dim=0),
            torch.cat([state.input_tensor for state in states], dim=0),
            torch.cat([state.adapter_residual for state in states], dim=0),
            states[0].layer_norm,
            (
                torch.cat([state.bottleneck_up for state in states], dim=0)
                if states[0].bottleneck_up is not None
                else None
            ),
            states[-1].last,
        )

    def repeat(self, state: BottleneckState, channels: int) -> BottleneckState:
        return BottleneckState(
            state.hidden_states.repeat(channels, 1, 1),
            state.input_tensor.repeat(channels, 1, 1),
            state.adapter_residual.repeat(channels, 1, 1),
            state.layer_norm,
            (
                state.bottleneck_up.repeat(channels, 1, 1)
                if state.bottleneck_up is not None
                else None
            ),
            state.last,
        )

    def mean(
        self, states: List[BottleneckState], weights: torch.Tensor
    ) -> BottleneckState:
        return BottleneckState(
            torch.mean(
                torch.stack([s.hidden_states for s in states], 0) * weights,
                dim=0,
            ),
            states[0].input_tensor,
            states[0].adapter_residual,
            states[0].layer_norm,
            states[0].bottleneck_up,
            states[-1].last,
        )

    def compose_single(
        self, adapter_setup: str, state: BottleneckState, lvl: int = 0
    ) -> BottleneckState:
        adapter_layer = self.adapters[adapter_setup]
        context = ForwardContext.get_context()
        output_gating = (
            context.output_adapter_gating_scores if context is not None else False
        )
        layer_output = adapter_layer(
            state.hidden_states,
            residual_input=state.adapter_residual,
            output_gating=output_gating,
        )
        hidden_states, up = layer_output[0], layer_output[2]
        if output_gating:
            self._store_gating_score(adapter_setup, layer_output[-1])

        return state._replace(
            hidden_states=hidden_states, bottleneck_up=up, last=adapter_setup
        )

    def compose_fuse(self, adapter_setup: Fuse, state: BottleneckState, lvl: int = 0):
        """
        Performs adapter fusion with the given adapters for the given input.
        """
        context = ForwardContext.get_context()

        # config of _last_ fused adapter is significant
        fusion_config, _ = self.adapters_config.get_fusion(adapter_setup.name)
        last = adapter_setup.last()
        last_adapter = self.adapters[last]
        hidden_states, query, residual = last_adapter.pre_forward(
            state.hidden_states,
            state.input_tensor,
            state.layer_norm,
            fusion_config=fusion_config,
        )
        state = state._replace(hidden_states=hidden_states, adapter_residual=residual)

        children_states = []
        for child in adapter_setup:
            if isinstance(child, AdapterCompositionBlock):
                self.check_composition_valid(adapter_setup, child, lvl)
                composition_func = self._get_compose_func(type(child))
                child_state = composition_func(child, state, lvl=lvl + 1)
                children_states.append(child_state)
            elif child in self.adapter_modules:
                child_state = self.compose_single(child, state, lvl=lvl + 1)
                children_states.append(child_state)
            else:
                pass

        if len(children_states) > 0:
            up_list = torch.stack([state.bottleneck_up for state in children_states])
            up_list = up_list.permute(1, 2, 0, 3)

            output_fusion_attns = (
                context.output_adapter_fusion_attentions
                if context is not None
                else False
            )
            fusion_output = self.adapter_fusion_layer[adapter_setup.name](
                query,
                up_list,
                up_list,
                state.adapter_residual,
                output_attentions=output_fusion_attns,
            )
            if output_fusion_attns:
                hidden_states = fusion_output[0]
                self._store_fusion_attentions(adapter_setup.name, fusion_output[-1])
            else:
                hidden_states = fusion_output

        return state._replace(hidden_states=hidden_states, last=last)

    def compose_split(self, adapter_setup: Split, state: BottleneckState, lvl: int = 0):
        """
        Splits the given input between the given adapters.
        """
        if sum(adapter_setup.splits) != state.hidden_states.shape[1]:
            raise IndexError(
                "The given input has sequence length {} which is not equal to the sum of splits {}".format(
                    state.hidden_states.shape[1], adapter_setup.splits
                )
            )

        state = self.pre_block(adapter_setup, state)

        children_states = []
        last = None
        for i, child in enumerate(adapter_setup):
            batch_idx = (
                sum(adapter_setup.splits[:i]),
                sum(adapter_setup.splits[: i + 1]),
            )
            child_state = BottleneckState(
                state.hidden_states[:, batch_idx[0] : batch_idx[1], :],
                state.input_tensor[:, batch_idx[0] : batch_idx[1], :],
                state.adapter_residual[:, batch_idx[0] : batch_idx[1], :],
                state.layer_norm,
                (
                    state.bottleneck_up[:, batch_idx[0] : batch_idx[1], :]
                    if state.bottleneck_up is not None
                    else None
                ),
            )
            if isinstance(child, AdapterCompositionBlock):
                self.check_composition_valid(adapter_setup, child, lvl)
                composition_func = self._get_compose_func(type(child))
                child_state = composition_func(child, child_state, lvl=lvl + 1)
                children_states.append(child_state)
                last = child_state.last or last
            elif child in self.adapter_modules:
                child_state = self.compose_single(child, child_state, lvl=lvl + 1)
                children_states.append(child_state)
                last = child_state.last or last
            else:
                pass

        hidden_states = torch.cat(
            [child.hidden_states for child in children_states], dim=1
        )
        return state._replace(hidden_states=hidden_states, last=last)

    def bottleneck_layer_forward(self, hidden_states, residual_input, layer_norm):
        # Batch sizes might be different due to prefix tuning w. Parallel block
        if residual_input is not None:
            (residual_input,) = adjust_tensors_for_parallel(
                hidden_states, residual_input
            )
            # Replicate in both directions as residual might be larger (e.g. GPT-J)
            (hidden_states,) = adjust_tensors_for_parallel(
                residual_input, hidden_states
            )
        adapter_setup = self.get_active_setup()
        if adapter_setup is not None:
            input_hidden_states = hidden_states

            state = BottleneckState(
                hidden_states, residual_input, residual_input, layer_norm
            )
            state = self.compose(adapter_setup, state)
            hidden_states, residual_input, _, _, _, last = state

            last_adapter = self.adapters[last]
            hidden_states = last_adapter.post_forward(
                hidden_states, input_hidden_states, residual_input, layer_norm
            )

        elif layer_norm is not None and not self.is_layer_hooked:
            hidden_states = layer_norm(hidden_states + residual_input)
        elif residual_input is not None and not self.is_layer_hooked:
            hidden_states = hidden_states + residual_input

        return hidden_states

    def forward(self, hidden_states, residual_input, layer_norm):
        return self.bottleneck_layer_forward(hidden_states, residual_input, layer_norm)


def build_full_config(adapter_config, model_config, save_id2label=False, **kwargs):
    config_dict = {
        "model_type": model_config.model_type,
        # some models such as encoder-decoder don't have a model-wide hidden size
        "hidden_size": getattr(model_config, "hidden_size", None),
    }
    config_dict.update(kwargs)
    if not hasattr(model_config, "prediction_heads") and save_id2label:
        config_dict["label2id"] = model_config.label2id
    if isinstance(adapter_config, AdapterConfig):
        config_dict["config"] = adapter_config.to_dict()
    else:
        config_dict["config"] = adapter_config
    # add lib name before version to distinguish from adapter-transformers
    config_dict["version"] = "adapters." + __version__
    return config_dict


class PrefixTuning(nn.Module, ModuleUtilsMixin):
    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        input_size: int,
        config: PrefixTuningConfig,
        n_embd_per_head: Optional[int] = None,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.input_size = input_size
        self.n_embd_per_head = n_embd_per_head or self.input_size // self.n_heads
        self.config = config

        # Set seed for reproducibility if specified in config
        fix_seed(self.config.init_weights_seed)
        self.wte = nn.Embedding(self.config.prefix_length, self.input_size)
        self.control_trans = nn.Sequential(
            nn.Linear(self.input_size, self.config.bottleneck_size),
            Activation_Function_Class(self.config.non_linearity.lower()),
            nn.Linear(
                self.config.bottleneck_size,
                self.n_layers * 2 * self.n_heads * self.n_embd_per_head,
            ),
        )
        self.dropout = nn.Dropout(self.config.dropout)

    def eject(self):
        input_tokens = torch.arange(self.config.prefix_length).long()
        input_tokens = input_tokens.unsqueeze(0).expand(1, -1).to(self.device)
        embs = self.wte(input_tokens)
        key_values = self.control_trans(
            embs
        )  # batch_size x prefix_length x n_layers*2*input_size
        key_values = key_values.view(
            self.config.prefix_length * self.n_layers * 2 * self.input_size
        )  # *2 for key and value

        return key_values

    def forward(self, batch_size):
        input_tokens = torch.arange(self.config.prefix_length).long()
        input_tokens = input_tokens.unsqueeze(0).expand(batch_size, -1).to(self.device)
        embs = self.wte(input_tokens)
        key_values = self.control_trans(
            embs
        )  # batch_size x prefix_length x n_layers*2*input_size
        key_values = key_values.view(
            batch_size,
            self.config.prefix_length,
            self.n_layers * 2,
            self.n_heads,
            self.n_embd_per_head,
        )  # *2 for key and value
        key_values = self.dropout(key_values)
        # n_layers * (2 x batch_size x n_heads x prefix_length x n_embd_per_head)
        key_values = key_values.permute(2, 0, 3, 1, 4).split(2)

        return key_values


class FlatPrefixTuning(nn.Module, ModuleUtilsMixin):
    def __init__(
        self,
        n_layers: int,
        n_heads: int,
        input_size: int,
        config: PrefixTuningConfig,
        n_embd_per_head: Optional[int] = None,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.input_size = input_size
        self.n_embd_per_head = n_embd_per_head or self.input_size // self.n_heads
        self.config = config

        # Set seed for reproducibility if specified in config
        fix_seed(self.config.init_weights_seed)

        self.control_trans = nn.Parameter(
            torch.randn(
                self.config.prefix_length
                * self.n_layers
                * 2
                * self.n_heads
                * self.n_embd_per_head
            )
        )

        self.dropout = nn.Dropout(self.config.dropout)

    def forward(self, batch_size):
        key_values = (
            self.control_trans.unsqueeze(0)
            .expand(batch_size, -1)
            .view(
                batch_size,
                self.config.prefix_length,
                self.n_layers * 2,
                self.n_heads,
                self.n_embd_per_head,
            )
            .to(self.device)
        )  # *2 for key and value
        key_values = self.dropout(key_values)
        # n_layers * (2 x batch_size x n_heads x prefix_length x n_embd_per_head)
        key_values = key_values.permute(2, 0, 3, 1, 4).split(2)

        return key_values


class PrefixTuningGroup(nn.ModuleDict):
    def __init__(self, module_configs, prefix_tuning_config):
        super().__init__()
        if prefix_tuning_config["flat"]:
            prefix_tuning_class = FlatPrefixTuning
        else:
            prefix_tuning_class = PrefixTuning
        for k, kwargs in module_configs.items():
            self[k] = prefix_tuning_class(**kwargs, config=prefix_tuning_config)

    def eject(self):
        """Converts all PrefixTuning modules into FlatPrefixTuning modules."""
        for k, v in self.items():
            if isinstance(v, PrefixTuning):
                config = v.config.replace(flat=True)
                self[k] = FlatPrefixTuning(v.n_layers, v.n_heads, v.input_size, config)
                weights = v.eject()
                self[k].control_trans = nn.Parameter(weights)

    def forward(self, batch_size):
        return {k: v(batch_size) for k, v in self.items()}


class PrefixTuningPool(nn.Module):
    def __init__(
        self,
        model_config: PretrainedConfig,
        adapters_config: ModelAdaptersConfig,
    ):
        super().__init__()
        self.model_config = model_config
        self.adapters_config = adapters_config
        self.prefix_counts = {}
        self.prefix_tunings = nn.ModuleDict()

    def indicate_prefix(self, prefix_name: str, location_key: str, **kwargs):
        if prefix_name not in self.prefix_counts:
            self.prefix_counts[prefix_name] = {location_key: {"count": 1, **kwargs}}
        elif location_key not in self.prefix_counts[prefix_name]:
            self.prefix_counts[prefix_name][location_key] = {
                "count": 1,
                **kwargs,
            }
        else:
            # TODO-AH: Check if kwargs are the same
            self.prefix_counts[prefix_name][location_key]["count"] += 1

        return self.prefix_counts[prefix_name][location_key]["count"] - 1

    def confirm_prefix(self, prefix_name: str) -> bool:
        """Create Prefix Tuning module based on shim layer infications."""
        prefix_tuning_config = self.adapters_config.match(
            prefix_name, PrefixTuningConfig
        )
        if prefix_tuning_config is None:
            return False

        if prefix_name not in self.prefix_counts:
            raise ValueError(f"Prefix {prefix_name} not found in PrefixTuningPool")

        module_configs = {}
        for location_key, location_config in self.prefix_counts[prefix_name].items():
            module_configs[location_key] = {
                "n_layers": location_config["count"],
                "n_heads": location_config["n_heads"],
                "input_size": location_config["input_size"],
                "n_embd_per_head": location_config["n_embd_per_head"],
            }
        prefix_tuning = PrefixTuningGroup(module_configs, prefix_tuning_config)
        prefix_tuning.train(self.training)  # make sure training mode is consistent
        self.prefix_tunings[prefix_name] = prefix_tuning
        del self.prefix_counts[prefix_name]
        return True

    def average_prefix(
        self,
        prefix_name: str,
        input_adapters: Dict[str, float],
        combine_strategy: str,
        **kwargs,
    ) -> bool:
        if self.confirm_prefix(prefix_name):
            # Prefix Tuning only support linear combination
            if combine_strategy != "linear":
                raise ValueError(
                    f"Combine strategy {combine_strategy} not supported for prefix tuning."
                )

            # average weights
            avg_state_dict = {}
            for name, weight in input_adapters.items():
                module = self.prefix_tunings[name]
                if module is not None:
                    for k, v in module.state_dict().items():
                        if k in avg_state_dict:
                            avg_state_dict[k] += weight * v
                        else:
                            avg_state_dict[k] = weight * v
            # load averaged weights
            self.prefix_tunings[prefix_name].load_state_dict(avg_state_dict)
            return True

        return False

    def delete_prefix(self, prefix_name: str):
        if prefix_name in self.prefix_tunings:
            del self.prefix_tunings[prefix_name]

    def enable_prefix(self, prefix_name: str):
        if prefix_name in self.prefix_tunings:
            for param in self.prefix_tunings[prefix_name].parameters():
                param.requires_grad = True

    def get_prefix(self, prefix_name: str):
        if prefix_name in self.prefix_tunings:
            return self.prefix_tunings[prefix_name]
        else:
            return None

    def forward(self, *args, **kwargs):
        context = AdapterSetup.get_context()
        if context is not None:
            adapter_setup = context.adapter_setup
        else:
            adapter_setup = self.adapters_config.active_setup

        prefix_states = {}
        if adapter_setup is not None:
            # Infer batch size
            input_tensor_names = [
                "input_ids",
                "decoder_input_ids",
                "attention_mask",
                "inputs_embeds",
                "pixel_values",
                "input_features",
            ]
            batch_size = None
            for name in input_tensor_names:
                if kwargs.get(name, None) is not None:
                    batch_size = kwargs[name].size(0)
                    break
            if batch_size is None:
                if len(args) > 0:
                    batch_size = args[0].size(0)
                else:
                    raise ValueError(
                        "Could not infer batch size for prefix tuning from inputs."
                    )

            # Pass to sub-layers
            for name in adapter_setup.flatten():
                if name in self.prefix_tunings:
                    prefix_states[name] = self.prefix_tunings[name](batch_size)

        return prefix_states


class PrefixTuningState(NamedTuple):
    key_states: torch.Tensor
    value_states: torch.Tensor
    residual_input: torch.Tensor
    attention_mask: Optional[torch.Tensor]
    invert_mask: bool
    idx_slice: Optional[slice] = None


class PrefixTuningLayer(ComposableAdapterLayerBase, nn.Module):
    adapter_modules_name = "prefixes"
    supported_compositions = [Stack, Parallel, BatchSplit]

    def __init__(
        self,
        location_key: str,
        model_config: PretrainedConfig,
        adapters_config: ModelAdaptersConfig,
        add_model_type_to_key: bool = False,
    ):
        super().__init__()
        self.model_config = model_config
        self.adapters_config = adapters_config
        self.location_key = location_key
        if add_model_type_to_key:
            self.location_key = f"{self.model_config.model_type}_{self.location_key}"
        self.prefixes = {}
        self.prefix_gates = nn.ModuleDict()

    def set_pool(self, pool: PrefixTuningPool):
        self.__setattr__("pool", pool)

    def add_adapter(self, adapter_name: str, layer_idx: int) -> bool:
        self.layer_idx = layer_idx
        # only match location keys for which we have config keys
        if self.location_key.startswith("cross") or self.location_key.startswith(
            "encoder"
        ):
            used_location_key = self.location_key
        else:
            used_location_key = None
        prefix_tuning_config = self.adapters_config.match(
            adapter_name,
            config_type=PrefixTuningConfig,
            layer_idx=self.layer_idx,
            location_key=used_location_key,
        )
        if prefix_tuning_config is not None:
            prefix_id = self.pool.indicate_prefix(
                adapter_name,
                self.location_key,
                n_heads=self.model_config.num_attention_heads,
                input_size=self.model_config.hidden_size,
                n_embd_per_head=getattr(
                    self.model_config, "d_kv", None
                ),  # this is currently specific to T5-3B
            )
            self.prefixes[adapter_name] = prefix_id

            if prefix_tuning_config.use_gating:
                gate_outputs = 1 if prefix_tuning_config.shared_gating else 2
                gate = nn.Linear(self.model_config.hidden_size, gate_outputs)
                gate.weight.data.normal_(mean=0.0, std=0.02)
                self.prefix_gates[adapter_name] = gate
            return True

        return False

    def average_adapter(
        self,
        adapter_name: str,
        input_adapters: Dict[str, float],
        combine_strategy: str,
        **kwargs,
    ) -> bool:
        # add new adapter
        if self.add_adapter(adapter_name, self.layer_idx):
            # Prefix Tuning only support linear combination
            if combine_strategy != "linear":
                raise ValueError(
                    f"Combine strategy {combine_strategy} not supported for prefix tuning."
                )

            # prefix averaging is handled in pool, only average gates here
            if adapter_name in self.prefix_gates:
                avg_state_dict = {}
                for name, weight in input_adapters.items():
                    if name in self.prefix_gates:
                        module = self.prefix_gates[name]
                        for k, v in module.state_dict().items():
                            if k in avg_state_dict:
                                avg_state_dict[k] += weight * v
                            else:
                                avg_state_dict[k] = weight * v
                    else:
                        self.delete_adapter(
                            adapter_name
                        )  # clean up before raising error
                        raise ValueError("Adapter {} not found.".format(name))
                # load averaged weights
                self.prefix_gates[adapter_name].load_state_dict(avg_state_dict)
            return True
        else:
            return False

    def delete_adapter(self, adapter_name: str):
        self.pool.delete_prefix(adapter_name)
        if adapter_name in self.prefixes:
            del self.prefixes[adapter_name]
        if adapter_name in self.prefix_gates:
            del self.prefix_gates[adapter_name]

    def enable_adapters(
        self,
        adapter_setup: AdapterCompositionBlock,
        unfreeze_adapters: bool,
        unfreeze_fusion: bool,
    ):
        if unfreeze_adapters:
            for prefix_tuning_name in adapter_setup.flatten():
                self.pool.enable_prefix(prefix_tuning_name)
                if prefix_tuning_name in self.prefix_gates:
                    for param in self.prefix_gates[prefix_tuning_name].parameters():
                        param.requires_grad = unfreeze_adapters

    def freeze_adapter(self, adapter_name: str, freeze: bool = True):
        if adapter_name in self.prefixes:
            self.pool.get_prefix(adapter_name)[self.location_key].train(not freeze)
            for param in self.pool.get_prefix(adapter_name)[
                self.location_key
            ].parameters():
                param.requires_grad = not freeze
            if adapter_name in self.prefix_gates:
                for param in self.prefix_gates[adapter_name].parameters():
                    param.requires_grad = not freeze

    def get_adapter(self, adapter_name):
        return_dict = nn.ModuleDict()
        # Make sure to only return params once
        if adapter_name in self.prefixes and self.prefixes[adapter_name] == 0:
            prefix_module = self.pool.get_prefix(adapter_name)
            if prefix_module is not None:
                return_dict["prefix"] = prefix_module[self.location_key]
        if adapter_name in self.prefix_gates:
            return_dict["gate"] = self.prefix_gates[adapter_name]
        if len(return_dict) > 0:
            return return_dict

        return None

    def vslice(
        self, state: PrefixTuningState, slice_obj: Union[slice, torch.Tensor]
    ) -> PrefixTuningState:
        if isinstance(slice_obj, torch.Tensor):
            split_idx_slice = None
        elif state.idx_slice is None:
            split_idx_slice = slice_obj
        else:
            split_idx_slice = slice(
                state.idx_slice.start + slice_obj.start,
                state.idx_slice.start + slice_obj.stop,
            )
        return PrefixTuningState(
            key_states=state.key_states[slice_obj],
            value_states=state.value_states[slice_obj],
            residual_input=state.residual_input[slice_obj],
            attention_mask=(
                state.attention_mask[slice_obj]
                if state.attention_mask is not None
                else None
            ),
            invert_mask=state.invert_mask,
            idx_slice=split_idx_slice,
        )

    def pad_and_concat(self, states: List[PrefixTuningState]) -> PrefixTuningState:
        """Pads all key & value states to the longest prefix length in the current batch.
        This is required e.g. for stacked prefix tunings.
        """
        max_prefix_length = max([state.key_states.shape[-2] for state in states])
        (
            all_key_states,
            all_value_states,
            all_residual_input,
            all_attention_mask,
        ) = ([], [], [], [])
        for state in states:
            key_states, value_states, residual_input, attention_mask = state[:4]
            # pad sizes
            pad_length = max_prefix_length - key_states.shape[-2]
            pad_size = (0, 0, pad_length, 0)
            key_states = F.pad(
                key_states, pad_size, "constant", self.model_config.pad_token_id
            )
            value_states = F.pad(
                value_states,
                pad_size,
                "constant",
                self.model_config.pad_token_id,
            )

            # pad attention mask
            if pad_length > 0 and attention_mask is not None:
                # Masking the padded tokens only works correctly if attention_mask is set
                attention_mask = F.pad(
                    attention_mask,
                    (max_prefix_length - attention_mask.shape[-1], 0),
                    "constant",
                    1.0 if state.invert_mask else 0.0,
                )

            all_key_states.append(key_states)
            all_value_states.append(value_states)
            all_residual_input.append(residual_input)
            all_attention_mask.append(attention_mask)

        all_key_states = torch.cat(all_key_states, dim=0)
        all_value_states = torch.cat(all_value_states, dim=0)
        all_residual_input = torch.cat(all_residual_input, dim=0)
        all_attention_mask = (
            torch.cat(all_attention_mask, dim=0) if attention_mask is not None else None
        )

        return PrefixTuningState(
            key_states=all_key_states,
            value_states=all_value_states,
            residual_input=all_residual_input,
            attention_mask=all_attention_mask,
            invert_mask=states[0].invert_mask,
            idx_slice=states[0].idx_slice,
        )

    def repeat(self, state: PrefixTuningState, channels: int) -> PrefixTuningState:
        if state.attention_mask is not None:
            if (
                state.attention_mask.dim() == 2
            ):  # e.g. for DistilBERT, attention_mask has shape (batch_size, seq_len)
                attention_mask = state.attention_mask.repeat(channels, 1)
            else:
                attention_mask = state.attention_mask.repeat(channels, 1, 1, 1)
        else:
            attention_mask = None
        return PrefixTuningState(
            key_states=state.key_states.repeat(channels, 1, 1, 1),
            value_states=state.value_states.repeat(channels, 1, 1, 1),
            residual_input=state.residual_input.repeat(channels, 1, 1),
            attention_mask=attention_mask,
            invert_mask=state.invert_mask,
            idx_slice=state.idx_slice,
        )

    def mean(
        self, states: List[PrefixTuningState], weights: torch.Tensor
    ) -> PrefixTuningState:
        # TODO implement average composition
        raise NotImplementedError()

    def compose_single(
        self, adapter_setup: str, state: PrefixTuningState, lvl: int = 0
    ) -> PrefixTuningState:
        prefix_id = self.prefixes[adapter_setup]
        batch_size = state.key_states.size(0)

        # Retrieve pre-computed prefix states from context
        context = ForwardContext.get_context()
        # batch_size x n_heads x prefix_length x n_embd_per_head
        prefix_keys, prefix_values = context.prefix_states[adapter_setup][
            self.location_key
        ][prefix_id]

        # Select index range for batch split
        # Ignore slices that go beyond the prefix states bsz
        # (this is the case for slices produced by Parallel blocks which operate on replicated kv states)
        # But, let pass slices go beyond which return empty tensor
        # (this is the case for last batch_size == 0 for BatchSplit blocks)
        if state.idx_slice is not None and (
            state.idx_slice.start < prefix_keys.size(0)
            or state.idx_slice.start == state.idx_slice.stop == prefix_keys.size(0)
        ):
            prefix_keys = prefix_keys[state.idx_slice]
            prefix_values = prefix_values[state.idx_slice]

        if adapter_setup in self.prefix_gates:
            gate = self.prefix_gates[adapter_setup]
            gate_output = torch.mean(torch.sigmoid(gate(state.residual_input)), dim=1)
            self._store_gating_score(adapter_setup, gate_output)
            gate_output_key = gate_output[:, 0].view(-1, 1, 1, 1)
            gate_output_value = gate_output[:, -1].view(-1, 1, 1, 1)
            prefix_keys = prefix_keys * gate_output_key
            prefix_values = prefix_values * gate_output_value

        # Replicate for Parallel block
        prefix_keys, prefix_values = adjust_tensors_for_parallel(
            state.key_states, prefix_keys, prefix_values
        )

        key_states = torch.cat([prefix_keys, state.key_states], dim=2)
        value_states = torch.cat([prefix_values, state.value_states], dim=2)
        if state.attention_mask is not None:
            if (
                state.attention_mask.dim() == 2
            ):  # e.g. for DistilBERT, attention_mask has shape (batch_size, seq_len)
                prefix_mask = torch.ones(batch_size, prefix_keys.size(2)).to(
                    device=state.attention_mask.device,
                    dtype=state.attention_mask.dtype,
                )
            else:
                prefix_mask = torch.ones(
                    batch_size,
                    1,
                    state.attention_mask.size(2),
                    prefix_keys.size(2),
                ).to(
                    device=state.attention_mask.device,
                    dtype=state.attention_mask.dtype,
                )
            if state.invert_mask:
                prefix_mask = 1.0 - prefix_mask
            (prefix_mask,) = adjust_tensors_for_parallel(
                state.attention_mask, prefix_mask
            )
            attention_mask = torch.cat([prefix_mask, state.attention_mask], dim=-1)
        else:
            attention_mask = None

        return state._replace(
            key_states=key_states,
            value_states=value_states,
            attention_mask=attention_mask,
        )

    def forward(
        self,
        key_states,
        value_states,
        residual_input,
        attention_mask=None,
        invert_mask=True,
    ):
        adapter_setup = self.get_active_setup()
        if adapter_setup is not None:
            state = PrefixTuningState(
                key_states,
                value_states,
                residual_input,
                attention_mask,
                invert_mask,
            )
            state = self.compose(adapter_setup, state)
            key_states, value_states, residual_input, attention_mask = state[:4]

        return key_states, value_states, attention_mask


class PromptTuning(nn.Module):
    prompt: nn.Module
    combination_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

    def __init__(
        self,
        adapter_name: str,
        prompt_tuning_config: PromptTuningConfig,
        model_config: PretrainedConfig,
        base_model_embeddings: nn.Module,
    ):
        super().__init__()

        self.name = adapter_name
        self.model_config = model_config
        self.prompt_tuning_config = prompt_tuning_config

        embedding_size = getattr(
            model_config, "embedding_size", model_config.hidden_size
        )

        self.prompt_embedding = nn.Embedding(
            num_embeddings=prompt_tuning_config.prompt_length,
            embedding_dim=embedding_size,
        )
        # Initialize prompt tokens
        self.prompt_tokens = torch.arange(prompt_tuning_config.prompt_length).long()

        self._init_prompt_embedding(base_model_embeddings)

        if prompt_tuning_config.combine == "prefix":
            self.combination_fn = lambda prompt, embedded_input: torch.cat(
                [prompt, embedded_input], dim=1
            )
        elif prompt_tuning_config.combine == "prefix_after_bos":
            self.combination_fn = lambda prompt, embedded_input: torch.cat(
                [
                    embedded_input[:, 0, np.newaxis],
                    prompt,
                    embedded_input[:, 1:],
                ],
                dim=1,
            )
        else:
            raise ValueError(
                f"Unknown combination function: {prompt_tuning_config.combine}. "
                "Must be one of 'prefix' or 'prefix_after_bos'."
            )

    def _init_prompt_embedding(self, base_model_embeddings: nn.Module) -> None:

        # Set seed for reproducibility if specified in config
        fix_seed(self.prompt_tuning_config.init_weights_seed)

        if self.prompt_tuning_config.prompt_init == "random_uniform":
            nn.init.uniform_(
                self.prompt_embedding.weight,
                a=-self.prompt_tuning_config.random_uniform_scale,
                b=self.prompt_tuning_config.random_uniform_scale,
            )

        elif self.prompt_tuning_config.prompt_init == "from_string":
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_config.tokenizer_name_or_path
            )
            prompt_length = self.prompt_tuning_config.prompt_length
            prompt_text = self.prompt_tuning_config.prompt_init_text
            if prompt_text is None:
                raise ValueError(
                    "Prompt text must be provided when using prompt_init='from_string'."
                )

            tokenized_prompt_text: list[int] = tokenizer(prompt_text)["input_ids"]  # type: ignore

            # If the prompt text tokens are shorter than the prompt length, we repeat the prompt text tokens until we reach the prompt length
            if len(tokenized_prompt_text) < prompt_length:
                num_reps = math.ceil(prompt_length / len(tokenized_prompt_text))
                tokenized_prompt_text = tokenized_prompt_text * num_reps

            # Adjust length of prompt text tokens to match prompt_length
            tokenized_prompt_text = tokenized_prompt_text[:prompt_length]

            # Initialize prompt embedding with tokenized prompt text
            word_embedding_weights = (
                base_model_embeddings(torch.LongTensor(tokenized_prompt_text))
                .detach()
                .clone()
            )
            word_embedding_weights = word_embedding_weights.to(torch.float32)
            self.prompt_embedding.weight = nn.Parameter(word_embedding_weights)

        else:
            raise ValueError(
                f"Unknown prompt initialization: {self.prompt_tuning_config.prompt_init}"
            )

    def forward(self, embedded_input):
        # Compute prompt embedding
        self.prompt_tokens = self.prompt_tokens.to(embedded_input.device)
        prompt = self.prompt_embedding(self.prompt_tokens)

        # Prompt to batch size
        batch_size = embedded_input.shape[0]
        prompt = torch.tile(
            torch.unsqueeze(prompt, dim=0),
            [batch_size] + [1 for _ in prompt.shape],
        )

        # Merge prompt and input
        output = self.combination_fn(prompt, embedded_input)

        # Adapt attention mask
        prefix_attention_mask_length = self.prompt_tuning_config.prompt_length

        return output, prefix_attention_mask_length


class PromptTuningLayer(AdapterLayerBase, nn.Module):

    adapter_modules_name = "prompt_tunings"

    def __init__(
        self,
        model_config: PretrainedConfig,
        adapters_config: ModelAdaptersConfig,
        base_model_embeddings: nn.Module,
    ):
        super().__init__()
        self.model_config = model_config
        self.adapters_config = adapters_config
        self.base_model_embeddings = base_model_embeddings
        self.prompt_tunings = nn.ModuleDict()

    def add_adapter(self, adapter_name: str, layer_idx: int) -> bool:
        # ignore layer_idx as prompt tunings are only added after the embedding layer
        prompt_tuning_config = self.adapters_config.match(
            adapter_name,
            config_type=PromptTuningConfig,
        )

        if prompt_tuning_config is not None:
            adapter = PromptTuning(
                adapter_name=adapter_name,
                prompt_tuning_config=prompt_tuning_config,  # type: ignore
                model_config=self.model_config,
                base_model_embeddings=self.base_model_embeddings,
            )
            adapter.train(self.training)  # make sure training mode is consistent
            self.prompt_tunings[adapter_name] = adapter
            return True

        return False

    def forward(self, hidden_states: torch.Tensor):
        prefix_attention_mask_length = None
        adapter_setup = self.get_active_setup()
        if adapter_setup is not None and len(adapter_setup) > 0:
            first_adapter = adapter_setup.first()
            if first_adapter in self.prompt_tunings:
                hidden_states, prefix_attention_mask_length = self.prompt_tunings[
                    first_adapter
                ](hidden_states)

        context = ForwardContext.get_context()
        if context is not None:
            context.prompt_tokens_length = prefix_attention_mask_length

        return hidden_states


def hook_fn(model, module, args, embedding_output):
    embedding_output = model.prompt_tuning.forward(embedding_output)
    return embedding_output


# TODO: this will only work for a limited set of models
def _attn_mask_hook_fn(module, args):
    attn_mask = args[1]
    attn_mask = prefix_attention_mask(attn_mask)
    return (args[0], attn_mask) + args[2:]


def init_prompt_tuning(model):
    model = model.base_model
    if not hasattr(model, "prompt_tuning"):
        model.support_prompt_tuning = True
        model.prompt_tuning = PromptTuningLayer(
            model.config, model.adapters_config, model.get_input_embeddings()
        )
        embed_layer = multigetattr(model, model.adapter_interface.model_embeddings)
        embed_layer.register_forward_hook(partial(hook_fn, model))

        for _, layer in model.iter_layers():
            layer.register_forward_pre_hook(_attn_mask_hook_fn)


class ReftUnit(nn.Module):
    def __init__(
        self,
        in_dim: int,
        r_dim: int,
        orthogonal: bool = False,
        subtract_projection: bool = True,
        non_linearity: str = None,
        dropout: float = 0.0,
        init_weights_seed: int = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.orthogonal = orthogonal

        # Set seed for reproducibility if specified in config
        fix_seed(init_weights_seed)
        self.learned_source = nn.Linear(in_dim, r_dim, bias=True, dtype=dtype)
        projection = nn.Linear(in_dim, r_dim, bias=False, dtype=dtype)

        if orthogonal:
            # orthogonal is not implemented for half precision
            if dtype in [torch.float16, torch.bfloat16]:
                logger.warning(
                    "Orthogonal parametrization is not supported for half precision dtypes. Converting REFT projection layer to float32.",
                    UserWarning,
                )
                projection = projection.to(dtype=torch.float32)
            if projection.weight.device == torch.device("meta"):
                projection = projection.to_empty(device="cpu")
            self.projection = nn.utils.parametrizations.orthogonal(projection)
        else:
            self.projection = projection

        self.subtract_projection = subtract_projection
        self.non_linearity = Activation_Function_Class(non_linearity)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        source_states = self.non_linearity(self.learned_source(x))
        if self.subtract_projection:
            projected_states = self.projection(x)
            source_states = source_states - projected_states
        adapted_output = x + torch.matmul(source_states, self.projection.weight)
        adapted_output = self.dropout(adapted_output)
        return adapted_output


class ReftModule(nn.Module):
    def __init__(self, in_features: int, config: ReftConfig):
        super().__init__()
        self.prefix_positions = config.prefix_positions
        self.suffix_positions = config.suffix_positions
        self.tied_weights = config.tied_weights
        n_units = 1 if config.tied_weights else 2
        dtype = getattr(torch, config.dtype) if config.dtype else None
        self.units = nn.ModuleList(
            [
                ReftUnit(
                    in_features,
                    config.r,
                    config.orthogonality,
                    config.subtract_projection,
                    config.non_linearity,
                    config.dropout,
                    config.init_weights_seed,
                    dtype,
                )
                for _ in range(n_units)
            ]
        )

    def _gather_adapted_states(self, hidden_states: torch.Tensor):
        context = ForwardContext.get_context()
        bsz, seq_len, ddim = hidden_states.size()

        # if cached indexing matrices are computed for different hidden_states size -> recompute
        cache_invalidated = False
        if hasattr(context, "pref_idx") and hasattr(context, "suff_idx"):
            cache_invalidated = (
                torch.max(context.suff_idx) >= seq_len  # indices out of bounds
                or bsz != context.suff_idx.size(0)  # batch size mismatch
                or ddim != context.suff_idx.size(2)  # hidden size mismatch
            )

        # no cached indexing matrices available -> compute now
        if (
            not hasattr(context, "pref_idx")
            and not hasattr(context, "suff_idx")
            or cache_invalidated
        ):
            # read offsets & lengths from context
            if hasattr(context, "seqlens"):
                first_non_padding = context.offsets
                last_non_padding = context.offsets + context.seqlens
            else:
                first_non_padding = torch.tensor([0] * hidden_states.size(0)).to(
                    hidden_states.device
                )
                last_non_padding = torch.tensor(
                    [hidden_states.size(1)] * hidden_states.size(0)
                ).to(hidden_states.device)
            # create indexing matrices for prefixes & suffixes
            if self.prefix_positions > 0:
                real_pref_len = min(self.prefix_positions, hidden_states.size(1))
                pref_idx = first_non_padding.view(-1, 1, 1) + (
                    torch.arange(real_pref_len)
                    .unsqueeze(-1)
                    .expand(bsz, real_pref_len, ddim)
                    .to(hidden_states.device)
                )
                # Cache for next layer
                context.pref_idx = pref_idx
            if self.suffix_positions > 0:
                real_suff_len = min(self.suffix_positions, hidden_states.size(1))
                suff_idx = last_non_padding.view(-1, 1, 1) + (
                    torch.arange(-real_suff_len, 0)
                    .unsqueeze(-1)
                    .expand(bsz, real_suff_len, ddim)
                    .to(hidden_states.device)
                )
                context.suff_idx = suff_idx

        # gather prefix & suffix states
        if self.prefix_positions > 0:
            prefix = hidden_states.gather(1, context.pref_idx)
        else:
            prefix = torch.zeros(bsz, 0, ddim, device=hidden_states.device)
        if self.suffix_positions > 0:
            suffix = hidden_states.gather(1, context.suff_idx)
        else:
            suffix = torch.zeros(bsz, 0, ddim, device=hidden_states.device)

        if self.tied_weights:
            adapted_states = [torch.cat([prefix, suffix], dim=1)]
        else:
            adapted_states = [prefix, suffix]

        return adapted_states

    def _scatter_adapted_states(
        self, hidden_states: torch.Tensor, adapted_states: List[torch.Tensor]
    ):
        context = ForwardContext.get_context()

        # merge prefix, suffix and adapted states
        adapted_output = torch.cat(adapted_states, dim=1).to(hidden_states.dtype)

        if self.prefix_positions > 0:
            hidden_states = torch.scatter(
                hidden_states,
                1,
                context.pref_idx,
                adapted_output[:, : self.prefix_positions, :],
            )
        if self.suffix_positions > 0:
            hidden_states = torch.scatter(
                hidden_states,
                1,
                context.suff_idx,
                adapted_output[:, -self.suffix_positions :, :],
            )

        return hidden_states

    def forward(self, hidden_states: torch.Tensor):
        # hidden_states shape: (batch_size, seq_len, hidden_size)
        adapted_states = self._gather_adapted_states(hidden_states)

        # apply reft
        for i, unit in enumerate(self.units):
            adapted_states[i] = unit(adapted_states[i])

        output = self._scatter_adapted_states(hidden_states, adapted_states)

        return output


class ReftLayer(AdapterLayerBase, nn.Module):
    adapter_modules_name = "refts"

    def __init__(self, location_key: str, model_config, adapters_config):
        super().__init__()
        self.location_key = location_key + "_reft"
        self.model_config = model_config
        self.adapters_config = adapters_config
        self.refts = nn.ModuleDict()

    def add_adapter(self, adapter_name: str, layer_idx: int) -> bool:
        self.layer_idx = layer_idx
        reft_config = self.adapters_config.match(
            adapter_name,
            config_type=ReftConfig,
            layer_idx=self.layer_idx,
            location_key=self.location_key,
        )
        if reft_config is not None and (
            reft_config.layers == "all" or self.layer_idx in reft_config.layers
        ):
            reft = ReftModule(
                self.model_config.hidden_size,
                reft_config,
            )
            reft.train(self.training)
            self.refts[adapter_name] = reft
            return True

        return False

    def forward(self, hidden_states: torch.Tensor):
        adapter_setup = self.get_active_setup()
        if adapter_setup is not None:
            first_adapter = adapter_setup.first()
            if first_adapter in self.refts:
                hidden_states = self.refts[first_adapter](hidden_states)

        return hidden_states

    def pre_save_adapters(self):
        # Make sure orthogonal parametrizations are contiguous, otherwise saving with safetensors will fail
        for reft in self.refts.values():
            for unit in reft.units:
                if unit.orthogonal:
                    unit.projection.parametrizations.weight[
                        0
                    ].base = unit.projection.parametrizations.weight[
                        0
                    ].base.contiguous()


def hook_fn(module, args, output):
    if isinstance(output, torch.Tensor):
        return module.reft_layer(output)
    else:
        return (module.reft_layer(output[0]),) + output[1:]


def init_reft(model):
    model = model.base_model
    for _, layer in model.iter_layers():
        if not hasattr(layer, "reft_layer"):
            layer.reft_layer = ReftLayer("output", model.config, model.adapters_config)
            layer.register_forward_hook(hook_fn)


def init_shared_parameters(config, in_features, device):
    """
    Create and initialize the parameters shared by all compacter modules
    """
    parameters = nn.ParameterDict()
    if config["shared_W_phm"]:
        if config["factorized_phm_W"]:
            out_features = in_features // config["reduction_factor"]
            _in_feats_per_axis = in_features // config["phm_dim"]
            _out_feats_per_axis = out_features // config["phm_dim"]
            W_down_left = torch.Tensor(
                size=(config["phm_dim"], _in_feats_per_axis, config["phm_rank"])
            )
            W_down_right = torch.Tensor(
                size=(config["phm_dim"], config["phm_rank"], _out_feats_per_axis)
            )
            W_up_left = torch.Tensor(
                size=(config["phm_dim"], _out_feats_per_axis, config["phm_rank"])
            )
            W_up_right = torch.Tensor(
                size=(config["phm_dim"], config["phm_rank"], _in_feats_per_axis)
            )
            init_W(config, W_left=W_down_left, W_right=W_down_right)
            init_W(config, W_left=W_up_left, W_right=W_up_right)
            parameters["W_down_left"] = nn.Parameter(W_down_left, requires_grad=True)
            parameters["W_down_right"] = nn.Parameter(W_down_right, requires_grad=True)
            parameters["W_up_left"] = nn.Parameter(W_up_left, requires_grad=True)
            parameters["W_up_right"] = nn.Parameter(W_up_right, requires_grad=True)
        else:
            W_down = torch.Tensor(
                size=(config["phm_dim"], _in_feats_per_axis, _out_feats_per_axis)
            )
            W_up = torch.Tensor(
                size=(config["phm_dim"], _out_feats_per_axis, _in_feats_per_axis)
            )
            init_W(config, W=W_down)
            init_W(config, W=W_up)
            parameters["W_down"] = nn.Parameter(W_down, requires_grad=True)
            parameters["W_up"] = nn.Parameter(W_up, requires_grad=True)
    if config["shared_phm_rule"]:
        if config["factorized_phm_rule"]:
            phm_rule_left = nn.Parameter(
                torch.FloatTensor(config["phm_dim"], config["phm_dim"], 1).to(device),
                requires_grad=config["learn_phm"],
            )
            phm_rule_right = nn.Parameter(
                torch.FloatTensor(config["phm_dim"], 1, config["phm_dim"]).to(device),
                requires_grad=config["learn_phm"],
            )
            if config["phm_c_init"] == "normal":
                phm_rule_left.data.normal_(mean=0, std=config["phm_init_range"])
                phm_rule_right.data.normal_(mean=0, std=config["phm_init_range"])
            elif config["phm_c_init"] == "uniform":
                phm_rule_left.data.uniform_(-1, 1)
                phm_rule_right.data.uniform_(-1, 1)
            else:
                raise NotImplementedError
            parameters["phm_rule_left"] = phm_rule_left
            parameters["phm_rule_right"] = phm_rule_right
        else:
            phm_rule = nn.Parameter(
                torch.FloatTensor(
                    config["phm_dim"], config["phm_dim"], config["phm_dim"]
                ),
                requires_grad=config["learn_phm"],
            )
            if config["phm_c_init"] == "normal":
                phm_rule.data.normal_(mean=0, std=config["phm_init_range"])
            elif config["phm_c_init"] == "uniform":
                phm_rule.data.uniform_(-1, 1)
            else:
                raise NotImplementedError
            parameters["phm_rule"] = phm_rule
    return parameters


def init_W(config, W_left=None, W_right=None, W=None):
    """
    Initialize the weights for the compacter module or the shared parameters
    """
    if config["factorized_phm_W"]:
        W_left = W_left
        W_right = W_right
    else:
        W = W
    if config["hypercomplex_nonlinearity"]:
        if config["factorized_phm_W"]:
            for i in range(config["phm_dim"]):
                W_left.data[i] = nn.init.xavier_normal_(W_left.data[i])
                W_right.data[i] = nn.init.xavier_normal_(W_right.data[i])
        else:
            for i in range(config["phm_dim"]):
                W.data[i] = nn.init.xavier_normal_(W.data[i])
    elif config["hypercomplex_nonlinearity"] == "glorot-uniform":
        if config["factorized_phm_W"]:
            for i in range(config["phm_dim"]):
                W_left.data[i] = nn.init.xavier_uniform_(W_left.data[i])
                W_right.data[i] = nn.init.xavier_uniform_(W_right.data[i])
        else:
            for i in range(config["phm_dim"]):
                W.data[i] = nn.init.xavier_uniform_(W.data[i])
    elif config["hypercomplex_nonlinearity"] == "normal":
        if config["factorized_phm_W"]:
            for i in range(config["phm_dim"]):
                W_left.data[i].normal_(mean=0, std=config["phm_init_range"])
                W_right.data[i].normal_(mean=0, std=config["phm_init_range"])
        else:
            for i in range(config["phm_dim"]):
                W.data[i].normal_(mean=0, std=config["phm_init_range"])
    else:
        raise ValueError


CONFIG_CLASS_KEYS_MAPPING = {
    "beit": {},
    "bert": {},
    "xlm_roberta": {},
}
SUBMODEL_NAMES = {
    "clip": ["vision_config", "text_config"],
    "encoder-decoder": ["encoder", "decoder"],
}


def wrap_config(config: PretrainedConfig):
    # Make sure each class has its own attribute_map
    type(config).attribute_map = copy.deepcopy(type(config).attribute_map)
    # Ensure missing keys are in class
    if config.model_type in CONFIG_CLASS_KEYS_MAPPING:
        for key, value in CONFIG_CLASS_KEYS_MAPPING[config.model_type].items():
            if key not in config.attribute_map:
                config.attribute_map[key] = value


def init_adapters_config(
    model: PreTrainedModel,
    model_config: PretrainedConfig,
    adapters_config: Optional[ModelAdaptersConfig] = None,
):
    # Make sure config is wrapped
    model.config = model_config
    wrap_config(model.config)

    # Init ModelAdaptersConfig
    if adapters_config is not None:
        model.adapters_config = adapters_config
    elif not hasattr(model_config, "adapters"):
        model.adapters_config = ModelAdaptersConfig()
    elif model_config.adapters is not None and not isinstance(
        model_config.adapters, ModelAdaptersConfig
    ):
        model.adapters_config = ModelAdaptersConfig(**model_config.adapters)
    if hasattr(model, "base_model") and model.base_model is not model:
        model.base_model.adapters_config = model.adapters_config

    # Convert AdapterFusions from old format for backwards compatibility
    fusion_models = getattr(model_config, "adapter_fusion_models", [])
    fusion_config = getattr(model_config, "adapter_fusion", None)
    for fusion_adapter_names in fusion_models:
        model.adapters_config.add_fusion(fusion_adapter_names, config=fusion_config)


class AdapterMethod:
    bottleneck = "bottleneck"
    prefix_tuning = "prefix_tuning"
    lora = "lora"
    prompt_tuning = "prompt_tuning"
    reft = "reft"
    invertible = "invertible"

    @staticmethod
    def get_from_config(config) -> List[str]:
        methods = []
        if getattr(config, "inv_adapter", False):
            methods.append(AdapterMethod.invertible)
        if config.architecture is None:
            methods.append(AdapterMethod.bottleneck)
        elif config.architecture == "union":
            for sub_config in config.configs:
                methods.extend(AdapterMethod.get_from_config(sub_config))
        else:
            methods.append(config.architecture)
        return methods


@dataclass
class AdapterModelInterface:
    adapter_methods: List[str]

    model_embeddings: str
    model_layers: str

    layer_self_attn: str
    layer_cross_attn: str

    attn_o_proj: Optional[str]

    layer_intermediate_proj: str
    layer_output_proj: str

    ###
    # Either all of these (this is the default and best working implementation):
    attn_k_proj: Optional[str] = None
    attn_q_proj: Optional[str] = None
    attn_v_proj: Optional[str] = None

    # Or this (for when query, key and value are stored in the same tensor as in GPT2 or ModernBERT):
    attn_qkv_proj: Optional[str] = None
    ###

    # Optional attributes for extended bottleneck adapter support
    layer_pre_self_attn: Optional[str] = None
    layer_pre_cross_attn: Optional[str] = None
    layer_pre_ffn: Optional[str] = None
    layer_ln_1: Optional[str] = None
    layer_ln_2: Optional[str] = None

    base_model: Optional[str] = None

    def to_dict(self):
        return asdict(self)

    def __post_init__(self):
        """Validate projection attributes after initialization."""

        has_separate_projections = (
            self.attn_k_proj is not None
            and self.attn_q_proj is not None
            and self.attn_v_proj is not None
        )
        has_combined_projection = self.attn_qkv_proj is not None

        if not has_separate_projections and not has_combined_projection:
            raise ValueError(
                "Must specify either individual projections (k,q,v) layers or combined qkv projection layer. You currently are neither specifying attn_qkv_proj nor attn_k_proj, attn_q_proj and attn_v_proj."
            )

        if has_separate_projections and has_combined_projection:
            raise ValueError(
                "Cannot specify both individual projections (k,q,v) and combined qkv projection. You specified attn_qkv_proj as well as attn_k_proj, attn_q_proj and attn_v_proj which makes no sense."
            )

    def _save(self, save_directory, model_config):
        config_dict = {
            "model_type": model_config.model_type,
            "interface": self.to_dict(),
            "version": "adapters." + __version__,
        }
        save_path = os.path.join(save_directory, INTERFACE_CONFIG_NAME)
        with open(save_path, "w") as f:
            json.dump(config_dict, f, indent=2, sort_keys=True)

    @classmethod
    def _load(cls, path_or_repo_id: str, **kwargs):
        resolved_file = cached_file(path_or_repo_id, INTERFACE_CONFIG_NAME, **kwargs)
        with open(resolved_file, "r") as f:
            config_dict = json.load(f)
        return AdapterModelInterface(**config_dict["interface"])


class InvertibleAdaptersMixin:
    """Mixin for Transformer models adding invertible adapters."""

    def init_adapters(self, model_config, adapters_config, **kwargs):

        self.invertible_adapters = nn.ModuleDict(dict())

        init_adapters_config(self, model_config, adapters_config)

        if hasattr(super(), "init_adapters"):
            super().init_adapters(self.config, self.adapters_config, **kwargs)

    def add_invertible_adapter(self, adapter_name: str) -> bool:
        """
        Adds an invertible adapter module for the adapter with the given name. If the given adapter does not specify an
        invertible adapter config, this method does nothing.

        Args:
            adapter_name (str): The name of the adapter for which to add an invertible adapter module.
        """
        if adapter_name in self.invertible_adapters:
            raise ValueError(
                f"Model already contains an adapter module for '{adapter_name}'."
            )
        embedding_size = getattr(self.config, "embedding_size", self.config.hidden_size)
        adapter_config = self.adapters_config.match(
            adapter_name,
            config_type=BnConfig,
            location_key="inv_adapter",
        )

        return False

    def invertible_adapters_forward(self, hidden_states, rev=False):
        # TODO: Currently no fusion over invertible adapters, takes only very first language adapter position
        adapter_setup = self._get_active_setup()
        if adapter_setup is not None and len(adapter_setup) > 0:
            first_adapter = adapter_setup.first()
            if first_adapter in self.invertible_adapters:
                hidden_states = self.invertible_adapters[first_adapter](
                    hidden_states, rev=rev
                )
        return hidden_states

    def _get_active_setup(self):
        if hasattr(self, "adapters_config"):
            # First check current context before falling back to defined setup
            context = AdapterSetup.get_context()
            if context is not None:
                adapter_setup = context.adapter_setup
            else:
                adapter_setup = self.adapters_config.active_setup
        else:
            adapter_setup = None
        if adapter_setup is not None and (len(adapter_setup.flatten()) > 0):
            return adapter_setup
        else:
            return None


class EmbeddingAdaptersMixin:
    """Mixin for Transformer models adding support for dynamically switching embeddings."""

    def init_adapters(self, model_config, adapters_config, **kwargs):
        self.loaded_embeddings = {}
        self._active_embedding = "default"

        init_adapters_config(self, model_config, adapters_config)

        super().init_adapters(self.config, self.adapters_config, **kwargs)

    @property
    def active_embeddings(self):
        return self._active_embedding


class ModelAdaptersMixin(ABC):
    """Mixin for transformer models adding support for loading/ saving adapters."""

    add_base_adapters = False
    support_lora_delta_w_svd = True  # If True, the model supports the "lora_delta_w_svd" combine_strategy to merge adapter weights.
    support_prompt_tuning = True  # If False, the prompt tuning layer is not added to the model. If True, the prompt tuning layer is added if add_base_adapters is True.

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)

    def _link_prefix_to_pool(self, layer):
        if isinstance(layer, PrefixTuningLayer):
            layer.set_pool(self.base_model.prefix_tuning)

    def _add_tied_weights_keys(self):
        """Internal method to add adapter-specific keys to the list of tied weights keys."""
        if self.base_model.support_prompt_tuning:
            prompt_tied_weights_keys = ["prompt_tuning.base_model_embeddings.*"]
            if self._tied_weights_keys is not None:
                self._tied_weights_keys += prompt_tied_weights_keys
            else:
                self._tied_weights_keys = prompt_tied_weights_keys

    @property
    def model_name(self):
        return self.config.name_or_path

    def _init_adapters_submodules(self, model_config, adapters_config):
        # Initialize adapters in all submodules
        for module in self.modules():
            # skip calling module
            if module == self:
                continue
            if hasattr(module, "init_adapters"):
                module.init_adapters(model_config, adapters_config)

    def _default_init_adapter_methods(self, model_config, adapters_config):
        init_reft(self.base_model)
        # Add prefix tuning
        self.base_model.prefix_tuning = PrefixTuningPool(model_config, adapters_config)
        # Add Prompt Tuning
        # breakpo .int()
        if self.add_base_adapters:
            if self.support_prompt_tuning:
                self.prompt_tuning = PromptTuningLayer(
                    model_config, adapters_config, self.get_input_embeddings()
                )

    def init_adapters(self, model_config, adapters_config):
        """
        This method initializes adapter modules and fusion modules from the model config.
        """
        self.base_model.shared_parameters = nn.ModuleDict()

        # Initialize adapters config
        init_adapters_config(self, model_config, adapters_config)

        self._default_init_adapter_methods(self.config, self.adapters_config)

        # Initialize adapters in all submodules
        self._init_adapters_submodules(self.config, self.adapters_config)

        # Link all prefix tunings
        if hasattr(self.base_model, "prefix_tuning"):
            self.apply_to_adapter_layers(
                lambda i, layer: self._link_prefix_to_pool(layer)
            )

        # Initialize adapters from config
        for adapter_name in self.adapters_config:
            self._add_adapter_weights(adapter_name)
        # Initialize fusion from config
        for fusion_name in self.adapters_config.fusions:
            self.apply_to_adapter_layers(
                lambda i, layer: layer.add_fusion_layer(fusion_name)
            )

        if isinstance(self, EmbeddingAdaptersMixin):
            self.loaded_embeddings["default"] = self.get_input_embeddings()

        self._add_tied_weights_keys()

    def supports_adapter(self, type_or_config: Union[str, AdapterConfig]) -> bool:
        """
        Checks if the model supports a given adapter type.

        Args:
            adapter_type (str): The adapter type to check.

        Returns:
            bool: True if the adapter type is supported, False otherwise.
        """
        if isinstance(type_or_config, AdapterConfig):
            types = AdapterMethod.get_from_config(type_or_config)
        else:
            types = [type_or_config]

        supported = []
        for _type in types:
            if getattr(self.base_model, "adapter_interface", None) is not None:
                supported.append(
                    _type in self.base_model.adapter_interface.adapter_methods
                )
            elif _type == AdapterMethod.prompt_tuning:
                supported.append(self.base_model.support_prompt_tuning)
            elif _type == AdapterMethod.invertible:
                supported.append(isinstance(self, InvertibleAdaptersMixin))
            else:
                supported.append(True)
        return all(supported)

    # These methods have to be implemented by every deriving class:

    @abstractmethod
    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        """
        Iterates over all layers of the model.

        This abstract method has to ne implemented by every implementing model.
        """
        pass

    def apply_to_adapter_layers(self, fn):
        """
        Applies a function to all adapter layers of the model.
        """
        for i, layer in self.iter_layers():
            for module in layer.modules():
                if isinstance(module, AdapterLayerBase):
                    fn(i, module)

    def apply_to_basemodel_childs(self, fn):
        """
        Applies a function to all direct childs of the model if they are a instance of AdapterLayerBase.
        """
        if self.base_model.add_base_adapters:
            for module in self.base_model.children():
                if isinstance(module, AdapterLayerBase):
                    # These childs don't have a layer index so we pass -1
                    fn(-1, module)

    def has_adapters(self):
        return len(self.adapters_config.adapters) > 0

    @property
    def has_parallel_adapters(self) -> bool:
        if self.adapters_config.active_setup:
            return self.adapters_config.active_setup.parallel_channels > 1
        else:
            return False

    @property
    def active_adapters(self) -> AdapterCompositionBlock:
        return self.adapters_config.active_setup

    @active_adapters.setter
    def active_adapters(self, adapter_setup: Union[list, AdapterCompositionBlock]):
        self.set_active_adapters(adapter_setup)

    def set_active_adapters(
        self,
        adapter_setup: Union[list, AdapterCompositionBlock],
        skip_layers: Optional[List[int]] = None,
    ):
        adapter_setup = parse_composition(
            adapter_setup, model_type=self.config.model_type
        )
        if adapter_setup:
            for adapter_name in adapter_setup.flatten():
                if adapter_name not in self.adapters_config.adapters:
                    raise ValueError(
                        f"No adapter with name '{adapter_name}' found. Please make sure that all specified adapters"
                        " are correctly loaded."
                    )

        # Make sure LoRA is reset
        self.reset_adapter()
        self.adapters_config.active_setup = adapter_setup
        self.adapters_config.skip_layers = skip_layers

    def add_adapter(
        self,
        adapter_name: str,
        config=None,
        overwrite_ok: bool = False,
        set_active: bool = False,
    ):
        config = AdapterConfig.load(config)  # ensure config is ok and up-to-date
        # check if config is valid for this model
        config_or_type = config or AdapterMethod.bottleneck
        if not self.supports_adapter(config_or_type):
            raise ValueError(
                f"Adapter config or type '{config_or_type}' is not supported by this model."
            )
        # In case adapter already exists and we allow overwriting, explicitly delete the existing one first
        if overwrite_ok and adapter_name in self.adapters_config:
            self.delete_adapter(adapter_name)
        self.adapters_config.add(adapter_name, config=config)
        try:
            self._add_adapter_weights(adapter_name)
        except ValueError as ex:
            self.delete_adapter(adapter_name)
            raise ex
        if set_active:
            self.set_active_adapters(adapter_name)

        # For VeRA adapters, register tied weights patterns
        if self.adapters_config.match(adapter_name, LoRAConfig):
            adapter_config = self.adapters_config.match(adapter_name, LoRAConfig)
            if isinstance(adapter_config.vera_d, float) or isinstance(
                adapter_config.vera_b, float
            ):
                vera_tied_weights_keys = [
                    f"shared_parameters\\.{adapter_name}\\.lora_A",
                    f"shared_parameters\\.{adapter_name}\\.lora_B",
                ]

                if self._tied_weights_keys is not None:
                    self._tied_weights_keys += vera_tied_weights_keys
                else:
                    self._tied_weights_keys = vera_tied_weights_keys

    def _add_adapter_weights(self, adapter_name: str):
        """Helper method that performs the actual parameter additions when adding a new adapter."""
        self.apply_to_adapter_layers(
            lambda i, layer: layer.add_adapter(adapter_name, i)
        )
        self.apply_to_basemodel_childs(
            lambda i, child: child.add_adapter(adapter_name, i)
        )

        # PHM Layer
        if self.adapters_config.match(adapter_name, BnConfig, location_key="phm_layer"):
            adapter_config = self.adapters_config.match(
                adapter_name, BnConfig, location_key="phm_layer"
            )
            if adapter_config["shared_phm_rule"] or adapter_config["shared_W_phm"]:
                if self.config.model_type in SUBMODEL_NAMES:
                    hidden_sizes = [
                        getattr(self.config, key).hidden_size
                        for key in SUBMODEL_NAMES[self.config.model_type]
                    ]
                    if all(hidden_sizes[0] == h for h in hidden_sizes):
                        self.base_model.shared_parameters[
                            adapter_name
                        ] = init_shared_parameters(
                            adapter_config, hidden_sizes[0], self.device
                        )
                    else:
                        raise ValueError(
                            "The model has different hidden sizes {}. Sharing compacter weights is only possible if"
                            " the hidden_sizes match.".format(hidden_sizes)
                        )
                else:
                    self.base_model.shared_parameters[
                        adapter_name
                    ] = init_shared_parameters(
                        adapter_config, self.config.hidden_size, self.device
                    )

        # Vera Initialization
        if self.adapters_config.match(adapter_name, LoRAConfig):
            # in above line - we need to check for LoRAConfig since adapter reinitilization
            # depends on the architecture field of the adapter config
            adapter_config = self.adapters_config.match(adapter_name, LoRAConfig)
            if isinstance(adapter_config.vera_d, float) or isinstance(
                adapter_config.vera_b, float
            ):
                # First, we need to check that the hidden size is the same for all submodels
                if self.config.model_type in SUBMODEL_NAMES:
                    hidden_sizes = [
                        getattr(self.config, key).hidden_size
                        for key in SUBMODEL_NAMES[self.config.model_type]
                    ]
                    if not (all(hidden_sizes[0] == h for h in hidden_sizes)):
                        raise ValueError(
                            "The model has different hidden sizes {}. Vera uses shared LoRA A and B matrices and thus initialization is only possible if the hidden_sizes match.".format(
                                hidden_sizes
                            )
                        )

                # Next, init the shared parameters of Vera
                shapes_info = self.adapters_config._vera_init_shapes[adapter_name]
                lora_A_shape = shapes_info["lora_A_shape"]
                lora_B_shape = shapes_info["lora_B_shape"]
                self.base_model.shared_parameters[
                    adapter_name
                ] = init_shared_vera_parameters(
                    lora_A_shape, lora_B_shape, adapter_config, self.device
                )

        # Prefix Tuning
        for module in self.modules():
            if isinstance(module, PrefixTuningPool):
                module.confirm_prefix(adapter_name)
        if isinstance(self, InvertibleAdaptersMixin):
            self.add_invertible_adapter(adapter_name)

    def freeze_model(self, freeze=True):
        """Freezes all weights of the model."""
        # first freeze/ unfreeze all model weights
        for param in self.base_model.parameters():
            param.requires_grad = not freeze
        self.model_frozen = freeze

    def forward_context(self, context: ForwardContext, *args, **kwargs):
        """
        This method is called by the ``ForwardContext`` at the beginning of the forward pass.
        """
        if "task_ids" in kwargs:
            context.task_ids = kwargs.pop("task_ids")

        # some warnings if we don't use available adapters
        active_adapters = (
            getattr(self, "active_adapters", None)
            or AdapterSetup.get_context_adapter_setup()
        )
        if not active_adapters:
            if self.has_adapters():
                logger.warning(
                    "There are adapters available but none are activated for the forward pass."
                )
            return

        context.adapters_parallelized = False
        # Check if already parallelized in encoder
        if context.adapter_input_parallelized:
            if active_adapters.parallel_channels > 1:
                context.adapters_parallelized = True
        # Add the shared parameters for the active adapters to the context
        context.shared_parameters = {
            name: param
            for name, param in self.base_model.shared_parameters.items()
            if name in active_adapters.flatten()
        }

        if hasattr(self.base_model, "prefix_tuning"):
            context.prefix_states = self.base_model.prefix_tuning(*args, **kwargs)

        # Read out offsets & seqlens from attention mask
        if "attention_mask" in kwargs:
            attention_mask = kwargs["attention_mask"]
        elif len(args) > 1:
            attention_mask = args[1]
        else:
            attention_mask = None
        if attention_mask is not None:
            context.seqlens = (attention_mask == 1).sum(dim=-1).squeeze()
            # return the first "1" in each row of the attention mask
            context.offsets = attention_mask.argmax(1)

        # Adapter gating and attention outputs
        context.adapter_gating_scores = defaultdict(dict)
        context.adapter_fusion_attentions = defaultdict(dict)

    def reset_adapter(self):
        """
        Resets weights of a LoRA module merged using `model.merge_adapter(name)`.
        """
        with ForwardContext(self, torch.empty(0, 1)):
            if self.base_model.shared_parameters:
                ForwardContext.get_context().shared_parameters = (
                    self.base_model.shared_parameters
                )

            for module in self.modules():
                if isinstance(module, LoRALayer):
                    module.reset_adapter()


@inherit_doc
class ModelBaseAdaptersMixin(ModelAdaptersMixin):
    adapter_interface: AdapterModelInterface = None
    add_base_adapters = True

    def init_adapters(self, model_config, adapters_config):
        super().init_adapters(model_config, adapters_config)

        patch_forward(self)

    # Adapter Interface Methods

    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        for i, layer in enumerate(
            multigetattr(self, self.adapter_interface.model_layers)
        ):
            yield i, layer

    def get_layer(self, idx: int) -> nn.Module:
        return multigetattr(self, self.adapter_interface.model_layers)[idx]

    def iter_attentions(
        self,
    ) -> Iterable[Tuple[int, Literal["self", "cross"], nn.Module]]:
        for i, layer in self.iter_layers():
            if multihasattr(layer, self.adapter_interface.layer_self_attn or ""):
                yield i, "self", multigetattr(
                    layer, self.adapter_interface.layer_self_attn
                )
            if multihasattr(layer, self.adapter_interface.layer_cross_attn or ""):
                yield i, "cross", multigetattr(
                    layer, self.adapter_interface.layer_cross_attn
                )

    def iter_layer_ffns(
        self,
    ) -> Iterable[Tuple[int, Literal["intermediate", "output"], nn.Module]]:
        for i, layer in self.iter_layers():
            if intermediate_proj := multigetattr(
                layer, self.adapter_interface.layer_intermediate_proj
            ):
                yield i, "intermediate", intermediate_proj
            if output_proj := multigetattr(
                layer, self.adapter_interface.layer_output_proj
            ):
                yield i, "output", output_proj

    def post_embedding_forward(self, module, args, embedding_output):
        if isinstance(self, InvertibleAdaptersMixin):
            embedding_output = self.invertible_adapters_forward(embedding_output)

        embedding_output = self.prompt_tuning.forward(embedding_output)

        return embedding_output

    @ForwardContext.wrap_base
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)


class BertSelfAttentionAdaptersMixin:
    """Adds adapters to the BertSelfAttention module."""

    def init_adapters(self, model_config, adapters_config):
        # Wrap layers for LoRA
        self.query = LoRALinear.wrap(
            self.query, "selfattn", model_config, adapters_config, attn_key="q"
        )
        self.key = LoRALinear.wrap(
            self.key, "selfattn", model_config, adapters_config, attn_key="k"
        )
        self.value = LoRALinear.wrap(
            self.value, "selfattn", model_config, adapters_config, attn_key="v"
        )

        self.prefix_tuning = PrefixTuningLayer(
            self.location_key + "_prefix" if self.location_key else None,
            model_config,
            adapters_config,
        )
        patch_forward(self)


# For backwards compatibility, BertSelfOutput inherits directly from BottleneckLayer
class BertSelfOutputAdaptersMixin(BottleneckLayer):
    """Adds adapters to the BertSelfOutput module."""

    def __init__(self):
        super().__init__("mh_adapter")

    def init_adapters(self, model_config, adapters_config):
        self.location_key = "mh_adapter"
        super().init_adapters(model_config, adapters_config)
        patch_forward(self)


# For backwards compatibility, BertOutput inherits directly from BottleneckLayer
class BertOutputAdaptersMixin(BottleneckLayer):
    """Adds adapters to the BertOutput module."""

    def __init__(self):
        super().__init__("output_adapter")

    def init_adapters(self, model_config, adapters_config):
        self.location_key = "output_adapter"
        super().init_adapters(model_config, adapters_config)
        patch_forward(self)


class BertLayerAdaptersMixin:
    """Adds adapters to the BertLayer module."""

    def init_adapters(self, model_config, adapters_config):
        # Wrap layers for LoRA
        self.intermediate.dense = LoRALinear.wrap(
            self.intermediate.dense, "intermediate", model_config, adapters_config
        )
        self.output.dense = LoRALinear.wrap(
            self.output.dense, "output", model_config, adapters_config
        )

        # Set location keys for prefix tuning
        self.attention.self.location_key = "self"
        if hasattr(self, "add_cross_attention") and self.add_cross_attention:
            self.crossattention.self.location_key = "cross"


class BertModelAdaptersMixin(
    EmbeddingAdaptersMixin, InvertibleAdaptersMixin, ModelBaseAdaptersMixin
):
    """Adds adapters to the BertModel module."""

    def init_adapters(self, model_config, adapters_config):
        super().init_adapters(model_config, adapters_config)

        # Set hook for parallel composition
        for _, layer in self.iter_layers():
            self._set_layer_hook_for_parallel(layer)

        # Register hook for post embedding forward
        self.embeddings.register_forward_hook(self.post_embedding_forward)

    def _set_layer_hook_for_parallel(self, layer: nn.Module):
        def hook(module, input):
            adjust_tensors_for_parallel_(input[0], input[1])
            return input

        layer.register_forward_pre_hook(hook)

    def iter_layers(self) -> Iterable[Tuple[int, nn.Module]]:
        for i, layer in enumerate(self.encoder.layer):
            yield i, layer


# Copied from transformers.models.roberta.modeling_roberta.RobertaSelfAttention with Roberta->XLMRoberta
class XLMRobertaSelfAttentionWithAdapters(
    BertSelfAttentionAdaptersMixin, XLMRobertaSelfAttention
):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        attention_mask = prefix_attention_mask(attention_mask)  # type: ignore

        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)
        # >>> START AH Changes <<<
        query_layer, key_layer, value_layer = match_attn_matrices_for_parallel(
            query_layer, key_layer, value_layer
        )
        (attention_mask,) = adjust_tensors_for_parallel(query_layer, attention_mask)
        # >>> END AH Changes <<<

        use_cache = past_key_value is not None
        if self.is_decoder:
            past_key_value = (key_layer, value_layer)

        # >>> START AH Changes <<<
        key_layer, value_layer, attention_mask = self.prefix_tuning(
            key_layer, value_layer, hidden_states, attention_mask
        )
        (query_layer,) = adjust_tensors_for_parallel(key_layer, query_layer)
        # >>> END AH Changes <<<

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if (
            self.position_embedding_type == "relative_key"
            or self.position_embedding_type == "relative_key_query"
        ):
            query_length, key_length = query_layer.shape[2], key_layer.shape[2]
            if use_cache:
                position_ids_l = torch.tensor(
                    key_length - 1, dtype=torch.long, device=hidden_states.device
                ).view(-1, 1)
            else:
                position_ids_l = torch.arange(
                    query_length, dtype=torch.long, device=hidden_states.device
                ).view(-1, 1)
            position_ids_r = torch.arange(
                key_length, dtype=torch.long, device=hidden_states.device
            ).view(1, -1)
            distance = position_ids_l - position_ids_r

            positional_embedding = self.distance_embedding(
                distance + self.max_position_embeddings - 1
            )
            positional_embedding = positional_embedding.to(
                dtype=query_layer.dtype
            )  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum(
                    "bhld,lrd->bhlr", query_layer, positional_embedding
                )
                relative_position_scores_key = torch.einsum(
                    "bhrd,lrd->bhlr", key_layer, positional_embedding
                )
                attention_scores = (
                    attention_scores
                    + relative_position_scores_query
                    + relative_position_scores_key
                )

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in XLMRobertaModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        outputs = (
            (context_layer, attention_probs) if output_attentions else (context_layer,)
        )

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class XLMRobertaSdpaSelfAttentionWithAdapters(
    BertSelfAttentionAdaptersMixin, XLMRobertaSdpaSelfAttention
):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # >>> START AH Changes <<<
        attention_mask = prefix_attention_mask(attention_mask, [2, 3])  # type: ignore
        # >>> END AH Changes <<<

        if (
            self.position_embedding_type != "absolute"
            or output_attentions
            or head_mask is not None
        ):
            return super().forward(
                hidden_states,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )

        bsz, tgt_len, _ = hidden_states.size()

        query_layer = self.transpose_for_scores(self.query(hidden_states))

        # If this is instantiated as a cross-attention module, the keys and values come from an encoder; the attention
        # mask needs to be such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        current_states = encoder_hidden_states if is_cross_attention else hidden_states
        attention_mask = (
            encoder_attention_mask if is_cross_attention else attention_mask
        )

        # Check `seq_length` of `past_key_value` == `len(current_states)` to support prefix tuning
        if (
            is_cross_attention
            and past_key_value
            and past_key_value[0].shape[2] == current_states.shape[1]
        ):
            key_layer, value_layer = past_key_value
        else:
            key_layer = self.transpose_for_scores(self.key(current_states))
            value_layer = self.transpose_for_scores(self.value(current_states))
            if past_key_value is not None and not is_cross_attention:
                key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
                value_layer = torch.cat([past_key_value[1], value_layer], dim=2)

        # >>> START AH Changes <<<
        query_layer, key_layer, value_layer = match_attn_matrices_for_parallel(
            query_layer, key_layer, value_layer
        )
        (attention_mask,) = adjust_tensors_for_parallel(query_layer, attention_mask)
        # >>> END AH Changes <<<

        if self.is_decoder:
            past_key_value = (key_layer, value_layer)

        # >>> START AH Changes <<<
        key_layer, value_layer, attention_mask = self.prefix_tuning(
            key_layer, value_layer, hidden_states, attention_mask
        )
        (query_layer,) = adjust_tensors_for_parallel(key_layer, query_layer)
        bsz = query_layer.size(0)
        if (
            self.require_contiguous_qkv
            and query_layer.device.type == "cuda"
            and attention_mask is not None
        ):
            query_layer = query_layer.contiguous()
            key_layer = key_layer.contiguous()
            value_layer = value_layer.contiguous()
        is_causal = (
            True
            if self.is_decoder
            and not is_cross_attention
            and attention_mask is None
            and tgt_len > 1
            else False
        )

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_layer,
            key_layer,
            value_layer,
            attn_mask=attention_mask,
            dropout_p=self.dropout_prob if self.training else 0.0,
            is_causal=is_causal,
        )

        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, self.all_head_size)

        outputs = (attn_output,)
        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


# Copied from transformers.models.roberta.modeling_roberta.RobertaSelfOutput with Roberta->XLMRoberta
class XLMRobertaSelfOutputWithAdapters(
    BertSelfOutputAdaptersMixin, XLMRobertaSelfOutput
):
    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.bottleneck_layer_forward(
            hidden_states, input_tensor, self.LayerNorm
        )
        return hidden_states


# Copied from transformers.models.roberta.modeling_roberta.RobertaOutput with Roberta->XLMRoberta
class XLMRobertaOutputWithAdapters(BertOutputAdaptersMixin, XLMRobertaOutput):
    def forward(
        self, hidden_states: torch.Tensor, input_tensor: torch.Tensor
    ) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.bottleneck_layer_forward(
            hidden_states, input_tensor, self.LayerNorm
        )
        return hidden_states


# IMPORTANT: Only add classes to this mapping that are not copied into the adapters package
MODEL_MIXIN_MAPPING = {
    "BertLayer": BertLayerAdaptersMixin,
    "BertModel": BertModelAdaptersMixin,
    "XLMRobertaLayer": BertLayerAdaptersMixin,
    "XLMRobertaModel": BertModelAdaptersMixin,
    "XmodLayer": BertLayerAdaptersMixin,
}


def replace_xlm_roberta_with_adapters(module: nn.Module) -> None:
    """Static replacement of XLM-RoBERTa classes with adapter versions."""
    # Import the adapter classes directly
    # Check if module is a base model class
    if module.__class__.__name__ in MODEL_MIXIN_MAPPING:
        # Create new wrapper model class
        model_class = type(
            module.__class__.__name__,
            (MODEL_MIXIN_MAPPING[module.__class__.__name__], module.__class__),
            {},
        )
        module.__class__ = model_class
    elif module.__class__.__module__.startswith("transformers.models"):
        # Static mapping for XLM-RoBERTa classes
        class_mapping = {
            "XLMRobertaSelfAttention": XLMRobertaSelfAttentionWithAdapters,  # MISSING - ADD THIS!
            "XLMRobertaSdpaSelfAttention": XLMRobertaSdpaSelfAttentionWithAdapters,
            "XLMRobertaSelfOutput": XLMRobertaSelfOutputWithAdapters,
            "XLMRobertaOutput": XLMRobertaOutputWithAdapters,
        }

        if module.__class__.__name__ in class_mapping:
            module.__class__ = class_mapping[module.__class__.__name__]


def init(
    model: PreTrainedModel,
    adapters_config: Optional[ModelAdaptersConfig] = None,
    interface: Optional[AdapterModelInterface] = None,
) -> None:
    if isinstance(model, ModelAdaptersMixin):
        return model

    submodules = list(model.modules())
    replace_xlm_roberta_with_adapters(submodules.pop(0))

    # Change the class of all child modules to their adapters class
    for module in submodules:
        replace_xlm_roberta_with_adapters(module)

    # Next, check if model class itself is not replaced and has an adapter-supporting base class
    if not isinstance(model, ModelAdaptersMixin):
        if hasattr(model, "base_model_prefix") and hasattr(
            model, model.base_model_prefix
        ):
            base_model = getattr(model, model.base_model_prefix)
            if isinstance(base_model, ModelAdaptersMixin):
                # HACK to preserve original forward method signature (e.g. for Trainer label names)
                temp_signature = ForwardContext.add_context_args_in_signature(
                    model.forward.__func__
                )
                # Create new wrapper model class
                model_class_name = model.__class__.__name__
                model_class = type(
                    model_class_name,
                    (
                        EmbeddingAdaptersWrapperMixin,
                        ModelWithHeadsAdaptersMixin,
                        model.__class__,
                    ),
                    {},
                )
                model.__class__ = model_class
                model.forward.__func__.__signature__ = temp_signature

    # Finally, initialize adapters
    model.init_adapters(model.config, adapters_config)


# The "layers" attributes in the configs below map from static head module names to flex head module names.
# In this context, "None" refers to a flex-head layer without weights (e.g. dropout, acts).
STATIC_TO_FLEX_HEAD_MAP = {
    # BERT
    "BertForSequenceClassification": {
        "config": {
            "head_type": "classification",
            "layers": 1,
            "activation_function": None,
            "use_pooler": True,
        },
        "layers": [None, "classifier"],
    },
    "BertForMultipleChoice": {
        "config": {
            "head_type": "multiple_choice",
            "layers": 1,
            "activation_function": None,
            "use_pooler": True,
        },
        "layers": [None, "classifier"],
    },
    "BertForTokenClassification": {
        "config": {
            "head_type": "tagging",
            "layers": 1,
            "activation_function": None,
        },
        "layers": [None, "classifier"],
    },
    "BertForQuestionAnswering": {
        "config": {
            "head_type": "question_answering",
            "layers": 1,
            "activation_function": None,
        },
        "layers": [None, "qa_outputs"],
    },
    "BertForMaskedLM": {
        "config": {
            "head_type": "masked_lm",
            "layers": 2,
            "activation_function": "gelu",
            "layer_norm": True,
            "bias": True,
        },
        "layers": [
            "cls.predictions.transform.dense",
            None,
            "cls.predictions.transform.LayerNorm",
            "cls.predictions.decoder",
        ],
    },
    "BertLMHeadModel": {
        "config": {
            "head_type": "causal_lm",
            "layers": 2,
            "activation_function": "gelu",
            "layer_norm": True,
            "bias": True,
        },
        "layers": [
            "cls.predictions.transform.dense",
            None,
            "cls.predictions.transform.LayerNorm",
            "cls.predictions.decoder",
        ],
    },
    # BertGeneration
    "BertGenerationDecoder": {
        "config": {
            "head_type": "causal_lm",
            "layers": 1,
            "activation_function": None,
            "bias": True,
        },
        "layers": [
            "lm_head.decoder",
        ],
    },
    # XLM-RoBERTa
    "XLMRobertaForSequenceClassification": {
        "config": {
            "head_type": "classification",
            "layers": 2,
            "activation_function": "tanh",
            "use_pooler": False,
        },
        "layers": [None, "classifier.dense", None, None, "classifier.out_proj"],
    },
    "XLMRobertaForMultipleChoice": {
        "config": {
            "head_type": "multiple_choice",
            "layers": 1,
            "activation_function": None,
            "use_pooler": True,
        },
        "layers": [None, "classifier"],
    },
    "XLMRobertaForTokenClassification": {
        "config": {
            "head_type": "tagging",
            "layers": 1,
            "activation_function": None,
        },
        "layers": [None, "classifier"],
    },
    "XLMRobertaForQuestionAnswering": {
        "config": {
            "head_type": "question_answering",
            "layers": 1,
            "activation_function": None,
        },
        "layers": [None, "qa_outputs"],
    },
    "XLMRobertaForMaskedLM": {
        "config": {
            "head_type": "masked_lm",
            "layers": 2,
            "activation_function": "gelu",
            "layer_norm": True,
            "bias": True,
        },
        "layers": ["lm_head.dense", None, "lm_head.layer_norm", "lm_head.decoder"],
    },
    "XLMRobertaForCausalLM": {
        "config": {
            "head_type": "causal_lm",
            "layers": 2,
            "activation_function": "gelu",
            "layer_norm": True,
            "bias": True,
        },
        "layers": ["lm_head.dense", None, "lm_head.layer_norm", "lm_head.decoder"],
    },
}


def _regex_list_rename_func(k, rename_list):
    for o, n in rename_list:
        new_k, count = re.subn(o, n, k)
        if count > 0:
            return new_k
    return k


def get_head_config_and_rename_list(
    model_class_name, head_name, label2id, num_labels=None, return_rename_func=True
):
    if label2id is None:
        logger.warning(
            "No valid map of labels in label2id. Falling back to default (num_labels=2). This may cause errors during"
            " loading!"
        )
        label2id = {"LABEL_" + str(i): i for i in range(2)}
    # num_labels is optional (e.g. for regression, when no map given)
    num_labels = num_labels or len(label2id)
    data = STATIC_TO_FLEX_HEAD_MAP[model_class_name]
    # copy config to keep original mapping untouched
    config = copy.deepcopy(data["config"])
    if config["head_type"] == "multiple_choice":
        config["num_choices"] = num_labels
        config["label2id"] = label2id
    elif config["head_type"] not in ["causal_lm", "masked_lm", "seq2seq_lm"]:
        config["num_labels"] = num_labels
        config["label2id"] = label2id
    # rename
    rename_list = []
    i = 0
    for name in data["layers"]:
        if name is not None:
            escaped_name = re.escape(name)
            rename_list.append(
                (rf"{escaped_name}\.(\S+)", f"heads.{head_name}.{i}.\\1")
            )
        i += 1
    if return_rename_func:
        rename_func = lambda k, rename_list=rename_list: _regex_list_rename_func(
            k, rename_list
        )

        return config, rename_func
    else:
        return config, {k: v for k, v in rename_list}


class WeightsLoaderHelper:
    """
    A class providing helper methods for saving and loading module weights.
    """

    def __init__(
        self,
        model,
        weights_name,
        config_name,
        use_safetensors: bool = False,
        safe_weights_name: Optional[str] = None,
    ):
        self.model = model
        self.weights_name = weights_name
        self.config_name = config_name
        self.use_safetensors = use_safetensors
        if use_safetensors and not safetensors_available:
            raise ValueError(
                "Safetensors package not available. Please install via `pip install safetensors`."
            )
        self.safe_weights_name = safe_weights_name or weights_name

    def state_dict(self, filter_func):
        return {k: v for (k, v) in self.model.state_dict().items() if filter_func(k)}

    def rename_state_dict(self, state_dict, *rename_funcs):
        new_state_dict = {}
        for k, v in state_dict.items():
            new_k = k
            for rename_func in rename_funcs:
                new_k = rename_func(new_k)
            new_state_dict[new_k] = v
        return new_state_dict

    def save_weights_config(self, save_directory, config, meta_dict=None):
        # add meta information if given
        if meta_dict:
            for k, v in meta_dict.items():
                if k not in config:
                    config[k] = v
        # save to file system
        output_config_file = join(save_directory, self.config_name)
        with open(output_config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, sort_keys=True)
        logger.info("Configuration saved in {}".format(output_config_file))

    def save_weights(self, save_directory, filter_func):
        if not exists(save_directory):
            mkdir(save_directory)
        else:
            assert isdir(
                save_directory
            ), "Saving path should be a directory where the module weights can be saved."

        # Get the state of all adapter modules for this task
        state_dict = self.state_dict(filter_func)
        # Save the adapter weights
        if self.use_safetensors:
            output_file = join(save_directory, self.safe_weights_name)
            save_file(state_dict, output_file)
        else:
            output_file = join(save_directory, self.weights_name)
            torch.save(state_dict, output_file)
        logger.info("Module weights saved in {}".format(output_file))

    def load_weights_config(self, save_directory):
        config_file = join(save_directory, self.config_name)
        logger.info("Loading module configuration from {}".format(config_file))
        # Load the config
        with open(config_file, "r", encoding="utf-8") as f:
            loaded_config = json.load(f)
        # For older versions translate the activation function to the new format
        if "version" not in loaded_config:
            if "config" in loaded_config and loaded_config["config"] is not None:
                if (
                    "non_linearity" in loaded_config["config"]
                    and loaded_config["config"]["non_linearity"] in ACTIVATION_RENAME
                ):
                    loaded_config["config"]["non_linearity"] = ACTIVATION_RENAME[
                        loaded_config["config"]["non_linearity"]
                    ]
        return loaded_config

    @staticmethod
    def _load_module_state_dict(module, state_dict, start_prefix=""):
        missing_keys = []
        unexpected_keys = []
        error_msgs = []

        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=""):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict,
                prefix,
                local_metadata,
                True,
                missing_keys,
                unexpected_keys,
                error_msgs,
            )
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + ".")

        load(module, prefix=start_prefix)

        if len(error_msgs) > 0:
            raise RuntimeError(
                "Error(s) in loading state_dict for {}:\n\t{}".format(
                    module.__class__.__name__, "\n\t".join(error_msgs)
                )
            )
        return missing_keys, unexpected_keys

    def load_weights(
        self,
        save_directory,
        filter_func,
        rename_func=None,
        loading_info=None,
        in_base_model=False,
    ):
        # Load the weights of the adapter
        try:
            if self.use_safetensors:
                weights_file = join(save_directory, self.safe_weights_name)
                if exists(weights_file):
                    state_dict = load_file(weights_file, device="cpu")
                else:
                    logger.info(
                        f"No safetensors file found in {save_directory}. Falling back to torch.load..."
                    )
                    weights_file = join(save_directory, self.weights_name)
                    state_dict = torch.load(
                        weights_file, map_location="cpu", weights_only=True
                    )
            else:
                weights_file = join(save_directory, self.weights_name)
                state_dict = torch.load(
                    weights_file, map_location="cpu", weights_only=True
                )
        except Exception:
            raise OSError("Unable to load weights from pytorch checkpoint file. ")
        logger.info("Loading module weights from {}".format(weights_file))

        return self.load_weights_from_state_dict(
            state_dict,
            filter_func,
            rename_func=rename_func,
            loading_info=loading_info,
            in_base_model=in_base_model,
        )

    def load_weights_from_state_dict(
        self,
        state_dict,
        filter_func,
        rename_func=None,
        loading_info=None,
        in_base_model=False,
        start_prefix="",
    ):
        # Rename weights if needed
        if rename_func:
            if isinstance(rename_func, Sequence):
                state_dict = self.rename_state_dict(state_dict, *rename_func)
            else:
                state_dict = self.rename_state_dict(state_dict, rename_func)

        # Add the weights to the model
        # Make sure we are able to load base models as well as derived models (with heads)
        model_to_load = self.model
        has_prefix_module = any(
            s.startswith(self.model.base_model_prefix) for s in state_dict.keys()
        )
        if (
            not start_prefix
            and not hasattr(self.model, self.model.base_model_prefix)
            and has_prefix_module
        ):
            start_prefix = self.model.base_model_prefix + "."
        if (
            in_base_model
            and hasattr(self.model, self.model.base_model_prefix)
            and not has_prefix_module
        ):
            model_to_load = self.model.base_model

        missing_keys, unexpected_keys = self._load_module_state_dict(
            model_to_load, state_dict, start_prefix=start_prefix
        )

        missing_keys = [k for k in missing_keys if filter_func(k)]

        if len(missing_keys) > 0:
            logger.info(
                "Some module weights could not be found in loaded weights file: {}".format(
                    ", ".join(missing_keys)
                )
            )
        if self.model._keys_to_ignore_on_load_unexpected:
            unexpected_keys = [
                k
                for k in unexpected_keys
                if k not in self.model._keys_to_ignore_on_load_unexpected
            ]
        if len(unexpected_keys) > 0:
            logger.info(
                "Some weights of the state_dict could not be loaded into model: {}".format(
                    ", ".join(unexpected_keys)
                )
            )

        if isinstance(loading_info, dict):
            if "missing_keys" not in loading_info:
                loading_info["missing_keys"] = []
            if "unexpected_keys" not in loading_info:
                loading_info["unexpected_keys"] = []
            loading_info["missing_keys"].extend(missing_keys)
            loading_info["unexpected_keys"].extend(unexpected_keys)

        return missing_keys, unexpected_keys


class WeightsLoader(ABC):
    """
    An abstract class providing basic methods for saving and loading weights of a model. Extend this class to build
    custom module weight loaders.
    """

    def __init__(
        self,
        model,
        weights_name,
        config_name,
        use_safetensors: bool = False,
        safe_weights_name: Optional[str] = None,
    ):
        self.model = model
        self.weights_helper = WeightsLoaderHelper(
            model,
            weights_name,
            config_name,
            use_safetensors=use_safetensors,
            safe_weights_name=safe_weights_name,
        )

    def save(self, save_directory, name, **kwargs):
        if not exists(save_directory):
            mkdir(save_directory)
        else:
            assert isdir(
                save_directory
            ), "Saving path should be a directory where weights and configuration can be saved."

        config_dict = build_full_config(
            None,
            self.model.config,
            model_name=self.model.model_name,
            name=name,
            model_class=self.model.__class__.__name__,
        )
        meta_dict = kwargs.pop("meta_dict", None)

        # Save the adapter configuration
        self.weights_helper.save_weights_config(
            save_directory, config_dict, meta_dict=meta_dict
        )

        # Save adapter weights
        filter_func = self.filter_func(name)
        self.weights_helper.save_weights(save_directory, filter_func)

    def load(
        self, save_directory, load_as=None, loading_info=None, **kwargs
    ) -> Tuple[str, str]:
        if not exists(join(save_directory, self.weights_helper.weights_name)):
            raise ValueError(
                "Loading path should be a directory where the weights are saved."
            )

        # Load config
        config = self.weights_helper.load_weights_config(save_directory)

        # Load head weights
        filter_func = self.filter_func(config["name"])
        if load_as:
            rename_func = self.rename_func(config["name"], load_as)
        else:
            rename_func = None
        self.weights_helper.load_weights(
            save_directory,
            filter_func,
            rename_func=rename_func,
            loading_info=loading_info,
        )

        return save_directory, load_as or config["name"]


class AdapterLoader(WeightsLoader):
    """
    A class providing methods for saving and loading adapter modules from the Hub, the filesystem or a remote url.

    Model classes passed to this loader must implement the `ModelAdaptersMixin` class.
    """

    def __init__(self, model, adapter_type=None, use_safetensors: bool = False):
        super().__init__(
            model,
            WEIGHTS_NAME,
            CONFIG_NAME,
            use_safetensors=use_safetensors,
            safe_weights_name=SAFE_WEIGHTS_NAME,
        )
        self.adapter_type = adapter_type
        if adapter_type and not AdapterType.has(self.adapter_type):
            raise ValueError("Invalid adapter type {}".format(self.adapter_type))

    def filter_func(self, adapter_name):
        return (
            lambda x: "_adapters.{}.".format(adapter_name) in x
            or ".adapters.{}.".format(adapter_name) in x
            or ".prefix_tunings.{}.".format(adapter_name) in x
            or ".prefix_gates.{}.".format(adapter_name) in x
            or ".loras.{}.".format(adapter_name) in x
            or ".refts.{}.".format(adapter_name) in x
            or ".prompt_tunings.{}.".format(adapter_name) in x
            or ".shared_parameters.{}.".format(adapter_name) in x
        )

    # This dict maps the original weight names to the currently used equivalents.
    # The mapping is used by rename_func() to support loading from older weights files.
    # Old adapters will be loaded and converted to the new format automatically.
    legacy_weights_mapping = {
        "attention_text_task_adapters": "adapters",
        "attention_text_lang_adapters": "adapters",
        "layer_text_task_adapters": "adapters",
        "layer_text_lang_adapters": "adapters",
        "invertible_lang_adapters": "invertible_adapters",
    }

    def _rename_legacy_weights(self, k):
        for old, new in self.legacy_weights_mapping.items():
            k = k.replace(old, new)
        return k

    def _fix_backward_compat(self, config):
        # Fix error in previous versions for LoRA/ (IA)^3
        ADAPTER_PREFIX = "adapters."
        MIN_VERSION = Version("1.1.0")

        version = config.get("version", "")
        if (
            version.startswith(ADAPTER_PREFIX)
            and Version(version[len(ADAPTER_PREFIX) :]) < MIN_VERSION
        ):
            if (
                config["config"].get("architecture", None) == "lora"
                and config["config"]["r"] != config["config"]["alpha"]
            ):
                logger.warning(
                    "Loading a LoRA trained using a faulty scaling implementation of a previous library version. Editing the configuration to make sure the adapter works as trained."
                    "See https://github.com/adapter-hub/adapters/pull/770 for more."
                )
                config["config"]["alpha"] = config["config"]["r"]

    # This method is used to remove unnecessary invertible adapters from task adapters using the old format.
    # In the old format, task adapters e.g. using seq_bn config specify inv. adapters but don't use them.
    # As inv. adapters would be incorrectly used in the new implementation,
    # catch this case here when loading pretrained adapters.
    def _fix_legacy_config(self, adapter_name, missing_keys):
        if self.adapter_type == AdapterType.text_task:
            inv_adapter_keys = [
                x for x in missing_keys if f"invertible_adapters.{adapter_name}." in x
            ]
            if len(inv_adapter_keys) > 0:
                del self.model.base_model.invertible_adapters[adapter_name]
                missing_keys = [k for k in missing_keys if k not in inv_adapter_keys]
                # remove invertible_adapter from config
                adapter_config_name = self.model.adapters_config.adapters[adapter_name]
                if adapter_config_name in self.model.adapters_config.config_map:
                    adapter_config = self.model.adapters_config.config_map[
                        adapter_config_name
                    ]
                    self.model.adapters_config.config_map[
                        adapter_config_name
                    ] = adapter_config.replace(
                        inv_adapter=None, inv_adapter_reduction_factor=None
                    )
        return missing_keys

    def rename_func(self, old_name, new_name):
        return (
            lambda k: self._rename_legacy_weights(k)
            .replace("adapters.{}.".format(old_name), "adapters.{}.".format(new_name))
            .replace(
                ".prefix_tunings.{}.".format(old_name),
                ".prefix_tunings.{}.".format(new_name),
            )
            .replace(
                ".prefix_gates.{}.".format(old_name),
                ".prefix_gates.{}.".format(new_name),
            )
            .replace(".loras.{}.".format(old_name), ".loras.{}.".format(new_name))
            .replace(
                ".shared_parameters.{}.".format(old_name),
                ".shared_parameters.{}.".format(new_name),
            )
            .replace(".refts.{}.".format(old_name), ".refts.{}.".format(new_name))
        )

    def save_to_state_dict(self, name: str):
        if name not in self.model.adapters_config.adapters:
            raise ValueError(
                "No adapter of this type with the given name is part of this model."
            )

        adapter_config = self.model.adapters_config.get(name)

        config_dict = build_full_config(
            adapter_config,
            self.model.config,
            model_name=self.model.model_name,
            name=name,
            model_class=self.model.__class__.__name__,
        )

        # Save adapter weights
        filter_func = self.filter_func(config_dict["name"])
        state_dict = self.weights_helper.state_dict(filter_func)

        return state_dict, config_dict

    def save(self, save_directory, name, meta_dict=None):
        """
        Saves an adapter and its configuration file to a directory, so that it can be reloaded using the `load()`
        method.

        Args:
            save_directory (str): a path to a directory where the adapter will be saved
            task_name (str): the name of the adapter to be saved
        """
        if not exists(save_directory):
            mkdir(save_directory)
        else:
            assert isdir(
                save_directory
            ), "Saving path should be a directory where adapter and configuration can be saved."
        assert (
            name in self.model.adapters_config.adapters
        ), "No adapter of this type with the given name is part of this model."

        adapter_config = self.model.adapters_config.get(name)

        self.model.apply_to_adapter_layers(lambda _, layer: layer.pre_save_adapters())

        config_dict = build_full_config(
            adapter_config,
            self.model.config,
            model_name=self.model.model_name,
            name=name,
            model_class=self.model.__class__.__name__,
        )

        # Save the adapter configuration
        self.weights_helper.save_weights_config(
            save_directory, config_dict, meta_dict=meta_dict
        )

        # Save adapter weights
        filter_func = self.filter_func(config_dict["name"])
        self.weights_helper.save_weights(save_directory, filter_func)

    def load_from_state_dict(
        self, state_dict, name, load_as=None, loading_info=None, start_prefix=""
    ):
        new_adapter_name = load_as or name
        if new_adapter_name not in self.model.adapters_config.adapters:
            raise ValueError(
                "No adapter of this type with the given name is part of this model."
            )

        # Load adapter weights
        filter_func = self.filter_func(name)
        rename_func = self.rename_func(name, new_adapter_name)
        missing_keys, _ = self.weights_helper.load_weights_from_state_dict(
            state_dict,
            filter_func,
            rename_func=rename_func,
            loading_info=loading_info,
            in_base_model=True,
            start_prefix=start_prefix,
        )
        missing_keys = self._fix_legacy_config(new_adapter_name, missing_keys)
        if isinstance(loading_info, Mapping):
            loading_info["missing_keys"] = missing_keys

    def load(
        self,
        adapter_name_or_path,
        config=None,
        version=None,
        model_name=None,
        load_as=None,
        loading_info=None,
        leave_out=None,
        set_active=False,
        **kwargs,
    ):
        # Warn about deprecated arguments
        if config is not None or model_name is not None:
            logger.warning(
                "The 'config' and 'model_name' arguments are specific to the now unsupported legacy Hub repo and will"
                " be removed."
                "Please switch to only providing the HF Model Hub identifier.",
            )
        requested_config = AdapterConfig.load(config) if config else None
        # Resolve the weights to be loaded based on the given identifier and the current adapter config
        model_name = self.model.model_name or model_name
        resolved_folder = resolve_adapter_path(
            adapter_name_or_path,
            model_name,
            adapter_config=requested_config,
            version=version,
            **kwargs,
        )

        # Load config of adapter
        config = self.weights_helper.load_weights_config(resolved_folder)
        if self.adapter_type and "type" in config:
            assert (
                config["type"] == self.adapter_type
            ), "Loaded adapter has to be a {} adapter.".format(self.adapter_type)
        elif "type" in config:
            self.adapter_type = config["type"]
        # post-loading drop of layers
        if leave_out is not None:
            if (
                "leave_out" in config["config"]
                and config["config"]["leave_out"] is not None
            ):
                # The conversion to a set and then back to a list removes all duplicates
                leave_out = list(set(leave_out + config["config"]["leave_out"]))
            config["config"]["leave_out"] = leave_out
        # Fix issues
        self._fix_backward_compat(config)

        adapter_name = load_as or config["name"]
        # If the adapter is not part of the model, add it
        if adapter_name not in self.model.adapters_config.adapters:
            self.model.add_adapter(
                adapter_name, config=config["config"], set_active=set_active
            )
        else:
            logger.warning("Overwriting existing adapter '{}'.".format(adapter_name))

        # Load adapter weights
        filter_func = self.filter_func(adapter_name)
        rename_func = self.rename_func(config["name"], adapter_name)
        missing_keys, _ = self.weights_helper.load_weights(
            resolved_folder,
            filter_func,
            rename_func=rename_func,
            loading_info=loading_info,
            in_base_model=True,
        )
        missing_keys = self._fix_legacy_config(adapter_name, missing_keys)
        if isinstance(loading_info, Mapping):
            loading_info["missing_keys"] = missing_keys

        return resolved_folder, adapter_name
