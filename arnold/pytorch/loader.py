# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Arnold DQN model loader implementation.

Arnold is a Deep Q-Network (DQN) implementation for ViZDoom reinforcement learning.
This model processes screen images and game variables to predict Q-values for actions.
"""
from typing import Optional
from dataclasses import dataclass
import torch

from ...config import (
    ModelConfig,
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    StrEnum,
)
from ...base import ForgeModel
from ...tools.utils import get_file
from .src.dqn_module import DQNModuleFeedforward, DQNModuleRecurrent


@dataclass
class ArnoldDQNConfig(ModelConfig):
    """Configuration specific to Arnold DQN models"""

    # Model architecture parameters
    height: int = 60
    width: int = 108
    n_fm: int = 3  # number of feature maps (RGB channels)
    hist_size: int = 4  # history size (number of frames)
    hidden_dim: int = 512
    n_actions: int = 3
    use_bn: bool = False
    dropout: float = 0.0
    dueling_network: bool = False

    # Network architecture type
    network_type: str = "dqn_ff"  # "dqn_ff" for feedforward, "dqn_rnn" for recurrent
    recurrence: str = ""  # "lstm", "gru", or "rnn" for recurrent networks
    n_rec_layers: int = 1  # number of recurrent layers

    # Game variables (health, ammo)
    game_variables: list = None
    n_variables: int = 2
    variable_dim: list = None
    bucket_size: list = None

    # Game features (optional)
    game_features: str = ""

    # Pretrained model path (optional)
    pretrained_model_path: str = ""


class ModelVariant(StrEnum):
    """Available Arnold DQN model variants."""

    # Original variants (maintaining backward compatibility)
    DEFEND_THE_CENTER = "defend_the_center"
    HEALTH_GATHERING = "health_gathering"
    DEATHMATCH_SHOTGUN = "deathmatch_shotgun"
    VIZDOOM_2017_TRACK1 = "vizdoom_2017_track1"
    VIZDOOM_2017_TRACK2 = "vizdoom_2017_track2"
    DEFAULT = "default"

    # Variants with explicit architecture suffixes for all 5 weights
    DEFEND_THE_CENTER_FF = "defend_the_center_ff"
    DEFEND_THE_CENTER_RNN = "defend_the_center_rnn"
    HEALTH_GATHERING_FF = "health_gathering_ff"
    HEALTH_GATHERING_RNN = "health_gathering_rnn"
    DEATHMATCH_SHOTGUN_FF = "deathmatch_shotgun_ff"
    DEATHMATCH_SHOTGUN_RNN = "deathmatch_shotgun_rnn"
    VIZDOOM_2017_TRACK1_FF = "vizdoom_2017_track1_ff"
    VIZDOOM_2017_TRACK1_RNN = "vizdoom_2017_track1_rnn"
    VIZDOOM_2017_TRACK2_FF = "vizdoom_2017_track2_ff"
    VIZDOOM_2017_TRACK2_RNN = "vizdoom_2017_track2_rnn"


class ModelLoader(ForgeModel):
    """Arnold DQN model loader implementation."""

    # Dictionary of available model variants
    _VARIANTS = {
        ModelVariant.DEFEND_THE_CENTER: ArnoldDQNConfig(
            pretrained_model_name="arnold_dqn_defend_the_center",
            pretrained_model_path="s3://tt-ci-models-private/test_files/pytorch/Arnold/defend_the_center.pth",
            height=60,
            width=108,
            n_fm=3,
            hist_size=4,
            hidden_dim=512,
            n_actions=3,
            use_bn=False,
            dropout=0.0,
            dueling_network=False,
            network_type="dqn_ff",
            recurrence="",
            n_rec_layers=1,
            game_variables=[("health", 101), ("sel_ammo", 301)],
            n_variables=2,
            variable_dim=[32, 32],
            bucket_size=[1, 1],
            game_features="",
        ),
        ModelVariant.HEALTH_GATHERING: ArnoldDQNConfig(
            pretrained_model_name="arnold_dqn_health_gathering",
            pretrained_model_path="s3://tt-ci-models-private/test_files/pytorch/Arnold/health_gathering.pth",
            height=60,
            width=108,
            n_fm=3,
            hist_size=4,
            hidden_dim=512,
            n_actions=3,
            use_bn=False,
            dropout=0.0,
            dueling_network=False,
            network_type="dqn_ff",
            recurrence="",
            n_rec_layers=1,
            game_variables=[("health", 101)],
            n_variables=1,
            variable_dim=[32],
            bucket_size=[1],
            game_features="",
        ),
        ModelVariant.DEATHMATCH_SHOTGUN: ArnoldDQNConfig(
            pretrained_model_name="arnold_dqn_deathmatch_shotgun",
            pretrained_model_path="s3://tt-ci-models-private/test_files/pytorch/Arnold/deathmatch_shotgun.pth",
            height=60,
            width=108,
            n_fm=3,
            hist_size=6,
            hidden_dim=512,
            n_actions=3,
            use_bn=False,
            dropout=0.5,
            dueling_network=False,
            network_type="dqn_rnn",
            recurrence="lstm",
            n_rec_layers=1,
            game_variables=[("health", 101), ("sel_ammo", 301)],
            n_variables=2,
            variable_dim=[32, 32],
            bucket_size=[10, 1],
            game_features="target,enemy",
        ),
        ModelVariant.VIZDOOM_2017_TRACK1: ArnoldDQNConfig(
            pretrained_model_name="arnold_dqn_vizdoom_2017_track1",
            pretrained_model_path="s3://tt-ci-models-private/test_files/pytorch/Arnold/vizdoom_2017_track1.pth",
            height=60,
            width=108,
            n_fm=3,
            hist_size=4,
            hidden_dim=512,
            n_actions=3,
            use_bn=False,
            dropout=0.5,
            dueling_network=False,
            network_type="dqn_rnn",
            recurrence="lstm",
            n_rec_layers=1,
            game_variables=[("health", 101), ("sel_ammo", 301)],
            n_variables=2,
            variable_dim=[32, 32],
            bucket_size=[10, 1],
            game_features="target,enemy",
        ),
        ModelVariant.VIZDOOM_2017_TRACK2: ArnoldDQNConfig(
            pretrained_model_name="arnold_dqn_vizdoom_2017_track2",
            pretrained_model_path="s3://tt-ci-models-private/test_files/pytorch/Arnold/vizdoom_2017_track2.pth",
            height=60,
            width=108,
            n_fm=3,
            hist_size=4,
            hidden_dim=512,
            n_actions=3,
            use_bn=False,
            dropout=0.5,
            dueling_network=False,
            network_type="dqn_rnn",
            recurrence="lstm",
            n_rec_layers=1,
            game_variables=[("health", 101), ("sel_ammo", 301)],
            n_variables=2,
            variable_dim=[32, 32],
            bucket_size=[10, 1],
            game_features="target,enemy",
        ),
        ModelVariant.DEFAULT: ArnoldDQNConfig(
            pretrained_model_name="arnold_dqn_default",
            pretrained_model_path="s3://tt-ci-models-private/test_files/pytorch/Arnold/defend_the_center.pth",
            height=60,
            width=108,
            n_fm=3,
            hist_size=4,
            hidden_dim=512,
            n_actions=3,
            use_bn=False,
            dropout=0.0,
            dueling_network=False,
            network_type="dqn_ff",
            recurrence="",
            n_rec_layers=1,
            game_variables=[("health", 101), ("sel_ammo", 301)],
            n_variables=2,
            variable_dim=[32, 32],
            bucket_size=[1, 1],
            game_features="",
        ),
        # Variants with explicit architecture suffixes for all 5 weights
        # defend_the_center with both architectures
        ModelVariant.DEFEND_THE_CENTER_FF: ArnoldDQNConfig(
            pretrained_model_name="arnold_dqn_defend_the_center_ff",
            pretrained_model_path="s3://tt-ci-models-private/test_files/pytorch/Arnold/defend_the_center.pth",
            height=60,
            width=108,
            n_fm=3,
            hist_size=4,
            hidden_dim=512,
            n_actions=3,
            use_bn=False,
            dropout=0.0,
            dueling_network=False,
            network_type="dqn_ff",
            recurrence="",
            n_rec_layers=1,
            game_variables=[("health", 101), ("sel_ammo", 301)],
            n_variables=2,
            variable_dim=[32, 32],
            bucket_size=[1, 1],
            game_features="",
        ),
        ModelVariant.DEFEND_THE_CENTER_RNN: ArnoldDQNConfig(
            pretrained_model_name="arnold_dqn_defend_the_center_rnn",
            pretrained_model_path="s3://tt-ci-models-private/test_files/pytorch/Arnold/defend_the_center.pth",
            height=60,
            width=108,
            n_fm=3,
            hist_size=4,
            hidden_dim=512,
            n_actions=3,
            use_bn=False,
            dropout=0.5,
            dueling_network=False,
            network_type="dqn_rnn",
            recurrence="lstm",
            n_rec_layers=1,
            game_variables=[("health", 101), ("sel_ammo", 301)],
            n_variables=2,
            variable_dim=[32, 32],
            bucket_size=[10, 1],
            game_features="target,enemy",
        ),
        # health_gathering with both architectures
        ModelVariant.HEALTH_GATHERING_FF: ArnoldDQNConfig(
            pretrained_model_name="arnold_dqn_health_gathering_ff",
            pretrained_model_path="s3://tt-ci-models-private/test_files/pytorch/Arnold/health_gathering.pth",
            height=60,
            width=108,
            n_fm=3,
            hist_size=4,
            hidden_dim=512,
            n_actions=3,
            use_bn=False,
            dropout=0.0,
            dueling_network=False,
            network_type="dqn_ff",
            recurrence="",
            n_rec_layers=1,
            game_variables=[("health", 101)],
            n_variables=1,
            variable_dim=[32],
            bucket_size=[1],
            game_features="",
        ),
        ModelVariant.HEALTH_GATHERING_RNN: ArnoldDQNConfig(
            pretrained_model_name="arnold_dqn_health_gathering_rnn",
            pretrained_model_path="s3://tt-ci-models-private/test_files/pytorch/Arnold/health_gathering.pth",
            height=60,
            width=108,
            n_fm=3,
            hist_size=4,
            hidden_dim=512,
            n_actions=3,
            use_bn=False,
            dropout=0.5,
            dueling_network=False,
            network_type="dqn_rnn",
            recurrence="lstm",
            n_rec_layers=1,
            game_variables=[("health", 101)],
            n_variables=1,
            variable_dim=[32],
            bucket_size=[1],
            game_features="",
        ),
        # deathmatch_shotgun with both architectures
        # Note: Checkpoint has n_fm=4 (labels_mapping="0"), health vocab=11, n_actions=29
        ModelVariant.DEATHMATCH_SHOTGUN_FF: ArnoldDQNConfig(
            pretrained_model_name="arnold_dqn_deathmatch_shotgun_ff",
            pretrained_model_path="s3://tt-ci-models-private/test_files/pytorch/Arnold/deathmatch_shotgun.pth",
            height=60,
            width=108,
            n_fm=4,  # 3 RGB + 1 label map (labels_mapping="0")
            hist_size=1,  # For feedforward, if n_fm=4, hist_size should be 1 to match checkpoint [32, 4, 8, 8]
            hidden_dim=512,
            n_actions=29,  # From checkpoint: proj_action_scores.weight [29, 512]
            use_bn=False,
            dropout=0.0,
            dueling_network=False,
            network_type="dqn_ff",
            recurrence="",
            n_rec_layers=1,
            game_variables=[
                ("health", 11),
                ("sel_ammo", 301),
            ],  # health vocab=11 from checkpoint
            n_variables=2,
            variable_dim=[32, 32],
            bucket_size=[1, 1],
            game_features="",
        ),
        ModelVariant.DEATHMATCH_SHOTGUN_RNN: ArnoldDQNConfig(
            pretrained_model_name="arnold_dqn_deathmatch_shotgun_rnn",
            pretrained_model_path="s3://tt-ci-models-private/test_files/pytorch/Arnold/deathmatch_shotgun.pth",
            height=60,
            width=108,
            n_fm=4,  # 3 RGB + 1 label map (labels_mapping="0")
            hist_size=6,  # From run.sh: --hist_size 6
            hidden_dim=512,
            n_actions=29,  # From checkpoint: proj_action_scores.weight [29, 512]
            use_bn=False,
            dropout=0.5,
            dueling_network=False,
            network_type="dqn_rnn",
            recurrence="lstm",
            n_rec_layers=1,
            game_variables=[
                ("health", 11),
                ("sel_ammo", 301),
            ],  # health vocab=11 from checkpoint
            n_variables=2,
            variable_dim=[32, 32],
            bucket_size=[10, 1],
            game_features="target,enemy",
        ),
        # vizdoom_2017_track1 with both architectures
        # Note: Checkpoint has n_actions=29 (action_combinations="attack+move_lr;turn_lr;move_fb")
        ModelVariant.VIZDOOM_2017_TRACK1_FF: ArnoldDQNConfig(
            pretrained_model_name="arnold_dqn_vizdoom_2017_track1_ff",
            pretrained_model_path="s3://tt-ci-models-private/test_files/pytorch/Arnold/vizdoom_2017_track1.pth",
            height=60,
            width=108,
            n_fm=3,
            hist_size=4,
            hidden_dim=512,
            n_actions=29,  # From checkpoint: proj_action_scores.weight [29, 512]
            use_bn=False,
            dropout=0.0,
            dueling_network=False,
            network_type="dqn_ff",
            recurrence="",
            n_rec_layers=1,
            game_variables=[("health", 101), ("sel_ammo", 301)],
            n_variables=2,
            variable_dim=[32, 32],
            bucket_size=[1, 1],
            game_features="",
        ),
        ModelVariant.VIZDOOM_2017_TRACK1_RNN: ArnoldDQNConfig(
            pretrained_model_name="arnold_dqn_vizdoom_2017_track1_rnn",
            pretrained_model_path="s3://tt-ci-models-private/test_files/pytorch/Arnold/vizdoom_2017_track1.pth",
            height=60,
            width=108,
            n_fm=3,
            hist_size=4,
            hidden_dim=512,
            n_actions=29,  # From checkpoint: proj_action_scores.weight [29, 512]
            use_bn=False,
            dropout=0.5,
            dueling_network=False,
            network_type="dqn_rnn",
            recurrence="lstm",
            n_rec_layers=1,
            game_variables=[("health", 101), ("sel_ammo", 301)],
            n_variables=2,
            variable_dim=[32, 32],
            bucket_size=[10, 1],
            game_features="target,enemy",
        ),
        # vizdoom_2017_track2 with both architectures
        # Note: Checkpoint has n_actions=29 (action_combinations="move_fb+move_lr;turn_lr;attack")
        ModelVariant.VIZDOOM_2017_TRACK2_FF: ArnoldDQNConfig(
            pretrained_model_name="arnold_dqn_vizdoom_2017_track2_ff",
            pretrained_model_path="s3://tt-ci-models-private/test_files/pytorch/Arnold/vizdoom_2017_track2.pth",
            height=60,
            width=108,
            n_fm=3,
            hist_size=4,
            hidden_dim=512,
            n_actions=29,  # From checkpoint: proj_action_scores.weight [29, 512]
            use_bn=False,
            dropout=0.0,
            dueling_network=False,
            network_type="dqn_ff",
            recurrence="",
            n_rec_layers=1,
            game_variables=[("health", 101), ("sel_ammo", 301)],
            n_variables=2,
            variable_dim=[32, 32],
            bucket_size=[1, 1],
            game_features="",
        ),
        ModelVariant.VIZDOOM_2017_TRACK2_RNN: ArnoldDQNConfig(
            pretrained_model_name="arnold_dqn_vizdoom_2017_track2_rnn",
            pretrained_model_path="s3://tt-ci-models-private/test_files/pytorch/Arnold/vizdoom_2017_track2.pth",
            height=60,
            width=108,
            n_fm=3,
            hist_size=4,
            hidden_dim=512,
            n_actions=29,  # From checkpoint: proj_action_scores.weight [29, 512]
            use_bn=False,
            dropout=0.5,
            dueling_network=False,
            network_type="dqn_rnn",
            recurrence="lstm",
            n_rec_layers=1,
            game_variables=[("health", 101), ("sel_ammo", 301)],
            n_variables=2,
            variable_dim=[32, 32],
            bucket_size=[10, 1],
            game_features="target,enemy",
        ),
    }

    # Default variant to use
    DEFAULT_VARIANT = ModelVariant.DEFEND_THE_CENTER

    def __init__(self, variant: Optional[ModelVariant] = None):
        """Initialize ModelLoader with specified variant.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.
        """
        super().__init__(variant)

    def unpack_forward_output(self, fwd_output):
        """Unpack forward pass output for training mode.

        Since our model returns a single tensor when game features are disabled,
        we just return it directly. If it's a tuple, return the first element (Q-values).

        Args:
            fwd_output: Output from the forward pass (tensor or tuple)

        Returns:
            torch.Tensor: Q-values tensor for backward pass
        """
        if isinstance(fwd_output, tuple):
            # If tuple, return the first element (Q-values)
            return fwd_output[0]
        # If single tensor, return it directly
        return fwd_output

    @classmethod
    def _get_model_info(cls, variant: Optional[ModelVariant] = None) -> ModelInfo:
        """Get model information for dashboard and metrics reporting.

        Args:
            variant: Optional ModelVariant specifying which variant to use.
                     If None, DEFAULT_VARIANT is used.

        Returns:
            ModelInfo: Information about the model and variant
        """
        if variant is None:
            variant = cls.DEFAULT_VARIANT

        return ModelInfo(
            model="arnold_dqn",
            variant=variant,
            group=ModelGroup.GENERALITY,
            task=ModelTask.ATOMIC_ML,  # Reinforcement learning
            source=ModelSource.CUSTOM,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None, pretrained_path=None):
        """Load and return the Arnold DQN model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
                           If not provided, the model will use its default dtype (typically float32).
            pretrained_path: Optional path to pretrained model weights. If None, uses path from config.

        Returns:
            torch.nn.Module: The Arnold DQN model instance.
        """
        config = self._variant_config

        # Create a simple params object for model initialization
        class Params:
            def __init__(self, config):
                self.height = config.height
                self.width = config.width
                self.n_fm = config.n_fm
                self.hist_size = config.hist_size
                self.hidden_dim = config.hidden_dim
                self.n_actions = config.n_actions
                self.use_bn = config.use_bn
                self.dropout = config.dropout
                self.dueling_network = config.dueling_network
                self.network_type = config.network_type
                self.recurrence = config.recurrence
                self.n_rec_layers = config.n_rec_layers
                self.game_variables = config.game_variables
                self.n_variables = config.n_variables
                self.variable_dim = config.variable_dim
                self.bucket_size = config.bucket_size
                self.game_features = config.game_features

        params = Params(config)

        # Create model based on architecture type
        if config.network_type == "dqn_rnn":
            base_model = DQNModuleRecurrent(params)
        else:
            base_model = DQNModuleFeedforward(params)
        base_model.eval()

        # Wrap model to ensure consistent output format (always return single tensor)
        # This prevents issues with test framework when game features are disabled
        class ModelWrapper(torch.nn.Module):
            def __init__(self, base_model, is_recurrent=False):
                super().__init__()
                self.base_model = base_model
                self.is_recurrent = is_recurrent

            def forward(self, x_screens, x_variables, prev_state=None):
                if self.is_recurrent:
                    # For recurrent models, we need prev_state
                    # For inference/testing, we can use None (model will initialize it)
                    if prev_state is None:
                        # Initialize hidden state for recurrent models
                        batch_size = x_screens.size(0)
                        device = x_screens.device
                        dtype = x_screens.dtype
                        if hasattr(self.base_model, "rnn"):
                            num_layers = self.base_model.rnn.num_layers
                            hidden_dim = self.base_model.hidden_dim
                            if isinstance(self.base_model.rnn, torch.nn.LSTM):
                                h_0 = torch.zeros(
                                    num_layers,
                                    batch_size,
                                    hidden_dim,
                                    device=device,
                                    dtype=dtype,
                                )
                                c_0 = torch.zeros(
                                    num_layers,
                                    batch_size,
                                    hidden_dim,
                                    device=device,
                                    dtype=dtype,
                                )
                                prev_state = (h_0, c_0)
                            else:
                                h_0 = torch.zeros(
                                    num_layers,
                                    batch_size,
                                    hidden_dim,
                                    device=device,
                                    dtype=dtype,
                                )
                                prev_state = h_0
                    output = self.base_model(x_screens, x_variables, prev_state)
                    # Recurrent models return (output_sc, output_gf, next_state)
                    # Return only Q-values for testing
                    if isinstance(output, tuple):
                        return output[0]  # Return only Q-values
                    return output
                else:
                    # Feedforward model
                    output = self.base_model(x_screens, x_variables)
                    # Ensure we always return a single tensor (not tuple with None)
                    if isinstance(output, tuple):
                        # If tuple, return only the first element (Q-values)
                        return output[0]
                    # If already a single tensor, return it
                    return output

        model = ModelWrapper(
            base_model, is_recurrent=(config.network_type == "dqn_rnn")
        )
        model.eval()

        # Load pretrained weights if available
        pretrained_path = pretrained_path or config.pretrained_model_path
        if pretrained_path:
            try:
                # Use get_file utility to load from S3 bucket (s3://tt-ci-models-private/test_files/pytorch/Arnold/)
                # This handles S3 downloads and local caching automatically
                weight_path = get_file(pretrained_path)

                # Load state dict - Arnold saves state_dict directly
                # Handle both direct state_dict and wrapped formats
                state_dict = torch.load(weight_path, map_location="cpu")
                if isinstance(state_dict, dict):
                    # If it's a full checkpoint, try to extract the module state dict
                    if "module" in state_dict:
                        state_dict = state_dict["module"]
                    elif "state_dict" in state_dict:
                        state_dict = state_dict["state_dict"]
                    # Arnold saves state_dict directly, so load it
                    # Load into the base model, not the wrapper
                    missing_keys, unexpected_keys = base_model.load_state_dict(
                        state_dict, strict=False
                    )
                    if missing_keys:
                        print(
                            f"Warning: Some model parameters were not loaded: {missing_keys[:5]}..."
                        )
                    if unexpected_keys:
                        print(
                            f"Warning: Some checkpoint keys were not used: {unexpected_keys[:5]}..."
                        )
                    print(f"Successfully loaded pretrained weights from: {weight_path}")
            except Exception as e:
                # If weight loading fails, continue with untrained model
                # This allows the model to work even if weights aren't available
                print(
                    f"Warning: Failed to load pretrained weights from {pretrained_path}: {e}"
                )
                print("Continuing with randomly initialized weights.")

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Arnold DQN model.

        Args:
            dtype_override: Optional torch.dtype to override the inputs' default dtype.
                           If not provided, inputs will use the default dtype (typically float32).
            batch_size: Optional batch size to override the default batch size of 1.

        Returns:
            tuple: (screens, variables) where:
                - For feedforward models:
                  - screens: Screen images tensor of shape (batch_size, hist_size * n_fm, height, width)
                  - variables: List of game variable tensors, each of shape (batch_size,)
                - For recurrent models:
                  - screens: Screen images tensor of shape (batch_size, seq_len, n_fm, height, width)
                  - variables: List of game variable tensors, each of shape (batch_size, seq_len)
        """
        config = self._variant_config

        if config.network_type == "dqn_rnn":
            # Recurrent models: input shape is (batch_size, seq_len, n_fm, h, w)
            # Use hist_size as seq_len for recurrent models
            seq_len = config.hist_size
            screens = torch.rand(
                batch_size, seq_len, config.n_fm, config.height, config.width
            )

            # Create sample game variables with sequence dimension
            variables = []
            for i, (name, n_values) in enumerate(config.game_variables):
                var_tensor = torch.randint(
                    0, n_values, (batch_size, seq_len), dtype=torch.long
                )
                variables.append(var_tensor)
        else:
            # Feedforward models: input shape is (batch_size, hist_size * n_fm, height, width)
            screens = torch.rand(
                batch_size, config.hist_size * config.n_fm, config.height, config.width
            )

            # Create sample game variables (health, ammo)
            variables = []
            for i, (name, n_values) in enumerate(config.game_variables):
                var_tensor = torch.randint(0, n_values, (batch_size,), dtype=torch.long)
                variables.append(var_tensor)

        # Only convert dtype if explicitly requested
        if dtype_override is not None:
            screens = screens.to(dtype_override)
            # Note: variables are long tensors (indices), so we don't convert their dtype

        return screens, variables
