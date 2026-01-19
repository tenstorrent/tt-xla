# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Arnold DQN model loader implementation.

Arnold is a Deep Q-Network (DQN) implementation for ViZDoom reinforcement learning.
This model processes screen images and game variables to predict Q-values for actions.
"""
from typing import Optional, Dict, Any
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

    # Required by ModelConfig base class
    pretrained_model_name: str

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

    # Pretrained model path (only set if matching checkpoint exists)
    pretrained_model_path: str = ""


class ModelVariant(StrEnum):
    """Available Arnold DQN model variants - only variants with matching checkpoints."""

    # Feedforward variants with matching checkpoints
    DEFEND_THE_CENTER_FF = "defend_the_center_ff"
    HEALTH_GATHERING_FF = "health_gathering_ff"
    # Recurrent variants with matching checkpoints
    DEATHMATCH_SHOTGUN_RNN = "deathmatch_shotgun_rnn"
    VIZDOOM_2017_TRACK1_RNN = "vizdoom_2017_track1_rnn"
    VIZDOOM_2017_TRACK2_RNN = "vizdoom_2017_track2_rnn"


class ModelLoader(ForgeModel):
    """Arnold DQN model loader implementation."""

    # Dictionary of available model variants
    # Each variant uses only weights meant for its specific architecture
    _VARIANTS = {
        # Feedforward variants with matching checkpoints
        ModelVariant.DEFEND_THE_CENTER_FF: ArnoldDQNConfig(
            pretrained_model_name="defend_the_center_ff",
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
        ModelVariant.HEALTH_GATHERING_FF: ArnoldDQNConfig(
            pretrained_model_name="health_gathering_ff",
            pretrained_model_path="s3://tt-ci-models-private/test_files/pytorch/Arnold/health_gathering.pth",
            height=60,
            width=108,
            n_fm=3,
            hist_size=4,
            hidden_dim=512,
            n_actions=8,
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
        # Recurrent variants with matching checkpoints
        ModelVariant.DEATHMATCH_SHOTGUN_RNN: ArnoldDQNConfig(
            pretrained_model_name="deathmatch_shotgun_rnn",
            pretrained_model_path="s3://tt-ci-models-private/test_files/pytorch/Arnold/deathmatch_shotgun.pth",
            height=60,
            width=108,
            n_fm=4,  # 3 RGB + 1 label map
            hist_size=6,
            hidden_dim=512,
            n_actions=29,
            use_bn=False,
            dropout=0.5,
            dueling_network=False,
            network_type="dqn_rnn",
            recurrence="lstm",
            n_rec_layers=1,
            game_variables=[("health", 11), ("sel_ammo", 301)],
            n_variables=2,
            variable_dim=[32, 32],
            bucket_size=[1, 1],
            game_features="target,enemy",
        ),
        ModelVariant.VIZDOOM_2017_TRACK1_RNN: ArnoldDQNConfig(
            pretrained_model_name="vizdoom_2017_track1_rnn",
            pretrained_model_path="s3://tt-ci-models-private/test_files/pytorch/Arnold/vizdoom_2017_track1.pth",
            height=60,
            width=108,
            n_fm=3,
            hist_size=4,
            hidden_dim=512,
            n_actions=35,
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
        ModelVariant.VIZDOOM_2017_TRACK2_RNN: ArnoldDQNConfig(
            pretrained_model_name="vizdoom_2017_track2_rnn",
            pretrained_model_path="s3://tt-ci-models-private/test_files/pytorch/Arnold/vizdoom_2017_track2.pth",
            height=60,
            width=108,
            n_fm=3,
            hist_size=4,
            hidden_dim=512,
            n_actions=29,
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
    DEFAULT_VARIANT = ModelVariant.DEFEND_THE_CENTER_FF

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
            return fwd_output[0]
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
            task=ModelTask.ATOMIC_ML,
            source=ModelSource.CUSTOM,
            framework=Framework.TORCH,
        )

    @staticmethod
    def _convert_s3_path_for_cache(s3_path: str) -> str:
        """Convert S3 path to format expected by IRD_LF_CACHE server.

        The cache server expects paths without the bucket name:
        - Input:  s3://tt-ci-models-private/test_files/pytorch/Arnold/file.pth
        - Output: test_files/pytorch/Arnold/file.pth

        Args:
            s3_path: S3 path starting with s3://

        Returns:
            Path without s3:// prefix and bucket name, or original path if not S3
        """
        if s3_path.startswith("s3://"):
            path_without_prefix = s3_path[5:]
            if "/" in path_without_prefix:
                _, path_after_bucket = path_without_prefix.split("/", 1)
                return path_after_bucket
            return path_without_prefix
        return s3_path

    def load_model(self, dtype_override=None, pretrained_path=None):
        """Load and return the Arnold DQN model instance.

        Args:
            dtype_override: Optional torch.dtype to override the model's default dtype.
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

        # Wrap model to ensure consistent output format
        class ModelWrapper(torch.nn.Module):
            def __init__(self, base_model, is_recurrent=False):
                super().__init__()
                self.base_model = base_model
                self.is_recurrent = is_recurrent

            def forward(self, x_screens, x_variables, prev_state=None):
                if self.is_recurrent:
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
                    if isinstance(output, tuple):
                        return output[0]  # Return only Q-values
                    return output
                else:
                    output = self.base_model(x_screens, x_variables)
                    if isinstance(output, tuple):
                        return output[0]
                    return output

        model = ModelWrapper(
            base_model, is_recurrent=(config.network_type == "dqn_rnn")
        )
        model.eval()

        # Load pretrained weights if available
        pretrained_path = pretrained_path or config.pretrained_model_path
        if pretrained_path:
            cache_path = self._convert_s3_path_for_cache(pretrained_path)
            weight_path = get_file(cache_path)
            state_dict = torch.load(weight_path, map_location="cpu")

            # Extract module state dict if wrapped
            if isinstance(state_dict, dict):
                if "module" in state_dict:
                    state_dict = state_dict["module"]
                elif "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]

            base_model.load_state_dict(state_dict, strict=False)

        if dtype_override is not None:
            model = model.to(dtype_override)

        return model

    def load_inputs(self, dtype_override=None, batch_size=1):
        """Load and return sample inputs for the Arnold DQN model.

        Args:
            dtype_override: Optional torch.dtype to override the inputs' default dtype.
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
            seq_len = config.hist_size
            screens = torch.rand(
                batch_size, seq_len, config.n_fm, config.height, config.width
            )
            variables = []
            for i, (name, n_values) in enumerate(config.game_variables):
                var_tensor = torch.randint(
                    0, n_values, (batch_size, seq_len), dtype=torch.long
                )
                variables.append(var_tensor)
        else:
            screens = torch.rand(
                batch_size, config.hist_size * config.n_fm, config.height, config.width
            )
            variables = []
            for i, (name, n_values) in enumerate(config.game_variables):
                var_tensor = torch.randint(0, n_values, (batch_size,), dtype=torch.long)
                variables.append(var_tensor)

        if dtype_override is not None:
            screens = screens.to(dtype_override)

        return screens, variables

    def post_process(
        self,
        output: torch.Tensor,
        return_q_values: bool = False,
    ) -> Dict[str, Any]:
        """
        Post-process model output (Q-values) to extract action ID.

        The model outputs Q-values (action scores) for each possible action.
        This method selects the action with the highest Q-value (argmax) and
        returns the action ID along with optional Q-value information.

        Args:
            output: Model output tensor containing Q-values.
                   Shape: (batch_size, n_actions) for feedforward models
                   Shape: (batch_size, seq_len, n_actions) for recurrent models
            return_q_values: If True, include Q-values in the returned dictionary.

        Returns:
            Dictionary containing:
                - action_id: Integer action ID (0 to n_actions-1) with highest Q-value
                - max_q_value: Maximum Q-value (if return_q_values is True)
                - q_values: All Q-values as a tensor (if return_q_values is True)
                - batch_size: Batch size of the input
                - n_actions: Number of possible actions
        """
        # Handle different output shapes based on model type
        if output.dim() == 3:
            # Recurrent model: (batch_size, seq_len, n_actions)
            # Use the last timestep for action selection
            scores = output[:, -1, :]  # Shape: (batch_size, n_actions)
        elif output.dim() == 2:
            # Feedforward model: (batch_size, n_actions)
            scores = output
        else:
            # Handle single sample case: (n_actions,) -> (1, n_actions)
            if output.dim() == 1:
                scores = output.unsqueeze(0)
            else:
                raise ValueError(
                    f"Unexpected output shape: {output.shape}. "
                    "Expected (batch_size, n_actions) or (batch_size, seq_len, n_actions)"
                )

        batch_size = scores.size(0)
        n_actions = scores.size(1)

        # Get action IDs with highest Q-value for each sample in batch
        max_q_values, action_ids = torch.max(scores, dim=1)
        action_ids = action_ids.cpu().numpy()

        # Prepare result dictionary
        result = {
            "action_id": action_ids[0] if batch_size == 1 else action_ids.tolist(),
            "batch_size": batch_size,
            "n_actions": n_actions,
        }

        # Add Q-value information if requested
        if return_q_values:
            result["max_q_value"] = (
                max_q_values[0].item()
                if batch_size == 1
                else max_q_values.cpu().tolist()
            )
            result["q_values"] = scores.cpu() if batch_size == 1 else scores.cpu()

        return result
