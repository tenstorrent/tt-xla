# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import torch
import os
from ...config import (
    ModelInfo,
    ModelGroup,
    ModelTask,
    ModelSource,
    Framework,
    ModelConfig,
    StrEnum,
)
from ...base import ForgeModel


class ModelVariant(StrEnum):
    """Available model variants."""

    BASE = "Hippynn"


class ModelLoader(ForgeModel):
    """Forge-compatible loader for the Hippynn model."""

    _VARIANTS = {
        ModelVariant.BASE: ModelConfig(pretrained_model_name="hippynn_model_pretrained")
    }

    DEFAULT_VARIANT = ModelVariant.BASE

    @classmethod
    def _get_model_info(cls, variant: ModelVariant = None):
        """Return model information for Forge dashboards and reporting."""
        return ModelInfo(
            model="hippynn",
            variant=variant,
            group=ModelGroup.PRIORITY,
            task=ModelTask.ATOMIC_ML,
            source=ModelSource.GITHUB,
            framework=Framework.TORCH,
        )

    def load_model(self, dtype_override=None):
        """Load and return the Hippynn model wrapped in a Torch module.

        Args:
            dtype_override: Optional torch.dtype to override model precision.
        """

        from hippynn.graphs import GraphModule, inputs, networks, targets

        network_params = {
            "possible_species": [0, 1, 6, 7, 8, 16],
            "n_features": 20,
            "n_sensitivities": 20,
            "dist_soft_min": 1.6,
            "dist_soft_max": 10.0,
            "dist_hard_max": 12.5,
            "n_interaction_layers": 2,
            "n_atom_layers": 3,
        }
        os.environ["HIPPYNN_USE_CUSTOM_KERNELS"] = "False"
        species = inputs.SpeciesNode(db_name="Z")
        positions = inputs.PositionsNode(db_name="R")
        network = networks.Hipnn(
            "hippynn_model", (species, positions), module_kwargs=network_params
        )
        henergy = targets.HEnergyNode("HEnergy", network, db_name="T")

        # Load model
        model = GraphModule([species, positions], [henergy.mol_energy])

        for param in model.parameters():
            param.requires_grad = False

        model.eval()
        if dtype_override is not None:
            model = model.to(dtype_override)

        return model, henergy.mol_energy

    def load_inputs(self, dtype_override=None):
        """Load and return sample inputs (species and positions) for the model.

        Args:
            dtype_override: Optional torch.dtype to override input precision.
        """

        import ase.build
        import ase.units

        atoms = ase.build.molecule("H2O")
        positions = (
            torch.as_tensor(atoms.positions / ase.units.Bohr)
            .unsqueeze(0)
            .to(torch.get_default_dtype())
        )

        species = torch.as_tensor(atoms.get_atomic_numbers()).unsqueeze(0)

        if dtype_override is not None:
            positions = positions.to(dtype_override)

        return (species, positions)
