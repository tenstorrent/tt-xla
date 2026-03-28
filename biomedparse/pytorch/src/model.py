# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
BiomedParse model wrapper.

Handles cloning the BiomedParse repository and loading the model
using its custom code. BiomedParse is a biomedical foundation model
for joint segmentation, detection, and recognition across 9 imaging modalities.

Reference: https://github.com/microsoft/BiomedParse
HuggingFace: https://huggingface.co/microsoft/BiomedParse
"""

import os
import sys
import subprocess
from pathlib import Path

# Repository details
BIOMEDPARSE_REPO_URL = "https://github.com/microsoft/BiomedParse.git"
BIOMEDPARSE_CACHE_DIR = Path.home() / ".cache" / "biomedparse"


def _ensure_repo_cloned():
    """Clone the BiomedParse repo to cache if not already present.

    Returns:
        Path: Path to the cloned repository.
    """
    repo_dir = BIOMEDPARSE_CACHE_DIR / "BiomedParse"

    if not repo_dir.exists():
        BIOMEDPARSE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", BIOMEDPARSE_REPO_URL, str(repo_dir)],
            check=True,
        )

    return repo_dir


def _add_repo_to_path(repo_dir):
    """Add the BiomedParse repo to sys.path so its modules can be imported.

    Args:
        repo_dir: Path to the cloned BiomedParse repository.
    """
    repo_str = str(repo_dir)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)


def build_biomedparse_model():
    """Build and return the BiomedParse model using the repository's code.

    Returns:
        torch.nn.Module: The BiomedParse model in eval mode.
    """
    repo_dir = _ensure_repo_cloned()
    _add_repo_to_path(repo_dir)

    from modeling.BaseModel import BaseModel
    from modeling import build_model
    from utilities.arguments import load_opt_from_config_files
    from utilities.distributed import init_distributed

    config_path = str(repo_dir / "configs" / "biomedparse_inference.yaml")
    opt = load_opt_from_config_files([config_path])
    opt = init_distributed(opt)

    pretrained_pth = "hf_hub:microsoft/BiomedParse"
    model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval()

    return model
