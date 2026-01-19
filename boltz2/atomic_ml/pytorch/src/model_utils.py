# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
# code apapted from :
# https://github.com/jwohlwend/boltz

MIT License

Copyright (c) 2024 Jeremy Wohlwend, Gabriele Corso, Saro Passaro

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

from pathlib import Path
import os
from typing import Optional
import torch
import click
from dataclasses import asdict, is_dataclass
from types import MethodType

from third_party.tt_forge_models.tools.utils import get_file

BOLTZ_CACHE_DIR = (Path.home() / ".cache/url_cache").expanduser().resolve()


def load_boltz2_inputs(dtype_override: Optional[torch.dtype] = torch.float32):
    """
    Load, preprocess, and package input features for Boltz-2 inference.

    This function:
      • Sets up Boltz-2 cache directories (unified under BOLTZ_CACHE_DIR)
      • Downloads required Boltz-2 resources (CCD, molecules, checkpoint cache)
      • Runs full preprocessing (MSA retrieval, templates, constraints, molecules)
      • Loads the processed manifest and constructs input batches
    """
    from rdkit import Chem
    from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
    from boltz.data.types import Manifest
    from boltz.main import (
        download_boltz2,
        process_inputs,
        filter_inputs_structure,
        BoltzProcessedInput,
        check_inputs,
    )

    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("highest")
    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

    # Unified cache directory
    cache = BOLTZ_CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)

    # MSA server setup
    click.echo("MSA server enabled: https://api.colabfold.com")

    data = get_file("test_files/pytorch/boltz2/protein.yaml")
    data = Path(data).expanduser()

    out_dir = BOLTZ_CACHE_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    download_boltz2(cache)
    ccd_path = cache / "ccd.pkl"
    mol_dir = cache / "mols"

    # Validate inputs
    data = check_inputs(data)

    # Preprocessing inputs
    process_inputs(
        data=data,
        out_dir=out_dir,
        ccd_path=ccd_path,
        mol_dir=mol_dir,
        use_msa_server=True,
        msa_server_url="https://api.colabfold.com",
        msa_pairing_strategy="greedy",
        msa_server_username=os.environ.get("BOLTZ_MSA_USERNAME"),
        msa_server_password=os.environ.get("BOLTZ_MSA_PASSWORD"),
        api_key_header=None,
        api_key_value=os.environ.get("MSA_API_KEY_VALUE"),
        boltz2=True,
        preprocessing_threads=1,
        max_msa_seqs=8192,
    )

    # Load manifest
    manifest = Manifest.load(out_dir / "processed" / "manifest.json")

    # Optionally filter out existing predictions
    filtered_manifest = filter_inputs_structure(
        manifest=manifest,
        outdir=out_dir,
        override=True,
    )

    # Load processed data
    processed_dir = out_dir / "processed"
    processed = BoltzProcessedInput(
        manifest=filtered_manifest,
        targets_dir=processed_dir / "structures",
        msa_dir=processed_dir / "msa",
        constraints_dir=(
            (processed_dir / "constraints")
            if (processed_dir / "constraints").exists()
            else None
        ),
        template_dir=(
            (processed_dir / "templates")
            if (processed_dir / "templates").exists()
            else None
        ),
        extra_mols_dir=(
            (processed_dir / "mols") if (processed_dir / "mols").exists() else None
        ),
    )

    # Create data module (Boltz-2)
    data_module = Boltz2InferenceDataModule(
        manifest=processed.manifest,
        target_dir=processed.targets_dir,
        msa_dir=processed.msa_dir,
        mol_dir=mol_dir,
        num_workers=0,
        constraints_dir=processed.constraints_dir,
        template_dir=processed.template_dir,
        extra_mols_dir=processed.extra_mols_dir,
        override_method=None,
    )

    # Prepare predict dataloader and return a single batch of features
    predict_loader = data_module.predict_dataloader()
    try:
        first_batch = next(iter(predict_loader))
    except StopIteration:
        raise RuntimeError("No inputs available in the processed dataset to load.")

    # Recursively convert dataclass instances (possibly frozen) to plain dicts
    # to avoid attribute mutation attempts during device movement.
    def _convert_dataclasses(obj):
        if is_dataclass(obj):
            return _convert_dataclasses(asdict(obj))
        if isinstance(obj, dict):
            return {k: _convert_dataclasses(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert_dataclasses(v) for v in obj]
        if isinstance(obj, tuple):
            return tuple(_convert_dataclasses(v) for v in obj)
        return obj

    first_batch = _convert_dataclasses(first_batch)

    def _cast_recursive(obj):
        if isinstance(obj, torch.Tensor):
            return obj.to(dtype_override) if obj.is_floating_point() else obj
        if isinstance(obj, dict):
            return {k: _cast_recursive(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_cast_recursive(v) for v in obj]
        if isinstance(obj, tuple):
            return type(obj)(_cast_recursive(v) for v in obj)
        return obj

    first_batch = _cast_recursive(first_batch)
    return [first_batch]


def load_boltz2_model(checkpoint: Optional[str | Path] = None):
    """
    Load and initialize the Boltz-2 model for inference.

    This function:
      • Ensures the unified Boltz-2 cache directory exists
      • Downloads Boltz-2 model resources if missing
      • Locates (or resolves) the model checkpoint
      • Instantiates the Boltz-2 model with predefined diffusion, MSA, steering,
        and Pairformer configuration arguments
    """
    from boltz.model.models.boltz2 import Boltz2
    from boltz.main import (
        download_boltz2,
        PairformerArgsV2,
        MSAModuleArgs,
        Boltz2DiffusionParams,
        BoltzSteeringParams,
    )

    # Unified cache directory
    cache = BOLTZ_CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    download_boltz2(cache)

    # Resolve checkpoint
    checkpoint_path = (
        Path(checkpoint).expanduser().resolve()
        if checkpoint is not None
        else cache / "boltz2_conf.ckpt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {checkpoint_path}. "
            f"Expected the default Boltz-2 checkpoint at {cache / 'boltz2_conf.ckpt'} "
            f"or pass a valid path via `checkpoint`."
        )

    model_cls = Boltz2
    diffusion_params = Boltz2DiffusionParams()
    diffusion_params.step_scale = 1.5
    recycling_steps = 3
    sampling_steps = 200
    diffusion_samples = 1
    pairformer_args = PairformerArgsV2()
    predict_args = {
        "recycling_steps": recycling_steps,
        "sampling_steps": sampling_steps,
        "diffusion_samples": diffusion_samples,
        "max_parallel_samples": None,
        "write_confidence_summary": False,
        "write_full_pae": False,
        "write_full_pde": False,
    }

    msa_args = MSAModuleArgs(
        subsample_msa=True,
        num_subsampled_msa=1024,
        use_paired_feature=True,
    )

    steering_args = BoltzSteeringParams()
    steering_args.fk_steering = False
    steering_args.physical_guidance_update = False

    model_module = model_cls.load_from_checkpoint(
        str(checkpoint_path),
        strict=True,
        predict_args=predict_args,
        map_location="cpu",
        diffusion_process_args=asdict(diffusion_params),
        ema=False,
        use_kernels=True,
        pairformer_args=asdict(pairformer_args),
        msa_args=asdict(msa_args),
        steering_args=asdict(steering_args),
    )
    model_module.eval()
    model = model_module.to(dtype=torch.float32)
    model.use_kernels = False

    _default_forward_kwargs = {
        "recycling_steps": recycling_steps,
        "num_sampling_steps": sampling_steps,
        "diffusion_samples": diffusion_samples,
        "max_parallel_samples": None,
        "run_confidence_sequentially": True,
    }

    _orig_forward_bound = model.forward

    def _forward_with_defaults(module_self, feats, **kwargs):
        merged_kwargs = {**_default_forward_kwargs, **kwargs}
        return _orig_forward_bound(feats, **merged_kwargs)

    model.forward = MethodType(_forward_with_defaults, model)
    return model
