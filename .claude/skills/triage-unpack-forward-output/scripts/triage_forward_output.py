# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Run a tt-forge-models loader on CPU and dump the forward-pass output structure.

Used by the triage-unpack-forward-output skill in Step 3 (CPU triage). The
output goes to stdout; the skill captures it to /tmp/triage_<...>.log via shell
redirection and reads it back with the Read tool.

Usage:
    python triage_forward_output.py --model-dir <model_dir> [--variant <NAME>] [--batch-size 2]

The repo root is computed from __file__ (skill lives at
.claude/skills/triage-unpack-forward-output/scripts/), so the script works on
any checkout without hard-coded paths.
"""

import argparse
import collections.abc
import importlib
import inspect
import re
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

_MODEL_DIR_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_/]*$")


def _validate_model_dir(value: str) -> str:
    if not _MODEL_DIR_RE.match(value):
        raise argparse.ArgumentTypeError(
            f"--model-dir must contain only letters, digits, underscores, and forward slashes (got {value!r})"
        )
    return value


def _validate_identifier(value: str) -> str:
    if not value.isidentifier():
        raise argparse.ArgumentTypeError(
            f"--variant must be a valid Python identifier (got {value!r})"
        )
    return value


def describe(x, depth=0, name="out"):
    pad = "  " * depth
    if isinstance(x, torch.Tensor):
        print(f"{pad}{name}: Tensor shape={tuple(x.shape)} dtype={x.dtype}")
    elif isinstance(x, dict):
        print(f"{pad}{name}: dict({len(x)} keys)")
        for k, v in x.items():
            describe(v, depth + 1, str(k))
    elif isinstance(x, (list, tuple)):
        print(f"{pad}{name}: {type(x).__name__}({len(x)})")
        for i, v in enumerate(x):
            describe(v, depth + 1, f"[{i}]")
    else:
        attrs = [
            a
            for a in dir(x)
            if not a.startswith("_")
            and isinstance(getattr(x, a, None), (torch.Tensor, list, tuple, dict))
        ]
        print(f"{pad}{name}: {type(x).__name__} attrs={attrs}")
        for a in attrs:
            describe(getattr(x, a), depth + 1, a)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model-dir",
        required=True,
        type=_validate_model_dir,
        help="Top-level model directory under third_party/tt_forge_models, "
        "e.g. yolov9 or bert/question_answering",
    )
    parser.add_argument(
        "--variant",
        default=None,
        type=_validate_identifier,
        help="ModelVariant enum name (e.g. T, Base, Squad2). Omit if the loader has no ModelVariant.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    module_path = (
        f"third_party.tt_forge_models.{args.model_dir.replace('/', '.')}.pytorch.loader"
    )
    module = importlib.import_module(module_path)
    ModelLoader = module.ModelLoader

    if args.variant is not None and hasattr(module, "ModelVariant"):
        variant = getattr(module.ModelVariant, args.variant)
        loader = ModelLoader(variant=variant)
    else:
        loader = ModelLoader()

    model = loader.load_model().train()

    load_inputs_kwargs = {}
    if "batch_size" in inspect.signature(loader.load_inputs).parameters:
        load_inputs_kwargs["batch_size"] = args.batch_size
    inputs = loader.load_inputs(**load_inputs_kwargs)

    with torch.no_grad():
        if isinstance(inputs, collections.abc.Mapping):
            out = model(**inputs)
        elif isinstance(inputs, (list, tuple)):
            out = model(*inputs)
        else:
            out = model(inputs)

    describe(out)
    print(f"OUTPUT_CLASS: {type(out).__name__}")


if __name__ == "__main__":
    main()
