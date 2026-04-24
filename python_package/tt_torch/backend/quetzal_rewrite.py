# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any

from tt_torch.backend.passes import run_selected_fusion_passes
from tt_torch.fusion_providers import FusionProvider
from ttxla_tools.logging import logger

QUETZAL_REWRITE_PASSES_OPTION = "tt_quetzal_rewrite_passes"
QUETZAL_REWRITE_PASSES_ENV = "TT_TORCH_QUETZAL_REWRITE_PASSES"

DEFAULT_QUETZAL_REWRITE_PASSES = [
    "fuse_gelu",
    "reconstruct_sdpa",
]


def get_quetzal_rewrite_passes(options: dict[str, Any] | None) -> list[str]:
    requested = None
    if options and options.get(QUETZAL_REWRITE_PASSES_OPTION) is not None:
        requested = str(options[QUETZAL_REWRITE_PASSES_OPTION]).strip()
    elif os.environ.get(QUETZAL_REWRITE_PASSES_ENV) is not None:
        requested = os.environ[QUETZAL_REWRITE_PASSES_ENV].strip()

    if not requested or requested.lower() == "none":
        return []

    if requested.lower() == "all":
        return list(DEFAULT_QUETZAL_REWRITE_PASSES)

    return [name.strip() for name in requested.split(",") if name.strip()]


def run_quetzal_rewrite_passes(gm, options: dict[str, Any] | None) -> None:
    provider_names = get_quetzal_rewrite_passes(options)
    if not provider_names:
        return

    known_provider_names = set(
        FusionProvider.get_registered_provider_names(include_default_disabled=True)
    )
    unknown = [name for name in provider_names if name not in known_provider_names]
    if unknown:
        logger.warning(
            f"[QuetzalRewrite] Ignoring unknown pass(es): {', '.join(unknown)}"
        )

    provider_names = [name for name in provider_names if name in known_provider_names]
    if not provider_names:
        return

    replacements = run_selected_fusion_passes(gm, provider_names)
    logger.info(
        "[QuetzalRewrite] "
        f"passes={','.join(provider_names)} "
        f"matches={sum(replacements.values())}"
    )
