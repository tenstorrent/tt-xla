# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""CLI entry point for streaming DeepSeek-V4-Flash inference.

The actual orchestration lives in `streaming.core.run_streaming`; this file
just wires the adapter, config, and a final decoded-output print.

Run from project root:

    python -m streaming.run

(`python streaming/run.py` doesn't put cwd on `sys.path`, so the
`from streaming.adapters` imports below would fail.)

Optional env vars (all have working defaults):

    STREAM_MODE             whole_graph   whole_graph | layer_eager
    STREAM_NUM_LAYERS       43            number of transformer layers
    STREAM_MAX_NEW_TOKENS   3             decode steps
    STREAM_BATCH_SIZE       8             inference batch
    STREAM_PROMPT_LEN       128           padded prompt length (left-pad)
    STREAM_EXPERT_DTYPE     bf16          MoE expert pack dtype: bf16 / bfp_bf8 /
                                          bfp_bf4. bf16 = no override.
    STREAM_ATTN_DTYPE       bf16          Attention weight pack dtype (same
                                          values). bf16 = no override.
"""
from __future__ import annotations

import logging
import sys
import warnings

from ttxla_tools.logging import logger

from streaming.adapters.deepseek_v4_flash import DeepSeekV4FlashAdapter
from streaming.config import StreamingConfig
from streaming.core import print_decoded, run_streaming


def _configure_logging() -> None:
    """Quiet 3rd-party noise (transformers FutureWarning, dynamo, copyreg,
    transformers model-type warnings, ...) and route loguru to stderr with
    a compact prefix so layer-by-layer progress is easy to follow.

    Called AFTER the streaming imports so it overrides
    `ttxla_tools.logging` (transitively imported by tt_torch) which sets
    loguru to WARNING and hides our streaming progress."""
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="{time:HH:mm:ss} | {name} | {message}",
    )


def main() -> None:
    _configure_logging()
    config = StreamingConfig.from_env()
    adapter = DeepSeekV4FlashAdapter()
    result = run_streaming(adapter, config)
    print_decoded(adapter, result)


if __name__ == "__main__":
    main()
