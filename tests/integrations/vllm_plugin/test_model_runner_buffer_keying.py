# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""On-device regression for tt-xla #5416: buffer keying under the SMEM clamp.

Companion to ``test_prepare_inputs_buffer_keying.py`` (pure-helper unit tests).
This builds a real ``TTModelRunner`` where the SMEM seq limit
(``num_reqs_max_model_len``) falls below ``max_num_seqs`` and asserts every
per-batch device buffer is keyed by the clamped count.

The clamp normally needs a >128K-context model; we reproduce it cheaply by
shrinking the KV block size: ``get_max_num_seqs`` scales with
``block_size / max_model_len``, so ``block_size=8`` at the native 2048 context
gives a limit of 512 < ``max_num_seqs``.
"""

import pytest
import torch_xla.core.xla_model as xm
from vllm.engine.arg_utils import EngineArgs
from vllm_tt.model_runner import TTModelRunner

# opt-125m: native context 2048, tiny (config cached, no weights loaded by
# TTModelRunner.__init__). block_size=8 -> get_max_num_seqs(2048, 8) == 512.
_MODEL = "facebook/opt-125m"
_MAX_MODEL_LEN = 2048
_BLOCK_SIZE = 8
_MAX_NUM_SEQS = 640  # > 512 so the SMEM limit clamps the batch
_EXPECTED_SMEM_LIMIT = 512


def _build_clamp_config(min_num_seqs, max_prefill_num_seqs):
    additional_config = {
        "enable_const_eval": True,
        "min_context_len": 32,
        "prefill_chunk_size": 32,  # enable chunked prefill (loose budget assert)
    }
    if min_num_seqs is not None:
        additional_config["min_num_seqs"] = min_num_seqs
    if max_prefill_num_seqs is not None:
        additional_config["max_prefill_num_seqs"] = max_prefill_num_seqs
    engine_args = EngineArgs(
        model=_MODEL,
        max_num_batched_tokens=1024,
        max_num_seqs=_MAX_NUM_SEQS,
        max_model_len=_MAX_MODEL_LEN,
        gpu_memory_utilization=0.001,
        additional_config=additional_config,
    )
    vllm_config = engine_args.create_engine_config()
    # The TT platform forces block_size=32; override so the SMEM seq limit drops
    # below max_num_seqs on a small model (see module docstring).
    vllm_config.cache_config.block_size = _BLOCK_SIZE
    return vllm_config


@pytest.mark.push
@pytest.mark.single_device
@pytest.mark.parametrize(
    "min_num_seqs,max_prefill_num_seqs",
    [
        pytest.param(None, None, id="defaults"),  # plain decode, features off
        pytest.param(4, None, id="min_num_seqs_4"),  # prefill request bucketing
        pytest.param(4, 256, id="max_prefill_256"),  # distinct prefill ceiling
    ],
)
def test_per_batch_buffers_keyed_by_smem_clamped_count(
    min_num_seqs, max_prefill_num_seqs
):
    """Every per-batch buffer is keyed by the SMEM-clamped reachable set, and the
    runner constructs (startup assert passes) even though the batch is clamped."""
    vllm_config = _build_clamp_config(min_num_seqs, max_prefill_num_seqs)
    # Construction runs the startup buffer-keying assert; a failure raises here.
    runner = TTModelRunner(vllm_config, xm.xla_device())

    # The clamp must actually be active, else the test proves nothing.
    assert runner.num_reqs_max_model_len == _EXPECTED_SMEM_LIMIT
    assert runner.num_reqs_max_model_len < runner.max_num_reqs

    reachable = runner._reachable_num_reqs
    input_ids_keys = {k[0] for k in runner._input_ids_dev}
    position_ids_keys = {k[0] for k in runner._position_ids_dev}

    # All per-batch buffers agree on the same clamped key set...
    assert input_ids_keys == reachable
    assert position_ids_keys == reachable
    assert set(runner._logits_indices_dev) == reachable
    assert set(runner._batch_idx_dev) == reachable
    # ...which includes the clamped decode target the runtime will request...
    assert runner.num_reqs_max_model_len in input_ids_keys
    # ...and page tables (nested per kv-cache group) share the max-path subset.
    assert runner._max_len_num_reqs <= set(runner._page_table_dev_max[0])

    # Regression guard: the pre-#5416 keying used max_num_reqs, which under the
    # clamp is absent from the (now consistent) key set.
    assert runner.max_num_reqs not in input_ids_keys
