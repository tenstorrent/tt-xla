# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from infra import Framework, run_op_test
from utils import Category


def _gather_int64_comparator(device_output, golden_output, args, kwargs):
    """Exact-match comparator for torch.gather on int64 index tensors.

    Any mismatch indicates a bug in the stablehlo.gather → ttnn.gather
    lowering (tt-xla issue #4329): gather was lowered to ttnn.embedding
    which casts integer values through bf16, corrupting vocab IDs.
    """
    device_value = device_output.cpu().item()
    golden_value = golden_output.item()
    local_idx = args[1].item()
    print(
        f"\n  local_idx={local_idx}  "
        f"cpu={golden_value}  dev={device_value}  "
        f"match={golden_value == device_value}"
    )
    assert golden_value == device_value, (
        f"gather int64 mismatch: cpu={golden_value} dev={device_value} "
        f"(local_idx={local_idx})"
    )


@pytest.mark.nightly
@pytest.mark.single_device
@pytest.mark.record_test_properties(
    category=Category.OP_TEST,
    torch_op_name="torch.gather",
)
@pytest.mark.parametrize(
    "candidates,vocab_size",
    [
        pytest.param(64, 50272, id="opt125m"),
        pytest.param(128, 128256, id="llama"),
    ],
)
def test_gather_int64(candidates, vocab_size):
    """torch.gather on int64 index tensors returns correct values.

    Regression for tt-xla issue #4329: stablehlo.gather was lowered to
    ttnn.embedding instead of ttnn.gather, casting integer vocab IDs
    through bf16 and corrupting them (e.g. 2814→2816, 91715→91648).
    Fixed by Het Shah's composite gather op (tenstorrent/tt-mlir#7773).

    Uses run_op_test (Framework.TORCH) so the op goes through torch.compile
    and the stablehlo.gather lowering path — eager execution does not
    insert composite op lowerings.
    """

    class GatherInt64(torch.nn.Module):
        def forward(self, idx, local):
            return torch.gather(idx, 1, local)

    torch.manual_seed(42)
    idx_cpu = torch.randperm(vocab_size, dtype=torch.int64)[:candidates].unsqueeze(0)
    local_cpu = torch.randint(0, candidates, (1, 1), dtype=torch.int64)

    run_op_test(
        GatherInt64(),
        [idx_cpu, local_cpu],
        framework=Framework.TORCH,
        custom_comparator=_gather_int64_comparator,
    )
