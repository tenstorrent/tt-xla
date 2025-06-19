# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from .multi_chip_tester import MultiChipTester


class TorchMultiChipTester(MultiChipTester):
    """A tester for evaluating operations in a multichip Torch execution environment."""

    raise NotImplementedError(
        "Support for torch multichip testing not yet implemented."
    )
