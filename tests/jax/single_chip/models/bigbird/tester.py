# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from typing import Dict, Sequence

from infra import ComparisonConfig, ModelTester, RunMode


class BigBirdTester(ModelTester):
    """Tester for BigBird Model variants."""

    def __init__(
        self,
        model_path: str,
        comparison_config: ComparisonConfig = ComparisonConfig(),
        run_mode: RunMode = RunMode.INFERENCE,
    ) -> None:
        self._model_path = model_path
        super().__init__(comparison_config, run_mode)

    # override
    def _get_static_argnames(self) -> Sequence[str]:
        return ["train"]
