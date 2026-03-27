# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile

import torch
import torch.nn as nn
from torch.nn.utils import parametrize
from tt_torch.weight_dtype import (
    WeightDtypeParametrization,
    apply_weight_dtype_overrides,
    dump_weight_names,
    remove_weight_dtype_overrides,
)


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(16, 32)
        self.linear2 = nn.Linear(32, 8)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))


class NestedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.ModuleDict(
            {
                "layers": nn.ModuleList(
                    [
                        nn.ModuleDict(
                            {
                                "mlp": nn.ModuleDict(
                                    {
                                        "fc1": nn.Linear(16, 32),
                                        "fc2": nn.Linear(32, 16),
                                    }
                                ),
                                "self_attn": nn.ModuleDict(
                                    {
                                        "q_proj": nn.Linear(16, 16),
                                        "k_proj": nn.Linear(16, 16),
                                    }
                                ),
                            }
                        )
                        for _ in range(2)
                    ]
                )
            }
        )

    def forward(self, x):
        for layer in self.model["layers"]:
            x = layer["mlp"]["fc2"](torch.relu(layer["mlp"]["fc1"](x)))
        return x


class TestWeightDtypeParametrization:

    def test_forward(self):
        linear = nn.Linear(4, 8)
        x = torch.randn(2, 4)
        parametrize.register_parametrization(
            linear, "weight", WeightDtypeParametrization("bfp_bf4")
        )
        out = linear(x)
        assert out.shape == (2, 8)


class TestApplyWeightDtypeOverrides:
    def test_dict_config(self):
        model = SimpleModel()
        applied = apply_weight_dtype_overrides(
            model, {"linear1.weight": "bfp_bf4", "linear2.weight": "bfp_bf8"}
        )
        assert len(applied) == 2
        assert ("linear1.weight", "bfp_bf4") in applied
        assert ("linear2.weight", "bfp_bf8") in applied
        assert parametrize.is_parametrized(model.linear1, "weight")
        assert parametrize.is_parametrized(model.linear2, "weight")

    def test_glob_pattern(self):
        model = NestedModel()
        applied = apply_weight_dtype_overrides(
            model, {"model.layers.*.mlp.*.weight": "bfp_bf4"}
        )
        # 2 layers x 2 MLP linears = 4
        assert len(applied) == 4
        assert all(dtype == "bfp_bf4" for _, dtype in applied)
        # Attention layers should NOT be parametrized
        assert not parametrize.is_parametrized(
            model.model["layers"][0]["self_attn"]["q_proj"], "weight"
        )

    def test_default_with_overrides(self):
        model = SimpleModel()
        applied = apply_weight_dtype_overrides(
            model, {"default": "bfp_bf8", "linear1.weight": "bfp_bf4"}
        )
        applied_dict = dict(applied)
        assert applied_dict["linear1.weight"] == "bfp_bf4"
        assert applied_dict["linear2.weight"] == "bfp_bf8"

    def test_json_file(self):
        model = SimpleModel()
        config = {"linear1.weight": "bfp_bf4"}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config, f)
            f.flush()
            applied = apply_weight_dtype_overrides(model, f.name)
        assert len(applied) == 1
        assert ("linear1.weight", "bfp_bf4") in applied

    def test_no_match_applies_nothing(self):
        model = SimpleModel()
        applied = apply_weight_dtype_overrides(
            model, {"nonexistent.layer.weight": "bfp_bf4"}
        )
        assert len(applied) == 0


class TestRemoveWeightDtypeOverrides:
    def test_remove(self):
        model = SimpleModel()
        apply_weight_dtype_overrides(
            model, {"linear1.weight": "bfp_bf4", "linear2.weight": "bfp_bf8"}
        )
        count = remove_weight_dtype_overrides(model)
        assert count == 2
        assert not parametrize.is_parametrized(model.linear1)
        assert not parametrize.is_parametrized(model.linear2)


class TestDumpWeightNames:
    def test_simple_model(self):
        model = SimpleModel()
        result = dump_weight_names(model)
        assert "linear1.weight" in result
        assert "linear2.weight" in result
        assert len(result) == 2
        assert all(v == "bfp_bf8" for v in result.values())

    def test_nested_model(self):
        model = NestedModel()
        result = dump_weight_names(model, default_dtype="bfp_bf4")
        # 2 layers x (2 MLP + 2 attn) = 8 weight parameters
        assert len(result) == 8
        assert all(v == "bfp_bf4" for v in result.values())
        assert "model.layers.0.mlp.fc1.weight" in result
        assert "model.layers.1.self_attn.q_proj.weight" in result
