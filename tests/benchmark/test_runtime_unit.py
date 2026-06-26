# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""CPU-only unit tests for runtime.build_compile_options.

These pin the compile-options builder to the exact dicts the vision / encoder /
imagegen / llm drivers used to assemble inline, so it stays a behavior-preserving
replacement for drivers that can't run on CPU here.
"""

from runtime import build_compile_options


def test_compile_options_vision_imagegen_shape():
    """Vision and imagegen pass no experimental options -> those keys absent."""
    opts = build_compile_options(
        optimization_level=2,
        export_model_name="m",
        ttnn_perf_metrics_output_file="f",
        enable_trace=True,
    )
    assert opts == {
        "optimization_level": 2,
        "enable_trace": True,
        "export_path": "modules",
        "export_model_name": "m",
        "ttnn_perf_metrics_enabled": True,
        "ttnn_perf_metrics_output_file": "f",
    }


def test_compile_options_encoder_shape():
    """Encoder always emits the experimental weight-dtype / permute keys."""
    opts = build_compile_options(
        optimization_level=0,
        export_model_name="m",
        ttnn_perf_metrics_output_file="f",
        enable_trace=False,
        experimental_weight_dtype="",
        experimental_enable_permute_matmul_fusion=False,
    )
    assert opts == {
        "optimization_level": 0,
        "enable_trace": False,
        "export_path": "modules",
        "export_model_name": "m",
        "ttnn_perf_metrics_enabled": True,
        "ttnn_perf_metrics_output_file": "f",
        "experimental_weight_dtype": "",
        "experimental_enable_permute_matmul_fusion": False,
    }


def test_compile_options_llm_full_shape():
    """LLM forwards all optionals; meaningfully-set ones appear in the dict."""
    opts = build_compile_options(
        optimization_level=1,
        export_model_name="m",
        ttnn_perf_metrics_output_file="f",
        enable_trace=True,
        experimental_weight_dtype="bfp_bf8",
        experimental_enable_permute_matmul_fusion=True,
        fp32_dest_acc_en=True,
        experimental_kv_cache_dtype="bfp_bf8",
        enable_create_d2m_subgraphs=True,
    )
    assert opts == {
        "optimization_level": 1,
        "enable_trace": True,
        "export_path": "modules",
        "export_model_name": "m",
        "ttnn_perf_metrics_enabled": True,
        "ttnn_perf_metrics_output_file": "f",
        "experimental_weight_dtype": "bfp_bf8",
        "experimental_enable_permute_matmul_fusion": True,
        "fp32_dest_acc_en": True,
        "experimental-kv-cache-dtype": "bfp_bf8",
        "enable_create_d2m_subgraphs": True,
    }


def test_compile_options_llm_optionals_omitted_when_unset():
    """None / False optionals are dropped, matching the historical builder."""
    opts = build_compile_options(
        optimization_level=2,
        export_model_name="m",
        ttnn_perf_metrics_output_file="f",
        enable_trace=False,
        experimental_weight_dtype="bfp_bf8",
        experimental_enable_permute_matmul_fusion=False,
        fp32_dest_acc_en=None,
        experimental_kv_cache_dtype=None,
        enable_create_d2m_subgraphs=False,
    )
    assert "fp32_dest_acc_en" not in opts
    assert "experimental-kv-cache-dtype" not in opts
    assert "enable_create_d2m_subgraphs" not in opts
