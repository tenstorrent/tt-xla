# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager
from enum import Enum
from typing import Callable, Optional

import jax
import jax.lax as jlx
import jax.numpy as jnp
import torch_xla
from infra import Framework


class StrEnum(Enum):
    def __str__(self) -> str:
        return self.value


class Category(StrEnum):
    OP_TEST = "op_test"
    GRAPH_TEST = "graph_test"
    MODEL_TEST = "model_test"
    OTHER = "other"


class ModelGroup(StrEnum):
    RED = "red"
    PRIORITY = "priority"
    GENERALITY = "generality"


class TTArch(StrEnum):
    WORMHOLE_B0 = "wormhole_b0"
    BLACKHOLE = "blackhole"


class ModelTask(StrEnum):
    """
    Classification of tasks models can perform.

    Implemented based on
    https://huggingface.co/docs/transformers/en/tasks/sequence_classification.
    """

    NLP_TEXT_CLS = "nlp_text_cls"
    NLP_TOKEN_CLS = "nlp_token_cls"
    NLP_QA = "nlp_qa"
    NLP_CAUSAL_LM = "nlp_causal_lm"
    NLP_MASKED_LM = "nlp_masked_lm"
    NLP_TRANSLATION = "nlp_translation"
    NLP_SUMMARIZATION = "nlp_summarization"
    NLP_MULTI_CHOICE = "nlp_multi_choice"
    AUDIO_CLS = "audio_cls"
    AUDIO_ASR = "audio_asr"
    CV_IMAGE_CLS = "cv_image_cls"
    CV_IMAGE_SEG = "cv_image_seg"
    CV_VIDEO_CLS = "cv_video_cls"
    CV_OBJECT_DET = "cv_object_det"
    CV_ZS_OBJECT_DET = "cv_zs_object_det"
    CV_ZS_IMAGE_CLS = "cv_zs_image_cls"
    CV_DEPTH_EST = "cv_depth_est"
    CV_IMG_TO_IMG = "cv_img_to_img"
    CV_IMAGE_FE = "cv_image_fe"
    CV_MASK_GEN = "cv_mask_gen"
    CV_KEYPOINT_DET = "cv_keypoint_det"
    CV_KNOW_DISTILL = "cv_know_distill"
    MM_IMAGE_CAPT = "mm_image_capt"
    MM_DOC_QA = "mm_doc_qa"
    MM_VISUAL_QA = "mm_visual_qa"
    MM_TTS = "mm_tts"
    MM_IMAGE_TTT = "mm_image_ttt"
    MM_VIDEO_TTT = "mm_video_ttt"


class ModelSource(StrEnum):
    HUGGING_FACE = "huggingface"
    CUSTOM = "custom"
    TORCH_HUB = "torch_hub"
    GITHUB = "github"
    TIMM = "timm"
    TORCHVISION = "torchvision"


class BringupStatus(Enum):
    FAILED_FE_COMPILATION = "failed_fe_compilation"
    FAILED_TTMLIR_COMPILATION = "failed_ttmlir_compilation"
    FAILED_RUNTIME = "failed_runtime"
    INCORRECT_RESULT = "incorrect_result"
    PASSED = "passed"
    UNKNOWN = "unknown"
    NOT_STARTED = "not_started"

    def __str__(self) -> str:
        return self.name


class ExecutionPass(Enum):
    FORWARD = "forward"
    BACKWARD = "backward"

    def __str__(self) -> str:
        return self.name


def failed_fe_compilation(reason: str) -> str:
    return f"{BringupStatus.FAILED_FE_COMPILATION}: {reason}"


def failed_ttmlir_compilation(reason: str) -> str:
    return f"{BringupStatus.FAILED_TTMLIR_COMPILATION}: {reason}"


def failed_runtime(reason: str) -> str:
    return f"{BringupStatus.FAILED_RUNTIME}: {reason}"


def incorrect_result(reason: str) -> str:
    return f"{BringupStatus.INCORRECT_RESULT}: {reason}"


def passed(reason: str) -> str:
    return f"{BringupStatus.PASSED}: {reason}"


def build_model_name(
    framework: Framework,
    model: str,
    variant: Optional[str],
    task: ModelTask,
    source: ModelSource,
) -> str:
    variant = variant if variant is not None else "base"
    return f"{framework}_{model}_{variant}_{task}_{source}"


@contextmanager
def enable_x64():
    """
    Context manager that temporarily enables x64 in jax.config.

    Isolated as a context manager so that it doesn't change global config for all jax
    imports and cause unexpected fails elsewhere.
    """
    try:
        # Set the config to True within this block, and yield back control.
        jax.config.update("jax_enable_x64", True)
        yield
    finally:
        # After `with` statement ends, turn it off again.
        jax.config.update("jax_enable_x64", False)


# NOTE TTNN does not support boolean data type, so bfloat16 is used instead.
# The output of logical operation (and other similar ops) is bfloat16. JAX can
# not perform any computation due to mismatch in output data type (in testing
# infrastructure). The following tests explicitly convert data type of logical
# operation output for the verification purposes.
# TODO Remove this workaround once the data type issue is resolved.
# https://github.com/tenstorrent/tt-xla/issues/93
# TODO investigate why this decorator cannot be removed. See issue
# https://github.com/tenstorrent/tt-xla/issues/156
def convert_output_to_bfloat16(f: Callable):
    """Decorator to work around the mentioned issue."""

    def wrapper(*args, **kwargs):
        res = f(*args, **kwargs)
        return jlx.convert_element_type(res, jnp.bfloat16)

    return wrapper


def is_single_device(request):
    return request.node.get_closest_marker("single_device") is not None


def is_dual_chip(request):
    return request.node.get_closest_marker("dual_chip") is not None


def is_llmbox(request):
    return request.node.get_closest_marker("llmbox") is not None

def is_galaxy(request):
    return request.node.get_closest_marker("galaxy") is not None

def get_torch_device_arch() -> TTArch:
    """Returns the architecture of the connected TT device."""
    device_kind = torch_xla._XLAC._xla_device_kind("xla")
    if "Wormhole_b0" in device_kind:
        return TTArch.WORMHOLE_B0
    elif "Blackhole" in device_kind:
        return TTArch.BLACKHOLE
    else:
        raise ValueError(f"Unknown TT device architecture: {device_kind}")
