# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
**Bracket** and **hybrid CPU/TT** sanities around ``action_in_proj`` (``nn.Linear``).

Log-derived summary (``tt_denoise_steps_*.log``, ``tt_embed_suffix*.log``,
``trim_failing_sanity.log`` / ``trim_failing_sanity_2`` when network + fixture succeed):

- The **bad lowering** shows up as ``ttnn::concat`` / ``ShapeBase[] index out of range``
  (or PJRT **error 13** after a bad device run) when the **biased** second-step linear
  runs at the end of the **full TT stem**: **prefix + first denoise +** ``state_proj`` +
  time sinusoid + ``F.linear(x_t, W, b)`` (see ``BracketThroughLinearWithBias`` /
  ``test_pi0_bracket_action_in_proj_with_bias_expects_tt_failure``).

- The **same stem with** ``F.linear(..., bias=None)`` **passes** PCC
  (``BracketThroughLinearMatmulNoBias`` / ``passing_one.log``-style runs).

- When **prefix and first denoise** are done **on CPU** and only a **suffix** (or even
  **only** ``F.linear`` on ``action_in_proj`` weights) runs on TT, **PCC passes**
  (``test_pi0_hybrid_linear_bias_cpu_xt_only``, ``test_pi0_hybrid_linear_bias_tt_prefix_cpu_xt``,
  ``test_pi0_micro_biased_linear_cpu_xt`` / ``trim_failing_sanity``). So the failure is
  tied to the **whole fused TT graph** through TT-produced ``x_t`` + stem, not to the
  biased-linear math alone in isolation on TT.

- ``test_pi0_unit_biased_action_in_proj_linear_only`` is the **smallest** ``run_op_test``
  slice: **only** ``F.linear(x, W, b)`` (Pi0 weights), ``x`` = CPU midpoint after one
  denoise (no ``state_proj``, no time emb, no residual).

Incremental bisect (``test_pi0_second_embed_suffix_incremental_tt``) suggested the TT
``ttnn::concat`` / ``ShapeBase`` failure appears when **bias is included** in
``F.linear(x_t, weight, bias)``, not on the matmul-only path.

Bracket (both run **prefix + first denoise + suffix time path** entirely on TT):

1. ``BracketThroughLinearMatmulNoBias`` — ``F.linear(x_t, weight, bias=None)`` only.
2. ``BracketThroughLinearWithBias`` — ``F.linear(x_t, weight, bias)``.

Expect (1) to pass PCC on TT. For (2), ``run_op_test`` would fail PCC; instead
``test_pi0_bracket_action_in_proj_with_bias_expects_tt_failure`` **requires** a TT-side
exception consistent with the known ``ttnn::concat`` / ``ShapeBase`` / PJRT error-13
class until the stack is fixed.

Hybrid ladder (answers: **is the full TT stem required**, or can we **move work to CPU**
and still reproduce the biased-linear failure on TT?):

In ``sample_actions`` terms: **prefix** (vision+lang → KV), **first denoise** (updates
``noise → x_t`` once), **second-step suffix** (here: ``state_proj`` + time embedding +
``action_in_proj`` with bias). The two hybrid tests keep **second-step biased linear**
on TT while moving **prefix + first denoise** off the TT graph (values from CPU).

- ``test_pi0_hybrid_linear_bias_cpu_xt_only`` — **No** prefix and **no** first denoise on
  TT; only state + time + biased ``action_in_proj``. Inputs ``x_t_mid``, ``time1`` come
  from CPU. When this **passes** PCC but the full-bracket with bias **throws** (as in
  ``trim_failing_sanity``), the concat failure **does** require **TT first denoise**
  (TT-produced ``x_t``) in the fused graph — CPU ``x_t_mid`` is not enough to reproduce it.

- ``test_pi0_hybrid_linear_bias_tt_prefix_cpu_xt`` — **Prefix KV on TT**, first denoise
  still **off** TT (CPU ``x_t_mid``). If this **fails** like the full stem but
  ``..._cpu_xt_only`` **passes**, **TT prefix** participates in the minimal failing graph;
  if it **passes** like ``..._cpu_xt_only``, prefix on TT alone is still insufficient vs
  full stem.

Full TT stem (prefix + first denoise + suffix time + biased linear on TT) is
``test_pi0_bracket_action_in_proj_matmul_no_bias`` / ``..._with_bias_expects_tt_failure``.

If both hybrids **pass** PCC while the with-bias bracket **expects** the TT throw,
the smallest failing graph **includes first denoise on TT** (TT-produced ``x_t``),
not only CPU-frozen ``x_t_mid``—matching earlier “CPU midpoint + TT slice” runs that
did not reproduce the concat error.

**Stem CPU vs TT (what each test fixes)**

+-------------------------------+--------------+----------------+---------------------------+
| Test                          | Prefix (A)   | 1st denoise (B)| TT traced body            |
+===============================+==============+================+===========================+
| ``unit_biased_linear_only``   | CPU          | CPU            | only ``F.linear(x,W,b)`` |
+-------------------------------+--------------+----------------+---------------------------+
| ``micro_biased_linear_cpu_xt``| CPU (inputs) | CPU            | ``F.linear`` + tiny residual |
+-------------------------------+--------------+----------------+---------------------------+
| ``hybrid_linear_bias_cpu_xt`` | CPU          | CPU            | state + time + biased lin |
+-------------------------------+--------------+----------------+---------------------------+
| ``hybrid_tt_prefix_cpu_xt``   | TT           | CPU            | prefix + state + time + … |
+-------------------------------+--------------+----------------+---------------------------+
| ``bracket_action_in_proj_*``  | TT           | TT             | full stem + linear       |
+-------------------------------+--------------+----------------+---------------------------+

``unit_*`` is ``F.linear`` only; ``micro_*`` adds a tiny weight residual. **Decomposed vs fused** on TT reproduces the same
device failure; ``test_pi0_bracket_decomposed_bias_add_matches_fused_on_cpu`` checks
``F.linear(x,w,None)+b`` equals ``F.linear(x,w,b)`` on CPU only (avoids PJRT error 13
after a bad TT run during PCC).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from infra import Framework, run_op_test
from tests.torch.ops.pi0_pipeline_shared import pi0_torch_cumsum_patch_like_model_py_forward
from tests.torch.ops.test_pi0_second_embed_suffix_incremental_tt import (
    _create_sinusoidal_pos_embedding,
    _state_for_suffix,
)
from tests.torch.ops.test_pi0_two_denoise_tt_fullgraph_bisect import (
    _prefix_and_first_denoise,
    _prefix_pad_masks_and_pkv,
)
from utils import Category


def _detach_cpu_tensors(obj):
    """``images`` in ``pi0_bundle`` may be a ``list`` of tensors; move leaves to CPU."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, list):
        return [_detach_cpu_tensors(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_detach_cpu_tensors(x) for x in obj)
    return obj


def _tiny_residual_from_tensors(scale: float, *parts: torch.Tensor) -> torch.Tensor:
    """Scalar tie-in without ``torch.mean`` (TTIR can reject some ``mean`` lowers)."""
    acc = parts[0].reshape(-1)[0] * 0.0
    for p in parts:
        acc = acc + p.reshape(-1)[0].to(dtype=acc.dtype)
    return scale * acc


def _is_expected_tt_bias_concat_class_failure(exc: BaseException) -> bool:
    """True if ``exc`` matches the known Pi0 ``action_in_proj`` + bias TT failure class."""
    parts: list[str] = []
    cur: BaseException | None = exc
    while cur is not None:
        parts.append(f"{type(cur).__name__}: {cur}")
        cur = cur.__cause__ or cur.__context__
    text = "\n".join(parts).lower()
    return any(
        needle in text
        for needle in (
            "shapebase",
            "ttnn::concat",
            "tt_throw",
            "error code: 13",
            "concatop",
            " index out of range",
        )
    )


def _cpu_xt_and_time1_after_first_denoise(core, images, img_masks, lang_tokens, lang_masks, state, noise):
    """Match ``num_steps == 2`` schedule; CPU ``no_grad`` for hybrid inputs.

    After other TT tests, ``core`` may live on XLA while bundle tensors stay on CPU,
    which breaks ``embed_prefix`` (FloatTensor vs XLAFloatType). Run the reference
    forward on CPU and restore the model device afterward.
    """
    with pi0_torch_cumsum_patch_like_model_py_forward():
        with torch.no_grad():
            param_dev = next(core.parameters()).device
            if param_dev.type != "cpu":
                core.to("cpu")
                images = _detach_cpu_tensors(images)
                img_masks = _detach_cpu_tensors(img_masks)
                lang_tokens = _detach_cpu_tensors(lang_tokens)
                lang_masks = _detach_cpu_tensors(lang_masks)
                state = _detach_cpu_tensors(state)
                noise = _detach_cpu_tensors(noise)
                try:
                    _ppm, _pkv, x_t, time1 = _prefix_and_first_denoise(
                        core,
                        images,
                        img_masks,
                        lang_tokens,
                        lang_masks,
                        state,
                        noise,
                        num_steps=2,
                    )
                finally:
                    core.to(param_dev)
            else:
                _ppm, _pkv, x_t, time1 = _prefix_and_first_denoise(
                    core,
                    images,
                    img_masks,
                    lang_tokens,
                    lang_masks,
                    state,
                    noise,
                    num_steps=2,
                )
    return x_t, time1


def _stem_after_tt_first_denoise_through_time_emb(
    m: torch.nn.Module,
    images,
    img_masks,
    lang_tokens,
    lang_masks,
    state,
    noise,
    num_steps: int,
):
    """Prefix + first denoise on TT, then ``embed_suffix`` state + time sinusoid."""
    with pi0_torch_cumsum_patch_like_model_py_forward():
        _ppm, _pkv, x_t, time1 = _prefix_and_first_denoise(
            m,
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            noise,
            num_steps=num_steps,
        )
        state_in = _state_for_suffix(m, state)

        def state_proj_func(s):
            return m.state_proj(s)

        state_emb = m._apply_checkpoint(state_proj_func, state_in)
        state_token = state_emb[:, None, :]

        time_emb = _create_sinusoidal_pos_embedding(
            time1,
            m.action_in_proj.out_features,
            min_period=m.config.min_period,
            max_period=m.config.max_period,
            device=time1.device,
        )
        time_emb = time_emb.type(dtype=time1.dtype)
    return x_t, time1, state_token, time_emb


class BracketThroughLinearMatmulNoBias(torch.nn.Module):
    """TT: stem through time emb, then ``F.linear(x_t, weight, None)``."""

    def __init__(self, core: torch.nn.Module, num_steps: int = 2):
        super().__init__()
        self.core = core
        self._num_steps = num_steps

    def forward(self, images, img_masks, lang_tokens, lang_masks, state, noise):
        m = self.core
        x_t, time1, state_token, time_emb = _stem_after_tt_first_denoise_through_time_emb(
            m,
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            noise,
            self._num_steps,
        )
        lin = m.action_in_proj
        y = F.linear(x_t, lin.weight, None)
        return y + _tiny_residual_from_tensors(1e-9, state_token, time_emb)


class BracketThroughLinearWithBias(torch.nn.Module):
    """TT: same stem, then ``F.linear(x_t, weight, bias)`` (bias path)."""

    def __init__(self, core: torch.nn.Module, num_steps: int = 2):
        super().__init__()
        self.core = core
        self._num_steps = num_steps

    def forward(self, images, img_masks, lang_tokens, lang_masks, state, noise):
        m = self.core
        x_t, time1, state_token, time_emb = _stem_after_tt_first_denoise_through_time_emb(
            m,
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            noise,
            self._num_steps,
        )
        lin = m.action_in_proj
        y = F.linear(x_t, lin.weight, lin.bias)
        return y + _tiny_residual_from_tensors(1e-9, state_token, time_emb)


class HybridLinearBiasCpuXtOnly(torch.nn.Module):
    """TT: only state + time + biased linear; ``x_t_mid`` / ``time1`` are inputs (CPU-fed)."""

    def __init__(self, core: torch.nn.Module):
        super().__init__()
        self.core = core

    def forward(self, state, x_t_mid, time1):
        m = self.core
        with pi0_torch_cumsum_patch_like_model_py_forward():
            state_in = _state_for_suffix(m, state)

            def state_proj_func(s):
                return m.state_proj(s)

            state_emb = m._apply_checkpoint(state_proj_func, state_in)
            state_token = state_emb[:, None, :]

            time_emb = _create_sinusoidal_pos_embedding(
                time1,
                m.action_in_proj.out_features,
                min_period=m.config.min_period,
                max_period=m.config.max_period,
                device=time1.device,
            )
            time_emb = time_emb.type(dtype=time1.dtype)

            lin = m.action_in_proj
            y = F.linear(x_t_mid, lin.weight, lin.bias)
        return y + _tiny_residual_from_tensors(1e-9, state_token, time_emb)


class UnitBiasedActionInProjLinearOnly(torch.nn.Module):
    """Single op on TT: ``F.linear(x, action_in_proj.weight, action_in_proj.bias)`` — no extras."""

    def __init__(self, core: torch.nn.Module):
        super().__init__()
        self.core = core

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lin = self.core.action_in_proj
        return F.linear(x, lin.weight, lin.bias)


class MicroBiasedLinearCpuXt(torch.nn.Module):
    """Smallest useful TT slice: only ``F.linear(x_t_mid, weight, bias)``.

    Prefix and first denoise run **only** to build ``x_t_mid`` off-graph (CPU); the
    traced forward is the biased linear on Pi0 ``action_in_proj`` weights.
    """

    def __init__(self, core: torch.nn.Module):
        super().__init__()
        self.core = core

    def forward(self, x_t_mid: torch.Tensor) -> torch.Tensor:
        lin = self.core.action_in_proj
        return F.linear(x_t_mid, lin.weight, lin.bias) + 1e-15 * lin.weight.mean()


class BracketDecomposedBiasAdd(torch.nn.Module):
    """Same numeric result as biased linear; bias applied via explicit ``+ bias``.

    Compare to ``BracketThroughLinearWithBias``: if matmul-only passes and both fused
    and decomposed-with-bias fail, inspect **how** bias is lowered (concat/broadcast).
    """

    def __init__(self, core: torch.nn.Module, num_steps: int = 2):
        super().__init__()
        self.core = core
        self._num_steps = num_steps

    def forward(self, images, img_masks, lang_tokens, lang_masks, state, noise):
        m = self.core
        x_t, _time1, state_token, time_emb = _stem_after_tt_first_denoise_through_time_emb(
            m,
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            noise,
            self._num_steps,
        )
        lin = m.action_in_proj
        y = F.linear(x_t, lin.weight, None) + lin.bias
        return y + _tiny_residual_from_tensors(1e-9, state_token, time_emb)


class HybridLinearBiasTtPrefixCpuXt(torch.nn.Module):
    """TT: prefix KV + state/time + biased linear on **CPU** ``x_t_mid`` (prefix in graph)."""

    def __init__(self, core: torch.nn.Module):
        super().__init__()
        self.core = core

    def forward(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        x_t_mid,
        time1,
    ):
        m = self.core
        with pi0_torch_cumsum_patch_like_model_py_forward():
            prefix_pad_masks, _pkv = _prefix_pad_masks_and_pkv(
                m, images, img_masks, lang_tokens, lang_masks
            )
            state_in = _state_for_suffix(m, state)

            def state_proj_func(s):
                return m.state_proj(s)

            state_emb = m._apply_checkpoint(state_proj_func, state_in)
            state_token = state_emb[:, None, :]

            time_emb = _create_sinusoidal_pos_embedding(
                time1,
                m.action_in_proj.out_features,
                min_period=m.config.min_period,
                max_period=m.config.max_period,
                device=time1.device,
            )
            time_emb = time_emb.type(dtype=time1.dtype)

            lin = m.action_in_proj
            y = F.linear(x_t_mid, lin.weight, lin.bias)
        pm = prefix_pad_masks.to(dtype=torch.float32)
        return y + _tiny_residual_from_tensors(
            1e-9, state_token, time_emb
        ) + _tiny_residual_from_tensors(1e-15, pm)


@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_pi0_bracket_action_in_proj_matmul_no_bias(request, pi0_bundle):
    """Bracket (1): linear **without** bias — expect TT PCC pass vs CPU."""
    core, _policy, (
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise,
    ) = pi0_bundle
    run_op_test(
        BracketThroughLinearMatmulNoBias(core, num_steps=2),
        [images, img_masks, lang_tokens, lang_masks, state, noise],
        framework=Framework.TORCH,
        request=request,
    )


@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_pi0_bracket_action_in_proj_with_bias_expects_tt_failure(request, pi0_bundle):
    """Bracket (2): full TT stem + biased linear — expect TT concat / ShapeBase class failure.

    ``run_op_test`` assumes PCC success; this path hits ``ttnn::concat`` (see logs).
    When the compiler/runtime is fixed, this test will fail (no exception); switch it
    back to ``run_op_test`` with ``BracketThroughLinearWithBias``.
    """
    core, _policy, (
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise,
    ) = pi0_bundle
    with pytest.raises(Exception) as excinfo:
        run_op_test(
            BracketThroughLinearWithBias(core, num_steps=2),
            [images, img_masks, lang_tokens, lang_masks, state, noise],
            framework=Framework.TORCH,
            request=request,
        )
    assert _is_expected_tt_bias_concat_class_failure(excinfo.value), (
        f"Expected TT concat/bias-class failure; got {type(excinfo.value).__name__}: {excinfo.value!r}"
    )


@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_pi0_unit_biased_action_in_proj_linear_only(request, pi0_bundle):
    """Unit: Pi0 ``action_in_proj`` **only** — ``F.linear(x, W, b)``, ``x`` from CPU stem."""
    core, _policy, (
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise,
    ) = pi0_bundle
    x_mid, _t1 = _cpu_xt_and_time1_after_first_denoise(
        core, images, img_masks, lang_tokens, lang_masks, state, noise
    )
    run_op_test(
        UnitBiasedActionInProjLinearOnly(core),
        [x_mid],
        framework=Framework.TORCH,
        request=request,
    )


@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_pi0_micro_biased_linear_cpu_xt(request, pi0_bundle):
    """TT graph ≈ single biased linear; ``x_t`` midpoint from CPU stem."""
    core, _policy, (
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise,
    ) = pi0_bundle
    x_mid, _t1 = _cpu_xt_and_time1_after_first_denoise(
        core, images, img_masks, lang_tokens, lang_masks, state, noise
    )
    run_op_test(
        MicroBiasedLinearCpuXt(core),
        [x_mid],
        framework=Framework.TORCH,
        request=request,
    )


@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_pi0_bracket_decomposed_bias_add_matches_fused_on_cpu(pi0_bundle):
    """CPU: ``F.linear(x,w,None)+b`` matches ``F.linear(x,w,b)`` on the same stem.

    A TT ``run_op_test`` for the decomposed form hits the same ``ttnn::concat`` failure
    as ``BracketThroughLinearWithBias``, then PCC can error with PJRT code 13 when
    syncing XLA outputs after a bad device run. This test pins the **numeric** split
    without that comparison path.
    """
    core, _policy, (
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise,
    ) = pi0_bundle
    dec = BracketDecomposedBiasAdd(core, num_steps=2)
    fused = BracketThroughLinearWithBias(core, num_steps=2)
    param_dev = next(core.parameters()).device
    if param_dev.type != "cpu":
        core.to("cpu")
        images = _detach_cpu_tensors(images)
        img_masks = _detach_cpu_tensors(img_masks)
        lang_tokens = _detach_cpu_tensors(lang_tokens)
        lang_masks = _detach_cpu_tensors(lang_masks)
        state = _detach_cpu_tensors(state)
        noise = _detach_cpu_tensors(noise)
    try:
        with pi0_torch_cumsum_patch_like_model_py_forward():
            with torch.no_grad():
                o_dec = dec(images, img_masks, lang_tokens, lang_masks, state, noise)
                o_fused = fused(images, img_masks, lang_tokens, lang_masks, state, noise)
        assert torch.allclose(o_dec, o_fused, rtol=1e-4, atol=1e-4)
    finally:
        if param_dev.type != "cpu":
            core.to(param_dev)


@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_pi0_hybrid_linear_bias_cpu_xt_only(request, pi0_bundle):
    """Hybrid: prefix + 1st denoise on **CPU** only; TT = state + time + biased linear.

    If this **passes** PCC while ``..._with_bias_expects_tt_failure`` **throws** (see logs),
    the concat bug needs **TT first denoise** in the full graph, not only CPU ``x_t_mid``.
    """
    core, _policy, (
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise,
    ) = pi0_bundle
    x_mid, t1 = _cpu_xt_and_time1_after_first_denoise(
        core, images, img_masks, lang_tokens, lang_masks, state, noise
    )
    run_op_test(
        HybridLinearBiasCpuXtOnly(core),
        [state, x_mid, t1],
        framework=Framework.TORCH,
        request=request,
    )


@pytest.mark.single_device
@pytest.mark.record_test_properties(category=Category.OP_TEST)
def test_pi0_hybrid_linear_bias_tt_prefix_cpu_xt(request, pi0_bundle):
    """Hybrid: **TT prefix KV** + suffix path + biased linear; ``x_t_mid`` from **CPU**.

    If this fails like the full stem but ``..._cpu_xt_only`` passes, **TT prefix** is
    in the minimal reproducer; first denoise may still be CPU-only.
    """
    core, _policy, (
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise,
    ) = pi0_bundle
    x_mid, t1 = _cpu_xt_and_time1_after_first_denoise(
        core, images, img_masks, lang_tokens, lang_masks, state, noise
    )
    run_op_test(
        HybridLinearBiasTtPrefixCpuXt(core),
        [images, img_masks, lang_tokens, lang_masks, state, x_mid, t1],
        framework=Framework.TORCH,
        request=request,
    )

