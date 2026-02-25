# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple


@dataclass
class ConfigBase:
    enabled: bool = True

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False


@dataclass
class EqualConfig(ConfigBase):
    pass


@dataclass
class AtolConfig(ConfigBase):
    required_atol: float = 1.6e-1


@dataclass
class AllcloseConfig(ConfigBase):
    rtol: float = 1e-2
    atol: float = 1e-2


@dataclass
class PccConfig(ConfigBase):
    required_pcc: float = 0.99
    # When tensors are too close, pcc will output NaN values. To prevent that, we do
    # allclose comparison in that case. For each test it should be possible to
    # separately tune the allclose config for which pcc won't be calculated and
    # therefore test will be able to pass without pcc comparison.
    allclose: AllcloseConfig = field(default_factory=AllcloseConfig)


@dataclass
class ComparisonConfig:
    equal: EqualConfig = field(default_factory=lambda: EqualConfig(False))
    atol: AtolConfig = field(default_factory=lambda: AtolConfig(False))
    pcc: PccConfig = field(default_factory=PccConfig)
    allclose: AllcloseConfig = field(default_factory=lambda: AllcloseConfig(False))
    assert_on_failure: bool = True  # Default to True for backwards compatibility

    def enable_all(self) -> None:
        self.equal.enable()
        self.atol.enable()
        self.allclose.enable()
        self.pcc.enable()

    def disable_all(self) -> None:
        self.equal.disable()
        self.atol.disable()
        self.allclose.disable()
        self.pcc.disable()


@dataclass
class QualityConfig:
    """Base quality config with common behavior settings.

    Subclass this for domain-specific quality configurations with
    their own thresholds and check_thresholds() implementations.
    """

    assert_on_failure: bool = True

    def check_thresholds(self, metrics: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Check computed metrics against configured thresholds.

        Override in subclasses to implement metric-specific threshold checking.

        Args:
            metrics: Dictionary of computed metric values

        Returns:
            Tuple of (passed: bool, error_message: Optional[str])
        """
        return True, None  # Base class passes everything


@dataclass
class ImageGenQualityConfig(QualityConfig):
    """Quality config for image generation models (SDXL, Stable Diffusion, etc.).

    Supports CLIP (image-text alignment) and FID (distribution quality) metrics.
    """

    min_clip_threshold: float = 25.0  # CLIP score: higher is better
    max_fid_threshold: float = 350.0  # FID score: lower is better

    def check_thresholds(self, metrics: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Check CLIP and FID metrics against thresholds."""
        errors = []
        passed = True

        if "clip_min" in metrics:
            # higher is better for CLIP score
            if metrics["clip_min"] < self.min_clip_threshold:
                passed = False
                errors.append(
                    f"CLIP quality check failed. "
                    f"Calculated: clip_min={metrics['clip_min']:.2f}. "
                    f"Required: min_clip_threshold={self.min_clip_threshold:.2f}."
                )

        if "fid" in metrics:
            # lower is better for FID score
            if metrics["fid"] > self.max_fid_threshold:
                passed = False
                errors.append(
                    f"FID quality check failed. "
                    f"Calculated: fid={metrics['fid']:.2f}. "
                    f"Required: max_fid_threshold={self.max_fid_threshold:.2f}."
                )

        return passed, " ".join(errors) if errors else None
