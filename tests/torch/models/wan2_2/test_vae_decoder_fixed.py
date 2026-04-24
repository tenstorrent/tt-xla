# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
Wan 2.2 TI2V-5B VAE decoder test with graph-break monkey patches applied.

Same test as test_vae_decoder.py, but patches diffusers' WanResample.forward
to replace the "Rep" string sentinel (which triggers a dynamo graph break on
Tensor.__ne__(str) / Tensor.__eq__(str)) with an object-identity sentinel.

See tmp/vae_decoder_480p_graph_break_report.md for the full analysis.
"""

import torch
import torch_xla
import torch_xla.runtime as xr
from diffusers.models.autoencoders import autoencoder_kl_wan as _akw
from infra import Framework, run_graph_test
from infra.evaluators import ComparisonConfig, PccConfig
from tests.infra.testers.compiler_config import CompilerConfig

from .shared import RESOLUTIONS, load_vae, shard_vae_decoder_specs, wan22_mesh

_CACHE_T = _akw.CACHE_T
_REP = object()


def _wan_resample_forward_patched(self, x, feat_cache=None, feat_idx=[0]):
    b, c, t, h, w = x.size()
    if self.mode == "upsample3d":
        if feat_cache is not None:
            idx = feat_idx[0]
            if feat_cache[idx] is None:
                feat_cache[idx] = _REP
                feat_idx[0] += 1
            else:
                cache_x = x[:, :, -_CACHE_T:, :, :].clone()
                if (
                    cache_x.shape[2] < 2
                    and feat_cache[idx] is not None
                    and feat_cache[idx] is not _REP
                ):
                    cache_x = torch.cat(
                        [
                            feat_cache[idx][:, :, -1, :, :]
                            .unsqueeze(2)
                            .to(cache_x.device),
                            cache_x,
                        ],
                        dim=2,
                    )
                if (
                    cache_x.shape[2] < 2
                    and feat_cache[idx] is not None
                    and feat_cache[idx] is _REP
                ):
                    cache_x = torch.cat(
                        [torch.zeros_like(cache_x).to(cache_x.device), cache_x], dim=2
                    )
                if feat_cache[idx] is _REP:
                    x = self.time_conv(x)
                else:
                    x = self.time_conv(x, feat_cache[idx])
                feat_cache[idx] = cache_x
                feat_idx[0] += 1

                x = x.reshape(b, 2, c, t, h, w)
                x = torch.stack((x[:, 0, :, :, :, :], x[:, 1, :, :, :, :]), 3)
                x = x.reshape(b, c, t * 2, h, w)
    t = x.shape[2]
    x = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
    x = self.resample(x)
    x = x.view(b, t, x.size(1), x.size(2), x.size(3)).permute(0, 2, 1, 3, 4)

    if self.mode == "downsample3d":
        if feat_cache is not None:
            idx = feat_idx[0]
            if feat_cache[idx] is None:
                feat_cache[idx] = x.clone()
                feat_idx[0] += 1
            else:
                cache_x = x[:, :, -1:, :, :].clone()
                x = self.time_conv(
                    torch.cat([feat_cache[idx][:, :, -1:, :, :], x], 2)
                )
                feat_cache[idx] = cache_x
                feat_idx[0] += 1
    return x


_akw.WanResample.forward = _wan_resample_forward_patched


class VAEDecoderWrapper(torch.nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, z):
        return self.vae.decode(z, return_dict=False)[0]


def test_vae_decoder_480p_fixed():
    _run(resolution="480p", sharded=False)


def test_vae_decoder_720p_fixed():
    _run(resolution="720p", sharded=False)


def test_vae_decoder_480p_sharded_fixed():
    _run(resolution="480p", sharded=True)


def test_vae_decoder_720p_sharded_fixed():
    _run(resolution="720p", sharded=True)


def _run(resolution: str, sharded: bool):
    xr.set_device_type("TT")
    compiler_config = CompilerConfig(
        optimization_level=1,
        experimental_enable_dram_space_saving_optimization=True,
        export_path="model",
        export_model_name="vae_decoder",
    )
    torch.manual_seed(42)
    shapes = RESOLUTIONS[resolution]

    wrapper = VAEDecoderWrapper(load_vae()).eval().bfloat16()

    z = torch.randn(
        1,
        48,
        shapes["latent_frames"],
        shapes["latent_h"],
        shapes["latent_w"],
        dtype=torch.bfloat16,
    )

    mesh = wan22_mesh() if sharded else None
    shard_spec_fn = (lambda m: shard_vae_decoder_specs(m.vae)) if sharded else None

    run_graph_test(
        wrapper,
        [z],
        framework=Framework.TORCH,
        mesh=mesh,
        shard_spec_fn=shard_spec_fn,
        compiler_config=compiler_config,
        comparison_config=ComparisonConfig(pcc=PccConfig(required_pcc=0.99)),
    )
