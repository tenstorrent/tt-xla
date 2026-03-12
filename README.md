[![Tests][tests badge]][tests]
[![Codecov][codecov badge]][codecov]
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/tenstorrent/tt-xla)

<div align="center">

<h1>

[Hardware](https://tenstorrent.com/cards/) | [Documentation](https://docs.tenstorrent.com/tt-xla/) | [Discord](https://discord.gg/tenstorrent) | [Join Us](https://job-boards.greenhouse.io/tenstorrent?gh_src=22e462047us) | [Bounty $](https://github.com/tenstorrent/tt-xla/issues?q=is%3Aissue%20state%3Aopen%20label%3Abounty)

</h1>
<picture>
  <img alt="Logo" src="docs/src/imgs/tt_xla_logo.png" height="250">
</picture>

</div>
<br>

Run **PyTorch** and **JAX** models on Tenstorrent hardware. Use `torch.compile()` or `jax.jit()` and TT-XLA handles the rest — compiling your model through [TT-MLIR](https://github.com/tenstorrent/tt-mlir) and executing it on Tenstorrent accelerators. Supports single-chip and multi-chip configurations via the [PJRT](https://opensource.googleblog.com/2023/05/pjrt-simplifying-ml-hardware-and-framework-integration.html) interface.

> **Part of the [TT-Forge](https://github.com/tenstorrent/tt-forge) AI compiler ecosystem.**

-----
# Run a Model

Install TT-XLA and run ResNet-50 on Tenstorrent hardware:

```bash
pip install pjrt-plugin-tt --extra-index-url https://pypi.eng.aws.tenstorrent.com/
pip install torchvision
```

```python
import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from tt_torch.backend.backend import xla_backend
from torchvision.models import resnet50, ResNet50_Weights

# Set device to Tenstorrent
xr.set_device_type("TT")
device = xm.xla_device()

# Load ResNet-50
model = resnet50(weights=ResNet50_Weights.DEFAULT).to(torch.bfloat16).eval()
compiled_model = torch.compile(model, backend=xla_backend)
compiled_model = compiled_model.to(device)

# Run inference on Tenstorrent
input_tensor = torch.randn(1, 3, 224, 224, dtype=torch.bfloat16).to(device)
with torch.no_grad():
    output = compiled_model(input_tensor)

predicted_class = output.cpu().argmax(dim=-1).item()
print(f"Predicted ImageNet class: {predicted_class}")
```

See the full [Getting Started Guide](docs/src/getting_started.md) for Docker, build-from-source, and JAX setup options.

-----
# Quick Links
- [Getting Started / How to Run a Model](docs/src/getting_started.md)
- [Demos](https://github.com/tenstorrent/tt-forge/tree/main/demos/tt-xla) — Ready-to-run models (ResNet, GPT-2, OPT, ALBERT, and more)
- [Benchmarks](https://github.com/tenstorrent/tt-forge/tree/main/benchmark/tt-xla) — Performance benchmarks

-----
# Supported Hardware
| Device | Configurations |
|--------|---------------|
| **Wormhole** | N150 (single-chip), N300 (dual-chip) |
| **Blackhole** | P150B |

-----
# Related Tenstorrent Projects
- [TT-Forge](https://github.com/tenstorrent/tt-forge) — Central hub for the TT-Forge compiler project (demos, benchmarks, releases)
- [TT-Forge-ONNX](https://github.com/tenstorrent/tt-forge-onnx) — Frontend for ONNX, TensorFlow, and PaddlePaddle (single-chip)
- [TT-MLIR](https://github.com/tenstorrent/tt-mlir) — Core MLIR-based compiler framework for Tenstorrent hardware
- [TT-Metal](https://github.com/tenstorrent/tt-metal) — Low-level programming model and kernel development for Tenstorrent hardware

-----
# Tenstorrent Bounty Program Terms and Conditions
This repo is a part of Tenstorrent’s bounty program. If you are interested in helping to improve tt-forge, please make sure to read the [Tenstorrent Bounty Program Terms and Conditions](https://docs.tenstorrent.com/bounty_terms.html) before heading to the issues tab. Look for the issues that are tagged with both “bounty” and difficulty level!

[codecov]: https://codecov.io/gh/tenstorrent/tt-xla
[tests]: https://github.com/tenstorrent/tt-xla/actions/workflows/on-push.yml?query=branch%3Amain
[codecov badge]: https://codecov.io/gh/tenstorrent/tt-xla/graph/badge.svg?token=XQJ3JVKIRI
[tests badge]: https://github.com/tenstorrent/tt-xla/actions/workflows/on-push.yml/badge.svg?query=branch%3Amain
[deepwiki]: https://deepwiki.com/tenstorrent/tt-xla
[deepwiki badge]: https://deepwiki.com/badge.svg
