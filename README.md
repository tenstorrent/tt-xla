[![Tests][tests badge]][tests]
[![Codecov][codecov badge]][codecov]
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/tenstorrent/tt-xla)

<div align="center">

<h1>

[Hardware](https://tenstorrent.com/cards/) | [Documentation](docs/src) | [Discord](https://discord.gg/tenstorrent) | [Join Us](https://job-boards.greenhouse.io/tenstorrent?gh_src=22e462047us)

</h1>
<picture>
  <img alt="Logo" src="docs/src/imgs/tt_xla_logo.png" height="250">
</picture>

</div>
<br>

TT-XLA leverages a PJRT interface to integrate JAX (and in the future other frameworks), `tt-mlir` and Tenstorrent hardware. It supports ingestion of JAX models via jit compile, providing a StableHLO (SHLO) graph to `tt-mlir` compiler.

-----
# Quick Links
- [Getting Started / How to Run a Model](docs/src/getting_started.md)

-----
# What is This Repo?
The tt-xla repository is primarily used to enable running JAX models on Tenstorrent's AI hardware. It's a backend integration between the JAX ecosystem and Tenstorrent's ML accelerators using the PJRT (Portable JAX Runtime) interface.

-----
# Current AI Framework Front End Projects
- [TT-Forge-FE](https://github.com/tenstorrent/tt-forge-fe)
  - A TVM based graph compiler designed to optimize and transform computational graphs for deep learning models. Supports ingestion of PyTorch, ONNX, TensorFlow, PaddlePaddle and similar ML frameworks via TVM ([TT-TVM](https://github.com/tenstorrent/tt-tvm)).
  - See [docs pages](https://github.com/tenstorrent/tt-forge-fe/blob/main/docs/src/getting_started.md) for an overview and getting started guide.

- [TT-Torch](https://github.com/tenstorrent/tt-torch)
  - A MLIR-native, open-source, PyTorch 2.X and torch-mlir based front-end. It provides stableHLO (SHLO) graphs to `tt-mlir`. Supports ingestion of PyTorch models via PT2.X compile and ONNX models via torch-mlir (ONNX->SHLO)
  - See [docs pages](https://github.com/tenstorrent/tt-torch/blob/main/docs/src/getting_started.md) for an overview and getting started guide.

- [TT-XLA](https://github.com/tenstorrent/tt-xla)
  - Leverages a PJRT interface to integrate JAX (and in the future other frameworks), TT-MLIR, and Tenstorrent hardware. Supports ingestion of JAX models via jit compile, providing StableHLO (SHLO) graph to TT-MLIR compiler
  - See [docs pages](https://github.com/tenstorrent/tt-xla/blob/main/docs/src/getting_started.md) for an overview and getting started guide.

-----
# Related Tenstorrent Projects
- [TT-Forge](https://github.com/tenstorrent/tt-forge)
- [TT-Forge-FE](https://github.com/tenstorrent/tt-forge-fe)
- [TT-Torch](https://github.com/tenstorrent/tt-torch)
- [TT-MLIR](https://github.com/tenstorrent/tt-mlir)
- [TT-Metalium](https://github.com/tenstorrent/tt-metal)
- [TT-TVM](https://github.com/tenstorrent/tt-tvm)

-----
# Tenstorrent Bounty Program Terms and Conditions
This repo is a part of Tenstorrent’s bounty program. If you are interested in helping to improve tt-forge, please make sure to read the [Tenstorrent Bounty Program Terms and Conditions](https://docs.tenstorrent.com/bounty_terms.html) before heading to the issues tab. Look for the issues that are tagged with both “bounty” and difficulty level!

[codecov]: https://codecov.io/gh/tenstorrent/tt-xla
[tests]: https://github.com/tenstorrent/tt-xla/actions/workflows/on-push.yml?query=branch%3Amain
[codecov badge]: https://codecov.io/gh/tenstorrent/tt-xla/graph/badge.svg?token=XQJ3JVKIRI
[tests badge]: https://github.com/tenstorrent/tt-xla/actions/workflows/on-push.yml/badge.svg?query=branch%3Amain
[deepwiki]: https://deepwiki.com/tenstorrent/tt-xla
[deepwiki badge]: https://deepwiki.com/badge.svg
