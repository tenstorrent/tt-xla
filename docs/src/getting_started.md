# Getting Started
This document walks you through how to set up TT-XLA. TT-XLA is a front end for TT-Forge that is primarily used to ingest JAX models via jit compile, providing a StableHLO (SHLO) graph to the TT-MLIR compiler. TT-XLA leverages [PJRT](https://github.com/openxla/xla/tree/main/xla/pjrt/c#pjrt---uniform-device-api) to integrate JAX, [TT-MLIR](https://github.com/tenstorrent/tt-mlir) and Tenstorrent hardware. Please see [this](https://opensource.googleblog.com/2023/05/pjrt-simplifying-ml-hardware-and-framework-integration.html) blog post for more information about PJRT project. This project started as a fork of [iree-pjrt](https://github.com/stellaraccident/iree-pjrt), but has since been refactored and diverged.

> **NOTE:** Currently, only Tenstorrent Nebula boards are supported.

This is the main Getting Started page. There are two additional Getting Started pages depending on what you want to do. They are all described here, with links provided to each.

The following topics are covered:

* [Setup Options](#setup-options)
* [Configuring Hardware](#configuring-hardware)
* [Installing a Wheel and Running an Example](#installing-a-wheel-and-running-an-example)
* [Other Setup Options](#other-set-up-options)
   * [Using a Docker Container to Run an Example](getting_started_docker.md)
   * [Building From Source](getting_started_build_from_source.md)
* [Where to Go Next](#where-to-go-next)

> **NOTE:** If you encounter issues, please request assistance on the
>[TT-XLA Issues](https://github.com/tenstorrent/tt-xla/issues) page.

## Setup Options
TT-XLA can be used to run JAX models on Tenstorrent's AI hardware. Because TT-XLA is open source, you can also develop and add features to it. Setup instructions differ based on the task. You have the following options, listed in order of difficulty:
* [Installing a Wheel and Running an Example](#installing-a-wheel-and-running-an-example) - You should choose this option if you want to run models.
* [Using a Docker Container to Run an Example](https://docs.tenstorrent.com/tt-xla/getting_started_docker.html) - Choose this option if you want to keep the environment for running models separate from your existing environment.
* [Building from Source](https://docs.tenstorrent.com/tt-xla/getting_started_build_from_source.html) - This option is best if you want to develop TT-XLA further. It's a more complex process you are unlikely to need if you want to stick with running a model.

## Configuring Hardware
Before setup can happen, you must configure your hardware. You can skip this section if you already completed the configuration steps. Otherwise, this section of the walkthrough shows you how to do a quick setup using TT-Installer.

1. Configure your hardware with TT-Installer using the [Software Installation section here.](https://docs.tenstorrent.com/getting-started/README.html#software-installation)

2. Reboot your machine.

3. Make sure **hugepages** is enabled:

```bash
sudo systemctl enable --now 'dev-hugepages\x2d1G.mount'
sudo systemctl enable --now tenstorrent-hugepages.service
```

4. Please ensure that after you run the TT-Installer script, after you complete reboot and set up hugepages, you activate the virtual environment it sets up - ```source ~/.tenstorrent-venv/bin/activate```.

5. After your environment is running, to check that everything is configured, type the following:

```bash
tt-smi
```

You should see the Tenstorrent System Management Interface. It allows you to view real-time stats, diagnostics, and health info about your Tenstorrent device.

![TT-SMI](./imgs/tt_smi.png)

## Installing a Wheel and Running an Example

To install a wheel and run an example model, do the following:

1. Make sure you are in an active virtual environment. This walkthrough uses the same environment you activated to look at TT-SMI in the [Configuring Hardware](#configuring-hardware) section. If you are using multiple TT-Forge front ends to run models, you may want to set up a separate virtual environment instead. For example:

```bash
python3 -m venv .xla-venv
source .xla-venv/bin/activate
```

2. Install the wheel in your active virtual environment:

```bash
pip install pjrt-plugin-tt --extra-index-url https://pypi.eng.aws.tenstorrent.com/
```

>**NOTE:** You can pull pre-releases (these may not be stable, so proceed with caution) by adding the `--pre` flag
> directly after `pip install`.
> You can also choose a wheel from the [nightly release page](https://github.com/tenstorrent/tt-xla/releases).

3. You are now ready to try running a model. Navigate to the section of the [TT-Forge repo that contains TT-XLA demos](https://github.com/tenstorrent/tt-forge/tree/main/demos/tt-xla).

4. For this walkthrough, the [demo in the **gpt2** folder](https://github.com/tenstorrent/tt-forge/tree/main/demos/tt-xla/gpt2) is used. In the **gpt2** folder, in the **requirements.txt** file, you can see that **flax** and **transformers** are necessary to run the demo. Install them:

```bash
pip install flax transformers
```

5. Download the [**demo.py** file from the **gpt2** folder](https://github.com/tenstorrent/tt-forge/blob/main/demos/tt-xla/gpt2/demo.py) inside your activated virtual environment in a place where you can run it. The demo you are about to run takes a piece of text and tries to predict the next word that logically follows.

6. Run the model:

```bash
python demo.py
```

7. If all goes well you should see the prompt "The capital of France is", the predicted next token, the probability it will occur, and a list of other ranked options that could follow instead.

## Other Setup Options
If you want to keep your environment completely separate in a Docker container, or you want to develop TT-XLA further, this section links you to the pages with those options:

* [Setting up a Docker Container](getting_started_docker.md)
* [Building from Source](getting_started_build_from_source.md)

## Where to Go Next

Now that you have set up the TT-XLA wheel, you can compile and run other demos. See the [TT-XLA folder in the TT-Forge repo](https://github.com/tenstorrent/tt-forge/tree/main/demos/tt-xla) for other demos you can try.
