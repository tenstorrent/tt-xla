# Getting Started with tt-xla
tt-xla leverages [PJRT](https://github.com/openxla/xla/tree/main/xla/pjrt/c#pjrt---uniform-device-api) to integrate JAX (and in the future other frameworks), [tt-mlir](https://github.com/tenstorrent/tt-mlir) and Tenstorrent hardware. Please see [this](https://opensource.googleblog.com/2023/05/pjrt-simplifying-ml-hardware-and-framework-integration.html) blog post for more information about PJRT project. This project started as a fork of [iree-pjrt](https://github.com/stellaraccident/iree-pjrt) but has since then been refactored and diverged.

> **Note:** Currently only Tenstorrent `nebula` boards are supported and `galaxy` boards are not yet supported.

<<<<<<< HEAD
# Getting Started
This document walks you through how to set up TT-XLA. TT-XLA is a front end for TT-Forge that is primarily used to ingest JAX models via jit compile, providing a StableHLO (SHLO) graph to the TT-MLIR compiler. TT-XLA leverages [PJRT](https://github.com/openxla/xla/tree/main/xla/pjrt/c#pjrt---uniform-device-api) to integrate JAX, [tt-mlir](https://github.com/tenstorrent/tt-mlir) and Tenstorrent hardware. Please see [this](https://opensource.googleblog.com/2023/05/pjrt-simplifying-ml-hardware-and-framework-integration.html) blog post for more information about PJRT project. This project is a fork of [iree-pjrt](https://github.com/stellaraccident/iree-pjrt).
=======
## Installation with Docker

We provide Docker images with all Forge frontends and their dependencies preinstalled and ready to use.

1. Install drivers and tt-smi tool on the host machine using [tt-installer](https://github.com/tenstorrent/tt-installer) tool:
```bash
/bin/bash -c "$(curl -fsSL https://github.com/tenstorrent/tt-installer/releases/latest/download/install.sh)"
```

2. Install the latest tt-forge Docker image and run it:
```bash
# Pull the latest docker
docker pull ghcr.io/tenstorrent/tt-forge/tt-forge-slim:latest

# Run it
docker run -it --rm \
  --device /dev/tenstorrent \
  -v /dev/hugepages-1G:/dev/hugepages-1G \
  ghcr.io/tenstorrent/tt-forge/tt-forge-slim:latest
```

3. Activate the virtual environment which comes prepared in the docker:
```bash
source /home/forge/venv-tt-xla/bin/activate
```

This environment has the latest tt-xla wheel installed. In case you want to install the custom version of tt-xla wheel you can find them on our [release page](https://github.com/tenstorrent/tt-xla/releases) or [build them](#build-from-source) from source.

4. Run the demo scripts:
```
# Clone the tt-forge repo
git clone https://github.com/tenstorrent/tt-forge.git

# Demo scripts are located in `tt-forge/demos`, in folders per frontend.
# Here are some example tt-xla demo scripts you can run:
python tt-forge/demos/tt-xla/albert_base_v2/demo.py
python tt-forge/demos/tt-xla/gpt2/demo.py
python tt-forge/demos/tt-xla/opt_125m/demo.py
```

## Build from source
tt-xla integration with tt-mlir compiler is still in progress. Currently tt-xla it depends on tt-mlir toolchain for build. This build flow provides an easy way to experiment with tt-xla, StableHLO, and the tt-mlir infrastructure. The build process will be updated in the future to enhance the user experience.
>>>>>>> 54b8d12de4451e3b14f6028ea863b79e1b8e6c19

This is the main Getting Started page. There are two additional Getting Started pages depending on what you want to do. They are all described here, with links provided to each.

The following topics are covered:

* [Setup Options](#setup-options)
* [Configuring Hardware](#configuring-hardware)
* [Installing a Wheel and Running an Example](#installing-a-wheel-and-running-an-example)
* [Other Setup Options](#other-set-up-options)
   * [Using a Docker Container to Run an Example]
   * [Building From Source]
* [Where to Go Next]

> **NOTE:** If you encounter issues, please request assistance on the
>[TT-XLA Issues](https://github.com/tenstorrent/tt-xla/issues) page.

## Setup Options
TT-XLA can be used to run JAX models on Tenstorrent's AI hardware. Because TT-XLA is open source, you can also develop and add features to it. Setup instructions differ based on the task. You have the following options, listed in order of difficulty:
* [Installing a Wheel and Running an Example](#installing-a-wheel-and-running-an-example) - You should choose this option if you want to run models.
* [Using a Docker Container to Run an Example](getting_started_docker.md) - Choose this option if you want to keep the environment for running models separate from your existing environment.
* [Building from Source](getting_started_build_from_source.md) - This option is best if you want to develop TT-XLA further. It's a more complex process you are unlikely to need if you want to stick with running a model.

## Configuring Hardware
Before setup can happen, you must configure your hardware. You can skip this section if you already completed the configuration steps. Otherwise, this section of the walkthrough shows you how to do a quick setup using TT-Installer.

1. Configure your hardware with TT-Installer using the [Quick Installation section here.](https://docs.tenstorrent.com/getting-started/README.html#quick-installation)

2. Reboot your machine.

3. Please ensure that after you run this script, after you complete reboot, you activate the virtual environment it sets up - ```source ~/.tenstorrent-venv/bin/activate```.

4. After your environment is running, to check that everything is configured, type the following:

```bash
tt-smi
```

You should see the Tenstorrent System Management Interface. It allows you to view real-time stats, diagnostics, and health info about your Tenstorrent device.

![TT-SMI](./imgs/tt_smi.png)

## Installing a Wheel and Running an Example

This section walks you through downloading and installing a wheel. You can install the wheel wherever you would like if it is for running a model.

1. Make sure you are in an active virtual environment. This walkthrough uses the same environment you activated to look at TT-SMI in the [Configuring Hardware](#configuring-hardware) section. If you are using multiple TT-Forge front ends to run models, you may want to set up a separate virtual environment instead. For example:

```bash
python3 -m venv .xla-venv
source .xla-venv/bin/activate
```

2. Navigate to the [Tenstorrent Nightly Releases](https://github.com/tenstorrent/tt-forge/releases) page.

3. TT-XLA requires one wheel for set up, which will start with **tt-xla**. You can also download the source code for everything as a **.zip** or a **tar.gz** file. Scroll through the releases for the latest wheel that starts with **tt-xla**.

4. When you reach a TT-XLA wheel, look for the **Assets** section and click **Assets**.

5. Right click on the title of the listed asset. (For TT-XLA, it starts with **pjrt_plugin**.)

6. Download the wheel in your active virtual environment:

```bash
pip install NAME_OF_WHEEL
```

7. You are now ready to try running a model. Navigate to the section of the [TT-Forge repo that contains TT-XLA demos](https://github.com/tenstorrent/tt-forge/tree/main/demos/tt-xla).

8. For this walkthrough, the demo int the **gpt2** folder is used. In the **gpt2** folder, in the **requirements.txt** file, you can see that **flax** and **transformers** are necessary to run the demo. Install them:

```bash
pip install flax transformers
```

9. Download the [**demo.py** file from the **gpt2** folder](https://github.com/tenstorrent/tt-forge/blob/main/demos/tt-xla/gpt2/demo.py) inside your activated virtual environment in a place where you can run it. The demo you are about to run takes a piece of text and tries to predict the next word that logically follows.

10. If all goes well you should see the prompt "The capital of France is", the predicted next token, the probability it will occur, and a list of other ranked options that could follow instead.

## Other Set up Options
If you want to keep your environment completely separate in a Docker container, or you want to develop TT-XLA further, this section links you to the pages with those options:

* [Setting up a Docker Container](getting_started_docker.md)
* [Building from Source](getting_started_build_from_source.md)

## Where to Go Next

Now that you have set up the TT-XLA wheel, you can compile and run other demos. See the [TT-XLA folder in the TT-Forge repo](https://github.com/tenstorrent/tt-forge/tree/main/demos/tt-xla) for other demos you can try.
