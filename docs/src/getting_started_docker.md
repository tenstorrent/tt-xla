# Getting Started with Docker
This document walks you through how to set up TT-XLA using a Docker image. There are two other available options for getting started:
* [Installing a Wheel](https://docs.tenstorrent.com/tt-xla/getting_started.html#installing-a-wheel-and-running-an-example) - if you do not want to use Docker, and prefer to use a virtual environment by itself instead, use this method.
* [Building from Source](https://docs.tenstorrent.com/tt-xla/getting_started_build_from_source.html) - if you plan to develop TT-XLA further, you must build from source, and should use this method.

The following topics are covered:

* [Configuring Hardware](#configuring-hardware)
* [Setting up the Docker Container](#setting-up-the-docker-container)
* [Running Models in Docker](#running-models-in-docker)
* [Where to Go Next](#where-to-go-next)

## Configuring Hardware
Before setup can happen, you must configure your hardware. You can skip this section if you already completed the configuration steps. Otherwise, follow the instructions on the [Getting Started page](getting_started.md#configuring-hardware).

## Setting up the Docker Container
This section walks through the installation steps for using a Docker container for your project.

To install, do the following:

1. Install Docker if you do not already have it:

```bash
sudo apt update
sudo apt install docker.io -y
sudo systemctl start docker
sudo systemctl enable docker
```

2. Test that Docker is installed:

```bash
docker --version
```

3. Add your user to the Docker group:

```bash
sudo usermod -aG docker $USER
newgrp docker
```

4. Run the Docker container:

```bash
docker run -it --rm \
  --device /dev/tenstorrent \
  -v /dev/hugepages-1G:/dev/hugepages-1G \
  ghcr.io/tenstorrent/tt-xla-slim:latest
```

>**NOTE:** You cannot isolate devices in containers. You must pass through all devices even if you are only using one. You can do this by passing ```--device /dev/tenstorrent```. Do not try to pass ```--device /dev/tenstorrent/1``` or similar, as this type of device-in-container isolation will result in fatal errors later on during execution.

5. If you want to check that it is running, open a new tab with the **Same Command** option and run the following:

```bash
docker ps
```

## Running Models in Docker
This section shows you how to run a model using Docker. The provided example is from the TT-Forge repo. Do the following:

1. Inside your running Docker container, clone the TT-Forge repo:

```bash
git clone https://github.com/tenstorrent/tt-forge.git
```

2. Set the path for Python:

```bash
export PYTHONPATH=/tt-forge:$PYTHONPATH
```

3. Navigate into TT-Forge and run the following command:

```bash
git submodule update --init --recursive
```

4. Run a model. For this example, the **demo.py** for **opt_125m** is used. Similar to **gpt2**, this model predicts what the next word in a sentence is likely to be.  The **requirements.txt** file shows that you need to install **flax** and **transformers**:

```bash
pip install flax transformers
```

5. After completing installation, run the following:

```bash
python demos/tt-xla/nlp/pytorch/opt_demo.py
```

If all goes well, you should get an example prompt saying 'The capital of France is.' The prediction for the next term is listed, along with the probability it will occur. This is followed by a table of other likely choices.

## Where to Go Next

Now that you have set up TT-XLA, you can compile and run your own models, or try some of the other demos. You can find [TT-XLA demos in the TT-Forge directory](https://github.com/tenstorrent/tt-forge/tree/main/demos/tt-xla).
