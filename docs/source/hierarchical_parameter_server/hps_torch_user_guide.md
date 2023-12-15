# Hierarchical Parameter Server Plugin for Torch

```{contents}
---
depth: 2
local: true
backlinks: none
---
```

## Introduction to the HPS Plugin for Torch

The Hierarchical Parameter Server (HPS) is a distributed inference framework designed to efficiently deploy large embedding tables and enable low-latency retrieval of embeddings. It achieves this through a combination of a high-performance GPU embedding cache and a hierarchical storage architecture that supports various database backends. **The HPS plugin for Torch** allows users to harness the HPS by incorporating it into their Torch model as a custom layer. By doing so, you can seamlessly deploy large embedding tables within your model.

## Installation

### Compute Capability

The plugin supports the following compute capabilities:

| Compute Capability | GPU                  | SM |
|--------------------|----------------------|-----|
| 7.0                | NVIDIA V100 (Volta)  | 70  |
| 7.5                | NVIDIA T4 (Turing)   | 75  |
| 8.0                | NVIDIA A100 (Ampere) | 80  |
| 9.0                | NVIDIA H100 (Hopper) | 90  |

### Installing HPS Using NGC Containers

While all NVIDIA Merlin components are open source, the most convenient way to leverage them is through Merlin NGC containers. These containers enable you to encapsulate your software application, libraries, dependencies, and runtime compilers within a self-contained environment. By installing HPS using NGC containers, you ensure that your application environment remains portable, consistent, reproducible, and independent of the underlying host system's software configuration.

HPS is available within the Merlin Docker containers, which can be accessed through the NVIDIA GPU Cloud (NGC) catalog. You can explore and obtain these containers from the catalog by visiting <https://catalog.ngc.nvidia.com/containers>.

To utilize these Docker containers, you will need to install the [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker) to provide GPU support for Docker.

The following sample commands pull and start the Merlin PyTorch container:

Merlin PyTorch
```shell
# Run the container in interactive mode
$ docker run --gpus=all --rm -it --cap-add SYS_NICE nvcr.io/nvidia/merlin/merlin-pytorch:23.12
```

You can check the existence of the HPS plugin for Torch after launching the container by running the following Python statements:
```python
import hps_torch
```

## Example Notebooks

We provide a collection of examples as [Jupyter Notebooks](../hps_torch/notebooks/index.md) that demonstrate how to apply HPS to the Torch model.
