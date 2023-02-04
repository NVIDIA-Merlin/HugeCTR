# Hierarchical Parameter Server Plugin for TensorRT

```{contents}
---
depth: 2
local: true
backlinks: none
---
```

## Introduction to the HPS Plugin for TensorRT

Hierarchical Parameter Server (HPS) is a distributed inference framework that is dedicated to deploying large embedding tables and realizing the low-latency retrieval of embeddings.
The framework combines a high-performance GPU embedding cache with a hierarchical storage architecture that encompasses different types of database backends.
The HPS plugin for TensorRT can be integrated into the TensorRT network as a custom layer to build the engine. The HPS-integrated engine can perform low-latency embedding lookup for large tables and TensorRT-accelerated forward propagation for dense network at the same time.

## Workflow

The workflow to leverage the HPS plugin for TensorRT can be summarized as:

* **Convert trained models to ONNX**: The models trained with different frameworks are converted to ONNX using the popular tools [tf2onnx](https://github.com/onnx/tensorflow-onnx), [torch.onnx](https://github.com/pytorch/pytorch/tree/master/torch/onnx), [hugectr2onnx](https://github.com/NVIDIA-Merlin/HugeCTR/tree/main/onnx_converter), and so on.

* **Perform ONNX graph surgery**: The node for embedding lookup in the ONNX graph is replaced by the placeholder of HPS plugin for TensorRT using the tool [ONNX GraphSurgeon](https://github.com/NVIDIA/TensorRT/tree/main/tools/onnx-graphsurgeon), as shown in Fig. 1.

* **Build the HPS-integrated TensorRT engine**: We can build the TensorRT engine based on the modified ONNX graph where the HPS can leveraged as a custom plugin layer.

* **Deploy the engine with Triton Inference Server TensorRT backend**: The HPS-integrated TensorRT engine is deployed with Triton Inference Server TensorRT backend. Set the `LD_PRELOAD=/usr/local/hps_trt/lib/libhps_plugin.so` environment variable to load the plugin shared library when you start Triton Inference Server.

<img src="hps_trt_user_guide_src/graph_surgeon.png" alt="Logical diagram of using ONNX GraphSurgeon to set the embedding lookup to the HPS plugin for TensorRT" width="720px" style="display:block;margin-left:auto;margin-right:auto;"/>

<div style="text-align:center;">Fig. 1: ONNX Graph Surgery</div>

<br></br>

## Installation

### Compute Capability

The plugin supports the following compute capabilities:

| Compute Capability | GPU                  | SM |
|--------------------|----------------------|-----|
| 7.0                | NVIDIA V100 (Volta)  | 70  |
| 7.5                | NVIDIA T4 (Turing)   | 75  |
| 8.0                | NVIDIA A100 (Ampere) | 80  |

### Installing HPS Using NGC Containers

All NVIDIA Merlin components are available as open source projects. However, a more convenient way to use these components is by using our Merlin NGC containers. These containers allow you to package your software application, libraries, dependencies, and runtime compilers in a self-contained environment. When installing HPS using NGC containers, the application environment remains portable, consistent, reproducible, and agnostic to the underlying host system's software configuration.

HPS is included in the Merlin Docker containers that are available from the NVIDIA GPU Cloud (NGC) catalog.
Access the catalog of containers at <https://catalog.ngc.nvidia.com/containers>.
To use these Docker containers, you must install the [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker) to provide GPU support for Docker.

The following sample commands pull and start the Merlin TensorFlow container, Merlin PyTorch container, or Merlin HugeCTR container:

Merlin TensorFlow
```shell
# Run the container in interactive mode
$ docker run --gpus=all --rm -it --cap-add SYS_NICE nvcr.io/nvidia/merlin/merlin-tensorflow:23.01
```

Merlin PyTorch
```shell
# Run the container in interactive mode
$ docker run --gpus=all --rm -it --cap-add SYS_NICE nvcr.io/nvidia/merlin/merlin-pytorch:23.01
```

Merlin HugeCTR
```shell
# Run the container in interactive mode
$ docker run --gpus=all --rm -it --cap-add SYS_NICE nvcr.io/nvidia/merlin/merlin-hugectr:23.01
```

You can check the existence of the HPS plugin for TensorRT after launching the container by running the following Python statements:
```python
import ctypes
handle = ctypes.CDLL("/usr/local/hps_trt/lib/libhps_plugin.so", mode=ctypes.RTLD_GLOBAL)
```

## Example Notebooks

We provide a collection of examples as [Jupyter Notebooks](../hps_trt/notebooks/index.md) that demonstrate how to build the HPS-integrated TensorRT engine for models trained with TensorFlow, PyTorch, or HugeCTR.
