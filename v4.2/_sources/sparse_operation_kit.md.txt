# Sparse Operation Kit

[Sparse Operation Kit (SOK)](https://github.com/NVIDIA-Merlin/HugeCTR/tree/master/sparse_operation_kit) is a Python package wrapped GPU accelerated operations dedicated for sparse training / inference cases. It is designed to be compatible with common deep learning (DL) frameworks like TensorFlow.
In sparse training / inference scenarios, for instance, CTR estimation, there are vast amounts of parameters which cannot fit into the memory of a single GPU. Many common DL frameworks only offer limited support for model parallelism (MP), because it can complicate using all available GPUs in a cluster to accelerate the whole training process.
SOK provides broad MP functionality to fully utilize all available GPUs, regardless of whether these GPUs are located in a single machine or multiple machines. Simultaneously, SOK takes advantage of existing data-parallel (DP) capabilities of DL frameworks to accelerate training while minimizing code changes. With SOK embedding layers, you can build a DNN model with mixed MP and DP. MP is used to shard large embedding parameter tables, such that they are distributed among the available GPUs to balance the workload, while DP is used for layers that only consume little GPU resources.

Please check this [SOK Documentation](https://nvidia-merlin.github.io/HugeCTR/sparse_operation_kit/master/index.html) for detail.

<img src="user_guide_src/workflow_of_embeddinglayer.png" width="1080px" align="center"/>