# Hierarchical Parameter Server

Hierarchical Parameter Server (HPS) is a recommender system parameter server for embedding lookup that is part of NVIDIA HugeCTR.

HPS offers a flexible deployment and configuration to meet site-specific recommender system needs.
The deployment

[HPS Database Backend](../hugectr_parameter_server.md)
:  Provides a three-level storage architecture.
   The first and highest performing level is GPU memory and is followed by CPU memory.
   The third layer can be high-speed local SSDs with or without a distributed database.
   The key benefit of the HPS database backend is serving embedding tables that exceed GPU and CPU memory while providing the highest possible performance.

[HPS plugin for TensorFlow](hps_tf_user_guide.md)
:  Provides high-performance, scalability, and low-latency access to embedding tables for deep learning models that have large embedding tables in TensorFlow.

[HPS Backend for Triton Inference Server](https://github.com/triton-inference-server/hugectr_backend/tree/main/hps_backend)
:  The backend for Triton Inference Server is an inference deployment framework that integrates HPS for end-to-end inference on Triton.
   Documentation for the backend is available from the `hugectr_backend` repository at the preceding URL.
