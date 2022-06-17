#!/usr/bin/env python
# coding: utf-8

# <img src="http://developer.download.nvidia.com/compute/machine-learning/frameworks/nvidia_logo.png" style="width: 90px; float: right;">
# 
# # Multi-GPU Offline Inference

# ## Overview
# 
# In HugeCTR version 3.4.1, we provide Python APIs to perform multi-GPU offline inference.
# This work leverages the [HugeCTR Hierarchical Parameter Server](https://nvidia-merlin.github.io/HugeCTR/master/hugectr_core_features.html#hierarchical-parameter-server) and enables concurrent execution on multiple devices.
# The `Norm` or `Parquet` dataset format is currently supported by multi-GPU offline inference.
# 
# This notebook explains how to perform multi-GPU offline inference with the HugeCTR Python APIs.
# For more details about the API, see the [HugeCTR Python Interface](https://nvidia-merlin.github.io/HugeCTR/master/api/python_interface.html#inference-api) documentation.

# ## Installation
# 
# ### Get HugeCTR from NGC
# 
# The HugeCTR Python module is preinstalled in the 22.06 and later [Merlin Training Container](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-training): `nvcr.io/nvidia/merlin/merlin-training:22.06`.
# 
# You can check the existence of required libraries by running the following Python code after launching this container.
# 
# ```bash
# $ python3 -c "import hugectr"
# ```
# 
# **Note**: This Python module contains both training APIs and offline inference APIs. For online inference with Triton Inference Server, refer to the [HugeCTR Backend](https://github.com/triton-inference-server/hugectr_backend) documentation.
# 
# > If you prefer to build HugeCTR from the source code instead of using the NGC container, refer to the [How to Start Your Development](https://nvidia-merlin.github.io/HugeCTR/master/hugectr_contributor_guide.html#how-to-start-your-development) documentation.

# ## Data Generation
# 
# HugeCTR provides a tool to generate synthetic datasets.
# The [Data Generator](https://nvidia-merlin.github.io/HugeCTR/master/api/python_interface.html#data-generator-api) class is capable of generating datasets in different formats and with different distributions.
# We will generate multi-hot Parquet datasets with a power-law distribution for this notebook:

# In[1]:


import hugectr
from hugectr.tools import DataGeneratorParams, DataGenerator

data_generator_params = DataGeneratorParams(
  format = hugectr.DataReaderType_t.Parquet,
  label_dim = 2,
  dense_dim = 2,
  num_slot = 3,
  i64_input_key = True,
  nnz_array = [2, 1, 3],
  source = "./multi_hot_parquet/file_list.txt",
  eval_source = "./multi_hot_parquet/file_list_test.txt",
  slot_size_array = [10000, 10000, 10000],
  check_type = hugectr.Check_t.Non,
  dist_type = hugectr.Distribution_t.PowerLaw,
  power_law_type = hugectr.PowerLaw_t.Short,
  num_files = 16,
  eval_num_files = 4)
data_generator = DataGenerator(data_generator_params)
data_generator.generate()


# ## Train from Scratch
# 
# We can train fom scratch by performing the following steps with Python APIs:
# 
# 1. Create the solver, reader and optimizer, then initialize the model.
# 2. Construct the model graph by adding input, sparse embedding and dense layers in order.
# 3. Compile the model and have an overview of the model graph.
# 4. Dump the model graph to a JSON file.
# 5. Fit the model, save the model weights and optimizer states implicitly.
# 6. Dump one batch of evaluation results to files.

# In[2]:


get_ipython().run_cell_magic('writefile', 'multi_hot_train.py', 'import hugectr\nfrom mpi4py import MPI\nsolver = hugectr.CreateSolver(model_name = "multi_hot",\n                              max_eval_batches = 1,\n                              batchsize_eval = 16384,\n                              batchsize = 16384,\n                              lr = 0.001,\n                              vvgpu = [[0]],\n                              i64_input_key = True,\n                              repeat_dataset = True,\n                              use_cuda_graph = True)\nreader = hugectr.DataReaderParams(data_reader_type = hugectr.DataReaderType_t.Parquet,\n                                  source = ["./multi_hot_parquet/file_list.txt"],\n                                  eval_source = "./multi_hot_parquet/file_list_test.txt",\n                                  check_type = hugectr.Check_t.Non,\n                                  slot_size_array = [10000, 10000, 10000])\noptimizer = hugectr.CreateOptimizer(optimizer_type = hugectr.Optimizer_t.Adam)\nmodel = hugectr.Model(solver, reader, optimizer)\nmodel.add(hugectr.Input(label_dim = 2, label_name = "label",\n                        dense_dim = 2, dense_name = "dense",\n                        data_reader_sparse_param_array = \n                        [hugectr.DataReaderSparseParam("data1", [2, 1], False, 2),\n                        hugectr.DataReaderSparseParam("data2", 3, False, 1),]))\nmodel.add(hugectr.SparseEmbedding(embedding_type = hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash, \n                            workspace_size_per_gpu_in_mb = 4,\n                            embedding_vec_size = 16,\n                            combiner = "sum",\n                            sparse_embedding_name = "sparse_embedding1",\n                            bottom_name = "data1",\n                            optimizer = optimizer))\nmodel.add(hugectr.SparseEmbedding(embedding_type = hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash, \n                            workspace_size_per_gpu_in_mb = 2,\n                            embedding_vec_size = 16,\n                            combiner = "sum",\n                            sparse_embedding_name = "sparse_embedding2",\n                            bottom_name = "data2",\n                            optimizer = optimizer))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Reshape,\n                            bottom_names = ["sparse_embedding1"],\n                            top_names = ["reshape1"],\n                            leading_dim=32))                            \nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Reshape,\n                            bottom_names = ["sparse_embedding2"],\n                            top_names = ["reshape2"],\n                            leading_dim=16))                            \nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Concat,\n                            bottom_names = ["reshape1", "reshape2", "dense"], top_names = ["concat1"]))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,\n                            bottom_names = ["concat1"],\n                            top_names = ["fc1"],\n                            num_output=1024))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,\n                            bottom_names = ["fc1"],\n                            top_names = ["relu1"]))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,\n                            bottom_names = ["relu1"],\n                            top_names = ["fc2"],\n                            num_output=2))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.MultiCrossEntropyLoss,\n                            bottom_names = ["fc2", "label"],\n                            top_names = ["loss"],\n                            target_weight_vec = [0.5, 0.5]))\nmodel.compile()\nmodel.summary()\nmodel.graph_to_json("multi_hot.json")\nmodel.fit(max_iter = 1100, display = 200, eval_interval = 1000, snapshot = 1000, snapshot_prefix = "multi_hot")\nmodel.export_predictions("multi_hot_pred_" + str(1000), "multi_hot_label_" + str(1000))\n')


# In[3]:


get_ipython().system('python3 multi_hot_train.py')


# ### Multi-GPU Offline Inference
# 
# We can demonstrate multi-GPU offline inference by performing the following steps with Python APIs:
# 
# 1. Configure the inference hyperparameters.
# 2. Initialize the inference model. The model is a collection of inference sessions deployed on multiple devices.
# 3. Make an inference from the evaluation dataset.
# 4. Check the correctness of the inference by comparing it with the dumped evaluation results.
# 
# **Note**: The `max_batchsize` configured within `InferenceParams` is the global batch size.
# The value for `max_batchsize` should be divisible by the number of deployed devices.
# The numpy array returned by `InferenceModel.predict` is of the shape `(max_batchsize * num_batches, label_dim)`.

# In[4]:


import hugectr
from hugectr.inference import InferenceModel, InferenceParams
import numpy as np
from mpi4py import MPI

model_config = "multi_hot.json"
inference_params = InferenceParams(
    model_name = "multi_hot",
    max_batchsize = 1024,
    hit_rate_threshold = 1.0,
    dense_model_file = "multi_hot_dense_1000.model",
    sparse_model_files = ["multi_hot0_sparse_1000.model", "multi_hot1_sparse_1000.model"],
    deployed_devices = [0, 1, 2, 3],
    use_gpu_embedding_cache = True,
    cache_size_percentage = 0.5,
    i64_input_key = True
)
inference_model = InferenceModel(model_config, inference_params)
pred = inference_model.predict(
    16,
    "./multi_hot_parquet/file_list_test.txt",
    hugectr.DataReaderType_t.Parquet,
    hugectr.Check_t.Non,
    [10000, 10000, 10000]
)
grount_truth = np.loadtxt("multi_hot_pred_1000")
print("pred: ", pred)
print("grount_truth: ", grount_truth)
diff = pred.flatten()-grount_truth
mse = np.mean(diff*diff)
print("mse: ", mse)

