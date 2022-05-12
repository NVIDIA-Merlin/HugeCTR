#!/usr/bin/env python
# coding: utf-8

# <img src="http://developer.download.nvidia.com/compute/machine-learning/frameworks/nvidia_logo.png" style="width: 90px; float: right;">
# 
# # Hierarchical Parameter Server Demo

# ## Overview
# 
# In HugeCTR version 3.5, we provide Python APIs for embedding table lookup with [HugeCTR Hierarchical Parameter Server (HPS)](https://nvidia-merlin.github.io/HugeCTR/master/hugectr_core_features.html#hierarchical-parameter-server)
# HPS supports different database backends and GPU embedding caches.
# 
# This notebook demonstrates how to use HPS with HugeCTR Python APIs. Without loss of generality, the HPS APIs are utilized together with the ONNX Runtime APIs to create an ensemble inference model, where HPS is responsible for embedding table lookup while the ONNX model takes charge of feed forward of dense neural networks.

# ## Installation
# 
# ### Get HugeCTR from NGC
# 
# The HugeCTR Python module is preinstalled in the 22.05 and later [Merlin Training Container](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-training): `nvcr.io/nvidia/merlin/merlin-training:22.05`.
# 
# You can check the existence of required libraries by running the following Python code after launching this container.
# 
# ```bash
# $ python3 -c "import hugectr"
# ```
# 
# **Note**: This Python module contains both training APIs and offline inference APIs. For online inference with Triton, please refer to [HugeCTR Backend](https://github.com/triton-inference-server/hugectr_backend).
# 
# > If you prefer to build HugeCTR from the source code instead of using the NGC container, please refer to the
# > [How to Start Your Development](https://nvidia-merlin.github.io/HugeCTR/master/hugectr_contributor_guide.html#how-to-start-your-development)
# > documentation.

# ## Data Generation
# 
# HugeCTR provides a tool to generate synthetic datasets. The [Data Generator](https://nvidia-merlin.github.io/HugeCTR/master/api/python_interface.html#data-generator-api) is capable of generating datasets of different file formats and different distributions. We will generate one-hot Parquet datasets with power-law distribution for this notebook:

# In[1]:


import hugectr
from hugectr.tools import DataGeneratorParams, DataGenerator

data_generator_params = DataGeneratorParams(
  format = hugectr.DataReaderType_t.Parquet,
  label_dim = 1,
  dense_dim = 10,
  num_slot = 4,
  i64_input_key = True,
  nnz_array = [1, 1, 1, 1],
  source = "./data_parquet/file_list.txt",
  eval_source = "./data_parquet/file_list_test.txt",
  slot_size_array = [10000, 10000, 10000, 10000],
  check_type = hugectr.Check_t.Non,
  dist_type = hugectr.Distribution_t.PowerLaw,
  power_law_type = hugectr.PowerLaw_t.Short,
  num_files = 16,
  eval_num_files = 4,
  num_samples_per_file = 40960)
data_generator = DataGenerator(data_generator_params)
data_generator.generate()


# ## Train from Scratch
# 
# We can train fom scratch by performing the following steps with Python APIs:
# 
# 1. Create the solver, reader and optimizer, then initialize the model.
# 2. Construct the model graph by adding input, sparse embedding and dense layers in order.
# 3. Compile the model and have an overview of the model graph.
# 4. Dump the model graph to the JSON file.
# 5. Fit the model, save the model weights and optimizer states implicitly.
# 6. Dump one batch of evaluation results to files.

# In[2]:


get_ipython().run_cell_magic('writefile', 'train.py', 'import hugectr\nfrom mpi4py import MPI\nsolver = hugectr.CreateSolver(model_name = "hps_demo",\n                              max_eval_batches = 1,\n                              batchsize_eval = 1024,\n                              batchsize = 1024,\n                              lr = 0.001,\n                              vvgpu = [[0]],\n                              i64_input_key = True,\n                              repeat_dataset = True,\n                              use_cuda_graph = True)\nreader = hugectr.DataReaderParams(data_reader_type = hugectr.DataReaderType_t.Parquet,\n                                  source = ["./data_parquet/file_list.txt"],\n                                  eval_source = "./data_parquet/file_list_test.txt",\n                                  check_type = hugectr.Check_t.Non,\n                                  slot_size_array = [10000, 10000, 10000, 10000])\noptimizer = hugectr.CreateOptimizer(optimizer_type = hugectr.Optimizer_t.Adam)\nmodel = hugectr.Model(solver, reader, optimizer)\nmodel.add(hugectr.Input(label_dim = 1, label_name = "label",\n                        dense_dim = 10, dense_name = "dense",\n                        data_reader_sparse_param_array = \n                        [hugectr.DataReaderSparseParam("data1", [1, 1], True, 2),\n                        hugectr.DataReaderSparseParam("data2", [1, 1], True, 2)]))\nmodel.add(hugectr.SparseEmbedding(embedding_type = hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash, \n                            workspace_size_per_gpu_in_mb = 4,\n                            embedding_vec_size = 16,\n                            combiner = "sum",\n                            sparse_embedding_name = "sparse_embedding1",\n                            bottom_name = "data1",\n                            optimizer = optimizer))\nmodel.add(hugectr.SparseEmbedding(embedding_type = hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash, \n                            workspace_size_per_gpu_in_mb = 8,\n                            embedding_vec_size = 32,\n                            combiner = "sum",\n                            sparse_embedding_name = "sparse_embedding2",\n                            bottom_name = "data2",\n                            optimizer = optimizer))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Reshape,\n                            bottom_names = ["sparse_embedding1"],\n                            top_names = ["reshape1"],\n                            leading_dim=32))                            \nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Reshape,\n                            bottom_names = ["sparse_embedding2"],\n                            top_names = ["reshape2"],\n                            leading_dim=64))                            \nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Concat,\n                            bottom_names = ["reshape1", "reshape2", "dense"], top_names = ["concat1"]))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,\n                            bottom_names = ["concat1"],\n                            top_names = ["fc1"],\n                            num_output=1024))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,\n                            bottom_names = ["fc1"],\n                            top_names = ["relu1"]))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,\n                            bottom_names = ["relu1"],\n                            top_names = ["fc2"],\n                            num_output=1))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.BinaryCrossEntropyLoss,\n                            bottom_names = ["fc2", "label"],\n                            top_names = ["loss"]))\nmodel.compile()\nmodel.summary()\nmodel.graph_to_json("hps_demo.json")\nmodel.fit(max_iter = 1100, display = 200, eval_interval = 1000, snapshot = 1000, snapshot_prefix = "hps_demo")\nmodel.export_predictions("hps_demo_pred_" + str(1000), "hps_demo_label_" + str(1000))\n')


# In[3]:


get_ipython().system('python3 train.py')


# ## Convert HugeCTR to ONNX
# 
# We will convert the saved HugeCTR models to ONNX using the HugeCTR to ONNX Converter. For more information about the converter, refer to the README in the [onnx_converter](https://github.com/NVIDIA-Merlin/HugeCTR/tree/master/onnx_converter) directory of the repository.
# 
# For the sake of double checking the correctness, we will investigate both cases of conversion depending on whether or not to convert the sparse embedding models.

# In[4]:


import hugectr2onnx
hugectr2onnx.converter.convert(onnx_model_path = "hps_demo_with_embedding.onnx",
                            graph_config = "hps_demo.json",
                            dense_model = "hps_demo_dense_1000.model",
                            convert_embedding = True,
                            sparse_models = ["hps_demo0_sparse_1000.model", "hps_demo1_sparse_1000.model"])

hugectr2onnx.converter.convert(onnx_model_path = "hps_demo_without_embedding.onnx",
                            graph_config = "hps_demo.json",
                            dense_model = "hps_demo_dense_1000.model",
                            convert_embedding = False)


# ## Inference with HPS & ONNX
# 
# We will make inference by performing the following steps with Python APIs:
# 
# 1. Configure the HPS hyperparameters.
# 2. Initialize the HPS object, which is responsible for embedding table lookup.
# 3. Loading the Parquet data.
# 4. Make inference with the HPS object and the ONNX inference session of `hps_demo_without_embedding.onnx`.
# 5. Check the correctness by comparing with dumped evaluation results.
# 6. Make inference with the ONNX inference session of `hps_demo_with_embedding.onnx` (double check).

# In[5]:


from hugectr.inference import HPS, ParameterServerConfig, InferenceParams

import pandas as pd
import numpy as np

import onnxruntime as ort

slot_size_array = [10000, 10000, 10000, 10000]
key_offset = np.insert(np.cumsum(slot_size_array), 0, 0)[:-1]
batch_size = 1024

# 1. Configure the HPS hyperparameters
ps_config = ParameterServerConfig(
           emb_table_name = {"hps_demo": ["sparse_embedding1", "sparse_embedding2"]},
           embedding_vec_size = {"hps_demo": [16, 32]},
           max_feature_num_per_sample_per_emb_table = {"hps_demo": [2, 2]},
           inference_params_array = [
              InferenceParams(
                model_name = "hps_demo",
                max_batchsize = batch_size,
                hit_rate_threshold = 1.0,
                dense_model_file = "",
                sparse_model_files = ["hps_demo0_sparse_1000.model", "hps_demo1_sparse_1000.model"],
                deployed_devices = [0],
                use_gpu_embedding_cache = True,
                cache_size_percentage = 0.5,
                i64_input_key = True)
           ])

# 2. Initialize the HPS object
hps = HPS(ps_config)

# 3. Loading the Parquet data.
df = pd.read_parquet("data_parquet/val/gen_0.parquet")
dense_input_columns = df.columns[1:11]
cat_input1_columns = df.columns[11:13]
cat_input2_columns = df.columns[13:15]
dense_input = df[dense_input_columns].loc[0:batch_size-1].to_numpy(dtype=np.float32)
cat_input1 = (df[cat_input1_columns].loc[0:batch_size-1].to_numpy(dtype=np.int64) + key_offset[0:2]).reshape((batch_size, 2, 1))
cat_input2 = (df[cat_input2_columns].loc[0:batch_size-1].to_numpy(dtype=np.int64) + key_offset[2:4]).reshape((batch_size, 2, 1))

# 4. Make inference from the HPS object and the ONNX inference session of `hps_demo_without_embedding.onnx`.
embedding1 = hps.lookup(cat_input1.flatten(), "hps_demo", 0).reshape(batch_size, 2, 16)
embedding2 = hps.lookup(cat_input2.flatten(), "hps_demo", 1).reshape(batch_size, 2, 32)
sess = ort.InferenceSession("hps_demo_without_embedding.onnx")
res = sess.run(output_names=[sess.get_outputs()[0].name],
               input_feed={sess.get_inputs()[0].name: dense_input,
               sess.get_inputs()[1].name: embedding1,
               sess.get_inputs()[2].name: embedding2})
pred = res[0]

# 5. Check the correctness by comparing with dumped evaluation results.
ground_truth = np.loadtxt("hps_demo_pred_1000")
print("ground_truth: ", ground_truth)
diff = pred.flatten()-ground_truth
mse = np.mean(diff*diff)
print("pred: ", pred)
print("mse between pred and ground_truth: ", mse)

# 6. Make inference with the ONNX inference session of `hps_demo_with_embedding.onnx` (double check).
sess_ref = ort.InferenceSession("hps_demo_with_embedding.onnx")
res_ref = sess_ref.run(output_names=[sess_ref.get_outputs()[0].name],
                   input_feed={sess_ref.get_inputs()[0].name: dense_input,
                   sess_ref.get_inputs()[1].name: cat_input1,
                   sess_ref.get_inputs()[2].name: cat_input2})
pred_ref = res_ref[0]
diff_ref = pred_ref.flatten()-ground_truth
mse_ref = np.mean(diff_ref*diff_ref)
print("pred_ref: ", pred_ref)
print("mse between pred_ref and ground_truth: ", mse_ref)

