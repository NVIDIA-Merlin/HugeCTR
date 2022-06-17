#!/usr/bin/env python
# coding: utf-8

# <img src="http://developer.download.nvidia.com/compute/machine-learning/frameworks/nvidia_logo.png" style="width: 90px; float: right;">
# 
# # HugeCTR to ONNX Converter

# ## Overview
# 
# To improve compatibility and interoperability with other deep-learning frameworks, we provide a Python module to convert HugeCTR models to ONNX.
# ONNX serves as an open-source format for AI models.
# Basically, this converter requires the model graph in JSON, dense model, and sparse models as inputs and saves the converted ONNX model to the specified path.
# All the required input files can be obtained with HugeCTR training APIs and the whole workflow can be accomplished seamlessly in Python.
# 
# This notebook demonstrates how to access and use the HugeCTR to ONNX converter.
# Please make sure that you are familiar with HugeCTR training APIs which will be covered here to ensure the completeness.
# For more details of the usage of this converter, refer to the HugeCTR to ONNX Converter in the [onnx_converter](https://github.com/NVIDIA-Merlin/HugeCTR/tree/master/onnx_converter) directory of the repository.

# ## Access the HugeCTR to ONNX Converter
# 
# Make sure that you start the notebook inside a running 22.06 or later NGC docker container: `nvcr.io/nvidia/merlin/merlin-training:22.06`.
# The module of the ONNX converter is installed to the path `/usr/local/lib/python3.8/dist-packages`.
# As for HugeCTR Python interface, a dynamic link to the `hugectr.so` library is installed to the path `/usr/local/hugectr/lib/`.
# You can access the ONNX converter as well as HugeCTR Python interface anywhere within the container.

# Run the following cell to confirm that the HugeCTR Python interface can be accessed correctly.

# In[1]:


import hugectr


# Run the following cell to confirm that the HugeCTR to ONNX converter can be accessed correctly.

# In[2]:


import hugectr2onnx


# ## Wide and Deep Model
# 
# ### Download and Preprocess Data
# 
# 1. Download the Criteo dataset using the following command:
# 
#    ```shell
#    $ cd ${project_root}/tools
#    $ wget https://storage.googleapis.com/criteo-cail-datasets/day_1.gz
#    ```
#    
#    In preprocessing, we will further reduce the amounts of data to speedup the preprocessing, fill missing values, remove the feature values whose occurrences are very rare, etc. Here we choose pandas preprocessing method to make the dataset ready for HugeCTR training.
# 
# 2. Preprocessing by Pandas using the following command:
# 
#    ```shell
#    $ bash preprocess.sh 1 wdl_data pandas 1 1 100
#    ```
#    
#    The first argument represents the dataset postfix. It is 1 here since day_1 is used. The second argument wdl_data is where the preprocessed data is stored. The fourth argument (one after pandas) 1 embodies that the normalization is applied to dense features. The fifth argument 1 means that the feature crossing is applied. The last argument 100 means the number of data files in each file list.
#    
# 3. Create a soft link to the dataset folder using the following command:
# 
#    ```shell
#    $ ln -s ${project_root}/tools/wdl_data ${project_root}/notebooks/wdl_data
#    ```

# ### Train the HugeCTR Model
# 
# We can train fom scratch, dump the model graph to a JSON file, and save the model weights and optimizer states by performing the following with Python APIs:
# 
# 1. Create the solver, reader and optimizer, then initialize the model.
# 2. Construct the model graph by adding input, sparse embedding and dense layers in order.
# 3. Compile the model and have an overview of the model graph.
# 4. Dump the model graph to the JSON file.
# 5. Fit the model, save the model weights and optimizer states implicitly.
# 
# Please note that the training mode is determined by `repeat_dataset` within `hugectr.CreateSolver`.
# If it is `True`, the non-epoch mode training is adopted and the maximum iterations should be specified by `max_iter` within `hugectr.Model.fit`.
# If it is `False`, the epoch-mode training is adopted and the number of epochs should be specified by `num_epochs` within `hugectr.Model.fit`.
# 
# The optimizer that is used to initialize the model applies to the weights of dense layers, while the optimizer for each sparse embedding layer can be specified independently within `hugectr.SparseEmbedding`.

# In[3]:


get_ipython().run_cell_magic('writefile', 'wdl_train.py', 'import hugectr\nfrom mpi4py import MPI\nsolver = hugectr.CreateSolver(max_eval_batches = 300,\n                              batchsize_eval = 16384,\n                              batchsize = 16384,\n                              lr = 0.001,\n                              vvgpu = [[0]],\n                              repeat_dataset = True)\nreader = hugectr.DataReaderParams(data_reader_type = hugectr.DataReaderType_t.Norm,\n                                  source = ["./wdl_data/file_list.txt"],\n                                  eval_source = "./wdl_data/file_list_test.txt",\n                                  check_type = hugectr.Check_t.Sum)\noptimizer = hugectr.CreateOptimizer(optimizer_type = hugectr.Optimizer_t.Adam,\n                                    update_type = hugectr.Update_t.Global,\n                                    beta1 = 0.9,\n                                    beta2 = 0.999,\n                                    epsilon = 0.0000001)\nmodel = hugectr.Model(solver, reader, optimizer)\nmodel.add(hugectr.Input(label_dim = 1, label_name = "label",\n                        dense_dim = 13, dense_name = "dense",\n                        data_reader_sparse_param_array = \n                        [hugectr.DataReaderSparseParam("wide_data", 2, True, 1),\n                        hugectr.DataReaderSparseParam("deep_data", 1, True, 26)]))\nmodel.add(hugectr.SparseEmbedding(embedding_type = hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash, \n                            workspace_size_per_gpu_in_mb = 75,\n                            embedding_vec_size = 1,\n                            combiner = "sum",\n                            sparse_embedding_name = "sparse_embedding2",\n                            bottom_name = "wide_data",\n                            optimizer = optimizer))\nmodel.add(hugectr.SparseEmbedding(embedding_type = hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash, \n                            workspace_size_per_gpu_in_mb = 1074,\n                            embedding_vec_size = 16,\n                            combiner = "sum",\n                            sparse_embedding_name = "sparse_embedding1",\n                            bottom_name = "deep_data",\n                            optimizer = optimizer))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Reshape,\n                            bottom_names = ["sparse_embedding1"],\n                            top_names = ["reshape1"],\n                            leading_dim=416))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Reshape,\n                            bottom_names = ["sparse_embedding2"],\n                            top_names = ["reshape2"],\n                            leading_dim=1))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Concat,\n                            bottom_names = ["reshape1", "dense"],\n                            top_names = ["concat1"]))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,\n                            bottom_names = ["concat1"],\n                            top_names = ["fc1"],\n                            num_output=1024))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,\n                            bottom_names = ["fc1"],\n                            top_names = ["relu1"]))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Dropout,\n                            bottom_names = ["relu1"],\n                            top_names = ["dropout1"],\n                            dropout_rate=0.5))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,\n                            bottom_names = ["dropout1"],\n                            top_names = ["fc2"],\n                            num_output=1024))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,\n                            bottom_names = ["fc2"],\n                            top_names = ["relu2"]))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Dropout,\n                            bottom_names = ["relu2"],\n                            top_names = ["dropout2"],\n                            dropout_rate=0.5))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,\n                            bottom_names = ["dropout2"],\n                            top_names = ["fc3"],\n                            num_output=1))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Add,\n                            bottom_names = ["fc3", "reshape2"],\n                            top_names = ["add1"]))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.BinaryCrossEntropyLoss,\n                            bottom_names = ["add1", "label"],\n                            top_names = ["loss"]))\nmodel.graph_to_json("wdl.json")\nmodel.compile()\nmodel.summary()\nmodel.fit(max_iter = 2300, display = 200, eval_interval = 1000, snapshot = 2000, snapshot_prefix = "wdl")\n')


# In[4]:


get_ipython().system('python3 wdl_train.py')


# ### Convert to ONNX
# 
# We can convert the trained HugeCTR model to ONNX with a call to `hugectr2onnx.converter.convert`. We can specify whether to convert the sparse embeddings via the flag `convert_embedding` and do not need to provide the sparse models if it is set as `False`. In this notebook, both dense and sparse parts of the HugeCTR model will be converted to ONNX, in order that we can check the correctness of the conversion more easily by comparing inference results based on HugeCTR and ONNX Runtime.

# In[5]:


import hugectr2onnx
hugectr2onnx.converter.convert(onnx_model_path = "wdl.onnx",
                            graph_config = "wdl.json",
                            dense_model = "wdl_dense_2000.model",
                            convert_embedding = True,
                            sparse_models = ["wdl0_sparse_2000.model", "wdl1_sparse_2000.model"])


# ### Inference with ONNX Runtime and HugeCTR
# 
# To make inferences with the ONNX runtime, we need to read samples from the data and feed them to the ONNX inference session. Specifically, we need to extract dense features, wide sparse features and deep sparse features from the preprocessed Wide&Deep dataset. To guarantee fair comparison with HugeCTR inference, we will use the first data file within `./wdl_data/file_list_test.txt`, i.e., `./wdl_data/val/sparse_embedding0.data`, and make inference for the same number of samples (should be less than the total number of samples within `./wdl_data/val/sparse_embedding0.data`).

# In[6]:


import struct
import numpy as np
def read_samples_for_wdl(data_file, num_samples, key_type="I32", slot_num=27):
    key_type_map = {"I32": ["I", 4], "I64": ["q", 8]}
    with open(data_file, 'rb') as file:
        # skip data_header
        file.seek(4 + 64 + 1, 0)
        batch_label = []
        batch_dense = []
        batch_wide_data = []
        batch_deep_data = []
        for _ in range(num_samples):
            # one sample
            length_buffer = file.read(4) # int
            length = struct.unpack('i', length_buffer)
            label_buffer = file.read(4) # int
            label = struct.unpack('i', label_buffer)[0]
            dense_buffer = file.read(4 * 13) # dense_dim * float
            dense = struct.unpack("13f", dense_buffer)
            keys = []
            for _ in range(slot_num):
                nnz_buffer = file.read(4) # int
                nnz = struct.unpack("i", nnz_buffer)[0]
                key_buffer = file.read(key_type_map[key_type][1] * nnz) # nnz * sizeof(key_type)
                key = struct.unpack(str(nnz) + key_type_map[key_type][0], key_buffer)
                keys += list(key)
            check_bit_buffer = file.read(1) # char
            check_bit = struct.unpack("c", check_bit_buffer)[0]
            batch_label.append(label)
            batch_dense.append(dense)
            batch_wide_data.append(keys[0:2])
            batch_deep_data.append(keys[2:28])
    batch_label = np.reshape(np.array(batch_label, dtype=np.float32), newshape=(num_samples, 1))
    batch_dense = np.reshape(np.array(batch_dense, dtype=np.float32), newshape=(num_samples, 13))
    batch_wide_data = np.reshape(np.array(batch_wide_data, dtype=np.int64), newshape=(num_samples, 1, 2))
    batch_deep_data = np.reshape(np.array(batch_deep_data, dtype=np.int64), newshape=(num_samples, 26, 1))
    return batch_label, batch_dense, batch_wide_data, batch_deep_data


# In[7]:


batch_size = 64
num_batches = 100
data_file = "./wdl_data/val/sparse_embedding0.data" # there are totally 40960 samples
onnx_model_path = "wdl.onnx"

label, dense, wide_data, deep_data = read_samples_for_wdl(data_file, batch_size*num_batches, key_type="I32", slot_num = 27)
import onnxruntime as ort
sess = ort.InferenceSession(onnx_model_path)
res = sess.run(output_names=[sess.get_outputs()[0].name],
                  input_feed={sess.get_inputs()[0].name: dense, sess.get_inputs()[1].name: wide_data, sess.get_inputs()[2].name: deep_data})
onnx_preds = res[0].reshape((batch_size*num_batches,))
print("ONNX Runtime Predicions:", onnx_preds)


# We can then make inference based on HugeCTR APIs and compare the prediction results.

# In[8]:


dense_model = "wdl_dense_2000.model"
sparse_models = ["wdl0_sparse_2000.model", "wdl1_sparse_2000.model"]
graph_config = "wdl.json"
data_source = "./wdl_data/file_list_test.txt"
import hugectr
from mpi4py import MPI
from hugectr.inference import InferenceParams, CreateInferenceSession
inference_params = InferenceParams(model_name = "wdl",
                                max_batchsize = batch_size,
                                hit_rate_threshold = 0.6,
                                dense_model_file = dense_model,
                                sparse_model_files = sparse_models,
                                device_id = 0,
                                use_gpu_embedding_cache = True,
                                cache_size_percentage = 0.6,
                                i64_input_key = False)
inference_session = CreateInferenceSession(graph_config, inference_params)
hugectr_preds = inference_session.predict(num_batches, data_source, hugectr.DataReaderType_t.Norm, hugectr.Check_t.Sum)
print("HugeCTR Predictions: ", hugectr_preds)


# In[9]:


print("Min absolute error: ", np.min(np.abs(onnx_preds-hugectr_preds)))
print("Mean absolute error: ", np.mean(np.abs(onnx_preds-hugectr_preds)))
print("Max absolute error: ", np.max(np.abs(onnx_preds-hugectr_preds)))


# ## API Signature for hugectr2onnx.converter

# ```bash
# NAME
#     hugectr2onnx.converter
# 
# FUNCTIONS
#     convert(onnx_model_path, graph_config, dense_model, convert_embedding=False, sparse_models=[], ntp_file=None, graph_name='hugectr')
#         Convert a HugeCTR model to an ONNX model
#         Args:
#             onnx_model_path: the path to store the ONNX model
#             graph_config: the graph configuration JSON file of the HugeCTR model
#             dense_model: the file of the dense weights for the HugeCTR model
#             convert_embedding: whether to convert the sparse embeddings for the HugeCTR model (optional)
#             sparse_models: the files of the sparse embeddings for the HugeCTR model (optional)
#             ntp_file: the file of the non-trainable parameters for the HugeCTR model (optional)
#             graph_name: the graph name for the ONNX model (optional)
# ```
