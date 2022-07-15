#!/usr/bin/env python
# coding: utf-8

# <img src="http://developer.download.nvidia.com/compute/machine-learning/frameworks/nvidia_logo.png" style="width: 90px; float: right;">
# 
# # HugeCTR Continuous Training

# ## Overview
# The notebook introduces how to use the Embedding Training Cache (ETC) feature in HugeCTR for the continuous training. The ETC feature is designed to handle recommendation models with huge embedding table by the incremental training method, which allows you to train such a model that the model size is much larger than the available GPU memory size.
# 
# To learn more about the ETC, see the [Embedding Training Cache](https://nvidia-merlin.github.io/HugeCTR/master/hugectr_core_features.html#embedding-training-cache) documentation.
# 
# To learn how to use the APIs of ETC, see the [HugeCTR Python Interface](https://nvidia-merlin.github.io/HugeCTR/master/api/python_interface.html) documentation.

# ## Installation
# 
# ### Get HugeCTR from NGC
# 
# The continuous training module is preinstalled in the 22.07 and later [Merlin Training Container](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-hugectr): `nvcr.io/nvidia/merlin/merlin-hugectr:22.07`.
# 
# You can check the existence of required libraries by running the following Python code after launching this container.
# 
# ```bash
# $ python3 -c "import hugectr"
# ```
# 
# > If you prefer to build HugeCTR from the source code instead of using the NGC container, refer to the
# > [How to Start Your Development](https://nvidia-merlin.github.io/HugeCTR/master/hugectr_contributor_guide.html#how-to-start-your-development)
# > documentation.

# ## Continuous Training
# 
# ### Data Preparation
# 
# 1. Download the Criteo dataset using the following command:
# 
#    ```shell
#    $ cd ${project_root}/tools
#    $ wget https://storage.googleapis.com/criteo-cail-datasets/day_1.gz
#    ```
#    
#    To preprocess the downloaded Kaggle Criteo dataset, we'll make the following operations: 
# 
#    * Reduce the amounts of data to speed up the preprocessing
#    * Fill missing values
#    * Remove the feature values whose occurrences are very rare, etc.
# 
# 2. Preprocessing by Pandas using the following command:
# 
#    ```shell
#    $ bash preprocess.sh 1 wdl_data pandas 1 1 100
#    ```
#    
#    Meanings of the command line arguments:
# 
#    * The 1st argument represents the dataset postfix. It is `1` here since `day_1` is used.
#    * The 2nd argument `wdl_data` is where the preprocessed data is stored.
#    * The 3rd argument `pandas` is the processing script going to use, here we choose `pandas`.
#    * The 4th argument `1` embodies that the normalization is applied to dense features.
#    * The 5th argument `1` means that the feature crossing is applied.
#    * The 6th argument `100` means the number of data files in each file list.
# 
#    For more details about the data preprocessing, please refer to the "Preprocess the Criteo Dataset" section of the README in the [samples/criteo](https://github.com/NVIDIA-Merlin/HugeCTR/tree/master/samples/criteo) directory of the repository on GitHub.
# 
# 3. Create a soft link of the dataset folder to the path of this notebook using the following command:
# 
#    ```shell
#    $ ln -s ${project_root}/tools/wdl_data ${project_root}/notebooks/wdl_data
#    ```

# ### Continuous Training with High-level API
# 
# This section gives the code sample of continuous training using a Keras-like high-level API. The high-level API encapsulates much of the complexity for users, making it easy to use and able to handle many of the scenarios in a production environment.
# 
# Meanwhile, in addition to a high-level API, HugeCTR also provides low-level APIs that enable you customize the training logic. A code sample using the low-level APIs is provided in the next section.
# 
# The code sample in this section trains a model from scratch using the embedding training cache, gets the incremental model, and saves the trained dense weights and sparse embedding weights. The following steps are required to achieve those logics:
# 
# 1. Create the `solver`, `reader`, `optimizer` and `etc`, then initialize the model.
# 2. Construct the model graph by adding input, sparse embedding, and dense layers in order.
# 3. Compile the model and overview the model graph.
# 4. Dump the model graph to the JSON file.
# 5. Train the sparse and dense model.
# 6. Set the new training datasets and their corresponding keysets.
# 7. Train the sparse and dense model incrementally.
# 8. Get the incrementally trained embedding table.
# 9. Save the model weights and optimizer states explicitly.
# 
# Note: `repeat_dataset` should be `False` when using the embedding training cache, while the argument `num_epochs` in `Model::fit` specifies the number of training epochs in this mode.

# In[1]:


get_ipython().run_cell_magic('writefile', 'wdl_train.py', 'import hugectr\nfrom mpi4py import MPI\nsolver = hugectr.CreateSolver(max_eval_batches = 5000,\n                              batchsize_eval = 1024,\n                              batchsize = 1024,\n                              lr = 0.001,\n                              vvgpu = [[0]],\n                              i64_input_key = False,\n                              use_mixed_precision = False,\n                              repeat_dataset = False,\n                              use_cuda_graph = True)\nreader = hugectr.DataReaderParams(data_reader_type = hugectr.DataReaderType_t.Norm,\n                          source = ["wdl_data/file_list."+str(i)+".txt" for i in range(2)],\n                          keyset = ["wdl_data/file_list."+str(i)+".keyset" for i in range(2)],\n                          eval_source = "wdl_data/file_list.2.txt",\n                          check_type = hugectr.Check_t.Sum)\noptimizer = hugectr.CreateOptimizer(optimizer_type = hugectr.Optimizer_t.Adam)\nhc_cnfg = hugectr.CreateHMemCache(num_blocks = 2, target_hit_rate = 0.5, max_num_evict = 0)\netc = hugectr.CreateETC(ps_types = [hugectr.TrainPSType_t.Staged, hugectr.TrainPSType_t.Cached],\n                        sparse_models = ["./wdl_0_sparse_model", "./wdl_1_sparse_model"],\n                        local_paths = ["./"], hmem_cache_configs = [hc_cnfg])\nmodel = hugectr.Model(solver, reader, optimizer, etc)\nmodel.add(hugectr.Input(label_dim = 1, label_name = "label",\n                        dense_dim = 13, dense_name = "dense",\n                        data_reader_sparse_param_array = \n                        [hugectr.DataReaderSparseParam("wide_data", 30, True, 1),\n                        hugectr.DataReaderSparseParam("deep_data", 2, False, 26)]))\nmodel.add(hugectr.SparseEmbedding(embedding_type = hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash, \n                            workspace_size_per_gpu_in_mb = 69,\n                            embedding_vec_size = 1,\n                            combiner = "sum",\n                            sparse_embedding_name = "sparse_embedding2",\n                            bottom_name = "wide_data",\n                            optimizer = optimizer))\nmodel.add(hugectr.SparseEmbedding(embedding_type = hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash, \n                            workspace_size_per_gpu_in_mb = 1074,\n                            embedding_vec_size = 16,\n                            combiner = "sum",\n                            sparse_embedding_name = "sparse_embedding1",\n                            bottom_name = "deep_data",\n                            optimizer = optimizer))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Reshape,\n                            bottom_names = ["sparse_embedding1"],\n                            top_names = ["reshape1"],\n                            leading_dim=416))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Reshape,\n                            bottom_names = ["sparse_embedding2"],\n                            top_names = ["reshape2"],\n                            leading_dim=1))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Concat,\n                            bottom_names = ["reshape1", "dense"], top_names = ["concat1"]))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,\n                            bottom_names = ["concat1"],\n                            top_names = ["fc1"],\n                            num_output=1024))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,\n                            bottom_names = ["fc1"],\n                            top_names = ["relu1"]))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Dropout,\n                            bottom_names = ["relu1"],\n                            top_names = ["dropout1"],\n                            dropout_rate=0.5))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,\n                            bottom_names = ["dropout1"],\n                            top_names = ["fc2"],\n                            num_output=1024))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,\n                            bottom_names = ["fc2"],\n                            top_names = ["relu2"]))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Dropout,\n                            bottom_names = ["relu2"],\n                            top_names = ["dropout2"],\n                            dropout_rate=0.5))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,\n                            bottom_names = ["dropout2"],\n                            top_names = ["fc3"],\n                            num_output=1))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Add,\n                            bottom_names = ["fc3", "reshape2"],\n                            top_names = ["add1"]))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.BinaryCrossEntropyLoss,\n                            bottom_names = ["add1", "label"],\n                            top_names = ["loss"]))\nmodel.compile()\nmodel.summary()\nmodel.graph_to_json(graph_config_file = "wdl.json")\nmodel.fit(num_epochs = 1, display = 500, eval_interval = 1000)\n# Get the updated embedding features in model.fit()\n# updated_model = model.get_incremental_model()\nmodel.set_source(source = ["wdl_data/file_list.3.txt", "wdl_data/file_list.4.txt"], keyset = ["wdl_data/file_list.3.keyset", "wdl_data/file_list.4.keyset"], eval_source = "wdl_data/file_list.5.txt")\nmodel.fit(num_epochs = 1, display = 500, eval_interval = 1000)\n# Get the updated embedding features in model.fit()\nupdated_model = model.get_incremental_model()\nmodel.save_params_to_files("wdl_etc")\n')


# In[2]:


get_ipython().system('python3 wdl_train.py')


# ### Continuous Training with the Low-level API
# 
# This section gives the code sample for continuous training using the low-level API.
# The program logic is the same as the preceding code sample.
# 
# Although the low-level APIs provide fine-grained control of the training logic, we encourage you to use the high-level API if it can satisfy your requirements because the naked data reader and embedding training cache logics are not straightforward and error prone.
# 
# For more about the low-level API, please refer to [Low-level Training API](https://nvidia-merlin.github.io/HugeCTR/master/api/python_interface.html#low-level-training-api) and samples of [Low-level Training](./hugectr_criteo.ipynb).

# In[3]:


get_ipython().run_cell_magic('writefile', 'wdl_etc.py', 'import hugectr\nfrom mpi4py import MPI\nsolver = hugectr.CreateSolver(max_eval_batches = 5000,\n                              batchsize_eval = 1024,\n                              batchsize = 1024,\n                              vvgpu = [[0]],\n                              i64_input_key = False,\n                              use_mixed_precision = False,\n                              repeat_dataset = False,\n                              use_cuda_graph = True)\nreader = hugectr.DataReaderParams(data_reader_type = hugectr.DataReaderType_t.Norm,\n                          source = ["wdl_data/file_list."+str(i)+".txt" for i in range(2)],\n                          keyset = ["wdl_data/file_list."+str(i)+".keyset" for i in range(2)],\n                          eval_source = "wdl_data/file_list.2.txt",\n                          check_type = hugectr.Check_t.Sum)\noptimizer = hugectr.CreateOptimizer(optimizer_type = hugectr.Optimizer_t.Adam)\nhc_cnfg = hugectr.CreateHMemCache(num_blocks = 2, target_hit_rate = 0.5, max_num_evict = 0)\netc = hugectr.CreateETC(ps_types = [hugectr.TrainPSType_t.Staged, hugectr.TrainPSType_t.Cached],\n                        sparse_models = ["./wdl_0_sparse_model", "./wdl_1_sparse_model"],\n                        local_paths = ["./"], hmem_cache_configs = [hc_cnfg])\nmodel = hugectr.Model(solver, reader, optimizer, etc)\nmodel.construct_from_json(graph_config_file = "wdl.json", include_dense_network = True)\nmodel.compile()\nlr_sch = model.get_learning_rate_scheduler()\ndata_reader_train = model.get_data_reader_train()\ndata_reader_eval = model.get_data_reader_eval()\netc = model.get_embedding_training_cache()\ndataset = [("wdl_data/file_list."+str(i)+".txt", "wdl_data/file_list."+str(i)+".keyset") for i in range(2)]\ndata_reader_eval.set_source("wdl_data/file_list.2.txt")\ndata_reader_eval_flag = True\niteration = 0\nfor file_list, keyset_file in dataset:\n  data_reader_train.set_source(file_list)\n  data_reader_train_flag = True\n  etc.update(keyset_file)\n  while True:\n    lr = lr_sch.get_next()\n    model.set_learning_rate(lr)\n    data_reader_train_flag = model.train()\n    if not data_reader_train_flag:\n      break\n    if iteration % 1000 == 0:\n      batches = 0\n      while data_reader_eval_flag:\n        if batches >= solver.max_eval_batches:\n          break\n        data_reader_eval_flag = model.eval()\n        batches += 1\n      if not data_reader_eval_flag:\n        data_reader_eval.set_source()\n        data_reader_eval_flag = True\n      metrics = model.get_eval_metrics()\n      print("[HUGECTR][INFO] iter: {}, metrics: {}".format(iteration, metrics))\n    iteration += 1\n  print("[HUGECTR][INFO] trained with data in {}".format(file_list))\n\ndataset = [("wdl_data/file_list."+str(i)+".txt", "wdl_data/file_list."+str(i)+".keyset") for i in range(3, 5)]\nfor file_list, keyset_file in dataset:\n  data_reader_train.set_source(file_list)\n  data_reader_train_flag = True\n  etc.update(keyset_file)\n  while True:\n    lr = lr_sch.get_next()\n    model.set_learning_rate(lr)\n    data_reader_train_flag = model.train()\n    if not data_reader_train_flag:\n      break\n    if iteration % 1000 == 0:\n      batches = 0\n      while data_reader_eval_flag:\n        if batches >= solver.max_eval_batches:\n          break\n        data_reader_eval_flag = model.eval()\n        batches += 1\n      if not data_reader_eval_flag:\n        data_reader_eval.set_source()\n        data_reader_eval_flag = True\n      metrics = model.get_eval_metrics()\n      print("[HUGECTR][INFO] iter: {}, metrics: {}".format(iteration, metrics))\n    iteration += 1\n  print("[HUGECTR][INFO] trained with data in {}".format(file_list))\nincremental_model = model.get_incremental_model()\nmodel.save_params_to_files("wdl_etc")\n')


# In[4]:


get_ipython().system('python3 wdl_etc.py')

