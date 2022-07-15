#!/usr/bin/env python
# coding: utf-8

# <img src="http://developer.download.nvidia.com/compute/machine-learning/frameworks/nvidia_logo.png" style="width: 90px; float: right;">
# 
# # Introduction to the HugeCTR Python Interface

# ## Overview
# 
# HugeCTR version 3.1 introduces an enhanced Python interface
# The interface supports continuous training and inference with high-level APIs.
# There are four main improvements.
# 
# * First, the model graph can be constructed and dumped to a JSON file with Python code and it saves users from writing JSON configuration files.
# * Second, the API supports the feature of embedding training cache with high-level APIs and extends it further for online training cases.
# (For learn about continuous training, you can view the [example notebook](./continuous_training.ipynb)).
# * Third, the freezing method is provided for both sparse embedding and dense network.
# This method enables transfer learning and fine-tuning for CTR tasks.
# * Finally, the pre-trained embeddings in other formats can be converted to HugeCTR sparse models and then loaded to facilitate the training process. This is shown in the Load Pre-trained Embeddings section of this notebook.
# 
# This notebook explains how to access and use the enhanced HugeCTR Python interface.
# Although the low-level training APIs are still maintained for users who want to have precise control of each training iteration, migrating to the high-level training APIs is strongly recommended.
# For more details of the usage of the Python API, refer to the [HugeCTR Python Interface](https://nvidia-merlin.github.io/HugeCTR/master/api/python_interface.html) documentation.

# ## Installation
# 
# ### Get HugeCTR from NGC
# 
# The HugeCTR Python module is preinstalled in the 22.07 and later [Merlin Training Container](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-hugectr): `nvcr.io/nvidia/merlin/merlin-hugectr:22.07`.
# 
# You can check the existence of the required libraries by running the following Python code after launching this container.
# 
# ```bash
# $ python3 -c "import hugectr"
# ```
# 
# > If you prefer to build HugeCTR from the source code instead of using the NGC container,
# > refer to the
# > [How to Start Your Development](https://nvidia-merlin.github.io/HugeCTR/master/hugectr_contributor_guide.html#how-to-start-your-development)
# > documentation.

# ## DCN Model
# 
# ### Download and Preprocess Data
# 
# 1. Download the Criteo dataset using the following command:
# 
#    ```shell
#    $ cd ${project-root}/tools
#    $ wget https://storage.googleapis.com/criteo-cail-datasets/day_1.gz
#    ```
#    
#    In preprocessing, we will further reduce the amounts of data to speedup the preprocessing, fill missing values, remove the feature values whose occurrences are very rare, etc. Here we choose pandas preprocessing method to make the dataset ready for HugeCTR training.
# 
# 2. Preprocessing by Pandas using the following command:
# 
#    ```shell
#    $ bash preprocess.sh 1 dcn_data pandas 1 0
#    ```
#    
#    The first argument represents the dataset postfix. It is 1 here since day_1 is used. The second argument dcn_data is where the preprocessed data is stored. The fourth argument (one after pandas) 1 embodies that the normalization is applied to dense features. The last argument 0 means that the feature crossing is not applied.
# 
# 3. Create a soft link to the dataset folder using the following command:
# 
#    ```shell
#    $ ln ${project-root}/tools/dcn_data ${project_root}/notebooks/dcn_data
#    ```
#    
# **Note**: It will take a while (dozens of minutes) to preprocess the dataset. Please make sure that it is finished successfully before moving forward to the next section.

# ### Train from Scratch
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

# In[1]:


import hugectr


# In[2]:


get_ipython().run_cell_magic('writefile', 'dcn_train.py', 'import hugectr\nfrom mpi4py import MPI\nsolver = hugectr.CreateSolver(max_eval_batches = 1500,\n                              batchsize_eval = 4096,\n                              batchsize = 4096,\n                              lr = 0.001,\n                              vvgpu = [[0]],\n                              i64_input_key = False,\n                              use_mixed_precision = False,\n                              repeat_dataset = True,\n                              use_cuda_graph = True)\nreader = hugectr.DataReaderParams(data_reader_type = hugectr.DataReaderType_t.Norm,\n                                  source = ["./dcn_data/file_list.txt"],\n                                  eval_source = "./dcn_data/file_list_test.txt",\n                                  check_type = hugectr.Check_t.Sum)\noptimizer = hugectr.CreateOptimizer(optimizer_type = hugectr.Optimizer_t.Adam)\nmodel = hugectr.Model(solver, reader, optimizer)\nmodel.add(hugectr.Input(label_dim = 1, label_name = "label",\n                        dense_dim = 13, dense_name = "dense",\n                        data_reader_sparse_param_array = \n                        [hugectr.DataReaderSparseParam("data1", 2, False, 26)]))\nmodel.add(hugectr.SparseEmbedding(embedding_type = hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash, \n                            workspace_size_per_gpu_in_mb = 264,\n                            embedding_vec_size = 16,\n                            combiner = "sum",\n                            sparse_embedding_name = "sparse_embedding1",\n                            bottom_name = "data1",\n                            optimizer = optimizer))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Reshape,\n                            bottom_names = ["sparse_embedding1"],\n                            top_names = ["reshape1"],\n                            leading_dim=416))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Concat,\n                            bottom_names = ["reshape1", "dense"], top_names = ["concat1"]))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.MultiCross,\n                            bottom_names = ["concat1"],\n                            top_names = ["multicross1"],\n                            num_layers=6))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,\n                            bottom_names = ["concat1"],\n                            top_names = ["fc1"],\n                            num_output=1024))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,\n                            bottom_names = ["fc1"],\n                            top_names = ["relu1"]))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Dropout,\n                            bottom_names = ["relu1"],\n                            top_names = ["dropout1"],\n                            dropout_rate=0.5))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,\n                            bottom_names = ["dropout1"],\n                            top_names = ["fc2"],\n                            num_output=1024))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,\n                            bottom_names = ["fc2"],\n                            top_names = ["relu2"]))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Dropout,\n                            bottom_names = ["relu2"],\n                            top_names = ["dropout2"],\n                            dropout_rate=0.5))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Concat,\n                            bottom_names = ["dropout2", "multicross1"],\n                            top_names = ["concat2"]))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,\n                            bottom_names = ["concat2"],\n                            top_names = ["fc3"],\n                            num_output=1))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.BinaryCrossEntropyLoss,\n                            bottom_names = ["fc3", "label"],\n                            top_names = ["loss"]))\nmodel.compile()\nmodel.summary()\nmodel.graph_to_json(graph_config_file = "dcn.json")\nmodel.fit(max_iter = 1200, display = 500, eval_interval = 100, snapshot = 1000, snapshot_prefix = "dcn")\n')


# In[3]:


get_ipython().system('python3 dcn_train.py')


# ### Continue Training
# 
# We can continue our training based on the saved model graph, model weights, and optimizer states by performing the following with Python APIs:
# 
# 1. Create the solver, reader and optimizer, then initialize the model.
# 2. Construct the model graph from the saved JSON file.
# 3. Compile the model and have an overview of the model graph.
# 4. Load the model weights and optimizer states.
# 5. Fit the model, save the model weights and optimizer states implicitly.

# In[4]:


get_ipython().system('ls *.model')


# In[5]:


get_ipython().run_cell_magic('writefile', 'dcn_continue.py', 'import hugectr\nfrom mpi4py import MPI\nsolver = hugectr.CreateSolver(max_eval_batches = 1500,\n                              batchsize_eval = 4096,\n                              batchsize = 4096,\n                              vvgpu = [[0]],\n                              i64_input_key = False,\n                              use_mixed_precision = False,\n                              repeat_dataset = True,\n                              use_cuda_graph = True)\nreader = hugectr.DataReaderParams(data_reader_type = hugectr.DataReaderType_t.Norm,\n                                  source = ["./dcn_data/file_list.txt"],\n                                  eval_source = "./dcn_data/file_list_test.txt",\n                                  check_type = hugectr.Check_t.Sum)\noptimizer = hugectr.CreateOptimizer(optimizer_type = hugectr.Optimizer_t.Adam)\nmodel = hugectr.Model(solver, reader, optimizer)\nmodel.construct_from_json(graph_config_file = "dcn.json", include_dense_network = True)\nmodel.compile()\nmodel.load_dense_weights("dcn_dense_1000.model")\nmodel.load_sparse_weights(["dcn0_sparse_1000.model"])\nmodel.load_dense_optimizer_states("dcn_opt_dense_1000.model")\nmodel.load_sparse_optimizer_states(["dcn0_opt_sparse_1000.model"])\nmodel.summary()\nmodel.fit(max_iter = 500, display = 50, eval_interval = 100, snapshot = 10000, snapshot_prefix = "dcn")\n')


# In[6]:


get_ipython().system('python3 dcn_continue.py')


# ### Inference
# 
# The HugeCTR inference is enabled by `hugectr.inference.InferenceSession.predict` method of InferenceSession.
# This method requires dense features, embedding columns, and row pointers of slots as the input and gives the prediction result as the output.
# We need to convert the Criteo data to inference format first.

# In[7]:


get_ipython().system('python3 ../tools/criteo_predict/criteo2predict.py --src_csv_path=dcn_data/val/test.txt --src_config=../tools/criteo_predict/dcn_data.json --dst_path=./dcn_csr.txt --batch_size=1024')


# We can then make inferences based on the saved model graph and model weights by performing the following with Python APIs:
# 
# 1. Configure the inference related parameters.
# 2. Create the inference session.
# 3. Make inference with the `InferenceSession.predict` method. 

# In[8]:


get_ipython().run_cell_magic('writefile', 'dcn_inference.py', 'from hugectr.inference import InferenceParams, CreateInferenceSession\nfrom mpi4py import MPI\n\ndef calculate_accuracy(labels, output):\n    num_samples = len(labels)\n    flags = [1 if ((labels[i] == 0 and output[i] <= 0.5) or (labels[i] == 1 and output[i] > 0.5)) else 0 for i in range(num_samples)]\n    correct_samples = sum(flags)\n    return float(correct_samples)/(float(num_samples)+1e-16)\n\ndata_file = open("dcn_csr.txt")\nconfig_file = "dcn.json"\nlabels = [int(item) for item in data_file.readline().split(\' \')]\ndense_features = [float(item) for item in data_file.readline().split(\' \') if item!="\\n"]\nembedding_columns = [int(item) for item in data_file.readline().split(\' \')]\nrow_ptrs = [int(item) for item in data_file.readline().split(\' \')]\n\n# create parameter server, embedding cache and inference session\ninference_params = InferenceParams(model_name = "dcn",\n                                max_batchsize = 1024,\n                                hit_rate_threshold = 0.6,\n                                dense_model_file = "./dcn_dense_1000.model",\n                                sparse_model_files = ["./dcn0_sparse_1000.model"],\n                                device_id = 0,\n                                use_gpu_embedding_cache = True,\n                                cache_size_percentage = 0.9,\n                                i64_input_key = False,\n                                use_mixed_precision = False)\ninference_session = CreateInferenceSession(config_file, inference_params)\noutput = inference_session.predict(dense_features, embedding_columns, row_ptrs)\naccuracy = calculate_accuracy(labels, output)\nprint("[HUGECTR][INFO] number samples: {}, accuracy: {}".format(len(labels), accuracy))\n')


# In[9]:


get_ipython().system('python3 dcn_inference.py')


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
#    $ ln ${project_root}/tools/wdl_data ${project_root}/notebooks/wdl_data
#    ```
#    
# **Note**: It will take a while (dozens of minutes) to preprocess the dataset. Please make sure that it is finished successfully before moving forward to the next section.

# ### Train from Scratch
# 
# We can train fom scratch, dump the model graph to a JSON file, and save the model weights and optimizer states by performing the same steps that we followed with the DCN Model.

# In[10]:


get_ipython().run_cell_magic('writefile', 'wdl_train.py', 'import hugectr\nfrom mpi4py import MPI\nsolver = hugectr.CreateSolver(max_eval_batches = 5000,\n                              batchsize_eval = 1024,\n                              batchsize = 1024,\n                              lr = 0.001,\n                              vvgpu = [[0]],\n                              i64_input_key = False,\n                              use_mixed_precision = False,\n                              repeat_dataset = False,\n                              use_cuda_graph = True)\nreader = hugectr.DataReaderParams(data_reader_type = hugectr.DataReaderType_t.Norm,\n                          source = ["wdl_data/file_list.0.txt"],\n                          eval_source = "wdl_data/file_list.1.txt",\n                          check_type = hugectr.Check_t.Sum)\noptimizer = hugectr.CreateOptimizer(optimizer_type = hugectr.Optimizer_t.Adam)\nmodel = hugectr.Model(solver, reader, optimizer)\nmodel.add(hugectr.Input(label_dim = 1, label_name = "label",\n                        dense_dim = 13, dense_name = "dense",\n                        data_reader_sparse_param_array = \n                        [hugectr.DataReaderSparseParam("wide_data", 30, True, 1),\n                        hugectr.DataReaderSparseParam("deep_data", 2, False, 26)]))\nmodel.add(hugectr.SparseEmbedding(embedding_type = hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash, \n                            workspace_size_per_gpu_in_mb = 69,\n                            embedding_vec_size = 1,\n                            combiner = "sum",\n                            sparse_embedding_name = "sparse_embedding2",\n                            bottom_name = "wide_data",\n                            optimizer = optimizer))\nmodel.add(hugectr.SparseEmbedding(embedding_type = hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash, \n                            workspace_size_per_gpu_in_mb = 1074,\n                            embedding_vec_size = 16,\n                            combiner = "sum",\n                            sparse_embedding_name = "sparse_embedding1",\n                            bottom_name = "deep_data",\n                            optimizer = optimizer))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Reshape,\n                            bottom_names = ["sparse_embedding1"],\n                            top_names = ["reshape1"],\n                            leading_dim=416))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Reshape,\n                            bottom_names = ["sparse_embedding2"],\n                            top_names = ["reshape2"],\n                            leading_dim=1))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Concat,\n                            bottom_names = ["reshape1", "dense"], top_names = ["concat1"]))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,\n                            bottom_names = ["concat1"],\n                            top_names = ["fc1"],\n                            num_output=1024))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,\n                            bottom_names = ["fc1"],\n                            top_names = ["relu1"]))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Dropout,\n                            bottom_names = ["relu1"],\n                            top_names = ["dropout1"],\n                            dropout_rate=0.5))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,\n                            bottom_names = ["dropout1"],\n                            top_names = ["fc2"],\n                            num_output=1024))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,\n                            bottom_names = ["fc2"],\n                            top_names = ["relu2"]))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Dropout,\n                            bottom_names = ["relu2"],\n                            top_names = ["dropout2"],\n                            dropout_rate=0.5))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,\n                            bottom_names = ["dropout2"],\n                            top_names = ["fc3"],\n                            num_output=1))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Add,\n                            bottom_names = ["fc3", "reshape2"],\n                            top_names = ["add1"]))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.BinaryCrossEntropyLoss,\n                            bottom_names = ["add1", "label"],\n                            top_names = ["loss"]))\nmodel.compile()\nmodel.summary()\nmodel.graph_to_json(graph_config_file = "wdl.json")\nmodel.fit(num_epochs = 1, display = 500, eval_interval = 500, snapshot = 4000, snapshot_prefix = "wdl")\n')


# In[11]:


get_ipython().system('python3 wdl_train.py')


# ### Fine-tuning
# 
# We can only load the sparse embedding layers, their corresponding weights, and then construct a new dense network. The dense weights will be trained first and the sparse weights will be fine-tuned later. We can achieve this by performing the following with Python APIs:
# 
# 1. Create the solver, reader and optimizer, then initialize the model.
# 2. Load the sparse embedding layers from the saved JSON file.
# 3. Add the dense layers on top of the loaded model graph.
# 4. Compile the model and have an overview of the model graph.
# 5. Load the sparse weights and freeze the sparse embedding layers.
# 6. Train the dense weights.
# 7. Unfreeze the sparse embedding layers and freeze the dense layers, reset the learning rate scheduler with a small rate.
# 8. Fine-tune the sparse weights.

# In[12]:


get_ipython().run_cell_magic('writefile', 'wdl_fine_tune.py', 'import hugectr\nfrom mpi4py import MPI\nsolver = hugectr.CreateSolver(max_eval_batches = 5000,\n                              batchsize_eval = 1024,\n                              batchsize = 1024,\n                              vvgpu = [[0]],\n                              i64_input_key = False,\n                              use_mixed_precision = False,\n                              repeat_dataset = False,\n                              use_cuda_graph = True)\nreader = hugectr.DataReaderParams(data_reader_type = hugectr.DataReaderType_t.Norm,\n                          source = ["wdl_data/file_list.2.txt"],\n                          eval_source = "wdl_data/file_list.3.txt",\n                          check_type = hugectr.Check_t.Sum)\noptimizer = hugectr.CreateOptimizer(optimizer_type = hugectr.Optimizer_t.Adam)\nmodel = hugectr.Model(solver, reader, optimizer)\nmodel.construct_from_json(graph_config_file = "wdl.json", include_dense_network = False)\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Reshape,\n                            bottom_names = ["sparse_embedding1"],\n                            top_names = ["reshape1"],\n                            leading_dim=416))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Reshape,\n                            bottom_names = ["sparse_embedding2"],\n                            top_names = ["reshape2"],\n                            leading_dim=1))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Concat,\n                            bottom_names = ["reshape1", "reshape2", "dense"], top_names = ["concat1"]))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,\n                            bottom_names = ["concat1"],\n                            top_names = ["fc1"],\n                            num_output=1024))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,\n                            bottom_names = ["fc1"],\n                            top_names = ["relu1"]))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Dropout,\n                            bottom_names = ["relu1"],\n                            top_names = ["dropout1"],\n                            dropout_rate=0.5))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,\n                            bottom_names = ["dropout1"],\n                            top_names = ["fc2"],\n                            num_output=1))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.BinaryCrossEntropyLoss,\n                            bottom_names = ["fc2", "label"],\n                            top_names = ["loss"]))\nmodel.compile()\nmodel.summary()\nmodel.load_sparse_weights(["wdl0_sparse_4000.model", "wdl1_sparse_4000.model"])\nmodel.freeze_embedding()\nmodel.fit(num_epochs = 1, display = 500, eval_interval = 1000, snapshot = 100000, snapshot_prefix = "wdl")\nmodel.unfreeze_embedding()\nmodel.freeze_dense()\nmodel.reset_learning_rate_scheduler(base_lr = 0.0001)\nmodel.fit(num_epochs = 2, display = 500, eval_interval = 1000, snapshot = 100000, snapshot_prefix = "wdl")\n')


# In[13]:


get_ipython().system('python3 wdl_fine_tune.py')


# ### Load Pre-trained Embeddings
# 
# If you have the pre-trained embeddings in other formats, you can convert them to the HugeCTR sparse models and then load them to facilitate the training process. For the sake of simplicity and generality, we represent the pretrained embeddings with the dictionary of randomly initialized numpy arrays, of which the keys indicate the embedding keys and the array values embody the embedding values. It is worth mentioning that there are two embedding tables for the Wide&Deep model, and here we only load the pre-trained embeddings for one table and freeze the corresponding embedding layer.

# In[14]:


get_ipython().run_cell_magic('writefile', 'wdl_load_pretrained.py', 'import hugectr\nfrom mpi4py import MPI\nimport numpy as np\nimport os\nimport struct\nsolver = hugectr.CreateSolver(max_eval_batches = 5000,\n                              batchsize_eval = 1024,\n                              batchsize = 1024,\n                              vvgpu = [[0]],\n                              i64_input_key = False,\n                              use_mixed_precision = False,\n                              repeat_dataset = False,\n                              use_cuda_graph = True)\nreader = hugectr.DataReaderParams(data_reader_type = hugectr.DataReaderType_t.Norm,\n                          source = ["wdl_data/file_list.0.txt"],\n                          eval_source = "wdl_data/file_list.1.txt",\n                          check_type = hugectr.Check_t.Sum)\noptimizer = hugectr.CreateOptimizer(optimizer_type = hugectr.Optimizer_t.Adam)\nmodel = hugectr.Model(solver, reader, optimizer)\nmodel.construct_from_json(graph_config_file = "wdl.json", include_dense_network = True)\nmodel.compile()\nmodel.summary()\n\ndef convert_pretrained_embeddings_to_sparse_model(pre_trained_sparse_embeddings, hugectr_sparse_model, embedding_vec_size):\n    os.system("mkdir -p {}".format(hugectr_sparse_model))\n    with open("{}/key".format(hugectr_sparse_model), \'wb\') as key_file, \\\n        open("{}/emb_vector".format(hugectr_sparse_model), \'wb\') as vec_file:\n      for key in pre_trained_sparse_embeddings:\n        vec = pre_trained_sparse_embeddings[key]\n        key_struct = struct.pack(\'q\', key)\n        vec_struct = struct.pack(str(embedding_vec_size) + "f", *vec)\n        key_file.write(key_struct)\n        vec_file.write(vec_struct)\n\n# Convert the pretrained embeddings\npretrained_embeddings = dict()\nhugectr_sparse_model = "wdl1_pretrained.model"\nembedding_vec_size = 16\nkey_range = (0, 100000)\nfor key in range(key_range[0], key_range[1]):\n    pretrained_embeddings[key] = np.random.randn(embedding_vec_size).astype(np.float32)\nconvert_pretrained_embeddings_to_sparse_model(pretrained_embeddings, hugectr_sparse_model, embedding_vec_size)\nprint("Successfully convert pretrained embeddings to {}".format(hugectr_sparse_model))\n\n# Load the pretrained sparse models\nmodel.load_sparse_weights({"sparse_embedding1": hugectr_sparse_model})\nmodel.freeze_embedding("sparse_embedding1")\nmodel.fit(num_epochs = 1, display = 500, eval_interval = 1000, snapshot = 100000, snapshot_prefix = "wdl")\n')


# In[15]:


get_ipython().system('python3 wdl_load_pretrained.py')


# ### Low-level Training
# 
# The low-level training APIs are maintained in the enhanced HugeCTR Python interface. If you want to have precise control of each training iteration and each evaluation step, you may find it helpful to use these APIs. Since the data reader behavior is different in epoch mode and non-epoch mode, we should pay attention to how to tweak the data reader when using low-level training.
# We will demonstrate how to write the low-level training scripts for non-epoch mode, epoch mode, and embedding training cache mode.

# In[16]:


get_ipython().run_cell_magic('writefile', 'wdl_non_epoch.py', 'import hugectr\nfrom mpi4py import MPI\nsolver = hugectr.CreateSolver(max_eval_batches = 5000,\n                              batchsize_eval = 1024,\n                              batchsize = 1024,\n                              vvgpu = [[0]],\n                              i64_input_key = False,\n                              use_mixed_precision = False,\n                              repeat_dataset = True,\n                              use_cuda_graph = True)\nreader = hugectr.DataReaderParams(data_reader_type = hugectr.DataReaderType_t.Norm,\n                          source = ["wdl_data/file_list.0.txt"],\n                          eval_source = "wdl_data/file_list.1.txt",\n                          check_type = hugectr.Check_t.Sum)\noptimizer = hugectr.CreateOptimizer(optimizer_type = hugectr.Optimizer_t.Adam)\nmodel = hugectr.Model(solver, reader, optimizer)\nmodel.construct_from_json(graph_config_file = "wdl.json", include_dense_network = True)\nmodel.compile()\nmodel.start_data_reading()\nlr_sch = model.get_learning_rate_scheduler()\nmax_iter = 2000\nfor i in range(max_iter):\n    lr = lr_sch.get_next()\n    model.set_learning_rate(lr)\n    model.train()\n    if (i%100 == 0):\n        loss = model.get_current_loss()\n        print("[HUGECTR][INFO] iter: {}; loss: {}".format(i, loss))\n    if (i%1000 == 0 and i != 0):\n        for _ in range(solver.max_eval_batches):\n            model.eval()\n        metrics = model.get_eval_metrics()\n        print("[HUGECTR][INFO] iter: {}, {}".format(i, metrics))\nmodel.save_params_to_files("./", max_iter)\n')


# In[17]:


get_ipython().system('python3 wdl_non_epoch.py')


# In[18]:


get_ipython().run_cell_magic('writefile', 'wdl_epoch.py', 'import hugectr\nfrom mpi4py import MPI\nsolver = hugectr.CreateSolver(max_eval_batches = 5000,\n                              batchsize_eval = 1024,\n                              batchsize = 1024,\n                              vvgpu = [[0]],\n                              i64_input_key = False,\n                              use_mixed_precision = False,\n                              repeat_dataset = False,\n                              use_cuda_graph = True)\nreader = hugectr.DataReaderParams(data_reader_type = hugectr.DataReaderType_t.Norm,\n                          source = ["wdl_data/file_list.0.txt"],\n                          eval_source = "wdl_data/file_list.1.txt",\n                          check_type = hugectr.Check_t.Sum)\noptimizer = hugectr.CreateOptimizer(optimizer_type = hugectr.Optimizer_t.Adam)\nmodel = hugectr.Model(solver, reader, optimizer)\nmodel.construct_from_json(graph_config_file = "wdl.json", include_dense_network = True)\nmodel.compile()\nlr_sch = model.get_learning_rate_scheduler()\ndata_reader_train = model.get_data_reader_train()\ndata_reader_eval = model.get_data_reader_eval()\ndata_reader_eval.set_source()\ndata_reader_eval_flag = True\niteration = 0\nfor epoch in range(2):\n  print("[HUGECTR][INFO] epoch: ", epoch)\n  data_reader_train.set_source()\n  data_reader_train_flag = True\n  while True:\n    lr = lr_sch.get_next()\n    model.set_learning_rate(lr)\n    data_reader_train_flag = model.train()\n    if not data_reader_train_flag:\n      break\n    if iteration % 1000 == 0:\n      batches = 0\n      while data_reader_eval_flag:\n        if batches >= solver.max_eval_batches:\n          break\n        data_reader_eval_flag = model.eval()\n        batches += 1\n      if not data_reader_eval_flag:\n        data_reader_eval.set_source()\n        data_reader_eval_flag = True\n      metrics = model.get_eval_metrics()\n      print("[HUGECTR][INFO] iter: {}, metrics: {}".format(iteration, metrics))\n    iteration += 1\nmodel.save_params_to_files("./", iteration)\n')


# In[19]:


get_ipython().system('python3 wdl_epoch.py')

