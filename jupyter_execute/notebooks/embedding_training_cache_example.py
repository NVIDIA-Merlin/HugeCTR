#!/usr/bin/env python
# coding: utf-8

# <img src="http://developer.download.nvidia.com/compute/machine-learning/frameworks/nvidia_logo.png" style="width: 90px; float: right;">
# 
# # Embedding Training Cache Example

# ## Overview
# [Embedding Training Cache](https://nvidia-merlin.github.io/HugeCTR/master/hugectr_embedding_training_cache.html) enables you to train huge models that cannot fit into GPU memory in one time. In this example, we will go through an end-to-end training procedure using the embedding training cache feature of HugeCTR. We are going to use the Criteo dataset as our data source and NVTabular as our data preprocessing tool. 

# ## Table of Contents
# -  [Installation](#installation)
# -  [Data Preparation](#data-preparation)
# -  [Extract keyset](#extract-keyset)
# -  [Training using HugeCTR](#training-using-hugectr)

# ## Installation
# 
# ### Get HugeCTR from NVIDIA GPU Cloud
# 
# The HugeCTR Python module is preinstalled in the 22.04 and later [Merlin Training Container](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-training): `nvcr.io/nvidia/merlin/merlin-training:22.04`.
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

# ## Data Preparation

# First, make a folder to store our data:

# In[1]:


get_ipython().system('mkdir etc_data')


# Second, make a script that uses the [HugeCTR Data Generator](https://nvidia-merlin.github.io/HugeCTR/master/hugectr_user_guide.html#generating-synthetic-data-and-benchmarks) to generate datasets:

# In[2]:


get_ipython().run_cell_magic('writefile', 'generate_data.py', '\nimport hugectr\nfrom hugectr.tools import DataGenerator, DataGeneratorParams\nfrom mpi4py import MPI\nimport argparse\nparser = argparse.ArgumentParser(description=("Data Generation"))\n\nparser.add_argument("--num_files", type=int, help="number of files in training data", default = 8)\nparser.add_argument("--eval_num_files", type=int, help="number of files in validation data", default = 2)\nparser.add_argument(\'--num_samples_per_file\', type=int, help="number of samples per file", default=1000000)\nparser.add_argument(\'--dir_name\', type=str, help="data directory name(Required)")\nargs = parser.parse_args()\n\ndata_generator_params = DataGeneratorParams(\n  format = hugectr.DataReaderType_t.Parquet,\n  label_dim = 1,\n  dense_dim = 13,\n  num_slot = 26,\n  num_files = args.num_files,\n  eval_num_files = args.eval_num_files,\n  i64_input_key = True,\n  num_samples_per_file = args.num_samples_per_file,\n  source = "./etc_data/" + args.dir_name + "/file_list.txt",\n  eval_source = "./etc_data/" + args.dir_name + "/file_list_test.txt",\n  slot_size_array = [12988, 7129, 8720, 5820, 15196, 4, 4914, 1020, 30, 14274, 10220, 15088, 10, 1518, 3672, 48, 4, 820, 15, 12817, 13908, 13447, 9447, 5867, 45, 33],\n  # for parquet, check_type doesn\'t make any difference\n  check_type = hugectr.Check_t.Non,\n  dist_type = hugectr.Distribution_t.PowerLaw,\n  power_law_type = hugectr.PowerLaw_t.Short)\ndata_generator = DataGenerator(data_generator_params)\ndata_generator.generate()\n')


# In[3]:


get_ipython().system('python generate_data.py --dir_name "file0"')


# In[4]:


get_ipython().system('python generate_data.py --dir_name "file1"')


# ## Extract Keyset

# The HugeCTR repository on GitHub includes a keyset generator script for Parquet datasets. See the `generate_keyset.py` file in the [keyset_scripts](https://github.com/NVIDIA-Merlin/HugeCTR/tree/master/tools/keyset_scripts) directory of the repository. We can use the script to generate keyset for our training datasets.

# In[5]:


get_ipython().system('python generate_keyset.py --src_dir_path ./etc_data/file0/train --keyset_path ./etc_data/file0/train/_hugectr.keyset  --slot_size_array 12988 7129 8720 5820 15196 4 4914 1020 30 14274 10220 15088 10 1518 3672 48 4 820 15 12817 13908 13447 9447 5867 45 33')


# Do the same thing for file2:

# In[6]:


get_ipython().system('python generate_keyset.py --src_dir_path ./etc_data/file1/train --keyset_path ./etc_data/file1/train/_hugectr.keyset  --slot_size_array 12988 7129 8720 5820 15196 4 4914 1020 30 14274 10220 15088 10 1518 3672 48 4 820 15 12817 13908 13447 9447 5867 45 33')


# Run `ls -l ./data` to make sure we have data and keyset ready:

# In[7]:


get_ipython().system('ls -l ./etc_data/file0/train')


# In[8]:


get_ipython().system('ls -l ./etc_data/file1/train')


# ## Training using HugeCTR

# In[9]:


get_ipython().run_cell_magic('writefile', 'etc_sample.py', 'import hugectr\nfrom mpi4py import MPI\nsolver = hugectr.CreateSolver(max_eval_batches = 5000,\n                              batchsize_eval = 1024,\n                              batchsize = 1024,\n                              lr = 0.001,\n                              vvgpu = [[0]],\n                              i64_input_key = True,\n                              use_mixed_precision = False,\n                              repeat_dataset = False)\nreader = hugectr.DataReaderParams(data_reader_type = hugectr.DataReaderType_t.Parquet,\n                          source = ["./etc_data/file0/file_list.txt"],\n                          keyset = ["./etc_data/file0/train/_hugectr.keyset"],\n                          eval_source = "./etc_data/file0/file_list_test.txt",\n                          slot_size_array = [12988, 7129, 8720, 5820, 15196, 4, 4914, 1020, 30, 14274, 10220, 15088, 10, 1518, 3672, 48, 4, 820, 15, 12817, 13908, 13447, 9447, 5867, 45, 33],\n                          check_type = hugectr.Check_t.Non)\noptimizer = hugectr.CreateOptimizer(optimizer_type = hugectr.Optimizer_t.Adam)\nhc_cnfg = hugectr.CreateHMemCache(num_blocks = 1, target_hit_rate = 0.5, max_num_evict = 0)\netc = hugectr.CreateETC(ps_types = [hugectr.TrainPSType_t.Cached],\n                       sparse_models = ["./dcn_sparse_model"],\n                       local_paths = ["./"], hmem_cache_configs = [hc_cnfg])\nmodel = hugectr.Model(solver, reader, optimizer, etc)\nmodel.add(hugectr.Input(label_dim = 1, label_name = "label",\n                        dense_dim = 13, dense_name = "dense",\n                        data_reader_sparse_param_array = \n                        [hugectr.DataReaderSparseParam("data1", 1, True, 26)]))\nmodel.add(hugectr.SparseEmbedding(embedding_type = hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash, \n                            workspace_size_per_gpu_in_mb = 5000,\n                            embedding_vec_size = 16,\n                            combiner = "sum",\n                            sparse_embedding_name = "sparse_embedding1",\n                            bottom_name = "data1",\n                            optimizer = optimizer))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Reshape,\n                            bottom_names = ["sparse_embedding1"],\n                            top_names = ["reshape1"],\n                            leading_dim=416))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Concat,\n                            bottom_names = ["reshape1", "dense"], top_names = ["concat1"]))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.MultiCross,\n                            bottom_names = ["concat1"],\n                            top_names = ["multicross1"],\n                            num_layers=6))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,\n                            bottom_names = ["concat1"],\n                            top_names = ["fc1"],\n                            num_output=1024))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,\n                            bottom_names = ["fc1"],\n                            top_names = ["relu1"]))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Dropout,\n                            bottom_names = ["relu1"],\n                            top_names = ["dropout1"],\n                            dropout_rate=0.5))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,\n                            bottom_names = ["dropout1"],\n                            top_names = ["fc2"],\n                            num_output=1024))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,\n                            bottom_names = ["fc2"],\n                            top_names = ["relu2"]))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Dropout,\n                            bottom_names = ["relu2"],\n                            top_names = ["dropout2"],\n                            dropout_rate=0.5))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Concat,\n                            bottom_names = ["dropout2", "multicross1"],\n                            top_names = ["concat2"]))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,\n                            bottom_names = ["concat2"],\n                            top_names = ["fc3"],\n                            num_output=1))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.BinaryCrossEntropyLoss,\n                            bottom_names = ["fc3", "label"],\n                            top_names = ["loss"]))\nmodel.compile()\nmodel.summary()\nmodel.graph_to_json(graph_config_file = "dcn.json")\nmodel.fit(num_epochs = 1, display = 500, eval_interval = 1000)\n\nmodel.set_source(source = ["etc_data/file1/file_list.txt"], keyset = ["etc_data/file1/train/_hugectr.keyset"], eval_source = "etc_data/file1/file_list_test.txt")\nmodel.fit(num_epochs = 1, display = 500, eval_interval = 1000)\n\nmodel.save_params_to_files("dcn_etc")\n')


# In[10]:


get_ipython().system('python3 etc_sample.py')

