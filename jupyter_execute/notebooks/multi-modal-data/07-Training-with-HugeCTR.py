#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Copyright 2021 NVIDIA Corporation. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


# <img src="http://developer.download.nvidia.com/compute/machine-learning/frameworks/nvidia_logo.png" style="width: 90px; float: right;">
# 
# # Training HugeCTR Model with Pre-trained Embeddings
# 
# In this notebook, we will train a deep neural network for predicting user's rating (binary target with 1 for ratings `>3` and 0 for  ratings `<=3`). The two categorical features are `userId` and `movieId`.
# 
# We will also make use of movie's pretrained embeddings, extracted in the previous notebooks.

# ## Loading pretrained movie features into non-trainable embedding layer

# In[2]:


# loading NVTabular movie encoding
import pandas as pd
import os

INPUT_DATA_DIR = './data'
movie_mapping = pd.read_parquet(os.path.join(INPUT_DATA_DIR, "workflow-hugectr/categories/unique.movieId.parquet"))


# In[3]:


movie_mapping.tail()


# In[4]:


feature_df = pd.read_parquet('feature_df.parquet')
print(feature_df.shape)
feature_df.head()


# In[5]:


feature_df.set_index('movieId', inplace=True)


# In[6]:


from tqdm import tqdm
import numpy as np

num_tokens = len(movie_mapping)
embedding_dim = 2048+1024
hits = 0
misses = 0

# Prepare embedding matrix
embedding_matrix = np.zeros((num_tokens, embedding_dim))

print("Loading pretrained embedding matrix...")
for i, row in tqdm(movie_mapping.iterrows(), total=len(movie_mapping)):
    movieId = row['movieId']
    if movieId in feature_df.index: 
        embedding_vector = feature_df.loc[movieId]
        # embedding found
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1
print("Found features for %d movies (%d misses)" % (hits, misses))


# In[7]:


embedding_dim


# In[8]:


embedding_matrix


# Next, we write the pretrained embedding to a raw format supported by HugeCTR.
# 
# Note: As of version 3.2, HugeCTR only supports a maximum embedding size of 1024. Hence, we shall be using the first 512 element of image embedding plus 512 element of text embedding.

# In[9]:


import struct

PRETRAINED_EMBEDDING_SIZE = 1024

def convert_pretrained_embeddings_to_sparse_model(keys, pre_trained_sparse_embeddings, hugectr_sparse_model, embedding_vec_size):
    os.system("mkdir -p {}".format(hugectr_sparse_model))
    with open("{}/key".format(hugectr_sparse_model), 'wb') as key_file, \
        open("{}/emb_vector".format(hugectr_sparse_model), 'wb') as vec_file:
                
        for i, key in enumerate(keys):
            vec = np.concatenate([pre_trained_sparse_embeddings[i,:int(PRETRAINED_EMBEDDING_SIZE/2)], pre_trained_sparse_embeddings[i, 1024:1024+int(PRETRAINED_EMBEDDING_SIZE/2)]])
            key_struct = struct.pack('q', key)
            vec_struct = struct.pack(str(embedding_vec_size) + "f", *vec)
            key_file.write(key_struct)
            vec_file.write(vec_struct)

keys = list(movie_mapping.index)
convert_pretrained_embeddings_to_sparse_model(keys, embedding_matrix, 'hugectr_pretrained_embedding.model', embedding_vec_size=PRETRAINED_EMBEDDING_SIZE) # HugeCTR not supporting embedding size > 1024


# ## Define and train model
# 
# In this section, we define and train the model. The model comprise trainable embedding layers for categorical features (`userId`, `movieId`) and pretrained (non-trainable) embedding layer for movie features.
# 
# We will write the model to `./model.py` and execute it afterwards.

# First, we need the cardinalities of each categorical feature to assign as `slot_size_array` in the model below.

# In[10]:


import nvtabular as nvt
from nvtabular.ops import get_embedding_sizes

workflow = nvt.Workflow.load(os.path.join(INPUT_DATA_DIR, "workflow-hugectr"))

embeddings = get_embedding_sizes(workflow)
print(embeddings)

#{'userId': (162542, 512), 'movieId': (56586, 512), 'movieId_duplicate': (56586, 512)}


# We use `graph_to_json` to convert the model to a JSON configuration, required for the inference.

# In[11]:


get_ipython().run_cell_magic('writefile', "'./model.py'", '\nimport hugectr\nfrom mpi4py import MPI  # noqa\nINPUT_DATA_DIR = \'./data/\'\n\nsolver = hugectr.CreateSolver(\n    vvgpu=[[0]],\n    batchsize=2048,\n    batchsize_eval=2048,\n    max_eval_batches=160,\n    i64_input_key=True,\n    use_mixed_precision=False,\n    repeat_dataset=True,\n)\noptimizer = hugectr.CreateOptimizer(optimizer_type=hugectr.Optimizer_t.Adam)\nreader = hugectr.DataReaderParams(\n    data_reader_type=hugectr.DataReaderType_t.Parquet,\n    source=[INPUT_DATA_DIR + "train-hugectr/_file_list.txt"],\n    eval_source=INPUT_DATA_DIR + "valid-hugectr/_file_list.txt",\n    check_type=hugectr.Check_t.Non,\n    slot_size_array=[162542, 56586, 21, 56586],\n)\n\nmodel = hugectr.Model(solver, reader, optimizer)\n\nmodel.add(\n    hugectr.Input(\n        label_dim=1,\n        label_name="label",\n        dense_dim=0,\n        dense_name="dense",\n        data_reader_sparse_param_array=[\n            hugectr.DataReaderSparseParam("data1", nnz_per_slot=[1, 1, 2], is_fixed_length=False, slot_num=3),\n            hugectr.DataReaderSparseParam("movieId", nnz_per_slot=[1], is_fixed_length=True, slot_num=1)\n        ],\n    )\n)\nmodel.add(\n    hugectr.SparseEmbedding(\n        embedding_type=hugectr.Embedding_t.LocalizedSlotSparseEmbeddingHash,\n        workspace_size_per_gpu_in_mb=3000,\n        embedding_vec_size=16,\n        combiner="sum",\n        sparse_embedding_name="sparse_embedding1",\n        bottom_name="data1",\n        optimizer=optimizer,\n    )\n)\n\n# pretrained embedding\nmodel.add(\n    hugectr.SparseEmbedding(\n        embedding_type=hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash,\n        workspace_size_per_gpu_in_mb=3000,\n        embedding_vec_size=1024,\n        combiner="sum",\n        sparse_embedding_name="pretrained_embedding",\n        bottom_name="movieId",\n        optimizer=optimizer,\n    )\n)\n\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Reshape,\n                            bottom_names = ["sparse_embedding1"],\n                            top_names = ["reshape1"],\n                            leading_dim=48))\n\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Reshape,\n                            bottom_names = ["pretrained_embedding"],\n                            top_names = ["reshape2"],\n                            leading_dim=1024))\n\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Concat,\n                            bottom_names = ["reshape1", "reshape2"],\n                            top_names = ["concat1"]))\n\nmodel.add(\n    hugectr.DenseLayer(\n        layer_type=hugectr.Layer_t.InnerProduct,\n        bottom_names=["concat1"],\n        top_names=["fc1"],\n        num_output=128,\n    )\n)\nmodel.add(\n    hugectr.DenseLayer(\n        layer_type=hugectr.Layer_t.ReLU,\n        bottom_names=["fc1"],\n        top_names=["relu1"],\n    )\n)\nmodel.add(\n    hugectr.DenseLayer(\n        layer_type=hugectr.Layer_t.InnerProduct,\n        bottom_names=["relu1"],\n        top_names=["fc2"],\n        num_output=128,\n    )\n)\nmodel.add(\n    hugectr.DenseLayer(\n        layer_type=hugectr.Layer_t.ReLU,\n        bottom_names=["fc2"],\n        top_names=["relu2"],\n    )\n)\nmodel.add(\n    hugectr.DenseLayer(\n        layer_type=hugectr.Layer_t.InnerProduct,\n        bottom_names=["relu2"],\n        top_names=["fc3"],\n        num_output=1,\n    )\n)\nmodel.add(\n    hugectr.DenseLayer(\n        layer_type=hugectr.Layer_t.BinaryCrossEntropyLoss,\n        bottom_names=["fc3", "label"],\n        top_names=["loss"],\n    )\n)\nmodel.compile()\nmodel.summary()\n\n# Load the pretrained embedding layer\nmodel.load_sparse_weights({"pretrained_embedding": "./hugectr_pretrained_embedding.model"})\nmodel.freeze_embedding("pretrained_embedding")\n\nmodel.fit(max_iter=10001, display=100, eval_interval=200, snapshot=5000)\nmodel.graph_to_json(graph_config_file="hugectr-movielens.json")\n')


# We train our model.

# In[ ]:


get_ipython().system('python model.py')

