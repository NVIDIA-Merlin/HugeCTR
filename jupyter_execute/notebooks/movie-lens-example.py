#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Copyright 2020 NVIDIA Corporation. All Rights Reserved.
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
# # HugeCTR demo on Movie lens data

# ## Overview
# 
# HugeCTR is a recommender-specific framework that is capable of distributed training across multiple GPUs and nodes for click-through-rate (CTR) estimation.
# HugeCTR is a component of NVIDIA Merlin ([documentation](https://nvidia-merlin.github.io/Merlin/main/README.html) | [GitHub](https://github.com/NVIDIA-Merlin/Merlin)).
# Merlin which is a framework that accelerates the entire pipeline from data ingestion and training to deploying GPU-accelerated recommender systems.
# 
# ### Learning objectives
# 
# * Training a deep-learning recommender model (DLRM) on the MovieLens 20M [dataset](https://grouplens.org/datasets/movielens/20m/).
# * Walk through data preprocessing, training a DLRM model with HugeCTR, and then using the movie embedding to answer item similarity queries.
# 

# ## Prerequisites
# 
# ### Docker containers
# 
# Start the notebook inside a running 22.06 or later NGC Docker container: `nvcr.io/nvidia/merlin/merlin-training:22.06`.
# The HugeCTR Python interface is installed to the path `/usr/local/hugectr/lib/` and the path is added to the environment variable `PYTHONPATH`.
# You can use the HugeCTR Python interface within the Docker container without any additional configuration.
# 
# ### Hardware
# 
# This notebook requires a Pascal, Volta, Turing, Ampere or newer GPUs, such as P100, V100, T4 or A100.
# You can view the GPU information with the `nvidia-smi` command:

# In[1]:


get_ipython().system('nvidia-smi')


# ## Data download and preprocessing
# 
# We first install a few extra utilities for data preprocessing.

# In[3]:


print("Downloading and installing 'tqdm' package.")
get_ipython().system('pip3 -q install torch tqdm')

print("Downloading and installing 'unzip' command")
get_ipython().system('conda install -y -q -c conda-forge unzip')


# Next, we download and unzip the MovieLens 20M [dataset](https://grouplens.org/datasets/movielens/20m/).

# In[4]:


print("Downloading and extracting 'Movie Lens 20M' dataset.")
get_ipython().system('wget -nc http://files.grouplens.org/datasets/movielens/ml-20m.zip -P data -q --show-progress')
get_ipython().system('unzip -n data/ml-20m.zip -d data')
get_ipython().system('ls ./data')


# ### MovieLens data preprocessing

# In[5]:


import pandas as pd
import torch
import tqdm

MIN_RATINGS = 20
USER_COLUMN = 'userId'
ITEM_COLUMN = 'movieId'


# In[6]:





# Next, we read the data into a Pandas dataframe and encode `userID` and `itemID` with integers.

# In[7]:


df = pd.read_csv('./data/ml-20m/ratings.csv')
print("Filtering out users with less than {} ratings".format(MIN_RATINGS))
grouped = df.groupby(USER_COLUMN)
df = grouped.filter(lambda x: len(x) >= MIN_RATINGS)

print("Mapping original user and item IDs to new sequential IDs")
df[USER_COLUMN], unique_users = pd.factorize(df[USER_COLUMN])
df[ITEM_COLUMN], unique_items = pd.factorize(df[ITEM_COLUMN])

nb_users = len(unique_users)
nb_items = len(unique_items)

print("Number of users: %d\nNumber of items: %d"%(len(unique_users), len(unique_items)))

# Save the mapping to do the inference later on
import pickle
with open('./mappings.pickle', 'wb') as handle:
    pickle.dump({"users": unique_users, "items": unique_items}, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Next, we split the data into a train and test set.
# The last movie each user has recently rated is used for the test set.

# In[8]:


# Need to sort before popping to get the last item
df.sort_values(by='timestamp', inplace=True)
    
# clean up data
del df['rating'], df['timestamp']
df = df.drop_duplicates() # assuming it keeps order

# now we have filtered and sorted by time data, we can split test data out
grouped_sorted = df.groupby(USER_COLUMN, group_keys=False)
test_data = grouped_sorted.tail(1).sort_values(by=USER_COLUMN)

# need to pop for each group
train_data = grouped_sorted.apply(lambda x: x.iloc[:-1])


# In[9]:


train_data['target']=1
test_data['target']=1
train_data.head()


# Because the MovieLens data contains only positive examples, first we define a utility function to generate negative samples.

# In[10]:


class _TestNegSampler:
    def __init__(self, train_ratings, nb_users, nb_items, nb_neg):
        self.nb_neg = nb_neg
        self.nb_users = nb_users 
        self.nb_items = nb_items 

        # compute unique ids for quickly created hash set and fast lookup
        ids = (train_ratings[:, 0] * self.nb_items) + train_ratings[:, 1]
        self.set = set(ids)

    def generate(self, batch_size=128*1024):
        users = torch.arange(0, self.nb_users).reshape([1, -1]).repeat([self.nb_neg, 1]).transpose(0, 1).reshape(-1)

        items = [-1] * len(users)

        random_items = torch.LongTensor(batch_size).random_(0, self.nb_items).tolist()
        print('Generating validation negatives...')
        for idx, u in enumerate(tqdm.tqdm(users.tolist())):
            if not random_items:
                random_items = torch.LongTensor(batch_size).random_(0, self.nb_items).tolist()
            j = random_items.pop()
            while u * self.nb_items + j in self.set:
                if not random_items:
                    random_items = torch.LongTensor(batch_size).random_(0, self.nb_items).tolist()
                j = random_items.pop()

            items[idx] = j
        items = torch.LongTensor(items)
        return items


# Next, we generate the negative samples for training.

# In[11]:


sampler = _TestNegSampler(df.values, nb_users, nb_items, 500)  # using 500 negative samples
train_negs = sampler.generate()
train_negs = train_negs.reshape(-1, 500)

sampler = _TestNegSampler(df.values, nb_users, nb_items, 100)  # using 100 negative samples
test_negs = sampler.generate()
test_negs = test_negs.reshape(-1, 100)


# In[12]:


import numpy as np

# generating negative samples for training
train_data_neg = np.zeros((train_negs.shape[0]*train_negs.shape[1],3), dtype=int)
idx = 0
for i in tqdm.tqdm(range(train_negs.shape[0])):
    for j in range(train_negs.shape[1]):
        train_data_neg[idx, 0] = i # user ID
        train_data_neg[idx, 1] = train_negs[i, j] # negative item ID
        idx += 1
    
# generating negative samples for testing
test_data_neg = np.zeros((test_negs.shape[0]*test_negs.shape[1],3), dtype=int)
idx = 0
for i in tqdm.tqdm(range(test_negs.shape[0])):
    for j in range(test_negs.shape[1]):
        test_data_neg[idx, 0] = i
        test_data_neg[idx, 1] = test_negs[i, j]
        idx += 1


# In[13]:


train_data_np= np.concatenate([train_data_neg, train_data.values])
np.random.shuffle(train_data_np)

test_data_np= np.concatenate([test_data_neg, test_data.values])
np.random.shuffle(test_data_np)


# In[14]:


# HugeCTR expect user ID and item ID to be different, so we use 0 -> nb_users for user IDs and
# nb_users -> nb_users+nb_items for item IDs.
train_data_np[:,1] += nb_users 
test_data_np[:,1] += nb_users 


# In[15]:


np.max(train_data_np[:,1])


# ### Write HugeCTR data files
# 
# After pre-processing, we write the data to disk using HugeCTR the [Norm](https://nvidia-merlin.github.io/HugeCTR/master/api/python_interface.html#norm) dataset format.

# In[16]:


from ctypes import c_longlong as ll
from ctypes import c_uint
from ctypes import c_float
from ctypes import c_int

def write_hugeCTR_data(huge_ctr_data, filename='huge_ctr_data.dat'):
    print("Writing %d samples"%huge_ctr_data.shape[0])
    with open(filename, 'wb') as f:
        #write header
        f.write(ll(0)) # 0: no error check; 1: check_num
        f.write(ll(huge_ctr_data.shape[0])) # the number of samples in this data file
        f.write(ll(1)) # dimension of label
        f.write(ll(1)) # dimension of dense feature
        f.write(ll(2)) # long long slot_num
        for _ in range(3): f.write(ll(0)) # reserved for future use

        for i in tqdm.tqdm(range(huge_ctr_data.shape[0])):
            f.write(c_float(huge_ctr_data[i,2])) # float label[label_dim];
            f.write(c_float(0)) # dummy dense feature
            f.write(c_int(1)) # slot 1 nnz: user ID
            f.write(c_uint(huge_ctr_data[i,0]))
            f.write(c_int(1)) # slot 2 nnz: item ID
            f.write(c_uint(huge_ctr_data[i,1]))


# #### Train data

# In[17]:


def generate_filelist(filelist_name, num_files, filename_prefix):
    with open(filelist_name, 'wt') as f:
        f.write('{0}\n'.format(num_files));
        for i in range(num_files):
            f.write('{0}_{1}.dat\n'.format(filename_prefix, i))


# In[18]:


get_ipython().system('rm -rf ./data/hugeCTR')
get_ipython().system('mkdir ./data/hugeCTR')

for i, data_arr in enumerate(np.array_split(train_data_np,10)):
    write_hugeCTR_data(data_arr, filename='./data/hugeCTR/train_huge_ctr_data_%d.dat'%i)

generate_filelist('./data/hugeCTR/train_filelist.txt', 10, './data/hugeCTR/train_huge_ctr_data')


# #### Test data

# In[19]:


for i, data_arr in enumerate(np.array_split(test_data_np,10)):
    write_hugeCTR_data(data_arr, filename='./data/hugeCTR/test_huge_ctr_data_%d.dat'%i)
    
generate_filelist('./data/hugeCTR/test_filelist.txt', 10, './data/hugeCTR/test_huge_ctr_data')


# ## HugeCTR DLRM training
# 
# In this section, we will train a DLRM network on the augmented movie lens data. First, we write the training Python script.

# In[2]:


get_ipython().run_cell_magic('writefile', 'hugectr_dlrm_movielens.py', 'import hugectr\nfrom mpi4py import MPI\nsolver = hugectr.CreateSolver(max_eval_batches = 1000,\n                              batchsize_eval = 65536,\n                              batchsize = 65536,\n                              lr = 0.1,\n                              warmup_steps = 1000,\n                              decay_start = 10000,\n                              decay_steps = 40000,\n                              decay_power = 2.0,\n                              end_lr = 1e-5,\n                              vvgpu = [[0]],\n                              repeat_dataset = True,\n                              use_mixed_precision = True,\n                              scaler = 1024)\nreader = hugectr.DataReaderParams(data_reader_type = hugectr.DataReaderType_t.Norm,\n                                  source = ["./data/hugeCTR/train_filelist.txt"],\n                                  eval_source = "./data/hugeCTR/test_filelist.txt",\n                                  check_type = hugectr.Check_t.Non)\noptimizer = hugectr.CreateOptimizer(optimizer_type = hugectr.Optimizer_t.SGD,\n                                    update_type = hugectr.Update_t.Local,\n                                    atomic_update = True)\nmodel = hugectr.Model(solver, reader, optimizer)\nmodel.add(hugectr.Input(label_dim = 1, label_name = "label",\n                        dense_dim = 1, dense_name = "dense",\n                        data_reader_sparse_param_array = \n                        [hugectr.DataReaderSparseParam("data1", 1, True, 2)]))\nmodel.add(hugectr.SparseEmbedding(embedding_type = hugectr.Embedding_t.LocalizedSlotSparseEmbeddingHash, \n                            workspace_size_per_gpu_in_mb = 41,\n                            embedding_vec_size = 64,\n                            combiner = "sum",\n                            sparse_embedding_name = "sparse_embedding1",\n                            bottom_name = "data1",\n                            optimizer = optimizer))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.FusedInnerProduct,\n                            bottom_names = ["dense"],\n                            top_names = ["fc1"],\n                            num_output=64))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.FusedInnerProduct,\n                            bottom_names = ["fc1"],\n                            top_names = ["fc2"],\n                            num_output=128))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.FusedInnerProduct,\n                            bottom_names = ["fc2"],\n                            top_names = ["fc3"],\n                            num_output=64))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Interaction,\n                            bottom_names = ["fc3","sparse_embedding1"],\n                            top_names = ["interaction1"]))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.FusedInnerProduct,\n                            bottom_names = ["interaction1"],\n                            top_names = ["fc4"],\n                            num_output=1024))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.FusedInnerProduct,\n                            bottom_names = ["fc4"],\n                            top_names = ["fc5"],\n                            num_output=1024))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.FusedInnerProduct,\n                            bottom_names = ["fc5"],\n                            top_names = ["fc6"],\n                            num_output=512))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.FusedInnerProduct,\n                            bottom_names = ["fc6"],\n                            top_names = ["fc7"],\n                            num_output=256))                                                  \nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,\n                            bottom_names = ["fc7"],\n                            top_names = ["fc8"],\n                            num_output=1))                                                                                           \nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.BinaryCrossEntropyLoss,\n                            bottom_names = ["fc8", "label"],\n                            top_names = ["loss"]))\nmodel.compile()\nmodel.summary()\nmodel.fit(max_iter = 50000, display = 1000, eval_interval = 3000, snapshot = 3000, snapshot_prefix = "./hugeCTR_saved_model_DLRM/")\n')


# In[21]:


get_ipython().system('rm -rf ./hugeCTR_saved_model_DLRM/')
get_ipython().system('mkdir ./hugeCTR_saved_model_DLRM/')


# In[3]:


get_ipython().system('CUDA_VISIBLE_DEVICES=0 python3 hugectr_dlrm_movielens.py')


# ## Answer item similarity with DLRM embedding
# 
# In this section, we demonstrate how the output of HugeCTR training can be used to carry out simple inference tasks. Specifically, we will show that the movie embeddings can be used for simple item-to-item similarity queries. Such a simple inference can be used as an efficient candidate generator to generate a small set of candidates prior to deep learning model re-ranking. 
# 
# First, we read the embedding tables and extract the movie embeddings.

# In[17]:


import struct 
import pickle
import numpy as np

key_type = 'I64'
key_type_map = {"I32": ["I", 4], "I64": ["q", 8]}

embedding_vec_size = 64

HUGE_CTR_VERSION = 2.21 # set HugeCTR version here, 2.2 for v2.2, 2.21 for v2.21

if HUGE_CTR_VERSION <= 2.2:
    each_key_size = key_type_map[key_type][1] + key_type_map[key_type][1] + 4 * embedding_vec_size
else:
    each_key_size = key_type_map[key_type][1] + 8 + 4 * embedding_vec_size


# In[18]:


embedding_table = {}
        
with open("./hugeCTR_saved_model_DLRM/0_sparse_9000.model" + "/key", 'rb') as key_file, \
     open("./hugeCTR_saved_model_DLRM/0_sparse_9000.model" + "/emb_vector", 'rb') as vec_file:
    try:
        while True:
            key_buffer = key_file.read(key_type_map[key_type][1])
            vec_buffer = vec_file.read(4 * embedding_vec_size)
            if len(key_buffer) == 0 or len(vec_buffer) == 0:
                break
            key = struct.unpack(key_type_map[key_type][0], key_buffer)[0]
            values = struct.unpack(str(embedding_vec_size) + "f", vec_buffer)

            embedding_table[key] = values

    except BaseException as error:
        print(error)


# In[20]:


item_embedding = np.zeros((26744, embedding_vec_size), dtype='float')
for i in range(len(embedding_table[1])):
    item_embedding[i] = embedding_table[1][i]
    


# ### Answer nearest neighbor queries
# 

# In[21]:


from scipy.spatial.distance import cdist

def find_similar_movies(nn_movie_id, item_embedding, k=10, metric="euclidean"):
    #find the top K similar items according to one of the distance metric: cosine or euclidean
    sim = 1-cdist(item_embedding, item_embedding[nn_movie_id].reshape(1, -1), metric=metric)
   
    return sim.squeeze().argsort()[-k:][::-1]


# In[22]:


with open('./mappings.pickle', 'rb') as handle:
    movies_mapping = pickle.load(handle)["items"]

nn_to_movies = movies_mapping
movies_to_nn = {}
for i in range(len(movies_mapping)):
    movies_to_nn[movies_mapping[i]] = i

import pandas as pd
movies = pd.read_csv("./data/ml-20m/movies.csv", index_col="movieId")


# In[23]:


for movie_ID in range(1,10):
    try:
        print("Query: ", movies.loc[movie_ID]["title"], movies.loc[movie_ID]["genres"])

        print("Similar movies: ")
        similar_movies = find_similar_movies(movies_to_nn[movie_ID], item_embedding)

        for i in similar_movies:
            print(nn_to_movies[i], movies.loc[nn_to_movies[i]]["title"], movies.loc[nn_to_movies[i]]["genres"])
        print("=================================\n")
    except Exception as e:
        pass

