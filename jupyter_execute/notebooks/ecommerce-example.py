#!/usr/bin/env python
# coding: utf-8

# <img src="http://developer.download.nvidia.com/compute/machine-learning/frameworks/nvidia_logo.png" style="width: 90px; float: right;">
# 
# # Merlin ETL, training, and inference with e-Commerce behavior data

# ## Overview
# 
# In this tutorial, we use the [eCommerce behavior data from multi category store](https://www.kaggle.com/mkechinov/ecommerce-behavior-data-from-multi-category-store) from [REES46 Marketing Platform](https://rees46.com/) as our dataset. This tutorial is built upon the NVIDIA RecSys 2020 [tutorial](https://recsys.acm.org/recsys20/tutorials/). 
# 
# This notebook provides the code to preprocess the dataset and generate the training, validation, and test sets for the remainder of the tutorial. We define our own goal and filter the dataset accordingly.
# 
# For our tutorial, we decided that our goal is to predict if a user purchased an item:
# 
# -  Positive: User purchased an item.
# -  Negative: User added an item to the cart, but did not purchase it (in the same session).
# 
# We split the dataset into training, validation, and test set by the timestamp:
# 
# - Training: October 2019 - February 2020
# - Validation: March 2020
# - Test: April 2020
# 
# We remove AddToCart Events from a session, if in the same session the same item was purchased.

# ## Data
# 
# First, we download and unzip the raw data.
# 
# Note: the dataset is approximately 11 GB and can require several minutes to download.

# In[ ]:


get_ipython().run_cell_magic('bash', '', 'export HOME=$PWD\npip install gdown --user\n~/.local/bin/gdown  https://drive.google.com/uc?id=1-Rov9fFtGJqb7_ePc6qH-Rhzxn0cIcKB\n~/.local/bin/gdown  https://drive.google.com/uc?id=1-Rov9fFtGJqb7_ePc6qH-Rhzxn0cIcKB\n~/.local/bin/gdown  https://drive.google.com/uc?id=1zr_RXpGvOWN2PrWI6itWL8HnRsCpyqz8\n~/.local/bin/gdown  https://drive.google.com/uc?id=1g5WoIgLe05UMdREbxAjh0bEFgVCjA1UL\n~/.local/bin/gdown  https://drive.google.com/uc?id=1qZIwMbMgMmgDC5EoMdJ8aI9lQPsWA3-P\n~/.local/bin/gdown  https://drive.google.com/uc?id=1x5ohrrZNhWQN4Q-zww0RmXOwctKHH9PT\n')


# In[2]:


import glob  

list_files = glob.glob('*.csv.gz')
list_files


# ### Data extraction and initial preprocessing
# 
# We extract a few relevant columns from the raw datasets and parse date columns into several atomic columns (day, month...).

# In[3]:


import pandas as pd
import numpy as np
from tqdm import tqdm

def process_files(file):
    df_tmp = pd.read_csv(file, compression='gzip')
    df_tmp['session_purchase'] =  df_tmp['user_session'] + '_' + df_tmp['product_id'].astype(str)
    df_purchase = df_tmp[df_tmp['event_type']=='purchase']
    df_cart = df_tmp[df_tmp['event_type']=='cart']
    df_purchase = df_purchase[df_purchase['session_purchase'].isin(df_cart['session_purchase'])]
    df_cart = df_cart[~(df_cart['session_purchase'].isin(df_purchase['session_purchase']))]
    df_cart['target'] = 0
    df_purchase['target'] = 1
    df = pd.concat([df_cart, df_purchase])
    df = df.drop('category_id', axis=1)
    df = df.drop('session_purchase', axis=1)
    df[['cat_0', 'cat_1', 'cat_2', 'cat_3']] = df['category_code'].str.split("\.", n = 3, expand = True).fillna('NA')
    df['brand'] = df['brand'].fillna('NA')
    df = df.drop('category_code', axis=1)
    df['timestamp'] = pd.to_datetime(df['event_time'].str.replace(' UTC', ''))
    df['ts_hour'] = df['timestamp'].dt.hour
    df['ts_minute'] = df['timestamp'].dt.minute
    df['ts_weekday'] = df['timestamp'].dt.weekday
    df['ts_day'] = df['timestamp'].dt.day
    df['ts_month'] = df['timestamp'].dt.month
    df['ts_year'] = df['timestamp'].dt.year
    df.to_csv('./dataset/' + file.replace('.gz', ''), index=False)
    
get_ipython().system('mkdir ./dataset')
for file in tqdm(list_files):
    print(file)
    process_files(file)


# ### Prepare the training, validation, and test datasets
# 
# Next, we split the data into training, validation, and test sets. We use 3 months for training, 1 month for validation, and 1 month for testing.

# In[4]:


lp = []
list_files = glob.glob('./dataset/*.csv')


# In[5]:


get_ipython().system('ls -l ./dataset/*.csv')


# In[6]:


for file in list_files:
    lp.append(pd.read_csv(file))


# In[7]:


df = pd.concat(lp)
df.shape


# In[8]:


df_test = df[df['ts_month']==4]
df_valid = df[df['ts_month']==3]
df_train = df[(df['ts_month']!=3)&(df['ts_month']!=4)]


# In[9]:


df_train.shape, df_valid.shape, df_test.shape


# In[10]:


get_ipython().system('mkdir -p ./data')
df_train.to_parquet('./data/train.parquet', index=False)
df_valid.to_parquet('./data/valid.parquet', index=False)
df_test.to_parquet('./data/test.parquet', index=False)


# In[11]:


df_train.head()


# ## Preprocessing with NVTabular
# 
# Next, we will use NVTabular for preprocessing and engineering more features. 
# 
# But first, we need to import the necessary libraries and initialize a Dask GPU cluster for computation.
# 
# ### Initialize Dask GPU cluster

# In[12]:


# Standard Libraries
import os
from time import time
import re
import shutil
import glob
import warnings

# External Dependencies
import numpy as np
import pandas as pd
import cupy as cp
import cudf
import dask_cudf
from dask_cuda import LocalCUDACluster
from dask.distributed import Client
from dask.utils import parse_bytes
from dask.delayed import delayed
import rmm

# NVTabular
import nvtabular as nvt
import nvtabular.ops as ops
from nvtabular.io import Shuffle
from nvtabular.utils import _pynvml_mem_size, device_mem_size

print(nvt.__version__)


# In[13]:


# define some information about where to get our data
BASE_DIR = "./nvtabular_temp"
get_ipython().system('rm -r $BASE_DIR && mkdir $BASE_DIR')
input_path = './dataset'
dask_workdir = os.path.join(BASE_DIR, "workdir")
output_path = os.path.join(BASE_DIR, "output")
stats_path = os.path.join(BASE_DIR, "stats")


# This example was tested on a DGX server with 8 GPUs. If you have less GPUs, modify the `NUM_GPUS` variable accordingly.

# In[14]:


NUM_GPUS = [0,1,2,3,4,5,6,7]
#NUM_GPUS = [0]

# Dask dashboard
dashboard_port = "8787"

# Deploy a Single-Machine Multi-GPU Cluster
protocol = "tcp"             # "tcp" or "ucx"
visible_devices = ",".join([str(n) for n in NUM_GPUS])  # Delect devices to place workers
device_limit_frac = 0.5      # Spill GPU-Worker memory to host at this limit.
device_pool_frac = 0.6
part_mem_frac = 0.05

# Use total device size to calculate args.device_limit_frac
device_size = device_mem_size(kind="total")
device_limit = int(device_limit_frac * device_size)
device_pool_size = int(device_pool_frac * device_size)
part_size = int(part_mem_frac * device_size)

# Check if any device memory is already occupied
"""
for dev in visible_devices.split(","):
    fmem = _pynvml_mem_size(kind="free", index=int(dev))
    used = (device_size - fmem) / 1e9
    if used > 1.0:
        warnings.warn(f"BEWARE - {used} GB is already occupied on device {int(dev)}!")
"""

cluster = None               # (Optional) Specify existing scheduler port
if cluster is None:
    cluster = LocalCUDACluster(
        protocol = protocol,
        n_workers=len(visible_devices.split(",")),
        CUDA_VISIBLE_DEVICES = visible_devices,
        device_memory_limit = device_limit,
        local_directory=dask_workdir,
        dashboard_address=":" + dashboard_port,
    )

# Create the distributed client
client = Client(cluster)
client


# In[15]:


get_ipython().system('nvidia-smi')


# In[16]:


# Initialize RMM pool on ALL workers
def _rmm_pool():
    rmm.reinitialize(
        # RMM may require the pool size to be a multiple of 256.
        pool_allocator=True,
        initial_pool_size=(device_pool_size // 256) * 256, # Use default size
    )
    
client.run(_rmm_pool)


# ### Define NVTabular dataset

# In[17]:


train_paths = glob.glob('./data/train.parquet')
valid_paths = glob.glob('./data/valid.parquet')
test_paths = glob.glob('./data/test.parquet')

train_dataset = nvt.Dataset(train_paths, engine='parquet', part_mem_fraction=0.15)
valid_dataset = nvt.Dataset(valid_paths, engine='parquet', part_mem_fraction=0.15)
test_dataset = nvt.Dataset(test_paths, engine='parquet', part_mem_fraction=0.15)


# In[18]:


train_dataset.to_ddf().head()


# In[19]:


len(train_dataset.to_ddf().columns)


# In[20]:


train_dataset.to_ddf().columns


# In[21]:


len(train_dataset.to_ddf())


# ### Preprocessing and feature engineering
# 
# In this notebook we will explore a few feature engineering technique with NVTabular:
# 
# - Creating cross features, e.g. `user_id` and `'brand`
# - Target encoding
# 
# The engineered features will then be preprocessed into a form suitable for machine learning model:
# 
# - Fill missing values
# - Encoding categorical features into integer values
# - Normalization of numeric features

# In[22]:


from nvtabular.ops import LambdaOp

# cross features
def user_id_cross_maker(col, gdf):
    return col.astype(str) + '_' + gdf['user_id'].astype(str)

user_id_cross_features = (
    nvt.ColumnGroup(['product_id', 'brand', 'ts_hour', 'ts_minute']) >>
    LambdaOp(user_id_cross_maker, dependency=['user_id']) >> 
    nvt.ops.Rename(postfix = '_user_id_cross')
)


def user_id_brand_cross_maker(col, gdf):
    return col.astype(str) + '_' + gdf['user_id'].astype(str) + '_' + gdf['brand'].astype(str)

user_id_brand_cross_features = (
    nvt.ColumnGroup(['ts_hour', 'ts_weekday', 'cat_0', 'cat_1', 'cat_2']) >>
    LambdaOp(user_id_brand_cross_maker, dependency=['user_id', 'brand']) >> 
    nvt.ops.Rename(postfix = '_user_id_brand_cross')
)

target_encode = (
    ['brand', 'user_id', 'product_id', 'cat_2', ['ts_weekday', 'ts_day']] >>
    nvt.ops.TargetEncoding(
        nvt.ColumnGroup('target'),
        kfold=5,
        p_smooth=20,
        out_dtype="float32",
        )
)

cat_feats = (user_id_brand_cross_features + user_id_cross_features) >> nvt.ops.Categorify()
cont_feats =  ['price', 'ts_weekday', 'ts_day', 'ts_month'] >> nvt.ops.FillMissing() >>  nvt.ops.Normalize()
cont_feats += target_encode >> nvt.ops.Rename(postfix = '_TE')


# In[23]:


output = cat_feats + cont_feats + 'target'
proc = nvt.Workflow(output)


# ### Visualize workflow as a DAG
# 

# In[ ]:


get_ipython().system('apt install -y graphviz')


# In[25]:


output.graph


# ### Executing the workflow
# 
# After having defined the workflow, calling the `fit()` method will start the actual computation to record the required statistics from the training data.

# In[26]:


get_ipython().run_cell_magic('time', '', 'time_preproc_start = time()\nproc.fit(train_dataset)\ntime_preproc = time()-time_preproc_start\n')


# In[27]:


cat_feats.output_columns.names


# In[28]:


output.output_columns.names


# In[29]:


CAT_FEATS = ['ts_hour_user_id_brand_cross',
 'ts_weekday_user_id_brand_cross',
 'cat_0_user_id_brand_cross',
 'cat_1_user_id_brand_cross',
 'cat_2_user_id_brand_cross',
 'product_id_user_id_cross',
 'brand_user_id_cross',
 'ts_hour_user_id_cross',
 'ts_minute_user_id_cross',]

CON_FEATS = ['price',
 'ts_weekday',
 'ts_day',
 'ts_month',
 'TE_brand_target_TE',
 'TE_user_id_target_TE',
 'TE_product_id_target_TE',
 'TE_cat_2_target_TE',
 'TE_ts_weekday_ts_day_target_TE']

dict_dtypes = {}
for col in CAT_FEATS:
    dict_dtypes[col] = np.int64
for col in CON_FEATS:
    dict_dtypes[col] = np.float32

dict_dtypes['target'] = np.float32


# Next, we call the `transform()` method to transform the datasets.

# In[30]:


output_train_dir = os.path.join(output_path, 'train/')
output_valid_dir = os.path.join(output_path, 'valid/')
output_test_dir = os.path.join(output_path, 'test/')
get_ipython().system(' rm -rf $output_train_dir && mkdir -p $output_train_dir')
get_ipython().system(' rm -rf $output_valid_dir && mkdir -p $output_valid_dir')
get_ipython().system(' rm -rf $output_test_dir && mkdir -p $output_test_dir')


# In[31]:


get_ipython().run_cell_magic('time', '', "\ntime_preproc_start = time()\nproc.transform(train_dataset).to_parquet(output_path=output_train_dir, dtypes=dict_dtypes,\n                                         shuffle=nvt.io.Shuffle.PER_PARTITION,\n                                         cats=CAT_FEATS,\n                                         conts=CON_FEATS,\n                                         labels=['target'])\ntime_preproc += time()-time_preproc_start\n")


# In[32]:


get_ipython().system('ls -l $output_train_dir')


# In[33]:


get_ipython().run_cell_magic('time', '', "\ntime_preproc_start = time()\nproc.transform(valid_dataset).to_parquet(output_path=output_valid_dir, dtypes=dict_dtypes,\n                                         shuffle=nvt.io.Shuffle.PER_PARTITION,\n                                         cats=CAT_FEATS,\n                                         conts=CON_FEATS,\n                                         labels=['target'])\ntime_preproc += time()-time_preproc_start\n")


# In[34]:


get_ipython().system('ls -l $output_valid_dir')


# In[35]:


get_ipython().run_cell_magic('time', '', "\ntime_preproc_start = time()\nproc.transform(test_dataset).to_parquet(output_path=output_test_dir, dtypes=dict_dtypes,\n                                         shuffle=nvt.io.Shuffle.PER_PARTITION,\n                                         cats=CAT_FEATS,\n                                         conts=CON_FEATS,\n                                         labels=['target'])\ntime_preproc += time()-time_preproc_start\n")


# In[36]:


time_preproc


# ### Verify the preprocessed data
# 
# Let's quickly read the data back and verify that all fields have the expected format.

# In[37]:


get_ipython().system('ls $output_train_dir')


# In[38]:


nvtdata = pd.read_parquet(output_train_dir+'/part_0.parquet')
nvtdata.head()


# In[39]:


get_ipython().system('ls $output_valid_dir')


# In[40]:


nvtdata_valid = pd.read_parquet(output_valid_dir+'/part_0.parquet')
nvtdata_valid.head()


# In[41]:


sum(nvtdata_valid['ts_hour_user_id_brand_cross']==0)


# In[42]:


len(nvtdata_valid)


# ### Getting the embedding size
# 
# Next, we need to get the embedding size for the categorical variables. This is an important input for defining the embedding table size to be used by HugeCTR.

# In[43]:


embeddings = ops.get_embedding_sizes(proc)
embeddings


# In[44]:


print([embeddings[x][0] for x in cat_feats.output_columns.names])


# In[45]:


cat_feats.output_columns.names


# In[46]:


embedding_size_str = "{}".format([embeddings[x][0] for x in cat_feats.output_columns.names])
embedding_size_str


# In[47]:


num_con_feates = len(CON_FEATS)
num_con_feates


# In[48]:


print([embeddings[x][0] for x in cat_feats.output_columns.names])


# Next, we'll shutdown our Dask client from earlier to free up some memory so that we can share it with HugeCTR.

# In[49]:


client.shutdown()
cluster.close()


# ### Preparing the training Python script for HugeCTR
# 
# The HugeCTR model can be defined by Python API. The following Python script defines a DLRM model and specifies the training resources. 
# 
# Several parameters that need to be edited to match this dataset are:
# 
# - `slot_size_array`: cardinalities for the categorical variables
# - `dense_dim`: number of dense features
# - `slot_num`: number of categorical variables
# 
# The model graph can be saved into a JSON file by calling `model.graph_to_json`, which will be used for inference afterwards.
# 
# In the following code, we train the network using 8 GPUs and a workspace of 4000 MB per GPU. Note that the total embedding size is `33653306*128*4/(1024**3)` = 16.432 GB.

# In[1]:


get_ipython().run_cell_magic('writefile', 'hugectr_dlrm_ecommerce.py', 'import hugectr\nfrom mpi4py import MPI\nsolver = hugectr.CreateSolver(max_eval_batches = 2720,\n                              batchsize_eval = 16384,\n                              batchsize = 16384,\n                              lr = 0.1,\n                              warmup_steps = 8000,\n                              decay_start = 48000,\n                              decay_steps = 24000,\n                              vvgpu = [[0,1,2,3,4,5,6,7]],\n                              repeat_dataset = True,\n                              i64_input_key = True)\nreader = hugectr.DataReaderParams(data_reader_type = hugectr.DataReaderType_t.Parquet,\n                                  source = ["./nvtabular_temp/output/train/_file_list.txt"],\n                                  eval_source = "./nvtabular_temp/output/valid/_file_list.txt",\n                                  check_type = hugectr.Check_t.Non,\n                                  slot_size_array = [4427037, 3961156, 2877223, 2890639, 2159304, 4398425, 3009092, 3999369, 5931061])\noptimizer = hugectr.CreateOptimizer(optimizer_type = hugectr.Optimizer_t.SGD,\n                                    update_type = hugectr.Update_t.Local,\n                                    atomic_update = True)\nmodel = hugectr.Model(solver, reader, optimizer)\nmodel.add(hugectr.Input(label_dim = 1, label_name = "label",\n                        dense_dim = 9, dense_name = "dense",\n                        data_reader_sparse_param_array = \n                        [hugectr.DataReaderSparseParam("data1", 1, True, 9)]))\nmodel.add(hugectr.SparseEmbedding(embedding_type = hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash,\n                            workspace_size_per_gpu_in_mb = 4000,\n                            embedding_vec_size = 128,\n                            combiner = \'sum\',\n                            sparse_embedding_name = "sparse_embedding1",\n                            bottom_name = "data1",\n                            optimizer = optimizer))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,\n                            bottom_names = ["dense"],\n                            top_names = ["fc1"],\n                            num_output=512))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,\n                            bottom_names = ["fc1"],\n                            top_names = ["relu1"]))                           \nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,\n                            bottom_names = ["relu1"],\n                            top_names = ["fc2"],\n                            num_output=256))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,\n                            bottom_names = ["fc2"],\n                            top_names = ["relu2"]))                            \nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,\n                            bottom_names = ["relu2"],\n                            top_names = ["fc3"],\n                            num_output=128))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,\n                            bottom_names = ["fc3"],\n                            top_names = ["relu3"]))                              \nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Interaction,\n                            bottom_names = ["relu3","sparse_embedding1"],\n                            top_names = ["interaction1"]))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,\n                            bottom_names = ["interaction1"],\n                            top_names = ["fc4"],\n                            num_output=1024))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,\n                            bottom_names = ["fc4"],\n                            top_names = ["relu4"]))                              \nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,\n                            bottom_names = ["relu4"],\n                            top_names = ["fc5"],\n                            num_output=1024))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,\n                            bottom_names = ["fc5"],\n                            top_names = ["relu5"]))                              \nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,\n                            bottom_names = ["relu5"],\n                            top_names = ["fc6"],\n                            num_output=512))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,\n                            bottom_names = ["fc6"],\n                            top_names = ["relu6"]))                               \nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,\n                            bottom_names = ["relu6"],\n                            top_names = ["fc7"],\n                            num_output=256))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,\n                            bottom_names = ["fc7"],\n                            top_names = ["relu7"]))                                                                              \nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,\n                            bottom_names = ["relu7"],\n                            top_names = ["fc8"],\n                            num_output=1))                                                                                           \nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.BinaryCrossEntropyLoss,\n                            bottom_names = ["fc8", "label"],\n                            top_names = ["loss"]))\nmodel.compile()\nmodel.summary()\nmodel.graph_to_json(graph_config_file = "dlrm_ecommerce.json")\nmodel.fit(max_iter = 12000, display = 1000, eval_interval = 3000, snapshot = 10000, snapshot_prefix = "./")\n')


# ## HugeCTR training
# 
# Now we are ready to train a DLRM model with HugeCTR.
# 
# 

# In[2]:


get_ipython().system('python3 hugectr_dlrm_ecommerce.py')


# ## HugeCTR inference
# 
# In this section, we read the test dataset and compute the AUC value. 
# 
# We will utilize the saved model graph in JSON format for inference.

# ### Prepare the inference session

# In[3]:


import sys
from hugectr.inference import InferenceParams, CreateInferenceSession
from mpi4py import MPI


# In[4]:


# create inference session
inference_params = InferenceParams(model_name = "dlrm",
                              max_batchsize = 4096,
                              hit_rate_threshold = 0.6,
                              dense_model_file = "./_dense_10000.model",
                              sparse_model_files = ["./0_sparse_10000.model"],
                              device_id = 0,
                              use_gpu_embedding_cache = True,
                              cache_size_percentage = 0.2,
                              i64_input_key = True)
inference_session = CreateInferenceSession("dlrm_ecommerce.json", inference_params)


# ### Reading and preparing the data
# 
# First, we read the NVTabular processed data.

# In[5]:


import pandas as pd

nvtdata_test = pd.read_parquet('./nvtabular_temp/output/test/part_0.parquet')
nvtdata_test.head()


# In[6]:


con_feats = ['price',
 'ts_weekday',
 'ts_day',
 'ts_month',
 'TE_brand_target_TE',
 'TE_user_id_target_TE',
 'TE_product_id_target_TE',
 'TE_cat_2_target_TE',
 'TE_ts_weekday_ts_day_target_TE']


# In[7]:


cat_feats = ['ts_hour_user_id_brand_cross',
 'ts_weekday_user_id_brand_cross',
 'cat_0_user_id_brand_cross',
 'cat_1_user_id_brand_cross',
 'cat_2_user_id_brand_cross',
 'product_id_user_id_cross',
 'brand_user_id_cross',
 'ts_hour_user_id_cross',
 'ts_minute_user_id_cross']


# In[8]:


emb_size = [4427037, 3961156, 2877223, 2890639, 2159304, 4398425, 3009092, 3999369, 5931061]


# ### Converting data to CSR format
# 
# HugeCTR expects data in CSR format for inference. One important thing to note is that NVTabular requires categorical variables to occupy different integer ranges. For example, if there are 10 users and 10 items, then the users should be encoded in the 0-9 range, while items should be in the 10-19 range. NVTabular encodes both users and items in the 0-9 ranges.
# 
# For this reason, we need to shift the keys of the categorical variable produced by NVTabular to comply with HugeCTR.

# In[9]:


import numpy as np
shift = np.insert(np.cumsum(emb_size), 0, 0)[:-1]


# In[10]:


cat_data = nvtdata_test[cat_feats].values + shift


# In[11]:


dense_data = nvtdata_test[con_feats].values


# In[12]:


def infer_batch(inference_session, dense_data_batch, cat_data_batch):
    dense_features = list(dense_data_batch.flatten())
    embedding_columns = list(cat_data_batch.flatten())
    row_ptrs= list(range(0,len(embedding_columns)+1))
    output = inference_session.predict(dense_features, embedding_columns, row_ptrs)
    return output


# Now we are ready to carry out inference on the test set.

# In[13]:


batch_size = 4096
num_batches = (len(dense_data) // batch_size) + 1
batch_idx = np.array_split(np.arange(len(dense_data)), num_batches)


# In[14]:


get_ipython().system('pip install tqdm')


# In[ ]:


from tqdm import tqdm

labels = []
for batch_id in tqdm(batch_idx):
    dense_data_batch = dense_data[batch_id]
    cat_data_batch = cat_data[batch_id]
    results = infer_batch(inference_session, dense_data_batch, cat_data_batch)
    labels.extend(results)


# In[16]:


len(labels)


# ### Computing the test AUC value

# In[17]:


ground_truth = nvtdata_test['target'].values


# In[18]:


from sklearn.metrics import roc_auc_score

roc_auc_score(ground_truth, labels)


# ## Conclusion
# 
# In this notebook, we have walked you through the process of preprocessing the data, train a DLRM model with HugeCTR, then carrying out inference with the HugeCTR Python interface. Try this workflow on your data and let us know your feedback.
# 
# 
