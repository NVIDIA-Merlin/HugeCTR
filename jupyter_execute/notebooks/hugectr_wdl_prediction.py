#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# # HugeCTR Wide and Deep Model with Criteo
# 
# ## Overview
# 
# In this notebook, we provide a tutorial that shows how to train a wide and deep model using the high-level Python API from HugeCTR on the original Criteo dataset as training data.
# We show how to produce prediction results based on different types of local database.

# ## Dataset Preprocessing
# 
# ### Generate training and validation data folders

# In[1]:


# define some data folder to store the original and preprocessed data
# Standard Libraries
import os
from time import time
import re
import shutil
import glob
import warnings
BASE_DIR = "/wdl_train"
train_path  = os.path.join(BASE_DIR, "train")
val_path = os.path.join(BASE_DIR, "val")
CUDA_VISIBLE_DEVICES = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
n_workers = len(CUDA_VISIBLE_DEVICES.split(","))
frac_size = 0.15
allow_multi_gpu = False
use_rmm_pool = False
max_day = None  # (Optional) -- Limit the dataset to day 0-max_day for debugging

if os.path.isdir(train_path):
    shutil.rmtree(train_path)
os.makedirs(train_path)

if os.path.isdir(val_path):
    shutil.rmtree(val_path)
os.makedirs(val_path)


# In[4]:


get_ipython().system('ls -l $train_path')


# ### Download the original Criteo dataset

# In[2]:


get_ipython().system('apt-get install wget')


# In[3]:


get_ipython().system('wget -P $train_path https://storage.googleapis.com/criteo-cail-datasets/day_0.gz')


# 
# Split the dataset into training and validation.

# In[2]:


get_ipython().system('gzip -d -c $train_path/day_0.gz > day_0')
get_ipython().system('head -n 45840617 day_0 > $train_path/train.txt')
get_ipython().system('tail -n 2000000 day_0 > $val_path/test.txt')


# ### Preprocessing with NVTabular

# In[5]:


get_ipython().run_cell_magic('writefile', '/wdl_train/preprocess.py', 'import os\nimport sys\nimport argparse\nimport glob\nimport time\nfrom cudf.io.parquet import ParquetWriter\nimport numpy as np\nimport pandas as pd\nimport concurrent.futures as cf\nfrom concurrent.futures import as_completed\nimport shutil\n\nimport dask_cudf\nfrom dask_cuda import LocalCUDACluster\nfrom dask.distributed import Client\nfrom dask.utils import parse_bytes\nfrom dask.delayed import delayed\n\nimport cudf\nimport rmm\nimport nvtabular as nvt\nfrom nvtabular.io import Shuffle\nfrom nvtabular.utils import device_mem_size\nfrom nvtabular.ops import Categorify, Clip, FillMissing, HashBucket, LambdaOp, Normalize, Rename, Operator, get_embedding_sizes\n#%load_ext memory_profiler\n\nimport logging\nlogging.basicConfig(format=\'%(asctime)s %(message)s\')\nlogging.root.setLevel(logging.NOTSET)\nlogging.getLogger(\'numba\').setLevel(logging.WARNING)\nlogging.getLogger(\'asyncio\').setLevel(logging.WARNING)\n\n# define dataset schema\nCATEGORICAL_COLUMNS=["C" + str(x) for x in range(1, 27)]\nCONTINUOUS_COLUMNS=["I" + str(x) for x in range(1, 14)]\nLABEL_COLUMNS = [\'label\']\nCOLUMNS =  LABEL_COLUMNS + CONTINUOUS_COLUMNS +  CATEGORICAL_COLUMNS\n#/samples/criteo mode doesn\'t have dense features\ncriteo_COLUMN=LABEL_COLUMNS +  CATEGORICAL_COLUMNS\n#For new feature cross columns\nCROSS_COLUMNS = []\n\n\nNUM_INTEGER_COLUMNS = 13\nNUM_CATEGORICAL_COLUMNS = 26\nNUM_TOTAL_COLUMNS = 1 + NUM_INTEGER_COLUMNS + NUM_CATEGORICAL_COLUMNS\n\n\n# Initialize RMM pool on ALL workers\ndef setup_rmm_pool(client, pool_size):\n    client.run(rmm.reinitialize, pool_allocator=True, initial_pool_size=pool_size)\n    return None\n\n#compute the partition size with GB\ndef bytesto(bytes, to, bsize=1024):\n    a = {\'k\' : 1, \'m\': 2, \'g\' : 3, \'t\' : 4, \'p\' : 5, \'e\' : 6 }\n    r = float(bytes)\n    return bytes / (bsize ** a[to])\n\nclass FeatureCross(Operator):\n    def __init__(self, dependency):\n        self.dependency = dependency\n\n    def transform(self, columns, gdf):\n        new_df = type(gdf)()\n        for col in columns.names:\n            new_df[col] = gdf[col] + gdf[self.dependency]\n        return new_df\n\n    def dependencies(self):\n        return [self.dependency]\n\n#process the data with NVTabular\ndef process_NVT(args):\n\n    if args.feature_cross_list:\n        feature_pairs = [pair.split("_") for pair in args.feature_cross_list.split(",")]\n        for pair in feature_pairs:\n            CROSS_COLUMNS.append(pair[0]+\'_\'+pair[1])\n\n\n    logging.info(\'NVTabular processing\')\n    train_input = os.path.join(args.data_path, "train/train.txt")\n    val_input = os.path.join(args.data_path, "val/test.txt")\n    PREPROCESS_DIR_temp_train = os.path.join(args.out_path, \'train/temp-parquet-after-conversion\')\n    PREPROCESS_DIR_temp_val = os.path.join(args.out_path, \'val/temp-parquet-after-conversion\')\n    PREPROCESS_DIR_temp = [PREPROCESS_DIR_temp_train, PREPROCESS_DIR_temp_val]\n    train_output = os.path.join(args.out_path, "train")\n    val_output = os.path.join(args.out_path, "val")\n\n    # Make sure we have a clean parquet space for cudf conversion\n    for one_path in PREPROCESS_DIR_temp:\n        if os.path.exists(one_path):\n            shutil.rmtree(one_path)\n        os.mkdir(one_path)\n\n\n    ## Get Dask Client\n\n    # Deploy a Single-Machine Multi-GPU Cluster\n    device_size = device_mem_size(kind="total")\n    cluster = None\n    if args.protocol == "ucx":\n        UCX_TLS = os.environ.get("UCX_TLS", "tcp,cuda_copy,cuda_ipc,sockcm")\n        os.environ["UCX_TLS"] = UCX_TLS\n        cluster = LocalCUDACluster(\n            protocol = args.protocol,\n            CUDA_VISIBLE_DEVICES = args.devices,\n            n_workers = len(args.devices.split(",")),\n            enable_nvlink=True,\n            device_memory_limit = int(device_size * args.device_limit_frac),\n            dashboard_address=":" + args.dashboard_port\n        )\n    else:\n        cluster = LocalCUDACluster(\n            protocol = args.protocol,\n            n_workers = len(args.devices.split(",")),\n            CUDA_VISIBLE_DEVICES = args.devices,\n            device_memory_limit = int(device_size * args.device_limit_frac),\n            dashboard_address=":" + args.dashboard_port\n        )\n\n\n\n    # Create the distributed client\n    client = Client(cluster)\n    if args.device_pool_frac > 0.01:\n        setup_rmm_pool(client, int(args.device_pool_frac*device_size))\n\n\n    #calculate the total processing time\n    runtime = time.time()\n\n    #test dataset without the label feature\n    if args.dataset_type == \'test\':\n        global LABEL_COLUMNS\n        LABEL_COLUMNS = []\n\n    ##-----------------------------------##\n    # Dask rapids converts txt to parquet\n    # Dask cudf dataframe = ddf\n\n    ## train/valid txt to parquet\n    train_valid_paths = [(train_input,PREPROCESS_DIR_temp_train),(val_input,PREPROCESS_DIR_temp_val)]\n\n    for input, temp_output in train_valid_paths:\n\n        ddf = dask_cudf.read_csv(input,sep=\'\\t\',names=LABEL_COLUMNS + CONTINUOUS_COLUMNS + CATEGORICAL_COLUMNS)\n\n        ## Convert label col to FP32\n        if args.parquet_format and args.dataset_type == \'train\':\n            ddf["label"] = ddf[\'label\'].astype(\'float32\')\n\n        # Save it as parquet format for better memory usage\n        ddf.to_parquet(temp_output,header=True)\n        ##-----------------------------------##\n\n    COLUMNS =  LABEL_COLUMNS + CONTINUOUS_COLUMNS + CROSS_COLUMNS + CATEGORICAL_COLUMNS\n    train_paths = glob.glob(os.path.join(PREPROCESS_DIR_temp_train, "*.parquet"))\n    valid_paths = glob.glob(os.path.join(PREPROCESS_DIR_temp_val, "*.parquet"))\n\n    categorify_op = Categorify(freq_threshold=args.freq_limit)\n    cat_features = CATEGORICAL_COLUMNS >> categorify_op\n    cont_features = CONTINUOUS_COLUMNS >> FillMissing() >> Clip(min_value=0) >> Normalize()\n    cross_cat_op = Categorify(freq_threshold=args.freq_limit)\n\n    features = LABEL_COLUMNS\n    \n    if args.criteo_mode == 0:\n        features += cont_features\n        if args.feature_cross_list:\n            feature_pairs = [pair.split("_") for pair in args.feature_cross_list.split(",")]\n            for pair in feature_pairs:\n                col0 = pair[0]\n                col1 = pair[1]\n                features += col0 >> FeatureCross(col1)  >> Rename(postfix="_"+col1) >> cross_cat_op\n            \n    features += cat_features\n\n    workflow = nvt.Workflow(features, client=client)\n\n    logging.info("Preprocessing")\n\n    output_format = \'hugectr\'\n    if args.parquet_format:\n        output_format = \'parquet\'\n\n    # just for /samples/criteo model\n    train_ds_iterator = nvt.Dataset(train_paths, engine=\'parquet\', part_size=int(args.part_mem_frac * device_size))\n    valid_ds_iterator = nvt.Dataset(valid_paths, engine=\'parquet\', part_size=int(args.part_mem_frac * device_size))\n\n    shuffle = None\n    if args.shuffle == "PER_WORKER":\n        shuffle = nvt.io.Shuffle.PER_WORKER\n    elif args.shuffle == "PER_PARTITION":\n        shuffle = nvt.io.Shuffle.PER_PARTITION\n\n    logging.info(\'Train Datasets Preprocessing.....\')\n\n    dict_dtypes = {}\n    for col in CATEGORICAL_COLUMNS:\n        dict_dtypes[col] = np.int64\n    if not args.criteo_mode:\n        for col in CONTINUOUS_COLUMNS:\n            dict_dtypes[col] = np.float32\n    for col in CROSS_COLUMNS:\n        dict_dtypes[col] = np.int64\n    for col in LABEL_COLUMNS:\n        dict_dtypes[col] = np.float32\n    \n    conts = CONTINUOUS_COLUMNS if not args.criteo_mode else []\n    \n    workflow.fit(train_ds_iterator)\n    \n    if output_format == \'hugectr\':\n        workflow.transform(train_ds_iterator).to_hugectr(\n                cats=CATEGORICAL_COLUMNS + CROSS_COLUMNS,\n                conts=conts,\n                labels=LABEL_COLUMNS,\n                output_path=train_output,\n                shuffle=shuffle,\n                out_files_per_proc=args.out_files_per_proc,\n                num_threads=args.num_io_threads)\n    else:\n        workflow.transform(train_ds_iterator).to_parquet(\n                output_path=train_output,\n                dtypes=dict_dtypes,\n                cats=CATEGORICAL_COLUMNS + CROSS_COLUMNS,\n                conts=conts,\n                labels=LABEL_COLUMNS,\n                shuffle=shuffle,\n                out_files_per_proc=args.out_files_per_proc,\n                num_threads=args.num_io_threads)\n        \n        \n        \n    ###Getting slot size###    \n    #--------------------##\n    embeddings_dict_cat = categorify_op.get_embedding_sizes(CATEGORICAL_COLUMNS)\n    embeddings_dict_cross = cross_cat_op.get_embedding_sizes(CROSS_COLUMNS)\n    embeddings = [embeddings_dict_cat[c][0] for c in CATEGORICAL_COLUMNS] + [embeddings_dict_cross[c][0] for c in CROSS_COLUMNS]\n    \n    print(embeddings)\n    ##--------------------##\n\n    logging.info(\'Valid Datasets Preprocessing.....\')\n\n    if output_format == \'hugectr\':\n        workflow.transform(valid_ds_iterator).to_hugectr(\n                cats=CATEGORICAL_COLUMNS + CROSS_COLUMNS,\n                conts=conts,\n                labels=LABEL_COLUMNS,\n                output_path=val_output,\n                shuffle=shuffle,\n                out_files_per_proc=args.out_files_per_proc,\n                num_threads=args.num_io_threads)\n    else:\n        workflow.transform(valid_ds_iterator).to_parquet(\n                output_path=val_output,\n                dtypes=dict_dtypes,\n                cats=CATEGORICAL_COLUMNS + CROSS_COLUMNS,\n                conts=conts,\n                labels=LABEL_COLUMNS,\n                shuffle=shuffle,\n                out_files_per_proc=args.out_files_per_proc,\n                num_threads=args.num_io_threads)\n\n    embeddings_dict_cat = categorify_op.get_embedding_sizes(CATEGORICAL_COLUMNS)\n    embeddings_dict_cross = cross_cat_op.get_embedding_sizes(CROSS_COLUMNS)\n    embeddings = [embeddings_dict_cat[c][0] for c in CATEGORICAL_COLUMNS] + [embeddings_dict_cross[c][0] for c in CROSS_COLUMNS]\n    \n    print(embeddings)\n    ##--------------------##\n\n    ## Shutdown clusters\n    client.close()\n    logging.info(\'NVTabular processing done\')\n\n    runtime = time.time() - runtime\n\n    print("\\nDask-NVTabular Criteo Preprocessing")\n    print("--------------------------------------")\n    print(f"data_path          | {args.data_path}")\n    print(f"output_path        | {args.out_path}")\n    print(f"partition size     | {\'%.2f GB\'%bytesto(int(args.part_mem_frac * device_size),\'g\')}")\n    print(f"protocol           | {args.protocol}")\n    print(f"device(s)          | {args.devices}")\n    print(f"rmm-pool-frac      | {(args.device_pool_frac)}")\n    print(f"out-files-per-proc | {args.out_files_per_proc}")\n    print(f"num_io_threads     | {args.num_io_threads}")\n    print(f"shuffle            | {args.shuffle}")\n    print("======================================")\n    print(f"Runtime[s]         | {runtime}")\n    print("======================================\\n")\n\n\ndef parse_args():\n    parser = argparse.ArgumentParser(description=("Multi-GPU Criteo Preprocessing"))\n\n    #\n    # System Options\n    #\n\n    parser.add_argument("--data_path", type=str, help="Input dataset path (Required)")\n    parser.add_argument("--out_path", type=str, help="Directory path to write output (Required)")\n    parser.add_argument(\n        "-d",\n        "--devices",\n        default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"),\n        type=str,\n        help=\'Comma-separated list of visible devices (e.g. "0,1,2,3"). \'\n    )\n    parser.add_argument(\n        "-p",\n        "--protocol",\n        choices=["tcp", "ucx"],\n        default="tcp",\n        type=str,\n        help="Communication protocol to use (Default \'tcp\')",\n    )\n    parser.add_argument(\n        "--device_limit_frac",\n        default=0.5,\n        type=float,\n        help="Worker device-memory limit as a fraction of GPU capacity (Default 0.8). "\n    )\n    parser.add_argument(\n        "--device_pool_frac",\n        default=0.9,\n        type=float,\n        help="RMM pool size for each worker  as a fraction of GPU capacity (Default 0.9). "\n        "The RMM pool frac is the same for all GPUs, make sure each one has enough memory size",\n    )\n    parser.add_argument(\n        "--num_io_threads",\n        default=0,\n        type=int,\n        help="Number of threads to use when writing output data (Default 0). "\n        "If 0 is specified, multi-threading will not be used for IO.",\n    )\n\n    #\n    # Data-Decomposition Parameters\n    #\n\n    parser.add_argument(\n        "--part_mem_frac",\n        default=0.125,\n        type=float,\n        help="Maximum size desired for dataset partitions as a fraction "\n        "of GPU capacity (Default 0.125)",\n    )\n    parser.add_argument(\n        "--out_files_per_proc",\n        default=1,\n        type=int,\n        help="Number of output files to write on each worker (Default 1)",\n    )\n\n    #\n    # Preprocessing Options\n    #\n\n    parser.add_argument(\n        "-f",\n        "--freq_limit",\n        default=0,\n        type=int,\n        help="Frequency limit for categorical encoding (Default 0)",\n    )\n    parser.add_argument(\n        "-s",\n        "--shuffle",\n        choices=["PER_WORKER", "PER_PARTITION", "NONE"],\n        default="PER_PARTITION",\n        help="Shuffle algorithm to use when writing output data to disk (Default PER_PARTITION)",\n    )\n\n    parser.add_argument(\n        "--feature_cross_list", default=None, type=str, help="List of feature crossing cols (e.g. C1_C2, C3_C4)"\n    )\n\n    #\n    # Diagnostics Options\n    #\n\n    parser.add_argument(\n        "--profile",\n        metavar="PATH",\n        default=None,\n        type=str,\n        help="Specify a file path to export a Dask profile report (E.g. dask-report.html)."\n        "If this option is excluded from the command, not profile will be exported",\n    )\n    parser.add_argument(\n        "--dashboard_port",\n        default="8787",\n        type=str,\n        help="Specify the desired port of Dask\'s diagnostics-dashboard (Default `3787`). "\n        "The dashboard will be hosted at http://<IP>:<PORT>/status",\n    )\n\n    #\n    # Format\n    #\n\n    parser.add_argument(\'--criteo_mode\', type=int, default=0)\n    parser.add_argument(\'--parquet_format\', type=int, default=1)\n    parser.add_argument(\'--dataset_type\', type=str, default=\'train\')\n\n    args = parser.parse_args()\n    args.n_workers = len(args.devices.split(","))\n    return args\nif __name__ == \'__main__\':\n\n    args = parse_args()\n\n    process_NVT(args)\n')


# In[10]:


import pandas as pd


# In[11]:


get_ipython().system("python3 /wdl_train/preprocess.py --data_path wdl_train/  --out_path wdl_train/ --freq_limit 6 --feature_cross_list C1_C2,C3_C4  --device_pool_frac 0.5  --devices '0' --num_io_threads 2")


# #### Check the preprocessed training data

# In[6]:


get_ipython().system('ls -ll /wdl_train/train')


# In[7]:


import pandas as pd
df = pd.read_parquet("/wdl_train/train/0.8870d61b8a1f4deca0f911acfb072999.parquet")
df.head(2)


# ### WDL Model Training

# In[3]:


get_ipython().run_cell_magic('writefile', "'./model.py'", 'import hugectr\n#from mpi4py import MPI\nsolver = hugectr.CreateSolver(max_eval_batches = 4000,\n                              batchsize_eval = 2720,\n                              batchsize = 2720,\n                              lr = 0.001,\n                              vvgpu = [[2]],\n                              repeat_dataset = True,\n                              i64_input_key = True)\n\nreader = hugectr.DataReaderParams(data_reader_type = hugectr.DataReaderType_t.Parquet,\n                                  source = ["./train/_file_list.txt"],\n                                  eval_source = "./val/_file_list.txt",\n                                  check_type = hugectr.Check_t.Non,\n                                  slot_size_array = [249058, 19561, 14212, 6890, 18592, 4, 6356, 1254, 52, 226170, 80508, 72308, 11, 2169, 7597, 61, 4, 923, 15, 249619, 168974, 243480, 68212, 9169, 75, 34, 278018, 415262])\noptimizer = hugectr.CreateOptimizer(optimizer_type = hugectr.Optimizer_t.Adam,\n                                    update_type = hugectr.Update_t.Global,\n                                    beta1 = 0.9,\n                                    beta2 = 0.999,\n                                    epsilon = 0.0000001)\nmodel = hugectr.Model(solver, reader, optimizer)\n\nmodel.add(hugectr.Input(label_dim = 1, label_name = "label",\n                        dense_dim = 13, dense_name = "dense",\n                        data_reader_sparse_param_array = \n                        [hugectr.DataReaderSparseParam("wide_data", 1, True, 2),\n                        hugectr.DataReaderSparseParam("deep_data", 2, False, 26)]))\n\nmodel.add(hugectr.SparseEmbedding(embedding_type = hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash, \n                            workspace_size_per_gpu_in_mb = 24,\n                            embedding_vec_size = 1,\n                            combiner = "sum",\n                            sparse_embedding_name = "sparse_embedding2",\n                            bottom_name = "wide_data",\n                            optimizer = optimizer))\nmodel.add(hugectr.SparseEmbedding(embedding_type = hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash, \n                            workspace_size_per_gpu_in_mb = 405,\n                            embedding_vec_size = 16,\n                            combiner = "sum",\n                            sparse_embedding_name = "sparse_embedding1",\n                            bottom_name = "deep_data",\n                            optimizer = optimizer))\n\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Reshape,\n                            bottom_names = ["sparse_embedding1"],\n                            top_names = ["reshape1"],\n                            leading_dim=416))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Reshape,\n                            bottom_names = ["sparse_embedding2"],\n                            top_names = ["reshape2"],\n                            leading_dim=2))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReduceSum,\n                            bottom_names = ["reshape2"],\n                            top_names = ["wide_redn"],\n                            axis = 1))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Concat,\n                            bottom_names = ["reshape1", "dense"],\n                            top_names = ["concat1"]))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,\n                            bottom_names = ["concat1"],\n                            top_names = ["fc1"],\n                            num_output=1024))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,\n                            bottom_names = ["fc1"],\n                            top_names = ["relu1"]))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Dropout,\n                            bottom_names = ["relu1"],\n                            top_names = ["dropout1"],\n                            dropout_rate=0.5))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,\n                            bottom_names = ["dropout1"],\n                            top_names = ["fc2"],\n                            num_output=1024))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,\n                            bottom_names = ["fc2"],\n                            top_names = ["relu2"]))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Dropout,\n                            bottom_names = ["relu2"],\n                            top_names = ["dropout2"],\n                            dropout_rate=0.5))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,\n                            bottom_names = ["dropout2"],\n                            top_names = ["fc3"],\n                            num_output=1))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Add,\n                            bottom_names = ["fc3", "wide_redn"],\n                            top_names = ["add1"]))\nmodel.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.BinaryCrossEntropyLoss,\n                            bottom_names = ["add1", "label"],\n                            top_names = ["loss"]))\nmodel.compile()\nmodel.summary()\nmodel.fit(max_iter = 21000, display = 1000, eval_interval = 4000, snapshot = 20000, snapshot_prefix = "wdl")\nmodel.graph_to_json(graph_config_file = "wdl.json")\n')


# In[1]:


get_ipython().system('python ./model.py')


# In[9]:


get_ipython().system('ls -ll')


# In[1]:


get_ipython().system('python /wdl_infer/wdl_python_infer.py "wdl" "/wdl_infer/model/wdl/1/wdl.json"  "/wdl_infer/model/wdl/1/wdl_dense_20000.model"  "/wdl_infer/model/wdl/1/wdl0_sparse_20000.model/,/wdl_infer/model/wdl/1/wdl1_sparse_20000.model"  "/wdl_infer/first_ten.csv"')


# ## Prepare Inference Request

# In[3]:


get_ipython().system('ls -l /wdl_train/val')


# In[5]:


import pandas as pd
df = pd.read_parquet("/wdl_train/val/0.110d099942694a5cbf1b71eb73e10f27.parquet")

df.head()


# In[6]:


df.head(10).to_csv('/wdl_train/infer_test.csv', sep=',', index=False,header=True)


# ## Create prediction scripts

# In[7]:


get_ipython().run_cell_magic('writefile', "'/wdl_train/wdl_predict.py'", 'from hugectr.inference import InferenceParams, CreateInferenceSession\nimport hugectr\nimport pandas as pd\nimport numpy as np\nimport sys\nfrom mpi4py import MPI\ndef wdl_inference(model_name, network_file, dense_file, embedding_file_list, data_file,enable_cache,dbtype=hugectr.Database_t.Local,rocksdb_path=""):\n    CATEGORICAL_COLUMNS=["C" + str(x) for x in range(1, 27)]+["C1_C2","C3_C4"]\n    CONTINUOUS_COLUMNS=["I" + str(x) for x in range(1, 14)]\n    LABEL_COLUMNS = [\'label\']\n    emb_size = [249058, 19561, 14212, 6890, 18592, 4, 6356, 1254, 52, 226170, 80508, 72308, 11, 2169, 7597, 61, 4, 923, 15, 249619, 168974, 243480, 68212, 9169, 75, 34, 278018, 415262]\n    shift = np.insert(np.cumsum(emb_size), 0, 0)[:-1]\n    test_df=pd.read_csv(data_file,sep=\',\')\n    config_file = network_file\n    row_ptrs = list(range(0,21))+list(range(0,261))\n    dense_features =  list(test_df[CONTINUOUS_COLUMNS].values.flatten())\n    test_df[CATEGORICAL_COLUMNS].astype(np.int64)\n    embedding_columns = list((test_df[CATEGORICAL_COLUMNS]+shift).values.flatten())\n    \n\n    # create parameter server, embedding cache and inference session\n    inference_params = InferenceParams(model_name = model_name,\n                                max_batchsize = 64,\n                                hit_rate_threshold = 0.5,\n                                dense_model_file = dense_file,\n                                sparse_model_files = embedding_file_list,\n                                device_id = 2,\n                                use_gpu_embedding_cache = enable_cache,\n                                cache_size_percentage = 0.9,\n                                i64_input_key = True,\n                                use_mixed_precision = False,\n                                db_type = dbtype,\n                                rocksdb_path=rocksdb_path,\n                                cache_size_percentage_redis=0.5)\n    inference_session = CreateInferenceSession(config_file, inference_params)\n    output = inference_session.predict(dense_features, embedding_columns, row_ptrs)\n    print("WDL multi-embedding table inference result is {}".format(output))\n\nif __name__ == "__main__":\n    model_name = sys.argv[1]\n    print("{} multi-embedding table prediction".format(model_name))\n    network_file = sys.argv[2]\n    print("{} multi-embedding table prediction network is {}".format(model_name,network_file))\n    dense_file = sys.argv[3]\n    print("{} multi-embedding table prediction dense file is {}".format(model_name,dense_file))\n    embedding_file_list = str(sys.argv[4]).split(\',\')\n    print("{} multi-embedding table prediction sparse files are {}".format(model_name,embedding_file_list))\n    data_file = sys.argv[5]\n    print("{} multi-embedding table prediction input data path is {}".format(model_name,data_file))\n    input_dbtype = sys.argv[6]\n    print("{} multi-embedding table prediction input dbtype path is {}".format(model_name,input_dbtype))\n    if input_dbtype=="local":\n        wdl_inference(model_name, network_file, dense_file, embedding_file_list, data_file, True, hugectr.Database_t.Local)\n    if input_dbtype=="rocksdb":\n        rocksdb_path = sys.argv[7]\n        print("{} multi-embedding table prediction rocksdb_path path is {}".format(model_name,rocksdb_path))\n        wdl_inference(model_name, network_file, dense_file, embedding_file_list, data_file, True, hugectr.Database_t.RocksDB,rocksdb_path)\n')


# ## Prediction
# 
# Use different types of databases as a local parameter server to get the wide and deep model prediction results.

# ### Load model embedding tables into local memory as parameter server

# In[2]:


get_ipython().system('python /wdl_train/wdl_predict.py "wdl" "/wdl_infer/model/wdl/1/wdl.json" "/wdl_infer/model/wdl/1/wdl_dense_20000.model" "/wdl_infer/model/wdl/1/wdl0_sparse_20000.model/,/wdl_infer/model/wdl/1/wdl1_sparse_20000.model" "/wdl_train/infer_test.csv" "local"')


# ### Load model embedding tables into local RocksDB as a parameter Server
# 
# Create a RocksDB directory with read and write permissions for storing model embedded tables.

# In[7]:


get_ipython().system('mkdir -p -m 700 /wdl_train/rocksdb')


# In[9]:


get_ipython().system('python /wdl_train/wdl_predict.py "wdl" "/wdl_infer/model/wdl/1/wdl.json"  "/wdl_infer/model/wdl/1/wdl_dense_20000.model"  "/wdl_infer/model/wdl/1/wdl0_sparse_20000.model/,/wdl_infer/model/wdl/1/wdl1_sparse_20000.model"  "/wdl_train/infer_test.csv"  "rocksdb"  "/wdl_train/rocksdb"')

