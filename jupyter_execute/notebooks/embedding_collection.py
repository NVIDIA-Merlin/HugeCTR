#!/usr/bin/env python
# coding: utf-8

# # HugeCTR Embedding Collection
# 
# ## About this Notebook
# 
# This notebook demonstrates the following:
# 
# - Introduces the API of the embedding collection.
# - Introduces the embedding table placement strategy (ETPS) and how to configure ETPS in embedding collection.
# - Shows how to use an embedding collection in a DLRM model with the Criteo dataset for training and evaluation.
#   The notebook shows two different ETPS as reference.
# 
# ## Concepts and API Reference
# 
# The following key classes and configuration file are used in this notebook:
# 
# - `hugectr.EmbeddingTableConfig`
# - `hugectr.EmbeddingPlanner`
# - JSON plan file for the ETPS
# 
# For the concepts and API reference information about the classes and file, see the [Overview of Using the HugeCTR Embedding Collection](https://nvidia-merlin.github.io/HugeCTR/master/api/hugectr_layer_book.html#overview-of-using-the-hugectr-embedding-collection) in the HugeCTR Layer Classes and Methods information.

# ## Use an Embedding Collection with a DLRM Model
# 
# ### Prepare the Data
# 
# Follow the instructions under heading "Preprocess the Dataset through NVTabular" from the README in the [samples/deepfm](https://github.com/NVIDIA-Merlin/HugeCTR/tree/master/samples/deepfm) directory of the repository to prepare data.
# 
# ### Prepare the Training Script
# 
# This notebook was developed with on single DGX-1 to run the DLRM model in this notebook. The GPU info in DGX-1 is as follows. It consists of 8 V100-SXM2 GPUs.
# 

# In[6]:


get_ipython().system(' nvidia-smi')


# The training script, `dlrm_train.py`, uses the the embedding collection API.
# The script accepts one command-line argument that specifies the plan file so we can run the script several times and evaluate different ETPS:
# 

# In[25]:


get_ipython().run_cell_magic('writefile', 'dlrm_train.py', 'import sys\nimport hugectr\n\nplan_file = sys.argv[1]\nslot_size_array = [203931, 18598, 14092, 7012, 18977, 4, 6385, 1245, 49,\n                   186213, 71328, 67288, 11, 2168, 7338, 61, 4, 932, 15,\n                   204515, 141526, 199433, 60919, 9137, 71, 34]\n\nsolver = hugectr.CreateSolver(\n    max_eval_batches=70,\n    batchsize_eval=65536,\n    batchsize=65536,\n    lr=0.5,\n    warmup_steps=300,\n    vvgpu=[[0, 1, 2, 3, 4, 5, 6, 7]],\n    repeat_dataset=True,\n    i64_input_key=True,\n    metrics_spec={hugectr.MetricsType.AverageLoss: 0.0},\n    use_embedding_collection=True,\n)\n\nreader = hugectr.DataReaderParams(\n    data_reader_type=hugectr.DataReaderType_t.Parquet,\n    source=["./deepfm_data_nvt/train/_file_list.txt"],\n    eval_source="./deepfm_data_nvt/val/_file_list.txt",\n    check_type=hugectr.Check_t.Non,\n    slot_size_array=slot_size_array\n)\n\noptimizer = hugectr.CreateOptimizer(\n    optimizer_type=hugectr.Optimizer_t.SGD,\n    update_type=hugectr.Update_t.Local,\n    atomic_update=True\n)\n\nmodel = hugectr.Model(solver, reader, optimizer)\n\nmodel.add(\n    hugectr.Input(\n        label_dim=1,\n        label_name="label",\n        dense_dim=13,\n        dense_name="dense",\n        data_reader_sparse_param_array=[\n            hugectr.DataReaderSparseParam("data{}".format(i), 1, False, 1)\n            for i in range(len(slot_size_array))\n        ],\n    )\n)\n\n# Create the embedding table.\nembedding_table_list = []\nfor i in range(len(slot_size_array)):\n    embedding_table_list.append(\n        hugectr.EmbeddingTableConfig(\n            table_id=i,\n            max_vocabulary_size=slot_size_array[i],\n            ev_size=128,\n            min_key=0,\n            max_key=slot_size_array[i],\n        )\n    )\n\n# Create the embedding planner and embedding collection.\nembedding_planner = hugectr.EmbeddingPlanner()\nemb_vec_list = []\nfor i in range(len(slot_size_array)):\n    embedding_planner.embedding_lookup(\n        table_config=embedding_table_list[i],\n        bottom_name="data{}".format(i),\n        top_name="emb_vec{}".format(i),\n        combiner="sum"\n    )\n\nembedding_collection = embedding_planner.create_embedding_collection(plan_file)\n\nmodel.add(embedding_collection)\n# need concat\nmodel.add(\n    hugectr.DenseLayer(\n        layer_type=hugectr.Layer_t.Concat,\n        bottom_names=["emb_vec{}".format(i) for i in range(len(slot_size_array))],\n        top_names=["sparse_embedding1"],\n        axis=1\n    )\n)\n\nmodel.add(\n    hugectr.DenseLayer(\n        layer_type=hugectr.Layer_t.InnerProduct,\n        bottom_names=["dense"],\n        top_names=["fc1"],\n        num_output=512\n    )\n)\n\nmodel.add(\n    hugectr.DenseLayer(\n        layer_type=hugectr.Layer_t.ReLU, bottom_names=["fc1"], top_names=["relu1"]\n    )\n)\n\nmodel.add(\n    hugectr.DenseLayer(\n        layer_type=hugectr.Layer_t.InnerProduct,\n        bottom_names=["relu1"],\n        top_names=["fc2"],\n        num_output=256\n    )\n)\n\nmodel.add(\n    hugectr.DenseLayer(\n        layer_type=hugectr.Layer_t.ReLU, bottom_names=["fc2"], top_names=["relu2"]\n    )\n)\n\nmodel.add(\n    hugectr.DenseLayer(\n        layer_type=hugectr.Layer_t.InnerProduct,\n        bottom_names=["relu2"],\n        top_names=["fc3"],\n        num_output=128\n    )\n)\n\nmodel.add(\n    hugectr.DenseLayer(\n        layer_type=hugectr.Layer_t.ReLU, bottom_names=["fc3"], top_names=["relu3"]\n    )\n)\n\nmodel.add(\n    hugectr.DenseLayer(\n        layer_type=hugectr.Layer_t.Interaction,  # interaction only support 3-D input\n        bottom_names=["relu3", "sparse_embedding1"],\n        top_names=["interaction1"],\n    )\n)\n\nmodel.add(\n    hugectr.DenseLayer(\n        layer_type=hugectr.Layer_t.InnerProduct,\n        bottom_names=["interaction1"],\n        top_names=["fc4"],\n        num_output=1024,\n    )\n)\n\nmodel.add(\n    hugectr.DenseLayer(\n        layer_type=hugectr.Layer_t.ReLU, bottom_names=["fc4"], top_names=["relu4"]\n    )\n)\n\nmodel.add(\n    hugectr.DenseLayer(\n        layer_type=hugectr.Layer_t.InnerProduct,\n        bottom_names=["relu4"],\n        top_names=["fc5"],\n        num_output=1024,\n    )\n)\n\nmodel.add(\n    hugectr.DenseLayer(\n        layer_type=hugectr.Layer_t.ReLU, bottom_names=["fc5"], top_names=["relu5"]\n    )\n)\n\nmodel.add(\n    hugectr.DenseLayer(\n        layer_type=hugectr.Layer_t.InnerProduct,\n        bottom_names=["relu5"],\n        top_names=["fc6"],\n        num_output=512,\n    )\n)\n\nmodel.add(\n    hugectr.DenseLayer(\n        layer_type=hugectr.Layer_t.ReLU, bottom_names=["fc6"], top_names=["relu6"]\n    )\n)\n\nmodel.add(\n    hugectr.DenseLayer(\n        layer_type=hugectr.Layer_t.InnerProduct,\n        bottom_names=["relu6"],\n        top_names=["fc7"],\n        num_output=256,\n    )\n)\n\nmodel.add(\n    hugectr.DenseLayer(\n        layer_type=hugectr.Layer_t.ReLU, bottom_names=["fc7"], top_names=["relu7"]\n    )\n)\n\nmodel.add(\n    hugectr.DenseLayer(\n        layer_type=hugectr.Layer_t.InnerProduct,\n        bottom_names=["relu7"],\n        top_names=["fc8"],\n        num_output=1,\n    )\n)\n\nmodel.add(\n    hugectr.DenseLayer(\n        layer_type=hugectr.Layer_t.BinaryCrossEntropyLoss,\n        bottom_names=["fc8", "label"],\n        top_names=["loss"],\n    )\n)\n\nmodel.compile()\nmodel.summary()\nmodel.fit(\n    max_iter=1000,\n    display=100,\n    eval_interval=100,\n    snapshot=10000000,\n    snapshot_prefix="dlrm",\n)\n')


# ### Embedding Table Placement Strategy: Data Parallel and Model Parallel
# 
# The following `generate_plan()` function shows how to configure small tables as data parallel and use model parallel for larger tables.
# Each table is on single GPU and different GPU will hold different table&mdash;the same way we work with data in `hugectr.LocalizedHashEmbedding`.

# In[3]:


def print_plan(plan):
    for id, single_gpu_plan in enumerate(plan):
        print("single_gpu_plan index = {}".format(id))
        for plan_attr in single_gpu_plan:
            for key in plan_attr:
                if key != "global_embedding_list":
                    print("\t{}:{}".format(key, plan_attr[key]))
                else:
                    prefix_len = len(key)
                    left_space_fill = " " * prefix_len
                    print("\t{}:{}".format(key, plan_attr[key][0]))
                    for index in range(1, len(plan_attr[key])):
                        print("\t{}:{}".format(left_space_fill, plan_attr[key][index]))


def generate_plan(slot_size_array, gpu_count, plan_file):

    mp_table = [i for i in range(len(slot_size_array)) if slot_size_array[i] > 6000]
    dp_table = [i for i in range(len(slot_size_array)) if slot_size_array[i] <= 6000]

    # Place the table across all GPUs.
    plan = []
    for gpu_id in range(gpu_count):
        single_gpu_plan = []
        mp_plan = {
            "local_embedding_list": [
                table_id
                for i, table_id in enumerate(mp_table)
                if i % gpu_count == gpu_id
            ],
            "table_placement_strategy": "mp",
        }
        dp_plan = {"local_embedding_list": dp_table, "table_placement_strategy": "dp"}
        single_gpu_plan.append(mp_plan)
        single_gpu_plan.append(dp_plan)
        plan.append(single_gpu_plan)

    # Generate the global view of table placement.
    mp_global_embedding_list = []
    dp_global_embedding_list = []
    for single_gpu_plan in plan:
        mp_global_embedding_list.append(single_gpu_plan[0]["local_embedding_list"])
        dp_global_embedding_list.append(single_gpu_plan[1]["local_embedding_list"])
    for single_gpu_plan in plan:
        single_gpu_plan[0]["global_embedding_list"] = mp_global_embedding_list
        single_gpu_plan[1]["global_embedding_list"] = dp_global_embedding_list
    print_plan(plan)

    # Write the plan file to disk.
    import json
    with open(plan_file, "w") as f:
        json.dump(plan, f, indent=4)


# In[4]:


slot_size_array = [
    203931,
    18598,
    14092,
    7012,
    18977,
    4,
    6385,
    1245,
    49,
    186213,
    71328,
    67288,
    11,
    2168,
    7338,
    61,
    4,
    932,
    15,
    204515,
    141526,
    199433,
    60919,
    9137,
    71,
    34,
]

generate_plan(
    slot_size_array=slot_size_array,
    gpu_count=8,
    plan_file="./dp_and_localized_plan.json",
)


# In[5]:


get_ipython().system('python3 dlrm_train.py ./dp_and_localized_plan.json')


# ### Embedding Table Placement Strategy: Distributed
# 
# The `generate_distributed_plan()` function shows how to distribute all tables across all GPUs
# This strategy is similar to `hugectr.DistributedHashEmbedding`.
# 

# In[6]:


def generate_distributed_plan(slot_size_array, gpu_count, plan_file):
    # Place the table across all GPUs.
    plan = []
    for gpu_id in range(gpu_count):
        distributed_plan = {
            "local_embedding_list": [
                table_id for table_id in range(len(slot_size_array))
            ],
            "table_placement_strategy": "mp",
            "shard_id": gpu_id,
            "shards_count": gpu_count,
        }
        plan.append([distributed_plan])

    # Generate the global view of table placement.
    distributed_global_embedding_list = []
    for single_gpu_plan in plan:
        distributed_global_embedding_list.append(
            single_gpu_plan[0]["local_embedding_list"]
        )
    for single_gpu_plan in plan:
        single_gpu_plan[0]["global_embedding_list"] = distributed_global_embedding_list
    print_plan(plan)

    # Write the plan file to disk.
    import json
    with open(plan_file, "w") as f:
        json.dump(plan, f, indent=4)


# In[7]:


slot_size_array = [
    203931,
    18598,
    14092,
    7012,
    18977,
    4,
    6385,
    1245,
    49,
    186213,
    71328,
    67288,
    11,
    2168,
    7338,
    61,
    4,
    932,
    15,
    204515,
    141526,
    199433,
    60919,
    9137,
    71,
    34,
]

generate_distributed_plan(
    slot_size_array=slot_size_array,
    gpu_count=8,
    plan_file="./distributed_plan.json"
)


# In[8]:


get_ipython().system('python3 dlrm_train.py ./distributed_plan.json')


# ### Performance Comparison for the Different ETPS
# 
# The iteration duration for the data parallel and model parallel strategy is `103.45s`.
# For the distributed strategy, the duration is `190.85s`.
# This comparison shows how different ETPS can greatly affect the performance of embedding.
# The results show that performance is better if you configure the embedding table as data parallel or localized when the table can fit on a single GPU.
