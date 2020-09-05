HUGECTR 2.2.1 USER GUIDE
===================================
<div align=center><img src ="user_guide_src/merlin_arch.png" width = '800' height ='400'/></div>
<div align=center>Fig. 1. Merlin Architecture</div>

## Introduction ##
As the training component of NVIDIA Merlin (Fig. 1), HugeCTR is a recommender specific framework which is capable of distributed training across multiple GPUs and nodes for Click-Through-Rate (CTR) estimation.
Its key missions are high-performance and ease-of-use.
The rest of this documentation is organized as follows.
We first summarize the changes in version 2.2.1 since the 2.2 release.
Then we describe HugeCTR architecture and what kinds of models it can support, whilst illustrating how to use it.
In the final section, we compare its performance quantitatively with Tensorflow CPU/GPU.

Highlighted Features of HugeCTR
* GPU Hash Table and dynamic insertion
* Multi-GPU and Multi-node training with large scale embedding support
* Mixed-precision training
* Support of four common networks and their variants: Wide and Deep Learning (WDL)[1], Deep Cross Network (DCN)[2], DeepFM [3] and DLRM[6].

## Release Notes ##
## What’s New in Version 2.2.1 ##
In HugeCTR version 2.2.1, we enriched the user-convenience features together with the refactoring efforts and bug fixes.
+ **Dataset in Parquet Format Support** :  HugeCTR data reader was extended to support [Parquet format](#parquet-data-format). The preprocessed dataset and its metadata can be generated with [nvTabular](https://github.com/NVIDIA/NVTabular)

+ **GPU-Powered Preprocessing Script** : The preprocessing script used for HugeCTR samples such as DCN, DeepFM and W&D was rewritten in nvTabular. This GPU accelerated script doesn’t use too much host memory anymore.

+ **Preprocessing Tool for DLRM Sample** : To make it easier for the user to run our DLRM sample, a preprocessing tool written in CUDA C++ was added.

+ **Use of RAPIDS MLPrims** : Some existing layers were rewritten to utilize the highly optimized machine learning primitives ([MLPrims](https://github.com/rapidsai/cuml/tree/branch-0.15/cpp/src_prims)) of RAPIDs. 
 
+ **Reorganization of Submodules** : All the submodules were moved to `third_party` directory.

+ **Revived Pascal Support** : The support for Pascal Architecture, e.g., P100 was added back. However, with a Pascal graphic card, `InteractionLayer` doesn’t support FP16.

+ **Compile Time Reduction** : By modularizing the embedding related code into several files, HugeCTR compile time was improved.

+ **Refactoring of `Tensor` and `GeneralBuffer`** : In HugeCTR, `Tensor` and `GeneralBuffer` are used for memory management and access control. In the version 2.2.1, they were refactored to clarify their responsibilities and support different memory kinds, .e.g, Host, Device and Unified. Check their interface changes if you are using them to add a new layer.

## What’s New in Version 2.2 ##
HugeCTR version 2.2 adds a lot of features to enhance its usability and performance.
HugeCTR is not only a high-performance reference design for framework designers but also a self contained training framework.
+ **Matrix Multiplication Algorithm Search** : HugeCTR runs an exhaustive [algorithm](https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmalgo_t) search for each fully connected layer to find the best performant one.

+ **AUC** : An user can choose to use AUC as an evaluation metric in addition to AverageLoss. It is also possible to stop training when AUC reaches a specified threshold.

+ **Batch Shuffle in Training Data Set** : Training data batch shuffling is supported.

+ **Different Batch Sizes for Training and Evaluation** : An user can specify the different batch sizes for training and evaluation. It can be useful to tune overall performance.

+ **Full FP16 pipeline** : In order to data and compute throughputs, We added the full FP16 pipeline.

+ **Fused Fully Connected Layer** : In FP16 mode, you can choose to use a specialized fully connected layer fused with ReLU activation function.

+ **Evaluation Data Caching on Device** : For GPUs with large memory capacity like A100, a user can choose to cache data batches for small evaluation data sets.

+ **Interaction Layer** : We added Interaction layer used for popular models such as DLRM.

+ **Optimized Data Reader for Raw Data Format** : RAW data format is supported to simplify the one hot data reading and achieve better performance.

+ **Deep Learning Recommendation Model (DLRM)** : We enabled and optimized the training of DLRM. Please find more details in [samples/dlrm](../samples/dlrm/README.md).

+ **Learning Rate Scheduling** : [Different learning rate scheduling](#sgd-optimizer--learning-rate-scheduling) is supported. <br>

+ **Weight Initialization Methods** : For each trainable layer, a user can choose which method ,e.g., XavierUnifrom, Zero, etc is used for its weight initialization.

+ **Ampere Support** : We tested and optimized HugeCTR for Ampere Architecture.

## Architecture and Supported Networks
To enable large embedding training, the embedding table in HugeCTR is model parallel and distributed across all GPUs in a homogeneous cluster, which consists of multiple nodes and multiple GPUs. Meanwhile, the dense model such as DNN is data parallel, which has one copy in each GPU (see Fig. 2).

HugeCTR supports flexible and various CTR networks with Embeddings e.g. WDL, DCN, DeepFM, in which Embedding has a three-stage workflow: table lookup, reducing the weights within a slot, concat the weights from different slots (see Fig. 4). Operations and layers supported in HugeCTR are listed as follows:
* Multi-slot embedding: Sum / Mean
* Layers: Fully Connected / Fused Fully Connected (FP16 only) / ReLU / ELU / Dropout / Split / Reshape / Concat / BatchNorm / Multiply / FmOrder2 / MultCross / ReduceSum / Interaction
* Optimizer: Adam/ Momentum SGD/ Nesterov
* Loss: CrossEntropy/ BinaryCrossEntropy

<div align=center><img src ="user_guide_src/fig1_hugectr_arch.png" width = '800' height ='400'/></div>
<div align=center>Fig. 2. HugeCTR Architecture</div>

<div align=center><img width = '600' height ='400' src ="user_guide_src/fig2_embedding_mlp.png"/></div>
<div align=center>Fig. 3. Embedding Architecture</div>

<div align=center><img width = '800' height ='300' src ="user_guide_src/fig3_embedding_mech.png"/></div>
<div align=center>Fig. 4. Embedding Mechanism</div>

## Highlighted Features
### GPU hashtable and dynamic insertion
GPU hashtable makes the data preprocessing easier and enables dynamic insertion in HugeCTR 2.x. The input training data are hash values (64bit long long type) instead of original indices. Thus embedding initialization is not required before training. A pair of <key,value> (random small weight) will be inserted during runtime only when a new key appears in the training data and the hashtable cannot find it. 
### Multi-node training and enabling large embedding
Multi-node training is supported to enable very large embedding. Thus, an embedding table of arbitrary size can be trained in HugeCTR. In multi-node solution, sparse model or embedding is also distributed across the nodes, and dense model is in data parallel. In our implementation, HugeCTR leverages NCCL and gossip[4] for high speed and scalable inter- and intra-node communication.
### Mixed precision training
In HugeCTR, to improve compute/memory throughputs with the memory footprint reduced, mixed precision training is supported.
In this mode, for matrix multiplication based layers such as `FullyConnectedLayer` and `InteractionLayer`, on Volta, Turing and Ampere architectures, TensorCores are used to boost performance. For the other layers including embeddings, the data type is changed to FP16, so that both memory bandwidth and capacity are saved.
Please note that loss scaling will be applied to avoid arithmetic underflow (see Fig. 5).  Mixed-precision training can be enabled in the JSON config file (see below).
<div align=center><img width = '500' height ='400' src ="user_guide_src/fig4_arithmetic_underflow.png"/></div>
<div align=center>Fig. 5. Arithmetic Underflow</div>

Currently, the layers and optimizers below support FP16:

* Add
* Concat
* Dropout
* FullyConnected
* FusedFullyConnected (FP16 only)
* Interaction
* ReLU
* Reshape
* Slice
* Adam
* MomentumSGD
* Nesterov
* SGD

## Usages
Training with one-shot instruction:
```shell
$ huge_ctr --train <config>.json
```
To load a snapshot, you can just modify config.json (`dense_model_file`, `sparse_model_file` in solver clause) according to the name of the snapshot. 

To run with multiple nodes: HugeCTR should be built with OpenMPI (GPUDirect support is recommended for high performance), then the configure file and model files should be located in "Network File System" and be visible to each of the processes. A sample of running in two nodes:
```shell
$ mpirun -N2 ./huge_ctr --train config.json
```
## Network Configurations
In this section, we describe how a HugeCTR JSON config file is organized. Please refer to [**sample configure file**](../samples/dlrm/dlrm_fp16_64k.json) for more details.

### Solver
Solver clause contains the configuration to training resource and task, items include:
* `lr_policy`: only supports `fixed` now.
* `display`: intervals to print loss on screen.
* `gpu`: GPU indices used in a training process, which has two levels. For example: [[0,1],[1,2]] indicates that two nodes are used; in the first node, GPU 0 and GPU 1 are used while GPU 1 and 2 are used for the second node.
It is also possible to specify noncontinuous GPU indices, e.g., [0, 2, 4, 7]
* `batchsize`: minibatch used in training.
* `batchsize_eval`: minibatch used in evaluation.
* `snapshot`: intervals to save a checkpoint in file with the prefix of `snapshot_prefix`
* `eval_interval`: intervals of evaluation on a test set.
* `eval_batches`: the number of batches will be used in loss calculation of evaluation. HugeCTR will print the average loss of the batches.
* `dense model_file`: (optional: no need to config if train from scratch) file of dense model.
* `sparse_model_file`: (optional: no need to config if train from scratch)file of sparse models. From v2.1 multi-embeddings are supported in one model. Each embedding will have one model file.
* `mixed_precision`: (optional) enabling mixed precision training with the scaler specified here. Only 128/256/512/1024 are supported.
* `eval_metrics`: (optional) The list of enabled evaluation metrics. You can choose to use both `AUC` and `AverageLoss` or one of them. For AUC, you can also set its threshold, e.g., ["AUC:0.8025"] so that the training terminates if it reaches the value. The default is AUC without the threshold.
* `"input_key_type`: if your dataset format is `Norm`, you can choose the data type of each input key. The default is I32. For a `Parquet` dataset generated by nvTabular, only I64 is allowed, while I32 must be specified to use `Raw` format.
```json
 "solver": {
    "lr_policy": "fixed",
    "display": 1000,
    "max_iter": 300000,
    "gpu": [0],
    "batchsize": 512,
    "batchsize_eval": 256,
    "snapshot": 10000000,
    "snapshot_prefix": "./",
    "eval_interval": 1000,
    "eval_batches": 60,
    "mixed_precision": 256,
    "eval_metrics": ["AUC:0.8025"],
    "dense model_file": "./dense_model.bin",
    "sparse_model_file": ["./sparse_model1.bin","./sparse_model2.bin"]
  }
```

#### Different batch size in training and evaluation ####
Different batch sizes can be set in training and evaluation respectively.
```
"batchsize": 512,
"batchsize_eval": 256,
```

#### Full FP16 pipeline ####
When `mixed_precission` is set, a full fp16 pipeline will be triggered. [7]

### Optimizer
The optimizer used in both dense and sparse models. In addition to Adam/MomentumSGD/Nesterov, We add SGD in version 2.2. Note that different optimizers can be supported in the dense and embedding parts of the model. To enable specific optimizers in embeddings, please just put the optimizer clause into the embedding layer. Otherwise, the embedding layer will use the same optimizer as the dense part. `global_update` can be specified in the optimizer. By default the optimizer will only update the hot columns of embedding in each iteration, but if you assign `true`, our optimizer will update all the columns.
Note that `global_update` will not have as good speed as not using it.
```json
"optimizer": {
  "type": "Adam",
  "global_update": true,
  "adam_hparam": {
    "learning_rate": 0.001,
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 0.0000001
  }
}
"optimizer": {
  "type": "MomentumSGD",
  "global_update": false,
  "momentum_sgd_hparam": {
    "learning_rate": 0.01,
    "momentum_factor": 0.0
  }
}
"optimizer": {
  "type": "Nesterov",
  "global_update": true,
  "nesterov_hparam": {
    "learning_rate": 0.01,
    "momentum_factor": 0.0
  }
}
```
### SGD optimizer & Learning rate scheduling
HugeCTR supports the learning rate scheduling and allows users to configure its hyperparameters.
You can set the base learning rate (`learning_rate`) and the number of initial steps used for warm-up (`warmup_steps`).
When the learning rate decay starts (`decay_start`) and the decay period in step (`decay_steps`) are also set.
Fig. 6 illustrates how these hyperparameters interact with the actual learning rate.
```json
"optimizer": {
  "type": "SGD",
  "global_update": false,
  "sgd_hparam": {
    "learning_rate": 24.0,
    "warmup_steps": 8000,
    "decay_start": 48000,
    "decay_steps": 24000
  }
}
```
<div align=center><img width = '500' src ="docs/user_guide_src/learning_rate_scheduling.png"/></div>
<div align=center>Fig. 6. Learning rate scheduling</div>

### Layers
Many different kinds of layers are supported in clause `layer`, which are categorized into data, dense layers, and sparse layers (embeddings).

#### Data
`Data` is considered as the first layer in a JSON config file. So following layers can access its dense and sparse inputs with their specified names. 
##### Norm Format
Data set in the `Norm` format (default) is consistent with the previous version. Its properties include the file name of training and testing (evaluation) set, maximum elements (key) in a sample, and the label dimensions like Fig. 7 (a).

* All the nodes will share the same file list in training.
* `dense` and `sparse` should be configured (dense_dim should be 0 if no dense feature is involved), where `dense` refers to the dense input and `sparse` refers to the `sparse` input. `sparse` should be an array here, since we support multiple `embedding` and each one requires a `sparse` input.
* The `type` of sparse input should be consistent with the following Embeddings.
* `slot_num` is the number of slots used in this training set. All the weight vectors get out of a slot will be reduced into one vector after embedding lookup (see Fig. 5).
* The sum of `slot_num` in each sparse input should be consistent with the slot number defined in the header of the training file.  
```json
    {
      "name": "data",
      "type": "Data",
      "source": "./file_list.txt",
      "eval_source": "./file_list_test.txt",
      "check": "Sum",
      "label": {
        "top": "label",
        "label_dim": 1
      },
      "dense": {
        "top": "dense",
        "dense_dim": 13
      },
      "sparse": [
        {
          "top": "data1",
          "type": "DistributedSlot",
          "max_feature_num_per_sample": 30,
          "slot_num": 26
        }
      ]
    }
```
* Caching evaluation data on device: To cache evaluation data on device, set can be specified to restrict the memory will be used.  `cache_eval_data`.
* Additional item "max_nnz": 1 can be specified to restrict the amount of memory used. 
```
"cache_eval_data": true,
```

##### Raw and Parquet Formats
We also support ‘Raw’ format, introduced in v2.2, and ‘Parquet’ format from v2.2.1. See Fig. 7 (b) and (c). Several additional item are configurable as below:
* `format":` `Raw` or `Parquet`
* `num_samples` should be specified because the `Raw` format file doesn't have a header like in the `Norm` format. It's the same to `eval_num_samples`. ‘Parquet’ format doesn’t need to specify this field.
* `check`: ‘Raw’ and ‘Parquet’ don’t use this field. So use the value `None` or omit `check` itself.
* `slot_size_array`: an array of table vocabulary size.
* `float_label_dense`: **This is valid only for `Raw` format.** If its value is set to `true`, the label and dense features of each sample are interpreted as `float` values. Otherwie, they are read as `int` values while the dense features are preprocessed with `log(dense[i] + 1.f)`. The default value is `false`.
```json
     {
	 "name": "data",
	  "type": "Data",
	  "format": "Raw",
	  "num_samples": 4195155968,
	  "slot_size_array": [39884406,    39043,    17289,     7420,    20263,    3,  7120,     1543,  63, 38532951,  2953546,   403346, 10,     2208,    11938,      155,        4,      976, 14, 39979771, 25641295, 39664984,   585935,    12972,  108,  36],
	  "source": "/etc/workspace/dataset/train_data.bin",
	  "eval_num_samples": 89137319,
	  "eval_source": "/etc/workspace/dataset/test_data.bin",
	  "check": "None",
	  "cache_eval_data": true,
	  "label": {
              "top": "label",
              "label_dim": 1
	  },
	  "dense": {
              "top": "dense",
              "dense_dim": 13
	  },
	  "sparse": [
              {
		  "top": "data1",
		  "type": "LocalizedSlot",
		  "max_feature_num_per_sample": 26,
                  "max_nnz": 1,
		  "slot_num": 26
              }
	  ]
    },


```
#### Embedding:
* An embedding table can be segmented into multiple slots (or feature fields), which spans multiple GPUs and multiple nodes.
* `type`: two types of embedding are supported: `LocalizedSlotSparseEmbeddingHash`, `DistributedSlotSparseEmbeddingHash`.
  * `LocalizedSlotSparseEmbeddingHash`: each individual slot will be located in each GPU in turn, and not shared. This type of embedding has the best scalability.
    * `plan_file`: a plan file should be specified when using `LocalizedSlotSparseEmbeddingHash`. To generate a plan file please refer to the [**README**](../samples/dcn/README.md) in dcn sample.
  * `DistributedSlotSparseEmbeddingHash`: Each GPU will have a portion of a slot. This type of embedding is useful when there exists the load imbalance among slots and potentially has OOM issues.
  * In single GPU training, for your convenience please use `DistributedSlotSparseEmbeddingHash`.
  * `LocalizedSlotSparseEmbeddingOneHot`: Optimized version of `LocalizedSlotSparseEmbeddingHash` but only supports single node training with p2p connections between each pair of GPUs and one hot input.
* `max_vocabulary_size_per_gpu`: the maximum possible size of embedding for one gpu.
* `embedding_vec_size`: The size of each embedding vector.
* `combiner`: 0 is sum and 1 is mean.
* `optimizer`: (optional) from v2.1, HugeCTR supports different optimizers in dense and sparse models. You can specify your optimizer of this Embedding here. If not specified, HugeCTR will reuse the optimizer of the dense model here.
```json
    {
      "name": "sparse_embedding1",
      "type": "LocalizedSlotSparseEmbeddingHash",
      "bottom": "data1",
      "top": "sparse_embedding1",
      "plan_file": "all2all_plan_bi_1.json",
      "sparse_embedding_hparam": {
        "max_vocabulary_size_per_gpu": 1737710,
        "embedding_vec_size": 16,
        "combiner": 0
      },
      "optimizer": {
        "type": "Adam",
        "learning_rate": true,
        "adam_hparam": {
          "learning_rate": 0.001,
          "beta1": 0.9,
          "beta2": 0.999,
          "epsilon": 0.0000001
        }
      }
    }

```
#### Layers:
* Reshape: the first layer after embedding should be `Reshape` to reshape the tensor from 3D to 2D. Reshape is the only layer that accepts both 3D and 2D input and the output must be 2D.
`leading_dim` in `Reshape` is the leading dimension of the output.

* Concat: you can `Concat` at most **five** tensors into one and list the name in the `bottom` array. Note that the second dimension (usually batch size) should be the same.

* Slice: opposite to concat, we support `Slice` layer to copy specific `ranges` of input tensor to named output tensors. In the example below, we duplicate input tensor with `Slice` (0 is inclusive, 429 is exclusive). 

```json

    {
      "name": "reshape1",
      "type": "Reshape",
      "bottom": "sparse_embedding1",
      "top": "reshape1",
      "leading_dim": 416
    }
    {
      "name": "concat1",
      "type": "Concat",
      "bottom": ["reshape1","dense"],
      "top": "concat1"
    }
    {
      "name": "slice1",
      "type": "Slice",
      "bottom": "concat1",
      "ranges": [[0,429], [0,429]],
      "top": ["slice11", "slice12"]
    }


```

* ELU: the type name is `ELU`, and a `elu_param` called `alpha` in it can be configured.
* Fully Connected (`InnerProduct`): bias is supported in fully connected layers and `num_output` is the dimension of output.
* Fused fully connected layer(`FusedInnerProduct`): Fused bias adding and relu activation into a single layer.
* Loss: different from the other layers, you can specify which `regularization` you will use. This is optional. By default no regularization will be used.
```json
{
  "name": "elu1",
  "type": "ELU",
  "bottom": "fc1",
  "top": "elu1",
  "elu_param": {
    "alpha": 1.0
  }
}
{
  "name": "fc8",
  "type": "InnerProduct",
  "bottom": "concat2",
  "top": "fc8",
  "fc_param": {
    "num_output": 1
  }
}
{
  "name": "fc2",
  "type": "FusedInnerProduct",
  "bottom": "fc1",
  "top": "fc2",
  "fc_param": {
    "num_output": 256
  }
}
{
  "name": "loss",
  "type": "BinaryCrossEntropyLoss",
  "bottom": ["fc8","label"],
  "regularizer": "L2",
  "top": "loss"
}

```
* Interaction layer: 
```json
{
  "name": "interaction1",
  "type": "Interaction",
  "bottom": ["fc3", "sparse_embedding1"],
  "top": "interaction1"
}
```
* For more details, please refer to [**parser.cpp**](../HugeCTR/src/parser.cpp)

## Data Format

<div align=center><img width="80%" height="80%" src ="user_guide_src/dataset_format.png"/></div>
<div align=center>Fig. 7 (a) Norm (b) Raw (c) Parquet dataset format </div>

### Norm Dataset
A `Norm` format dataset in HugeCTR consists of an ASCII format file list and a set of data files in binary format to maximize the performance of data loading and minimize the storage. Note that a data file is the minimum reading granularity for a reading thread, so at least 10 files in each file list are required for best performance. 
#### File List
A file list starts with a number which indicates the number of files in the file list, followed by the paths of the data files.
```shell
$ cat simple_sparse_embedding_file_list.txt
10
./simple_sparse_embedding/simple_sparse_embedding0.data
./simple_sparse_embedding/simple_sparse_embedding1.data
./simple_sparse_embedding/simple_sparse_embedding2.data
./simple_sparse_embedding/simple_sparse_embedding3.data
./simple_sparse_embedding/simple_sparse_embedding4.data
./simple_sparse_embedding/simple_sparse_embedding5.data
./simple_sparse_embedding/simple_sparse_embedding6.data
./simple_sparse_embedding/simple_sparse_embedding7.data
./simple_sparse_embedding/simple_sparse_embedding8.data
./simple_sparse_embedding/simple_sparse_embedding9.data
```

#### Data File
A data file (binary) contains a header followed by the data samples.

Header field:
```c
typedef struct DataSetHeader_ {
  long long error_check;        // 0: no error check; 1: check_num
  long long number_of_records;  // the number of samples in this data file
  long long label_dim;          // dimension of label
  long long dense_dim;          // dimension of dense feature
  long long slot_num;           // slot_num for each embedding
  long long reserved[3];        // reserved for future use
} DataSetHeader;

```

Data Definition (each sample):
```c
typedef struct Data_{
  int length;                   // bytes in this sample (optional: only in check_sum mode )
  float label[label_dim];       
  float dense[dense_dim];
  Slot slots[slot_num];          
  char checkbits;                // checkbit for this sample (optional: only in checksum mode)
} Data;

typedef struct Slot_{
  int nnz;
  unsigned int*  keys; // changeable to `long long` with `"input_key_type"` in `solver` object of JSON config file.
} Slot;
```

Data field often has a lot of samples. Each sample starts with the labels in integer type, followed by `nnz` (number of nonzero) and key in long long (or unsigned int) type like Fig. 7 (a).

### Raw Dataset
Fig. 7 (b) shows the structure of a `Raw` dataset sample. To use the format, you need to specify the number of train samples and the number of evaluation samples and the slot size of the embedding in the `"data"` clause.

```json
"format": "Raw",
"num_samples": 4195155968,
"slot_size_array": [39884406,    39043,    17289,     7420,    20263,    3,  7120,     1543,  63, 38532951,  2953546,   403346, 10,     2208,    11938,      155,        4,      976, 14, 39979771, 25641295, 39664984,   585935,    12972,  108,  36],
"eval_num_samples": 89137319,
```
A proxy in C struct for a sample:
```c
typedef struct Data_{
  int label[label_dim];       
  int dense[dense_dim];
  int category[sparse_dim];
} Data;
```
### Parquet Data Format
Parquet is a column-oriented data format of Apache Hadoop ecosystem, which is free and open-source. To reduce the file sizes, it supports compression and encoding. For more information, check out [its official documentation](https://parquet.apache.org/documentation/latest/).

Fig. 7 (c)  shows an example Parquet dataset in terms of HugeCTR. Currently nested column types are not supported in HugeCTR Parquet data loader. Any missing values in a column are not allowed. Like `Norm` format, the label and dense feature columns should be in float format. Slot feature columns are expected to be in Int64 format. The data columns inside Parquet file can be arranged in any order.  A separate metadata file, named `_metadata.json` is required by HugeCTR to get required information on number of rows in each parquet file and column index mapping for each of label, dense (numerical) and slot (categorical) features.  The example is shown below:
```
{
"file_stats": [{"file_name": "file1.parquet", "num_rows": 6528076}, {"file_name": "file2.parquet", "num_rows": 6528076}], 
"cats": [{"col_name": "C11", "index": 24}, {"col_name": "C24", "index": 37}, {"col_name": "C17", "index": 30}, {"col_name": "C7", "index": 20}, {"col_name": "C6", "index": 19}],
"conts": [{"col_name": "I5", "index": 5}, {"col_name": "I13", "index": 13}, {"col_name": "I2", "index": 2}, {"col_name": "I10", "index": 10}],
"labels": [{"col_name": "label", "index": 0}]
}
```

### Non-Trainable Parameters
Some of the layers will generate statistical results during training like Batch Norm. Such parameters are outputs of CTR training (called “non-trainable parameters”) and used in inference.

In HugeCTR such parameters will be written into a JSON format file along with weights during training.
```json
{
  "layers": [
    {
      "type": "BatchNorm",
      "mean": [-0.192325, 0.003050, -0.323447, -0.034817, -0.091861], 
      "var": [0.738942, 0.410794, 1.370279, 1.156337, 0.638146] 
    },
    {
      "type": "BatchNorm",
      "mean": [-0.759954, 0.251507, -0.648882, -0.176316, 0.515163],
      "var": [1.434012, 1.422724, 1.001451, 1.756962, 1.126412]
    },
    {
      "type": "BatchNorm",
      "mean": [0.851878, -0.837513, -0.694674, 0.791046, -0.849544],
      "var": [1.694500, 5.405566, 4.211646, 1.936811, 5.659098]
    }
  ]
}
```
## Performance on DGX-1
In this section, we test the scalability of HugeCTR and compare its performance and result with TensorFlow on NVIDIA V100 GPUs available in a single DGX-1 system. In summary, we can achieve about 114x speedup over multi-thread Tensorflow CPU with only one V100 and generate almost the same loss curves for both evaluation and training (see Fig. 10 and Fig. 11).

Test environment:
* CPU Server: Dual 20-core Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz
* TensorFlow version 2.0.0
* V100 16GB: NVIDIA DGX1 servers

Network:
* `Wide Deep Learning`: Nx 1024-unit FC layers with ReLU and dropout, emb_dim: 16; Optimizer: Adam for both Linear and DNN models
* `Deep Cross Network`: Nx 1024-unit FC layers with ReLU and dropout, emb_dim: 16, 6x cross layers; Optimizer: Adam for both Linear and DNN models

Data set:
The data is provided by CriteoLabs [5]. The original training set contains 45,840,617 examples. Each example contains a label (1 if the ad was clicked, otherwise 0) and 39 features (13 integer features and 26 categorical features). 

Preprocessing:
* Common: preprocessed by using the scripts available in tools/criteo_script
* HugeCTR: converted to HugeCTR data format with criteo2hugectr
* TF: converted to TFRecord format for the efficient training on Tensorflow

### HugeCTR
The good scalability of HugeCTR as the number of active GPUs is increased, is mainly because of the high efficient data exchange and the three-stage processing pipeline. In this pipeline, we overlap the data reading from file, host to device data transaction (inter- and intra- node) and GPU training.  The following chart shows the scalability of HugeCTR with the configuration of Batch Size=16384, Layers=7 on DGX1 Servers.
<div align=center><img width = '800' height ='400' src ="user_guide_src/fig12_multi_gpu_performance.PNG"/></div>
<div align=center>Fig. 9 Multi-GPU Performance of HugeCTR</div>

### TensorFlow
In the TensorFlow test case below, HugeCTR shows up to 114x speedup to a CPU server with TensorFlow with only one V100 GPU and almost the same loss curve.


<div align=center><img width = '800' height ='400' src ="user_guide_src/WDL.JPG"/></div>
<div align=center>Fig. 10 WDL performance and loss curve comparison with TensorFlow v2.0 </div>


<div align=center><img width = '800' height ='400' src ="user_guide_src/DCN.JPG"/></div>
<div align=center>Fig. 11 WDL performance and loss curve comparison with TensorFlow v2.0</div>

## Performance on DGX-2 and DGX A100
We submitted the DLRM benchmark with HugeCTR v2.2 to [MLPerf Training v0.7](https://mlperf.org/training-results-0-7). The dataset was [Criteo Terabyte Click Logs](https://labs.criteo.com/2013/12/download-terabyte-click-logs/) (of 4 billion user & item interactions over 24 days). The target machine was DGX-2 with 16 V100 GPUs and DGX A100 with 8 A100 GPUs. It set records among the commercially available products. Check out [this blog post](https://developer.nvidia.com/blog/optimizing-ai-performance-for-mlperf-v0-7-training/) for more details.

## Known Issues

* The auto plan file generator doesn't support generating a plan file for 1 GPU system. In this case, user need to manually create the json plan file with the following content:
` {"type": "all2all", "num_gpus": 1, "main_gpu": 0, "num_steps": 1, "num_chunks": 1, "plan": [[0, 0]], "chunks": [1]} ` and rename the json plan file to the name listed in the HugeCTR configuration file.
* For a 2-GPU system, if there are 2 NVLinks between GPUs, then the auto plan file generator will print some warnings `RuntimeWarning: divide by zero encountered in true_divide`. This will not affect the generated json plan file.
* The current plan file generator doesn't support the system that is only partially connected by NVLink. That is the system which has NVLink but exists 2 GPUs where data cannot travel through NVLink between them.
* Users need to set an environment variable: `export CUDA_DEVICE_ORDER=PCI_BUS_ID` to ensure that CUDA runtime and driver have consistent ordering of GPUs.
* `LocalizedSlotSparseEmbeddingOneHot` only supports single node training with p2p connections between each pair of involved GPUs.
* In v2.2.1, training halts in running DLRM sample on DGX2 due to a CUDA Graph related issue. To run the sample on DGX2, disable the use of CUDA Graph with `"cuda_graph": false` even if it degrades the performance a bit. We are workin on fixing this issue. There isn't such a problem on DGX A100.

## Reference
[1] Wide and Deep Learning: https://arxiv.org/abs/1606.07792

[2] Deep Cross Network: https://arxiv.org/abs/1708.05123

[3] DeepFM: https://arxiv.org/abs/1703.04247 

[4] Gossip: https://github.com/Funatiq/gossip

[5] CriteoLabs: http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/

[6] DLRM: https://ai.facebook.com/blog/dlrm-an-advanced-open-source-deep-learning-recommendation-model/

[7] Mixed Precision Training: https://arxiv.org/abs/1710.03740




