HUGECTR 2.1 USER GUIDE
===================================
## Introduction
HugeCTR is a high-efficiency GPU framework designed for Click-Through-Rate (CTR) estimation training, which targets both high performance and easy usage. The rest of this documentation is organized as follows. We first introduce the new features added in the version 2.1. We then describe HugeCTR architecture and what kinds of model it can support, whilst illustrating how to use it. In the final section, we compare its performance quantitatively with Tensorflow CPU/GPU.

Highlighted features of HugeCTR
* GPU Hashtable and dynamic insertion
* Multi-node training and very large embedding support
* Mixed-precision training

## New Features in Version 2.1
HugeCTR version 2.1 is a major update which aims to provide a flexible, fast, scalable and reliable solution for CTR Training. Framework designers can consider it as a high-performance reference design. 

* Supporting three important networks: Wide and Deep Learning (WDL)[1], Deep Cross Network (DCN)[2] and DeepFM [3] 
* A new embedding implementation `LocalizedSlotSparseEmbedding` which reduces the memory transactions across GPUs and nodes resiliently to the number of GPUs.
* Supporting multiple Embeddings in one network
* Supporting dense feature input, which doesn't need any embdding layer
* Supporting new layers like: Dropout / Split / Reshape / Multiply / FmOrder2 / MultCross / Add
* Check bits in data reader to enable data check and error skip.
* L1 / L2 Regularization

## Architecture and Supported Networks
To enable large embedding training, the embedding table in HugeCTR is model parallel and distributed across all the GPUs in a homogeneous cluster, which consists of multiple nodes and multiple GPUs. Meanwhile, the dense model such as DNN is data parallel, which has one copy in each GPU (see Fig.1).

HugeCTR supports flexible and various CTR networks with Embeddings e.g. WDL, DCN, DeepFM, in which Embedding has a three-stage workflow: table lookup, reducing the weights within a slot, concat the weights from different slots (see Fig.3). Operations and layers supported in HugeCTR are listed as follows:
* Multi-slot embedding: Sum / Mean
* Layers: Fully Connected / ReLU / ELU / Dropout / Split / Reshape / Concat / BN / Multiply / FmOrder2 / MultCross / add
* Optimizer: Adam/ Momentum SGD/ Nesterov
* Loss: CrossEngtropy/ BinaryCrossEntropy

<div align=center><img src ="user_guide_src/fig1_hugectr_arch.png" width = '800' height ='400'/></div>
<div align=center>Fig.1 HugeCTR Architecture</div>

<div align=center><img width = '600' height ='400' src ="user_guide_src/fig2_embedding_mlp.png"/></div>
<div align=center>Fig. 2 Embedding Architecture</div>

<div align=center><img width = '800' height ='300' src ="user_guide_src/fig3_embedding_mech.png"/></div>
<div align=center>Fig. 3 Embedding Mechanism</div>

## Highlighted Features
### GPU hashtable and dynamic insertion
GPU Hashtable makes the data preprocessing easier and enables dynamic insertion in HugeCTR 2.x. The input training data are hash values (64bit long long type) instead of original indices. Thus embedding initialization is not required before training. A pair of <key,value> (random small weight) will be inserted during runtime only when a new key appears in the training data and hashtable cannot find it. 
### Multi-node training and enabling large embedding
Multi-node training is supported to enable very large embedding. Thus, an embedding table of arbitrary size can be trained in HugeCTR. In multi-node solution, sparse model or embedding is also distributed across the nodes, and dense model is in data parallel. In our implementation, HugeCTR leverages NCCL and gossip[4] for high speed and scalable inner- and intra-node communication.
### Mixed precision training
Mixed-precision is an important feature in latest NVIDIA GPUs. On volta and Turing GPUs, fully connected layer can be configured to run on TensorCore with FP16. Please note that loss scaling will be applied to avoid arithmetic underflow (see Fig. 4).  Mixed-precision training can be enabled in cmake options (see README.md).

<div align=center><img width = '500' height ='400' src ="user_guide_src/fig4_arithmetic_underflow.png"/></div>
<div align=center>Fig 4. Arithmetic Underflow</div>

## Usages
Training with one-shoot instruction:
```shell
$ huge_ctr –-train config.json
```
To load a snapshot, you can just modify config.json (`dense_model_file`, `sparse_model_file` in solver clause) according to the name of the snapshot. 

To run with multiple node: HugeCTR should be built with OpenMPI (GPUDirect support is recommended for high performance), then the configure file and model files should be located in "Network File System" and be visible to each of the processes. A sample of runing in two nodes:
```shell
$ mpirun -N2 ./huge_ctr --train config.json
```
## Network Configurations
Please refer to [**sample configure file**](utest/simple_sparse_embedding.json)

### Solver
Solver clause contains the configuration to training resource and task, items include:
* `lr_policy`: only supports `fixed` now.
* `display`: intervals to print loss on screen.
* `gpu`: GPU indices used in a training process, which has two levels. For example: [[0,1],[2,3]] means that two node are used, and in the first node GPUs with index 0 and 1 are used and 2, 3 in the second node.
* `batchsize`: minibatch used in training.
* `snapshot`: intervals to save a checkpoint in file with the prefix of `snapshot_prefix`
* `eval_interval`: intervals of evaluation on test set.
* `eval_batches`: the number of batches will be used in loss calculation of evaluation. HugeCTR will print the average loss of the batches.
* `dense model_file`: (optional: no need to config if train from scratch) file of dense model.
* `sparse_model_file`: (optional: no need to config if train from scratch)file of sparse models. In v2.1 multi-embeddings are supported in one model. Each embedding will have one model file.
* `mixed_precision`: (optional) enabling mixed precision training with the scaler specified here. Only 128/256/512/1024 are supported.
```json
 "solver": {
    "lr_policy": "fixed",
    "display": 1000,
    "max_iter": 300000,
    "gpu": [0],
    "batchsize": 512,
    "snapshot": 10000000,
    "snapshot_prefix": "./",
    "eval_interval": 1000,
    "eval_batches": 60,
    "mixed_precision": 256,
    "dense model_file": "./dense_model.bin",
    "sparse_model_file": ["./sparse_model1.bin","./sparse_model2.bin"]
  }
```
### Optimizer
The optimizer used in both dense and sparse models. Adam/MomentumSGD/Nesterov are supported in v2.1. Note that different optimizers can be supported in dense model and each embeddings.
To enable specific optimizers in embeddings, please just put the optimizer clause into the embedding layer. Otherwise, the embedding layer will use the same optimizer as dense model. 
`global_update` can be specified in optimizer. By default optimizer will only update the hot columns of embedding in each iterations, but if you assign `true`, our optimizer will update all the columns.
Note that `global_update` will not have as good speed as not using it.
```json
"optimizer": {
  "type": "Adam",
  "global_update": true,
  "adam_hparam": {
    "alpha": 0.005,
    "beta1": 0.9,
    "beta2": 0.999,
    "epsilon": 0.000001
  }
}
"optimizer": {
  "type": "MomentumSGD",
  "global_update": false,
  "momentum_sgd_hparam": {
    "learning_rate": 0.005,
    "momentum_factor": 0.0
  }
}
"optimizer": {
  "type": "Nesterov",
  "global_update": true,
  "nesterov_hparam": {
    "learning_rate": 0.005,
    "momentum_factor": 0.0
  }
}
```

### Layers
Many different kinds of layers are supported in clause `layer`, which includes data layer; dense model like: Fully Connected / ReLU / ELU / Dropout / Split / Reshape / Concat / BN / Multiply / FmOrder2 / MultCross / add , and sparse model LocalizedSlotSparseEmbeddingHash / DistributedSlotSparseEmbeddingHash. `Embedding` should always be the first layer after data layer.

Data:
Data set properties include file name of training and testing (evaluation) set, maximum elements (key) in a sample, and label dimensions (see fig. 5).
* From v2.1, `Data` will be one of the layer in the layers list. So that the name of dense input and sparse input can be reference in the following layers. 
* All the nodes will share the same file list in training.
* `dense` and `sparse` should be configured (dense_dim should be 0 if no dense feature involved), where `dense` refers to the dense input and `sparse` refers to the `sparse` input. `sparse` should be an array here, since we support multiple `embedding` and each one requires a `sparse` input.
* The `type` of sparse input should be consistent with the following Embeddings.
* `slot_num` is the number of slots used in this training set. All the weight vectors get out of a slot will be reduced into one vector after embedding lookup (see Fig.3).
* The sum of `slot_num` in each sparse input should be consistent with the slot number defined in the header of training file.  
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

Embedding:
* Slots (tables or feature fields) are distributed in GPUs and nodes.
* `type`: two types of embedding are supported: `LocalizedSlotSparseEmbeddingHash`, `DistributedSlotSparseEmbeddingHash`.
  * `LocalizedSlotSparseEmbeddingHash`: each individual slot will be located in each GPU in turn, and not shared. This type of embedding has the best scalability.
    * `plan_file`: a plan file should be specified when use `LocalizedSlotSparseEmbeddingHash`. To generate a plan file please refer to the [**README**](samples/dcn/README.md) in dcn sample.
  * `DistributedSlotSparseEmbeddingHash`: Each GPU will has a portion of a slot. This type of embedding is useful when there exists the load imbalance among slots and potentially has OOM issue.
  * In single GPU training, for your convenience please use `DistributedSlotSparseEmbeddingHash`.
* `vocabulary_size`: the maximum possible size of embedding.
* `load_factor`: as embedding is implemented with hashtable, `load_factor` is the ratio of loaded vocabulary to capacity of the hashtable.
* `embedding_vec_size`: the vector size of an embedding weight (value). Then the memory used in this hashtable will be vocabulary_size*embedding_vec_size/load_factor.
* `combiner`: 0 is sum and 1 is mean.
* `optimizer`: (optional) from v2.1 HugeCTR supports different optimizers in dense and sparse models. You can specify your optimizer of this Embedding here. If not specified, HugeCTR will reuse the optimizer of dense model here.
```json
    {
      "name": "sparse_embedding1",
      "type": "LocalizedSlotSparseEmbeddingHash",
      "bottom": "data1",
      "top": "sparse_embedding1",
      "plan_file": "all2all_plan_bi_1.json",
      "sparse_embedding_hparam": {
        "vocabulary_size": 1737710,
        "load_factor": 0.75,
        "embedding_vec_size": 16,
        "combiner": 0
      },
      "optimizer": {
        "type": "Adam",
        "global_update": true,
        "adam_hparam": {
          "alpha": 0.005,
          "beta1": 0.9,
          "beta2": 0.999,
          "epsilon": 0.000001
        }
      }
    }

```
Reshape: in v2.1 the first layer after embedding should be `Reshape` to reshape the tensor from 3D to 2D. Reshape is the only layer accept both 3D and 2D input and the output must be 2D.
`leading_dim` in `Reshape` is the leading dimension of the output.

Concat: you can `Concat` at most five tensors into one and list the name in `bottom` array. Note that the second dimension (usually batch size) should be the same.

Slice: opposite to concat, we support slice layer to copy specific `ranges` of input tensor to named output tensors. In the sample below, we duplicate input tensor with `Slice` (0 is inclusive, 429 is exclusive). 

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

The Others
* ELU: the type name is `ELU`, and a `elu_param` called `alpha` in it can be configured.
* Fully Connected (`InnerProduct`): bias is supported in fully connected layer and `num_output` is the dimension of output.
* Loss: different from the other layers, you can specify which `regularization` will you use. This is optional. By default no regularization will be used.
* For more details please refer to [**parser.cu**](HugeCTR/src/parser.cpp)
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
  "name": "loss",
  "type": "BinaryCrossEntropyLoss",
  "bottom": ["fc8","label"],
  "regularizer": "L2",
  "top": "loss"
}

```
## Data Format
A data set in HugeCTR includes an ASCII format file list and a set of data files in binary format to maximize the performance of data loading and minimize the storage. Note that data file is the minimum reading granularity for a reading thread, so at least 10 files in each file list are required for best performance.
### File List
A file list starts with a number which indicates the number of files in the file list, then comes with the path of each data file.
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
### Data File
A data file (binary) contains a header and the following data (many samples).

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
  long long*  keys; 
} Slot;
```

Data field often has a lot of samples. Each sample starts with the labels in integer type, followed by `nnz` (number of nonzero) and key in long long type (see Fig. 6).

<div align=center><img width = '1000' height ='150' src ="user_guide_src/fig10_data_field.png"/></div>
<div align=center>Fig. 6 Data Field</div>

### No Trained Parameters
Some of the layers will generate statistic result during training like Batch Norm. Such parameters are outputs of CTR training (called “no trained parameters”) and used in inference.

In HugeCTR such parameters will be written into a JSON format file along with weight during training.
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
## Performance
In this section, we test the scalability of HugeCTR and compare its performance and result with TensorFlow on NVIDIA V100 GPUs. In summary, we can achieve about 114x speedup over multi-thread Tensorflow CPU with only one V100 and generate almost the same loss curves for both evaluation and training (see Fig. 9).

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
The good scalability of HugeCTR as the number of active GPUs is increased, is mainly because of the high efficient data exchange and the three-stage processing pipeline. In this pipeline, we overlap the data reading from file, host to device data transaction (inter- and intra- node) and GPU training.  The following chart shows the scalability of HugeCTR with the configration of Batch Size=16384, Layers=7 on DGX1 Servers.
<div align=center><img width = '800' height ='400' src ="user_guide_src/fig12_multi_gpu_performance.PNG"/></div>
<div align=center>Fig. 7 Multi-GPU Performance of HugeCTR</div>

### TensorFlow
In the TensorFlow test case below, HugeCTR shows up to 114x speedup to a CPU server with TensorFlow with only one V100 GPU and almost the same loss curve.


<div align=center><img width = '800' height ='400' src ="user_guide_src/WDL.JPG"/></div>
<div align=center>Fig. 8 WDL performance and loss curve comparsion with TensorFlow v2.0 </div>


<div align=center><img width = '800' height ='400' src ="user_guide_src/DCN.JPG"/></div>
<div align=center>Fig. 9 WDL performance and loss curve comparsion with TensorFlow v2.0</div>

## Known Issues

* The auto plan file generator doesn't support to generate plan file for 1 GPU system. In this case, user need to manually create the json plan file with the following content:
` {"type": "all2all", "num_gpus": 1, "main_gpu": 0, "num_steps": 1, "num_chunks": 1, "plan": [[0, 0]], "chunks": [1]} ` and rename the json plan file to the name listed in the HugeCTR configuration file.
* For 2 GPU system, if there are 2 NVLinks between GPUs, then the auto plan file generator will print some warnings `RuntimeWarning: divide by zero encountered in true_divide`. This will not affact the generated json plan file.
* The current plan file generator doesn't suppprt the system that is only partially connected by NVLink. That is the system which has NVLink but exists 2 GPUs where data cannot travel through NVLink between them.
    + Systems such as DGX-1 where some GPU pairs don't have direct NVLink between them are supported because all the GPUs can reach each other by NVLink directly(through NVLink between them) or indirectly(through NVLink from other GPUs).
* If a system is not a fully-connected NVLink system which is a system where all GPU pairs has the same number of NVLink between them, then the maximum supported NVLinks between any two GPU is 2.
* User need to set environment variable: `export CUDA_DEVICE_ORDER=PCI_BUS_ID` to ensure that CUDA runtime and driver have consistent ordering of GPUs.


## Reference
[1] Wide and Deep Learning: https://arxiv.org/abs/1606.07792

[2] Deep Cross Network: https://arxiv.org/abs/1708.05123

[3] DeepFM: https://arxiv.org/abs/1703.04247 

[4] Gossip: https://github.com/Funatiq/gossip

[5] CriteoLabs: http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/





