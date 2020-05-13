# Questions and Answers #

### 1. Who are the target users of HugeCTR? ###
We are trying to provide a fast and dedicated framework to the industry users who requires high-efficiency solution of their online or offline CTR training. 
HugeCTR is also a reference design for the framework developers to help them enhance their current GPU CTR training solutions.
### 2. Which models can be supported in HugeCTR? ###
HugeCTR v2.1 is flexible. We support DNN / WDL / DCN / DeepFM models which are widely used in industrial recommender systems. The users can find useful samples from “samples” [folder](../samples) of HugeCTR 
### 3. Does HugeCTR support TensorFlow? ###
HugeCTR v2.1 has no TF interface, but the Trained model is compatible with TensorFlow. It's encouraged to deploy the trained model to TensorFlow for inference. See more details in [tutorial](../tutorial/dump_to_tf) 
### 4. Does HugeCTR support multiple nodes CTR training? ###
Yes. HugeCTR supports single-gpu, or multi-gpu single-node, or multi-gpu multi-node for CTR training. Please refer to samples/dcn2node for more details.
### 5. How to deal with the huge embedding table that cannot be stored in a single GPU memory? ###
Embedding table in HugeCTR is model-parallel stored across GPUs and nodes.  So if you have very large size of embedding table, just use as many GPUs as you need to store it. That’s why we have the name “HugeCTR”. Suppose you have 1TB embedding table and 16xV100-32GB in a GPU server node, you can take 2 nodes for such case. 
### 6. Which GPUs are supported in HugeCTR? ###
HugeCTR supports GPUs with Compute Compatibility > 6.0, for example P100, P4, P40, P6, V100, T4.
### 7. Is DGX the only GPU server that is required in HugeCTR? ###
DGX is not required, but recommended, because the performance of CTR training highly relies on the performance of inter-GPUs transactions. DGX has NVLink and NVSwitch inside, so that you can expect 150GB/s per direction per GPU. It’s 9.3x to PCI-E 3.0.
### 8. Can HugeCTR run without InfiniBand?###
For multi-node training, InfiniBand is recommended but not required. You can use any solution with UCX support. InfiniBand with GPU RDMA support will maximize performance of inter-node transactions.
### 9. Is there any requirement of CPU configuration for HugeCTR execution? ###
HugeCTR has very low CPU requirements, because it offload almost all the computation to GPUs where CPU is only used in file reading.
### 10.	What is the specific format of files as input in HugeCTR? ###
We have specific file format support. Please refer to the [tutorial](../tutorial/dump_to_tf) in HugeCTR.
### 11.	Does HugeCTR support Python interface? ###
Not currently. We will consider to add Python interface in the future version.
### 12. Does HugeCTR do synchronous training or asynchronous training? ###
HugeCTR only supports synchronous training.
### 13.	Does HugeCTR support stream training? ###
Yes, hashtable based embedding in HugeCTR supports dynamic insertion, which is designed for stream training. New features can be added into embedding in runtime. 
HugeCTR also support data check. Error data will be skipped in training.
### 14. What is a “slot” in HugeCTR? ###
In HugeCTR, slot is feature field or table. The features in a slot can be one-hot or multi-hot. Number of features in different slots can be variant. There are `slot_num` of slots in one sample of training dataset. This item is configurable in data layer of configuration file.  
### 15.	What are the differences between LocalizedSlotEmbedding and DistributedSlotEmbedding? ###
There are two sub-classes of Embedding layer, LocalizedSlotEmbedding and DistributedSlotEmbedding. They are distinguished by different method of distributing embedding table on multiple GPUs as model parallelism. For LocalizedSlotEmbedding, the features in the same slot will be stored in one GPU (that is why we call it “localized slot”), and different slots may be stored in different GPUs according to the index number of the slot; For DistributedSlotEmbedding, all the features are distributed to different GPUs according to the index number of the feature, regardless of the index number of the slot. That means the features in the same slot may be stored in different GPUs (that is why we call it “distributed slot”).
### 16. For multi-node，is DataReader required to read the same batch of data on each node for each step? ### 
Yes, each node in training will read the same data in each iteration. 
### 17.	As model parallelism in embedding layer, how does it get all the embedding lookup features from multi-node / multi-gpu? ###
After embedding lookup, the embedding features in one slot need to be combined (or reduced) into one embedding vector. There are 2 steps: 1) local reduction in single GPU in forward kernel function; 2) global reduction across multi-node / multi-gpu by collective communications libraries such as NCCL. 
### 18.	How to set data clauses, if there are two embeddings needed? ### 
There should only be one source where the "sparse" is an array. Suppose there are 26 features (slots), first 13 features belong to the first embedding and the last 13 features belong to the second embedding, you can have two elements in "sparse" array as below: 
```
"sparse": [
{
 "top": "data1",
 "type": "DistributedSlot",
 "max_feature_num_per_sample": 30,
 "slot_num": 13
},
{
 "top": "data2",
 "type": "DistributedSlot",
 "max_feature_num_per_sample": 30,
 "slot_num": 13
}
]
```
### 19.	How to save and load model in HugeCTR? ### 
In HugeCTR, the model is saved in binary raw format. For model saving, you can set the “snapshot” in .json file to set the intervals of saving a checkpoint in file with the prefix of “snapshot_prefix”; For model loading, you can just modify the “dense_model_file”, “sparse_model_file” in .json file (in solver clause) according to the name of the snapshot.
### 20.	Could the post training model from HugeCTR be imported into other frameworks such as TensorFlow for inference deployment? ### 
Yes. The training model in HugeCTR is saved in raw format, and you can import it to other frameworks by writing some scripts . We provide a tutorial to demonstrate how to import HugeCTR post training model to TensorFlow. Please refer to the [tutorial](../tutorial/dump_to_tf) .
### 21. Does HugeCTR support overlap between different slots? ###
Features in different slots must be unique (no overlap). You may want to preprocess the data if you have overlaps e.g. offset or use hash function.
### 22. What if there's no value in a slot? ###
nnz=0 is supported in HugeCTR input. That means no features will be looked up.
### 23. How can I benchmark my network? ###
Firstly, you should construct your own configure file. You can refer to our [User Guide](hugectr_user_guide.md) and samples. Secondly, using our `data_generator` to generate a random dataset. Seeing [introductions](../README.md#benchmark).
Thirdly, run with `./huge_ctr --train ./your_config.json`
### 24. What is "plan_file"? How to provide it?  ###
Plan_file is used by Gossip communication library. In this file, the GPUs topology and connection in the server is defined. Gossip needs the plan_file as input to optimize the collective communication efficiency among GPUs. We provide a plan_file generator [tool](../tools/plan_generation) for the user to generate plan_file easily based on your server configuration. And there are samples of how to use the tool to generate plan_file, please refer to [dcn](../samples/dcn) or [dcn2nodes](../samples/dcn2nodes). Please note that if you don't want to use Gossip library but NCCL library to perform all2all communication, you don't need to provide a plan_file. Please refer to our [README](../README) "Build with NCCL All2All Supported".
### 25. How to set vocabulary_size and load_fator in .json file?  ###
vocabulary_size reprensents the max number of features in hashtable / embedding_table. 
In DistributedSlotEmbedding, we suppose the ditribution of features(keys) is uniform distribution, and we will allocate the same memory space on GPUs as max_vocabulary_size_per_gpu=vocabulary_size/gpu_num. So if the feature distuibution is nonuniform, and the number of features on each GPU is very different, the users need to set a big enough value of vocabulary_size in order to guarantee there wiil not be overflow for each GPU. 
In LocalizedSlotEmbedding, we suppose the each slot has the same number of features. If the size of each slot is very different, we recommend the users give a big enough number since we will calculate the max_vocabulary_size_per_gpu according to this [formula](../HugeCTR/include/embeddings/localized_slot_sparse_embedding_hash.hpp#L255). 
load_factor is a factor for GPU hashtable, and the size of hashtable will be vocabualry_size/load_factor after you set this value to < 1. load_factor is a very important factor that can affect the performance of hashtable get() / insert() operation very much. We recommend to set this value to 0.75 ad default to get better performance and suitable redundant memory allocation according to our benchmark results. 
### 26. Is GPU with nvlink that is required in HugeCTR? ###
GPU with nvlink is not required, but recommended, because the performance of CTR training highly relies on the performance of inter-GPUs communication. GPUs with PCIE is also supported in HugeCTR.