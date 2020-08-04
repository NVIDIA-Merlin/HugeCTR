# Questions and Answers #

### 1. Who are the target users of HugeCTR? ###
We are trying to provide a recommender specific framework to users from various industries, who needs high-efficient solution of their online/offline CTR training. 
HugeCTR is also a reference design for framework developers who want to port their CPU solutions to GPU or optimize their current GPU solutions.
### 2. Which models can be supported in HugeCTR? ###
HugeCTR v2.2 supports DNN, WDL, DCN, DeepFM, DLRM and theier vairants, which are widely used in industrial recommender systems.
See [samples](../samples) [folder] to try them with HUgeCTR.
HugeCTR's expressiveness is not confined to the aforementioned models.
You can construct your own models by combining the layers supported by HugeCTR.
### 3. Does HugeCTR support TensorFlow? ###
HugeCTR v2.2 has no TF interface yet, but a HugeCTR Trained model is compatible with TensorFlow.
We recommend that you export a trained model to TensorFlow for inference by following the instructions in our [tutorial](../tutorial/dump_to_tf) 
### 4. Does HugeCTR support multiple nodes CTR training? ###
Yes. HugeCTR supports single-GPU, multi-GPU and multi-node training. Cehck out samples/dcn2node for more details.
### 5. How to deal with the huge embedding table that cannot be stored in a single GPU memory? ###
Embedding table in HugeCTR is model-parallel stored across GPUs and nodes.  So if you have very large size of embedding table, just use as many GPUs as you need to store it. That’s why we have the name “HugeCTR”. Suppose you have 1TB embedding table and 16xV100-32GB in a GPU server node, you can take 2 nodes for such case. 
### 6. Which GPUs are supported in HugeCTR? ###
HugeCTR supports GPUs with Compute Compatibility > 7.0 such as V100, T4 and A100.
### 7. Must we use the DGX family such as DGX A100 to run HugeCTR? ###
A DGX machine is not mandatory but recommended to achieve the best performance by exploiting NVSwitch's high inter-GPU bandwidth. 
### 8. Can HugeCTR run without InfiniBand? ###
For multi-node training, InfiniBand is recommended but not required. You can use any solution with UCX support.
However, InfiniBand with GPU RDMA support will maximize performance of inter-node transactions.
### 9. Is there any requirement of CPU configuration for HugeCTR execution? ###
HugeCTR's approach is to offload the computational workloads to GPUs with the memory operations overlapped with them.
So HugeCTR performance is mainly decided by what kinds of GPUs and I/O devices are used.
### 10.	What is the specific format of files as input in HugeCTR? ###
We have specific file format support. Please refer to the [tutorial](../tutorial/dump_to_tf) in HugeCTR.
### 11.	Does HugeCTR support Python interface? ###
Not currently. We will consider to add Python interface in the future version.
### 12. Does HugeCTR do synchronous training with multiple GPUs (and nodes)? Otherwise, does it do asynchronous training? ###
HugeCTR only supports synchronous training.
### 13.	Does HugeCTR support stream training? ###
Yes, hashtable based embedding in HugeCTR supports dynamic insertion, which is designed for stream training. New features can be added into embedding in runtime. 
HugeCTR also support data check. Error data will be skipped in training.
### 14. What is a “slot” in HugeCTR? ###
In HugeCTR, slot is feature field or table.
The features in a slot can be one-hot or multi-hot.
The number of features in different slots can be various.
You can specify the number of slots (`slot_num`) in data layer of your configuration file.
### 15.	What are the differences between LocalizedSlotEmbedding and DistributedSlotEmbedding? ###
There are two sub-classes of Embedding layer, LocalizedSlotEmbedding and DistributedSlotEmbedding.
They are distinguished by different method of distributing embedding table on multiple GPUs as model parallelism.
For LocalizedSlotEmbedding, the features in the same slot will be stored in one GPU (that is why we call it “localized slot”), and different slots may be stored in different GPUs according to the index number of the slot.
For DistributedSlotEmbedding, all the features are distributed to different GPUs according to the index number of the feature, regardless of the index number of the slot.
That means the features in the same slot may be stored in different GPUs (that is why we call it “distributed slot”).
### 16. For multi-node，is DataReader required to read the same batch of data on each node for each step? ### 
Yes, each node in training will read the same data in each iteration. 
### 17.	As model parallelism in embedding layer, how does it get all the embedding lookup features from multi-node / multi-gpu? ###
After embedding lookup, the embedding features in one slot need to be combined (or reduced) into one embedding vector.
There are 2 steps:
1) local reduction in single GPU in forward kernel function;
2) global reduction across multi-node / multi-gpu by collective communications libraries such as NCCL. 
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
Firstly, you should construct your own configure file. You can refer to our [User Guide](hugectr_user_guide.md) and samples.
Secondly, using our `data_generator` to generate a random dataset. Seeing [introductions](../README.md#benchmark).
Thirdly, run with `./huge_ctr --train ./your_config.json`
### 24. What is "plan_file"? How to provide it?  ###
Plan_file is used by Gossip communication library. the GPUs topology and connection in the server are defined in this file. We provide a plan_file generator [tool](../tools/plan_generation) for users to generate plan_file easily based on your server configuration. Please refer to [dcn](../samples/dcn) or [dcn2nodes](../samples/dcn2nodes). Please note that if you prefer NCCL library other than Gossip here you don't have to provide a plan_file. Please refer to our [README](../README.md) "Build with NCCL All2All Supported".
### 25. How to set max_vocabulary_size_per_gpu and slot_size_array in .json file? ###
As embeddings are model parallel in HugeCTR,
it's a reference number for HugeCTR to allocate GPU memory accordingly and not necessarily the exact number of features in your dataset.
In practice, we usually set it larger than the real size because of the non-uniform distribution of the keys.
In DistributedSlotEmbedding, HugeCTR will allocate the same size of memory on each GPU which is `max_vocabulary_size_per_gpu`.
Users have to set this parameter big enough to make sure no overflow in any GPUs.
In LocalizedSlotEmbedding, user can also provide `max_vocabulary_size_per_gpu`,
if the slot sizes are significantly different, we recommend that user give a number large enough to prevent overflow.
Another approach for LocalizedSlotEmbedding is that users can provide the exact size for each slot,
which is `slot_size_array` and HugeCTR will calculate the `max_vocabulary_size_per_gpu` according to the given slot sizes.
### 26. Is nvlink required in HugeCTR? ###
GPU with nvlink is not required, but recommended because the performance of CTR training highly relies on the performance of inter-GPUs communication. GPU servers with PCIE connections are also supported.
### 27. Is DGX the only GPU server that is required in HugeCTR? ###
DGX is not required, but recommended, because the performance of CTR training highly relies on the performance of inter-GPUs transactions. DGX has NVLink and NVSwitch inside, so that you can expect 150GB/s per direction per GPU. It’s 9.3x to PCI-E 3.0.
