# Questions and Answers

## 1. Who are the target users of HugeCTR?

We are trying to provide a recommender specific framework to users from various industries, who need high-efficient solutions for their online/offline CTR training.
HugeCTR is also a reference design for framework developers who want to port their CPU solutions to GPU or optimize their current GPU solutions.

## 2. Which models can be supported in HugeCTR?

HugeCTR v2.2 supports DNN, WDL, DCN, DeepFM, DLRM and their variants, which are widely used in industrial recommender systems.
Refer to the [samples](https://github.com/NVIDIA-Merlin/HugeCTR/tree/master/samples) directory in the HugeCTR repository on GitHub to try them with HugeCTR.
HugeCTR's expressiveness is not confined to the aforementioned models.
You can construct your own models by combining the layers supported by HugeCTR.

## 3. Does HugeCTR support TensorFlow?

HugeCTR v2.2 has no TF interface yet, but a HugeCTR Trained model is compatible with TensorFlow.
We recommend that you export a trained model to TensorFlow for inference by following the instructions in dump_to_tf tutorial that is in the [tutorial](https://github.com/NVIDIA-Merlin/HugeCTR/tree/master/tutorial) directory of the HugeCTR repository on GitHub.

## 4. Does HugeCTR support multiple nodes CTR training?

Yes. HugeCTR supports single-GPU, multi-GPU and multi-node training. Check out samples/dcn2node for more details.

## 5. How to deal with the huge embedding table that cannot be stored in a single GPU memory?

Embedding table in HugeCTR is model-parallel stored across GPUs and nodes.  So if you have a very large embedding table, just use as many GPUs as you need to store it. That’s why we have the name “HugeCTR”. Suppose you have 1TB embedding table and 16xV100-32GB in a GPU server node, you can take 2 nodes for such a case.

## 6. Which GPUs are supported in HugeCTR?

HugeCTR supports GPUs with Compute Compatibility > 7.0 such as V100, T4 and A100.

## 7. Must we use the DGX family such as DGX A100 to run HugeCTR?

A DGX machine is not mandatory but recommended to achieve the best performance by exploiting NVSwitch's high inter-GPU bandwidth.

## 8. Can HugeCTR run without InfiniBand?

For multi-node training, InfiniBand is recommended but not required. You can use any solution with UCX support.
However, InfiniBand with GPU RDMA support will maximize performance of inter-node transactions.

## 9. Is there any requirement of CPU configuration for HugeCTR execution?

HugeCTR's approach is to offload the computational workloads to GPUs with the memory operations overlapped with them.
So HugeCTR performance is mainly decided by what kinds of GPUs and I/O devices are used.

## 10. What is the specific format of files as input in HugeCTR?

We have specific file format support.
Refer to the [Dataset formats](./api/python_interface.md#dataset-formats) section of the Python API documentation.

## 11.	 Does HugeCTR support Python interface?

Yes we introduced our first version of Python interface.
Check out our [example notebooks](/hugectr_example_notebooks) and Python [API documentation](./api/python_interface.md).

## 12. Does HugeCTR do synchronous training with multiple GPUs (and nodes)? Otherwise, does it do asynchronous training?

HugeCTR only supports synchronous training.

## 13. Does HugeCTR support stream training?

Yes, hashtable based embedding in HugeCTR supports dynamic insertion, which is designed for stream training. New features can be added into embedding in runtime.
HugeCTR also supports data check. Error data will be skipped in training.

## 14. What is a “slot” in HugeCTR?

In HugeCTR, a slot is a feature field or table.
The features in a slot can be one-hot or multi-hot.
The number of features in different slots can be various.
You can specify the number of slots (`slot_num`) in the data layer of your configuration file.

## 15. What are the differences between LocalizedSlotEmbedding and DistributedSlotEmbedding?

There are two sub-classes of Embedding layer, LocalizedSlotEmbedding and DistributedSlotEmbedding.
They are distinguished by different methods of distributing embedding tables on multiple GPUs as model parallelism.
For LocalizedSlotEmbedding, the features in the same slot will be stored in one GPU (that is why we call it “localized slot”), and different slots may be stored in different GPUs according to the index number of the slot.
For DistributedSlotEmbedding, all the features are distributed to different GPUs according to the index number of the feature, regardless of the index number of the slot.
That means the features in the same slot may be stored in different GPUs (that is why we call it “distributed slot”).

Thus LocalizedSlotEmbedding is optimized for the case each embedding is smaller than the memory size of GPU. As local reduction per slot is used in LocalizedSlotEmbedding and no global reduce between GPUs, the overall data transaction in Embedding is much less than DistributedSlotEmbedding. DistributedSlotEmbedding is made for the case some of the embeddings are larger than the memory size of GPU. As global reduction is required. DistributedSlotEmbedding has much more memory trasactions between GPUs.

## 16. For multi-node，is DataReader required to read the same batch of data on each node for each step?

Yes, each node in training will read the same data in each iteration.

## 17. As model parallelism in embedding layers, how does it get all the embedding lookup features from multi-node / multi-gpu?

After embedding lookup, the embedding features in one slot need to be combined (or reduced) into one embedding vector.
There are 2 steps:

1) local reduction in single GPU in forward kernel function;
2) global reduction across multi-node / multi-gpu by collective communications libraries such as NCCL.

## 18. How to set data clauses, if there are two embeddings needed?

There should only be one source where the "sparse" is an array. Suppose there are 26 features (slots), first 13 features belong to the first embedding and the last 13 features belong to the second embedding, you can have two elements in "sparse" array as below:

```json
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

## 19. How to save and load models in HugeCTR?

In HugeCTR, the model is saved in binary raw format. For model saving, you can set the “snapshot” in .json file to set the intervals of saving a checkpoint in file with the prefix of “snapshot_prefix”; For model loading, you can just modify the “dense_model_file”, “sparse_model_file” in .json file (in solver clause) according to the name of the snapshot.

## 20. Could the post training model from HugeCTR be imported into other frameworks such as TensorFlow for inference deployment?

Yes. The training model in HugeCTR is saved in raw format, and you can import it to other frameworks by writing some scripts.
We provide a tutorial to demonstrate how to import the HugeCTR trained model to TensorFlow.
Refer to the dump_to_tf tutorial in the [tutorial](https://github.com/NVIDIA-Merlin/HugeCTR/tree/master/tutorial) directory of the HugeCTR repository on GitHub.

## 21. Does HugeCTR support overlap between different slots?

Features in different slots must be unique (no overlap). You may want to preprocess the data if you have overlaps e.g. offset or use hash function.

## 22. What if there's no value in a slot?

nnz=0 is supported in HugeCTR input. That means no features will be looked up.

## 23. How can I benchmark my network?

Firstly, you should construct your own configure file. You can refer to our [User Guide](hugectr_user_guide.md) and samples.
Secondly, using our `data_generator` to generate a random dataset. See the [Getting Started](https://github.com/NVIDIA-Merlin/HugeCTR#getting-started) section of the HugeCTR repository README for an example.
Thirdly, run with `./huge_ctr --train ./your_config.json`

## 24. How to set workspace_size_per_gpu_in_mb and slot_size_array?

As embeddings are model parallel in HugeCTR,
`workspace_size_per_gpu_in_mb` is a reference number for HugeCTR to allocate GPU memory accordingly and not necessarily the exact number of features in your dataset. It is depending on vocabulary size per gpu, embedding vector size and optimizer type.
Refer to the embedding workspace calculator in the [tools](https://github.com/NVIDIA-Merlin/HugeCTR/tree/master/tools) directory of the HugeCTR repository on GitHub.
Use the calculator to calculate the vocabulary size per GPU and workspace_size per GPU for different embedding types, embedding vector size and optimizer type.

In practice, we usually set it larger than the real size because of the non-uniform distribution of the keys.

`slot_size_array` has 2 usages. It can be used as a replacement for `workspace_size_per_gpu_in_mb` to avoid wasting memory caused by imbalance vocabulary size. And it can also be used as a reference to add offset for keys in different slot.

The relation between embedding type, `workspace_size_per_gpu_in_mb` and `slot_size_array` is:

* For `DistributedSlotEmbedding`, `workspace_size_per_gpu_in_mb` is needed and `slot_size_array` is not needed. Each GPU will allocate the same amount of memory for embedding table usage.
* For `LocalizedSlotSparseEmbeddingHash`, only one of `workspace_size_per_gpu_in_mb` and `slot_size_array` is needed. If users can provide the exact size for each slot, we recommand users to specify `slot_size_array`. It can help avoid wasting memory caused by imbalance vocabulary size. Or you can specify `workspace_size_per_gpu_in_mb` so each GPU will allocate the same amount of memory for embedding table usage. If you specify both `slot_size_array` and `workspace_size_per_gpu_in_mb`, HugeCTR will use `slot_size_array` for `LocalizedSlotSparseEmbeddingHash`.
* For `LocalizedSlotSparseEmbeddingOneHot`, `slot_size_array` is needed. It is used for allocating memory and adding offset for each slot.
* For `HybridSparseEmbedding`, both `workspace_size_per_gpu_in_mb` and `slot_size_array` is needed. `workspace_size_per_gpu_in_md` is used for allocating memory while `slot_size_array` is used for adding offset

## 25. Is nvlink required in HugeCTR?

GPU with nvlink is not required, but recommended because the performance of CTR training highly relies on the performance of inter-GPUs communication. GPU servers with PCIE connections are also supported.

## 26. Is DGX the only GPU server that is required in HugeCTR?

DGX is not required, but recommended, because the performance of CTR training highly relies on the performance of inter-GPUs transactions. DGX has NVLink and NVSwitch inside, so that you can expect 150GB/s per direction per GPU. It’s 9.3x to PCI-E 3.0.

## 27. Can HugeCTR run without InfiniBand?

For multi-node training, InfiniBand is recommended but not required. You can use any solution with UCX support. InfiniBand with GPU RDMA support will maximize performance of inter-node transactions.

## 28. Does HugeCTR support loading pretrained embeddings in other formats?

You can convert the pretrained embeddings to the HugeCTR sparse models and then load them to facilitate the training process. You can refer to [save_params_to_files](/api/python_interface.md#save-params-to-files-method) to get familiar with the HugeCTR sparse model format. We demonstrate the usage in 3.4 Load Pre-trained Embeddings of [hugectr_criteo.ipynb](../notebooks/hugectr_criteo.ipynb).

## 29. How to construct the model graph with branch topology in HugeCTR?

The branch topology is inherently supported by HugeCTR model graph, and extra layers are abstracted away in HugeCTR Python Interface.
Refer to the [Slice Layer](/api/hugectr_layer_book.md#slice-layer) for information about model graphs with branches and sample code.

## 30. What is the good practice of configuring the embedding vector size?

The embedding vector size is related to the size of Cooperative Thread Array (CTA) for HugeCTR kernal launching, so first and foremost it should not exceed the maximum number of threads per block. It would be better that it is configured to a multiple of the warp size for the sake of occupancy. Still, you can set the embedding vector size freely according to the specific model architecture as long as it complies with the limit.
