# <img src="docs/user_guide_src/merlin_logo.png" alt="logo" width="85"/> Merlin: HugeCTR #
[![v30](docs/user_guide_src/version.JPG)](release_notes.md)

HugeCTR is a GPU-accelerated recommender framework designed to distribute training across multiple GPUs and nodes and estimate Click-Through Rates (CTRs). HugeCTR supports model-parallel embedding tables and data-parallel neural networks and their variants such as [Deep Interest Network (DIN)](https://arxiv.org/pdf/1706.06978.pdf), [NCF](https://arxiv.org/abs/1708.05031), [Wide and Deep Learning (WDL)](https://arxiv.org/abs/1606.07792), [Deep Cross Network (DCN)](https://arxiv.org/abs/1708.05123), [DeepFM](https://arxiv.org/abs/1703.04247), and [Deep Learning Recommendation Model (DLRM)](https://arxiv.org/abs/1906.00091). HugeCTR is a component of [NVIDIA Merlin Open Beta](https://developer.nvidia.com/nvidia-merlin), which is used to build large-scale deep learning recommender systems. For additional information, see [HugeCTR User Guide](docs/hugectr_user_guide.md).

Design Goals:
* **Fast**: HugeCTR is a speed-of-light CTR model framework that can [outperform](performance.md) popular recommender systems such as TensorFlow (TF).
* **Efficient**: HugeCTR provides the essentials so that you can efficiently train your CTR model.
* **Easy**: Regardless of whether you are a data scientist or machine learning practitioner, we've made it easy for anybody to use HugeCTR.

## Table of Contents
* [Core Features](#core-features)
* [Getting Started](#getting-started)
* [Support and Feedback](#support-and-feedback)
* [Contribute to HugeCTR](#contribute-to-hugectr)

## Core Features ##
HugeCTR supports a variety of features, including the following:
* [high-level abstracted recsys specific user interface](docs/python_interface.md)
* [model parallel training](docs/hugectr_user_guide.md#model-parallel-training)
* [well optimized full GPU workflow](performance.md)
* [multi-node training](docs/hugectr_user_guide.md#multi-node-training)
* [mixed precision training](docs/hugectr_user_guide.md#mixed-precision-training)
* [embedding training cache](docs/hugectr_user_guide.md#embedding-training-cache)
* [caching of most frequent embedding for inference](https://github.com/triton-inference-server/hugectr_backend/blob/main/docs/architecture.md#enabling-the-gpu-embedding-cache)
* [GPU / CPU memory sharing mechanism across different inference instances](https://github.com/triton-inference-server/hugectr_backend/blob/main/docs/architecture.md#hugectr-backend-framework)
* [ONNX Converter](docs/hugectr_user_guide.md#onnx-converter)

To learn about our latest enhancements, see our [release notes](release_notes.md).

## Getting Started ##
If you'd like to quickly train a model using the Python interface, follow these steps:
1. Start a NGC container with your local host directory (/your/host/dir mounted) by running the following command:
   ```
   docker run --runtime=nvidia --rm -v /your/host/dir:/your/container/dir -w /your/container/dir -it -u $(id -u):$(id -g) -it nvcr.io/nvidia/merlin/merlin-training:0.6
   ```

   **NOTE**: The **/your/host/dir** directory is just as visible as the **/your/container/dir** directory. The **/your/host/dir** directory is also your starting directory.

2. Activate the merlin conda environment by running the following command:  
   ```shell.
   source activate merlin
   ```

3. Write a simple Python script to generate synthetic dataset:
   ```
   # dcn_norm_generate.py
   import hugectr
   from hugectr.tools import DataGeneratorParams, DataGenerator
   data_generator_params = DataGeneratorParams(
     format = hugectr.DataReaderType_t.Norm,
     label_dim = 1,
     dense_dim = 13,
     num_slot = 26,
     i64_input_key = False,
     source = "./dcn_norm/file_list.txt",
     eval_source = "./dcn_norm/file_list_test.txt",
     slot_size_array = [39884, 39043, 17289, 7420, 20263, 3, 7120, 1543, 39884, 39043, 17289, 7420, 20263, 3, 7120, 1543, 63, 63, 39884, 39043, 17289, 7420, 20263, 3, 7120, 1543],
     check_type = hugectr.Check_t.Sum,
     dist_type = hugectr.Distribution_t.PowerLaw,
     power_law_type = hugectr.PowerLaw_t.Short)
   data_generator = DataGenerator(data_generator_params)
   data_generator.generate()
   ```

4. Generate the Norm dataset for DCN model by running the following command:
   ```
   python dcn_norm_generate.py
   ```
   **NOTE**: The generated dataset will reside in the folder `./dcn_norm`, which includes both training data and evaluation data.

5. Write a simple Python script for training:
   ```
   # dcn_norm_train.py
   import hugectr
   from mpi4py import MPI
   solver = hugectr.CreateSolver(max_eval_batches = 1280,
                                 batchsize_eval = 1024,
                                 batchsize = 1024,
                                 lr = 0.001,
                                 vvgpu = [[0]],
                                 repeat_dataset = True)
   reader = hugectr.DataReaderParams(data_reader_type = hugectr.DataReaderType_t.Norm,
                                    source = ["./dcn_norm/file_list.txt"],
                                    eval_source = "./dcn_norm/file_list_test.txt",
                                    check_type = hugectr.Check_t.Sum)
   optimizer = hugectr.CreateOptimizer(optimizer_type = hugectr.Optimizer_t.Adam,
                                       update_type = hugectr.Update_t.Global)
   model = hugectr.Model(solver, reader, optimizer)
   model.add(hugectr.Input(label_dim = 1, label_name = "label",
                           dense_dim = 13, dense_name = "dense",
                           data_reader_sparse_param_array = 
                           [hugectr.DataReaderSparseParam("data1", 1, True, 26)]))
   model.add(hugectr.SparseEmbedding(embedding_type = hugectr.Embedding_t.DistributedSlotSparseEmbeddingHash, 
                              workspace_size_per_gpu_in_mb = 25,
                              embedding_vec_size = 16,
                              combiner = "sum",
                              sparse_embedding_name = "sparse_embedding1",
                              bottom_name = "data1",
                              optimizer = optimizer))
   model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Reshape,
                              bottom_names = ["sparse_embedding1"],
                              top_names = ["reshape1"],
                              leading_dim=416))
   model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Concat,
                              bottom_names = ["reshape1", "dense"], top_names = ["concat1"]))
   model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Slice,
                              bottom_names = ["concat1"],
                              top_names = ["slice11", "slice12"],
                              ranges=[(0,429),(0,429)]))
   model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.MultiCross,
                              bottom_names = ["slice11"],
                              top_names = ["multicross1"],
                              num_layers=6))
   model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                              bottom_names = ["slice12"],
                              top_names = ["fc1"],
                              num_output=1024))
   model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.ReLU,
                              bottom_names = ["fc1"],
                              top_names = ["relu1"]))
   model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Dropout,
                              bottom_names = ["relu1"],
                              top_names = ["dropout1"],
                              dropout_rate=0.5))
   model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.Concat,
                              bottom_names = ["dropout1", "multicross1"],
                              top_names = ["concat2"]))
   model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                              bottom_names = ["concat2"],
                              top_names = ["fc2"],
                              num_output=1))
   model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.BinaryCrossEntropyLoss,
                              bottom_names = ["fc2", "label"],
                              top_names = ["loss"]))
   model.compile()
   model.summary()
   model.graph_to_json(graph_config_file = "dcn.json")
   model.fit(max_iter = 5120, display = 200, eval_interval = 1000, snapshot = 5000, snapshot_prefix = "dcn")
   ```
   **NOTE**: Please make sure that the paths to the synthetic datasets are correct with respect to this Python script. Besides, `data_reader_type`, `check_type`, `label_dim`, `dense_dim` and `data_reader_sparse_param_array` should be consistent with the generated dataset.

6. Train the model by running the following command:
   ```
   python dcn_norm_train.py
   ```
   **NOTE**: It is expected that the value of evaluation AUC is not good given that randomly generated datasets are being used. When the training is done, you will see the files of dumped graph JSON, saved model weights and optimizer states.

For additional information, see the [HugeCTR User Guide](docs/hugectr_user_guide.md).

## Support and Feedback ##
If you encounter any issues and/or have questions, please file an issue [here](https://github.com/NVIDIA/HugeCTR/issues) so that we can provide you with the necessary resolutions and answers. To further advance the Merlin/HugeCTR Roadmap, we encourage you to share all the details regarding your recommender system pipeline using this [survey](https://developer.nvidia.com/merlin-devzone-survey).

## Contribute to HugeCTR ##
HugeCTR is an open source project, and we encourage you to join the development directly. All of your contributions will be appreciated and can help us to improve our quality and performance. Please find more about how to contribute and the developer specific instructions on our [HugeCTR Contributor Guide](docs/hugectr_contributor_guide.md)

