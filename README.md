# <img src="docs/user_guide_src/merlin_logo.png" alt="logo" width="85"/> Merlin: HugeCTR #
[![v30](docs/user_guide_src/version.JPG)](release_notes.md)

HugeCTR is a GPU-accelerated recommender framework designed to distribute training across multiple GPUs and nodes and estimate Click-Through Rates (CTRs). HugeCTR supports model-parallel embedding tables and data-parallel neural networks and their variants such as [Deep Interest Network (DIN)](https://arxiv.org/pdf/1706.06978.pdf), [NCF](https://arxiv.org/abs/1708.05031), [Wide and Deep Learning (WDL)](https://arxiv.org/abs/1606.07792), [Deep Cross Network (DCN)](https://arxiv.org/abs/1708.05123), [DeepFM](https://arxiv.org/abs/1703.04247), and [Deep Learning Recommendation Model (DLRM)](https://arxiv.org/abs/1906.00091). HugeCTR is a component of [NVIDIA Merlin Open Beta](https://developer.nvidia.com/nvidia-merlin), which is used to build large-scale deep learning recommender systems. For more information, refer to [HugeCTR User Guide](docs/hugectr_user_guide.md).

Design Goals:
* **Fast**: HugeCTR is a speed-of-light CTR model framework that can [outperform](performance.md) popular recommender systems such as TensorFlow (TF).
* **Efficient**: HugeCTR provides the essentials so that you can efficiently train your CTR model.
* **Easy**: Regardless of whether you are a data scientist or machine learning practitioner, we've made it easy for anybody to use HugeCTR.

## Table of Contents
* [Core Features](#core-features)
* [Getting Started](#getting-started)
* [HugeCTR SDK](#hugectr-sdk)
* [Support and Feedback](#support-and-feedback)
* [Contributing to HugeCTR](#contributing-to-hugectr)
* [Additional Resources](#additional-resources)

## Core Features ##
HugeCTR supports a variety of features, including the following:
* [High-Level abstracted Python interface](docs/python_interface.md)
* [Model parallel training](docs/hugectr_user_guide.md#model-parallel-training)
* [Optimized GPU workflow](performance.md)
* [Multi-node training](docs/hugectr_user_guide.md#multi-node-training)
* [Mixed precision training](docs/hugectr_user_guide.md#mixed-precision-training)
* [Embedding training cache](docs/hugectr_user_guide.md#embedding-training-cache)
* [GPU embedding cache](https://github.com/triton-inference-server/hugectr_backend/blob/main/docs/architecture.md#enabling-the-gpu-embedding-cache)
* [GPU / CPU memory sharing mechanism across various inference instances](https://github.com/triton-inference-server/hugectr_backend/blob/main/docs/architecture.md#hugectr-backend-framework)
* [HugeCTR to ONNX Converter](docs/hugectr_user_guide.md#onnx-converter)
* [Hierarchical Parameter Server](docs/hugectr_user_guide.md#hierarchical-parameter-server)

To learn about our latest enhancements, refer to our [release notes](release_notes.md).

## Getting Started ##
If you'd like to quickly train a model using the Python interface, do the following:

1. Start a NGC container with your local host directory (/your/host/dir mounted) by running the following command:
   ```
   docker run --gpus=all --rm -it --cap-add SYS_NICE -v /your/host/dir:/your/container/dir -w /your/container/dir -it -u $(id -u):$(id -g) nvcr.io/nvidia/merlin/merlin-
   training:21.11
   ```

   **NOTE**: The **/your/host/dir** directory is just as visible as the **/your/container/dir** directory. The **/your/host/dir** directory is also your starting directory.

2. Activate the Merlin conda environment by running the following command:  
   ```shell.
   source activate merlin
   ```

3. Write a simple Python script to generate a synthetic dataset:
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
     slot_size_array = [39884, 39043, 17289, 7420, 20263, 3, 7120, 1543, 39884, 39043, 17289, 7420, 20263, 3, 7120, 1543, 63, 63, 39884, 39043, 17289, 7420, 20263, 3, 7120, 
     1543],
     check_type = hugectr.Check_t.Sum,
     dist_type = hugectr.Distribution_t.PowerLaw,
     power_law_type = hugectr.PowerLaw_t.Short)
   data_generator = DataGenerator(data_generator_params)
   data_generator.generate()
   ```

4. Generate the Norm dataset for your DCN model by running the following command:
   ```
   python dcn_norm_generate.py
   ```
   **NOTE**: The generated dataset will reside in the folder `./dcn_norm`, which contains training and evaluation data.

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
   model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.MultiCross,
                              bottom_names = ["concat1"],
                              top_names = ["multicross1"],
                              num_layers=6))
   model.add(hugectr.DenseLayer(layer_type = hugectr.Layer_t.InnerProduct,
                              bottom_names = ["concat1"],
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
   **NOTE**: Ensure that the paths to the synthetic datasets are correct with respect to this Python script. `data_reader_type`, `check_type`, `label_dim`, `dense_dim`, and 
   `data_reader_sparse_param_array` should be consistent with the generated dataset.

6. Train the model by running the following command:
   ```
   python dcn_norm_train.py
   ```
   **NOTE**: It is presumed that the evaluation AUC value is incorrect since randomly generated datasets are being used. When the training is done, files that contain the 
   dumped graph JSON, saved model weights, and optimizer states will be generated.

For more information, refer to the [HugeCTR User Guide](docs/hugectr_user_guide.md).

## HugeCTR SDK ##
We're able to support external developers who can't use HugeCTR directly by exporting important HugeCTR components using:
* [Sparse Operation Kit](sparse_operation_kit): a python package wrapped with GPU accelerated operations dedicated for sparse training/inference cases.
* [GPU Embedding Cache](gpu_cache): embedding cache available on the GPU memory designed for CTR inference workload.

## Support and Feedback ##
If you encounter any issues or have questions, go to [https://github.com/NVIDIA/HugeCTR/issues](https://github.com/NVIDIA/HugeCTR/issues) and submit an issue so that we can provide you with the necessary resolutions and answers. To further advance the HugeCTR Roadmap, we encourage you to share all the details regarding your recommender system pipeline using this [survey](https://developer.nvidia.com/merlin-devzone-survey).

## Contributing to HugeCTR ##
With HugeCTR being an open source project, we welcome contributions from the general public. With your contributions, we can continue to improve HugeCTR's quality and performance. To learn how to contribute, refer to our [HugeCTR Contributor Guide](docs/hugectr_contributor_guide.md).

## Additional Resources ##
|Webpages|
|--------|
|[NVIDIA Merlin](https://developer.nvidia.com/nvidia-merlin)|
|[NVIDIA HugeCTR](https://developer.nvidia.com/nvidia-merlin/hugectr)|

### Talks ###
|Conference / Website|Title|Date|Speaker|Language|
|--------------------|-----|----|-------|--------|
|APSARA 2021|[GPU 推荐系统 Merlin](https://yunqi.aliyun.com/2021/agenda/session205?spm=5176.23948577a2c4e.J_6988780170.27.5ae7379893BcVp)|Oct 2021|Joey Wang|中文|
|GTC Spring 2021|[Learn how Tencent Deployed an Advertising System on the Merlin GPU Recommender Framework](https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31820/)|April 2021|Xiangting Kong, Joey Wang|English|
|GTC Spring 2021|[Merlin HugeCTR: Deep Dive Into Performance Optimization](https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31269/)|April 2021|Minseok Lee|English|
|GTC Spring 2021|[Integrate HugeCTR Embedding with TensorFlow](https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31425/)|April 2021|Jianbing Dong|English|
|GTC China 2020|[MERLIN HUGECTR ：深入研究性能优化](https://on-demand-gtc.gputechconf.com/gtcnew/sessionview.php?sessionName=cns20516-merlin+hugectr+%ef%bc%9a%e6%b7%b1%e5%85%a5%e7%a0%94%e7%a9%b6%e6%80%a7%e8%83%bd%e4%bc%98%e5%8c%96)|Oct 2020|Minseok Lee|English|
|GTC China 2020|[性能提升 7 倍 + 的高性能 GPU 广告推荐加速系统的落地实现](https://on-demand-gtc.gputechconf.com/gtcnew/sessionview.php?sessionName=cns20483-%E6%80%A7%E8%83%BD%E6%8F%90%E5%8D%87+7+%E5%80%8D+%2B+%E7%9A%84%E9%AB%98%E6%80%A7%E8%83%BD+gpu+%E5%B9%BF%E5%91%8A%E6%8E%A8%E8%8D%90%E5%8A%A0%E9%80%9F%E7%B3%BB%E7%BB%9F%E7%9A%84%E8%90%BD%E5%9C%B0%E5%AE%9E%E7%8E%B0)|Oct 2020|Xiangting Kong|中文|
|GTC China 2020|[使用 GPU EMBEDDING CACHE 加速 CTR 推理过程](https://on-demand-gtc.gputechconf.com/gtcnew/sessionview.php?sessionName=cns20626-%E4%BD%BF%E7%94%A8+gpu+embedding+cache+%E5%8A%A0%E9%80%9F+ctr+%E6%8E%A8%E7%90%86%E8%BF%87%E7%A8%8B)|Oct 2020|Fan Yu|中文|
|GTC China 2020|[将 HUGECTR EMBEDDING 集成于 TENSORFLOW](https://on-demand-gtc.gputechconf.com/gtcnew/sessionview.php?sessionName=cns20377-%E5%B0%86+hugectr+embedding+%E9%9B%86%E6%88%90%E4%BA%8E+tensorflow)|Oct 2020|Jianbing Dong|中文|
|GTC Spring 2020|[HugeCTR: High-Performance Click-Through Rate Estimation Training](https://www.nvidia.com/en-us/on-demand/session/gtcsj20-s21455/)|March 2020|Minseok Lee, Joey Wang|English|
|GTC China 2019|[HUGECTR: GPU 加速的推荐系统训练](https://on-demand-gtc.gputechconf.com/gtcnew/sessionview.php?sessionName=cn9794-hugectr%3A+gpu+%E5%8A%A0%E9%80%9F%E7%9A%84%E6%8E%A8%E8%8D%90%E7%B3%BB%E7%BB%9F%E8%AE%AD%E7%BB%83)|Oct 2019|Joey Wang|中文|

### Blogs ### 
|Conference / Website|Title|Date|Authors|Language|
|--------------------|-----|----|-------|--------|
|NVIDIA Devblog|[Accelerating Embedding with the HugeCTR TensorFlow Embedding Plugin](https://developer.nvidia.com/blog/accelerating-embedding-with-the-hugectr-tensorflow-embedding-plugin/)|Sept 2021|Vinh Nguyen, Ann Spencer, Joey Wang and Jianbing Dong|English|
|medium.com|[Optimizing Meituan’s Machine Learning Platform: An Interview with Jun Huang](https://medium.com/nvidia-merlin/optimizing-meituans-machine-learning-platform-an-interview-with-jun-huang-7e046143131f)|Sept 2021|Sheng Luo and Benedikt Schifferer|English|
|medium.com|[Leading Design and Development of the Advertising Recommender System at Tencent: An Interview with Xiangting Kong](https://medium.com/nvidia-merlin/leading-design-and-development-of-the-advertising-recommender-system-at-tencent-an-interview-with-37f1eed898a7)|Sept 2021|Xiangting Kong, Ann Spencer|English|
|NVIDIA Devblog|[扩展和加速大型深度学习推荐系统 – HugeCTR 系列第 1 部分](https://developer.nvidia.com/zh-cn/blog/scaling-and-accelerating-large-deep-learning-recommender-systems-hugectr-series-part-1/)|June 2021|Minseok Lee|中文|
|NVIDIA Devblog|[使用 Merlin HugeCTR 的 Python API 训练大型深度学习推荐模型 – HugeCTR 系列第 2 部分](https://developer.nvidia.com/zh-cn/blog/training-large-deep-learning-recommender-models-with-merlin-hugectrs-python-apis-hugectr-series-part2/)|June 2021|Vinh Nguyen|中文|
|medium.com|[Training large Deep Learning Recommender Models with Merlin HugeCTR’s Python APIs — HugeCTR Series Part 2](https://medium.com/nvidia-merlin/training-large-deep-learning-recommender-models-with-merlin-hugectrs-python-apis-hugectr-series-69a666e0bdb7)|May 2021|Minseok Lee, Joey Wang, Vinh Nguyen and Ashish Sardana|English|
|medium.com|[Scaling and Accelerating large Deep Learning Recommender Systems — HugeCTR Series Part 1](https://medium.com/nvidia-merlin/scaling-and-accelerating-large-deep-learning-recommender-systems-hugectr-series-part-1-c19577acfe9d)|May 2021|Minseok Lee|English|
|IRS 2020|[Merlin: A GPU Accelerated Recommendation Framework](https://irsworkshop.github.io/2020/publications/paper_21_Oldridge_Merlin.pdf)|Aug 2020|Even Oldridge etc.|English|
|NVIDIA Devblog|[Introducing NVIDIA Merlin HugeCTR: A Training Framework Dedicated to Recommender Systems](https://developer.nvidia.com/blog/introducing-merlin-hugectr-training-framework-dedicated-to-recommender-systems/)|July 2020|Minseok Lee and Joey Wang|English|
