# [HugeCTR](README.md)

[![Version](https://img.shields.io/github/v/release/NVIDIA-Merlin/HugeCTR?color=orange)](release_notes.md/)
[![LICENSE](https://img.shields.io/github/license/NVIDIA-Merlin/HugeCTR)](https://github.com/NVIDIA-Merlin/HugeCTR/blob/main/LICENSE)
[![Documentation](https://img.shields.io/badge/documentation-blue.svg)](https://nvidia-merlin.github.io/HugeCTR/main/hugectr_user_guide.html)

HugeCTR is a GPU-accelerated recommender framework designed for training and inference of large deep learning models. 

Design Goals:
* **Fast**: HugeCTR performs outstandingly in recommendation [benchmarks](https://nvidia-merlin.github.io/HugeCTR/main/performance.html) including MLPerf.
* **Easy**: Regardless of whether you are a data scientist or machine learning practitioner, we've made it easy for anybody to use HugeCTR with plenty of [documents](https://nvidia-merlin.github.io/HugeCTR/main/hugectr_user_guide.html), [notebooks](https://github.com/NVIDIA-Merlin/HugeCTR/tree/main/notebooks) and [samples](https://github.com/NVIDIA-Merlin/HugeCTR/tree/main/samples).
* **Domain Specific**: HugeCTR provides the [essentials](https://github.com/NVIDIA-Merlin/HugeCTR#core-features), so that you can efficiently deploy your recommender models with very large embedding.

## Table of Contents
* [Core Features](https://nvidia-merlin.github.io/HugeCTR/main/hugectr_core_features.html)
* [Getting Started](https://nvidia-merlin.github.io/HugeCTR/main/hugectr_user_guide.html#installing-and-building-hugectr)
* [HugeCTR SDK](https://nvidia-merlin.github.io/HugeCTR/main/hugectr_user_guide.html#tools)
* [Support and Feedback](https://nvidia-merlin.github.io/HugeCTR/main/hugectr_contributor_guide.html)
* [Contributing to HugeCTR](https://nvidia-merlin.github.io/HugeCTR/main/hugectr_contributor_guide.html)
* [Additional Resources](https://nvidia-merlin.github.io/HugeCTR/main/additional_resources.html)

## Core Features ##
HugeCTR supports a variety of features, including the following:

* [High-Level abstracted Python interface](https://nvidia-merlin.github.io/HugeCTR/main/api/python_interface.html)
* [Model parallel training](https://nvidia-merlin.github.io/HugeCTR/main/hugectr_core_features.html#model-parallel-training)
* [Optimized GPU workflow](performance.md)
* [Multi-node training](https://nvidia-merlin.github.io/HugeCTR/main/hugectr_core_features.html#multi-node-training)
* [Mixed precision training](https://nvidia-merlin.github.io/HugeCTR/main/hugectr_core_features.html#mixed-precision-training)
* [Embedding training cache](https://nvidia-merlin.github.io/HugeCTR/main/hugectr_core_features.html#embedding-training-cache)
* [GPU embedding cache](https://github.com/triton-inference-server/hugectr_backend/blob/main/docs/architecture.md#enabling-the-gpu-embedding-cache)
* [GPU / CPU memory sharing mechanism across various inference instances](https://github.com/triton-inference-server/hugectr_backend/blob/main/docs/architecture.md#hugectr-backend-framework)
* [HugeCTR to ONNX Converter](https://nvidia-merlin.github.io/HugeCTR/main/hugectr_core_features.html#hugectr-to-onnx-converter)
* [Hierarchical Parameter Server](https://nvidia-merlin.github.io/HugeCTR/main/hugectr_core_features.html#hierarchical-parameter-server)
* [Sparse Operation Kit](https://github.com/NVIDIA-Merlin/HugeCTR/tree/main/sparse_operation_kit)


To learn about our latest enhancements, refer to our [release notes](release_notes.md).

## Getting Started ##
If you'd like to quickly train a model using the Python interface, do the following:

1. Start a NGC container with your local host directory (/your/host/dir mounted) by running the following command:
   ```
   docker run --gpus=all --rm -it --cap-add SYS_NICE -v /your/host/dir:/your/container/dir -w /your/container/dir -it -u $(id -u):$(id -g) nvcr.io/nvidia/merlin/merlin-hugectr:23.03
   ```

   **NOTE**: The **/your/host/dir** directory is just as visible as the **/your/container/dir** directory. The **/your/host/dir** directory is also your starting directory.

   **NOTE**: HugeCTR uses NCCL to share data between ranks, and NCCL may requires shared memory for IPC and pinned (page-locked) system memory resources. It is recommended that you increase these resources by issuing the following options in the `docker run` command.
   ```text
   -shm-size=1g -ulimit memlock=-1
   ```

2. Write a simple Python script to generate a synthetic dataset:
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

3. Generate the Norm dataset for your DCN model by running the following command:
   ```
   python dcn_norm_generate.py
   ```
   **NOTE**: The generated dataset will reside in the folder `./dcn_norm`, which contains training and evaluation data.

4. Write a simple Python script for training:
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
                              workspace_size_per_gpu_in_mb = 75,
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

5. Train the model by running the following command:
   ```
   python dcn_norm_train.py
   ```
   **NOTE**: It is presumed that the evaluation AUC value is incorrect since randomly generated datasets are being used. When the training is done, files that contain the
   dumped graph JSON, saved model weights, and optimizer states will be generated.

For more information, refer to the [HugeCTR User Guide](https://nvidia-merlin.github.io/HugeCTR/main/hugectr_user_guide.html).

## HugeCTR SDK ##
We're able to support external developers who can't use HugeCTR directly by exporting important HugeCTR components using:
* Sparse Operation Kit [directory](sparse_operation_kit) | [documentation](https://nvidia-merlin.github.io/HugeCTR/sparse_operation_kit/master/): a python package wrapped with GPU accelerated operations dedicated for sparse training/inference cases.
* [GPU Embedding Cache](gpu_cache): embedding cache available on the GPU memory designed for CTR inference workload.

## Support and Feedback ##
If you encounter any issues or have questions, go to [https://github.com/NVIDIA/HugeCTR/issues](https://github.com/NVIDIA/HugeCTR/issues) and submit an issue so that we can provide you with the necessary resolutions and answers. To further advance the HugeCTR Roadmap, we encourage you to share all the details regarding your recommender system pipeline using this [survey](https://developer.nvidia.com/merlin-devzone-survey).

## Contributing to HugeCTR ##
With HugeCTR being an open source project, we welcome contributions from the general public. With your contributions, we can continue to improve HugeCTR's quality and performance. To learn how to contribute, refer to our [HugeCTR Contributor Guide](https://nvidia-merlin.github.io/HugeCTR/main/hugectr_contributor_guide.html).

## Additional Resources ##
|Webpages|
|--------|
|[NVIDIA Merlin](https://developer.nvidia.com/nvidia-merlin)|
|[NVIDIA HugeCTR](https://developer.nvidia.com/nvidia-merlin/hugectr)|

### Talks ###
|Conference / Website|Title|Date|Speaker|Language|
|--------------------|-----|----|-------|--------|
|Short Videos Episode 1|[Merlin HugeCTR：GPU 加速的推荐系统框架](https://www.bilibili.com/video/BV1jT411E7VJ/)|May 2022|Joey Wang|中文|
|Short Videos Episode 2|[HugeCTR 分级参数服务器如何加速推理](https://www.bilibili.com/video/BV1PW4y127UA/)|May 2022|Joey Wang|中文|
|Short Videos Episode 3|[使用 HugeCTR SOK 加速 TensorFlow 训练](https://www.bilibili.com/video/BV1mG411n7XH/)|May 2022|Gems Guo|中文|
|GTC Sping 2022|[Merlin HugeCTR: Distributed Hierarchical Inference Parameter Server Using GPU Embedding Cache](https://www.nvidia.com/en-us/on-demand/session/gtcspring22-s41126/)|March 2022|Matthias Langer, Yingcan Wei, Yu Fan|English|
|APSARA 2021|[GPU 推荐系统 Merlin](https://yunqi.aliyun.com/2021/agenda/session205?spm=5176.23948577a2c4e.J_6988780170.27.5ae7379893BcVp)|Oct 2021|Joey Wang|中文|
|GTC Spring 2021|[Learn how Tencent Deployed an Advertising System on the Merlin GPU Recommender Framework](https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31820/)|April 2021|Xiangting Kong, Joey Wang|English|
|GTC Spring 2021|[Merlin HugeCTR: Deep Dive Into Performance Optimization](https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31269/)|April 2021|Minseok Lee|English|
|GTC Spring 2021|[Integrate HugeCTR Embedding with TensorFlow](https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31425/)|April 2021|Jianbing Dong|English|
|GTC China 2020|[MERLIN HUGECTR ：深入研究性能优化](https://www.nvidia.cn/on-demand/session/gtccn2020-cns20516/)|Oct 2020|Minseok Lee|English|
|GTC China 2020|[性能提升 7 倍 + 的高性能 GPU 广告推荐加速系统的落地实现](https://www.nvidia.cn/on-demand/session/gtccn2020-cns20483/)|Oct 2020|Xiangting Kong|中文|
|GTC China 2020|[使用 GPU EMBEDDING CACHE 加速 CTR 推理过程](https://www.nvidia.cn/on-demand/session/gtccn2020-cns20626/)|Oct 2020|Fan Yu|中文|
|GTC China 2020|[将 HUGECTR EMBEDDING 集成于 TENSORFLOW](https://www.nvidia.cn/on-demand/session/gtccn2020-cns20377/)|Oct 2020|Jianbing Dong|中文|
|GTC Spring 2020|[HugeCTR: High-Performance Click-Through Rate Estimation Training](https://www.nvidia.com/en-us/on-demand/session/gtcsj20-s21455/)|March 2020|Minseok Lee, Joey Wang|English|
|GTC China 2019|[HUGECTR: GPU 加速的推荐系统训练](https://www.nvidia.cn/on-demand/session/gtcchina2019-cn9794/)|Oct 2019|Joey Wang|中文|

### Blogs ###
|Conference / Website|Title|Date|Authors|Language|
|--------------------|-----|----|-------|--------|
|Wechat Blog|[Merlin HugeCTR 分级参数服务器系列之三：集成到TensorFlow](https://mp.weixin.qq.com/s/sFmJXZ53Qj4J7iGkzGvQbw)|Nov. 2022|Kingsley Liu|中文|
|NVIDIA Devblog|[Scaling Recommendation System Inference with Merlin Hierarchical Parameter Server/使用 Merlin 分层参数服务器扩展推荐系统推理](https://developer.nvidia.com/zh-cn/blog/scaling-recommendation-system-inference-with-merlin-hierarchical-parameter-server/)|August 2022|Shashank Verma, Wenwen Gao, Yingcan Wei, Matthias Langer, Jerry Shi, Fan Yu, Kingsley Liu, Minseok Lee|English/中文|
|NVIDIA Devblog|[Merlin HugeCTR Sparse Operation Kit 系列之二](https://developer.nvidia.cn/zh-cn/blog/merlin-hugectr-sparse-operation-kit-series-2/)|June 2022|Kunlun Li|中文|
|NVIDIA Devblog|[Merlin HugeCTR Sparse Operation Kit 系列之一](https://developer.nvidia.com/zh-cn/blog/merlin-hugectr-sparse-operation-kit-part-1/)|March 2022|Gems Guo, Jianbing Dong|中文|
|Wechat Blog|[Merlin HugeCTR 分级参数服务器系列之二](https://mp.weixin.qq.com/s/z-K3UNg6-ysrfKe3C6McZg)|March 2022|Yingcan Wei, Matthias Langer, Jerry Shi|中文|
|Wechat Blog|[Merlin HugeCTR 分级参数服务器系列之一](https://mp.weixin.qq.com/s/5_AKe6f_nJjddCLZU28P2A)|Jan. 2022|Yingcan Wei, Jerry Shi|中文|
|NVIDIA Devblog|[Accelerating Embedding with the HugeCTR TensorFlow Embedding Plugin](https://developer.nvidia.com/blog/accelerating-embedding-with-the-hugectr-tensorflow-embedding-plugin/)|Sept 2021|Vinh Nguyen, Ann Spencer, Joey Wang and Jianbing Dong|English|
|medium.com|[Optimizing Meituan’s Machine Learning Platform: An Interview with Jun Huang](https://medium.com/nvidia-merlin/optimizing-meituans-machine-learning-platform-an-interview-with-jun-huang-7e046143131f)|Sept 2021|Sheng Luo and Benedikt Schifferer|English|
|medium.com|[Leading Design and Development of the Advertising Recommender System at Tencent: An Interview with Xiangting Kong](https://medium.com/nvidia-merlin/leading-design-and-development-of-the-advertising-recommender-system-at-tencent-an-interview-with-37f1eed898a7)|Sept 2021|Xiangting Kong, Ann Spencer|English|
|NVIDIA Devblog|[扩展和加速大型深度学习推荐系统 – HugeCTR 系列第 1 部分](https://developer.nvidia.com/zh-cn/blog/scaling-and-accelerating-large-deep-learning-recommender-systems-hugectr-series-part-1/)|June 2021|Minseok Lee|中文|
|NVIDIA Devblog|[使用 Merlin HugeCTR 的 Python API 训练大型深度学习推荐模型 – HugeCTR 系列第 2 部分](https://developer.nvidia.com/zh-cn/blog/training-large-deep-learning-recommender-models-with-merlin-hugectrs-python-apis-hugectr-series-part2/)|June 2021|Vinh Nguyen|中文|
|medium.com|[Training large Deep Learning Recommender Models with Merlin HugeCTR’s Python APIs — HugeCTR Series Part 2](https://medium.com/nvidia-merlin/training-large-deep-learning-recommender-models-with-merlin-hugectrs-python-apis-hugectr-series-69a666e0bdb7)|May 2021|Minseok Lee, Joey Wang, Vinh Nguyen and Ashish Sardana|English|
|medium.com|[Scaling and Accelerating large Deep Learning Recommender Systems — HugeCTR Series Part 1](https://medium.com/nvidia-merlin/scaling-and-accelerating-large-deep-learning-recommender-systems-hugectr-series-part-1-c19577acfe9d)|May 2021|Minseok Lee|English|
|IRS 2020|[Merlin: A GPU Accelerated Recommendation Framework](https://irsworkshop.github.io/2020/publications/paper_21_Oldridge_Merlin.pdf)|Aug 2020|Even Oldridge etc.|English|
|NVIDIA Devblog|[Introducing NVIDIA Merlin HugeCTR: A Training Framework Dedicated to Recommender Systems](https://developer.nvidia.com/blog/introducing-merlin-hugectr-training-framework-dedicated-to-recommender-systems/)|July 2020|Minseok Lee and Joey Wang|English|
