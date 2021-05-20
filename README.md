# <img src="docs/user_guide_src/merlin_logo.png" alt="logo" width="85"/> Merlin: HugeCTR #

HugeCTR, a component of [NVIDIA Merlin Open Beta](https://developer.nvidia.com/nvidia-merlin#getstarted), is a GPU-accelerated recommender framework. It was designed to distribute training across multiple GPUs and nodes and estimate Click-Through Rates (CTRs). HugeCTR supports model-parallel embedding tables and data-parallel neural networks and their variants such as [Wide and Deep Learning (WDL)](https://arxiv.org/abs/1606.07792), [Deep Cross Network (DCN)](https://arxiv.org/abs/1708.05123), [DeepFM](https://arxiv.org/abs/1703.04247), and [Deep Learning Recommendation Model (DLRM)](https://ai.facebook.com/blog/dlrm-an-advanced-open-source-deep-learning-recommendation-model/). For additional information, see [HugeCTR User Guide](docs/hugectr_user_guide.md).

Design Goals:
* Fast: HugeCTR is a speed-of-light CTR model framework.
* Dedicated: HugeCTR provides the essentials so that you can train your CTR model in an efficient manner.
* Easy: Regardless of whether you are a data scientist or machine learning practitioner, we've made it easy for anybody to use HugeCTR.

## Table of Contents
* [Performance](#performance)
* [Release Notes](#release-notes)
* [Getting Started](#getting-started)
* [Support and Feedback](#support-and-feedback)

## Performance ##
We've tested HugeCTR's performance on the following systems:
* DGX-2 and DGX A100
* Versus TensorFlow (TF)

### Evaluating HugeCTR's Performance on the DGX-2 and DGX A100
We submitted the DLRM benchmark with HugeCTR version 2.2 to [MLPerf Training v0.7](https://mlperf.org/training-results-0-7). The dataset was [Criteo Terabyte Click Logs](https://labs.criteo.com/2013/12/download-terabyte-click-logs/), which contains 4 billion user and item interactions over 24 days. The target machines were DGX-2 with 16 V100 GPUs and DGX A100 with eight A100 GPUs. Fig. 1 summarizes the performance. For more details, see [this blog post](https://developer.nvidia.com/blog/accelerating-recommender-systems-training-with-nvidia-merlin-open-beta/).

<div align=center><img  width='624' height='385' src ="docs/user_guide_src/mlperf_dlrm_training.png"/></div>
<div align=center>Fig. 1 MLPerf v0.7 DLRM training performance across different platforms.</div>

#### HugeCTR Strong Scaling Results on DGX A100
Fig. 2 shows the strong scaling result for both the full precision mode (FP32) and mixed-precision mode (FP16) on a single NVIDIA DGX A100. Bars represent the average iteration time in ms.
<div align=center><img  width='600' height='371' src ="docs/user_guide_src/a100_scaling.png"/></div>
<div align=center>Fig. 2 The strong scaling result of HugeCTR with a W&D model on a single DGX A100, the lower the better.</div>

### Evaluating HugeCTR's Performance on the TensorFlow
In the TensorFlow test case below, HugeCTR exhibits a speedup up to 114x compared to a CPU server that is running TensorFlow with only one V100 GPU and almost the same loss curve.

* Test environment:
  - CPU Server: Dual 20-core Intel(R) Xeon(R) CPU E5-2698 v4 @ 2.20GHz
  - TensorFlow version 2.0.0
  - V100 16GB: NVIDIA DGX1 servers

* Network:
  - `Wide Deep Learning`: Nx 1024-unit FC layers with ReLU and dropout, emb_dim: 16; Optimizer: Adam for both Linear and DNN models
  - `Deep Cross Network`: Nx 1024-unit FC layers with ReLU and dropout, emb_dim: 16, 6x cross layers; Optimizer: Adam for both Linear and DNN models

* Dataset:
  - The data is provided by [CriteoLabs](http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/). The original training set contains 45,840,617 examples. Each example contains a label (0 by default OR 1 if the ad was clicked) and 39 features in which 13 are integer and 26 are categorical.

* Preprocessing:
  - Common: Preprocessed by using the scripts available in tools/criteo_script.
  - HugeCTR: Converted to the HugeCTR data format with criteo2hugectr.
  - TF: Converted to the TFRecord format for the efficient training on Tensorflow.

<div align=center><img width='800' height='339' src ="docs/user_guide_src/WDL.JPG"/></div>
<div align=center>Fig. 3 WDL Performance and Loss Curve Comparison with TensorFlow Version 2.0</div>

<br></br>

<div align=center><img width='800' height='339' src ="docs/user_guide_src/DCN.JPG"/></div>
<div align=center>Fig. 4 DCN performance and Loss Curve Comparison with TensorFlow Version 2.0</div>

## Release Notes ##
Bigger model and large scale training are always the main requirments in recommendation system. In v3.1, we provide a set of new optimizations for good scalability as below, and now they are avaliable in this beta version.  
- Distributed Hybrid embedding - Model/data parallel split of embeddings based on statistical access frequency to minimize embedding exchange traffic.
- Optimized communication collectives - Hierarchical multi-node all-to-all for NVLINK aggregation and oneshot algorithm for All-reduce.
- Optimized data reader - Async I/O based data reader to maximize I/O utilization, minimize interference with collectives and eval caching.
- MLP fusions - Fused GEMM + Relu + Bias fprop and GEMM + dRelu + bgrad bprop.
- Compute-communication overlap - Generalized embedding and bottom MLP overlap. 
- Holistic cuda graph - Full iteration graph capture to reduce launch latencies and jitter.

## Getting Started ##
If you'd like to quickly train a model using the Python interface, follow these steps:
1. Start a NGC container with your local host directory (/your/host/dir mounted) by running the following command:
   ```
   docker run --runtime=nvidia --rm -v /your/host/dir:/your/container/dir -w /your/container/dir -it -u $(id -u):$(id -g) -it nvcr.io/nvidia/merlin/merlin-training:0.5
   ```

   **NOTE**: The **/your/host/dir** directory is just as visible as the **/your/container/dir** directory. The **/your/host/dir** directory is also your starting directory.

2. Activate the merlin conda environment by running the following command:  
   ```shell.
   source activate merlin
   ```

3. Inside the container, copy the DCN configuration file to our mounted directory (/your/container/dir).

   This config file specifies the DCN model architecture and its optimizer. With any Python use case, the solver clause within the configuration file is not used at all.

4. Generate a synthetic dataset based on the configuration file by running the following command:
   ```
   ./data_generator --config-file dcn.json --voc-size-array 39884,39043,17289,7420,20263,3,7120,1543,39884,39043,17289,7420,20263,3,7120,1543,63,63,39884,39043,17289,7420,20263,3,7120,1543 --distribution powerlaw --alpha -1.2
   ```

5. Write a simple Python code using the hugectr module as shown here:
   ```
   # train.py
   import sys
   import hugectr
   from mpi4py import MPI

   def train(json_config_file):
     solver_config = hugectr.solver_parser_helper(batchsize = 16384,
                                                  batchsize_eval = 16384,
                                                  vvgpu = [[0,1,2,3,4,5,6,7]],
                                                  repeat_dataset = True)
     sess = hugectr.Session(solver_config, json_config_file)
     sess.start_data_reading()
     for i in range(10000):
       sess.train()
       if (i % 100 == 0):
         loss = sess.get_current_loss()
         print("[HUGECTR][INFO] iter: {}; loss: {}".format(i, loss))

   if __name__ == "__main__":
     json_config_file = sys.argv[1]
     train(json_config_file)

   ```

   **NOTE**: Update the vvgpu (the active GPUs), batchsize, and batchsize_eval parameters according to your GPU system.

6. Train the model by running the following command:
   ```
   python train.py dcn.json
   ```

For additional information, see the [HugeCTR User Guide](docs/hugectr_user_guide.md).

## Support and Feedback ##
If you encounter any issues and/or have questions, please file an issue [here](https://github.com/NVIDIA/HugeCTR/issues) so that we can provide you with the necessary resolutions and answers. To further advance the Merlin/HugeCTR Roadmap, we encourage you to share all the details regarding your recommender system pipeline using this [survey](https://developer.nvidia.com/merlin-devzone-survey).
