# <img src="docs/user_guide_src/merlin_logo.png" alt="logo" width="85"/> Merlin: HugeCTR #

HugeCTR is a GPU-accelerated recommender framework that was designed to distribute training across multiple GPUs and nodes and estimate Click-Through Rates (CTRs). HugeCTR supports model-parallel embedding tables and data-parallel neural networks and their variants such as [Wide and Deep Learning (WDL)](https://arxiv.org/abs/1606.07792), [Deep Cross Network (DCN)](https://arxiv.org/abs/1708.05123), [DeepFM](https://arxiv.org/abs/1703.04247), and [Deep Learning Recommendation Model (DLRM)](https://ai.facebook.com/blog/dlrm-an-advanced-open-source-deep-learning-recommendation-model/). HugeCTR is a component of [NVIDIA Merlin Open Beta](https://developer.nvidia.com/nvidia-merlin#getstarted), which is used for building large-scale deep learning recommender systems. For additional information, see [HugeCTR User Guide](docs/hugectr_user_guide.md).

Design Goals:
* **Fast**: HugeCTR is a speed-of-light CTR model framework that can [outperform](performance.md) popular recommender systems such as TensorFlow (TF).
* **Efficient**: HugeCTR provides the essentials so that you can train your CTR model in an efficient manner.
* **Easy**: Regardless of whether you are a data scientist or machine learning practitioner, we've made it easy for anybody to use HugeCTR.

## Table of Contents
* [Core Features](#core-features)
* [Getting Started](#getting-started)
* [Support and Feedback](#support-and-feedback)

## Core Features ##
HugeCTR supports a variety of features including the following:
* [multi-node training](docs/hugectr_user_guide.md#multi-node-training)
* [mixed precision training](docs/hugectr_user_guide.md#mixed-precision-training)
* [SGD optimizer and learning rate scheduling](docs/hugectr_user_guide.md#sgd-optimizer-and-learning-rate-scheduling)
* [model oversubscription](docs/hugectr_user_guide.md#model-oversubscription)

To learn about our latest enhancements, see our [release notes](release_notes.md).

## Getting Started ##
If you'd like to quickly train a model using the Python interface, follow these steps:
1. Start a NGC container with your local host directory (/your/host/dir mounted) by running the following command:
   ```
   docker run --runtime=nvidia --rm -v /your/host/dir:/your/container/dir -w /your/container/dir -it -u $(id -u):$(id -g) -it nvcr.io/nvidia/hugectr:v2.3
   ```

   **NOTE**: The **/your/host/dir** directory is just as visible as the **/your/container/dir** directory. The **/your/host/dir** directory is also your starting directory.

2. Inside the container, copy the DCN configuration file to our mounted directory (/your/container/dir).

   This config file specifies the DCN model architecture and its optimizer. With any Python use case, the solver clause within the configuration file is not used at all.

3. Generate a synthetic dataset based on the configuration file by running the following command:
   ```
   data_generator ./dcn.json ./dataset_dir 434428 1
   ```

   The following set of files are created: ./file_list.txt, ./file_list_test.txt, and ./dataset_dir/*.

4. Write a simple Python code using the hugectr module as shown here:
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

5. Train the model by running the following command:
   ```
   python train.py dcn.json
   ```

For additional information, see the [HugeCTR User Guide](../docs/hugectr_user_guide.md).

## Support and Feedback ##
If you encounter any issues and/or have questions, please file an issue [here](https://github.com/NVIDIA/HugeCTR/issues) so that we can provide you with the necessary resolutions and answers. To further advance the Merlin/HugeCTR Roadmap, we encourage you to share all the details regarding your recommender system pipeline using this [survey](https://developer.nvidia.com/merlin-devzone-survey).