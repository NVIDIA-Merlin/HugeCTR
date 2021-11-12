# Contributing to HugeCTR

We're grateful for your interest in HugeCTR and value your contributions. You can contribute to HugeCTR by:
* submitting a [feature, documentation, or bug request](https://github.com/NVIDIA/HugeCTR/issues/new/choose).

  **NOTE**: After we review your request, we'll assign it to a future release. If you think the issue should be prioritized over others, comment on the issue.
  
* proposing and implementing a new feature.

  **NOTE**: Once we agree to the proposed design, you can go ahead and implement the new feature using the steps outlined in the [Contribute New Code section](#contribute-new-code).
  
* implementing a pending feature or fixing a bug.

  **NOTE**: Use the steps outlined in the [Contribute New Code section](#contribute-new-code). If you need more information about a particular issue, 
  add your comments on the issue.

## Contribute New Code

1. Build HugeCTR or Sparse Operation Kit (SOK) from source using the steps outlined in the [Set Up the Development Environment section](#set-up-the-development-environment).
2. [File an issue](https://github.com/NVIDIA/HugeCTR/issues/new/choose) and add a comment stating that you'll work on it.
3. Start coding.
 
   **NOTE**: Don't forget to add or update the unit tests properly.
   
4. [Create a pull request](https://github.com/nvidia/HugeCTR/compare) for you work.
5. Wait for a maintainer to review your code.

   You may be asked to make additional edits to your code if necessary. Once approved, a maintainer will merge your pull request.

If you have any questions or need clarification, don't hesitate to add comments to your issue and we'll respond promptly.

## Set Up the Development Environment With Light-weight Containers

In HugeCTR we provide the development environment by addtional light-weight containers as below for easier modification. By using such containers, you don't need to build source code under NGC Container.

**Note**: the message on terminal below is not errors if you are working in such containers.
```
groups: cannot find name for group ID 1007
I have no name!@56a762eae3f8:/hugectr
```

### Build HugeCTR from Source

To build HugeCTR from source, do the following:

1. Build the `hugectr:devel` image using the steps outlined [here](../tools/dockerfiles#build-container-for-model-training).
   
   Be sure to choose a **Development mode**.

2. Download the HugeCTR repository and the third-party modules that it relies on by running the following commands:
   ```shell
   $ git clone https://github.com/NVIDIA/HugeCTR.git
   $ cd HugeCTR
   $ git submodule update --init --recursive
   ```
   
3. Build HugeCTR from scratch using one or any combination of the following options:
   - **SM**: You can use this option to build HugeCTR with a specific compute capability (DSM=80) or multiple compute capabilities (DSM="70;75"). The default compute capability 
     is 70, which uses the NVIDIA V100 GPU. For more information, refer to [Compute Capability](https://github.com/NVIDIA/HugeCTR/blob/master/docs/hugectr_user_guide.md#compute-capability). 60 is not supported for inference deployments. For more information, refer to [Quick Start](https://github.com/triton-inference-server/hugectr_backend#quick-start).
   - **CMAKE_BUILD_TYPE**: You can use this option to build HugeCTR with Debug or Release. When using Debug to build, HugeCTR will print more verbose logs and execute GPU tasks 
     in a synchronous manner.
   - **VAL_MODE**: You can use this option to build HugeCTR in validation mode, which was designed for framework validation. In this mode, loss of training will be shown as the 
     average of eval_batches results. Only one thread and chunk will be used in the data reader. Performance will be lower when in validation mode. This option is set to OFF by 
     default.
   - **ENABLE_MULTINODES**: You can use this option to build HugeCTR with multiple nodes. This option is set to OFF by default. For more information, refer to [samples/dcn2nodes](../samples/dcn).
   - **ENABLE_INFERENCE**: You can use this option to build HugeCTR in inference mode, which was designed for the inference framework. In this mode, an inference shared library 
     will be built for the HugeCTR Backend. Only interfaces that support the HugeCTR Backend can be used. Therefore, you can’t train models in this mode. This option is set to 
     OFF by default.

   Here are some examples of how you can build HugeCTR using these build options:
   ```shell
   $ mkdir -p build && cd build
   $ cmake -DCMAKE_BUILD_TYPE=Release -DSM=70 .. # Target is NVIDIA V100 with all others by default
   $ make -j && make install
   ```

   ```shell
   $ mkdir -p build && cd build
   $ cmake -DCMAKE_BUILD_TYPE=Release -DSM="70;80" -DVAL_MODE=ON .. # Target is NVIDIA V100 / A100 with Validation mode on.
   $ make -j && make install
   ```

   ```shell
   $ mkdir -p build && cd build
   $ cmake -DCMAKE_BUILD_TYPE=Release -DSM="70;80" -DCMAKE_BUILD_TYPE=Debug .. # Target is NVIDIA V100 / A100 with Debug mode.
   $ make -j && make install
   ```

   ```shell
   $ mkdir -p build && cd build
   $ cmake -DCMAKE_BUILD_TYPE=Release -DSM="70;80" -DENABLE_INFERENCE=ON .. # Target is NVIDIA V100 / A100 with Validation mode on.
   $ make -j && make install
   ```

### Build Sparse Operation Kit (SOK) from Source

To build the Sparse Operation Kit component in HugeCTR, do the following:

1. Build the `hugectr:tf-plugin` docker image using the steps noted [here](../tools/dockerfiles/README.md#build-container-for-tensorflow-plugin).
   
   Be sure to choose a **Development mode**.

2. Download the HugeCTR repository by running the following command:
   ```shell
   $ git clone https://github.com/NVIDIA/HugeCTR.git hugectr
   ```
   
3. Build and install libraries to the system paths by running the following commands:
   ```shell
   $ cd hugectr/sparse_operation_kit
   $ bash ./install.sh --SM=[GPU Compute Capability] --USE_NVTX=[ON/OFF]
   ```
   
   If you want to profile this module with nvtx, you can enable nvtx marks by setting `--USE_NVTX=ON`.
