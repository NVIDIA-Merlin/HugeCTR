# DCN MULTI-NODES SAMPLE #
A sample of building and training Deep & Cross Network with HugeCTR on multi-nodes [(link)](https://arxiv.org/pdf/1708.05123.pdf).

## Dataset and preprocess ##
In running this sample, [Criteo 1TB Click Logs dataset](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/) is used.
The dataset contains 24 files, each of which corresponds to one day of data.
To spend less time on preprocessing, we use only one of them.
Each sample consists of a label (1 if the ad was clicked, otherwise 0) and 39 features (13 integer features and 26 categorical features).
The dataset also has the significant amounts of missing values across the feature columns, which should be preprocessed accordingly.

### 1. Download the dataset and preprocess

Go to [this link](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/),
and download one of 24 files into the directory "${project_root}/tools, 
or execute the following command:
```
$ cd ${project_root}/tools
$ wget http://azuremlsampleexperiments.blob.core.windows.net/criteo/day_1.gz
```
- **NOTE**: Replace 1 with a value from [0, 23] to use a different day.

In preprocessing, we will further reduce the amounts of data to speedup the preprocessing, fill missing values, remove the feature values whose occurrences are very rare, etc.
Please choose one of the following two methods to make the dataset ready for HugeCTR training.

#### Preprocessing by Pandas ####
```shell
$ bash preprocess.sh 1 criteo_data pandas 1 0
```
- **NOTE**: The first argument represents the dataset postfix.  For instance, if `day_1` is used, it is 1.
- **NOTE**: the second argument `criteo_data` is where the preprocessed data is stored.
You may want to change it in case where multiple datasets for different purposes are generated concurrently.
If you change it, `source` and `eval_source` in your JSON config file must be changed as well.
- **NOTE**: the fourth arguement (one after `pandas`) represents if the normalization is applied to dense features (1=ON, 0=OFF).
- **NOTE**: the last argument decides if the feature crossing is applied (1=ON, 0=OFF).
It must remains 0 unless the sample is not `wdl`.

### 2. Build HugeCTR with **multi-nodes training supported** (refer to the README in home directory).
You can chose to use [our NGC docker image](../docs/hugectr_user_guide.md#build-hugectr-with-the-docker-image) where the multi-node mode is enabled. If you have to build your own docker image and HugeCTR yourself, refer to [our instructions](../docs/hugectr_user_guide.md#build-with-multi-nodes)

## Plan file generation ##
If gossip communication library is used, a plan file is needed to be generated first as below. If NCCL communication library is used, there is no need to generate a plan file, just skip this step. 

Login to your GPU cluster and gets two nodes. For example, if on a SLURM system:  
```shell
# We will use two nodes, i.e., -N 2, in this example
$ srun -N 2 --pty bash -i
$ export CUDA_DEVICE_ORDER=PCI_BUS_ID
$ mpirun python3 plan_generation/plan_generator.py ../samples/dcn2nodes/dcn8l8gpu2nodes.json
```
**NOTE:** If your cluster is unequpped with a job scheduler, please refer to [our tutorial](../tutorial/multinode-training/README.md/)

## Training with HugeCTR ##

1. Run huge_ctr
```shell
$ mpirun --bind-to none ../build/bin/huge_ctr --train /samples/dcn2nodes/dcn8l8gpu2nodes.json
```
