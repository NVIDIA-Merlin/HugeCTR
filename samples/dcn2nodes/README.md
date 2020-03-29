# DCN MULTI-NODES SAMPLE #
A sample of building and training Deep & Cross Network with HugeCTR on multi-nodes [(link)](https://arxiv.org/pdf/1708.05123.pdf).

## Dataset and preprocess ##
The data is provided by CriteoLabs (http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/).
The original training set contains 45,840,617 examples.
Each example contains a label (1 if the ad was clicked, otherwise 0) and 39 features (13 integer features and 26 categorical features).
The dataset also has the significant amounts of missing values across the feature columns, which should be preprocessed acordingly.
The original test set doesn't contain labels, so it's not used.

### Requirements ###
* Python >= 3.6.9
* Pandas 1.0.1
* Sklearn 0.22.1

1. Download the dataset and preprocess

Go to [(link)](http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/)
and download kaggle-display dataset into the folder "${project_home}/tools/criteo_script/".
The script `preprocess.sh` fills the missing values by mapping them to the unused unique integer or category.
It also replaces unique values which appear less than six times across the entire dataset with the unique value for missing values.
Its purpose is to redcue the vocabulary size of each columm while not losing too much information.
In addition, it normalizes the integer feature values to the range [0, 1],
but it doesn't create any feature crosses.

```shell
# The preprocessing can take 1-4 hours based on the system configuration.
$ cd ../../tools/criteo_script/
$ bash preprocess.sh dcn2nodes 1 0
$ cd ../../samples/dcn2nodes/
```

2. Convert the dataset to HugeCTR format
```shell
$ cp ../../build/bin/criteo2hugectr ./
$ ./criteo2hugectr ../../tools/criteo_script/dcn2nodes_data/train criteo/sparse_embedding file_list.txt
$ ./criteo2hugectr ../../tools/criteo_script/dcn2nodes_data/val criteo_test/sparse_embedding file_list_test.txt
```

## Plan file generation ##
Login to your GPU cluster and acquire two nodes. For example, if on a SLURM system:  
```shell
$ srun -N 2 --pty bash -i
$ export CUDA_DEVICE_ORDER=PCI_BUS_ID
$ mpirun python ../../tools/plan_generation/plan_generator.py dcn8l8gpu2nodes.json
```

## Training with HugeCTR ##

1. Build HugeCTR with the instructions on README.md under home directory.

2. Copy huge_ctr to samples/dcn2nodes
```shell
$ cp ../../build/bin/huge_ctr ./
```

3. Run huge_ctr
```shell
$ mpirun --bind-to none ./huge_ctr --train dcn8l8gpu2nodes.json
```

4. If you use docker container as the development environment, for multi-node, you should build the HugeCTR inside a docker container, and create docker image from the docker container, then distribut the image to all nodes, then run hugectr commands through docker

```shell
$ mpirun --bind-to none docker run --runtime=nvidia --rm -v $(pwd):/dataset hugectr:latest huge_ctr --train /dataset/dcn8l8gpu2nodes.json
```
