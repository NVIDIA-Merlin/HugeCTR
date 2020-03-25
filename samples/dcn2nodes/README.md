# DCN MULTI-NODES SAMPLE #
A sample of building and training Deep & Cross Network with HugeCTR on multi-nodes [(link)](https://arxiv.org/pdf/1708.05123.pdf).

## Dataset and preprocess ##
The data is provided by CriteoLabs (http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/).
The original training set contains 45,840,617 examples.
Each example contains a label (1 if the ad was clicked, otherwise 0) and 39 features (13 integer features and 26 categorical features).
Because the original test set doesn't contain labels, it is not used.

### Requirements ###
Python >= 3.6.9
Pandas 1.0.1
Sklearn 0.22.1

1. Download the dataset and preprocess

Go to [(link)](http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/)
and download kaggle-display dataset into the folder "${project_home}/tools/criteo_script/".

```shell
$ cd ../../tools/criteo_script/
$ bash preprocess.sh dcn 1 0
$ cd ../../samples/dcn/
```

2. Convert the dataset to HugeCTR format
```shell
$ cp ../../build/bin/criteo2hugectr ./
$ ./criteo2hugectr ../../tools/criteo_script/train. criteo/sparse_embedding file_list.txt
$ ./criteo2hugectr ../../tools/criteo_script/val criteo_test/sparse_embedding file_list_test.txt
```

## Plan file generation ##
Login to your GPU cluster and acquire two nodes. For example, if on a SLURM system:  
```shell
$ srun -N 2 --pty bash -i
$ export CUDA_DEVICE_ORDER=PCI_BUS_ID
$ mpirun python ../../tools/plan_generation/plan_from_topology_asynch.py dcn8l8gpu2nodes.json
```

## Training with HugeCTR ##

1. Build HugeCTR with the instructions on README.md under home directory.

2. Copy huge_ctr to samples/criteo
```shell
$ cp ../../build/bin/huge_ctr ./
```

3. Run huge_ctr
```shell
$ mpirun --bind-to none ./huge_ctr --train dcn8l8gpu2nodes.json
```


