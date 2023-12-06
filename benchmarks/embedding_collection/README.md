# Introduction
This folder contains the scripts used to benchmark embedding collection.

# Usage Guide
## Dataset Generation
The folder `./dataset` contains scripts used to generate dataset. We provide 4 datasets generation scripts. You can refer `dataset/generation.sh` for how to use the scripts. 

## Training Benchmark
The folder `./hugectr` contains scripts used to benchmark training. It consists:
* `train.py`: the starting script.
* `sharding/`: An automatic sharding algorithm implementation and its test.

You can use `benchmark.sh` to run the benchmark. You should map generated dataset to `/workdir/dataset` to match with configuration in `benchmark.sh`. `benchmark.sh` should be used as follows:
```
bash benchmark.sh $test_case $batchsize
```
args:
$test_case: required. It supports `180table_70B_hotness80`, `7table_470B_hotness20`, `dcnv2`, `510table_110B_hotness5`, `200table_100B_hotness20`.
$batchsize: required. The global batch_size for training

# ENV
Those env variable can be used to skip specific component during training.
* SKIP_H2D
* SKIP_DATA_DISTRIBUTOR
* SKIP_EMBEDDING
* SKIP_ALL2ALL
* SKIP_BOTTOM_MLP
* SKIP_TOP_MLP
* SKIP_ALLREDUCE