# Introduction
This folder contains the scripts used to compare perf between HugeCTR and TorchRec.

You can use docker image `gitlab-master.nvidia.com:5005/dl/hugectr/hugectr/devel_optimized_torchrec:latest`, which containes torchrec v0.5.0-rc1 and fbgemm v0.5.0-rc0.

About the data, you should map `/lustre/fsw/devtech/hpc-hugectr/aleliu1/benchmark_ebc/dataset` to `/workdir/dataset` when you do benchmarking on selene. Currently data is only accessible on selene.

You can use `benchmark.sh` to run the benchmark. It supports following use case:
```
bash benchmark.sh $framework $test_case $batchsize $nsys_result_path
```
args:
$framework: required. can be hugectr or torchrec, when using torchrec, please note you should add `--ntasks-per-node=8` because torchrec is using 1 process 1 GPU mode
$test_case: required. right now can be middle, middle_only_sparse, middle_small_vec, middle_small_vec_only_sparse, middle20, middle70. Different test_case uses different data or model
$batchsize: required. The global batchsize for training
$nsys_result_path: optional. If set, will run nsys and generate profiling result on the specified path.

