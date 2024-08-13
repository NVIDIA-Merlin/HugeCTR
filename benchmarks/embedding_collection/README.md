# Introduction
This folder contains the scripts used to reproduce performance result in paper `Embedding Optimization for Training Large-scale Deep Learning Recommendation Systems with EMBark` published in RecSys 24.

You can use dockerfile unster this folder to build image for testing, which containes torchrec v0.6.0-rc2 and fbgemm v0.6.0-rc2.

About the data, you can use scripts under dataset to generate syncthetic dataset. Or you can refer [torchrec_dlrm](https://github.com/mlcommons/training/tree/master/recommendation_v2/torchrec_dlrm) to generate dataset used for MLPerf DLRM v2 benchmark.

You can use `benchmark.sh` to run the benchmark. It supports using in this way:
```
bash benchmark.sh $test_case $batchsize
```
args:
$test_case: required. right now can be utest, 180table_70B_hotness80, 7table_470B_hotness20, dcnv2, 510table_110B_hotness5, 200table_100B_hotness20. Different test_case uses different data or model. Please make sure you have generated corresponding data and placed them as the same place configured in benchmark.sh.

$batchsize: required. The global batchsize for training

