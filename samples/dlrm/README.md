# DLRM CTR SAMPLE #
A sample of building and training DLRM model with HugeCTR [(link)](https://ai.facebook.com/blog/dlrm-an-advanced-open-source-deep-learning-recommendation-model/).

## Dataset and preprocess ##
[(Terabyte Click Logs)](https://labs.criteo.com/2013/12/download-terabyte-click-logs/) provided by CriteoLabs is used in this sample. The up limit of row count of each embedding table is limited to 40 million.
We process the date in the same way as dlrm [(repo)](https://github.com/facebookresearch/dlrm#benchmarking). Each sample has 40 32bits integers, where the first integer is label,
the next 13 integers are dense feature and the following 26 integers are category feature.

## Requirements ##
* DGX A100 or
* DGX2 (32GB V100)


## Run ##
1. Given two processed training data file: train.bin (671.2GB) and test.bin(14.3GB), copy them into this directory (samples/dlrm)

2. Build HugeCTR with the instructions on README.md under home directory.

3. Run either of the four json configure files in this directory: e.g.
```shell
$ ./huge_ctr --train ./dlrm_fp16_64k.json
```

Note that `cache_eval_data` is only supported in DGX A100 if you are runing in DGX2 please disable this in json files


