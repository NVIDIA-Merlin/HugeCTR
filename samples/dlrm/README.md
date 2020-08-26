# DLRM CTR SAMPLE #
A sample of building and training DLRM model with HugeCTR [(link)](https://ai.facebook.com/blog/dlrm-an-advanced-open-source-deep-learning-recommendation-model/).

## Dataset and preprocess for Terabyte Click Logs ##
[(Terabyte Click Logs)](https://labs.criteo.com/2013/12/download-terabyte-click-logs/) provided by CriteoLabs is used in this sample. The up limit of row count of each embedding table is limited to 40 million.
We process the data in the same way as dlrm [(repo)](https://github.com/facebookresearch/dlrm#benchmarking). Each sample has 40 32bits integers, where the first integer is label,
the next 13 integers are dense features and the following 26 integers are category features.

## Requirements ##
* Requirements on [README](../../README.md#Requirements) 
* DGX A100 or DGX2 (32GB V100)


## Run ##
1. Download TeraBytes datasets from [(Terabyte Click Logs)](https://labs.criteo.com/2013/12/download-terabyte-click-logs/). Unzip them and name as `day_0`, `day_1`, ..., `day_23`.

2. Build HugeCTR with the instructions on [README.md](../../README.md#build) under home directory.

3. Preprocess Datasets. This operation will generate `train.bin(671.2GB)` and `test.bin(14.3GB)`.
```bash
# Usage: ./dlrm_raw input_dir output_dir --train {days for training} --test {days for testing}
$ cp ../../build/bin/dlrm_raw ./
$ ./dlrm_raw ./ ./ \
--train 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22 \
--test 23
```

4. Run either of the four json configure files in this directory: e.g.
```shell
$ ./huge_ctr --train ./terabyte_fp16_64k.json
```

**Note** that `cache_eval_data` is only supported in DGX A100. If you are running in DGX2, please disable this in json files.


## Dataset and preprocess for Kaggle Criteo Logs ##
The data is provided by CriteoLabs (http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/).
The original training set contains 45,840,617 examples.
Each example contains a label (1 if the ad was clicked, otherwise 0) and 39 features (13 integer features and 26 categorical features).
The dataset also has the significant amounts of missing values across the feature columns, which should be preprocessed acordingly.
The original test set doesn't contain labels, so it's not used.

## Requirements ##
+ Python >= 3.6.9
+ libcudf >= 0.15
+ librmm >= 0.15


## Run ##
1. Download the dataset and preprocess <br>
Go to ([link](http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/)) and download kaggle-display dataset into the folder `"${project_home}/samples/dlrm/"`. The tool `dlrm_raw` converts original datas to HugeCTR's raw format and fills the missing values. Only `train.txt` will be used to generate raw datas.
```bash
# download and preprocess
$ cd ../../samples/dlrm/
$ tar zxvf dac.tar.gz
```

2. Build HugeCTR with the instructions on README.md under home directory.

3. Convert the dataset to HugeCTR Raw format
```bash
# Usage: ./dlrm_raw input_dir out_dir
$ cp ../../build/bin/dlrm_raw ./
$ ./dlrm_raw ./ ./ 
```

4. Train with HugeCTR
```bash
$ cp ../../build/bin/huge_ctr ./
$ ./huge_ctr --train ./kaggle_fp32.json
```
