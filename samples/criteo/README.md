# CRITEO CTR SAMPLE #
In this sample we aim to demostrate the basic usage of huge_ctr.

## Dataset and preprocess ##
The data is provided by CriteoLabs (http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/).
The original training set contains 45,840,617 examples.
Each example contains a label (1 if the ad was clicked, otherwise 0) and 39 features (13 integer features and 26 categorical features).
The dataset also has the significant amounts of missing values across the feature columns, which should be preprocessed acordingly.
The original test set doesn't contain labels, so it's not used.

1. Download the dataset and preprocess

Go to [(link)](http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/)
and download kaggle-display dataset into the folder "${project_home}/tools/criteo_script/".
The script `usage.sh` fills the missing values by mapping them to the unusned unique integer or category.
It also replaces unique values which appear less than six times across the entire dataset with the unique value for missing values.
Its purpose is to redcue the vocabulary size of each columm while not losing too much information.

```shell
$ cd ../../tools/criteo_script_legacy/ && bash usage.sh && cd ../../samples/criteo/
```

2. Translate the dataset to HugeCTR format
```shell
$ cp ../../build/bin/criteo2hugectr_legacy ./
$ ./criteo2hugectr_legacy 1 ../../tools/criteo_script_legacy/train.out criteo/sparse_embedding file_list.txt
$ ./criteo2hugectr_legacy 1 ../../tools/criteo_script_legacy/test.out criteo_test/sparse_embedding file_list_test.txt
```

## Training with HugeCTR ##

1. Build HugeCTR with the instructions on README.md under home directory.

2. Copy huge_ctr to samples/criteo
```shell
$ cp ../../build/bin/huge_ctr ./
```

3. Run huge_ctr
```shell
$ ./huge_ctr --train ./criteo.json
```
