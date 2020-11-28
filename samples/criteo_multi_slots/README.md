# CRITEO CTR SAMPLE #
In this sample we aim to demonstrate the basic usage of SparseEmbeddingHash and multiple slots.

## Dataset and preprocess ##
The data is provided by CriteoLabs (http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/).
The original training set contains 45,840,617 examples.
Each example contains a label (1 if the ad was clicked, otherwise 0) and 39 features (13 integer features and 26 categorical features).
The dataset also has the significant amounts of missing values across the feature columns, which should be preprocessed accordingly.
The original test set doesn't contain labels, so it's not used.

1. Download the dataset and preprocess

Go to [(link)](http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/)
and download the kaggle-display dataset into the folder "${project_home}/tools/criteo_script_legacy/".
The script `preprocess.sh` fills the missing values by mapping them to the unused unique integer or category.
It also replaces unique values which appear less than six times across the entire dataset with the unique value for missing values.
Its purpose is to reduce the vocabulary size of each column while not losing too much information.

```shell
$ cd ../../tools/criteo_script_legacy/
$ bash preprocess.sh
$ cd ../../samples/criteo_multi_slots/
```

2. Build HugeCTR with the instructions on README.md under home directory.

3. Translate the dataset to HugeCTR format
```shell
$ cp ../../build/bin/criteo2hugectr_legacy ./
$ ./criteo2hugectr_legacy 10 ../../tools/criteo_script_legacy/train.out criteo/sparse_embedding file_list.txt
$ ./criteo2hugectr_legacy 10 ../../tools/criteo_script_legacy/test.out criteo_test/sparse_embedding file_list_test.txt
```

## Training with HugeCTR ##

1. Copy huge_ctr to samples/criteo_multi_slots
```shell
$ cp ../../build/bin/huge_ctr ./
```

2. Run huge_ctr
```shell
$ ./huge_ctr --train ./criteo_multi_slots.json
```
