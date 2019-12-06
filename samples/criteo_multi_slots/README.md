# CRITEO CTR SAMPLE #
In this sample we aim to demostrate the basic usage of SparseEmbeddingHash and multiple slots.

## Dataset and preprocess ##
The data is provided by CriteoLabs (http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/). The original training set contains 45,840,617 examples. Each example contains a label (1 if the ad was clicked, otherwise 0) and 39 features (13 integer features and 26 categorical features). The original test set doesn't contain labels, so it's not used. To demostrate how slot works, the keys in this dataset is distributed into 10 groups with mod e.g. 11 mod 10 is 1

1. Download the dataset and preprocess

Go to [(link)](http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/) and download kaggle-display dataset into the folder "${project_home}/tools/criteo_script/".

```shell
$ cd ../../tools/criteo_script/ && bash usage.sh && cd ../../samples/criteo_multi_slots/
```

2. Translate the dataset to HugeCTR format
```shell
$ g++ -DNDEBUG -o criteo2hugectr10slots -std=c++11 criteo2hugectr10slots.cpp  
$ ./criteo2hugectr10slots ../../tools/criteo_script/train.out criteo/sparse_embedding file_list.txt
$ ./criteo2hugectr10slots ../../tools/criteo_script/test.out criteo_test/sparse_embedding file_list_test.txt
```

## Training with HugeCTR ##

1. Build HugeCTR with the instructions on README.md under home directory.

2. Copy huge_ctr to samples/criteo_multi_slots
```shell
$ cd build/bin/
$ cp ./huge_ctr ../../samples/criteo_multi_slots/
```

3. Run huge_ctr
```shell
$ ./huge_ctr --model-init ./criteo.json
$ ./huge_ctr --train ./criteo.json
```


