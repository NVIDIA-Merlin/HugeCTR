# DCN CTR SAMPLE #
A sample of building and training Deep & Cross Network with HugeCTR [(link)](https://arxiv.org/pdf/1708.05123.pdf).

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

## Training with HugeCTR ##

1. Build HugeCTR with the instructions on README.md under home directory.

2. Copy huge_ctr to samples/criteo
```shell
$ cd build/bin/
$ cp ./huge_ctr ../../samples/dcn/
```

3. Run huge_ctr
```shell
$ ./huge_ctr --model-init ./dcn.json
$ ./huge_ctr --train ./dcn.json
```


