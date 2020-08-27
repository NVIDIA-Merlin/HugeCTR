# WDL CTR SAMPLE #
A sample of building and training Wide & Deep Network with HugeCTR [(link)](https://arxiv.org/abs/1606.07792).

## Dataset and preprocess ##
The data is provided by CriteoLabs (http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/).
The original training set contains 45,840,617 examples.
Each example contains a label (1 if the ad was clicked, otherwise 0) and 39 features (13 integer features and 26 categorical features).
The dataset also has the significant amounts of missing values across the feature columns, which should be preprocessed accordingly.
The original test set doesn't contain labels, so it's not used.

### Requirements ###
* Python >= 3.6.9
* Pandas 1.0.1
* Sklearn 0.22.1

1. Download the dataset and preprocess

Go to [(link)](http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/)
and download the kaggle-display dataset into the folder "${project_home}/tools/criteo_script/".
The script `preprocess.sh` fills the missing values by mapping them to the unused unique integer or category.
It also replaces unique values which appear less than six times across the entire dataset with the unique value for missing values.
Its purpose is to reduce the vocabulary size of each column while not losing too much information.
In addition, it doesn't only normalize the integer feature values to the range [0, 1],
but it also creates the two feature crosses.
Please choose one of the following two methods for data preprocessing.

#### Preprocessing by Pandas ####
```shell
# The preprocessing can take 40 minutes to 1 hour based on the system configuration.
$ cd ../../tools/criteo_script/
$ bash preprocess.sh wdl 1 1
$ cd ../../samples/wdl/
```
Convert the dataset to HugeCTR format
```shell
$ cp ../../build/bin/criteo2hugectr ./
$ ./criteo2hugectr ../../tools/criteo_script/wdl_data/train criteo/sparse_embedding file_list.txt 2
$ ./criteo2hugectr ../../tools/criteo_script/wdl_data/val criteo_test/sparse_embedding file_list_test.txt 2
```

#### Preprocessing by NVTabular ####
HugeCTR supports data processing by NVTabular since version 2.2.1. Please make sure NVTabular docker environment has been set up successfully according to [NVTAbular github](https://github.com/NVIDIA/NVTabular).
And bind mount HugeCTR ${project_home} volume to NVTabular docker. Run NVTabular docker and execute the following preprocessing commands.
Go to [(link)](http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/)
download kaggle-display dataset into the folder "${project_home}/samples/wdl/". 
```shell
$ tar zxvf dac.tar.gz 
$ mkdir -p wdl_data/train
$ mkdir -p wdl_data/val 
$ head -n 36672493 train.txt > wdl_data/train/train.txt 
$ tail -n 9168124 train.txt > wdl_data/val/test.txt 
$ cp ../../tools/criteo_script/preprocess_nvt.py ./
#The default output of NVTabular is the parquet format, if need the norm binary format, please add argument --parquet_format=0
$ python3 preprocess_nvt.py --src_csv_path=wdl_data/train/train.txt --dst_csv_path=wdl_data/train/ --normalize_dense=1 --feature_cross=1 --criteo_mode=0
$ python3 preprocess_nvt.py --src_csv_path=wdl_data/val/test.txt --dst_csv_path=wdl_data/val/ --normalize_dense=1 --feature_cross=1 --criteo_mode=0
```
Exit from the NVTabular docker environment and then run HugeCTR docker with interaction mode under home directory again.

2. Build HugeCTR with the instructions on README.md under home directory.


## Training with HugeCTR ##

1. Copy huge_ctr to samples/wdl
```shell
$ cp ../../build/bin/huge_ctr ./
```

2. Run huge_ctr

#### For Pandas Preprocessing ####
```shell
$ ./huge_ctr --train ./wdl.json
```

#### For NVTabular Preprocessing ####

Parquet output
```shell
$ ./huge_ctr --train ./wdl_parquet.json
```

Binary output
```shell
$ ./huge_ctr --train ./wdl_bin.json
```
