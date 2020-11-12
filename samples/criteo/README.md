# CRITEO CTR SAMPLE #
In this sample we aim to demonstrate the basic usage of huge_ctr.

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
Please choose one of the following two methods for data preprocessing.

#### Preprocessing by Perl ####
```shell
$ cd ../../tools/criteo_script_legacy/
$ bash preprocess.sh
$ cd ../../samples/criteo/
```
Translate the dataset to HugeCTR format
```shell
$ cp ../../build/bin/criteo2hugectr_legacy ./
$ ./criteo2hugectr_legacy 1 ../../tools/criteo_script_legacy/train.out criteo/sparse_embedding file_list.txt
$ ./criteo2hugectr_legacy 1 ../../tools/criteo_script_legacy/test.out criteo_test/sparse_embedding file_list_test.txt
```

#### Preprocessing by NVTabular ####

HugeCTR supports data processing by NVTabular since version 2.2.1. Please make sure NVTabular docker environment has been set up successfully according to [NVTAbular github](https://github.com/NVIDIA/NVTabular). Make sure to use the latest version(0.2) of NVTabular.
And bind mount HugeCTR ${project_home} volume to NVTabular docker. Run NVTabular docker and execute the following preprocessing commands.
Go to [(link)](http://labs.criteo.com/2014/02/kaggle-display-advertising-challenge-dataset/)
download kaggle-display dataset into the folder "${project_home}/samples/criteo/". 
```shell
$ tar zxvf dac.tar.gz 
$ mkdir -p criteo_data/train
$ mkdir -p criteo_data/val 
$ head -n 36672493 train.txt > criteo_data/train/train.txt 
$ tail -n 9168124 train.txt > criteo_data/val/test.txt 
$ cp ../../tools/criteo_script/preprocess_nvt.py ./
#--help:show help message and explan usage of each parameters.
#--parquet_format=1 The default output of NVTabular is the parquet format, if need the norm binary format, please add argument with 0
#--device_limit_frac：Worker device-memory limit as a fraction of GPU capacity, which should be determined by the gpu with the leatest memory
#--device_pool_frac：The RMM pool frac is the same for all GPUs, make sure each one has enough memory size
#--num_io_threads: Number of threads to use when writing output data.
$ python3 preprocess_nvt.py --data_path criteo_data/train/train.txt --out_path criteo_data/train/ --freq_limit 6 --device_limit_frac 0.2 --device_pool_frac 0.2 --out_files_per_proc 8  --devices "0" --num_io_threads 2 --criteo_mode=1
$ python3 preprocess_nvt.py --data_path criteo_data/val/test.txt --out_path criteo_data/val/ --freq_limit 6 --device_limit_frac 0.2 --device_pool_frac 0.2 --out_files_per_proc 8  --devices "0" --num_io_threads 2 --criteo_mode=1
```
Exit from the NVTabular docker environment and then run HugeCTR docker with interaction mode under home directory again.

2. Build HugeCTR with the instructions on README.md under home directory.


## Training with HugeCTR ##

1. Copy huge_ctr to samples/criteo
```shell
$ cp ../../build/bin/huge_ctr ./
```

2. Run huge_ctr

#### For Pandas Preprocessing ####
```shell
$ ./huge_ctr --train ./criteo.json
```

#### For NVTabular Preprocessing ####

Parquet output
```shell
$ ./huge_ctr --train ./criteo_parquet.json
```
Binary output
```shell
$ ./huge_ctr --train ./criteo_bin.json
```

