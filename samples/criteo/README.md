# CRITEO CTR SAMPLE #
In this sample we aim to demonstrate the basic usage of huge_ctr.

## Dataset and preprocess ##
In running this sample, [Criteo 1TB Click Logs dataset](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/) is used.
The dataset contains 24 files, each of which corresponds to one day of data.
To spend less time on preprocessing, we use only one of them.
Each sample consists of a label (1 if the ad was clicked, otherwise 0) and 39 features (13 integer features and 26 categorical features).
The dataset also has the significant amounts of missing values across the feature columns, which should be preprocessed accordingly.


### 1. Download the dataset and preprocess

Go to [this link](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/),
and download one of 24 files into the directory "${project_root}/tools", 
or execute the following command:
```
$ cd ${project_root}/tools
$ wget http://azuremlsampleexperiments.blob.core.windows.net/criteo/day_1.gz
```
- **NOTE**: Replace 1 with a value from [0, 23] to use a different day.

In preprocessing, we will further reduce the amounts of data to speedup the preprocessing, fill missing values, remove the feature values whose occurrences are very rare, etc.
Please choose one of the following two methods to make the dataset ready for HugeCTR training.

#### Preprocessing by Perl ####
```shell
$ bash preprocess.sh 1 criteo_data perl 1
```
- **NOTE**: The first argument represents the dataset postfix.  For instance, if `day_1` is used, it is 1.
- **NOTE**: the second argument `criteo_data` is where the preprocessed data is stored.
You may want to change it in case where multiple datasets for different purposes are generated concurrently.
If you change it, `source` and `eval_source` in your JSON config file must be changed as well.
- **NOTE**: the last arguemnt represents the number of slots used in this sample.

#### Preprocessing by NVTabular ####

HugeCTR supports data processing by NVTabular since version 2.2.1.
Please make sure NVTabular docker environment has been set up successfully according to [NVTAbular github](https://github.com/NVIDIA/NVTabular).
Make sure to use the latest version of NVTabular,
and mount HugeCTR ${project_root} volume to NVTabular docker.
Run NVTabular docker and execute the following preprocessing commands:
```shell
$ bash preprocess.sh 1 criteo_data nvt 1 1 0
```
- **NOTE**: The first and second arguments are as the same as Perl's (see above).
- **NOTE**: If you want to generate a binary data in `Norm` format data, instead of the Parquet format data, set the fourth argument (one after `nvt`) to 0. It can take much longer than the Parquet mode becuase of the additional conversion process.
Otherwise, a Parquet dataset is generated. Use this NVTabular binary mode if you encounter an  issue with the Pandas mode.
- **NOTE**: the fifth argument(one after `nvt`)  must be set to 1 for `criteo` sample. Otherwise, it is 0.
- **NOTE**: the last argument decides if the feature crossing is applied (1=ON, 0=OFF).
It must remains 0 unless the sample is not `wdl`.

Exit from the NVTabular docker environment and then run HugeCTR docker with interaction mode under home directory again.

### 2. Build HugeCTR with the instructions on README.md under home directory.


## Training with HugeCTR ##

#### After Perl Preprocessing ####
```shell
$ ../build/bin/huge_ctr --train ../samples/criteo/criteo.json
```

#### After NVTabular Preprocessing ####

Parquet output
```shell
$ ../build/bin/huge_ctr --train ../samples/criteo/criteo_parquet.json
```
Binary output (See NOTE above)
```shell
$ ../build/bin/huge_ctr --train ../samples/criteo/criteo_bin.json
```
