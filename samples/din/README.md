# DIN CTR SAMPLE #
The purpose of this sample is to demonstrate how to build and train a [Deep Interest Network](https://arxiv.org/pdf/1706.06978.pdf) with HugeCTR.

## Set Up the HugeCTR Docker Environment ##
You can set up the HugeCTR Docker environment by doing one of the following:
- [Pull the NGC Docker](#pull-the-ngc-docker)
- [Build the HugeCTR Docker Container on Your Own](#build-the-hugectr-docker-container-on-your-own)

### Pull the NGC Docker ###
HugeCTR is available as buildable source code, but the easiest way to install and run HugeCTR is to pull the pre-built Docker image, which is available on the NVIDIA GPU Cloud (NGC). This method provides a self-contained, isolated, and reproducible environment for repetitive experiments.

1. Pull the HugeCTR NGC Docker by running the following command:
   ```bash
   $ docker pull nvcr.io/nvidia/merlin/merlin-training:22.05
   ```
2. Launch the container in interactive mode with the HugeCTR root directory mounted into the container by running the following command:
   ```bash
   $ docker run --gpus=all --rm -it --cap-add SYS_NICE -u $(id -u):$(id -g) -v $(pwd):/hugectr -w /hugectr nvcr.io/nvidia/merlin/merlin-training:22.05
   ```

### Build the HugeCTR Docker Container on Your Own ###
If you want to build the HugeCTR Docker container on your own, refer to [How to Start Your Development](https://nvidia-merlin.github.io/HugeCTR/master/hugectr_contributor_guide.html#how-to-start-your-development).

You should make sure that HugeCTR is built and installed in `/usr/local/hugectr` within the Docker container. Remember to set the option `ENABLE_MULTINODES` as `ON` when building HugeCTR if you want to try the multinode training sample. You can launch the container in interactive mode in the same manner as shown above, and then set the `PYTHONPATH` environment variable inside the Docker container using the following command:
```shell
$ export PYTHONPATH=/usr/local/hugectr/lib:$PYTHONPATH
``` 

## Download the Dataset ##
Make a folder under the project root directory to store the raw data:
```bash
cd ${project_root}
mkdir raw_data
```
Go [here](https://jmcauley.ucsd.edu/data/amazon/) and download the `review_Electronics_5` and `meta_Electronics` dataset files into the "${project_root}/raw_data" directory. 

As an alternative, you can run the following command:
```bash
cd raw_data
wget -c http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz
gzip -d reviews_Electronics_5.json.gz
wget -c http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Electronics.json.gz
gzip -d meta_Electronics.json.gz
```
## Preprocess the Dataset ##
Make a folder under the project root directory to store the processed data:
```bash
mkdir din_data
```
Run python scripts to preprocess the data:
```bash
cd utils
bash preprocess.sh
```

## Train with HugeCTR ##
With the above steps, the preprocessed data is saved locally in the `din_data/` directory. To train the DIN model using the data, simply run:
```
python din_parquet.py
```
By default, this will run 8000 iterations and the peak of the AUC could reach higher than 0.8.

There is another implementation of the DIN model in `din_try.py`, which employs the Matrix Multiply layer. It can achieve AUC 0.75 after 88,000 iterations.
