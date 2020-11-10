# Multinode Training with HugeCTR
This is a tutorial on how to do multi-node training with HugeCTR. 

## On Cluster with Job Scheduler
If your cluster is already equipped with a job scheduler such as SLURM,
refer to the instructions in [dcn2nodes](../../samples/dcn2nodes/README.md).

## On Cluster from Scratch
Follow the instructions below.

### Requirements
* Make your nodes connected with one other, so that each node can visit another by ssh-nopassword
* A shared directory each node can visit. 
* OpenMPI >= 4.0. 
* Docker >= 19.03.8

### Prepare HugeCTR Executable, Config File and Dataset
* Prepare a dcn dataset (refer to [dcn2nodes](../../samples/dcn2nodes/README.md)) and cater the dataset in the same directory for all the nodes.

* Build a docker image (refer to [README](../README.md)) in each node. You can also build an image once and use `docker save`/`docker load` to distribute the same image to all the nodes.
  
* Build HugeCTR: Build HugeCTR with **multi-nodes training supported** (refer to [README](../README.md)) and copy it to the same shared directory in each node

* Config Json file
The [dcn8l8gpu2nodes.json](../../samples/dcn2nodes/dcn8l8gpu2nodes.json) is using 2 8-GPU nodes. You can change `"gpu"` setting based on your own environment, and then copy the config file into a directory where HugeCTR executable file is located.
    ```bash
    cp ../../samples/dcn2nodes/dcn8l8gpu2nodes.json ../../build/bin/
    ```

### Configure `run_multinode.sh`
The [run_multinode.sh](./run_multinode.sh) uses `mpirun` to start the built docker container in each node. To use it, you must specify the variables below.

* WORK_DIR: The parent path where you put the `hugectr` executable and your json config file. 
* BIN: the `hugectr` executable path relative to `WORK_DIR`
* CONFIG_NAME: Json config file path relative to `WORK_DIR`
* DATASET: The real dataset path.
* VOL_DATASET: The dataset path shown inside your docker container as a mapping from `DATASET`. HugeCTR only sees this path.
* IMAGENAME: The name of your Docker image.

After you set all the variables properly, run `bash run_multinode.sh` to start multi-node training.

## Advanced
1. MPI needs to use tcp to build connections among nodes. If there exist multiple network names for a given host, it may lead to no response or crash. You may need to exclude some network names by setting "btl_tcp_if_exclude". The detailed setting is dependent upon your environment.
2. You can also consider customizing arguments of `mpirun` to achieve better performance on your own system, e.g., setting --bind-to none or sockets. 
3. If you are using DGX1 or DGXA100, you can configure InfiniBand (IB) devices by running `source ./config_DGX1.sh` or `source ./config_DGXA100.sh` respectively.
