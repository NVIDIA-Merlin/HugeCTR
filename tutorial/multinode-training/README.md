# Multinode training in HugeCTR
A tutorial about how to do multinode training in HugeCTR. 

## If You Have a GPU Cluster
If you have GPU cluster like SLURM system, you can follow steps in [dcn2nodes](../../samples/dcn2nodes/README.md).

## If You Do Not Have a GPU Cluster
### Requirements
* Make your nodes connected with one other, so that each node can visit another by ssh-nopassword
* A shared directory each node can visit. 
* OpenMPI >= 4.0. 
* Docker >= 19.03.8

### Prepare hugectr training resources

* Datasets: Prepare dcn dataset (refer to [dcn2nodes](../../samples/dcn2nodes/README.md)) and need put dataset in the same directory in each node.

* Build Docker Image: Build a docker image (refer to [README](../README.md)) in each node. You can also build an image once and use `docker save`/`docker load` to distribute the same image to all the nodes.
  
* Build HugeCTR: Build HugeCTR with **multi-nodes training supported** (refer to [README](../README.md)) and copy it to the same shared directory in each node

* Config Jsonfile: The [dcn8l8gpu2nodes.json](../../samples/dcn2nodes/dcn8l8gpu2nodes.json) is using 2 8-GPU nodes. You can change `"gpu"` setting according to your needs. You can change `"gpu"` setting based on your own environment, and then copy the config file into the directory where HugeCTR executable file is located.
    ```bash
    cp ../../samples/dcn2nodes/dcn8l8gpu2nodes.json ../../build/bin/
    ```

### Configure `run_multinode.sh`
The [run_multinode.sh](./run_multinode.sh) uses mpirun to start docker container in each. To use it, you must specify the variables below..

* WORK_DIR: The path under which you put hugectr bin file and your json config. 
* BIN: Hugectr bin file path relative to WORK_DIR
* CONFIG_NAME: Json config path relative to WORK_DIR
* DATASET: Dataset path in disk.
* VOL_DATASET: Dataset path in docker container. This is the path hugectr really use.
* IMAGENAME: Docker image name

After you set all the variables properly, you can use `bash run_multinode.sh` to start multinode training.

## Advanced
1. MPI needs to use tcp to build connection among servers. If there are multi network names on the host, it may be leading MPI hang or crash. You may need to exclude some network names by setting "btl_tcp_if_exclude". You can adjust these parts according to your environment.
2. You can also adjust mpirun args for better performance on your special platform. Example, setting --bind-to none or sockets. 
3. If you are using DGX1 or DGXA100, you can configure IB Devices by `source ./config_DGX1.sh` or `source ./config_DGXA100.sh` accordingly.