# Performance Regression for SparseOperationKit #
This folder contains the files used to obtain the performance for SOK. 

## requirements ##
TensorFlow 2.5 <br>
MPI <br>
cupy <br>
nvtx <br>
Nsight System

# DLRM Model #
DLRM is a standard CTRs model. <br>
Followed these blogs and codes to profiling DLRM with TensorFlow:
+ https://cloud.google.com/tpu/docs/tutorials/dlrm-dcn-2.x
+ https://github.com/tensorflow/models/tree/master/official/recommendation/ranking#overview

## Dataset ##
Use [Criteo Terabyte](https://labs.criteo.com/2013/12/download-terabyte-click-logs/) dataset.
1. Download the dataset. Follow these [instructions](https://labs.criteo.com/2013/12/download-terabyte-click-logs/).
2. If you have binary files created by HugeCTR team, you can skip those time consuming preprocessing steps, and just convert it to CSV files. If not, please follow [these instructions](https://github.com/tensorflow/models/tree/master/official/recommendation/ranking/preprocessing) to prepare dataset.
3. **[Optional]** If you used binary files created by HugeCTR team, then the following commands should be executed:
    + Convert train binary files to CSV files.
    ```shell
    $ python3 bin2csv.py \
        --input_file="/mnt/scratch/criteo_terabyte/train_data.bin" \
        --num_output_files=1024 \
        --output_path="./train/" \
        --save_prefix="train_" 
    ```
    + Convert test binary files to CSV files.
    ```shell
    $ python3 bin2csv.py \
        --input_file="/mnt/scratch/criteo_terabyte/test_data.bin" \
        --num_output_files=64 \
        --output_path="./test/" \
        --save_prefix="test_"
    ```

## profiling with TF ##
1. [Option1] Run the codes from [Repo](https://github.com/tensorflow/models.git)
    + Install Prerequisites. <br>
    Please be noted that the following commands will upgrade TensorFlow, so that you need to build SparseOperationKit again.
    ```shell
    $ pip install tensorflow-recommenders tensorflow-addons tensorflow_model_optimization gin-config pyyaml
    ```
    + Download codes from [Repo](https://github.com/tensorflow/models.git).
    ```shell
    $ git clone https://github.com/tensorflow/models.git
    ```
    + Run profiling commands
    ```shell
    $ bash run_tf_ranking.sh
    ```
2. [Option2] Run the codes writen in current folder.
    + Set common params
    ```shell
    $ export EMBEDDING_DIM=32
    ```
    + With MirroredStrategy
    ```shell
    $ python3 main.py \
        --global_batch_size=16384 \
        --train_file_pattern="./train/*.csv" \
        --test_file_pattern="./test/*.csv" \
        --embedding_layer="TF" \
        --embedding_vec_size=$EMBEDDING_DIM \
        --bottom_stack 512 256 $EMBEDDING_DIM \
        --top_stack 1024 1024 512 256 1 \
        --distribute_strategy="mirrored" \
        --gpu_num=4 \
        --TF_MP=1
    ```
    + With MultiWorkerMirroredStrategy
    ```shell
    $ mpiexec --allow-run-as-root -np 4 \
        python3 main.py \
            --global_batch_size=16384 \
            --train_file_pattern="./train/*.csv" \
            --test_file_pattern="./test/*.csv" \
            --embedding_layer="TF" \
            --embedding_vec_size=$EMBEDDING_DIM \
            --bottom_stack 512 256 $EMBEDDING_DIM \
            --top_stack 1024 1024 512 256 1 \
            --distribute_strategy="multiworker" \
            --TF_MP=1
    ```
    + With Horovod
    ```shell
    $ horovodrun -np 4 -H localhost:4 \
        python3 main.py \
            --global_batch_size=16384 \
            --train_file_pattern="./train/*.csv" \
            --test_file_pattern="./test/*.csv" \
            --embedding_layer="TF" \
            --embedding_vec_size=$EMBEDDING_DIM \
            --bottom_stack 512 256 $EMBEDDING_DIM \
            --top_stack 1024 1024 512 256 1 \
            --distribute_strategy="horovod" \
            --TF_MP=1
    ```

## profiling with SOK ##
+ With MirroredStrategy
```shell
$ python3 main.py \
    --global_batch_size=16384 \
    --train_file_pattern="./train/*.csv" \
    --test_file_pattern="./test/*.csv" \
    --embedding_layer="SOK" \
    --embedding_vec_size=$EMBEDDING_DIM \
    --bottom_stack 512 256 $EMBEDDING_DIM \
    --top_stack 1024 1024 512 256 1 \
    --distribute_strategy="mirrored" \
    --gpu_num=4 
```
+ With MultiWorkerMirroredStrategy
```shell
$ mpiexec --allow-run-as-root -np 4 \
    python3 main.py \
        --global_batch_size=16384 \
        --train_file_pattern="./train/*.csv" \
        --test_file_pattern="./test/*.csv" \
        --embedding_layer="SOK" \
        --embedding_vec_size=$EMBEDDING_DIM \
        --bottom_stack 512 256 $EMBEDDING_DIM \
        --top_stack 1024 1024 512 256 1 \
        --distribute_strategy="multiworker" 
```
+ With Horovod
```shell
$ horovodrun -np 4 -H localhost:4 \
    python3 main.py \
        --global_batch_size=16384 \
        --train_file_pattern="./train/*.csv" \
        --test_file_pattern="./test/*.csv" \
        --embedding_layer="SOK" \
        --embedding_vec_size=$EMBEDDING_DIM \
        --bottom_stack 512 256 $EMBEDDING_DIM \
        --top_stack 1024 1024 512 256 1 \
        --distribute_strategy="horovod"
```

# Demo Model #
## generate synthetic dataset ##
```shell
$ python3 gen_data.py \
    --global_batch_size=65536 \
    --slot_num=100 \
    --nnz_per_slot=10 \
    --iter_num=30 
```

## split whole dataset into multiple shards ##
```shell
$ python3 split_data.py \
    --filename="./data.file" \
    --split_num=8 \
    --save_prefix="data_"
```

## profiling tf on single GPU ##
```shell
$ python3 run_tf.py \
    --global_batch_size=8192 \
    --slot_num=100 \
    --nnz_per_slot=10 \
    --embedding_vec_size=4 \
    --num_dense_layers=6 \
    --vocabulary_size=8192 \
    --early_stop_iter=-1 \
    --filename="./data_0.file" \
    --data_splited=0 \
    --sparse_keys=0
```

## profiling sok on single GPU ##
```shell
$ python3 run_sok.py \
    --global_batch_size=8192 \
    --slot_num=100 \
    --nnz_per_slot=10 \
    --embedding_vec_size=4 \
    --num_dense_layers=6 \
    --vocabulary_size=8192 \
    --early_stop_iter=-1 \
    --filename="./data_0.file" \
    --data_splited=0 \
    --sparse_keys=0 \
    --whether_single_gpu=1
```

## profiling sok on multi GPU ##
```shell
$ mpiexec --allow-run-as-root -np 8 \
    python3 run_sok.py \
    --global_batch_size=65536 \
    --slot_num=100 \
    --nnz_per_slot=10 \
    --embedding_vec_size=4 \
    --num_dense_layers=6 \
    --vocabulary_size=8192 \
    --early_stop_iter=-1 \
    --filename="./data_" \
    --data_splited=1 \
    --sparse_keys=0 \
    --whether_single_gpu=0
```

## nsight system profiling command ##
```shell
$ nsys profile --sample=none --backtrace=none --cudabacktrace=none --cpuctxsw=none -f true -o filename --trace-fork-before-exec=true executable [executable-args]
```

## profiling shell script ##
You can run the following code to collect nsys profiling timelines.
```shell
$ bash performance_profiling.sh
```