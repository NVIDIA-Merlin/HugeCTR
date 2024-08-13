#!/usr/bin/env bash

export DENSE_UNIQUE_RATIO=0.3
export WGRAD_UNIQUE_RATIO=0.3

test_config=$1
batchsize=$2
nsys_result=$3

test_command=""

hugectr_framework="
python3 /workdir/benchmarks/embedding_collection/hugectr/train.py \
--bmlp_layer_sizes 512,256,128 \
--tmlp_layer_sizes 1024,1024,512,256,1 \
--display_interval 10 \
--sharding_plan uniform \
--use_mixed_precision \
--mem_comm_bw_ratio 60 \
--dp_threshold 0 \
--dense_threshold 0 \
--disable_train_intra_iteration_overlap \
--disable_train_inter_iteration_overlap \
"
# --disable_train_intra_iteration_overlap \
# --disable_train_inter_iteration_overlap \
#  --disable_fuse_sparse_embedding
# --mem_comm_bw_ratio 3000/50 eos
# --mem_comm_bw_ratio 2000/25 selene
# --num_gpus_per_node 2 \
# --perf_logging

# torchx run -s local_cwd dist.ddp -j 1x2 --script torchrec_dlrm/dlrm_main.py --
torchrec_framework="
python3 /workdir/benchmarks/embedding_collection/torchrec_dlrm/dlrm_main.py  \
--dense_arch_layer_sizes 512,256,128 \
--over_arch_layer_sizes 1024,1024,512,256,1 \
--dcn_num_layers=3 \
--dcn_low_rank_dim=512 \
--learning_rate 0.005 \
--print_sharding_plan \
--use_criteo_multi_hot_data \
--bmlp_overlap \
--enable_cuda_graph \
"

# test_command=$hugectr_framework
test_command=$torchrec_framework

utest_dataset="
--dataset_path /home/scratch.aleliu_sw/hugectr_workdir/hugectr/benchmarks/embedding_collection/dataset/180table_70B_hotness80_synthetic_alpha2.bin.part0 \
--train_num_samples 6553600 \
--eval_num_samples 6553600 \
--num_table 1,1,5,2 \
--vocabulary_size_per_table 5000,5000,1000,5000 \
--nnz_per_table 100,100,100,10 \
--ev_size_per_table 4,8,4,8 \
--dense_dim 13 \
"
dataset_180table_70B_hotness80="
--dataset_path /workdir/dataset/180table_70B_hotness80_synthetic_alpha1.1.bin \
--train_num_samples 6553600 \
--eval_num_samples 6553600 \
--num_table 5,5,5,5,20,30,10,20,10,10,10,5,40,1,1 \
--vocabulary_size_per_table 10000,4000000,4000000,50000000,1000,10000,5000000,4000000,10,1000,10000,100000,4000000,50000000,500000000 \
--nnz_per_table 100,50,30,50,50,30,20,20,100,10,100,100,200,100,100 \
--ev_size_per_table 128,64,64,32,128,128,256,128,128,64,128,64,64,128,32 \
--dense_dim 13 \
--optimizer sgd \
"
dataset_180table_70B_hotness80_alpha2="
--dataset_path /workdir/dataset/180table_70B_hotness80_synthetic_alpha2.bin \
--train_num_samples 6553600 \
--eval_num_samples 6553600 \
--num_table 5,5,5,5,20,30,10,20,10,10,10,5,40,1,1 \
--vocabulary_size_per_table 10000,4000000,4000000,50000000,1000,10000,5000000,4000000,10,1000,10000,100000,4000000,50000000,500000000 \
--nnz_per_table 100,50,30,50,50,30,20,20,100,10,100,100,200,100,100 \
--ev_size_per_table 128,64,64,32,128,128,256,128,128,64,128,64,64,128,32 \
--dense_dim 13 \
--optimizer sgd \
"
dataset_7table_470B_hotness20="
--dataset_path /workdir/dataset/7table_470B_hotness20_synthetic_alpha1.1.bin \
--train_num_samples 6553600 \
--eval_num_samples 6553600 \
--num_table 1,1,1,1,1,1,1 \
--vocabulary_size_per_table 10000000,400000000,1000000000,5000000000,1000000000,10000000,10000000 \
--nnz_per_table 80,20,20,40,1,1,1 \
--ev_size_per_table 256,64,128,32,128,64,128 \
--dense_dim 13 \
--i64_input_key \
--optimizer sgd \
"
dataset_dcnv2="
--dataset_path /data/train_data.bin \
--train_num_samples 6553600 \
--eval_num_samples 6553600 \
--num_table 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1 \
--vocabulary_size_per_table 40000000,39060,17295,7424,20265,3,7122,1543,63,40000000,3067956,405282,10,2209,11938,155,4,976,14,40000000,40000000,40000000,590152,12973,108,36 \
--nnz_per_table 3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1 \
--ev_size_per_table 128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128 \
--dense_dim 13 \
--optimizer adagrad \
"
dataset_510table_110B_hotness5="
--dataset_path /workdir/dataset/510table_110B_hotness5_synthetic_alpha1.1.bin \
--train_num_samples 6553600 \
--eval_num_samples 6553600 \
--num_table 100,150,20,50,150,20,20 \
--vocabulary_size_per_table 1000,100000,1000000,2000000,4000000,4000000,4000000 \
--nnz_per_table 1,1,1,1,1,10,100 \
--ev_size_per_table 128,128,128,128,128,128,128 \
--dense_dim 13 \
--optimizer sgd \
"
dataset_510table_110B_hotness5_alpha2="
--dataset_path /workdir/dataset/510table_110B_hotness5_synthetic_alpha2.bin \
--train_num_samples 6553600 \
--eval_num_samples 6553600 \
--num_table 100,150,20,50,150,20,20 \
--vocabulary_size_per_table 1000,100000,1000000,2000000,4000000,4000000,4000000 \
--nnz_per_table 1,1,1,1,1,10,100 \
--ev_size_per_table 128,128,128,128,128,128,128 \
--dense_dim 13 \
--optimizer sgd \
"
dataset_200table_100B_hotness20="
--dataset_path /workdir/dataset/200table_100B_hotness20_synthetic_alpha1.1.bin \
--train_num_samples 6553600 \
--eval_num_samples 6553600 \
--num_table 10,10,10,10,20,10,10,10,10,10,10,20,20,10,10,10,10 \
--vocabulary_size_per_table 100,1000,1000,10000,10000,10000,100000,1000000,2000000,2000000,4000000,4000000,2000000,4000000,4000000,4000000,50000000 \
--nnz_per_table 1,1,5,20,100,1,1,1,1,1,1,1,10,20,30,50,100 \
--ev_size_per_table 128,128,128,128,128,128,128,128,128,128,128,128,64,128,128,128,128 \
--dense_dim 13 \
--optimizer sgd \
"
dataset_200table_100B_hotness20_alpha2="
--dataset_path /workdir/dataset/200table_100B_hotness20_synthetic_alpha2.bin \
--train_num_samples 6553600 \
--eval_num_samples 6553600 \
--num_table 10,10,10,10,20,10,10,10,10,10,10,20,20,10,10,10,10 \
--vocabulary_size_per_table 100,1000,1000,10000,10000,10000,100000,1000000,2000000,2000000,4000000,4000000,2000000,4000000,4000000,4000000,50000000 \
--nnz_per_table 1,1,5,20,100,1,1,1,1,1,1,1,10,20,30,50,100 \
--ev_size_per_table 128,128,128,128,128,128,128,128,128,128,128,128,64,128,128,128,128 \
--dense_dim 13 \
--optimizer sgd \
"
case $test_config in
utest)
  test_command+="$utest_dataset"
  ;;
180table_70B_hotness80)
  test_command+="${dataset_180table_70B_hotness80}"
  ;;
7table_470B_hotness20)
  test_command+="${dataset_7table_470B_hotness20}"
  ;;
dcnv2)
  test_command+="${dataset_dcnv2}"
  ;;
510table_110B_hotness5)
  test_command+="${dataset_510table_110B_hotness5}"
  ;;
200table_100B_hotness20)
  test_command+="${dataset_200table_100B_hotness20}"
  ;;
*)
  echo "unknown test config"
  exit -1
  ;;
esac

if [ -z "$batchsize" ]; then
  echo "empty batchsize"
  exit -1
fi
test_command+="--batchsize $batchsize"

nsys_profile_command="
nsys profile  -s none \
-t cuda -f true \
-o $nsys_result \
-c cudaProfilerApi \
--cpuctxsw none --cuda-flush-interval 100 --capture-range-end stop --cuda-graph-trace=node
"

if [ ! -z "$nsys_result" ]; then
  test_command="$nsys_profile_command $test_command"
fi

echo $test_command
$test_command

#torchx run -s local_cwd dist.ddp -j 1x2 --script /workdir/benchmarks/embedding_collection/torchrec_dlrm/dlrm_main.py -- \
