#!/usr/bin/env bash

framework=$1
test_config=$2
batchsize=$3
nsys_result=$4

test_command=""

hugectr_framework="
python3 /workdir/benchmarks/embedding_collection/hugectr/train.py \
--sharding_plan hier_auto \
--optimizer adagrad
"
#  --disable_fuse_sparse_embedding
hugectr_framework_disable_fuse_sparse_embedding="
python3 /workdir/benchmarks/embedding_collection/hugectr/train.py \
--sharding_plan hier_auto \
--optimizer adagrad \
--disable_fuse_sparse_embedding
"

torchrec_framework="
python3 /workdir/benchmarks/embedding_collection/torchrec_dlrm/dlrm_main.py  \
  --dcn_num_layers=3 \
  --dcn_low_rank_dim=512 \
  --adagrad \
  --learning_rate 0.005 \
  --print_sharding_plan
"
# --pin_memory \

case $framework in
hugectr_disable_fuse_sparse)
  test_command=$hugectr_framework_disable_fuse_sparse_embedding
  ;;
hugectr)
  test_command=$hugectr_framework
  ;;
torchrec)
  test_command=$torchrec_framework
  ;;
*)
  echo "unknown framework"
  exit -1
  ;;
esac

utest_dataset="
--dataset_path /workdir/benchmarks/embedding_collection/dcnv2_synthetic_alpha1.01.bin \
--train_num_samples 6553600 \
--eval_num_samples 6553600 \
--num_table 1,1,1,1 \
--vocabulary_size_per_table 100,1000,100,1000 \
--nnz_per_table 1,2,1,1 \
--ev_size_per_table 4,8,4,8 \
--combiner_per_table c,s,s,c \
--dense_dim 13 \
--bmlp_layer_sizes 512,256,128 \
--tmlp_layer_sizes 1024,1024,512,256,1 \
"
#large_dataset="
#--dataset_path /workdir/dataset/large_dcnv2_synthetic_alpha1.1.bin \
#--train_num_samples 6553600 \
#--eval_num_samples 6553600 \
#--num_table 20,16,1,1,1,80,100,200,50,80 \
#--vocabulary_size_per_table 100000,15000000,200000000,200000000,200000000,15000000,10000,100000,500000,15000000 \
#--nnz_per_table 100,100,100,1,1,1,1,1,1,1 \
#--ev_size_per_table 64,64,128,128,128,32,32,64,64,64 \
#--dense_dim 13 \
#--combiner_per_table s,s,s,c,c,c,c,c,c,c \
#--bmlp_layer_sizes 512,256,128 \
#--tmlp_layer_sizes 1024,1024,512,256,1 \
#"
middle_dataset="
--dataset_path /workdir/dataset/middle_dcnv2_synthetic_alpha1.1.bin \
--train_num_samples 6553600 \
--eval_num_samples 6553600 \
--num_table 5,3,5,1,1,10,10,10,5,30 \
--vocabulary_size_per_table 10000,4000000,50000000,50000000,50000000,10,1000,10000,100000,4000000 \
--nnz_per_table 100,30,30,1,1,1,1,1,1,1 \
--ev_size_per_table 8,128,128,8,128,128,128,8,8,128 \
--dense_dim 13 \
--combiner_per_table s,s,s,c,c,c,c,c,c,c \
--bmlp_layer_sizes 512,256,128 \
--tmlp_layer_sizes 1024,1024,512,256,1 \
"
middle_dataset_only_sparse="
--dataset_path /workdir/dataset/middle_dcnv2_synthetic_alpha1.1.bin \
--train_num_samples 6553600 \
--eval_num_samples 6553600 \
--num_table 5,3,5,1,1,10,10,10,5,30 \
--vocabulary_size_per_table 10000,4000000,50000000,50000000,50000000,10,1000,10000,100000,4000000 \
--nnz_per_table 100,30,30,1,1,1,1,1,1,1 \
--ev_size_per_table 8,128,128,8,128,128,128,8,8,128 \
--dense_dim 13 \
--combiner_per_table s,s,s,s,s,s,s,s,s,s \
--bmlp_layer_sizes 512,256,128 \
--tmlp_layer_sizes 1024,1024,512,256,1 \
"
middle_dataset_small_vec="
--dataset_path /workdir/dataset/middle_dcnv2_synthetic_alpha1.1.bin \
--train_num_samples 6553600 \
--eval_num_samples 6553600 \
--num_table 5,3,5,1,1,10,10,10,5,30 \
--vocabulary_size_per_table 10000,4000000,50000000,50000000,50000000,10,1000,10000,100000,4000000 \
--nnz_per_table 100,30,30,1,1,1,1,1,1,1 \
--ev_size_per_table 4,4,4,4,4,4,4,4,4,4 \
--dense_dim 13 \
--combiner_per_table s,s,s,c,c,c,c,c,c,c \
--bmlp_layer_sizes 512,256,128 \
--tmlp_layer_sizes 1024,1024,512,256,1 \
"
middle_dataset_small_vec_only_sparse="
--dataset_path /workdir/dataset/middle_dcnv2_synthetic_alpha1.1.bin \
--train_num_samples 6553600 \
--eval_num_samples 6553600 \
--num_table 5,3,5,1,1,10,10,10,5,30 \
--vocabulary_size_per_table 10000,4000000,50000000,50000000,50000000,10,1000,10000,100000,4000000 \
--nnz_per_table 100,30,30,1,1,1,1,1,1,1 \
--ev_size_per_table 4,4,4,4,4,4,4,4,4,4 \
--dense_dim 13 \
--combiner_per_table s,s,s,s,s,s,s,s,s,s \
--bmlp_layer_sizes 512,256,128 \
--tmlp_layer_sizes 1024,1024,512,256,1 \
"
middle_hotness20_dataset="
--dataset_path /workdir/dataset/middle_dcnv2_synthetic_alpha1.1.hotness20.bin \
--train_num_samples 6553600 \
--eval_num_samples 6553600 \
--num_table 10,3,20,1,1,10,10,10,5,10 \
--vocabulary_size_per_table 10000,4000000,50000000,50000000,50000000,10,1000,10000,100000,4000000 \
--nnz_per_table 100,30,30,1,1,1,1,1,1,1 \
--ev_size_per_table 8,128,128,8,128,128,128,8,8,128 \
--dense_dim 13 \
--combiner_per_table s,s,s,c,c,c,c,c,c,c \
--bmlp_layer_sizes 512,256,128 \
--tmlp_layer_sizes 1024,1024,512,256,1 \
"
middle_hotness20_dataset_only_sparse="
--dataset_path /workdir/dataset/middle_dcnv2_synthetic_alpha1.1.hotness20.bin \
--train_num_samples 6553600 \
--eval_num_samples 6553600 \
--num_table 10,3,20,1,1,10,10,10,5,10 \
--vocabulary_size_per_table 10000,4000000,50000000,50000000,50000000,10,1000,10000,100000,4000000 \
--nnz_per_table 100,30,30,1,1,1,1,1,1,1 \
--ev_size_per_table 8,128,128,8,128,128,128,8,8,128 \
--dense_dim 13 \
--combiner_per_table s,s,s,s,s,s,s,s,s,s \
--bmlp_layer_sizes 512,256,128 \
--tmlp_layer_sizes 1024,1024,512,256,1 \
"
middle_hotness70_dataset="
--dataset_path /workdir/dataset/middle_dcnv2_synthetic_alpha1.1.hotness70.bin \
--train_num_samples 6553600 \
--eval_num_samples 6553600 \
--num_table 10,3,20,1,1,10,10,10,5,10 \
--vocabulary_size_per_table 10000,4000000,50000000,50000000,50000000,10,1000,10000,100000,4000000 \
--nnz_per_table 100,30,30,100,100,80,100,80,100,50 \
--ev_size_per_table 8,128,128,8,128,128,128,8,8,128 \
--dense_dim 13 \
--combiner_per_table s,s,s,s,s,s,s,s,s,s \
--bmlp_layer_sizes 512,256,128 \
--tmlp_layer_sizes 1024,1024,512,256,1 \
"
case $test_config in
utest)
  test_command+="$utest_dataset"
  ;;
middle)
  test_command+="$middle_dataset"
  ;;
middle_only_sparse)
  test_command+="$middle_dataset_only_sparse"
  ;;
middle_small_vec)
  test_command+="$middle_dataset_small_vec"
  ;;
middle_small_vec_only_sparse)
  test_command+="$middle_dataset_small_vec_only_sparse"
  ;;
middle20)
  test_command+="$middle_hotness20_dataset"
  ;;
middle70)
  test_command+="$middle_hotness70_dataset"
  ;;
middle20_only_sparse)
  test_command+="$middle_hotness20_dataset_only_sparse"
  ;;
large)
  test_command+="$large_dataset"
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

# nsys profile  -s none -t cuda -f true -o /workdir/16node_large_alpha1.1_nsys/16node_large_alpha1.1_sparse_and_dense_65536_%h_%p  -  c cudaProfilerApi  --cpuctxsw none --cuda-flush-interval 100 --capture-range-end stop --cuda-graph-trace=node \
# python3 /workdir/benchmarks/embedding_collection/hugectr/train.py \
#     --dataset_path /workdir/dataset/large_dcnv2_synthetic_alpha1.1.bin \
#     --batchsize 65536 \
#     --train_num_samples 6553600 \
#     --eval_num_samples 6553600 \
#     --sharding_plan hier_auto \
#     --num_table 20,16,1,1,1,80,100,200,50,80 \
#     --vocabulary_size_per_table 100000,15000000,200000000,200000000,200000000,15000000,10000,100000,500000,15000000 \
#     --nnz_per_table 100,100,100,1,1,1,1,1,1,1 \
#     --ev_size_per_table 64,64,128,128,128,32,32,64,64,64 \
#     --combiner_per_table s,s,s,c,c,c,c,c,c,c \
#     --dense_dim 13 \
#     --bmlp_layer_sizes 512,256,128 \
#     --tmlp_layer_sizes 1024,1024,512,256,1 \
#     --optimizer sgd

#nsys profile -s none -t cuda -f true -o /workdir/16node_large_alpha1.1_nsys/16node_large_alpha1. 1_sparse_65536_diable_fuse_sparse_embedding_%h_%p -c cudaProfilerApi --cpuctxsw none --cuda-flush-interval 100 --capture-range-end stop --cuda-graph-trace=node \
#  python3 /workdir/benchmarks/embedding_collection/hugectr/train.py \
#  --dataset_path /workdir/dataset/large_dcnv2_synthetic_alpha1.1.bin \
#  --batchsize 65536 \
#  --train_num_samples 6553600 \
#  --eval_num_samples 6553600 \
#  --sharding_plan hier_auto \
#  --num_table 20,16,1,1,1,80,100,200,50,80 \
#  --vocabulary_size_per_table 100000,15000000,200000000,200000000,200000000,15000000,10000,100000,500000,15000000 \
#  --nnz_per_table 100,100,100,1,1,1,1,1,1,1 \
#  --ev_size_per_table 64,64,128,128,128,32,32,64,64,64 \
#  --combiner_per_table s,s,s,s,s,s,s,s,s,s \
#  --dense_dim 13 \
#  --bmlp_layer_sizes 512,256,128 \
#  --tmlp_layer_sizes 1024,1024,512,256,1 \
#  --optimizer sgd \
#  --disable_fuse_sparse_embedding

#python3 /workdir/benchmarks/embedding_collection/torchrec_dlrm/dlrm_main.py \
#  --dataset_path /workdir/dataset/large_dcnv2_synthetic_alpha1.1.bin \
#  --batch_size_per_gpu 512 \
#  --limit_train_batches 6553600 \
#  --num_table 20,16,1,1,1,80,100,200,50,80 \
#  --vocabulary_size_per_table 100000,15000000,200000000,200000000,200000000,15000000,10000,100000,500000,15000000 \
#  --nnz_per_table 100,100,100,1,1,1,1,1,1,1 \
#  --ev_size_per_table 64,64,128,128,128,32,32,64,64,64 \
#  --combiner_per_table s,s,s,s,s,s,s,s,s,s \
#  --dense_dim 13 \
#  --dense_arch_layer_sizes 512,256,128 \
#  --over_arch_layer_sizes 1024,1024,512,256,1 \
#  --pin_memory \
#  --dcn_num_layers=3 \
#  --dcn_low_rank_dim=512 \
#  --adagrad \
#  --learning_rate 0.005

#torchx run -s local_cwd dist.ddp -j 1x2 --script /workdir/benchmarks/embedding_collection/torchrec_dlrm/dlrm_main.py -- \
