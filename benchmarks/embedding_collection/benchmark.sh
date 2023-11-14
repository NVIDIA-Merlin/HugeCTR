#!/usr/bin/env bash

framework=$1
test_config=$2
batchsize=$3
nsys_result=$4

test_command=""

hugectr_framework="
python3 /workdir/benchmarks/embedding_collection/hugectr/train.py \
--sharding_plan hier_auto \
--optimizer sgd \
--use_mixed_precision \
--use_column_wise_shard \
--dp_threshold 0
"
# --disable_train_intra_iteration_overlap \
# --disable_train_inter_iteration_overlap \
#  --disable_fuse_sparse_embedding
# --mem_comm_bw_ratio 3000/50 eos
# --mem_comm_bw_ratio 2000/25 selene

hugectr_framework_disable_fuse_sparse_embedding="
python3 /workdir/benchmarks/embedding_collection/hugectr/train.py \
--sharding_plan hier_auto \
--optimizer sgd \
--use_mixed_precision \
--disable_fuse_sparse_embedding
"

torchrec_framework="
python3 /workdir/benchmarks/embedding_collection/torchrec_dlrm/dlrm_main.py  \
  --dcn_num_layers=3 \
  --dcn_low_rank_dim=512 \
  --learning_rate 0.005 \
  --print_sharding_plan \
  --skip_h2d
"
# --pin_memory \
# --adagrad \
# --skip_input_dist \

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
--num_table 2,2 \
--vocabulary_size_per_table 1000,10000 \
--nnz_per_table 1,1 \
--ev_size_per_table 4,8 \
--combiner_per_table s,c \
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
dataset_80table_55B_hotness10="
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
dataset_80table_55B_hotness10_only_sparse="
--dataset_path /workdir/dataset/middle_dcnv2_synthetic_alpha1.1.bin \
--train_num_samples 6553600 \
--eval_num_samples 6553600 \
--num_table 5,3,5,1,1,10,10,10,5,30 \
--vocabulary_size_per_table 10000,4000000,50000000,50000000,50000000,10,1000,10000,100000,4000000 \
--nnz_per_table 100,30,30,1,1,1,1,1,1,1 \
--ev_size_per_table 256,128,128,8,128,128,128,8,8,128 \
--dense_dim 13 \
--combiner_per_table s,s,s,s,s,s,s,s,s,s \
--bmlp_layer_sizes 512,256,128 \
--tmlp_layer_sizes 1024,1024,512,256,1 \
"
dataset_80table_55B_hotness10_small_vec="
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
dataset_80table_55B_hotness10_small_vec_only_sparse="
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
dataset_80table_55B_hotness20="
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
dataset_80table_55B_hotness20_only_sparse="
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
dataset_80table_55B_hotness70="
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
dataset_130table_110B_hotness20="
--dataset_path /workdir/dataset/scale_dcnv2_synthetic_alpha1.1.bin \
--train_num_samples 6553600 \
--eval_num_samples 6553600 \
--num_table 10,5,5,5,10,10,10,10,10,10,10,5,30,2 \
--vocabulary_size_per_table 10000,4000000,4000000,50000000,1000,10000,50000000,4000000,10,10000,10000,100000,4000000,50000000 \
--nnz_per_table 100,50,30,50,50,30,2,2,1,1,1,1,1,1 \
--ev_size_per_table 64,256,64,32,64,32,128,128,128,128,128,128,128,128 \
--dense_dim 13 \
--combiner_per_table s,s,s,s,s,s,s,s,s,s,s,s,s,s \
--bmlp_layer_sizes 512,256,128 \
--tmlp_layer_sizes 1024,1024,512,256,1 \
"
dataset_180table_130B_hotness80="
--dataset_path /workdir/dataset/large_hotness_dcnv2_synthetic_alpha1.1.bin \
--train_num_samples 6553600 \
--eval_num_samples 6553600 \
--num_table 5,5,5,5,20,30,10,20,10,10,10,5,40,1,1 \
--vocabulary_size_per_table 10000,4000000,4000000,50000000,1000,10000,5000000,4000000,10,1000,10000,100000,4000000,50000000,500000000 \
--nnz_per_table 100,50,30,50,50,30,20,20,100,10,100,100,200,100,100 \
--ev_size_per_table 64,256,256,128,64,32,128,128,128,64,128,64,32,64,128 \
--dense_dim 13 \
--combiner_per_table s,s,s,s,s,s,s,s,s,s,s,s,s,s,s \
--bmlp_layer_sizes 512,256,128 \
--tmlp_layer_sizes 1024,1024,512,256,1 \
"
dataset_7table_470B_hotness20="
--dataset_path /workdir/dataset/large_tables_dcnv2_synthetic_alpha1.1.bin \
--train_num_samples 6553600 \
--eval_num_samples 6553600 \
--num_table 1,1,1,1,1,1,1 \
--vocabulary_size_per_table 10000000,400000000,1000000000,5000000000,1000000000,10000000,10000000 \
--nnz_per_table 80,20,20,40,1,1,1 \
--ev_size_per_table 256,64,128,32,128,64,128 \
--dense_dim 13 \
--combiner_per_table s,s,s,s,s,s,s \
--bmlp_layer_sizes 512,256,128 \
--tmlp_layer_sizes 1024,1024,512,256,1 \
--i64_input_key \
"
case $test_config in
utest)
  test_command+="$utest_dataset"
  ;;
80table_55B_hotness10)
  test_command+="${dataset_80table_55B_hotness10}"
  ;;
80table_55B_hotness10_only_sparse)
  test_command+="${dataset_80table_55B_hotness10_only_sparse}"
  ;;
80table_55B_hotness10_small_vec)
  test_command+="${dataset_80table_55B_hotness10_small_vec}"
  ;;
80table_55B_hotness10_small_vec_only_sparse)
  test_command+="${dataset_80table_55B_hotness10_small_vec_only_sparse}"
  ;;
80table_55B_hotness20)
  test_command+="${dataset_80table_55B_hotness20}"
  ;;
80table_55B_hotness70)
  test_command+="${dataset_80table_55B_hotness70}"
  ;;
80table_55B_hotness20_only_sparse)
  test_command+="${dataset_80table_55B_hotness20_only_sparse}"
  ;;
large)
  test_command+="$large_dataset"
  ;;
130table_110B_hotness20)
  test_command+="${dataset_130table_110B_hotness20}"
  ;;
180table_130B_hotness80)
  test_command+="${dataset_180table_130B_hotness80}"
  ;;
7table_470B_hotness20)
  test_command+="${dataset_7table_470B_hotness20}"
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
