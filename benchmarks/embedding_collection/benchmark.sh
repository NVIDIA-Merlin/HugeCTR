#!/usr/bin/env bash

export DENSE_UNIQUE_RATIO=0.3
export WGRAD_UNIQUE_RATIO=0.3

framework=$1
test_config=$2
batchsize=$3
nsys_result=$4

test_command=""

hugectr_framework="
python3 /workdir/benchmarks/embedding_collection/hugectr/train.py \
--display_interval 10 \
--sharding_plan auto \
--optimizer sgd \
--use_mixed_precision \
--mem_comm_bw_ratio 60 \
--dp_threshold 0 \
--dense_threshold 2 \
--num_gpus_per_node 2 \
"
# --disable_train_intra_iteration_overlap \
# --disable_train_inter_iteration_overlap \
#  --disable_fuse_sparse_embedding
# --mem_comm_bw_ratio 3000/50 eos
# --mem_comm_bw_ratio 2000/25 selene
# --num_gpus_per_node 2 \
# --perf_logging
# --use_column_wise_shard \
# --gen_loss_summary \

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

mlp_config="
--bmlp_layer_sizes 512,256,128 \
--tmlp_layer_sizes 1024,1024,512,256,1 \
"
test_command+="${mlp_config}"

utest_dataset="
--dataset_path /workdir/benchmarks/embedding_collection/dcnv2_synthetic_alpha1.01.bin \
--train_num_samples 6553600 \
--eval_num_samples 6553600 \
--num_table 2,2,2 \
--vocabulary_size_per_table 10000,1000,1000 \
--nnz_per_table 1,10,2 \
--ev_size_per_table 4,8,8 \
--dense_dim 13 \
--i64_input_key \
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
#dataset_80table_55B_hotness10="
#--dataset_path /workdir/dataset/middle_dcnv2_synthetic_alpha1.1.bin \
#--train_num_samples 6553600 \
#--eval_num_samples 6553600 \
#--num_table 5,3,5,1,1,10,10,10,5,30 \
#--vocabulary_size_per_table 10000,4000000,50000000,50000000,50000000,10,1000,10000,100000,4000000 \
#--nnz_per_table 100,30,30,1,1,1,1,1,1,1 \
#--ev_size_per_table 8,128,128,8,128,128,128,8,8,128 \
#--dense_dim 13 \
#"
dataset_80table_55B_hotness10="
--dataset_path /workdir/dataset/80table_55B_hotness10_1.1.bin \
--train_num_samples 6553600 \
--eval_num_samples 6553600 \
--num_table 10,10,10,5,30,1,1,3,5,5 \
--vocabulary_size_per_table 10,1000,10000,100000,4000000,50000000,50000000,4000000,50000000,10000 \
--nnz_per_table 1,1,1,1,1,1,1,30,30,100 \
--ev_size_per_table 256,256,256,256,256,256,256,256,256,256 \
--dense_dim 13 \
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
"
dataset_180table_130B_hotness80="
--dataset_path /workdir/dataset/large_hotness_dcnv2_synthetic_alpha1.1.bin \
--train_num_samples 6553600 \
--eval_num_samples 6553600 \
--num_table 5,5,5,5,20,30,10,20,10,10,10,5,40,1,1 \
--vocabulary_size_per_table 10000,4000000,4000000,50000000,1000,10000,5000000,4000000,10,1000,10000,100000,4000000,50000000,500000000 \
--nnz_per_table 100,50,30,50,50,30,20,20,100,10,100,100,200,100,100 \
--ev_size_per_table 128,128,128,128,128,128,128,128,128,128,128,128,128,128,128 \
--dense_dim 13 \
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
--i64_input_key \
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
"
dataset_180table_110B_hotness25="
--dataset_path /workdir/dataset/180table_110B_hotness25_1.1.bin \
--train_num_samples 6553600 \
--eval_num_samples 6553600 \
--num_table 10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10 \
--vocabulary_size_per_table 10,1000,1000,10000,10000,10000,100000,1000000,2000000,4000000,4000000,4000000,4000000,4000000,4000000,4000000,4000000,50000000 \
--nnz_per_table 100,50,30,20,10,1,1,1,1,1,1,1,10,10,20,30,50,100 \
--ev_size_per_table 128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128 \
--dense_dim 13 \
"
dataset_180table_110B_hotness13="
--dataset_path /workdir/dataset/180table_110B_hotness13_1.1.bin \
--train_num_samples 6553600 \
--eval_num_samples 6553600 \
--num_table 10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10 \
--vocabulary_size_per_table 10,1000,1000,10000,10000,10000,100000,1000000,2000000,4000000,4000000,4000000,4000000,4000000,4000000,4000000,4000000,50000000 \
--nnz_per_table 1,1,1,1,1,1,1,1,1,1,1,1,10,10,20,30,50,100 \
--ev_size_per_table 128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128 \
--dense_dim 13 \
"
dataset_180table_75B_hotness10="
--dataset_path /workdir/dataset/180table_75B_hotness10_1.1.bin \
--train_num_samples 6553600 \
--eval_num_samples 6553600 \
--num_table 10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10 \
--vocabulary_size_per_table 1000,1000,1000,1000,1000,10000,10000,10000,10000,10000,10000,10000,100000,100000,100000,4000000,4000000,50000000 \
--nnz_per_table 1,1,1,1,1,2,2,2,2,2,2,2,10,10,10,20,20,100 \
--ev_size_per_table 128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128 \
--dense_dim 13 \
"
dataset_180table_75B_hotness7="
--dataset_path /workdir/dataset/180table_75B_hotness7_1.1.bin \
--train_num_samples 6553600 \
--eval_num_samples 6553600 \
--num_table 10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10 \
--vocabulary_size_per_table 1000,1000,1000,1000,1000,10000,10000,10000,10000,10000,10000,10000,100000,100000,100000,4000000,4000000,50000000 \
--nnz_per_table 1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,20,100 \
--ev_size_per_table 128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128,128 \
--dense_dim 13 \
"
dataset_510table_110B_hotness5="
--dataset_path /workdir/dataset/510table_110B_hotness5_1.1.bin \
--train_num_samples 6553600 \
--eval_num_samples 6553600 \
--num_table 100,150,20,50,150,20,20 \
--vocabulary_size_per_table 1000,100000,1000000,2000000,4000000,4000000,4000000 \
--nnz_per_table 1,1,1,1,1,10,100 \
--ev_size_per_table 128,128,128,128,128,128,128 \
--dense_dim 13 \
"
dataset_200table_100B_hotness20="
--dataset_path /workdir/dataset/200table_100B_hotness20_1.1.bin \
--train_num_samples 6553600 \
--eval_num_samples 6553600 \
--num_table 10,10,10,10,20,10,10,10,10,10,10,20,20,10,10,10,10 \
--vocabulary_size_per_table 100,1000,1000,10000,10000,10000,100000,1000000,2000000,2000000,4000000,4000000,2000000,4000000,4000000,4000000,50000000 \
--nnz_per_table 1,1,5,20,100,1,1,1,1,1,1,1,10,20,30,50,100 \
--ev_size_per_table 128,128,128,128,128,128,128,128,128,128,128,128,64,128,128,128,128 \
--dense_dim 13 \
"
case $test_config in
utest)
  test_command+="$utest_dataset"
  ;;
80table_55B_hotness10)
  test_command+="${dataset_80table_55B_hotness10}"
  ;;
80table_55B_hotness10_small_vec)
  test_command+="${dataset_80table_55B_hotness10_small_vec}"
  ;;
80table_55B_hotness20)
  test_command+="${dataset_80table_55B_hotness20}"
  ;;
80table_55B_hotness70)
  test_command+="${dataset_80table_55B_hotness70}"
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
dcnv2)
  test_command+="${dataset_dcnv2}"
  ;;
180table_110B_hotness25)
  test_command+="${dataset_180table_110B_hotness25}"
  ;;
180table_110B_hotness13)
  test_command+="${dataset_180table_110B_hotness13}"
  ;;
180table_75B_hotness10)
  test_command+="${dataset_180table_75B_hotness10}"
  ;;
180table_75B_hotness7)
  test_command+="${dataset_180table_75B_hotness7}"
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
