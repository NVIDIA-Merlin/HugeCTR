#!/usr/bin/env bash
#
# Copyright (c) 2023, NVIDIA CORPORATION.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

set -e
export PS4='\n\033[0;33m+[${BASH_SOURCE}:${LINENO}]: \033[0m'
set -x

embedding_vec_size=4
global_batch_size=$1

nsys start -c nvtx -o tf_1gpu_bs$global_batch_size -f true 
nsys launch --sample=none -w true --backtrace=none --cudabacktrace=none --cpuctxsw=none --trace-fork-before-exec=true \
	-p Capture@Capture -e NSYS_NVTX_PROFILER_REGISTER_ONLY=0 \
	python3 main.py \
	--global_batch_size=$global_batch_size \
	--train_file_pattern="/mnt/portion/train/*.csv" \
	--test_file_pattern="/mnt/portion/test/*.csv" \
	--embedding_layer="TF" \
	--embedding_vec_size=$embedding_vec_size \
	--bottom_stack 512 256 $embedding_vec_size \
	--top_stack 1024 1024 512 256 1 \
	--distribute_strategy="mirrored" \
	--gpu_num=1 \
	--TF_MP=1 \
	--train_steps=500 \
	--nvtx_begin_step=450 \
	--nvtx_end_step=500

nsys start -c nvtx -o tf_2gpu_bs$global_batch_size -f true
nsys launch --sample=none -w true --backtrace=none --cudabacktrace=none --cpuctxsw=none --trace-fork-before-exec=true \
	        -p Capture@Capture -e NSYS_NVTX_PROFILER_REGISTER_ONLY=0 \
	mpiexec --allow-run-as-root -np 2 \
	python3 main.py \
	--global_batch_size=$global_batch_size \
	--train_file_pattern="/mnt/portion/train/*.csv" \
	--test_file_pattern="/mnt/portion/test/*.csv" \
	--embedding_layer="TF" \
	--embedding_vec_size=$embedding_vec_size \
	--bottom_stack 512 256 $embedding_vec_size \
	--top_stack 1024 1024 512 256 1 \
	--distribute_strategy="multiworker" \
	--TF_MP=1 \
	--train_steps=500 \
	--nvtx_begin_step=450 \
	--nvtx_end_step=500

nsys start -c nvtx -o tf_4gpu_bs$global_batch_size -f true
nsys launch --sample=none -w true --backtrace=none --cudabacktrace=none --cpuctxsw=none --trace-fork-before-exec=true \
	        -p Capture@Capture -e NSYS_NVTX_PROFILER_REGISTER_ONLY=0 \
	mpiexec --allow-run-as-root -np 4 \
	python3 main.py \
	--global_batch_size=$global_batch_size \
	--train_file_pattern="/mnt/portion/train/*.csv" \
	--test_file_pattern="/mnt/portion/test/*.csv" \
	--embedding_layer="TF" \
	--embedding_vec_size=$embedding_vec_size \
	--bottom_stack 512 256 $embedding_vec_size \
	--top_stack 1024 1024 512 256 1 \
	--distribute_strategy="multiworker" \
	--TF_MP=1 \
	--train_steps=500 \
	--nvtx_begin_step=450 \
	--nvtx_end_step=500

nsys start -c nvtx -o tf_8gpu_bs$global_batch_size -f true
nsys launch --sample=none -w true --backtrace=none --cudabacktrace=none --cpuctxsw=none --trace-fork-before-exec=true \
	        -p Capture@Capture -e NSYS_NVTX_PROFILER_REGISTER_ONLY=0 \
	mpiexec --allow-run-as-root -np 8 \
	python3 main.py \
	--global_batch_size=$global_batch_size \
	--train_file_pattern="/mnt/portion/train/*.csv" \
	--test_file_pattern="/mnt/portion/test/*.csv" \
	--embedding_layer="TF" \
	--embedding_vec_size=$embedding_vec_size \
	--bottom_stack 512 256 $embedding_vec_size \
	--top_stack 1024 1024 512 256 1 \
	--distribute_strategy="multiworker" \
	--TF_MP=1 \
	--train_steps=500 \
	--nvtx_begin_step=480 \
	--nvtx_end_step=500

## -------------------- SOK -------------------------------- ##
nsys start -c nvtx -o sok_1gpu_bs$global_batch_size -f true
nsys launch --sample=none -w true --backtrace=none --cudabacktrace=none --cpuctxsw=none --trace-fork-before-exec=true \
	        -p Capture@Capture -e NSYS_NVTX_PROFILER_REGISTER_ONLY=0 \
	python3 main.py \
	--global_batch_size=$global_batch_size \
	--train_file_pattern="/mnt/portion/train/*.csv" \
	--test_file_pattern="/mnt/portion/test/*.csv" \
	--embedding_layer="SOK" \
	--embedding_vec_size=$embedding_vec_size \
	--bottom_stack 512 256 $embedding_vec_size \
	--top_stack 1024 1024 512 256 1 \
	--distribute_strategy="mirrored" \
	--gpu_num=1 \
	--train_steps=500 \
	--nvtx_begin_step=450 \
	--nvtx_end_step=500

nsys start -c nvtx -o sok_2gpu_bs$global_batch_size -f true
nsys launch --sample=none -w true --backtrace=none --cudabacktrace=none --cpuctxsw=none --trace-fork-before-exec=true \
	                -p Capture@Capture -e NSYS_NVTX_PROFILER_REGISTER_ONLY=0 \
	mpiexec --allow-run-as-root -np 2 \
	python3 main.py \
	--global_batch_size=$global_batch_size \
	--train_file_pattern="/mnt/portion/train/*.csv" \
	--test_file_pattern="/mnt/portion/test/*.csv" \
	--embedding_layer="SOK" \
	--embedding_vec_size=$embedding_vec_size \
	--bottom_stack 512 256 $embedding_vec_size \
	--top_stack 1024 1024 512 256 1 \
	--distribute_strategy="multiworker" \
	--train_steps=500 \
	--nvtx_begin_step=450 \
	--nvtx_end_step=500

nsys start -c nvtx -o sok_4gpu_bs$global_batch_size -f true
nsys launch --sample=none -w true --backtrace=none --cudabacktrace=none --cpuctxsw=none --trace-fork-before-exec=true \
	                -p Capture@Capture -e NSYS_NVTX_PROFILER_REGISTER_ONLY=0 \
	mpiexec --allow-run-as-root -np 4 \
	python3 main.py \
	--global_batch_size=$global_batch_size \
	--train_file_pattern="/mnt/portion/train/*.csv" \
	--test_file_pattern="/mnt/portion/test/*.csv" \
	--embedding_layer="SOK" \
	--embedding_vec_size=$embedding_vec_size \
	--bottom_stack 512 256 $embedding_vec_size \
	--top_stack 1024 1024 512 256 1 \
	--distribute_strategy="multiworker" \
	--train_steps=500 \
	--nvtx_begin_step=450 \
	--nvtx_end_step=500


nsys start -c nvtx -o sok_8gpu_bs$global_batch_size -f true
nsys launch --sample=none -w true --backtrace=none --cudabacktrace=none --cpuctxsw=none --trace-fork-before-exec=true \
	                -p Capture@Capture -e NSYS_NVTX_PROFILER_REGISTER_ONLY=0 \
	mpiexec --allow-run-as-root -np 8 \
	python3 main.py \
	--global_batch_size=$global_batch_size \
	--train_file_pattern="/mnt/portion/train/*.csv" \
	--test_file_pattern="/mnt/portion/test/*.csv" \
	--embedding_layer="SOK" \
	--embedding_vec_size=$embedding_vec_size \
	--bottom_stack 512 256 $embedding_vec_size \
	--top_stack 1024 1024 512 256 1 \
	--distribute_strategy="multiworker" \
	--train_steps=500 \
	--nvtx_begin_step=450 \
	--nvtx_end_step=500
