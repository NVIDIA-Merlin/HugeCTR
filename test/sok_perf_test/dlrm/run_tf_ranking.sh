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

export EXPERIMENT_NAME=tf_ranking
export DATA_DIR="./"
export EMBEDDING_DIM=8
export PYTHONPATH=$PYTHONPATH:$(pwd)/models
# export TF_GPU_ALLOCATOR=cuda_malloc_async

python3 models/official/recommendation/ranking/train.py --mode=train_and_eval \
--model_dir=./${EXPERIMENT_NAME} --params_override="
runtime:
    distribution_strategy: 'mirrored'
    num_gpus: 4
task:
    use_synthetic_data: false
    train_data:
        input_path: '${DATA_DIR}/train/*'
        global_batch_size: 16384
    validation_data:
        input_path: '${DATA_DIR}/test/*'
        global_batch_size: 16384
    model:
        num_dense_features: 13
        bottom_mlp: [512,256,${EMBEDDING_DIM}]
        embedding_dim: ${EMBEDDING_DIM}
        top_mlp: [1024,1024,512,256,1]
        interaction: 'dot'
        vocab_sizes: [39884407, 39043, 17289, 7420, 20263, 
                    3, 7120, 1543, 63, 38532952, 2953546, 
                    403346, 10, 2208, 11938, 155, 4, 976, 
                    14, 39979772, 25641295, 39664985, 585935, 
                    12972, 108, 36]
trainer:
    use_orbit: true
    validation_interval: 85352
    checkpoint_interval: 85352
    validation_steps: 5440
    train_steps: 256054
    steps_per_loop: 1000
"

unset EXPERIMENT_NAME
unset DATA_DIR
unset EMBEDDING_DIM
# unset TF_GPU_ALLOCATOR