"""
 Copyright (c) 2023, NVIDIA CORPORATION.
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import math


def cal_workspace_size_per_gpu_from_vocabulary_size_per_gpu(
    vocabulary_size_per_gpu, emb_vec_size, num_gpus, optimizer, optimizer_update_type
):
    assert optimizer in ["adam", "adagrad", "momentumsgd", "nesterov", "sgd"]
    assert optimizer_update_type in ["local", "global", "lazy_global"]
    num_opt_state_copies = 0
    if optimizer == "adam":
        if optimizer_update_type == "lazy_global":
            num_opt_state_copies = 3
        else:
            num_opt_state_copies = 2
    elif optimizer == "adagrad" or optimizer == "momentumsgd" or optimizer == "nesterov":
        num_opt_state_copies = 1
    else:
        num_opt_state_copies = 0
    return math.ceil(
        (vocabulary_size_per_gpu * emb_vec_size * 4 * (1 + num_opt_state_copies)) / (1024 * 1024)
    )


def cal_vocabulary_size_per_gpu_for_distributed_slot(total_vocabulary_size, num_gpus):
    return math.ceil(total_vocabulary_size / num_gpus)


def cal_vocabulary_size_per_gpu_for_localized_slot(slot_size_array, num_gpus):
    vocal_size_per_gpu = [0 for _ in range(num_gpus)]
    for i in range(len(slot_size_array)):
        vocal_size_per_gpu[i % num_gpus] += slot_size_array[i]
    return math.ceil(max(vocal_size_per_gpu))


if __name__ == "__main__":
    vvgpu = [[0]]
    emb_vec_size = 16
    optimizer = "adam"
    optimizer_update_type = "global"
    slot_size_array = [
        39884,
        39043,
        17289,
        7420,
        20263,
        3,
        7120,
        1543,
        39884,
        39043,
        17289,
        7420,
        20263,
        3,
        7120,
        1543,
        63,
        63,
        39884,
        39043,
        17289,
        7420,
        20263,
        3,
        7120,
        1543,
    ]

    num_gpus = sum([len(local_gpu) for local_gpu in vvgpu])
    # DistributedSlotSparseEmbedding
    total_vocabulary_size = sum(slot_size_array)
    voc_size_per_gpu_distributed = cal_vocabulary_size_per_gpu_for_distributed_slot(
        total_vocabulary_size, num_gpus
    )
    workspace_size_per_gpu_distributed = cal_workspace_size_per_gpu_from_vocabulary_size_per_gpu(
        voc_size_per_gpu_distributed, emb_vec_size, num_gpus, optimizer, optimizer_update_type
    )
    # LocalizedSlotSparseEmbedding
    voc_size_per_gpu_localized = cal_vocabulary_size_per_gpu_for_localized_slot(
        slot_size_array, num_gpus
    )
    workspace_size_per_gpu_localized = cal_workspace_size_per_gpu_from_vocabulary_size_per_gpu(
        voc_size_per_gpu_localized, emb_vec_size, num_gpus, optimizer, optimizer_update_type
    )

    print(
        f"DistributedSlotSparseEmbedding, total_vocabulary_size: {total_vocabulary_size}, num_gpus: {num_gpus}, voc_size_per_gpu_distributed: {voc_size_per_gpu_distributed}, workspace_size_per_gpu: {workspace_size_per_gpu_distributed}MB"
    )
    print(
        f"LocalizedSlotSparseEmbedding, slot_size_array: {slot_size_array}, num_gpus: {num_gpus}, voc_size_per_gpu_distributed: {voc_size_per_gpu_localized}, workspace_size_per_gpu:{workspace_size_per_gpu_localized}MB"
    )
