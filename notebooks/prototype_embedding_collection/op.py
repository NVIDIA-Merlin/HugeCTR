from ast import Starred
from enum import unique
from statistics import mode
import typing
import numpy as np
from itertools import accumulate, chain, compress
import collections
import abc
from common import *
from utils import *
from threading import get_ident


def segmented_sort(keys_in, num_items, values_in, begin_offsets, end_offsets, num_segments):
    assert len(begin_offsets) == num_segments
    assert len(end_offsets) == num_segments
    assert num_items == end_offsets[-1]
    keys_out = []
    values_out = []
    for i in range(num_segments):
        start = begin_offsets[i]
        end = end_offsets[i]
        tmp = [(v, p) for v, p in zip(keys_in[start:end], values_in[start:end])]
        sorted_tmp = sorted(tmp, key=lambda x: x[0])
        keys_out.extend([v[0] for v in sorted_tmp])
        values_out.extend([v[1] for v in sorted_tmp])
    return keys_out, values_out


def segmented_reduce_sum(d_in, num_sgements, begin_offsets, end_offsets):
    output = [0 for _ in range(num_sgements)]
    for i in range(num_sgements):
        start = begin_offsets[i]
        end = end_offsets[i]
        output[i] = sum(d_in[start:end])
    return output


def clear_tensor(tensor):
    for e in tensor:
        e = 0.0


# class FlattenConcatembeddingOp:
#     def __init__(self, local_embedding_list) -> None:
#         self.local_embedding_list = local_embedding_list
#         pass

#     def compute(self, bucket_range, batch_size):
#         pass


# class LocalizedIndexCalculation:
#     def __init__(self, local_embedding_list) -> None:
#         self.flatten_concat_embedding_op = FlattenConcatembeddingOp(
#             local_embedding_list)
#         pass

#     def cal_model_idx(self, key, bucket_range, batch_size):
#         return [], []


class ModelIndexCalcualtion:
    def __init__(
        self,
        num_local_embedding,
        local_hotness_list,
        hotness_list,
        universal_batch_size,
        key_type,
        offset_type
    ) -> None:
        local_hotness_sum = sum(local_hotness_list)
        hotness_list_sum = sum(hotness_list)
        self.num_local_embedding = num_local_embedding
        self.model_key = [0 for _ in range(universal_batch_size * local_hotness_sum) if key_type == 'int32_t' or key_type == 'int64_t' ]
        self.model_idx_offsets = [0 for _ in range(universal_batch_size * num_local_embedding + 1) if offset_type == 'uint32_t' or offset_type == 'uint64_t' ]
        self.num_key_in_bucket_for_combiner = [0 for _ in range(universal_batch_size * num_local_embedding) if offset_type == 'uint32_t' or offset_type == 'uint64_t' ]
        self.num_model_key = 0
        self.flag = [int(0) for _ in range(universal_batch_size * hotness_list_sum)]

    def compute(self, key, bucket_range, local_embedding_list, sharding_id, num_sharding, batch_size):
        clear_tensor(self.flag)
        self.model_idx_offsets[0] = 0

        if self.num_local_embedding > 0:
            for idx in range(self.num_local_embedding):
                embedding_id = local_embedding_list[idx]
                for batch_id in range(batch_size):
                    bucket_start = bucket_range[batch_size * embedding_id + batch_id]
                    bucket_end = bucket_range[batch_size * embedding_id + batch_id + 1]
                    for r in range(bucket_start, bucket_end):
                        if key[r] % num_sharding == sharding_id:
                            self.flag[r] = 1

            for idx in range(self.num_local_embedding):
                embedding_id = local_embedding_list[idx]
                for batch_id in range(batch_size):
                    bucket_start = bucket_range[batch_size * embedding_id + batch_id]
                    bucket_end = bucket_range[batch_size * embedding_id + batch_id + 1]
                    self.model_idx_offsets[1 + idx * batch_size + batch_id] = sum(self.flag[bucket_start:bucket_end])
                    self.num_key_in_bucket_for_combiner[idx * batch_size + batch_id] = bucket_end - bucket_start
            # cub select
            self.num_model_key = sum(self.flag)
            self.model_key[: self.num_model_key] = list(compress(key, self.flag))
            # cub inclusive scan
            self.model_idx_offsets = list(accumulate(self.model_idx_offsets))
        return self.model_key, self.model_idx_offsets, self.num_model_key, self.num_key_in_bucket_for_combiner


class NetworkIndexCalculation:
    def compute(self, num_gpus, global_embedding_list, h_ev_size_offset):
        flatten_global_embedding_list = list(chain.from_iterable(global_embedding_list))
        dst_embedding_id_list = sorted(list(set(flatten_global_embedding_list)))
        num_dst_embedding_id = len(dst_embedding_id_list)
        # all array datatype is int
        network_ev_offsets = []
        for embedding_list in global_embedding_list:
            ev_list = [h_ev_size_offset[b + 1] - h_ev_size_offset[b] for b in embedding_list]
            ev_offset = [0] + list(accumulate(ev_list))
            network_ev_offsets.append(ev_offset)
        network_idx = [] 
        network_gpu_idx = [] 
        network_offset = [0 for _ in range(1 + num_dst_embedding_id)] 
        # this can be placed on cpu
        for local_embedding_id in range(len(dst_embedding_id_list)):
            dst_embedding_id = dst_embedding_id_list[local_embedding_id]
            for gpu_id in range(num_gpus):
                # because in python -1 is a valid index
                try:
                    idx = global_embedding_list[gpu_id].index(dst_embedding_id)
                except ValueError:
                    continue
                network_idx.append(network_ev_offsets[gpu_id][idx])
                network_gpu_idx.append(gpu_id)
                network_offset[1 + local_embedding_id] += 1
        network_offset = list(accumulate(network_offset))
        return (
            dst_embedding_id_list,
            num_dst_embedding_id,
            network_idx,
            network_gpu_idx,
            network_offset
        )


class CompressOffset:
    def __init__(self, stride) -> None:
        self.compressed_offset = [0 for _ in range(1 + stride)] # uint32_t
        self.stride = stride

    def compute(self, offset, batch_size):
        self.compressed_offset[0] = 0
        for s in range(self.stride):
            self.compressed_offset[1 + s] = offset[(s + 1) * batch_size] - offset[s * batch_size]
        return list(accumulate(self.compressed_offset))


class GradIndexCalculation:
    def __init__(self, num_local_embedding, h_local_hotness_list, universal_batch_size, key_type, offset_type) -> None:
        self.num_local_embedding = num_local_embedding
        local_hotness_sum = sum(h_local_hotness_list)
        self.unique_key_flag = [0 for _ in range(universal_batch_size * local_hotness_sum)] # int32_t
        self.unique_key = [0 for _ in range(universal_batch_size * local_hotness_sum) if key_type == 'int32_t' or key_type == 'int64_t'] 
        self.num_unique_key = 0
        self.unique_key_flag_scan = [0 for _ in range(universal_batch_size * local_hotness_sum)] # int32_t
        
        self.unique_key_bucket_idx = [0 for _ in range(universal_batch_size * local_hotness_sum) if offset_type == 'uint32_t' or offset_type == 'uint64_t']
        self.unique_key_bucket_idx_offset = [0 for _ in range(universal_batch_size * local_hotness_sum) if offset_type == 'uint32_t' or offset_type == 'uint64_t']
        self.unique_key_ev_size_offset = [0 for _ in range(1 + universal_batch_size * local_hotness_sum)] # int
        
        self.unique_key_id_space_offset = [0 for _ in range(1 + num_local_embedding)] # int
        self.unique_key_ev_id_space_offset = [0 for _ in range(1 + num_local_embedding)] # int

    def compute(self, model_key, bucket_idx, num_model_key, id_space_offset, d_local_ev_size_offset):
        if self.num_local_embedding > 0:
            # cub segmented radix sort pairs
            sorted_key, sorted_bucket_idx = segmented_sort(
                model_key,
                num_model_key,
                bucket_idx,
                id_space_offset[: self.num_local_embedding],
                id_space_offset[1:],
                self.num_local_embedding,
            )

            # cub select
            for idx in range(self.num_local_embedding):
                embedding_start = id_space_offset[idx]
                embedding_end = id_space_offset[idx + 1]
                for local_i in range(embedding_end - embedding_start):
                    i = local_i + embedding_start
                    if i == 0:
                        self.unique_key_flag[i] = 1
                    elif sorted_key[i] != sorted_key[i - 1]:
                        self.unique_key_flag[i] = 1
                    else:
                        self.unique_key_flag[i] = 0


            self.num_unique_key = sum(self.unique_key_flag[:num_model_key])
            self.unique_key[:self.num_unique_key] = list(compress(sorted_key, self.unique_key_flag))
            self.unique_key_id_space_offset[1:] = segmented_reduce_sum(
                self.unique_key_flag,
                self.num_local_embedding,
                id_space_offset[: self.num_local_embedding],
                id_space_offset[1:],
            )

            self.unique_key_bucket_idx[:num_model_key] = sorted_bucket_idx[:num_model_key]
            self.unique_key_flag_scan[:1 + num_model_key] = list(accumulate([0] + self.unique_key_flag[:num_model_key]))
            for i in range(1, num_model_key):
                if self.unique_key_flag_scan[i] != self.unique_key_flag_scan[i + 1]:
                    self.unique_key_bucket_idx_offset[self.unique_key_flag_scan[i]] = i
            self.unique_key_bucket_idx_offset[self.num_unique_key] = num_model_key
            self.unique_key_id_space_offset = list(accumulate(self.unique_key_id_space_offset))
            
            for embedding_idx in range(self.num_local_embedding):
                start = self.unique_key_id_space_offset[embedding_idx]
                end = self.unique_key_id_space_offset[embedding_idx + 1]
                for s in range(start, end):
                    self.unique_key_ev_size_offset[1 + s] = d_local_ev_size_offset[embedding_idx + 1] - d_local_ev_size_offset[embedding_idx]
            self.unique_key_ev_size_offset[:self.num_unique_key + 1] = list(accumulate(self.unique_key_ev_size_offset[:self.num_unique_key + 1]))
            
            for idx in range(self.num_local_embedding):
                self.unique_key_ev_id_space_offset[1 + idx] = (self.unique_key_id_space_offset[idx + 1] - self.unique_key_id_space_offset[idx]) * (d_local_ev_size_offset[idx + 1] - d_local_ev_size_offset[idx])
            self.unique_key_ev_id_space_offset = list(accumulate(self.unique_key_ev_id_space_offset))

        return (
            self.unique_key,
            self.num_unique_key,
            self.unique_key_id_space_offset,
            self.unique_key_bucket_idx,
            self.unique_key_bucket_idx_offset,
            self.unique_key_ev_size_offset,
            self.unique_key_ev_id_space_offset
        )


class ModelBackwardIndexCalculation:
    def __init__(
        self,
        num_gpus,
        num_local_embedding, 
        h_local_hotness_list, 
        universal_batch_size,
        key_type,
        offset_type
    ) -> None:
        self.num_gpus = num_gpus
        self.grad_index_calculation = GradIndexCalculation(
            num_local_embedding, h_local_hotness_list, universal_batch_size, key_type, offset_type
        )
        self.num_local_embedding = num_local_embedding
        local_hotness_sum = sum(h_local_hotness_list)
        self.bucket_idx = [0 for _ in range(universal_batch_size * local_hotness_sum)  if offset_type == 'uint32_t' or offset_type == 'uint64_t']
        

    def compute(self, model_key, num_model_key, model_offset, id_space_offset, d_local_ev_size_offset, batch_size):
        batch_size_per_gpu = batch_size // self.num_gpus
        if self.num_local_embedding > 0:
            # target idx: model comm buffer 
            for idx in range(batch_size * self.num_local_embedding):
                local_embedding_id = idx // batch_size
                batch_id = idx % batch_size
                gpu_id = batch_id // batch_size_per_gpu
                local_batch_id = batch_id % batch_size_per_gpu
                
                swizzle_idx = local_batch_id + local_embedding_id * batch_size_per_gpu + gpu_id * batch_size_per_gpu * self.num_local_embedding
                start = model_offset[idx]
                end = model_offset[idx + 1]
                for i in range(start, end):
                    self.bucket_idx[i] = swizzle_idx
        return self.grad_index_calculation.compute(
            model_key, 
            self.bucket_idx,
            num_model_key, 
            id_space_offset, 
            d_local_ev_size_offset
        )

class DPLocalReduceIndexCalculation:
    def __init__(
        self,
        num_local_embedding,
        num_embedding,
        h_local_hotness_list,
        h_hotness_list,
        universal_batch_size,
        key_type,
        offset_type
    ) -> None:
        self.grad_index_calculation = GradIndexCalculation(
            num_local_embedding, h_local_hotness_list, universal_batch_size, key_type, offset_type
        )
        self.num_local_embedding = num_local_embedding
        self.num_embedding = num_embedding
        local_hotness_sum = sum(h_local_hotness_list)
        hotness_sum = sum(h_hotness_list)
        self.all_dp_key = [0 for _ in range(universal_batch_size * local_hotness_sum) if key_type == 'int32_t' or key_type == 'int64_t']
        self.all_dp_bucket_idx = [0 for _ in range(universal_batch_size * local_hotness_sum) if offset_type == 'uint32_t' or offset_type == 'uint64_t']
        self.all_key_flag = [0 for _ in range(universal_batch_size * hotness_sum)] #int
        self.all_key_bucket_idx = [0 for _ in range(universal_batch_size * hotness_sum) if offset_type == 'uint32_t' or offset_type == 'uint64_t']
        self.all_dp_id_space_offset = [0 for _ in range(num_embedding + 1)] #int
        self.compress_id_space_offset = CompressOffset(self.num_embedding)

    def compute(self, key, bucket_range, d_local_embedding_list, d_local_ev_size_offset, batch_size):
        id_space_offset = self.compress_id_space_offset.compute(bucket_range, batch_size)
        for idx in range(self.num_local_embedding):
            embedding_id = d_local_embedding_list[idx]
            for batch_id in range(batch_size):
                start = bucket_range[embedding_id * batch_size + batch_id]
                end = bucket_range[embedding_id * batch_size + batch_id + 1]
                for i in range(start, end):
                    self.all_key_flag[i] = 1
                    self.all_key_bucket_idx[i] = embedding_id * batch_size + batch_id
        self.num_all_dp_key = sum(self.all_key_flag)
        self.all_dp_key[: self.num_all_dp_key] = list(compress(key, self.all_key_flag))
        self.all_dp_bucket_idx[: self.num_all_dp_key] = list(compress(self.all_key_bucket_idx, self.all_key_flag))

        self.all_dp_id_space_offset = segmented_reduce_sum(
            self.all_key_flag,
            self.num_embedding,
            id_space_offset[: self.num_embedding],
            id_space_offset[1:],
        )
        self.all_dp_id_space_offset = [0] + [
            self.all_dp_id_space_offset[d_local_embedding_list[idx]] for idx in range(self.num_local_embedding)
        ]
        self.all_dp_id_space_offset = list(accumulate(self.all_dp_id_space_offset))
        # print('all_dp_key', self.all_dp_key)
        # print('all_dp_bucket_idx', self.all_dp_bucket_idx)
        # print('id_space_offset', id_space_offset)
        # print('all_dp_id_space_offset', self.all_dp_id_space_offset)

        return self.grad_index_calculation.compute(
            self.all_dp_key,
            self.all_dp_bucket_idx,
            self.num_all_dp_key,
            self.all_dp_id_space_offset,
            d_local_ev_size_offset
        )


# class DPIndexCalculation:
#     def __init__(self, gpu_id, num_gpus, local_embedding_list) -> None:
#         self.gpu_id = gpu_id
#         self.num_gpus = num_gpus
#         self.local_embedding_id = local_embedding_list

#     def cal_dp_idx(self, key, bucket_range, batch_size):
#         return [], []


class DPIndexCalculation:
    def __init__(
        self,
        gpu_id,
        num_gpus,
        num_local_embedding,
        h_local_hotness_list,
        h_hotness_list,
        universal_batch_size,
        key_type,
        offset_type
    ) -> None:
        self.gpu_id = gpu_id
        self.num_gpus = num_gpus
        self.num_local_embedding = num_local_embedding
        universal_batch_size_per_gpu = universal_batch_size // num_gpus
        local_hotness_sum = sum(h_local_hotness_list)
        hotness_sum = sum(h_hotness_list)

        self.dp_key = [0 for _ in range(universal_batch_size_per_gpu * local_hotness_sum) if key_type == 'int32_t' or key_type == 'int64_t']
        self.dp_offset = [0 for _ in range(universal_batch_size_per_gpu * self.num_local_embedding + 1) if offset_type == 'uint32_t' or offset_type == 'uint64_t']
        self.flag = [0 for _ in range(universal_batch_size * hotness_sum)]
        self.num_dp_key = 0

    def compute(self, key, bucket_range, d_local_embedding_list, batch_size):
        batch_size_per_gpu = batch_size // self.num_gpus
        # mask flag
        for idx in range(self.num_local_embedding):
            embedding_id = d_local_embedding_list[idx]
            local_batch_start = batch_size * embedding_id + batch_size_per_gpu * self.gpu_id
            batch_bucket_start = bucket_range[local_batch_start]
            batch_bucket_end = bucket_range[local_batch_start + batch_size_per_gpu]
            self.flag[batch_bucket_start:batch_bucket_end] = [1 for _ in range(batch_bucket_end - batch_bucket_start)]
            for batch_id in range(batch_size_per_gpu):
                start = bucket_range[local_batch_start + batch_id]
                end = bucket_range[local_batch_start + batch_id + 1]
                self.dp_offset[1 + idx * batch_size_per_gpu + batch_id] = end - start
        # select
        self.num_dp_key = sum(self.flag)
        self.dp_key[: self.num_dp_key] = list(compress(key, self.flag))
        # cub inclusive sum
        self.dp_offset = list(accumulate(self.dp_offset))
        return self.dp_key, self.dp_offset, self.num_dp_key


class DPModelForward:
    def __init__(self, num_gpus, local_embedding_list) -> None:
        self.num_gpus = num_gpus
        self.num_local_embedding = len(local_embedding_list)

    def compute(
        self,
        dp_ev,
        dp_offset,
        output_buffer,
        d_local_embedding_list,
        d_local_combiner_list,
        d_ev_size_offset,
        batch_size,
    ):
        batch_size_per_gpu = batch_size // self.num_gpus
        # print('DPModelForward', dp_ev, dp_offset, d_local_embedding_list, batch_size_per_gpu)
        for idx in range(self.num_local_embedding):
            embedding_id = d_local_embedding_list[idx]
            ev_size = d_ev_size_offset[embedding_id + 1] - d_ev_size_offset[embedding_id]
            combiner = d_local_combiner_list[idx]
            assert combiner != "concat"
            for batch_id in range(batch_size_per_gpu):
                start = dp_offset[idx * batch_size_per_gpu + batch_id]
                end = dp_offset[idx * batch_size_per_gpu + batch_id + 1]
                accumulate_ev = [0.0 for _ in range(ev_size)]
                for i in range(start, end):
                    for ev_id in range(ev_size):
                        accumulate_ev[ev_id] += dp_ev[i][ev_id]
                if combiner == "mean" and end > start:
                    accumulate_ev = [v / (end - start) for v in accumulate_ev]
                output_buffer[
                    d_ev_size_offset[embedding_id] * batch_size_per_gpu
                    + batch_id * ev_size : d_ev_size_offset[embedding_id] * batch_size_per_gpu
                    + (batch_id + 1) * ev_size
                ] = accumulate_ev


class ModelForward:
    def __init__(self, num_gpus, local_embedding_list) -> None:
        self.num_gpus = num_gpus
        self.num_local_embedding = len(local_embedding_list)

    def compute(
        self,
        mp_ev,
        model_offset,
        num_key_in_bucket_for_combiner,
        model_comm_buffer,
        d_local_combiner_list,
        d_local_ev_size_offset,
        batch_size,
    ):
        batch_size_per_gpu = batch_size // self.num_gpus
        if self.num_local_embedding > 0:
            for idx in range(self.num_local_embedding):
                combiner = d_local_combiner_list[idx]
                ev_size = d_local_ev_size_offset[idx + 1] - d_local_ev_size_offset[idx]
                for batch_id in range(batch_size):
                    bucket_start = model_offset[idx * batch_size + batch_id]
                    bucket_end = model_offset[idx * batch_size + batch_id + 1]
                    tmp_ev = [0.0 for _ in range(ev_size)]
                    for bucket_id in range(bucket_start, bucket_end):
                        for ev_id in range(ev_size):
                            tmp_ev[ev_id] += mp_ev[bucket_id][ev_id]
                    if combiner == "mean" and bucket_end > bucket_start:
                        for ev_id in range(ev_size):
                            tmp_ev[ev_id] /= num_key_in_bucket_for_combiner[idx * batch_size + batch_id]
                    gpu_id = batch_id // batch_size_per_gpu
                    local_batch_id = batch_id % batch_size_per_gpu
                    local_comm_buffer_offset = (
                        d_local_ev_size_offset[idx] * batch_size_per_gpu + local_batch_id * ev_size
                    )
                    for ev_id in range(ev_size):
                        model_comm_buffer[gpu_id][local_comm_buffer_offset + ev_id] = tmp_ev[ev_id]


class NetworkForward:
    def __init__(self, num_gpus) -> None:
        self.num_gpus = num_gpus

    def compute(
        self,
        network_comm_buffer,
        dst_embedding_id_list,
        num_dst_embedding_id,
        network_idx,
        network_gpu_idx,
        network_offset,
        output_buffer,
        d_combiner_list,
        d_ev_size_offset,
        batch_size,
    ):
        batch_size_per_gpu = batch_size // self.num_gpus
        for idx in range(num_dst_embedding_id):
            dst_embedding_id = dst_embedding_id_list[idx]
            start = network_offset[idx]
            end = network_offset[idx + 1]
            ev_size = d_ev_size_offset[dst_embedding_id + 1] - d_ev_size_offset[dst_embedding_id]
            combiner =  d_combiner_list[dst_embedding_id]
            batch_dst_ev_offset = d_ev_size_offset[dst_embedding_id] * batch_size_per_gpu
            # handle missing key
            if end == start:
                output_buffer[batch_dst_ev_offset: batch_dst_ev_offset + ev_size * batch_size_per_gpu] = 0.
            
            for i in range(start, end):
                n_i = network_idx[i]
                ng_i = network_gpu_idx[i]
                batch_ev = network_comm_buffer[ng_i][n_i* batch_size_per_gpu: (n_i + ev_size) * batch_size_per_gpu]
                for e in range(ev_size * batch_size_per_gpu):
                    output_buffer[batch_dst_ev_offset + e] += batch_ev[e]

class NetworkBackward:
    def __init__(self, num_gpus) -> None:
        self.num_gpus = num_gpus

    def compute(
        self,
        top_grad,
        network_comm_buffer,
        dst_embedding_id_list,
        num_dst_embedding_id,
        network_idx,
        network_gpu_idx,
        network_offset,
        d_ev_size_offset,
        batch_size,
    ):
        # print('network_backward:')
        # print('top_grad={}'.format(top_grad))
        # print('network_comm_buffer={}'.format(network_comm_buffer))
        # print('dst_embedding_id_list={}'.format(dst_embedding_id_list))
        # print('num_dst_embedding_id={}'.format(num_dst_embedding_id))
        # print('network_idx={}'.format(network_idx))
        # print('network_gpu_idx={}'.format(network_gpu_idx))
        # print('network_offset={}'.format(network_offset))
        # print('d_ev_size_offset={}'.format(d_ev_size_offset))
        batch_size_per_gpu = batch_size // self.num_gpus
        for idx in range(num_dst_embedding_id):
            dst_embedding_id = dst_embedding_id_list[idx]
            start = network_offset[idx]
            end = network_offset[idx + 1]
            ev_size = d_ev_size_offset[dst_embedding_id + 1] - d_ev_size_offset[dst_embedding_id]
            batch_dst_ev_offset = d_ev_size_offset[dst_embedding_id] * batch_size_per_gpu
            
            for i in range(start, end):
                n_i = network_idx[i]
                ng_i = network_gpu_idx[i]
                batch_ev = top_grad[batch_dst_ev_offset: batch_dst_ev_offset + ev_size * batch_size_per_gpu]
                for e in range(ev_size * batch_size_per_gpu):
                    network_comm_buffer[ng_i][n_i* batch_size_per_gpu + e] = batch_ev[e]


class ModelBackward:
    def __init__(
        self,
        num_gpus,
        num_local_embedding,
        h_local_hotness_list,
        h_local_ev_size_list,
        universal_batch_size,
        emb_type
    ) -> None:
        self.num_local_embedding = num_local_embedding
        self.num_gpus = num_gpus

        max_unique_key_ev_buffer_size = sum(
            [hotness * ev_size for hotness, ev_size in zip(h_local_hotness_list, h_local_ev_size_list)]
        )
        self.grad_ev = [0.0 for _ in range(universal_batch_size * max_unique_key_ev_buffer_size) if emb_type == 'float' or emb_type == 'half']

    def compute(
        self,
        model_comm_buffer,
        unique_key_ev_size_offset,
        unique_key_bucket_idx,
        unique_key_bucket_idx_offset,
        num_unique_key,
        d_local_ev_size_offset,
        batch_size,
    ):
        batch_size_per_gpu = batch_size // self.num_gpus
        num_bucket_per_gpu = self.num_local_embedding * batch_size_per_gpu

        for unique_key_idx in range(num_unique_key):
            start = unique_key_bucket_idx_offset[unique_key_idx]
            end = unique_key_bucket_idx_offset[unique_key_idx + 1]
            ev_size = unique_key_ev_size_offset[unique_key_idx + 1] - unique_key_ev_size_offset[unique_key_idx]
            
            accumulate_ev = [0.0 for _ in range(ev_size)]
            for i in range(start, end):
                r = unique_key_bucket_idx[i]
                gpu_id = r // num_bucket_per_gpu
                local_r = r - gpu_id * num_bucket_per_gpu
                local_embedding_id = local_r // batch_size_per_gpu
                batch_id = local_r % batch_size_per_gpu
                ev = model_comm_buffer[gpu_id][batch_size_per_gpu * d_local_ev_size_offset[local_embedding_id] + batch_id * ev_size: batch_size_per_gpu * d_local_ev_size_offset[local_embedding_id] + (batch_id + 1) * ev_size]
                for ev_id in range(ev_size):
                    accumulate_ev[ev_id] += ev[ev_id]
            self.grad_ev[unique_key_ev_size_offset[unique_key_idx]: unique_key_ev_size_offset[unique_key_idx + 1]] = accumulate_ev
        
        return self.grad_ev


class DPLocalReduce:
    def __init__(
        self,
        gpu_id,
        num_gpus,
        num_local_embedding,
        h_local_hotness_list,
        h_local_ev_size_list,
        universal_batch_size,
        emb_type
    ) -> None:
        self.gpu_id = gpu_id
        self.num_gpus = num_gpus
        self.num_local_embedding = num_local_embedding

        max_unique_key_ev_buffer_size = sum(
            [hotness * ev_size for hotness, ev_size in zip(h_local_hotness_list, h_local_ev_size_list)]
        )
        self.grad_ev = [0.0 for _ in range(universal_batch_size * max_unique_key_ev_buffer_size)  if emb_type == 'float' or emb_type == 'half']

    def compute(
        self,
        top_grad,
        unique_key_ev_size_offset,
        unique_key_bucket_idx,
        unique_key_bucket_idx_offset,
        num_unique_key,
        d_ev_size_offset,
        batch_size,
    ):
        batch_size_per_gpu = batch_size // self.num_gpus
        local_batch_start = self.gpu_id * batch_size_per_gpu
        local_batch_end = (self.gpu_id + 1) * batch_size_per_gpu
        # print('top_grad={}'.format(top_grad))
        # print('unique_key_ev_size_offset={}'.format(unique_key_ev_size_offset))
        # print('unique_key_bucket_idx={}'.format(unique_key_bucket_idx))
        # print('unique_key_bucket_idx_offset={}'.format(unique_key_bucket_idx_offset))
        # print('num_unique_key={}'.format(num_unique_key))
        # print('d_ev_size_offset={}'.format(d_ev_size_offset))
        for unique_key_idx in range(num_unique_key):
            start = unique_key_bucket_idx_offset[unique_key_idx]
            end = unique_key_bucket_idx_offset[unique_key_idx + 1]
            ev_size = unique_key_ev_size_offset[unique_key_idx + 1] - unique_key_ev_size_offset[unique_key_idx]
            accumulate_ev = [0.0 for _ in range(ev_size)]
            for i in range(start, end):
                r = unique_key_bucket_idx[i]
                embedding_id = r // batch_size
                batch_id = r % batch_size
                # maybe we should filter in dp local reduce index calculation
                if batch_id >= local_batch_start and batch_id < local_batch_end:
                    local_batch_id = batch_id - local_batch_start
                    ev = top_grad[batch_size_per_gpu * d_ev_size_offset[embedding_id] + local_batch_id * ev_size: batch_size_per_gpu * d_ev_size_offset[embedding_id] + (local_batch_id + 1) * ev_size]
                    for ev_id in range(ev_size):
                        accumulate_ev[ev_id] += ev[ev_id]
            self.grad_ev[unique_key_ev_size_offset[unique_key_idx]: unique_key_ev_size_offset[unique_key_idx + 1]] = accumulate_ev
        return self.grad_ev


class All2All:
    def __init__(self, gpu_id, num_gpus) -> None:
        self.gpu_id = gpu_id
        self.num_gpus = num_gpus

    def communication(self, send_tensors, send_counts, recv_tensors, recv_counts):
        with nccl_communication("all2all", self.gpu_id, self.num_gpus) as comm:
            for i in range(self.num_gpus):
                nccl_send(send_tensors[i], send_counts[i], i, comm)
                nccl_recv(recv_tensors[i], recv_counts[i], i, comm)


class AllreduceInplace:
    def __init__(self, gpu_id, num_gpus) -> None:
        self.gpu_id = gpu_id
        self.num_gpus = num_gpus

    def communication(self, send_recv_buffer, count):
        with nccl_communication("allreduce", self.gpu_id, self.num_gpus) as comm:
            nccl_allreduce(send_recv_buffer, count, comm)
