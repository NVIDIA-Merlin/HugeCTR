/*
 * Copyright (c) 2023, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <embedding/common.hpp>
#include <embedding/operators/generic_lookup.cuh>
#include <embedding/operators/network_backward.hpp>
#include <embedding/operators/network_forward.hpp>
#include <utils.hpp>

namespace embedding {

using namespace core;
namespace {

void network_backward_from_feature_major_top_grad(const core23::Tensor& bucket_range,
                                                  const EmbeddingOutput& top_grad,
                                                  const NetworkIndices& network_indices,
                                                  NetworkBuffer& network_buffer, int batch_size,
                                                  int gpu_id, int num_gpus, cudaStream_t stream) {
  auto& top_grad_attr = top_grad.attr;
  auto& network_attr = network_buffer.attr;
  int batch_size_per_gpu = batch_size / num_gpus;
  int max_ev_size = top_grad_attr.max_ev_size;

  DISPATCH_INTEGRAL_FUNCTION_CORE23(bucket_range.data_type().type(), offset_t, [&] {
    DISPATCH_FLOAT_AND_HALF_FUNCTION_CORE23(top_grad.data.data_type().type(), emb_t, [&] {
      DISPATCH_FLOAT_AND_HALF_FUNCTION_CORE23(network_attr.type.type(), dst_emb_t, [&] {
        const offset_t* bucket_range_ptr = bucket_range.data<offset_t>();
        const int* network_ids_ptr = network_indices.network_ids.data<int>();
        const int* network_gpu_ids_ptr = network_indices.network_gpu_ids.data<int>();
        const int* network_offsets_ptr = network_indices.network_offsets.data<int>();
        const int* network_dst_lookup_ids_ptr = network_indices.network_dst_lookup_ids.data<int>();
        int** network_ev_sizes_ptr = (int**)network_attr.id_to_ev_size.data();
        int** network_ev_offsets_ptr = (int**)network_attr.id_to_ev_start_indices.data();
        const int* d_ev_size_offset_ptr = top_grad_attr.id_to_ev_start_indices.data<int>();
        const emb_t* top_grad_ptr = top_grad.data.data<emb_t>();
        dst_emb_t** network_comm_buffer_ptr = (dst_emb_t**)network_buffer.data.data();
        const char* combiner_ptr = top_grad_attr.id_to_combiner.data<char>();
        int num_network_dst_lookup_ids = network_indices.network_dst_lookup_ids.num_elements();

        auto one_to_multi_desc = make_MultiToOne<emb_t, dst_emb_t>(
            num_network_dst_lookup_ids * batch_size_per_gpu,
            [=] __device__(int i) {
              int bid = i / num_network_dst_lookup_ids;
              int lookup_id = i % num_network_dst_lookup_ids;
              return bid * network_offsets_ptr[num_network_dst_lookup_ids] +
                     network_offsets_ptr[lookup_id];
            },
            [=] __device__(int i) {
              int bid = i / num_network_dst_lookup_ids;
              int lookup_id = network_dst_lookup_ids_ptr[i % num_network_dst_lookup_ids];

              if (combiner_ptr[lookup_id] == static_cast<char>(Combiner::Average)) {
                int start = batch_size * lookup_id + gpu_id * batch_size_per_gpu + bid;
                return static_cast<int>(bucket_range_ptr[start + 1] - bucket_range_ptr[start]);
              } else {
                return 1;
              }
            },
            [=] __device__(int i) {
              int dst_lookup_id = network_dst_lookup_ids_ptr[i % num_network_dst_lookup_ids];
              return d_ev_size_offset_ptr[dst_lookup_id + 1] - d_ev_size_offset_ptr[dst_lookup_id];
            },
            [=] __device__(int i) {
              int bid = i / num_network_dst_lookup_ids;
              int lookup_id = network_dst_lookup_ids_ptr[i % num_network_dst_lookup_ids];

              int ev_offset = d_ev_size_offset_ptr[lookup_id] * batch_size_per_gpu;
              int ev_size = d_ev_size_offset_ptr[lookup_id + 1] - d_ev_size_offset_ptr[lookup_id];
              return top_grad_ptr + ev_offset + bid * ev_size;
            },
            [=] __device__(int i) {
              int bid = i / network_offsets_ptr[num_network_dst_lookup_ids];
              int id = i % network_offsets_ptr[num_network_dst_lookup_ids];

              int network_gpu_id = network_gpu_ids_ptr[id];
              int network_id = network_ids_ptr[id];
              int ev_offset =
                  network_ev_offsets_ptr[network_gpu_id][network_id] * batch_size_per_gpu;
              int ev_size = network_ev_sizes_ptr[network_gpu_id][network_id];

              return network_comm_buffer_ptr[network_gpu_id] + ev_offset + bid * ev_size;
            });
        copy_one_to_multi(one_to_multi_desc, max_ev_size, stream);
      });
    });
  });
}

void network_backward_from_batch_major_top_grad(const core23::Tensor& bucket_range,
                                                const EmbeddingOutput& top_grad,
                                                const NetworkIndices& network_indices,
                                                NetworkBuffer& network_buffer, int batch_size,
                                                int gpu_id, int num_gpus, cudaStream_t stream) {
  auto& top_grad_attr = top_grad.attr;
  auto& network_attr = network_buffer.attr;
  int batch_size_per_gpu = batch_size / num_gpus;
  int max_ev_size = top_grad_attr.max_ev_size;
  int num_lookup = top_grad_attr.id_to_ev_size.num_elements();

  DISPATCH_INTEGRAL_FUNCTION_CORE23(bucket_range.data_type().type(), offset_t, [&] {
    DISPATCH_FLOAT_AND_HALF_FUNCTION_CORE23(top_grad.data.data_type().type(), emb_t, [&] {
      DISPATCH_FLOAT_AND_HALF_FUNCTION_CORE23(network_attr.type.type(), dst_emb_t, [&] {
        const offset_t* bucket_range_ptr = bucket_range.data<offset_t>();
        const int* network_ids_ptr = network_indices.network_ids.data<int>();
        const int* network_gpu_ids_ptr = network_indices.network_gpu_ids.data<int>();
        const int* network_offsets_ptr = network_indices.network_offsets.data<int>();
        const int* network_dst_lookup_ids_ptr = network_indices.network_dst_lookup_ids.data<int>();
        int** network_ev_sizes_ptr = (int**)network_attr.id_to_ev_size.data();
        int** network_ev_offsets_ptr = (int**)network_attr.id_to_ev_start_indices.data();
        const int* d_ev_size_offset_ptr = top_grad_attr.id_to_ev_start_indices.data<int>();
        const emb_t* top_grad_ptr = top_grad.data.data<emb_t>();
        dst_emb_t** network_comm_buffer_ptr = (dst_emb_t**)network_buffer.data.data();
        const char* combiner_ptr = top_grad_attr.id_to_combiner.data<char>();
        int num_network_dst_lookup_ids = network_indices.network_dst_lookup_ids.num_elements();

        auto one_to_multi_desc = make_MultiToOne<emb_t, dst_emb_t>(
            num_network_dst_lookup_ids * batch_size_per_gpu,
            [=] __device__(int i) {
              int bid = i / num_network_dst_lookup_ids;
              int lookup_id = i % num_network_dst_lookup_ids;
              return bid * network_offsets_ptr[num_network_dst_lookup_ids] +
                     network_offsets_ptr[lookup_id];
            },
            [=] __device__(int i) {
              int bid = i / num_network_dst_lookup_ids;
              int lookup_id = network_dst_lookup_ids_ptr[i % num_network_dst_lookup_ids];

              if (combiner_ptr[lookup_id] == static_cast<char>(Combiner::Average)) {
                int start = batch_size * lookup_id + gpu_id * batch_size_per_gpu + bid;
                return static_cast<int>(bucket_range_ptr[start + 1] - bucket_range_ptr[start]);
              } else {
                return 1;
              }
            },
            [=] __device__(int i) {
              int dst_lookup_id = network_dst_lookup_ids_ptr[i % num_network_dst_lookup_ids];
              return d_ev_size_offset_ptr[dst_lookup_id + 1] - d_ev_size_offset_ptr[dst_lookup_id];
            },
            [=] __device__(int i) {
              int bid = i / num_network_dst_lookup_ids;
              int lookup_id = network_dst_lookup_ids_ptr[i % num_network_dst_lookup_ids];

              int ev_offset = d_ev_size_offset_ptr[num_lookup] * bid;
              int ev_size = d_ev_size_offset_ptr[lookup_id + 1] - d_ev_size_offset_ptr[lookup_id];
              return top_grad_ptr + ev_offset + d_ev_size_offset_ptr[lookup_id];
            },
            [=] __device__(int i) {
              int bid = i / network_offsets_ptr[num_network_dst_lookup_ids];
              int id = i % network_offsets_ptr[num_network_dst_lookup_ids];

              int network_gpu_id = network_gpu_ids_ptr[id];
              int network_id = network_ids_ptr[id];
              int ev_offset =
                  network_ev_offsets_ptr[network_gpu_id][network_id] * batch_size_per_gpu;
              int ev_size = network_ev_sizes_ptr[network_gpu_id][network_id];

              return network_comm_buffer_ptr[network_gpu_id] + ev_offset + bid * ev_size;
            });
        copy_one_to_multi(one_to_multi_desc, max_ev_size, stream);
      });
    });
  });
}

}  // namespace
void NetworkBackward::compute(const core23::Tensor& bucket_range, const EmbeddingOutput& top_grad,
                              const NetworkIndices& network_indices, NetworkBuffer& network_buffer,
                              int batch_size) {
  HugeCTR::CudaDeviceContext ctx(core_->get_device_id());
  auto stream = core_->get_local_gpu()->get_stream();
  int gpu_id = core_->get_global_gpu_id();
  int num_gpus = core_->get_global_gpu_count();

  if (top_grad.attr.layout == EmbeddingLayout::FeatureMajor) {
    network_backward_from_feature_major_top_grad(bucket_range, top_grad, network_indices,
                                                 network_buffer, batch_size, gpu_id, num_gpus,
                                                 stream);
  } else {
    HCTR_ASSERT(top_grad.attr.layout == EmbeddingLayout::BatchMajor);
    network_backward_from_batch_major_top_grad(bucket_range, top_grad, network_indices,
                                               network_buffer, batch_size, gpu_id, num_gpus,
                                               stream);
  }
}

void NetworkBackward::compute(
    const core23::Tensor& row_lengths, const core23::Tensor& d_combiner_list,
    const core23::Tensor& top_grad, const core23::Tensor& network_ids,
    const core23::Tensor& network_gpu_ids, const core23::Tensor& network_offsets,
    const core23::Tensor& network_dst_lookup_ids, const core23::Tensor& network_ev_sizes,
    const core23::Tensor& network_ev_offsets, core23::Tensor& network_comm_buffer,
    const core23::Tensor& d_ev_size_offset, int batch_size, int max_ev_size) {
  HugeCTR::CudaDeviceContext ctx(core_->get_device_id());
  int batch_size_per_gpu = batch_size / num_gpus_;
  auto stream = core_->get_local_gpu()->get_stream();

  DISPATCH_INTEGRAL_FUNCTION_CORE23(row_lengths.data_type().type(), offset_t, [&] {
    DISPATCH_FLOAT_AND_HALF_FUNCTION_CORE23(top_grad.data_type().type(), emb_t, [&] {
      DISPATCH_FLOAT_AND_HALF_FUNCTION_CORE23(
          network_comm_buffer.data_type().type(), dst_emb_t, [&] {
            const offset_t** row_lengths_ptr = (const offset_t**)row_lengths.data();
            const int* network_ids_ptr = network_ids.data<int>();
            const int* network_gpu_ids_ptr = network_gpu_ids.data<int>();
            const int* network_offsets_ptr = network_offsets.data<int>();
            const int* network_dst_lookup_ids_ptr = network_dst_lookup_ids.data<int>();
            const int** network_ev_sizes_ptr = (const int**)network_ev_sizes.data();
            const int** network_ev_offsets_ptr = (const int**)network_ev_offsets.data();
            const int* d_ev_size_offset_ptr = d_ev_size_offset.data<int>();
            const emb_t** top_grad_ptr = (const emb_t**)top_grad.data();
            dst_emb_t** network_comm_buffer_ptr = (dst_emb_t**)network_comm_buffer.data();
            const char* combiner_ptr = d_combiner_list.data<char>();
            int num_network_dst_lookup_ids = network_dst_lookup_ids.num_elements();
            int gpu_id = core_->get_global_gpu_id();

            auto one_to_multi_desc = make_MultiToOne<emb_t, dst_emb_t>(
                num_network_dst_lookup_ids * batch_size_per_gpu,
                [=] __device__(int i) {
                  int bid = i / num_network_dst_lookup_ids;
                  int lookup_id = i % num_network_dst_lookup_ids;
                  return bid * network_offsets_ptr[num_network_dst_lookup_ids] +
                         network_offsets_ptr[lookup_id];
                },
                [=] __device__(int i) {
                  int bid = i / num_network_dst_lookup_ids;
                  int lookup_id = network_dst_lookup_ids_ptr[i % num_network_dst_lookup_ids];

                  if (combiner_ptr[lookup_id] == static_cast<char>(Combiner::Average)) {
                    return static_cast<int>(row_lengths_ptr[lookup_id][bid]);
                  } else {
                    return 1;
                  }
                },
                [=] __device__(int i) {
                  int dst_lookup_id = network_dst_lookup_ids_ptr[i % num_network_dst_lookup_ids];
                  return d_ev_size_offset_ptr[dst_lookup_id + 1] -
                         d_ev_size_offset_ptr[dst_lookup_id];
                },
                [=] __device__(int i) {
                  int bid = i / num_network_dst_lookup_ids;
                  int lookup_id = network_dst_lookup_ids_ptr[i % num_network_dst_lookup_ids];

                  int ev_size =
                      d_ev_size_offset_ptr[lookup_id + 1] - d_ev_size_offset_ptr[lookup_id];
                  return top_grad_ptr[lookup_id] + bid * ev_size;
                },
                [=] __device__(int i) {
                  int bid = i / network_offsets_ptr[num_network_dst_lookup_ids];
                  int id = i % network_offsets_ptr[num_network_dst_lookup_ids];

                  int network_gpu_id = network_gpu_ids_ptr[id];
                  int network_id = network_ids_ptr[id];
                  int ev_offset =
                      network_ev_offsets_ptr[network_gpu_id][network_id] * batch_size_per_gpu;
                  int ev_size = network_ev_sizes_ptr[network_gpu_id][network_id];

                  return network_comm_buffer_ptr[network_gpu_id] + ev_offset + bid * ev_size;
                });
            copy_one_to_multi(one_to_multi_desc, max_ev_size, stream);
          });
    });
  });
}
}  // namespace embedding
