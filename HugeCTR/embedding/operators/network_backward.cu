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

void network_backward_from_feature_major_top_grad(const core23::Tensor& dp_num_keys_per_bucket,
                                                  const EmbeddingOutput& top_grad,
                                                  const NetworkIndices& network_indices,
                                                  const HugeCTR::core23::KernelParams kernel_params,
                                                  NetworkBuffer& network_buffer, int batch_size,
                                                  int gpu_id, int num_gpus, cudaStream_t stream) {
  auto& top_grad_attr = top_grad.attr;
  auto& network_attr = network_buffer.attr;
  int batch_size_per_gpu = batch_size / num_gpus;
  int max_ev_size = top_grad_attr.max_ev_size;

  DISPATCH_INTEGRAL_FUNCTION_CORE23(dp_num_keys_per_bucket.data_type().type(), offset_t, [&] {
    DISPATCH_FLOAT_AND_HALF_FUNCTION_CORE23(top_grad.data.data_type().type(), emb_t, [&] {
      DISPATCH_FLOAT_AND_HALF_FUNCTION_CORE23(network_attr.type.type(), dst_emb_t, [&] {
        const offset_t* dp_num_keys_per_bucket_ptr = dp_num_keys_per_bucket.data<offset_t>();
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
                int idx = batch_size_per_gpu * lookup_id + +bid;
                return static_cast<int>(dp_num_keys_per_bucket_ptr[idx]);
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
        copy_one_to_multi(one_to_multi_desc, kernel_params, max_ev_size, stream);
      });
    });
  });
}

void network_backward_from_batch_major_top_grad(const core23::Tensor& dp_num_keys_per_bucket,
                                                const EmbeddingOutput& top_grad,
                                                const NetworkIndices& network_indices,
                                                const HugeCTR::core23::KernelParams kernel_params,
                                                NetworkBuffer& network_buffer, int batch_size,
                                                int gpu_id, int num_gpus, cudaStream_t stream) {
  auto& top_grad_attr = top_grad.attr;
  auto& network_attr = network_buffer.attr;
  int batch_size_per_gpu = batch_size / num_gpus;
  int max_ev_size = top_grad_attr.max_ev_size;
  int num_lookup = top_grad_attr.id_to_ev_size.num_elements();

  DISPATCH_INTEGRAL_FUNCTION_CORE23(dp_num_keys_per_bucket.data_type().type(), offset_t, [&] {
    DISPATCH_FLOAT_AND_HALF_FUNCTION_CORE23(top_grad.data.data_type().type(), emb_t, [&] {
      DISPATCH_FLOAT_AND_HALF_FUNCTION_CORE23(network_attr.type.type(), dst_emb_t, [&] {
        const offset_t* dp_num_keys_per_bucket_ptr = dp_num_keys_per_bucket.data<offset_t>();
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
                int idx = batch_size_per_gpu * lookup_id + bid;
                return static_cast<int>(dp_num_keys_per_bucket_ptr[idx]);
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
        copy_one_to_multi(one_to_multi_desc, kernel_params, max_ev_size, stream);
      });
    });
  });
}

size_t calc_num_valid_network_tensor(const EmbeddingInput& embedding_input) {
  auto& h_recv_k_per_gpu =
      embedding_input.dense_compression_input.model_parallel_compression_input.h_recv_k_per_gpu;

  size_t num_network_tensor = 0;
  DISPATCH_INTEGRAL_FUNCTION_CORE23(h_recv_k_per_gpu.data_type().type(), offset_t, [&] {
    for (int64_t i = 0; i < h_recv_k_per_gpu.num_elements(); ++i) {
      num_network_tensor += static_cast<size_t>(h_recv_k_per_gpu.data<offset_t>()[i]);
    }
  });
  return num_network_tensor;
}

template <typename src_emb_t, typename dst_emb_t, typename offset_t>
struct DenseNetworkBackwardBatchMajorOneToOneAtomicDesc {
  using SrcT = src_emb_t;
  using DstT = dst_emb_t;

  HOST_DEVICE_INLINE size_t num_vec() { return num_vec_; }

  HOST_DEVICE_INLINE int get_vec_length(int i) { return ev_size; }
  // we need a transform to src id use num_model_revers_idx
  HOST_DEVICE_INLINE const SrcT* get_src_ptr(int i) {
    int hotness_id = bucket_id_ptr[i] / batch_size_per_gpu;
    int64_t lookup_id = bs_upper_bound_sub_one(hotness_range, range_num, hotness_id);
    offset_t bucket_id = bucket_id_ptr[i];
    hotness_id = hotness_id - hotness_range[lookup_id];
    int bid = bucket_id % batch_size_per_gpu;
    return src_ptr + bid * global_ev_offset + ev_start_indices[lookup_id] + hotness_id * ev_size;
  }
  HOST_DEVICE_INLINE DstT* get_dst_ptr(int i) { return dst_ptr + reverse_id_ptr[i] * ev_size; }

  size_t num_vec_;
  int ev_size;
  int batch_size_per_gpu;
  int range_num;
  int global_ev_offset;

  const int* hotness_range;
  const int* ev_start_indices;

  const offset_t* __restrict__ reverse_id_ptr;
  const offset_t* __restrict__ bucket_id_ptr;
  const src_emb_t* __restrict__ src_ptr;
  dst_emb_t* __restrict__ dst_ptr;
};

template <typename src_emb_t, typename dst_emb_t, typename offset_t>
struct DenseNetworkBackwardFeatureMajorOneToOneAtomicDesc {
  using SrcT = src_emb_t;
  using DstT = dst_emb_t;

  HOST_DEVICE_INLINE size_t num_vec() { return num_vec_; }

  HOST_DEVICE_INLINE int get_vec_length(int i) { return ev_size; }
  // we need a transform to src id use num_model_revers_idx
  HOST_DEVICE_INLINE const SrcT* get_src_ptr(int i) {
    int hotness_id = bucket_id_ptr[i] / batch_size_per_gpu;
    int64_t lookup_id = bs_upper_bound_sub_one(hotness_range, range_num, hotness_id);
    offset_t bucket_id = bucket_id_ptr[i];
    hotness_id = hotness_id - hotness_range[lookup_id];
    int bid = bucket_id % batch_size_per_gpu;
    return src_ptr + batch_size_per_gpu * ev_start_indices[lookup_id] +
           bid * hotness_list[lookup_id] * ev_size + hotness_id * ev_size;
  }
  HOST_DEVICE_INLINE DstT* get_dst_ptr(int i) { return dst_ptr + reverse_id_ptr[i] * ev_size; }

  size_t num_vec_;
  int ev_size;
  int batch_size_per_gpu;

  int range_num;
  const int* hotness_range;
  const int* ev_start_indices;
  const int* hotness_list;

  const offset_t* __restrict__ reverse_id_ptr;
  const offset_t* __restrict__ bucket_id_ptr;
  const src_emb_t* __restrict__ src_ptr;
  dst_emb_t* __restrict__ dst_ptr;
};

void dense_network_backward_from_feature_major_top_grad(
    const EmbeddingInput& embedding_input, const EmbeddingOutput& top_grad,
    const DenseNetworkIndices& network_indices, const HugeCTR::core23::KernelParams kernel_params,
    DenseNetworkBuffer& network_buffer, int batch_size, int gpu_id, int num_gpus,
    cudaStream_t stream) {
  int batch_size_per_gpu = batch_size / num_gpus;

  int ev_size = network_buffer.attr.ev_size;
  auto& reverse_idx =
      embedding_input.dense_compression_input.model_parallel_compression_input.network_reverse_idx;
  auto& bucket_ids = embedding_input.dense_compression_input.model_parallel_compression_input
                         .network_dst_bucket_ids;
  auto& num_network_reverse_idx = embedding_input.dense_compression_input
                                      .model_parallel_compression_input.num_network_reverse_idx;

  DISPATCH_INTEGRAL_FUNCTION_CORE23(reverse_idx.data_type().type(), offset_t, [&] {
    DISPATCH_FLOAT_AND_HALF_FUNCTION_CORE23(top_grad.data.data_type().type(), src_emb_t, [&] {
      DISPATCH_FLOAT_AND_HALF_FUNCTION_CORE23(
          network_buffer.data.data_type().type(), dst_emb_t, [&] {
            const src_emb_t* top_grad_ptr = top_grad.data.data<src_emb_t>();
            dst_emb_t* network_comm_buffer_ptr = network_buffer.data.data<dst_emb_t>();
            offset_t* reverse_idx_ptr = reverse_idx.data<offset_t>();
            offset_t* bucket_ids_ptr = bucket_ids.data<offset_t>();

            int range_num = network_indices.local_lookup_num + 1;

            auto hotness_range_ptr = network_indices.d_local_hotness_range.data<int>();
            auto ev_start_indices_ptr = network_indices.d_ev_start_indices.data<int>();
            auto hotness_list = network_indices.d_local_hotness.data<int>();

            using CopyDesc =
                DenseNetworkBackwardFeatureMajorOneToOneAtomicDesc<src_emb_t, dst_emb_t, offset_t>;

            CopyDesc one_to_one_atomic_desc = {num_network_reverse_idx,
                                               ev_size,
                                               batch_size_per_gpu,
                                               range_num,
                                               hotness_range_ptr,
                                               ev_start_indices_ptr,
                                               hotness_list,
                                               reverse_idx_ptr,
                                               bucket_ids_ptr,
                                               top_grad_ptr,
                                               network_comm_buffer_ptr};

            size_t num_valid_network_tensor = calc_num_valid_network_tensor(embedding_input);
            HCTR_LIB_THROW(cudaMemsetAsync(
                network_buffer.data.data(), 0,
                num_valid_network_tensor * ev_size * network_buffer.data.data_type().size(),
                stream));
            one_to_one_atomic(one_to_one_atomic_desc, kernel_params, ev_size,
                              num_network_reverse_idx, stream);
          });
    });
  });
}

void dense_network_backward_from_batch_major_top_grad(
    const EmbeddingInput& embedding_input, const EmbeddingOutput& top_grad,
    const DenseNetworkIndices& network_indices, const HugeCTR::core23::KernelParams kernel_params,
    DenseNetworkBuffer& network_buffer, int batch_size, int gpu_id, int num_gpus,
    cudaStream_t stream) {
  int batch_size_per_gpu = batch_size / num_gpus;

  int ev_size = network_buffer.attr.ev_size;
  size_t num_key = embedding_input.dense_compression_input.model_parallel_compression_input
                       .num_network_reverse_idx;
  auto& reverse_idx =
      embedding_input.dense_compression_input.model_parallel_compression_input.network_reverse_idx;
  auto& bucket_ids = embedding_input.dense_compression_input.model_parallel_compression_input
                         .network_dst_bucket_ids;
  auto& num_network_reverse_idx = embedding_input.dense_compression_input
                                      .model_parallel_compression_input.num_network_reverse_idx;

  DISPATCH_INTEGRAL_FUNCTION_CORE23(reverse_idx.data_type().type(), offset_t, [&] {
    DISPATCH_FLOAT_AND_HALF_FUNCTION_CORE23(top_grad.data.data_type().type(), src_emb_t, [&] {
      DISPATCH_FLOAT_AND_HALF_FUNCTION_CORE23(
          network_buffer.data.data_type().type(), dst_emb_t, [&] {
            const src_emb_t* top_grad_ptr = top_grad.data.data<src_emb_t>();
            dst_emb_t* network_comm_buffer_ptr = network_buffer.data.data<dst_emb_t>();
            offset_t* reverse_idx_ptr = reverse_idx.data<offset_t>();
            offset_t* bucket_ids_ptr = bucket_ids.data<offset_t>();
            auto hotness_range_ptr = network_indices.d_local_hotness_range.data<int>();
            auto ev_start_indices_ptr = network_indices.d_ev_start_indices.data<int>();
            int range_num = network_indices.local_lookup_num + 1;
            int global_ev_offset = network_indices.global_ev_offset;
            using CopyDesc =
                DenseNetworkBackwardBatchMajorOneToOneAtomicDesc<src_emb_t, dst_emb_t, offset_t>;
            CopyDesc one_to_one_atomic_desc = {
                num_network_reverse_idx, ev_size,           batch_size_per_gpu,     range_num,
                global_ev_offset,        hotness_range_ptr, ev_start_indices_ptr,   reverse_idx_ptr,
                bucket_ids_ptr,          top_grad_ptr,      network_comm_buffer_ptr};

            size_t num_valid_network_tensor = calc_num_valid_network_tensor(embedding_input);
            HCTR_LIB_THROW(cudaMemsetAsync(
                network_buffer.data.data(), 0,
                num_valid_network_tensor * ev_size * network_buffer.data.data_type().size(),
                stream));
            one_to_one_atomic(one_to_one_atomic_desc, kernel_params, ev_size,
                              num_network_reverse_idx, stream);
          });
    });
  });
}

}  // namespace
void NetworkBackward::sparse_backward(const core23::Tensor& dp_num_keys_per_bucket,
                                      const EmbeddingOutput& top_grad,
                                      const NetworkIndices& network_indices,
                                      NetworkBuffer& network_buffer, int batch_size) {
  HugeCTR::CudaDeviceContext ctx(core_->get_device_id());
  auto stream = core_->get_local_gpu()->get_stream();
  int gpu_id = core_->get_global_gpu_id();
  int num_gpus = core_->get_global_gpu_count();

  if (top_grad.attr.layout == EmbeddingLayout::FeatureMajor) {
    network_backward_from_feature_major_top_grad(dp_num_keys_per_bucket, top_grad, network_indices,
                                                 core_->get_kernel_param(), network_buffer,
                                                 batch_size, gpu_id, num_gpus, stream);
  } else {
    HCTR_ASSERT(top_grad.attr.layout == EmbeddingLayout::BatchMajor);
    network_backward_from_batch_major_top_grad(dp_num_keys_per_bucket, top_grad, network_indices,
                                               core_->get_kernel_param(), network_buffer,
                                               batch_size, gpu_id, num_gpus, stream);
  }
}

void NetworkBackward::dense_backward(const EmbeddingInput& embedding_input,
                                     const EmbeddingOutput& top_grad,
                                     const DenseNetworkIndices& network_indices,
                                     DenseNetworkBuffer& network_buffer, int batch_size) {
  HugeCTR::CudaDeviceContext ctx(core_->get_device_id());
  auto stream = core_->get_local_gpu()->get_stream();
  int gpu_id = core_->get_global_gpu_id();
  int num_gpus = core_->get_global_gpu_count();

  if (top_grad.attr.layout == EmbeddingLayout::FeatureMajor) {
    dense_network_backward_from_feature_major_top_grad(embedding_input, top_grad, network_indices,
                                                       core_->get_kernel_param(), network_buffer,
                                                       batch_size, gpu_id, num_gpus, stream);
  } else {
    HCTR_ASSERT(top_grad.attr.layout == EmbeddingLayout::BatchMajor);
    dense_network_backward_from_batch_major_top_grad(embedding_input, top_grad, network_indices,
                                                     core_->get_kernel_param(), network_buffer,
                                                     batch_size, gpu_id, num_gpus, stream);
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
  int batch_size_per_gpu = batch_size / core_->get_global_gpu_count();
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
