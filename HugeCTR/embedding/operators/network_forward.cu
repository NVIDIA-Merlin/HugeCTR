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

#include <embedding/operators/generic_lookup.cuh>
#include <embedding/operators/network_forward.hpp>
#include <utils.hpp>

namespace embedding {

void NetworkIndices::init(std::shared_ptr<CoreResourceManager> core,
                          const std::vector<std::vector<int>>& h_global_lookup_ids) {
  int num_gpus = static_cast<int>(h_global_lookup_ids.size());
  h_network_ids.clear();
  h_network_gpu_ids.clear();
  h_network_offsets.clear();
  h_network_dst_lookup_ids.clear();

  std::vector<std::tuple<int, int, int>> h_network_buffer_meta_info;
  for (int ggpu_id = 0; ggpu_id < num_gpus; ++ggpu_id) {
    int network_id = 0;
    for (int lookup_id : h_global_lookup_ids[ggpu_id]) {
      h_network_buffer_meta_info.push_back({ggpu_id, network_id, lookup_id});
      network_id += 1;
    }
  }

  std::sort(h_network_buffer_meta_info.begin(), h_network_buffer_meta_info.end(),
            [](const auto& lhs, const auto& rhs) { return std::get<2>(lhs) < std::get<2>(rhs); });

  for (size_t i = 0; i < h_network_buffer_meta_info.size(); ++i) {
    const auto& meta_info = h_network_buffer_meta_info[i];
    int network_gpu_id = std::get<0>(meta_info);
    int network_id = std::get<1>(meta_info);
    h_network_ids.push_back(network_id);
    h_network_gpu_ids.push_back(network_gpu_id);
  }

  int network_offset = 0;
  for (size_t i = 0; i < h_network_buffer_meta_info.size(); ++i) {
    const auto& meta_info = h_network_buffer_meta_info[i];
    int lookup_id = std::get<2>(meta_info);
    if (i == 0 || lookup_id != std::get<2>(h_network_buffer_meta_info[i - 1])) {
      h_network_offsets.push_back(network_offset);
    }
    network_offset += 1;
  }
  h_network_offsets.push_back(network_offset);

  for (size_t i = 0; i < h_network_buffer_meta_info.size(); ++i) {
    const auto& meta_info = h_network_buffer_meta_info[i];
    int lookup_id = std::get<2>(meta_info);
    if (i == 0 || lookup_id != std::get<2>(h_network_buffer_meta_info[i - 1])) {
      h_network_dst_lookup_ids.push_back(lookup_id);
    }
  }

  HugeCTR::CudaDeviceContext context(core->get_device_id());
  core23::Device device(core23::DeviceType::GPU, core->get_device_id());
  core23::BufferParams buffer_params;
  buffer_params.unitary = false;
  core23::TensorParams params = core23::TensorParams().device(device).buffer_params(buffer_params);

  this->network_ids = core23::Tensor(params.shape({static_cast<int64_t>(h_network_ids.size())})
                                         .data_type(core23::ScalarType::Int32));
  core23::copy_sync(this->network_ids, h_network_ids);
  this->network_gpu_ids =
      core23::Tensor(params.shape({static_cast<int64_t>(h_network_gpu_ids.size())})
                         .data_type(core23::ScalarType::Int32));
  core23::copy_sync(this->network_gpu_ids, h_network_gpu_ids);
  this->network_offsets =
      core23::Tensor(params.shape({static_cast<int64_t>(h_network_offsets.size())})
                         .data_type(core23::ScalarType::Int32));
  core23::copy_sync(this->network_offsets, h_network_offsets);
  this->network_dst_lookup_ids =
      core23::Tensor(params.shape({static_cast<int64_t>(h_network_dst_lookup_ids.size())})
                         .data_type(core23::ScalarType::Int32));
  core23::copy_sync(this->network_dst_lookup_ids, h_network_dst_lookup_ids);
}

void NetworkBufferAttr::init(std::shared_ptr<CoreResourceManager> core,
                             const EmbeddingCollectionParam& ebc_param, size_t grouped_id,
                             const std::vector<std::vector<int>>& h_global_lookup_ids) {
  const auto& group_params = ebc_param.grouped_lookup_params[grouped_id];
  HCTR_CHECK_HINT(group_params.embedding_group_type == EmbeddingGroupType::SparseModelParallel,
                  "UniformModelParallelEmbeddingMeta must be initialized by SparseModelParallel");

  this->num_gpus = static_cast<int>(h_global_lookup_ids.size());

  std::vector<std::vector<int>> h_id_to_ev_size;
  h_id_to_ev_size.resize(num_gpus);
  for (int ggpu_id = 0; ggpu_id < num_gpus; ++ggpu_id) {
    for (int lookup_id : h_global_lookup_ids[ggpu_id]) {
      int ev_size = ebc_param.lookup_params[lookup_id].ev_size;
      h_id_to_ev_size[ggpu_id].push_back(ev_size);
    }
  }

  std::vector<std::vector<int>> h_id_ev_start_indices;
  h_id_ev_start_indices.resize(num_gpus);
  for (int ggpu_id = 0; ggpu_id < num_gpus; ++ggpu_id) {
    h_id_ev_start_indices[ggpu_id].push_back(0);
    std::partial_sum(h_id_to_ev_size[ggpu_id].begin(), h_id_to_ev_size[ggpu_id].end(),
                     std::back_inserter(h_id_ev_start_indices[ggpu_id]));
  }

  HugeCTR::CudaDeviceContext context(core->get_device_id());
  core23::Device device(core23::DeviceType::GPU, core->get_device_id());
  core23::BufferParams buffer_params;
  buffer_params.unitary = false;
  core23::TensorParams params = core23::TensorParams().device(device).buffer_params(buffer_params);

  for (int ggpu_id = 0; ggpu_id < num_gpus; ++ggpu_id) {
    this->id_to_ev_size_list.emplace_back(
        params.shape({static_cast<int64_t>(h_id_to_ev_size[ggpu_id].size())})
            .data_type(core23::ScalarType::Int32));
    this->id_to_ev_start_indices_list.emplace_back(
        params.shape({static_cast<int64_t>(h_id_ev_start_indices[ggpu_id].size())})
            .data_type(core23::ScalarType::Int32));

    core23::copy_sync(this->id_to_ev_size_list[ggpu_id], h_id_to_ev_size[ggpu_id]);
    core23::copy_sync(this->id_to_ev_start_indices_list[ggpu_id], h_id_ev_start_indices[ggpu_id]);
  }
  this->id_to_ev_size =
      core23::init_tensor_list<int32_t>(this->id_to_ev_size_list, params.device().index());
  this->id_to_ev_start_indices =
      core23::init_tensor_list<int32_t>(this->id_to_ev_start_indices_list, params.device().index());

  this->gpu_id_to_max_ev_elements.clear();
  for (int ggpu_id = 0; ggpu_id < num_gpus; ++ggpu_id) {
    this->gpu_id_to_max_ev_elements.push_back(h_id_ev_start_indices[ggpu_id].back());
  }
  this->layout = EmbeddingLayout::FeatureMajor;
  this->max_ev_size = 0;
  for (int ggpu_id = 0; ggpu_id < num_gpus; ++ggpu_id) {
    for (auto ev_size : h_id_to_ev_size[ggpu_id]) {
      this->max_ev_size = std::max(this->max_ev_size, ev_size);
    }
  }
  this->type = ebc_param.emb_type;
}

void NetworkBuffer::init(std::shared_ptr<CoreResourceManager> core, const NetworkBufferAttr& attr,
                         int batch_size) {
  this->attr = attr;
  this->data_list.clear();

  int batch_size_per_gpu = batch_size / core->get_global_gpu_count();
  HugeCTR::CudaDeviceContext context(core->get_device_id());
  core23::Device device(core23::DeviceType::GPU, core->get_device_id());
  core23::TensorParams params = core23::TensorParams().device(device);
  for (int ggpu_id = 0; ggpu_id < attr.num_gpus; ++ggpu_id) {
    this->data_list.emplace_back(
        params.shape({batch_size_per_gpu * attr.gpu_id_to_max_ev_elements[ggpu_id]})
            .data_type(attr.type));
  }
  DISPATCH_FLOAT_AND_HALF_FUNCTION_CORE23(attr.type.type(), emb_t, [&] {
    this->data = core23::init_tensor_list<emb_t>(data_list, params.device().index());
  });
}
void DenseNetworkIndices::init(std::shared_ptr<CoreResourceManager> core,
                               const std::vector<int>& h_local_hotness_range_input,
                               const std::vector<int>& h_local_hotness_input,
                               const std::vector<int>& h_ev_start_indices_input,
                               const int local_lookup_num_input, const int global_ev_offset_input) {
  this->h_local_hotness_range.assign(h_local_hotness_range_input.begin(),
                                     h_local_hotness_range_input.end());
  this->h_local_hotness.assign(h_local_hotness_input.begin(), h_local_hotness_input.end());
  this->h_ev_start_indices.assign(h_ev_start_indices_input.begin(), h_ev_start_indices_input.end());
  this->local_lookup_num = local_lookup_num_input;
  this->global_ev_offset = global_ev_offset_input;

  HugeCTR::CudaDeviceContext context(core->get_device_id());
  core23::Device device(core23::DeviceType::GPU, core->get_device_id());
  core23::BufferParams buffer_params;
  buffer_params.unitary = false;
  core23::TensorParams params = core23::TensorParams().device(device).buffer_params(buffer_params);

  this->d_local_hotness_range =
      core23::Tensor(params.shape({static_cast<int64_t>(this->h_local_hotness_range.size())})
                         .data_type(core23::ScalarType::Int32));
  core23::copy_sync(this->d_local_hotness_range, this->h_local_hotness_range);

  this->d_local_hotness =
      core23::Tensor(params.shape({static_cast<int64_t>(h_local_hotness.size())})
                         .data_type(core23::ScalarType::Int32));
  core23::copy_sync(this->d_local_hotness, this->h_local_hotness);

  this->d_ev_start_indices =
      core23::Tensor(params.shape({static_cast<int64_t>(h_ev_start_indices.size())})
                         .data_type(core23::ScalarType::Int32));
  core23::copy_sync(this->d_ev_start_indices, this->h_ev_start_indices);
}

void DenseNetworkBufferAttr::init(std::shared_ptr<CoreResourceManager> core,
                                  const EmbeddingCollectionParam& ebc_param, size_t grouped_id,
                                  int max_hotness) {
  const auto& group_params = ebc_param.grouped_lookup_params[grouped_id];
  HCTR_CHECK_HINT(
      group_params.embedding_group_type == EmbeddingGroupType::DenseModelParallel ||
          group_params.embedding_group_type == EmbeddingGroupType::DenseModelParallelWithReduction,
      "DenseNetworkBufferAttr must be initialized by DenseModelParallel or "
      "DenseModelParallelWithReduction");

  this->num_lookup = group_params.lookup_ids.size();

  HCTR_CHECK_HINT(this->num_lookup > 0, "DenseNetworkBufferAttr must have lookup , but now is <=0");
  const auto& lookup_params = ebc_param.lookup_params;
  this->ev_size = lookup_params[group_params.lookup_ids[0]].ev_size;
  this->max_hotness = max_hotness;
  this->layout = EmbeddingLayout::FeatureMajor;
  this->type = ebc_param.emb_type;
}

void DenseNetworkBuffer::init(std::shared_ptr<CoreResourceManager> core,
                              const DenseNetworkBufferAttr& attr, int batch_size) {
  this->attr = attr;

  HugeCTR::CudaDeviceContext context(core->get_device_id());

  core23::Device device(core23::DeviceType::GPU, core->get_device_id());
  core23::TensorParams params = core23::TensorParams().device(device);

  double dense_unique_ratio = get_dense_unique_ratio();
  int64_t max_num_elements = static_cast<int64_t>(batch_size) * attr.max_hotness * attr.ev_size;
  int64_t num_elements =
      static_cast<int64_t>(dense_unique_ratio * static_cast<double>(max_num_elements));
  this->data = core23::Tensor(params.shape({num_elements}).data_type(attr.type));
}

NetworkForward::NetworkForward(std::shared_ptr<CoreResourceManager> core) : core_(core) {}

namespace {
// sparse
void network_forward_to_batch_major_output(const core23::Tensor& dp_num_keys_per_bucket,
                                           const NetworkBuffer& network_buffer,
                                           const NetworkIndices& network_indices,
                                           const HugeCTR::core23::KernelParams& kernel_params,
                                           EmbeddingOutput& embedding_output, int batch_size,
                                           int gpu_id, int num_gpus, cudaStream_t stream) {
  int batch_size_per_gpu = batch_size / num_gpus;
  auto& network_comm_buffer = network_buffer.data;
  auto& output_buffer = embedding_output.data;
  auto& network_attr = network_buffer.attr;
  auto& output_attr = embedding_output.attr;
  int max_ev_size = output_attr.max_ev_size;
  int num_lookup = output_attr.id_to_ev_size.num_elements();

  DISPATCH_INTEGRAL_FUNCTION_CORE23(dp_num_keys_per_bucket.data_type().type(), offset_t, [&] {
    DISPATCH_FLOAT_AND_HALF_FUNCTION_CORE23(network_comm_buffer.data_type().type(), emb_t, [&] {
      DISPATCH_FLOAT_AND_HALF_FUNCTION_CORE23(output_buffer.data_type().type(), dst_emb_t, [&] {
        const offset_t* dp_num_keys_per_bucket_ptr = dp_num_keys_per_bucket.data<offset_t>();
        const int* network_ids_ptr = network_indices.network_ids.data<int>();
        const int* network_gpu_ids_ptr = network_indices.network_gpu_ids.data<int>();
        const int* network_offsets_ptr = network_indices.network_offsets.data<int>();
        const int* network_dst_lookup_ids_ptr = network_indices.network_dst_lookup_ids.data<int>();
        const int** network_ev_sizes_ptr = (const int**)network_attr.id_to_ev_size.data();
        const int** network_ev_offsets_ptr =
            (const int**)network_attr.id_to_ev_start_indices.data();
        const emb_t** network_comm_buffer_ptr = (const emb_t**)network_comm_buffer.data();
        const int* dst_ev_start_indices_ptr = output_attr.id_to_ev_start_indices.data<int>();
        const char* dst_combiner_ptr = output_attr.id_to_combiner.data<char>();
        dst_emb_t* output_buffer_ptr = output_buffer.data<dst_emb_t>();
        int num_network_dst_lookup_ids = network_indices.network_dst_lookup_ids.num_elements();

        auto multi_to_one_desc = make_MultiToOne<emb_t, dst_emb_t>(
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

              if (dst_combiner_ptr[lookup_id] == static_cast<char>(Combiner::Average)) {
                int idx = batch_size_per_gpu * lookup_id + bid;
                return static_cast<int>(dp_num_keys_per_bucket_ptr[idx]);
              } else {
                return 1;
              }
            },
            [=] __device__(int i) {
              int dst_lookup_id = network_dst_lookup_ids_ptr[i % num_network_dst_lookup_ids];
              return dst_ev_start_indices_ptr[dst_lookup_id + 1] -
                     dst_ev_start_indices_ptr[dst_lookup_id];
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
            },
            [=] __device__(int i) {
              int bid = i / num_network_dst_lookup_ids;
              int lookup_id = network_dst_lookup_ids_ptr[i % num_network_dst_lookup_ids];
              int ev_offset = dst_ev_start_indices_ptr[num_lookup] * bid;

              return output_buffer_ptr + ev_offset + dst_ev_start_indices_ptr[lookup_id];
            });
        copy_multi_to_one(multi_to_one_desc, kernel_params, max_ev_size, stream);
      });
    });
  });
}
// sparse

void network_forward_to_feature_major_output(const core23::Tensor& dp_num_keys_per_bucket,
                                             const NetworkBuffer& network_buffer,
                                             const NetworkIndices& network_indices,
                                             const HugeCTR::core23::KernelParams& kernel_params,
                                             EmbeddingOutput& embedding_output, int batch_size,
                                             int gpu_id, int num_gpus, cudaStream_t stream) {
  int batch_size_per_gpu = batch_size / num_gpus;
  auto& network_comm_buffer = network_buffer.data;
  auto& output_buffer = embedding_output.data;
  auto& network_attr = network_buffer.attr;
  auto& output_attr = embedding_output.attr;
  int max_ev_size = output_attr.max_ev_size;

  DISPATCH_INTEGRAL_FUNCTION_CORE23(dp_num_keys_per_bucket.data_type().type(), offset_t, [&] {
    DISPATCH_FLOAT_AND_HALF_FUNCTION_CORE23(network_comm_buffer.data_type().type(), emb_t, [&] {
      DISPATCH_FLOAT_AND_HALF_FUNCTION_CORE23(output_buffer.data_type().type(), dst_emb_t, [&] {
        const offset_t* dp_num_keys_per_bucket_ptr = dp_num_keys_per_bucket.data<offset_t>();
        const int* network_ids_ptr = network_indices.network_ids.data<int>();
        const int* network_gpu_ids_ptr = network_indices.network_gpu_ids.data<int>();
        const int* network_offsets_ptr = network_indices.network_offsets.data<int>();
        const int* network_dst_lookup_ids_ptr = network_indices.network_dst_lookup_ids.data<int>();
        const int** network_ev_sizes_ptr = (const int**)network_attr.id_to_ev_size.data();
        const int** network_ev_offsets_ptr =
            (const int**)network_attr.id_to_ev_start_indices.data();
        const emb_t** network_comm_buffer_ptr = (const emb_t**)network_comm_buffer.data();
        const int* dst_ev_start_indices_ptr = output_attr.id_to_ev_start_indices.data<int>();
        const char* dst_combiner_ptr = output_attr.id_to_combiner.data<char>();
        dst_emb_t* output_buffer_ptr = output_buffer.data<dst_emb_t>();
        int num_network_dst_lookup_ids = network_indices.network_dst_lookup_ids.num_elements();

        auto multi_to_one_desc = make_MultiToOne<emb_t, dst_emb_t>(
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

              if (dst_combiner_ptr[lookup_id] == static_cast<char>(Combiner::Average)) {
                int idx = batch_size_per_gpu * lookup_id + bid;
                return static_cast<int>(dp_num_keys_per_bucket_ptr[idx]);
              } else {
                return 1;
              }
            },
            [=] __device__(int i) {
              int dst_lookup_id = network_dst_lookup_ids_ptr[i % num_network_dst_lookup_ids];
              return dst_ev_start_indices_ptr[dst_lookup_id + 1] -
                     dst_ev_start_indices_ptr[dst_lookup_id];
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
            },
            [=] __device__(int i) {
              int bid = i / num_network_dst_lookup_ids;
              int lookup_id = network_dst_lookup_ids_ptr[i % num_network_dst_lookup_ids];

              int ev_offset = dst_ev_start_indices_ptr[lookup_id] * batch_size_per_gpu;
              int ev_size =
                  dst_ev_start_indices_ptr[lookup_id + 1] - dst_ev_start_indices_ptr[lookup_id];
              return output_buffer_ptr + ev_offset + bid * ev_size;
            });
        copy_multi_to_one(multi_to_one_desc, kernel_params, max_ev_size, stream);
      });
    });
  });
}

template <typename src_emb_t, typename dst_emb_t, typename offset_t>
struct DenseNetworkForwardFeatureMajorOneToOneDesc {
  using SrcT = src_emb_t;
  using DstT = dst_emb_t;

  HOST_DEVICE_INLINE int num_vec() { return num_vec_; }
  HOST_DEVICE_INLINE bool need_copy(int i) { return true; }

  HOST_DEVICE_INLINE int get_vec_length(int i) { return ev_size; }
  // we need a transform to src id use num_model_revers_idx
  HOST_DEVICE_INLINE const SrcT* get_src_ptr(int i) {
    return src_ptr + reverse_id_ptr[i] * ev_size;
  }
  HOST_DEVICE_INLINE DstT* get_dst_ptr(int i) {
    int hotness_id = bucket_id_ptr[i] / batch_size_per_gpu;
    int64_t lookup_id = bs_upper_bound_sub_one(hotness_range, range_num, hotness_id);
    offset_t bucket_id = bucket_id_ptr[i];
    hotness_id = hotness_id - hotness_range[lookup_id];
    int bid = bucket_id % batch_size_per_gpu;
    return dst_ptr + batch_size_per_gpu * ev_start_indices[lookup_id] +
           bid * hotness_list[lookup_id] * ev_size + hotness_id * ev_size;
  }

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

template <typename src_emb_t, typename dst_emb_t, typename offset_t>
struct DenseNetworkForwardBatchMajorOneToOneDesc {
  using SrcT = src_emb_t;
  using DstT = dst_emb_t;
  HOST_DEVICE_INLINE int num_vec() { return num_vec_; }

  HOST_DEVICE_INLINE bool need_copy(int i) { return true; }

  HOST_DEVICE_INLINE int get_vec_length(int i) { return ev_size; }
  // we need a transform to src id use num_model_revers_idx
  HOST_DEVICE_INLINE const SrcT* get_src_ptr(int i) {
    return src_ptr + reverse_id_ptr[i] * ev_size;
  }
  HOST_DEVICE_INLINE DstT* get_dst_ptr(int i) {
    int hotness_id = bucket_id_ptr[i] / batch_size_per_gpu;
    int64_t lookup_id = bs_upper_bound_sub_one(hotness_range, range_num, hotness_id);
    offset_t bucket_id = bucket_id_ptr[i];
    hotness_id = hotness_id - hotness_range[lookup_id];
    int bid = bucket_id % batch_size_per_gpu;

    return dst_ptr + bid * global_ev_offset + ev_start_indices[lookup_id] + hotness_id * ev_size;
  }

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

// dense
void dense_network_forward_to_batch_major_output(const EmbeddingInput& embedding_input,
                                                 const DenseNetworkBuffer& network_buffer,
                                                 const DenseNetworkIndices& network_indices,
                                                 const HugeCTR::core23::KernelParams& kernel_params,
                                                 EmbeddingOutput& embedding_output, int batch_size,
                                                 int gpu_id, int num_gpus, cudaStream_t stream,
                                                 bool do_reduction) {
  int batch_size_per_gpu = batch_size / num_gpus;

  int ev_size = network_buffer.attr.ev_size;
  size_t num_key = embedding_input.dense_compression_input.model_parallel_compression_input
                       .num_network_reverse_idx;

  auto& network_comm_buffer = network_buffer.data;
  auto& output_buffer = embedding_output.data;
  auto& reverse_idx =
      embedding_input.dense_compression_input.model_parallel_compression_input.network_reverse_idx;
  auto& bucket_ids = embedding_input.dense_compression_input.model_parallel_compression_input
                         .network_dst_bucket_ids;
  auto& num_network_reverse_idx = embedding_input.dense_compression_input
                                      .model_parallel_compression_input.num_network_reverse_idx;
  DISPATCH_INTEGRAL_FUNCTION_CORE23(reverse_idx.data_type().type(), offset_t, [&] {
    DISPATCH_FLOAT_AND_HALF_FUNCTION_CORE23(network_comm_buffer.data_type().type(), emb_t, [&] {
      DISPATCH_FLOAT_AND_HALF_FUNCTION_CORE23(output_buffer.data_type().type(), dst_emb_t, [&] {
        const emb_t* network_comm_buffer_ptr = (const emb_t*)network_comm_buffer.data();
        dst_emb_t* output_buffer_ptr = output_buffer.data<dst_emb_t>();
        offset_t* reverse_idx_ptr = reverse_idx.data<offset_t>();
        offset_t* bucket_ids_ptr = bucket_ids.data<offset_t>();
        auto hotness_range_ptr = network_indices.d_local_hotness_range.data<int>();
        auto ev_start_indices_ptr = network_indices.d_ev_start_indices.data<int>();
        int range_num = network_indices.local_lookup_num + 1;
        int global_ev_offset = network_indices.global_ev_offset;
        using CopyDesc = DenseNetworkForwardBatchMajorOneToOneDesc<emb_t, dst_emb_t, offset_t>;

        CopyDesc one_to_one_desc = {num_network_reverse_idx, ev_size,
                                    batch_size_per_gpu,      range_num,
                                    global_ev_offset,        hotness_range_ptr,
                                    ev_start_indices_ptr,    reverse_idx_ptr,
                                    bucket_ids_ptr,          network_comm_buffer_ptr,
                                    output_buffer_ptr};
        if (do_reduction) {
          copy_one_to_one(one_to_one_desc, kernel_params, ev_size, stream, true);
          one_to_one_atomic(one_to_one_desc, kernel_params, ev_size, num_network_reverse_idx,
                            stream);

        } else {
          copy_one_to_one(one_to_one_desc, kernel_params, ev_size, stream, false);
        }
      });
    });
  });
}

// network is input;
// output is embedding_output
void dense_network_forward_to_feature_major_output(
    const EmbeddingInput& embedding_input, const DenseNetworkBuffer& network_buffer,
    const DenseNetworkIndices& network_indices, const HugeCTR::core23::KernelParams& kernel_params,
    EmbeddingOutput& embedding_output, int batch_size, int gpu_id, int num_gpus,
    cudaStream_t stream, bool do_reduction) {
  int batch_size_per_gpu = batch_size / num_gpus;

  int ev_size = network_buffer.attr.ev_size;
  size_t num_key = embedding_input.dense_compression_input.model_parallel_compression_input
                       .num_network_reverse_idx;

  auto& network_comm_buffer = network_buffer.data;
  auto& output_buffer = embedding_output.data;
  auto& reverse_idx =
      embedding_input.dense_compression_input.model_parallel_compression_input.network_reverse_idx;
  auto& bucket_ids = embedding_input.dense_compression_input.model_parallel_compression_input
                         .network_dst_bucket_ids;
  auto& num_network_reverse_idx = embedding_input.dense_compression_input
                                      .model_parallel_compression_input.num_network_reverse_idx;
  DISPATCH_INTEGRAL_FUNCTION_CORE23(reverse_idx.data_type().type(), offset_t, [&] {
    DISPATCH_FLOAT_AND_HALF_FUNCTION_CORE23(network_comm_buffer.data_type().type(), emb_t, [&] {
      DISPATCH_FLOAT_AND_HALF_FUNCTION_CORE23(output_buffer.data_type().type(), dst_emb_t, [&] {
        const emb_t* network_comm_buffer_ptr = (const emb_t*)network_comm_buffer.data();
        dst_emb_t* output_buffer_ptr = output_buffer.data<dst_emb_t>();
        offset_t* reverse_idx_ptr = reverse_idx.data<offset_t>();
        offset_t* bucket_ids_ptr = bucket_ids.data<offset_t>();
        int range_num = network_indices.local_lookup_num + 1;

        auto hotness_range_ptr = network_indices.d_local_hotness_range.data<int>();
        auto ev_start_indices_ptr = network_indices.d_ev_start_indices.data<int>();
        auto hotness_list = network_indices.d_local_hotness.data<int>();
        using CopyDesc = DenseNetworkForwardFeatureMajorOneToOneDesc<emb_t, dst_emb_t, offset_t>;
        CopyDesc one_to_one_desc = {num_network_reverse_idx,
                                    ev_size,
                                    batch_size_per_gpu,
                                    range_num,
                                    hotness_range_ptr,
                                    ev_start_indices_ptr,
                                    hotness_list,
                                    reverse_idx_ptr,
                                    bucket_ids_ptr,
                                    network_comm_buffer_ptr,
                                    output_buffer_ptr};

        if (do_reduction) {
          copy_one_to_one(one_to_one_desc, kernel_params, ev_size, stream, true);
          one_to_one_atomic(one_to_one_desc, kernel_params, ev_size, num_network_reverse_idx,
                            stream);
        } else {
          copy_one_to_one(one_to_one_desc, kernel_params, ev_size, stream, false);
        }
      });
    });
  });
}

}  // namespace

void NetworkForward::sparse_forward(const core23::Tensor& dp_num_keys_per_bucket,
                                    const NetworkBuffer& network_buffer,
                                    const NetworkIndices& network_indices,
                                    EmbeddingOutput& embedding_output, int batch_size) {
  HugeCTR::CudaDeviceContext ctx(core_->get_device_id());
  auto stream = core_->get_local_gpu()->get_stream();
  int gpu_id = core_->get_global_gpu_id();
  int num_gpus = core_->get_global_gpu_count();

  if (embedding_output.attr.layout == EmbeddingLayout::FeatureMajor) {
    network_forward_to_feature_major_output(dp_num_keys_per_bucket, network_buffer, network_indices,
                                            core_->get_kernel_param(), embedding_output, batch_size,
                                            gpu_id, num_gpus, stream);
  } else {
    HCTR_ASSERT(embedding_output.attr.layout == EmbeddingLayout::BatchMajor);
    network_forward_to_batch_major_output(dp_num_keys_per_bucket, network_buffer, network_indices,
                                          core_->get_kernel_param(), embedding_output, batch_size,
                                          gpu_id, num_gpus, stream);
  }
}

void NetworkForward::dense_forward(const EmbeddingInput& embedding_input,
                                   const DenseNetworkBuffer& network_buffer,
                                   const DenseNetworkIndices& network_indices,
                                   EmbeddingOutput& embedding_output, int batch_size,
                                   bool do_reduction) {
  HugeCTR::CudaDeviceContext ctx(core_->get_device_id());
  auto stream = core_->get_local_gpu()->get_stream();
  int gpu_id = core_->get_global_gpu_id();
  int num_gpus = core_->get_global_gpu_count();

  if (embedding_output.attr.layout == EmbeddingLayout::FeatureMajor) {
    dense_network_forward_to_feature_major_output(
        embedding_input, network_buffer, network_indices, core_->get_kernel_param(),
        embedding_output, batch_size, gpu_id, num_gpus, stream, do_reduction);
  } else {
    HCTR_ASSERT(embedding_output.attr.layout == EmbeddingLayout::BatchMajor);
    dense_network_forward_to_batch_major_output(embedding_input, network_buffer, network_indices,
                                                core_->get_kernel_param(), embedding_output,
                                                batch_size, gpu_id, num_gpus, stream, do_reduction);
  }
}

void NetworkForward::compute(
    const core23::Tensor& row_lengths, const core23::Tensor& d_combiner_list,
    const core23::Tensor& network_comm_buffer, const core23::Tensor& network_ids,
    const core23::Tensor& network_gpu_ids, const core23::Tensor& network_offsets,
    const core23::Tensor& network_dst_lookup_ids, const core23::Tensor& network_ev_sizes,
    const core23::Tensor& network_ev_offsets, core23::Tensor& output_buffer,
    const core23::Tensor& d_ev_size_offset, int batch_size, int max_ev_size) {
  HugeCTR::CudaDeviceContext ctx(core_->get_device_id());
  int batch_size_per_gpu = batch_size / core_->get_global_gpu_count();
  DISPATCH_INTEGRAL_FUNCTION_CORE23(row_lengths.data_type().type(), offset_t, [&] {
    DISPATCH_FLOAT_AND_HALF_FUNCTION_CORE23(network_comm_buffer.data_type().type(), emb_t, [&] {
      DISPATCH_FLOAT_AND_HALF_FUNCTION_CORE23(output_buffer.data_type().type(), dst_emb_t, [&] {
        auto stream = core_->get_local_gpu()->get_stream();

        const offset_t** row_lengths_ptr = (const offset_t**)row_lengths.data();
        const int* network_ids_ptr = network_ids.data<int>();
        const int* network_gpu_ids_ptr = network_gpu_ids.data<int>();
        const int* network_offsets_ptr = network_offsets.data<int>();
        const int* network_dst_lookup_ids_ptr = network_dst_lookup_ids.data<int>();
        const int** network_ev_sizes_ptr = (const int**)network_ev_sizes.data();
        const int** network_ev_offsets_ptr = (const int**)network_ev_offsets.data();
        const emb_t** network_comm_buffer_ptr = (const emb_t**)network_comm_buffer.data();
        const int* d_ev_size_offset_ptr = d_ev_size_offset.data<int>();
        const char* combiner_ptr = d_combiner_list.data<char>();
        dst_emb_t** output_buffer_ptr = (dst_emb_t**)output_buffer.data();
        int num_network_dst_lookup_ids = network_dst_lookup_ids.num_elements();
        int gpu_id = core_->get_global_gpu_id();

        auto multi_to_one_desc = make_MultiToOne<emb_t, dst_emb_t>(
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
              return d_ev_size_offset_ptr[dst_lookup_id + 1] - d_ev_size_offset_ptr[dst_lookup_id];
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
            },
            [=] __device__(int i) {
              int bid = i / num_network_dst_lookup_ids;
              int lookup_id = network_dst_lookup_ids_ptr[i % num_network_dst_lookup_ids];

              int ev_size = d_ev_size_offset_ptr[lookup_id + 1] - d_ev_size_offset_ptr[lookup_id];
              return output_buffer_ptr[lookup_id] + bid * ev_size;
            });
        copy_multi_to_one(multi_to_one_desc, max_ev_size, stream);
      });
    });
  });
}

}  // namespace embedding
