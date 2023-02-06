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
using namespace core;

std::vector<size_t> cal_network_comm_buffer_size(
    int universal_batch_size, int num_gpus,
    const std::vector<std::vector<int>>& global_lookup_id_list,
    const std::vector<int>& ev_size_list) {
  int batch_size_per_gpu = universal_batch_size / num_gpus;

  std::vector<size_t> network_comm_buffer_size;
  for (int global_gpu_id = 0; global_gpu_id < num_gpus; ++global_gpu_id) {
    auto& remote_lookup_id_list = global_lookup_id_list[global_gpu_id];
    size_t num_ev_elements = 0;
    for (int lookup_id : remote_lookup_id_list) {
      num_ev_elements += ev_size_list[lookup_id] * batch_size_per_gpu;
    }
    network_comm_buffer_size.push_back(num_ev_elements);
  }
  return network_comm_buffer_size;
}

void NetworkIndices::init(std::shared_ptr<CoreResourceManager> core,
                          const std::vector<std::vector<int>>& h_global_lookup_ids) {
  int num_gpus = core->get_global_gpu_count();

  std::vector<std::tuple<int, int, int>> h_network_buffer_meta_info;
  for (int ggpu_id = 0; ggpu_id < num_gpus; ++ggpu_id) {
    int network_id = 0;
    for (int lookup_id : h_global_lookup_ids[ggpu_id]) {
      h_network_buffer_meta_info.push_back({ggpu_id, network_id, lookup_id});
      network_id += 1;
    }
  }

  std::sort(h_network_buffer_meta_info.begin(), h_network_buffer_meta_info.end(),
            [](const auto& lhs, const auto& rhs) { return std::get<2>(lhs) <= std::get<2>(rhs); });

  std::vector<int> h_network_ids;
  std::vector<int> h_network_gpu_ids;
  for (size_t i = 0; i < h_network_buffer_meta_info.size(); ++i) {
    const auto& meta_info = h_network_buffer_meta_info[i];
    int network_gpu_id = std::get<0>(meta_info);
    int network_id = std::get<1>(meta_info);
    h_network_ids.push_back(network_id);
    h_network_gpu_ids.push_back(network_gpu_id);
  }

  std::vector<int> h_network_offsets;
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

  std::vector<int> h_network_dst_lookup_ids;
  for (size_t i = 0; i < h_network_buffer_meta_info.size(); ++i) {
    const auto& meta_info = h_network_buffer_meta_info[i];
    int lookup_id = std::get<2>(meta_info);
    if (i == 0 || lookup_id != std::get<2>(h_network_buffer_meta_info[i - 1])) {
      h_network_dst_lookup_ids.push_back(lookup_id);
    }
  }

  HugeCTR::CudaDeviceContext context(core->get_device_id());
  auto buffer_ptr = GetBuffer(core);
  this->network_ids =
      buffer_ptr->reserve({h_network_ids.size()}, DeviceType::GPU, TensorScalarType::Int32);
  this->network_gpu_ids =
      buffer_ptr->reserve({h_network_gpu_ids.size()}, DeviceType::GPU, TensorScalarType::Int32);
  this->network_offsets =
      buffer_ptr->reserve({h_network_offsets.size()}, DeviceType::GPU, TensorScalarType::Int32);
  this->network_dst_lookup_ids = buffer_ptr->reserve({h_network_dst_lookup_ids.size()},
                                                     DeviceType::GPU, TensorScalarType::Int32);
  buffer_ptr->allocate();

  this->network_ids.copy_from(h_network_ids);
  this->network_gpu_ids.copy_from(h_network_gpu_ids);
  this->network_offsets.copy_from(h_network_offsets);
  this->network_dst_lookup_ids.copy_from(h_network_dst_lookup_ids);
}

void NetworkBufferAttr::init(std::shared_ptr<CoreResourceManager> core,
                             const EmbeddingCollectionParam& ebc_param, size_t grouped_id,
                             const std::vector<std::vector<int>>& h_global_lookup_ids) {
  HugeCTR::CudaDeviceContext context(core->get_device_id());
  const auto& group_params = ebc_param.grouped_emb_params[grouped_id];
  HCTR_CHECK_HINT(group_params.table_placement_strategy == TablePlacementStrategy::ModelParallel,
                  "UniformModelParallelEmbeddingMeta must be initialized by ModelParallel");

  this->num_gpus = core->get_global_gpu_count();

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

  auto buffer_ptr = GetBuffer(core);
  for (int ggpu_id = 0; ggpu_id < num_gpus; ++ggpu_id) {
    this->id_to_ev_size_list.push_back(buffer_ptr->reserve(
        {h_id_to_ev_size[ggpu_id].size()}, DeviceType::GPU, TensorScalarType::Int32));
    this->id_to_ev_start_indices_list.push_back(buffer_ptr->reserve(
        {h_id_ev_start_indices[ggpu_id].size()}, DeviceType::GPU, TensorScalarType::Int32));
  }
  buffer_ptr->allocate();
  for (int ggpu_id = 0; ggpu_id < num_gpus; ++ggpu_id) {
    this->id_to_ev_size_list[ggpu_id].copy_from(h_id_to_ev_size[ggpu_id]);
    this->id_to_ev_start_indices_list[ggpu_id].copy_from(h_id_ev_start_indices[ggpu_id]);
  }
  this->id_to_ev_size =
      TensorList(core.get(), id_to_ev_size_list, DeviceType::GPU, TensorScalarType::Int32);
  this->id_to_ev_start_indices =
      TensorList(core.get(), id_to_ev_start_indices_list, DeviceType::GPU, TensorScalarType::Int32);

  this->gpu_id_to_max_ev_elements.clear();
  for (int ggpu_id = 0; ggpu_id < num_gpus; ++ggpu_id) {
    this->gpu_id_to_max_ev_elements.push_back(h_id_ev_start_indices[ggpu_id].back());
  }
  this->layout = EmbeddingLayout::FeatureMajor;
  int max_ev_size = 0;
  bool aligned = true;
  bool ragged = false;
  for (int ggpu_id = 0; ggpu_id < num_gpus; ++ggpu_id) {
    for (auto ev_size : h_id_to_ev_size[ggpu_id]) {
      max_ev_size = std::max(max_ev_size, ev_size);
      if (ev_size % 4 != 0) aligned = false;
      if (max_ev_size != ev_size) ragged = true;
    }
  }
  this->max_ev_size = max_ev_size;
  this->is_ragged = ragged;
  this->is_aligned = aligned;
  this->type = ebc_param.emb_type;
}

void NetworkBuffer::init(std::shared_ptr<CoreResourceManager> core, const NetworkBufferAttr& attr,
                         int batch_size) {
  this->attr = attr;
  this->data_list.clear();

  int batch_size_per_gpu = batch_size / attr.num_gpus;
  HugeCTR::CudaDeviceContext context(core->get_device_id());
  auto buffer_ptr = GetBuffer(core);
  for (int ggpu_id = 0; ggpu_id < attr.num_gpus; ++ggpu_id) {
    this->data_list.push_back(buffer_ptr->reserve(
        batch_size_per_gpu * attr.gpu_id_to_max_ev_elements[ggpu_id], DeviceType::GPU, attr.type));
  }
  buffer_ptr->allocate();

  this->data = TensorList(core.get(), data_list, DeviceType::GPU, attr.type);
}

NetworkForward::NetworkForward(std::shared_ptr<CoreResourceManager> core, int num_gpus)
    : core_(core), num_gpus_(num_gpus) {}

namespace {

void network_forward_to_batch_major_output(const Tensor& bucket_range,
                                           const NetworkBuffer& network_buffer,
                                           const NetworkIndices& network_indices,
                                           EmbeddingOutput& embedding_output, int batch_size,
                                           int gpu_id, int num_gpus, cudaStream_t stream) {
  int batch_size_per_gpu = batch_size / num_gpus;
  auto& network_comm_buffer = network_buffer.data;
  auto& output_buffer = embedding_output.data;
  auto& network_attr = network_buffer.attr;
  auto& output_attr = embedding_output.attr;
  int max_ev_size = output_attr.max_ev_size;
  int num_lookup = output_attr.id_to_ev_size.get_num_elements();

  DISPATCH_INTEGRAL_FUNCTION(bucket_range.dtype().type(), offset_t, [&] {
    DISPATCH_FLOAT_AND_HALF_FUNCTION(network_comm_buffer.dtype().type(), emb_t, [&] {
      DISPATCH_FLOAT_AND_HALF_FUNCTION(output_buffer.dtype().type(), dst_emb_t, [&] {
        const offset_t* bucket_range_ptr = bucket_range.get<offset_t>();
        const int* network_ids_ptr = network_indices.network_ids.get<int>();
        const int* network_gpu_ids_ptr = network_indices.network_gpu_ids.get<int>();
        const int* network_offsets_ptr = network_indices.network_offsets.get<int>();
        const int* network_dst_lookup_ids_ptr = network_indices.network_dst_lookup_ids.get<int>();
        const int** network_ev_sizes_ptr = network_attr.id_to_ev_size.get<int>();
        const int** network_ev_offsets_ptr = network_attr.id_to_ev_start_indices.get<int>();
        const emb_t** network_comm_buffer_ptr = network_comm_buffer.get<emb_t>();
        const int* dst_ev_start_indices_ptr = output_attr.id_to_ev_start_indices.get<int>();
        const char* dst_combiner_ptr = output_attr.id_to_combiner.get<char>();
        dst_emb_t* output_buffer_ptr = output_buffer.get<dst_emb_t>();
        int num_network_dst_lookup_ids = network_indices.network_dst_lookup_ids.get_num_elements();

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
                int start = batch_size * lookup_id + gpu_id * batch_size_per_gpu + bid;
                return static_cast<int>(bucket_range_ptr[start + 1] - bucket_range_ptr[start]);
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
        copy_multi_to_one(multi_to_one_desc, max_ev_size, stream);
      });
    });
  });
}

void network_forward_to_feature_major_output(const Tensor& bucket_range,
                                             const NetworkBuffer& network_buffer,
                                             const NetworkIndices& network_indices,
                                             EmbeddingOutput& embedding_output, int batch_size,
                                             int gpu_id, int num_gpus, cudaStream_t stream) {
  int batch_size_per_gpu = batch_size / num_gpus;
  auto& network_comm_buffer = network_buffer.data;
  auto& output_buffer = embedding_output.data;
  auto& network_attr = network_buffer.attr;
  auto& output_attr = embedding_output.attr;
  int max_ev_size = output_attr.max_ev_size;

  DISPATCH_INTEGRAL_FUNCTION(bucket_range.dtype().type(), offset_t, [&] {
    DISPATCH_FLOAT_AND_HALF_FUNCTION(network_comm_buffer.dtype().type(), emb_t, [&] {
      DISPATCH_FLOAT_AND_HALF_FUNCTION(output_buffer.dtype().type(), dst_emb_t, [&] {
        const offset_t* bucket_range_ptr = bucket_range.get<offset_t>();
        const int* network_ids_ptr = network_indices.network_ids.get<int>();
        const int* network_gpu_ids_ptr = network_indices.network_gpu_ids.get<int>();
        const int* network_offsets_ptr = network_indices.network_offsets.get<int>();
        const int* network_dst_lookup_ids_ptr = network_indices.network_dst_lookup_ids.get<int>();
        const int** network_ev_sizes_ptr = network_attr.id_to_ev_size.get<int>();
        const int** network_ev_offsets_ptr = network_attr.id_to_ev_start_indices.get<int>();
        const emb_t** network_comm_buffer_ptr = network_comm_buffer.get<emb_t>();
        const int* dst_ev_start_indices_ptr = output_attr.id_to_ev_start_indices.get<int>();
        const char* dst_combiner_ptr = output_attr.id_to_combiner.get<char>();
        dst_emb_t* output_buffer_ptr = output_buffer.get<dst_emb_t>();
        int num_network_dst_lookup_ids = network_indices.network_dst_lookup_ids.get_num_elements();

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
                int start = batch_size * lookup_id + gpu_id * batch_size_per_gpu + bid;
                return static_cast<int>(bucket_range_ptr[start + 1] - bucket_range_ptr[start]);
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
        copy_multi_to_one(multi_to_one_desc, max_ev_size, stream);
      });
    });
  });
}

}  // namespace

void NetworkForward::compute(const Tensor& bucket_range, const NetworkBuffer& network_buffer,
                             const NetworkIndices& network_indices,
                             EmbeddingOutput& embedding_output, int batch_size) {
  HugeCTR::CudaDeviceContext ctx(core_->get_device_id());
  auto stream = core_->get_local_gpu()->get_stream();
  int gpu_id = core_->get_global_gpu_id();
  int num_gpus = core_->get_global_gpu_count();

  if (embedding_output.attr.layout == EmbeddingLayout::FeatureMajor) {
    network_forward_to_feature_major_output(bucket_range, network_buffer, network_indices,
                                            embedding_output, batch_size, gpu_id, num_gpus, stream);
  } else {
    HCTR_ASSERT(embedding_output.attr.layout == EmbeddingLayout::BatchMajor);
    network_forward_to_batch_major_output(bucket_range, network_buffer, network_indices,
                                          embedding_output, batch_size, gpu_id, num_gpus, stream);
  }
}

void NetworkForward::compute(const TensorList& row_lengths, const Tensor& d_combiner_list,
                             const TensorList& network_comm_buffer, const Tensor& network_ids,
                             const Tensor& network_gpu_ids, const Tensor& network_offsets,
                             const Tensor& network_dst_lookup_ids,
                             const TensorList& network_ev_sizes,
                             const TensorList& network_ev_offsets, TensorList& output_buffer,
                             const Tensor& d_ev_size_offset, int batch_size, int max_ev_size) {
  HugeCTR::CudaDeviceContext ctx(core_->get_device_id());
  int batch_size_per_gpu = batch_size / num_gpus_;
  DISPATCH_INTEGRAL_FUNCTION(row_lengths.dtype().type(), offset_t, [&] {
    DISPATCH_FLOAT_AND_HALF_FUNCTION(network_comm_buffer.dtype().type(), emb_t, [&] {
      DISPATCH_FLOAT_AND_HALF_FUNCTION(output_buffer.dtype().type(), dst_emb_t, [&] {
        auto stream = core_->get_local_gpu()->get_stream();

        const offset_t** row_lengths_ptr = row_lengths.get<offset_t>();
        const int* network_ids_ptr = network_ids.get<int>();
        const int* network_gpu_ids_ptr = network_gpu_ids.get<int>();
        const int* network_offsets_ptr = network_offsets.get<int>();
        const int* network_dst_lookup_ids_ptr = network_dst_lookup_ids.get<int>();
        const int** network_ev_sizes_ptr = network_ev_sizes.get<int>();
        const int** network_ev_offsets_ptr = network_ev_offsets.get<int>();
        const emb_t** network_comm_buffer_ptr = network_comm_buffer.get<emb_t>();
        const int* d_ev_size_offset_ptr = d_ev_size_offset.get<int>();
        const char* combiner_ptr = d_combiner_list.get<char>();
        dst_emb_t** output_buffer_ptr = output_buffer.get<dst_emb_t>();
        int num_network_dst_lookup_ids = network_dst_lookup_ids.get_num_elements();
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
